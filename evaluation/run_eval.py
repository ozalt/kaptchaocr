"""End-to-end evaluation of a running KaptchaOCR server, through the Python client.

Suites (each prints a JSON summary and writes it to evaluation/results/<suite>.json):
  rec     the six public held-out recognition splits (zaltahir/kaptchaocr-rec)
  synth   fresh out-of-distribution text CAPTCHAs (python `captcha` lib + a PIL generator)
  math    PIL-drawn arithmetic CAPTCHAs through /solve
  click   the public synthetic click set (zaltahir/kaptchaocr-det, test split) through /solve/click
  slider  procedurally generated jigsaw sliders through /slide_match

Usage:
  python evaluation/run_eval.py --data DIR [--url http://127.0.0.1:8000] [--suites rec,synth,...] [--limit N]
DIR must contain rec/data/test_*.parquet and det/data/test-*.parquet (huggingface snapshot_download).
"""

import argparse
import io
import json
import random
import string
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "python"))
from kaptchaocr import Client  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def cer(pred, gold):
    # Levenshtein distance / len(gold)
    prev = list(range(len(gold) + 1))
    for i, p in enumerate(pred, 1):
        cur = [i]
        for j, g in enumerate(gold, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != g)))
        prev = cur
    return prev[-1] / max(len(gold), 1)


def score_text(pairs, ms):
    n = len(pairs)
    return {
        "n": n,
        "exact": round(sum(p == g for p, g in pairs) / n, 4),
        "exact_ci": round(sum(p.lower() == g.lower() for p, g in pairs) / n, 4),
        "cer": round(sum(cer(p, g) for p, g in pairs) / n, 4),
        "mean_ms": round(sum(ms) / n, 1),
        "errors_sample": [(g, p) for p, g in pairs if p != g][:10],
    }


def ocr_many(client, items):
    pairs, ms = [], []
    for img_bytes, gold in items:
        t = time.perf_counter()
        pred = client.ocr(img_bytes).text
        ms.append((time.perf_counter() - t) * 1000)
        pairs.append((pred, gold))
    return pairs, ms


def suite_rec(client, data, limit):
    out = {}
    for f in sorted((data / "rec" / "data").glob("test_*.parquet")):
        df = pd.read_parquet(f)
        if limit:
            df = df.sample(min(limit, len(df)), random_state=0)
        items = [(r.image["bytes"], r.text) for r in df.itertuples()]
        out[f.name.split("-")[0]] = score_text(*ocr_many(client, items))
    n = sum(v["n"] for v in out.values())
    out["overall_micro"] = {k: round(sum(v[k] * v["n"] for v in out.values()) / n, 4)
                            for k in ("exact", "exact_ci", "cer")}
    return out


def pil_captcha(text, rng):
    w, h = 40 + 28 * len(text), 64
    img = Image.new("RGB", (w, h), tuple(rng.randint(200, 255) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(6):  # noise lines
        d.line([(rng.randint(0, w), rng.randint(0, h)) for _ in range(2)],
               fill=tuple(rng.randint(60, 180) for _ in range(3)), width=rng.randint(1, 2))
    x = 18
    for ch in text:
        font = ImageFont.truetype(FONT, rng.randint(28, 38))
        glyph = Image.new("RGBA", (50, 60), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((8, 5), ch, font=font,
                                   fill=tuple(rng.randint(0, 120) for _ in range(3)) + (255,))
        glyph = glyph.rotate(rng.uniform(-25, 25), resample=Image.BICUBIC)
        img.paste(glyph, (x - 8, rng.randint(-4, 6)), glyph)
        x += rng.randint(22, 30)
    for _ in range(300):  # speckle
        d.point((rng.randint(0, w - 1), rng.randint(0, h - 1)), fill=(0, 0, 0))
    return img.filter(ImageFilter.SMOOTH)


def suite_synth(client, data, limit):
    from captcha.image import ImageCaptcha

    rng = random.Random(1234)
    alphabet = string.ascii_letters + string.digits
    n = limit or 300
    gen = ImageCaptcha(width=200, height=70)
    out = {}
    for name, make in [
        ("python_captcha_lib", lambda t: gen.generate_image(t)),
        ("pil_rotated_noise", lambda t: pil_captcha(t, rng)),
    ]:
        items = []
        for _ in range(n):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 6)))
            items.append((png(make(text)), text))
        out[name] = score_text(*ocr_many(client, items))
    return out


def suite_math(client, data, limit):
    rng = random.Random(7)
    ok, rows = 0, []
    n = limit or 200
    for _ in range(n):
        a, b, op = rng.randint(1, 20), rng.randint(1, 9), rng.choice("+-*")
        text = f"{a}{'x' if op == '*' else op}{b}=?"
        gold = str(eval(f"{a}{op}{b}"))
        res = client.solve(png(pil_captcha(text, rng)))
        got = res.math.answer if res.math else None
        ok += got == gold
        if got != gold and len(rows) < 10:
            rows.append({"image_text": text, "ocr": res.text, "answer": got, "gold": gold})
    return {"n": n, "solved": round(ok / n, 4), "errors_sample": rows}


def suite_click(client, data, limit):
    df = pd.read_parquet(next((data / "det" / "data").glob("test-*.parquet")))
    if limit:
        df = df.head(limit)
    stats = {}
    for mode in ("prompt_only", "prompt+charset_range"):
        solved = targets = hit = 0
        ms = []
        by_type = {"latin": [0, 0], "cjk": [0, 0]}  # [hits, total]
        for r in df.itertuples():
            objs = list(r.objects)
            chars = [o["category"] for o in objs]
            prompt = "请依次点击【" + ",".join(chars) + "】"
            t = time.perf_counter()
            res = client.solve_click(r.image["bytes"], prompt,
                                     charset_range="".join(chars) if "charset" in mode else None)
            ms.append((time.perf_counter() - t) * 1000)
            # points come back in prompt order for matched chars only; map via unmatched list
            matched = [c for c in chars]
            for c in res.unmatched:
                matched.remove(c)
            ok_all = not res.unmatched
            for ch in chars:
                by_type["latin" if ch.isascii() else "cjk"][1] += 1
            for ch, pt in zip(matched, res.points):
                o = objs[chars.index(ch)]
                x, y, w, h = o["bbox"]
                inside = x <= pt.x <= x + w and y <= pt.y <= y + h
                hit += inside
                by_type["latin" if ch.isascii() else "cjk"][0] += inside
                ok_all &= inside
            targets += len(chars)
            solved += ok_all
        stats[mode] = {"n_images": len(df), "image_solved": round(solved / len(df), 4),
                       "target_click_hit": round(hit / targets, 4),
                       "hit_by_type": {k: round(h / max(t, 1), 4) for k, (h, t) in by_type.items()},
                       "mean_ms": round(sum(ms) / len(ms), 1)}
    return stats


def make_slider(rng):
    W, H, S = 320, 160, 50
    bg = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(bg)
    for y in range(H):  # gradient
        d.line([(0, y), (W, y)], fill=(rng.randint(0, 30) + y, 90 + y // 3, 200 - y))
    for _ in range(40):  # texture
        x, y, r = rng.randint(0, W), rng.randint(0, H), rng.randint(5, 40)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(rng.randint(0, 255) for _ in range(3)))
    bg = bg.filter(ImageFilter.GaussianBlur(1))
    gx, gy = rng.randint(90, W - S - 10), rng.randint(10, H - S - 10)
    mask = Image.new("L", (S, S), 0)
    m = ImageDraw.Draw(mask)
    m.rectangle([0, 10, S - 10, S - 1], fill=255)
    m.ellipse([S - 22, 18, S - 1, 38], fill=255)  # right knob
    m.ellipse([12, 0, 32, 20], fill=255)  # top knob
    piece = Image.new("RGBA", (S, S))
    piece.paste(bg.crop((gx, gy, gx + S, gy + S)), (0, 0), mask)
    ImageDraw.Draw(piece).bitmap((0, 0), mask.filter(ImageFilter.FIND_EDGES), fill=(255, 255, 255, 255))
    target = Image.new("RGBA", (S + 4, H), (0, 0, 0, 0))  # tall transparent strip, piece at gy
    target.paste(piece, (2, gy), piece)
    shade = Image.new("RGB", (S, S), (0, 0, 0))
    region = bg.crop((gx, gy, gx + S, gy + S))
    bg.paste(Image.blend(region, shade, 0.55), (gx, gy), mask)
    return png(target), png(bg), gx


def suite_slider(client, data, limit):
    rng = random.Random(42)
    n = limit or 200
    errs, ms = [], []
    for _ in range(n):
        tgt, bg, gx = make_slider(rng)
        t = time.perf_counter()
        res = client.slide_match(tgt, bg)
        ms.append((time.perf_counter() - t) * 1000)
        errs.append(abs(res.target[0] - gx))
    errs.sort()
    return {"n": n, "within_3px": round(sum(e <= 3 for e in errs) / n, 4),
            "within_8px": round(sum(e <= 8 for e in errs) / n, 4),
            "median_err_px": errs[n // 2], "mean_ms": round(sum(ms) / n, 1)}


SUITES = {"rec": suite_rec, "synth": suite_synth, "math": suite_math,
          "click": suite_click, "slider": suite_slider}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    client = Client(a.url, timeout=60)
    OUT.mkdir(exist_ok=True)
    for name in a.suites.split(","):
        res = SUITES[name](client, a.data, a.limit)
        print(name, json.dumps(res, ensure_ascii=False, indent=1))
        (OUT / f"{name}{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
