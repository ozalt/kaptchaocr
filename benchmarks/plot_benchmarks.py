#!/usr/bin/env python3
"""Generate README-ready benchmark charts from the eval CSV results.

Usage:
    python3 benchmarks/plot_benchmarks.py

Inputs (committed):
    benchmarks/results/rec_2026-11-03.csv
    benchmarks/results/det_2026-11-03.csv

Outputs:
    docs/assets/benchmarks/*_light.png, *_dark.png, *.svg
    benchmarks/results/rec_summary.csv
    benchmarks/results/det_summary.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "kaptchaocr"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "assets" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = [
    "test_archive1",
    "test_archive2",
    "test_archive3",
    "test_hammer888",
    "test_kaggle",
    "test_lcsd",
]
SPLIT_LABEL = {s: s.removeprefix("test_") for s in SPLITS}

REC_ORDER = [
    "stock-common",
    "stock-old",
    "base-tiny",
    "base-small",
    "base-medium",
    "ft-tiny",
    "ft-small",
    "ft-medium",
]
REC_LABEL = {
    "stock-common": "ddddocr v2",
    "stock-old": "ddddocr legacy",
    "base-tiny": "PP-OCRv6 tiny",
    "base-small": "PP-OCRv6 small",
    "base-medium": "PP-OCRv6 medium",
    "ft-tiny": "KaptchaOCR ft-tiny",
    "ft-small": "KaptchaOCR ft-small",
    "ft-medium": "KaptchaOCR ft-medium",
}
REC_FAMILY = {
    "stock-common": "ddddocr",
    "stock-old": "ddddocr",
    "base-tiny": "PP-OCRv6 base",
    "base-small": "PP-OCRv6 base",
    "base-medium": "PP-OCRv6 base",
    "ft-tiny": "KaptchaOCR fine-tuned",
    "ft-small": "KaptchaOCR fine-tuned",
    "ft-medium": "KaptchaOCR fine-tuned",
}
DET_LABEL = {
    "stock-det": "ddddocr det",
    "ppocrv6-tiny-det": "PP-OCRv6 tiny",
    "ppocrv6-small-det": "PP-OCRv6 small",
    "ppocrv6-medium-det": "PP-OCRv6 medium",
    "ft-small-det": "KaptchaOCR ft-small",
    "ft-medium-det": "KaptchaOCR ft-medium",
}
DET_FAMILY = {
    "stock-det": "ddddocr",
    "ppocrv6-tiny-det": "PP-OCRv6 base",
    "ppocrv6-small-det": "PP-OCRv6 base",
    "ppocrv6-medium-det": "PP-OCRv6 base",
    "ft-small-det": "KaptchaOCR fine-tuned",
    "ft-medium-det": "KaptchaOCR fine-tuned",
}
FAMILIES = ["ddddocr", "PP-OCRv6 base", "KaptchaOCR fine-tuned"]
COLORS = {
    "light": {
        "ddddocr": "#475569",
        "PP-OCRv6 base": "#2563eb",
        "KaptchaOCR fine-tuned": "#059669",
    },
    "dark": {
        "ddddocr": "#94a3b8",
        "PP-OCRv6 base": "#60a5fa",
        "KaptchaOCR fine-tuned": "#34d399",
    },
}
THEMES = {
    "light": {
        "bg": "#ffffff",
        "panel": "#ffffff",
        "text": "#0f172a",
        "muted": "#475569",
        "grid": "#e2e8f0",
        "edge": "#cbd5e1",
    },
    "dark": {
        "bg": "#0b1220",
        "panel": "#0f172a",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "grid": "#1e293b",
        "edge": "#334155",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summary_sizes(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        return {r["model"]: float(r["size_mb"]) for r in read_csv(path)}
    except (KeyError, ValueError):
        return {}


def style_ax(ax, t, grid_axis="x"):
    ax.set_facecolor(t["panel"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["edge"])
    ax.tick_params(colors=t["muted"])
    ax.xaxis.label.set_color(t["text"])
    ax.yaxis.label.set_color(t["text"])
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def header(ax, title: str, subtitle: str, t):
    ax.text(
        0.0,
        1.075,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15.5,
        fontweight="bold",
        color=t["text"],
    )
    ax.text(
        0.0,
        1.012,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=t["muted"],
    )


def family_legend(ax, t, loc="lower right", ncol=1):
    handles = [
        Patch(facecolor=COLORS["light" if t["bg"] == "#ffffff" else "dark"][fam], label=fam)
        for fam in FAMILIES
    ]
    ax.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        fontsize=8.5,
        labelcolor=t["muted"],
        ncol=ncol,
    )


def save(fig, stem: str, theme_name: str, t):
    fig.savefig(
        OUT / f"{stem}_{theme_name}.png",
        dpi=200,
        facecolor=t["bg"],
        bbox_inches="tight",
        pad_inches=0.18,
    )
    if theme_name == "light":
        fig.savefig(
            OUT / f"{stem}.svg",
            facecolor=t["bg"],
            bbox_inches="tight",
            pad_inches=0.18,
            metadata={"Date": None},
        )
    plt.close(fig)


def load_rec():
    rows = read_csv(RES / "rec_2026-11-03.csv")
    data: dict[str, dict[str, dict[str, float | int]]] = {s: {} for s in SPLITS}
    for r in rows:
        data[r["split"]][r["model"]] = {
            "exact": float(r["exact_rate"]),
            "char": float(r["char_acc"]),
            "mean_ms": float(r["mean_ms"]),
            "p50_ms": float(r["p50_ms"]),
            "p95_ms": float(r["p95_ms"]),
            "n": int(r["n"]),
        }
    overall = {}
    sizes = summary_sizes(RES / "rec_summary.csv")
    for m in REC_ORDER:
        n = sum(int(data[s][m]["n"]) for s in SPLITS)
        overall[m] = {
            "n": n,
            "exact": sum(float(data[s][m]["exact"]) * int(data[s][m]["n"]) for s in SPLITS) / n,
            "char": sum(float(data[s][m]["char"]) * int(data[s][m]["n"]) for s in SPLITS) / n,
            "mean_ms": sum(float(data[s][m]["mean_ms"]) * int(data[s][m]["n"]) for s in SPLITS) / n,
            "size_mb": sizes.get(m, 0.0),
        }
    return data, overall


def load_det():
    rows = read_csv(RES / "det_2026-11-03.csv")
    sizes = summary_sizes(RES / "det_summary.csv")
    data = {}
    for r in rows:
        data[r["model"]] = {
            "f1_50": float(r["f1_50"]),
            "f1_75": float(r["f1_75"]),
            "p_50": float(r["precision_50"]),
            "r_50": float(r["recall_50"]),
            "miou_50": float(r["mean_iou_50"]),
            "mean_ms": float(r["mean_ms"]),
            "images": int(r["images"]),
            "gt_boxes": int(r["gt_boxes"]),
            "size_mb": sizes.get(r["model"], 0.0),
        }
    return data


def rec_heatmap(data, overall):
    models = REC_ORDER
    matrix = np.array(
        [[float(data[s][m]["exact"]) * 100 for s in SPLITS] for m in models]
    )
    overall_col = np.array([[overall[m]["exact"] * 100] for m in models])
    values = np.hstack([matrix, overall_col])
    counts = [int(data[s][models[0]]["n"]) for s in SPLITS]
    cols = [f"{SPLIT_LABEL[s]}\n{counts[i]:,}" for i, s in enumerate(SPLITS)] + [
        f"all\n{overall[models[0]]['n']:,}"
    ]

    for theme_name, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(12.4, 5.4))
        fig.patch.set_facecolor(t["bg"])
        im = ax.imshow(
            values,
            cmap="YlGnBu" if theme_name == "light" else "viridis",
            vmin=0,
            vmax=100,
            aspect="auto",
        )
        ax.set_facecolor(t["panel"])

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if theme_name == "light":
                    color = "#ffffff" if v >= 52 else "#0f172a"
                else:
                    color = "#0f172a" if v >= 55 else "#e2e8f0"
                weight = "bold" if j == values.shape[1] - 1 else "normal"
                ax.text(
                    j,
                    i,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8.6 if j < values.shape[1] - 1 else 9.2,
                    fontweight=weight,
                )

        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, fontsize=8.8, color=t["muted"])
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([REC_LABEL[m] for m in models], fontsize=9.4, color=t["text"])
        ax.tick_params(length=0)

        for side in ax.spines.values():
            side.set_visible(False)
        ax.axvline(len(SPLITS) - 0.5, color=t["edge"], linewidth=1.2)

        header(
            ax,
            "Exact-match accuracy on held-out CAPTCHA sets",
            "4,399 images · held out from our training · higher is better",
            t,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
        cbar.set_label("Exact match (%)", color=t["muted"], fontsize=9)
        cbar.ax.tick_params(colors=t["muted"])
        cbar.outline.set_edgecolor(t["edge"])
        fig.subplots_adjust(top=0.83)
        save(fig, "rec_exact_heatmap", theme_name, t)

    return values


def rec_macro(overall):
    models = sorted(REC_ORDER, key=lambda m: overall[m]["exact"], reverse=True)
    exact = np.array([overall[m]["exact"] * 100 for m in models])
    char = np.array([overall[m]["char"] * 100 for m in models])
    y = np.arange(len(models))

    for theme_name, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(10.2, 5.6))
        fig.patch.set_facecolor(t["bg"])
        colors = [COLORS[theme_name][REC_FAMILY[m]] for m in models]
        ax.barh(y - 0.18, exact, height=0.34, color=colors)
        ax.barh(y + 0.18, char, height=0.34, color=colors, alpha=0.38)
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels([REC_LABEL[m] for m in models], fontsize=9.5, color=t["text"])
        ax.set_xlim(0, 105)
        ax.set_xlabel("Accuracy (%)")
        ax.set_xticks([0, 20, 40, 60, 80, 100])

        for i, (e, c) in enumerate(zip(exact, char)):
            ax.text(e + 1.2, i - 0.18, f"{e:.0f}%", va="center", fontsize=8.2, color=t["muted"])
            ax.text(c + 1.2, i + 0.18, f"{c:.0f}%", va="center", fontsize=8.2, color=t["muted"])

        style_ax(ax, t)
        header(
            ax,
            "Recognition accuracy across all held-out sets",
            "4,399 images · solid = exact match, faded = character accuracy",
            t,
        )
        family_legend(ax, t, loc="lower right")
        fig.subplots_adjust(left=0.25, top=0.84)
        save(fig, "rec_macro_accuracy", theme_name, t)


def rec_pareto(overall):
    for theme_name, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(9.6, 5.6))
        fig.patch.set_facecolor(t["bg"])
        offsets = {
            "stock-common": (8, 10),
            "stock-old": (8, -16),
            "base-tiny": (-18, -22),
            "base-small": (8, 9),
            "base-medium": (-70, 12),
            "ft-tiny": (9, 10),
            "ft-small": (10, -14),
            "ft-medium": (-86, -20),
        }
        for m in REC_ORDER:
            x = overall[m]["mean_ms"]
            y = overall[m]["exact"] * 100
            s = 40 + math.sqrt(max(overall[m]["size_mb"], 0.1)) * 22
            ax.scatter(
                x,
                y,
                s=s,
                color=COLORS[theme_name][REC_FAMILY[m]],
                edgecolor=t["bg"],
                linewidth=1.5,
                zorder=3,
            )
            ax.annotate(
                REC_LABEL[m],
                (x, y),
                xytext=offsets[m],
                textcoords="offset points",
                fontsize=8.2,
                color=t["muted"],
                arrowprops=dict(arrowstyle="-", color=t["edge"], lw=0.7),
            )
        ax.set_xlabel("Mean latency (ms)")
        ax.set_ylabel("Exact match (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 92)
        style_ax(ax, t, grid_axis="both")
        header(
            ax,
            "Recognition accuracy vs. latency",
            "Bubble size = model size · CPU, single process · mean latency",
            t,
        )
        family_legend(ax, t, loc="lower right")
        fig.subplots_adjust(top=0.84)
        save(fig, "rec_accuracy_latency", theme_name, t)


def det_f1(data):
    models = sorted(data, key=lambda m: data[m]["f1_50"], reverse=True)
    f50 = np.array([data[m]["f1_50"] * 100 for m in models])
    f75 = np.array([data[m]["f1_75"] * 100 for m in models])
    y = np.arange(len(models))

    for theme_name, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(10.2, 4.9))
        fig.patch.set_facecolor(t["bg"])
        colors = [COLORS[theme_name][DET_FAMILY[m]] for m in models]
        ax.barh(y - 0.18, f50, height=0.34, color=colors)
        ax.barh(y + 0.18, f75, height=0.34, color=colors, alpha=0.38)
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels([DET_LABEL[m] for m in models], fontsize=9.5, color=t["text"])
        ax.set_xlim(0, 105)
        ax.set_xlabel(
            "F1 (%)   ·   PP-OCRv6 det is line-level; scored against glyph-level GT",
            fontsize=9,
        )
        ax.set_xticks([0, 20, 40, 60, 80, 100])

        for i, (a, b) in enumerate(zip(f50, f75)):
            ax.text(a + 1.2, i - 0.18, f"{a:.0f}%", va="center", fontsize=8.2, color=t["muted"])
            ax.text(b + 1.2, i + 0.18, f"{b:.0f}%", va="center", fontsize=8.2, color=t["muted"])

        style_ax(ax, t)
        header(
            ax,
            "Glyph detection F1 on unseen data",
            "400 synthetic images · 1,800 glyph boxes · solid = IoU 0.5, faded = IoU 0.75",
            t,
        )
        family_legend(ax, t, loc="lower right")
        fig.subplots_adjust(left=0.22, top=0.83)
        save(fig, "det_f1", theme_name, t)


def det_pareto(data):
    for theme_name, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(9.6, 5.5))
        fig.patch.set_facecolor(t["bg"])
        offsets = {
            "stock-det": (8, 9),
            "ppocrv6-tiny-det": (8, -17),
            "ppocrv6-small-det": (8, 9),
            "ppocrv6-medium-det": (-96, -18),
            "ft-small-det": (10, -16),
            "ft-medium-det": (-104, -18),
        }
        for m in data:
            x = data[m]["mean_ms"]
            y = data[m]["f1_50"] * 100
            s = 40 + math.sqrt(max(data[m]["size_mb"], 0.1)) * 22
            ax.scatter(
                x,
                y,
                s=s,
                color=COLORS[theme_name][DET_FAMILY[m]],
                edgecolor=t["bg"],
                linewidth=1.5,
                zorder=3,
            )
            ax.annotate(
                DET_LABEL[m],
                (x, y),
                xytext=offsets[m],
                textcoords="offset points",
                fontsize=8.2,
                color=t["muted"],
                arrowprops=dict(arrowstyle="-", color=t["edge"], lw=0.7),
            )
        ax.set_xscale("log")
        ax.set_xlim(90, 1800)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Mean latency (ms, log scale)")
        ax.set_ylabel("F1@0.5 (%)")
        style_ax(ax, t, grid_axis="both")
        header(
            ax,
            "Detection F1 vs. latency",
            "Bubble size = model size · CPU, single process · PP-OCRv6 det is line-level",
            t,
        )
        family_legend(ax, t, loc="lower right")
        fig.subplots_adjust(top=0.84)
        save(fig, "det_accuracy_latency", theme_name, t)


def write_summaries(overall, det):
    with (RES / "rec_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "exact_rate", "char_acc", "mean_ms", "size_mb"])
        for m in REC_ORDER:
            w.writerow(
                [
                    m,
                    REC_FAMILY[m],
                    f"{overall[m]['exact']:.6f}",
                    f"{overall[m]['char']:.6f}",
                    f"{overall[m]['mean_ms']:.3f}",
                    f"{overall[m]['size_mb']:.2f}",
                ]
            )

    with (RES / "det_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "family",
                "f1_50",
                "f1_75",
                "precision_50",
                "recall_50",
                "mean_iou_50",
                "mean_ms",
                "size_mb",
            ]
        )
        for m, r in det.items():
            w.writerow(
                [
                    m,
                    DET_FAMILY[m],
                    f"{r['f1_50']:.6f}",
                    f"{r['f1_75']:.6f}",
                    f"{r['p_50']:.6f}",
                    f"{r['r_50']:.6f}",
                    f"{r['miou_50']:.6f}",
                    f"{r['mean_ms']:.3f}",
                    f"{r['size_mb']:.2f}",
                ]
            )


def main():
    data, overall = load_rec()
    det = load_det()
    rec_heatmap(data, overall)
    rec_macro(overall)
    rec_pareto(overall)
    det_f1(det)
    det_pareto(det)
    write_summaries(overall, det)
    print("wrote charts to", OUT)
    for p in sorted(OUT.glob("*")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
