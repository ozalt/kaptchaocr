# Models

KaptchaOCR ships three model families and can load arbitrary ONNX recognizers
and detectors through a manifest.

## What is bundled

| Family | Task | Tiers | Size | Notes |
| --- | --- | --- | ---: | --- |
| ddddocr (MIT) | recognition | v2, legacy | 52 / 13 MB | Stock `stock_common` / `stock_old`; embedded fallback |
| ddddocr (MIT) | detection | stock | 19 MB | `stock_det`; embedded fallback |
| PP-OCRv6 (Apache-2.0) | recognition | tiny / small / medium | 4.3 / 20 / 73 MB | Base models; reference points |
| PP-OCRv6 (Apache-2.0) | detection | tiny / small / medium | 1.7 / 9.4 / 59 MB | Line-level detectors |
| KaptchaOCR fine-tuned | recognition | tiny / small / medium | 4.3 / 20 / 73 MB | `ft-tiny` / `ft-small` / `ft-medium` |
| KaptchaOCR fine-tuned | detection | small / medium | 9.5 / 59 MB | `ft-small-det` / `ft-medium-det`; glyph-level |

The fine-tuned models are PP-OCRv6 models fine-tuned on a CAPTCHA dataset and
exported to ONNX. The base `model/ppocrv6/` files are **gitignored** — see
[`model/README.md`](../model/README.md) for download/export notes.

## Profiles

`model/models.json` defines three profiles:

| Profile | Recognition | Detection |
| --- | --- | --- |
| `fast` | `ft-tiny` | `stock-det` |
| `balanced` | `ft-small` | `stock-det` |
| `accurate` | `ft-medium` | `stock-det` |

For click captchas, select the glyph detector explicitly:

```sh
kaptchaocr --ocr --det --manifest model/models.json \
  --rec-model ft-small --det-model ft-small-det
```

Without `--manifest`, the server uses the embedded stock models. With a
manifest, `--profile`, `--rec-model` and `--det-model` select entries; the
explicit model flags take precedence over the profile.

## Manifest format

`model/models.json` has `profiles` plus a `models` array. Each entry:

```json
{
  "name": "ft-small",
  "task": "rec",
  "path": "model/finetuned/small_rec.onnx",
  "charset": "model/finetuned/small_rec.json"
}
```

Detection entries add a `det` object (`type: "db"` or `"yolo"` with thresholds).
Paths are resolved relative to the working directory, so run the server from
the repository root (or adjust paths).

## Custom models

Any ONNX recognition model can be paired with a charset JSON:

```rust
let mut ocr = kaptchaocr::KaptchaOcr::with_model_charset(
    "captcha_model.onnx",
    "charset.json",
)?;
let image = std::fs::read("sample.jpg")?;
println!("{}", ocr.classification(image)?);
```

The charset file uses the same schema as the bundled `.json` files; see
[`model/README.md`](../model/README.md). For a server-side custom model, add a
manifest entry and point `--manifest` at it. Detection models need a `det`
configuration (type + thresholds).

## Accuracy snapshot

Held-out benchmark, 4,399 recognition images / 400 detection images:

| Model | Task | Metric | Score |
| --- | --- | --- | ---: |
| `ft-medium` | rec | exact match | **90.7%** |
| `ft-small` | rec | exact match | **80.4%** |
| `ft-tiny` | rec | exact match | **67.0%** |
| PP-OCRv6 medium (base) | rec | exact match | 28.6% |
| ddddocr v2 / legacy | rec | exact match | 14.3 / 15.3% |
| `ft-small-det` | det | F1@0.5 | **95.8%** |
| `ft-medium-det` | det | F1@0.5 / F1@0.75 | 95.7% / **63.1%** |
| `ddddocr det` | det | F1@0.5 | 57.5% |

Full tables, per-set results and latency: [Benchmarks](../benchmarks/README.md).

## Licenses

The stock models and portions of the original implementation are from
[ddddocr](https://github.com/sml2h3/ddddocr) under the MIT license. PP-OCRv6
base models are Apache-2.0. The required notices are in
[`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).
