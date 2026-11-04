# Models

KaptchaOCR ships five fine-tuned models and can load arbitrary ONNX
recognizers and detectors through a manifest.

## What is bundled

| Model | Task | Size | Notes | Hugging Face |
| --- | --- | ---: | --- | --- |
| `ft-tiny` | recognition | 4.3 MB | `fast` preset | [tiny-rec](https://huggingface.co/zaltahir/kaptchaocr-tiny-rec) |
| `ft-small` | recognition | 20 MB | `balanced` preset (default) | [small-rec](https://huggingface.co/zaltahir/kaptchaocr-small-rec) |
| `ft-medium` | recognition | 73 MB | `accurate` preset | [medium-rec](https://huggingface.co/zaltahir/kaptchaocr-medium-rec) |
| `ft-small-det` | glyph detection | 9.5 MB | click CAPTCHAs | [small-det](https://huggingface.co/zaltahir/kaptchaocr-small-det) |
| `ft-medium-det` | glyph detection | 59 MB | click CAPTCHAs, tighter boxes | [medium-det](https://huggingface.co/zaltahir/kaptchaocr-medium-det) |

Every model has its own repo in the
[KaptchaOCR collection](https://huggingface.co/collections/zaltahir/kaptchaocr).

The recognition models are PP-OCRv6 models fine-tuned on CAPTCHA images and
exported to ONNX. The detectors are PaddleOCR-style DB models trained for
glyph-level boxes. Everything lives in [`model/finetuned/`](../model/finetuned);
see [`model/README.md`](../model/README.md) for the manifest and export notes.

The ddddocr and PP-OCRv6 base weights in the benchmarks are comparison models
only and are **not** included in this repo.

## Profiles

`model/models.json` defines three recognition profiles:

| Profile | Recognition |
| --- | --- |
| `fast` | `ft-tiny` |
| `balanced` | `ft-small` |
| `accurate` | `ft-medium` |

Detection is selected separately; the first det entry — and therefore the
default — is `ft-small-det`.

For click captchas, select the glyph detector explicitly:

```sh
kaptchaocr --ocr --det --manifest model/models.json \
  --rec-model ft-small --det-model ft-small-det
```

Without `--manifest`, the default build uses the embedded `ft-small` recognizer
and `ft-small-det` detector; non-inline builds read those files from
`model/finetuned/`. With a manifest, `--profile`, `--rec-model` and
`--det-model` select entries; the explicit model flags take precedence over the
profile.

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
| `ft-small-det` | det | F1@0.5 | **95.8%** |
| `ft-medium-det` | det | F1@0.5 / F1@0.75 | 95.7% / **63.1%** |
| PP-OCRv6 medium (base, not bundled) | rec | exact match | 28.6% |
| ddddocr v2 / legacy (not bundled) | rec | exact match | 14.3 / 15.3% |
| ddddocr det (not bundled) | det | F1@0.5 | 57.5% |

Full tables, per-set results and latency: [Benchmarks](../benchmarks/README.md).

## Licenses

The fine-tuned models and code are Apache-2.0. They are derived from PP-OCRv6
base models (Apache-2.0). Portions of the original implementation are from
[ddddocr](https://github.com/sml2h3/ddddocr) under the MIT license; its stock
weights are not redistributed. The required notices are in
[`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).
