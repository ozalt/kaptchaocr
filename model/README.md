# model/ — Models & how to produce your own

Directory layout:

| File | Purpose |
| --- | --- |
| `stock_common.onnx` + `stock_common.json` | Stock recognition model (custom PyTorch CRNN/CTC, grayscale, H=64). `stock_common.json` = Charset config (`word`, `image`, `channel`, `charset`, optional `mean`/`std`/`bgr`). |
| `stock_old.onnx` + `stock_old.json` | Stock "old" recognition model (INT8-quantized variant). |
| `stock_det.onnx` | Stock detection model (custom YOLOv5-style, 416×416, anchor grid). |
| `ppocrv6/` | PP-OCRv6 ONNX exports (tiny/small/medium × rec/det). **Gitignored** — re-download from HuggingFace, see `paddleOCR.md` §3. |
| `finetuned/` | Captcha-fine-tuned PP-OCRv6 ONNX: rec (tiny/small/medium) + charsets, and `{small,medium}_det.onnx` (glyph-level detectors for click captchas, manifest entries `ft-small-det` / `ft-medium-det`). Use the tier's `ppocrv6/<tier>_rec.json` charset (blank at index 0); training artifacts live in `../captchowmin-rnd/model-training/`. |
| `models.json` | Model manifest consumed by the server (`--manifest`) and the eval harness. Lets you wire arbitrary ONNX rec/det models without rebuilding. |

Full reference on the PP-OCRv6 family (I/O, preprocessing, dicts) is in
[`../paddleOCR.md`](../paddleOCR.md).

---

## Wiring a model via the manifest

`model/models.json` lists every model with its task, ONNX path, charset
(rec only) and post-processing config (det only). Start the server against it:

```bash
cargo run -- --manifest model/models.json --ocr --det
```

The manifest's **first `rec` entry** becomes the OCR model and the **first
`det` entry** the detection model. Paths are relative to the working
directory. `--rec-model <name>` picks a specific rec entry, `--det-model
<name>` a specific det entry (`ft-small-det` / `ft-medium-det` are the
glyph-level click-captcha fine-tunes: F1@0.5 ~95–97% on the synthetic scatter
val sets vs 60.6% for `stock-det`; medium is ~4× slower with tighter boxes),
and `--profile <name>` uses the manifest's `profiles` map
(shipped: `fast` → `ft-tiny`, `balanced` → `ft-small`, `accurate` →
`ft-medium`). With the canonical `model/models.json`, plain `--manifest`
already serves the captcha-fine-tuned `ft-small`. `charset_range` works with
the official models **and** with manifest/fine-tuned PP-OCRv6 (CTC) models:
the per-frame choice is restricted to the requested characters before the CTC
merge, which is what `/solve/click` uses to constrain single-glyph
recognition to the prompt's characters. For a DB (PaddleOCR-style) detector:

```json
{ "name": "my-det", "task": "det", "path": "model/foo_det.onnx",
  "det": { "type": "db", "thresh": 0.2, "box_thresh": 0.4,
           "unclip_ratio": 1.4, "max_candidates": 3000,
           "nms_thr": 0.3, "score_thr": 0.1 } }
```

---

## PaddleOCR → ONNX conversion

Use PaddleOCR/PaddleX to export a trained model to ONNX (choose one route):

### Route A — PaddleX export (PP-OCRv6/v5 style, PIR)

```bash
pip install paddleocr "paddleocr[all]" paddle2onnx

# export rec
paddlex --install paddleocr
python - <<'EOF'
from paddleocr import TextRecognition
m = TextRecognition(model_name="PP-OCRv6_tiny_rec", device="cpu")
m.export(save_dir="./export/rec", dynamic_shape=True, input_shape=[1,3,48,-1])
EOF

# export det
python - <<'EOF'
from paddleocr import TextDetection
m = TextDetection(model_name="PP-OCRv6_tiny_det", device="cpu")
m.export(save_dir="./export/det", dynamic_shape=True)
EOF
```

### Route B — classic `paddle2onnx`

```bash
paddle2onnx \
  --model_dir ./inference \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference.onnx \
  --opset_version 12 \
  --enable_onnx_checker True \
  --input_shape_dict "{'x':[-1,3,48,-1]}"      # rec: [N,3,48,W]
  # det: "{'x':[-1,3,-1,-1]}"
```

> Keep **dynamic shapes** (or a fixed H, dynamic W for rec) — the server feeds
> variable-width text. If a det export fails under ORT with dynamic input
> sizes (internal `Resize` buffer mismatch), feed a fixed square input; the
> server's DB path already pads to 736×736.

### Route C — fine-tuned model

Train with PaddleOCR/PaddleX, then export the fine-tuned model exactly like
above. Two gotchas learned from the shipped fine-tunes (rec:
`finetuned/*_rec.onnx`, det: `finetuned/{small,medium}_det.onnx`):

- **PIR off**: `tools/export_model.py` on Paddle 3.x defaults to the PIR format
  (`inference.json`), which paddle2onnx 2.1.0 cannot read — pass
  `-o Global.export_with_pir=False` to write the classic `inference.pdmodel` /
  `inference.pdiparams`.
- **Flush subnormals**: zero any float weight `< 1.175e-38` after conversion
  (ORT convs can run ~6× slower otherwise; the det fine-tune had 541k such
  values). Full recipe:
  `../captchowmin-rnd/dataset-curation-det/convert_det_onnx.sh`.

---

## Shrinking (quantization / half precision)

Three standard knobs, in increasing order of aggression:

### 1. FP16 (half) — ~2× smaller, usually lossless for inference

```bash
pip install onnxruntime
python -m onnxruntime.transformers.float16 \
  --input inference.onnx --output inference_fp16.onnx
```

### 2. INT8 dynamic quantization — ~4× smaller, minimal code

```bash
python - <<'EOF'
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("inference.onnx", "inference_int8.onnx",
                 weight_type=QuantType.QInt8)
EOF
```

### 3. INT8 QAT / calibration-aware — best accuracy at INT8

Quantize with calibration data (a few hundred real/synthetic captchas):

```bash
python - <<'EOF'
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
# ... subclass CalibrationDataReader feeding your eval samples ...
quantize_static("inference.onnx", "inference_qat.onnx", dr,
                weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
EOF
```

> `model/stock_old.onnx` is an example of a dynamic-INT8-quantized model
> already in this repo (`onnx.quantize` producer). Validate accuracy with the
> eval harness before/after shrinking:
>
> ```bash
> cargo run --release --example eval -- --n 200 --lang both
> ```

---

## Validating a new model

Add it to `model/models.json`, then:

```bash
# rec: does it recognize?
curl -s -X POST -H 'Content-Type: application/json' \
  -d "{\"image\":\"$(base64 -w0 image/3.png)\"}" \
  http://127.0.0.1:8000/ocr

# det: does it box the text?
curl -s -X POST -H 'Content-Type: application/json' \
  -d "{\"image\":\"$(base64 -w0 image/3.png)\"}" \
  http://127.0.0.1:8000/det
```
