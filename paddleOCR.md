# PaddleOCR / PP-OCRv6 — Model Reference

> Status: **REFERENCE ONLY** (2026-08-12). Nothing here is wired into the
> inference pipeline yet. It documents the PP-OCRv6 model family we plan to
> use, what each model needs to run (I/O, preprocessing, postprocessing), and
> how it maps onto the existing `kaptchaocr` pipeline. Integration is Phase 3 /
> Phase 4 work; empirical-verification items are flagged `[verify]`.

## 1. TL;DR — what models do we need

We are going to use **all six** PP-OCRv6 ONNX exports, covering every tier:

| Tier   | Recognition (rec)                 | Detection (det)                 |
| ------ | --------------------------------- | ------------------------------- |
| tiny   | `PP-OCRv6_tiny_rec_onnx`  (~4.5 MB ONNX, 1.11M params)  | `PP-OCRv6_tiny_det_onnx`  (~1.8 MB ONNX, 0.44M params)  |
| small  | `PP-OCRv6_small_rec_onnx` (~21 MB ONNX, 5.29M params)    | `PP-OCRv6_small_det_onnx` (~9.9 MB ONNX, 2.48M params)  |
| medium | `PP-OCRv6_medium_rec_onnx`(~77 MB ONNX, 19.2M params)    | `PP-OCRv6_medium_det_onnx`(~62 MB ONNX, 22M params)     |

- **Recognition** (`text_recognition` / CTC word model): `x` [N,3,48,W_dyn] → `fetch_name_0` [N,T,classes].
- **Detection** (`text_detection` / DB detector): `x` [N,3,H_dyn,W_dyn] → `fetch_name_0` [N,1,H_r,W_r] probability map.
- Only the **ONNX** repos matter for `kaptchaocr` (ONNX Runtime). The plain Paddle and `.safetensors` repos use different runtimes (see §4).

These replace/complement the stock bundled models today:
- `model/stock_common.onnx` / `stock_old.onnx` — rec, height 32, channel 3, small vocab.
- `model/stock_det.onnx` — **anchor-based YOLO-style** detector (different family from PP-OCRv6 DB det).

Collection page: https://huggingface.co/collections/PaddlePaddle/pp-ocrv6

## 2. What PP-OCRv6 is

PP-OCRv6 is a lightweight OCR system (paper arXiv:2606.13108). Three tiers
(medium/small/tiny) share the same MetaFormer-style building blocks:

- **Backbone:** LCNetV4 (structural reparameterization).
- **Det neck:** RepLKFPN (dilated reparameterizable depthwise convs).
- **Rec neck/head:** EncoderWithLightSVTR (local–global attention, additive
  skip connections) + CTC + NRTR multi-head decoder.

Supported: 50 languages (rec supports 49 in the plain models; the bundled
dict is much larger). Claimed accuracy beats billion-scale VLMs with orders of
magnitude fewer parameters.

Reported accuracy (from HF READMEs):
- **rec** (W-Avg over 16 benchmarks): medium 83.2, small 81.3, tiny 73.5
  (vs PP-OCRv5_server 78.1).
- **det**: medium 86.2 Hmean (only medium figure published on the page).

## 3. Model formats on HuggingFace

Each model exists in **three** repos (all `PaddlePaddle/...`):

| Suffix               | Contents                                    | Runtime                       | Relevant to us? |
| -------------------- | ------------------------------------------- | ----------------------------- | --------------- |
| (none, e.g. `PP-OCRv6_tiny_rec`) | `inference.pdiparams` + `inference.yml` + `inference.json` | PaddlePaddle native inference | No              |
| `_onnx`              | `inference.onnx` + `inference.yml` + `inference.json`     | ONNX Runtime                  | **YES**         |
| `_safetensors`       | `model.safetensors` + `config.json` + `preprocessor_config.json` | HuggingFace Transformers (image-to-text) | No              |

`inference.json` is the exported PIR program (PaddlePaddle IR, op-level graph —
not useful for us). `inference.yml` is the authoritative config: input specs,
pre/post-processing. `inference.onnx` is the model we consume.

Download a model, e.g.:

```bash
curl -sL https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.onnx -o model/ppocrv6_tiny_rec.onnx
curl -sL https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/main/inference.onnx -o model/ppocrv6_tiny_det.onnx
```

## 4. Recognition model — how it works

### 4.1 I/O (verified from ONNX graphs)

```
INPUT  x           : f32, [N, 3, 48, W_dyn]        # NCHW; H fixed 48, W dynamic
OUTPUT fetch_name_0: f32, [N, T, classes]
```
- Tiny:   `classes = 6906`  (dict len 6904)
- Small:  `classes = 18710` (dict len 18708)
- Medium: `classes = 18710` (dict len 18708)
- `T` shrinks with width (downsampled ~4×; exact expression baked in the graph).
- Classes = dict_len + 2: PaddleOCR CTC convention puts **blank at index 0**;
  the extra trailing index (`dict_len + 1`) is unverified `[verify]`.

### 4.2 Preprocessing (from `inference.yml`)

```yaml
DecodeImage:   { channel_first: false, img_mode: BGR }
MultiLabelEncode: { gtc_encode: NRTRLabelEncode }   # training-only
RecResizeImg:  { image_shape: [3, 48, 320] }
KeepKeys:      [image, label_ctc, label_gtc, length, valid_ratio]  # training-only
```
Inference-essential steps:
1. Decode to **BGR**.
2. **Resize to fixed height 48**, width scaled to keep aspect ratio (bounded
   by the `image_shape` width, typically 320 for a single line; dynamic W is
   supported by the ONNX graph).
3. HWC → CHW.
4. Normalization: **`(x/255 − 0.5) / 0.5`** per channel is the PaddleOCR v3
   default — the `.yml` above does **not** list `NormalizeImage`, so the exact
   expected pixel scale is `[verify]` (the ONNX graph's first op is `Conv`, so
   normalization is **not** baked in; it must be applied by the caller).

### 4.3 Postprocessing

`CTCLabelDecode` — greedy/beam CTC decoding over the logits:
- Take per-timestep argmax, collapse repeated labels, drop blank (index 0),
  map remaining indices 1..=dict_len to the `character_dict` strings.

### 4.4 Dict

The full character dict (49 languages: ASCII, Latin-1/Latin-ext, Greek,
Cyrillic, currency, CJK ~6.7k hanzi, Japanese kana, punctuation, emoji for
small/medium) is embedded in `inference.yml` → `PostProcess.character_dict`.
For `kaptchaocr` this becomes the `Charset` JSON's `charset` list. Small/medium
share one 18,708-char dict; tiny has a 6,904-char dict.

## 5. Detection model — how it works

### 5.1 I/O (verified from ONNX graphs)

```
INPUT  x           : f32, [N, 3, H_dyn, W_dyn]     # NCHW, fully dynamic
OUTPUT fetch_name_0: f32, [N, 1, H_r, W_r]         # DB probability map (1 ch)
```
- Map resolution relative to input is ~1/4 `[verify]` (FPN output + small
  transpose head in the graph).

### 5.2 Preprocessing (from `inference.yml`)

```yaml
DecodeImage:     { channel_first: false, img_mode: BGR }
DetResizeForTest: null                              # default: limit_side_len 736, limit_type max
NormalizeImage:  { mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225],
                   scale: 1./255., order: hwc }
ToCHWImage:      null
```
Inference-essential steps:
1. Decode to **BGR**.
2. **DetResizeForTest**: resize so the *max* side = 736 (`limit_side_len`)
   preserving aspect ratio (no padding).
3. `(x/255 − mean)/std` with ImageNet mean/std (order HWC), then CHW.
4. Feed NCHW f32.

### 5.3 Postprocessing — `DBPostProcess`

```yaml
name: DBPostProcess
thresh: 0.2          # binarize threshold for the prob map
box_thresh: 0.4      # min score of a kept box
unclip_ratio: 1.4    # unclip polygon expansion factor
max_candidates: 3000
```
Steps: binarize map at `thresh` → find connected components → polygon
contours → unclip polygons by `unclip_ratio` → score boxes with
`box_thresh` → (optionally NMS). This is the standard DB (Differentiable
Binarization) postprocess.

## 6. How this maps to the existing `kaptchaocr` pipeline

### 6.1 Recognition — close fit, small deltas

The existing classification path (`preprocess` + `classification_internal` /
`ctc_greedy_decode`) already:
- normalizes official 3-channel models with `(x/255 − 0.5)/0.5`
  (`src/lib.rs` `preprocess`, official branch);
- resizes to a fixed height keeping aspect ratio when `Charset.image == [-1, H]`;
- CTC-decodes word models dropping **blank class 0** (U2 `ctc_greedy_decode`).

To run PP-OCRv6 rec it needs:
- `Charset` JSON with `word: true`, `channel: 3`, `image: [-1, 48]`, and the
  PP-OCRv6 dict as `charset`.
- Height 48 fixed (existing official models are height 32).
- Dynamic width OK if `image[0] == -1` (keeps aspect ratio) — matches the
  dynamic-W ONNX input.
- `classes = dict_len + 2` — the decoder must tolerate the trailing extra
  index (`[verify]`).
- **Channel order: BGR vs RGB.** PP-OCRv6 is trained on BGR (`DecodeImage
  img_mode: BGR`); the existing official path feeds RGB. Expect to feed BGR
  for PP-OCRv6 (`[verify]` empirically on a known sample).
- Normalization scale `[verify]` (§4.2).

### 6.2 Detection — now supported via a DB postprocess path

The bundled `stock_det.onnx` is an **anchor-based YOLO-style** detector:
fixed 320/416 padded input, raw 0–255 BGR, output `[1, i, 6]`
(cx,cy,w,h,obj,cls) decoded over `STRIDES` grids (`src/lib.rs`
`detection_with_options`). 

PP-OCRv6 det is a **DB detector** (dynamic size, normalized input, 1-channel
probability-map output). Implemented in `src/lib.rs`:
- `DetType::{Yolo,Db}` + `DbParams`; `detection_*` dispatches on the type.
- `detection_db` + `db_decode_prob_map`: resize longest side → 736, **center
  the content in a 736×736 padded square** (this export produces stable
  internal `Resize` shapes only at a fixed input; variable sizes raise ORT
  buffer-reuse errors), BGR + ImageNet normalize, then threshold → connected
  components → bbox → score filter (`box_thresh`) → unclip (`unclip_ratio`)
  → map back to original coords → NMS. This is a simplified axis-aligned
  approximation of `DBPostProcess` (no rotated polygons).
- Configure any det model (YOLO or DB) from `model/models.json` via the
  manifest (`--manifest`), no code changes. See `model/README.md`.

Verified on host 2026-08-12: PP-OCRv6 tiny/small/medium det run through the
manifest and return text-region boxes on sample images.

### 6.3 Feature notes

- CPU threads / concurrency: orthogonal — existing `--num-threads` /
  `--max-concurrency` apply unchanged.
- The PP-OCRv6 rec and det models are **word/CTC** and **DB** respectively —
  exactly the two new model "types" the Phase 3 manifest + Phase 1 U2 CTC
  decode were built for.

## 7. Integration checklist (for the follow-up work)

- [x] Download the 6 `_onnx` repos into `model/` (see §3) — about 176 MB total.
- [x] Export rec `character_dict` from each `inference.yml` into `Charset`
  JSONs (tiny 6,904 / small+medium 18,708).
- [x] Wire rec: `word: true`, `channel: 3`, `image: [-1, 48]` + verify
  normalization and BGR vs RGB on known samples.
- [x] Implement DB postprocess for det (see §6.2) + manifest
  (`model/models.json`, `--manifest`).
- [ ] Benchmark all six tiers with the eval harness (Phase 3 item 1) and
  compare against the bundled stock models — rec done 2026-08-12
  (PP-OCRv6 medium ~90% char-acc vs ~72% stock on synthetic Chinese-English); det
  comparison still TODO (needs box-vs-ground-truth scoring in the harness).

## 8. References

- Collection: https://huggingface.co/collections/PaddlePaddle/pp-ocrv6
- Paper: arXiv:2606.13108 "PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks"
- PaddleOCR repo: https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR docs: https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html
