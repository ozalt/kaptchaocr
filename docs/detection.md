# Detection

The detector returns rectangular boxes for text regions or glyphs. It is used
directly through `/det`, and internally by `/solve/click`.

> This is not general object detection. It finds text/character boxes, not cars,
> trucks or other objects. Picking pictures in a CAPTCHA is still work in
> progress and is not supported yet.

```rust
let image = std::fs::read("captcha.png").unwrap();
let mut det = kaptchaocr::kaptchaocr_detection().unwrap();
let boxes = det.detection(image).unwrap();
println!("{:?}", boxes);
```

Each box is `{x1, y1, x2, y2}` in image pixels. `/det` also returns a confidence
score for every box.

## Model choice

Without a manifest, the server uses the bundled ddddocr detector
(`stock_det.onnx`). For click captchas, use the glyph-level fine-tuned models:

```sh
kaptchaocr --ocr --det --manifest model/models.json \
  --det-model ft-small-det --rec-model ft-small
```

- `ft-small-det` — best speed/quality tradeoff; F1@0.5 **95.8%** on the
  fresh 400-image benchmark at 270 ms/image.
- `ft-medium-det` — same F1@0.5, stronger at IoU 0.75 (**63.1%** vs 50.5%),
  but ~5× slower.
- `stock-det` — bundled fallback; 57.5% F1@0.5.
- PP-OCRv6 base detectors are **line-level** models; they are not a drop-in
  replacement for per-glyph click detection.

Numbers and charts: [Benchmarks](../benchmarks/README.md).

## HTTP

`POST /det` with `{"image": "base64..."}` returns:

```json
{"code":200,"msg":"success","data":{"bboxes":[[80,3,98,21],[56,6,76,25]],"scores":[0.98,0.96]}}
```

Multipart upload: `POST /det/upload` with a `file` field.

## Click captchas

Detection alone gives boxes without characters. `/solve/click` combines
detection + per-box recognition + prompt ordering; see
[Solve API](solve.md#click-captchas-post-solveclick).
