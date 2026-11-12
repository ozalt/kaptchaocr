# Solve API

`/solve`, `/slide/solve` and `/solve/click` are higher-level endpoints for
scraping workflows: one request returns a ready-to-use answer instead of raw
model output.

## Text and math captchas: `POST /solve`

Requires `--ocr`. Same fields as `/ocr`, plus `math`:
`auto` (default) or `off`.

```json
{
  "image": "base64...",
  "math": "auto",
  "charset_range": "0123456789+-x/=",
  "png_fix": false
}
```

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "text": "5+?=8",
    "math": {"expression": "5+?=8", "result": 3.0, "answer": "3"},
    "probability": null,
    "charset": null,
    "confidence": 0.941,
    "model": "ft-small"
  }
}
```

`math` is `null` when the text is not an expression, or when the text contains
no binary operator (`+ - * /`). That guard prevents OCR damage such as
`113=?` (a dropped minus sign) from becoming a confident but wrong answer.
Supported forms include:

- `3+5=`, `9÷3`, `4x6`, full-width `４×６＝`
- single-unknown equations: `5+?=8`, `?×6=42`, `12/?=4`
- non-integral results keep up to two decimals (`10/3` → `3.33`)
- Chinese numerals/operators: `九乘六等于？` → `54`,
  `十二减四等于几` → `8`, `五加几等于八` → `3`
- traditional `減`/`幾` and compounds such as `两千`, `一百零三`

`model` is the loaded recognition model name when the server was started with
`--manifest`.

## Slider captchas: `POST /slide/solve`

Extends `/slide_match` with the slider handle position, calibration, and
trajectory generation.

| Field | Description |
| --- | --- |
| `target_image`, `background_image` | Base64 images |
| `simple_target` | Use simple template matching |
| `handle_x`, `handle_y` | Handle start position on the page |
| `offset` | Calibration added to the gap x |
| `duration_ms` | Drag duration; derived from distance by default |
| `seed` | Reproducible trajectory |
| `trajectory` | Include trajectory (default `true`) |

`distance = gap_x + offset - handle_x` (or just the gap x when no handle is
given). `trajectory` points are **deltas relative to the handle start**, sampled
at ~60 Hz, with the last point exactly at `(distance, 0)`:

```python
for p in trajectory:
    mouse.move(handle_x + p.x, handle_y + p.y)
```

```json
{
  "target_image": "base64...",
  "background_image": "base64...",
  "handle_x": 20,
  "handle_y": 40,
  "seed": 7
}
```

## Click captchas: `POST /solve/click`

Requires both recognition and detection (`--ocr --det`). Pipeline: detect glyph
boxes → recognize each box → order clicks according to the prompt.

| Field | Description |
| --- | --- |
| `image` | Base64 image |
| `prompt` | e.g. `请依次点击【明,天,会】` or `click in order: A B C` |
| `score_threshold` | Minimum detection score (default `0.3`) |
| `charset_range` | Constrain single-glyph recognition |

```json
{
  "image": "base64...",
  "prompt": "请依次点击【B,l,R,T】",
  "score_threshold": 0.3,
  "charset_range": "BlRT"
}
```

Response fields:

- `prompt` — parsed target characters, in click order
- `points` — click coordinates (box centres, relative to the image)
- `boxes` — every detected glyph box with recognized character and score
- `unmatched` — prompt characters that could not be matched

When `prompt` is present and `charset_range` is not, recognition is restricted
to the unique prompt characters. Boxes are then assigned globally by model
score (not first-match), so an early target cannot take the box a later target
needs. `boxes[].ch` therefore uses the prompt's character set. Without
`prompt`, only the boxes are returned and recognition uses the full charset.

For click solving, use the fine-tuned glyph detector + recognizer:

```sh
kaptchaocr --address 127.0.0.1:8000 --ocr --det \
  --manifest model/models.json --det-model ft-small-det --rec-model ft-small
```

`ft-small-det` scores **95.8% F1@0.5** on the fresh 400-image benchmark;
`ft-medium-det` is ~5× slower but tighter at IoU 0.75 (63.1% vs 50.5%). See
[Benchmarks](../benchmarks/README.md).

The prompt-derived charset is automatic. Pass `charset_range` only when you
want to override it (for example, to include a confusable character). The
fine-tuned PP-OCRv6 CTC models support it.
