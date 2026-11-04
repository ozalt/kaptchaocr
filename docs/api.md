# HTTP API

The server exposes JSON routes plus multipart upload variants. All JSON image
fields are standard base64 strings; do not add a `data:image/...;base64,`
prefix. Request bodies are limited to 50 MB.

```sh
# start everything the benchmarks use
kaptchaocr --address 127.0.0.1:8000 --ocr --det --slide \
  --manifest model/models.json --profile balanced
```

## Routes

| Route | Method | Enabled by | Purpose |
| --- | --- | --- | --- |
| `/status` | GET | always | Service status + enabled features |
| `/healthz` | GET | always | Liveness probe (plain `ok`) |
| `/metrics` | GET | always | Prometheus text metrics |
| `/docs` | GET | always | Swagger UI |
| `/docs.json` | GET | always | OpenAPI document |
| `/ocr` | POST | `--ocr` | Recognize one image |
| `/ocr/upload` | POST | `--ocr` | Same, as `multipart/form-data` |
| `/solve` | POST | `--ocr` | OCR + arithmetic answer |
| `/det` | POST | `--det` | Bounding boxes |
| `/det/upload` | POST | `--det` | Same, as `multipart/form-data` |
| `/solve/click` | POST | `--ocr --det` | Click-captcha points in prompt order |
| `/slide_match` | POST | `--slide` | Template slider matching |
| `/slide_comparison` | POST | `--slide` | Gap comparison |
| `/slide/solve` | POST | `--slide` | Match + drag distance / trajectory |
| `/mcp` | POST | `--mcp` / `--only-mcp` | MCP JSON-RPC |

All JSON routes use the envelope:

```json
{"code": 200, "msg": "success", "data": {}}
```

`/healthz` returns plain `ok`; `/metrics` returns Prometheus text; `/docs` is
the Swagger UI. Errors use the envelope with a non-200 `code` and a
human-readable `msg`.
When the server runs with `--api-key`, every route except `/status`,
`/healthz`, `/metrics` and the docs requires
`Authorization: Bearer <key>` or `x-api-key: <key>`.

## Status and health

```sh
curl -s http://127.0.0.1:8000/status
```

```json
{"code":200,"msg":"success","data":{"service_status":"running","enabled_features":["ocr","det","slide"]}}
```

`/healthz` returns `ok`; `/metrics` returns Prometheus counters/gauges such as
`kaptchaocr_requests_total`.

## OCR

`POST /ocr` — recognize a single-line CAPTCHA.

| Field | Type | Description |
| --- | --- | --- |
| `image` | string | Base64 image (required) |
| `png_fix` | bool | Treat transparent PNG backgrounds as white |
| `probability` | bool | Return the full probability matrix + charset |
| `charset_range` | string | Restrict candidates; see [OCR](ocr.md#charsets) |
| `color_filter` | string / array | Preset color name(s) or HSV ranges; see [OCR](ocr.md#color-filters) |

```sh
IMG="$(base64 < captcha.png | tr -d '\n')"
curl -s http://127.0.0.1:8000/ocr \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\"}"
```

```json
{"code":200,"msg":"success","data":{"text":"8A62N1","probability":null,"charset":null,"confidence":null}}
```

## Solve

`POST /solve` — OCR plus arithmetic solving for text/math CAPTCHAs. Same fields
as `/ocr`, plus `math` (`auto` default, or `off`).

```sh
curl -s http://127.0.0.1:8000/solve \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\"}"
```

```json
{"code":200,"msg":"success","data":{"text":"5+?=8","math":{"expression":"5+?=8","result":3.0,"answer":"3"},"probability":null,"charset":null,"confidence":null,"model":"ft-small"}}
```

See [Solve API](solve.md#text-and-math-captchas-post-solve) for the supported
expression forms.

## Detection

`POST /det` — return bounding boxes and confidence scores.

```sh
curl -s http://127.0.0.1:8000/det \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\"}"
```

```json
{"code":200,"msg":"success","data":{"bboxes":[[80,3,98,21],[56,6,76,25]],"scores":[0.98,0.96]}}
```

## Click captchas

`POST /solve/click` — detect glyphs, recognize each box, and order points for
the prompt. Requires `--ocr --det`.

```json
{
  "image": "base64...",
  "prompt": "请依次点击【B,l,R,T】",
  "score_threshold": 0.3,
  "charset_range": "BlRT"
}
```

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "prompt": "BlRT",
    "points": [{"x": 42.0, "y": 18.0}, {"x": 81.0, "y": 22.0}],
    "boxes": [
      {"x1": 30, "y1": 5, "x2": 54, "y2": 31, "ch": "B", "score": 0.99}
    ],
    "unmatched": []
  }
}
```

See [Solve API](solve.md#click-captchas-post-solveclick).

## Slider matching

`POST /slide_match` — target image + background image:

```json
{
  "target_image": "base64...",
  "background_image": "base64...",
  "simple_target": true
}
```

```json
{"code":200,"msg":"success","data":{"target":[215,45,261,91],"target_x":0,"target_y":45}}
```

`POST /slide_comparison` — two images where one has the gap; returns
`{"x": 144, "y": 76}`.

`POST /slide/solve` — match + drag distance and optional human-like trajectory:

```json
{
  "target_image": "base64...",
  "background_image": "base64...",
  "handle_x": 20,
  "handle_y": 40,
  "offset": 0,
  "duration_ms": 480,
  "seed": 7,
  "trajectory": true
}
```

```json
{"code":200,"msg":"success","data":{"target":[140,69,200,124],"target_x":0,"target_y":0,"gap_x":140,"gap_y":69,"distance":120.0,"duration_ms":482.0,"trajectory":[{"x":0.06,"y":-0.02,"t":0.0},{"x":1.2,"y":0.3,"t":16.6}]}}
```

See [Solve API](solve.md#slider-captchas-post-slidesolve).

## Multipart uploads

`/ocr/upload` and `/det/upload` accept `multipart/form-data` with the image in
the first file field:

```sh
curl -s http://127.0.0.1:8000/ocr/upload -F "file=@captcha.png"
```

Optional form fields for `/ocr/upload`: `png_fix`, `probability`,
`charset_range`, `color_filter` (JSON-encoded string).

## API key, limits and docs

```sh
kaptchaocr --ocr --api-key secret --rate-limit 20 --max-concurrency 4
```

- `--rate-limit N` allows N requests per IP per 10 seconds (0 = unlimited).
- `--max-concurrency N` caps simultaneous inference tasks (0 = CPU count).
- Swagger UI: <http://127.0.0.1:8000/docs>

The full request/response schemas are served at `/docs.json`.
