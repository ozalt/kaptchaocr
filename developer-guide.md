# kaptchaocr (Rust) — Developer Guide

This document explains what lives in this repository and how every piece works
technically. It is written against the current state of `master`.

## 1. What This Repo Is

`kaptchaocr` is a Rust re-implementation of the popular Python CAPTCHA
recognition library of the same name. The crate contains two artifacts in one
repository:

1. **`kaptchaocr` — the library** (`src/lib.rs`): CAPTCHA OCR (classification),
   object detection (for "click-to-select" captchas), slide-captcha matching,
   color filtering, and per-call character-set restriction. Inference runs
   through ONNX Runtime via the `ort` crate.
2. **The API server binary** (`src/main.rs`): an HTTP server built on
   the Salvo web framework exposing the library through a REST API, a
   hand-rolled MCP (Model Context Protocol) endpoint, OpenAPI/Swagger docs,
   and optional ACME (Let's Encrypt) TLS.

It ships with three pre-trained ONNX models (embedded inside the binary by
default) so deployment is a single cross-platform executable with no OpenCV
or Python dependency.

## 2. Repository Layout

```
.
├── Cargo.toml                      # Crate/feature/dependency definitions
├── Dockerfile                      # Multi-stage image: Alpine/musl build stage + runtime (:8000)
├── .dockerignore                   # Keeps target/ and ppocrv6 models out of the build context
├── .github/workflows/build.yml    # 13-platform CI build + release workflow
├── initialize.json                 # MCP `initialize` response (compiled into binary)
├── list.json                       # MCP `tools/list` response (compiled into binary)
├── model/
│   ├── models.json                 # Model manifest + fast/balanced/accurate profiles
│   ├── stock_common.onnx      (~54 MB)   # New OCR (classification) model
│   ├── stock_common.json                 # Charset/config for stock_common.onnx
│   ├── stock_old.onnx  (~14 MB)   # Older OCR model
│   ├── stock_old.json             # Charset/config for stock_old.onnx
│   ├── stock_det.onnx  (~20 MB)   # Object-detection model (YOLO-style, 1 class)
│   ├── finetuned/                  # PP-OCRv6 rec tiny/small/medium fine-tuned on captchas
│   └── ppocrv6/                    # Base PP-OCRv6 rec/det exports (not needed in images)
├── src/
│   ├── lib.rs                      # The kaptchaocr library
│   ├── manifest.rs                 # ONNX model manifest (`models.json`) loader
│   ├── math.rs                     # Arithmetic captcha parser/solver (ASCII + Chinese)
│   ├── trajectory.rs               # Human-like slider drag trajectory generator
│   └── main.rs                     # The API server binary
├── clients/python/                 # Zero-dependency Python client + Playwright examples
├── image/                          # Sample images used by the unit tests
├── test_api.py                     # Python smoke test hitting every REST + MCP endpoint
├── playwright-cookbook.md          # End-to-end browser recipes (text/math/slider)
├── developer-guide.md / paddleOCR.md
└── README.md                       # User documentation
```

## 3. Cargo Features (Cargo.toml)

| Feature            | Default | Effect |
|--------------------|---------|--------|
| `inline-model`     | ✅      | Embed the three ONNX models + charsets into the binary via `include_bytes!`/`include_str!`. Without it, models are read from a `model/` directory at startup. |
| `download-binaries`| ✅      | Auto-download ONNX Runtime native libs at build time (via `ort`). |
| `load-dynamic`     | ❌      | Load a user-supplied ONNX Runtime `.so`/`.dll` at runtime (`KaptchaOcr::set_onnxruntime_path`). |
| `cuda`             | ❌      | Enable the CUDA execution provider (only in the library; the server binary does not expose CUDA). |

Note: `ort` is pulled from a **git fork** (`https://github.com/86maid/ort.git`)
with features `ndarray`, `half`, `copy-dylibs`, not from crates.io. There is no
`rev`/`tag` pin on that dependency.

## 4. Library (`src/lib.rs`)

### 4.1 Module-level model initializers

Feature-gated convenience constructors:

- `kaptchaocr_classification()` → `KaptchaOcr` with `stock_common.onnx` + `stock_common.json`.
  Panics if `--ocr-path` models are expected but inline-model is off and files
  are missing (the non-inline variants use `std::fs::read` directly).
- `kaptchaocr_classification_old()` → old model + charset.
- `kaptchaocr_detection()` → `stock_det.onnx` only (no charset).
- `*_cuda(device_id)` variants require the `cuda` feature.

### 4.2 The `KaptchaOcr` struct

```rust
pub struct KaptchaOcr<'a> {
    diy: bool,                          // true for any model other than the two official ones
    session: ort::Session,              // ONNX Runtime session (models load from memory)
    charset: Option<Cow<'a, Charset>>,  // charset JSON; None for detection sessions
    charset_range: Vec<String>,         // restrict probability output (set via set_ranges)
}
```

- `unsafe impl Send/Sync` — ONNX Runtime sessions are thread-safe; this allows
  sharing one session across `tokio::task::spawn_blocking` threads.
- Constructors: `new`/`new_ref`, `new_model`, `with_model_charset`,
  `with_model`, plus `*_cuda` variants. All load models from memory buffers
  (`commit_from_memory`); path variants just `fs::read` first.
- `is_diy(model)`: computes the SHA-256 of the model bytes and compares
  against two hard-coded hashes (of `stock_common.onnx` and `stock_old.onnx`).
  Anything else is treated as a "DIY"/custom model — any third-party
  ONNX + charset pair — which changes preprocessing + decoding behavior.
- `set_onnxruntime_path` (feature `load-dynamic` only) uses
  `catch_unwind`/`panic::set_hook` to convert ORT's panic-on-load into an
  `anyhow::Error`.

### 4.3 `Charset` and config JSON

```rust
pub struct Charset {
    pub word: bool,      // is it a "word" (CTC-sequence) CNN model?
    pub image: [i64; 2], // [width, height]; width == -1 => auto width (keeps aspect ratio), height must be a multiple of 16
    pub channel: i64,    // 1 = grayscale, 3 = RGB
    pub charset: Vec<String>, // label list (index = class id)
}
```

### 4.4 The classification (OCR) pipeline

Both `classification()` and `classification_probability()` funnel into
`classification_*_with_options()`. Steps:

1. **Load** the image (`image::load_from_memory`), optionally after a
   `ColorFilter::filter()` pass.
2. **Resize** according to `charset.image` with Lanczos3:
   - fixed `[w, h]` → `resize_exact(w, h)`;
   - `[-1, h]` + `word` → square `h × h`;
   - `[-1, h]` otherwise → keep aspect ratio, height `h`.
3. **Channel conversion**: `channel == 1` → grayscale bytes; `png_fix` →
   fill transparent pixels with white via `png_rgba_black_preprocess`, then
   RGB; otherwise straight RGB.
4. **Tensor**: `(1, channel, height, width)` as `f32`, NHWC→NCHW via
   `from_shape_vec`.
5. **Normalization** per pixel:
   - official model: `(x/255 − 0.5) / 0.5`;
   - DIY model: ImageNet-style mean/std `(x/255 − μ)/σ` with channel-specific
     μ/σ (grayscale uses the "G" mean/std pair).
6. **Inference**: `session.run(ort::inputs![tensor])`.
7. **Decoding** by model kind (shared `classification_internal_conf`
   returns `Vec<(class_idx, confidence)>`; `classification_internal`
   maps to text, `classification_with_confidence` exposes `(String, f32)`):
   - `word == true` (**CTC**): proper CTC greedy decode. If any output
     extracts as `f32` logits, use it (`ctc_greedy_decode`: softmax over
     class axis, argmax per timestep, collapse consecutive duplicates,
     drop blank class `0`; per-char confidence = **max** softmax prob of
     the collapsed run). Otherwise fall back to the `i64` argmax output
     (`ctc_greedy_decode_indices`, confidence `1.0`). Supports `[1, T, C]`
     and `[T, C]` output shapes.
   - official non-word: output index 0 is `f32` per-position logits; per row,
      argmax → class id (also returns the row's softmax of argmax as
      confidence); greedy dedup: drop class id `0` and consecutive duplicates.
   - DIY non-word: same but on the `u32` output (marked "todo: untested" in
     the source; confidence reported as `1.0`).

### 4.5 Probability output

`classification_probability()` returns `CharacterProbability`:

```rust
pub struct CharacterProbability {
    pub text: Option<String>,
    pub charset: Vec<String>,       // column labels for `probability`
    pub probability: Vec<Vec<f32>>, // per character-position, per label
    pub confidence: Option<f64>,
}
```

- `get_text()` lazily applies argmax per position.
- `get_confidence()` lazily averages per-position max probability.
- Softmax is computed with `mapv(f32::exp)`, divided by the sum over the
  whole tensor (numerically redundant), then each row is re-normalized by its
  own sum (`sum_axis(Axis(2))`).
- `set_ranges()` / `calc_ranges()` restrict the columns returned. Unavailable
  labels get probability `−1.0`. Range selection happens in either
  `classification_probability_with_options` (per request) or precomputed via
  `calc_ranges` (global).
- **Panics on DIY models**: `classification_probability*` calls
  `panic!("can only use the official model")` if `diy`.

### 4.6 `CharsetRange`

Alphabet restriction abstraction:

- `0`…`7` map to fixed built-in presets via `From<i32>` (see README table):
  Digit, Lowercase, Uppercase, Lower+Upper, Lower+Digit, Upper+Digit,
  Lower+Upper+Digit, and "default charset minus alnum".
- Any string (`From<&str>`) → `Other(String)` i.e. a custom character set.
- `Charset(vec)` → already-computed list.
- `calc_ranges` dedups via `HashSet` (⚠️ non-deterministic ordering) and always
  appends `""` (the blank/unknown class).

### 4.7 Color filter

`ColorFilter` is an untagged enum accepting either:

- `Color` presets (`"red"`, `"green"`, …) — mapped to fixed HSV ranges
  (`IntoHsvRange`), e.g. red uses `[(0,50,50),(10,255,255)]` +
  `[(170,50,50),(180,255,255)]` because hue wraps;
- `[Color; N]` / `Vec<Color>` / `Vec<String>` mixes;
- raw `Vec<((u8,u8,u8),(u8,u8,u8))>` HSV ranges.

`ColorFilter::filter()` converts RGB→HSV in pure Rust (per-pixel loops with
`ndarray` `Array3<u8>`), builds a boolean mask from the HSV ranges, and
outputs `RgbImage` where non-matching pixels become white. It is applied
before classification — matching pixels are kept, so "filter green" means
"recognize only the green parts".

### 4.8 Object detection within `detection()`

This is a YOLOv5-style single-class detector (`stock_det.onnx`):

1. **Letterbox**: scale so the shorter side fits 416 (triangle filter), pad
   with gray `[114,114,114]` to 416×416.
2. **Tensor** `(1,3,416,416)`, filled BGR (OpenCV convention).
3. **Decode** over strides `[8, 16, 32]` (grid totals `52² + 26² + 13²`):
   `score = obj_conf × cls_conf`, skip `score < SCORE_THR` (0.1); convert
   `(x,y,cx,cy,tw,th)` → box via `exp(w/h) × stride`; divide by the resize
   ratio `r` back to original pixel space.
4. **Greedy NMS** with `NMS_THR = 0.45`: repeatedly take the highest-score
   box, drop boxes with IoU above the threshold.
5. **Clamp** to image bounds; return `Vec<BBox>`.

⚠️ Note: `GRIDS`/`EXPANDED_STRIDES` lazy statics (`lazy_static` +
`transmute_copy`) are computed at startup for the decode, but `detection()`
recomputes grid offsets inline — those statics are currently **dead code**.

### 4.9 Slide matching

All slide endpoints are imagery algorithms (not neural networks):

- `slide_match()` — for PNG sliders with transparent background:
  - crops the target to its opaque bounding box (records the crop offset as
    `target_x`/`target_y`);
  - Canny edge detection (thresholds 100/200) on both target and background;
  - normalized cross-correlation template matching (`imageproc::match_template`);
  - `find_extremes()` → max location → `SlideBBox { x1,y1,x2,y2, target_x, target_y }`.
- `simple_slide_match()` — same but skips the alpha-crop (for JPG/BMP
  sliders that have little background).
- `slide_comparison()` — for the "one full image vs the same image with the
  gap filled" variant:
  - requires identical image dimensions (`anyhow::ensure`);
  - pixel-wise diff `> 80` → white, else black (per-byte `abs_diff`);
  - scans columns left→right; the first column containing ≥ 5 changed pixels
    yields `x` (`i + 2`) and `y` (`j − 5`).
- `*_with_path()` variants just `fs::read` then delegate.

### 4.10 `classification_bbox()`

Crops each `BBox` from the source image, re-encodes the crop to PNG in
memory, and runs `classification()` on the bytes; returns `Vec<(BBox, String)>`.

### 4.11 Unit tests

`cargo test --no-default-features --features download-binaries` runs tests
against the `image/` samples: classification (new/old/probability/filter),
detection (writes `output1.jpg`/`output2.jpg` with red boxes), bbox OCR,
slide match, comparison, the math solver and the trajectory generator.

### 4.12 `math` module (`src/math.rs`)

`math::solve(text) -> Option<MathSolution>` turns an OCR string into an
arithmetic answer. `normalize()` maps full-width characters, OCR aliases
(`x`/`X`/`*`, `÷`, `＝`, …) and Chinese numerals/operator words (`九乘六等于？`,
`十二减四等于几`) into an ASCII expression; a small recursive-descent parser
(`expr → term → factor → primary`) evaluates it with standard precedence, and
a single `?` (`5+?=8`, `12/?=4`) is solved by inverting the AST. Results are
rounded to ≤2 decimals; `answer` is the string a client types. The module is
also reachable through `POST /solve`.

### 4.13 `trajectory` module (`src/trajectory.rs`)

`trajectory::human_trajectory(distance, duration_ms, seed)` samples a
slider-drag path at ~60 Hz: smootherstep easing, a small overshoot near the
end, decaying tremor and jitter, with the last point exactly
`(distance, 0)`. Points are deltas from the handle start; the same seed
reproduces the same path. `default_duration_ms = clamp(350 + 1.1·d, 300, 1500)`.
Served by `POST /slide/solve`.

## 5. The Server Binary (`src/main.rs`)

### 5.1 CLI (`clap` derive)

| Flag | Default | Notes |
|---|---|---|
| `--address` | `0.0.0.0:8000` | bind address |
| `--mcp` | off | enable MCP endpoint; mutually exclusive with `--only-mcp` |
| `--only-mcp` | off | run only MCP, hide REST routes (except `/status`); mutually exclusive with `--mcp` |
| `--ocr` | off | new classification model; exclusive with `--old` |
| `--old` | off | old classification model |
| `--det` | off | detection model |
| `--slide` | off | slide_match / slide_comparison |
| `--ocr-charset-range` | unset | global default charset (0–7 = preset, else custom string) |
| `--ocr-path` | `model/stock_common.onnx` | overrides model+charset path (non-inline builds) |
| `--det-path` | `model/stock_det.onnx` | overrides det model path (non-inline) |
| `--acme <domain>` | off | obtain a TLS cert from Let's Encrypt for that domain |
| `--max-concurrency` | CPU cores | max concurrent ONNX inference tasks (0 = unlimited) |
| `--rate-limit` | 0 (off) | requests allowed per IP per 10 s |
| `--api-key` | unset | require `Authorization: Bearer <key>` or `x-api-key: <key>` |
| `--num-threads` | 0 (ORT default) | ORT `IntraOpNumThreads` per inference |
| `--config` | unset | optional TOML config; CLI args win |
| `--manifest` | unset | load rec/det models from `model/models.json` (preferred path) |
| `--profile` | unset | manifest profile: `fast` / `balanced` / `accurate` (needs `--manifest`) |
| `--rec-model` | unset | pick a manifest entry by name; wins over `--profile` |
| `--det-model` | unset | pick a manifest **det** entry by name (e.g. `ft-small-det`) |

If no feature flag is given at all, everything is enabled (`ocr + old + det
+ slide`) with a warning.

### 5.2 Global singletons

- `ARGS: OnceLock<Args>` — parsed CLI args (also read by `route_status`).
- `OCR: OnceLock<KaptchaOcr>` — classification session (new or old).
- `DET: OnceLock<KaptchaOcr>` — detection session.
- `CACHE: LazyLock<Mutex<LruCache<String, Vec<String>>>>(64)` — memoizes
  `calc_ranges` results keyed by the raw charset string so repeated
  `charset_range` requests skip recomputation.
- `MODEL_NAME: OnceLock<String>` — the loaded rec model's manifest entry name
  (or `stock-common` / `stock-old` / file stem); returned by `/solve`.

### 5.3 Routing

Salvo router with the `Logger` hoop:

- `GET  /status` — `{code, msg, data:{service_status, enabled_features}}`.
- `GET  /healthz` — plain `ok` liveness probe; `GET /metrics` — Prometheus text.
- `POST /ocr` — base64 `image` (+ `png_fix`, `probability`,
  `charset_range`, `color_filter`).
- `POST /ocr/upload`, `POST /det/upload` — multipart file variants.
- `POST /solve` — `/ocr` + arithmetic answer; `math` is `auto` (default) or
  `off`, response adds `math{expression,result,answer}` and `model`.
- `POST /det` — base64 `image` → `bboxes` + `scores`.
- `POST /slide_match` — `target_image`, `background_image`,
  `simple_target` → `target: [x1,y1,x2,y2]` + `target_x/y`.
- `POST /slide_comparison` — two base64 images → `{x, y}`.
- `POST /slide/solve` — slide match + `handle_x/handle_y/offset/duration_ms/
  seed/trajectory` → `gap_x/gap_y/distance/trajectory` (relative deltas).
- `POST /mcp` — JSON-RPC/MCP (see below).
- `GET  /docs.json` + `GET /docs` — OpenAPI + Swagger UI.

`/solve` is registered with `--ocr`/`--old`, `/slide/solve` with `--slide`.

REST routes are only registered when their feature flag is on and
`--only-mcp` is absent. All heavy work runs in `tokio::task::spawn_blocking`
to keep the async runtime responsive (ORT session shared across threads via
the `unsafe Sync` impl).

Response envelope for every REST call:

```json
{ "code": 200, "msg": "success", "data": { ... } }
```

Errors (invalid base64, bad JSON, model errors) fall through to
`default_error_handler`, which serializes the Salvo error into the same
envelope with the HTTP error code, via a custom `Catcher`.

### 5.4 MCP endpoint (`route_mcp`)

A hand-rolled JSON-RPC 2.0 over HTTP server (uses `rmcp` **only** for the
message/error types, not for serving):

- `initialize` → the static `InitializeResult` compiled from
  `initialize.json` (protocol 2025-03-26, serverInfo kaptchaocr 1.0.0).
- `tools/list` → `ListToolsResult` built from `list.json`, **filtered at
  startup to the enabled features** (`ocr`/`old` → ocr, `det` → det,
  `slide` → slide_match + slide_comparison); MCP tool names use underscores
  (`slide_match`, `slide_comparison`), matching the REST route names.
- `tools/call` → builds a synthetic Salvo `Request` containing the tool
  arguments as JSON body and **re-dispatches to the regular REST handlers**
  (`route_ocr.handle(...)` etc.). If the tool's feature is disabled, returns
  a JSON-RPC error `"the tool exists, but the server is not enabled"`. The
  `APIResponse<Value>` body is then converted to `CallToolResult::success(text)`
  or `::error(text)` (errors also include any Salvo error body, e.g. 500s).
- Notifications (e.g. `notifications/initialized`) are silently ignored.
- Unknown methods → `METHOD_NOT_FOUND` JSON-RPC error.

### 5.5 HTTP details

- `request::set_global_secure_max_size(50 * 1024 * 1024)` guards the body size.
- ACME mode: `TcpListener::new(...).acme().add_domain(domain).bind()` — TLS via
  rustls, auto cert obtain/renew.
- Logging: `tracing_subscriber` with `RUST_LOG` env filter (default `info`);
  ANSI enabled only when the terminal supports it.

## 6. CI/CD (`.github/workflows/build.yml`)

Manual `workflow_dispatch` with inputs `version`, `description`, `publish`.
Thirteen nearly identical build jobs producing zips for:

- Windows `i686` / `x86_64`-msvc (inline + non-inline)
- Linux `x86_64`-gnu (inline + non-inline)
- macOS `x86_64` (inline + non-inline)
- AArch64 Linux (inline + non-inline)
- AArch64 macOS (inline + non-inline)
- `x86_64`-linux-musl via the `Dockerfile` (inline + non-inline)

Non-inline zips bundle the `model/` folder next to the executable. A final
`release` job (gated on `publish == true`) creates a GitHub Release
(`actions/create-release@v1` + `softprops/action-gh-release@v1`) and attaches
all 13 zips.

The `Dockerfile` is **multi-stage**: stage 1 is the Alpine build environment
(rust:1.90.0-alpine + `build-base`, `musl-dev`, `openssl-libs-static`,
`cmake`, …) — also used directly by the musl CI jobs — and stage 2 is a small
runtime image holding the binary plus `model/stock_*` and
`model/finetuned/`, defaulting to
`--ocr --det --slide --manifest model/models.json --profile balanced`
(`HEALTHCHECK` on `/healthz`). `.dockerignore` keeps `target/` and
`model/ppocrv6/` out of the build context.

## 7. Python tooling

### 7.1 `test_api.py`

Hits every endpoint once a server is running on `127.0.0.1:8000`:
`/status`, `/ocr` (plain, + color filter string/array/HSV, + charset_range,
+ probability), `/solve` (math + `math:"off"`), `/det`, `/slide_match`
(simple + full), `/slide_comparison`, `/slide/solve`, and MCP `initialize` /
`tools/list` / `tools/call` / `notifications/initialized`. For each case it
prints the equivalent `curl` command and the raw response.

### 7.2 `clients/python/` (client library)

Zero-dependency (stdlib `urllib`) synchronous client:

- `kaptchaocr/client.py` — `Client` with `solve`, `solve_file`, `ocr`, `det`,
  `slide_match`, `slide_comparison`, `slide_solve`, `status`, `healthz`;
  automatic retries for connection errors / 5xx / 429 (exponential backoff,
  `Retry-After` respected) and `x-api-key` support. The HTTP transport is
  injectable, which is what the tests use.
- `kaptchaocr/models.py` / `errors.py` — typed dataclasses and
  `ServerError` / `NetworkError`.
- `tests/test_client.py` — 15 offline `unittest` tests
  (`python3 -m unittest discover`), no server required.
- `examples/playwright_text.py` / `playwright_slider.py` — browser recipes.

### 7.3 `playwright-cookbook.md`

End-to-end recipes: model/profile selection (including why Chinese captchas
need the stock model), text captcha loop, math (ASCII + Chinese), slider drag
replay and `offset` calibration, retry etiquette, a Scrapy note, and a
troubleshooting table.

## 8. Model files

| File | Size | Purpose |
|---|---|---|
| `stock_common.onnx` | 54 MB | Primary OCR classification model (charset `stock_common.json`) |
| `stock_old.onnx` | 14 MB | Legacy OCR model (`stock_old.json`) |
| `stock_det.onnx` | 20 MB | Detection model: 416×416 YOLO-style, 1 class |
| `finetuned/{tiny,small,medium}_rec.onnx` | 4.5 / 21 / 77 MB | PP-OCRv6 rec fine-tuned on captchas (`fast` / `balanced` / `accurate`) |
| `finetuned/small_det.onnx` | 9.5 MB | PP-OCRv6 det fine-tuned for glyph-level click captchas (manifest `ft-small-det`; F1@0.5 96.8%) |
| `finetuned/medium_det.onnx` | 60 MB | Larger glyph detector (manifest `ft-medium-det`): F1@0.5 ~95–96%, much tighter boxes at IoU ≥ 0.75, ~4× slower |
| `ppocrv6/*` | ~169 MB | Base PP-OCRv6 rec/det exports; manifest entries, not embedded |

With `inline-model` enabled the three stock models are `include_bytes!`-embedded,
making the binary ~90 MB but fully self-contained; with it disabled the binary
reads `model/` relative to the working directory at startup. Fine-tuned and
PP-OCRv6 models are always loaded from the manifest (`--manifest
model/models.json`, optionally `--profile` / `--rec-model` / `--det-model`).

## 9. Request flow at a glance

```
curl -X POST /ocr {"image":"<b64>","color_filter":"green","charset_range":"0123456789+-x/="}
   │
   ├─ route_ocr (async, salvo)
   │    ├─ base64 decode
   │    ├─ parse color_filter → ColorFilter
   │    ├─ charset_range → CACHE.get_or_insert(calc_ranges(...))   [LRU, keyed by string]
   │    └─ spawn_blocking ─► classification*_with_options
   │         ├─ filter() : RGB→HSV mask, keep matched → RgbImage
   │         ├─ resize (Lanczos3) per charset.image
   │         ├─ grayscale / png_fix / RGB → bytes
   │         ├─ NCHW f32 tensor + normalize ((x/255-0.5)/0.5 | DIY: ImageNet)
   │         ├─ ort::Session.run
   │         ├─ CTC decode (dedupe + drop blank; logits/probability outputs detected)
   │         └─ probability: charset column subsetting (softmax only for logits outputs)
   └─ APIResponse{code:200,msg:"success",data:{text,probability,charset,confidence}}
```

`charset_range` works with the official models and with manifest/fine-tuned
PP-OCRv6 (CTC) models: for CTC models the per-frame argmax is restricted to
the requested characters (plus blank) before the greedy merge, so single-glyph
crops are decoded into the target set.

`POST /solve` reuses the same pipeline through the shared `classify()` helper
and then runs `math::solve()` on the text; `POST /slide/solve` runs
`slide_match()` and then `trajectory::human_trajectory()`.

The same core runs through MCP: `tools/call` → synthetic Salvo Request →
`route_*` handler → `CallToolResult`.