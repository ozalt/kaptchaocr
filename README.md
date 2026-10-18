# Introduction

Kaptchowmin is an easy-to-use, general-purpose CAPTCHA recognition library and
API server written in Rust. It handles text/OCR recognition, glyph detection and
slide matching, ships as a cross-platform static binary (no OpenCV dependency),
and exposes both a REST API and an MCP endpoint.

# Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Environment Support](#environment-support)
- [Installation steps](#installation-steps)
- [User Documentation](#user-documentation)
  - [OCR Recognition](#ocr-recognition)
    - [Content Recognition](#content-recognition)
    - [Old Model](#old-model)
    - [Supports recognizing transparent black PNG format images using the png\_fix parameter](#supports-recognizing-transparent-black-png-format-images-using-the-png_fix-parameter)
    - [Color Filter](#color-filter)
    - [Reference Example Image](#reference-example-image)
  - [Object Detection](#object-detection)
    - [Reference Example Image](#reference-example-image-1)
  - [Slider Matching](#slider-matching)
    - [Algorithm 1](#algorithm-1)
    - [Algorithm 2](#algorithm-2)
  - [OCR Probability Output](#ocr-probability-output)
  - [Solve API (captcha answering)](#solve-api-captcha-answering)
  - [Custom OCR Model Import](#custom-ocr-model-import)
  - [Fine-tuned models (PP-OCRv6 captcha recognition)](#fine-tuned-models-pp-ocrv6-captcha-recognition)
- [API server example](#api-server-example)
  - [Running method](#running-method)
  - [Running Examples](#running-examples)
  - [Docker](#docker)
  - [API Documentation](#api-documentation)
  - [API test examples, see the `test_api.py` file for complete tests](#api-test-examples-see-the-test_apipy-file-for-complete-tests)
  - [Python client](#python-client)
  - [MCP Protocol Support](#mcp-protocol-support)
    - [Tool Invocation Request](#tool-invocation-request)
    - [Tool Call Response](#tool-call-response)
- [Difficult and complicated cases](#difficult-and-complicated-cases)

# Environment Support

| System           | CPU | GPU | Remarks                                                                                                                                       |
| ---------------- | --- | --- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Windows 64-bit   | √   | ?   | Some versions of Windows require installing <a href="https://www.ghxi.com/yxkhj.html">vc runtime library</a>                                  |
| Windows 32-bit   | √   | ?   | Static linking is not supported. Some versions of Windows require installing <a href="https://www.ghxi.com/yxkhj.html">vc runtime library</a> |
| Linux 64 / ARM64 | √   | ?   | May need to upgrade the glibc version, <a href=https://www.cnblogs.com/fireinstone/p/18169273>upgrade glibc version</a>                       |
| Linux 64 / MUSL  | √   | ?   | No glibc required, statically linked                                                                                                          |
| Linux 32         | ×   | ?   |                                                                                                                                               |
| Macos X64        | √   | ?   | M1/M2/M3 ... Chip reference #67                                                   |

# Installation steps

`lib.rs` implements `kaptchowmin`.

`main.rs` implements the API server.

`model` is the directory holding the models and character sets.

Depend on this library `kaptchowmin = { path = "path/to/captchowmin" }`

Enable the `cuda` feature `kaptchowmin = { path = "path/to/captchowmin", features = ["cuda"] }`

Supports static and dynamic linking, uses static linking by default, and will automatically download the link library during construction. Please set up the proxy. The `cuda` feature does not support static linking (it will download the dynamic link library itself).

If you have more questions, please jump to the [Troubleshooting](#difficult-and-complicated-cases) section.

# User Documentation

## OCR Recognition

### Content Recognition
Mainly used to recognize single-line text, which occupies the main part of the image, such as common alphanumeric verification codes. This project can handle Chinese, English (with random case or by setting the range to specify case), numbers, and certain special characters.

```rust
let image = std::fs::read("target.png").unwrap();
let mut ocr = kaptchowmin::kaptchowmin_classification().unwrap();
let res = ocr.classification(image).unwrap();
println!("{:?}", res);
```

### Old Model

```rust
let image = std::fs::read("target.png").unwrap();
let mut ocr = kaptchowmin::kaptchowmin_classification_old().unwrap();
let res = ocr.classification(image).unwrap();
println!("{:?}", res);
```

### Supports recognizing transparent black PNG format images using the png_fix parameter

```
classification_with_png_fix(image, true);
```

### Color Filter

Supports the following preset colors: red, blue, green, yellow, orange, purple, cyan, black, white, gray.

```rust
let ocr = kaptchowmin_classification().unwrap();

// Keep only green
println!(
    "{}",
    ocr
    .classification_with_filter(include_bytes!("../image/4.png"), "green")
    .unwrap()
);

// Only keep red and green
println!(
    "{}",
    ocr
    .classification_with_filter(include_bytes!("../image/4.png"), ["red", "green"])
    .unwrap()
);

// HSV range, each element is a (min_hsv, max_hsv) tuple.
println!(
    "{}",
    ocr
    .classification_with_filter(
        include_bytes!("../image/4.png"),
        [((40, 50, 50), (80, 255, 255))]
    )
    .unwrap()
);
```
### Reference Example Image

<img src="https://cdn.wenanzhe.com/img/20210715211733855.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/78b7f57d-371d-4b65-afb2-d19608ae1892.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211226142305.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211226142325.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/2AMLyA_fd83e1f1800e829033417ae6dd0e0ae0.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/aabd_181ae81dd5526b8b89f987d1179266ce.jpg" alt="captcha" width="150">
<br />
<img src="https://cdn.wenanzhe.com/img/2bghz_b504e9f9de1ed7070102d21c6481e0cf.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/0000_z4ecc2p65rxc610x.jpg" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/2acd_0586b6b36858a4e8a9939db8a7ec07b7.jpg" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/2a8r_79074e311d573d31e1630978fe04b990.jpg" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/aftf_C2vHZlk8540y3qAmCM.bmp" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20211226144057.png" alt="captcha" width="150">

## Object Detection

```rust
let image = std::fs::read("target.png").unwrap();
let mut det = kaptchowmin::kaptchowmin_detection().unwrap();
let res = det.detection(image).unwrap();
println!("{:?}", res);
```

### Reference Example Image

![Test](https://cdn.wenanzhe.com/img/page1_1.jpg)
![Test](https://cdn.wenanzhe.com/img/page1_2.jpg)
![Test](https://cdn.wenanzhe.com/img/page1_3.jpg)
![Test](https://cdn.wenanzhe.com/img/page1_4.jpg)
![Test](https://cdn.wenanzhe.com/img/result.jpg)
![Test](https://cdn.wenanzhe.com/img/result2.jpg)
![Test](https://cdn.wenanzhe.com/img/result4.jpg)

The above are just the click verification code images I can currently find, and I have done a simple test.

## Slider Matching

The algorithm is not implemented with a deep neural network.

### Algorithm 1
The small slider is a separate PNG image with a transparent background, as shown below:

![Test](https://cdn.wenanzhe.com/img/b.png)

Then the background has a small slider slot, as shown below:

![Test](https://cdn.wenanzhe.com/img/a.png)

```rust
let target_bytes = std::fs::read("target.png").unwrap();
let background_bytes = std::fs::read("background.png").unwrap();
let res = kaptchowmin::slide_match(target_bytes, background_bytes).unwrap();
println!("{:?}", res);
```

If the small image does not have too much background, you can use simple_slide_match, usually in jpg or bmp format.

```rust
let target_bytes = std::fs::read("target.png").unwrap();
let background_bytes = std::fs::read("background.png").unwrap();
let res = kaptchowmin::simple_slide_match(target_bytes, background_bytes).unwrap();
println!("{:?}", res);
```

### Algorithm 2
One image is the original image with a pit, as shown below:

![Test](https://cdn.wenanzhe.com/img/bg.jpg)

One image is the original image, as shown below:

![Test](https://cdn.wenanzhe.com/img/fullpage.jpg)

```rust
let target_bytes = std::fs::read("target.png").unwrap();
let background_bytes = std::fs::read("background.png").unwrap();
let res = kaptchowmin::slide_comparison(target_bytes, background_bytes).unwrap();
println!("{:?}", res);
```

## OCR Probability Output

In order to provide more flexible control and range limitation of OCR results, the project supports range limitation of OCR results.

You can return the probability of the full character table by calling `classification_probability`.

Of course, you can also limit the returned results by setting the output character range through `set_ranges`.

| Parameter Value | Meaning                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------- |
| 0               | Pure integer 0-9                                                                         |
| 1               | Pure lowercase letters a-z                                                               |
| 2               | Pure uppercase letters A-Z                                                               |
| 3               | Lowercase letters a-z + Uppercase letters A-Z                                            |
| 4               | Lowercase letters a-z + Integers 0-9                                                     |
| 5               | Uppercase letters A-Z + Integers 0-9                                                     |
| 6               | Lowercase letters a-z + Uppercase letters A-Z + Integers 0-9                             |
| 7               | Default character library - Lowercase letters a-z - Uppercase letters A-Z - Integers 0-9 |

If the value is of type string, please pass in a piece of text that does not contain spaces, where each character is a candidate word, for example: `"0123456789+-x/="`

```rust
let image = std::fs::read("image.png").unwrap();
let mut ocr = kaptchowmin::kaptchowmin_classification().unwrap();

// The number 3 corresponds to the enumeration CharsetRange::LowercaseUppercase, no need to write the enumeration
// ocr.set_ranges(3);

// Set the global character set
ocr.set_ranges("0123456789+-x/=");

// Or, the character set for single recognition
// ocr.classification_probability_with_ranges(image, "0123456789+-x/=");

let result = ocr.classification_probability(image).unwrap();

println!("Recognition result: {}", result.get_text());
println!("Recognition confidence: {}", result.get_confidence());

// Oh, it seems there's a bit too much data, be careful of freezing!
println!("Probability: {}", result.json());
```

## Solve API (captcha answering)

`/solve` and `/slide/solve` target scraping workflows: one request returns a
ready-to-use answer instead of raw text. They require the server to run with
`--ocr` (`/solve`) and `--slide` (`/slide/solve`).

### Text / math captchas: `POST /solve`

- Same request fields as `/ocr` (`image`, `png_fix`, `probability`,
  `charset_range`, `color_filter`) plus `math`: `auto` (default) or `off`.
- Returns `text` and `math` (`null` when the text is not an expression):
  `{"expression":"5+?=8","result":3.0,"answer":"3"}`.
- Supported: `3+5=`, `9÷3`, `4x6`, full-width `４×６＝`, and single-unknown
  equations `5+?=8`, `?×6=42`, `12/?=4`; non-integral results keep up to two
  decimals (`10/3` → `3.33`).
- Chinese numerals and operator words: `九乘六等于？` → `54`,
  `十二减四等于几` → `8`, `五加几等于八` → `3`; traditional `減`/`幾` and
  `两千` / `一百零三` style compounds included (up to thousands).
- `model` in the response is the loaded model name (when started with
  `--manifest`).

### Slider captchas: `POST /slide/solve`

- Extends `/slide_match` with `handle_x` / `handle_y` (slider handle position
  from the page), `offset` (calibration), `duration_ms`, `seed` (reproducible
  trajectory) and `trajectory` (default true).
- `distance` = gap x + `offset` − `handle_x` (or just the gap x when no handle
  is given).
- `trajectory` holds **deltas relative to the handle start**, sampled at
  ~60 Hz, with the last point exactly at `(distance, 0)`; replay with
  `mouse.move(handle_x + p.x, handle_y + p.y)`.

### Click captchas: `POST /solve/click`

Requires both recognition and detection (`--ocr --det`). Pipeline: detect glyph
boxes → recognize each box → order the clicks for the prompt.

- Request: `image`, optional `prompt` (e.g. `请依次点击【明,天,会】`,
  `click in order: A B C`), `score_threshold` (default 0.3), `charset_range`.
- Response: `prompt` (parsed target characters), `points` (click coordinates
  in prompt order, relative to the captcha image, box centres), `boxes` (all
  glyph boxes with recognized characters and detection scores, reading order),
  `unmatched` (prompt characters that could not be matched).
- Matching: each target takes the first unused box, preferring an exact
  character match over an ASCII-case-insensitive one. Without a `prompt` only
  the boxes are returned.
- For click solving use the fine-tuned glyph detector and recognizer:
  `--manifest model/models.json --det-model ft-small-det --rec-model ft-small`
  (detection F1@0.5 **96.8%** on the synthetic click set vs 60.6% for the
  bundled `stock-det`; `ft-medium-det` trades ~4× latency for tighter boxes),
  and pass `charset_range` with the prompt's characters to constrain
  single-glyph recognition.
- `charset_range` now supports both the official models and fine-tuned
  PP-OCRv6 (CTC) models; without a range the official `stock-common`
  (wider charset, CJK included) recognizer is also a good fit.

## Custom OCR Model Import

Any third-party ONNX recognition model can be imported when paired with a
charset JSON file (same schema as the bundled configs — see `model/README.md`):

```rust
use kaptchowmin::*;

let mut ocr = Kaptchowmin::with_model_charset(
    "captcha_model.onnx",
    "charset.json",
)
.unwrap();
let image_bytes = std::fs::read("sample.jpg").unwrap();
let res = ocr.classification(&image_bytes).unwrap();
println!("{:?}", res);
```

## Fine-tuned models (PP-OCRv6 captcha recognition)

The repo ships PP-OCRv6 recognition models fine-tuned on a captcha dataset
(tiny / small / medium), wired in through the `model/models.json` manifest:

| model | exact | char | mean latency (CPU, 8 threads) | size |
| --- | --- | --- | --- | --- |
| `ft-tiny` | 67.0% | 88.8% | 7–12 ms | 4.5 MB |
| `ft-small` | **80.4%** | **93.0%** | 18–33 ms | 21 MB |
| `ft-medium` | **90.7%** | **96.0%** | 48–93 ms | 77 MB |
| `stock-common` (bundled) | 14.3% | 42.3% | 24–43 ms | — |
| `stock-old` (bundled) | 15.2% | 42.2% | 19–33 ms | — |
| base PP-OCRv6 medium (not fine-tuned) | 28.6% | 59.0% | 45–109 ms | 77 MB |

> Measured on 6 per-source test sets (4,399 images); exact = whole string
> correct, case-sensitive. The fine-tuned models are case-sensitive; the base
> (non-fine-tuned) PP-OCRv6 models collapse case, scoring ~0 exact on the
> lowercase/10-char splits. Dataset, harness and reproduction live in the R&D
> folder `../captchowmin-rnd/`.

The same pipeline produced a fine-tuned **glyph detector** for click captchas:

| detector | P@0.5 | R@0.5 | F1@0.5 | F1@0.75 | test set |
| --- | --- | --- | --- | --- | --- |
| `ft-small-det` | 95.4% | 98.3% | **96.8%** | 53.9% | val (seed 1) |
| `ft-small-det` | 93.2% | 97.3% | **95.2%** | 49.3% | val2 (seed 21) |
| `ft-medium-det` | 92.3% | 97.9% | 95.0% | **61.9%** | val (seed 1) |
| `ft-medium-det` | 93.8% | 98.0% | **95.9%** | **59.8%** | val2 (seed 21) |
| `stock-det` (bundled) | 87.3% | 46.4% | 60.6% | 37.9% | val |

> Greedy glyph matching at IoU 0.5 / 0.75 (harness: `eval_det` in
> `../captchowmin-rnd/`). `ft-small-det` is the default recommendation
> (fastest); `ft-medium-det` is ~4× slower but produces clearly tighter boxes
> (better at IoU ≥ 0.75). For `/solve/click` use `--det-model ft-small-det`
> (or `ft-medium-det`) with `--rec-model ft-small` and pass `charset_range`
> with the prompt characters.

Server-side profile selection (requires `--manifest`):

```sh
# recommended: small (balanced — also the manifest default)
kaptchowmin --ocr --manifest model/models.json --profile balanced

kaptchowmin --ocr --manifest model/models.json --profile fast      # tiny, fastest
kaptchowmin --ocr --manifest model/models.json --profile accurate  # medium, most accurate
kaptchowmin --ocr --manifest model/models.json --rec-model stock-common  # any entry by name
```

Without `--manifest`, the bundled (inline) `stock_common` / `stock_old` models are
used. From Rust, load directly with `Kaptchowmin::new(bytes, charset)` (charset:
`model/finetuned/<tier>_rec.json`). Conversion notes: `model/README.md` and the
R&D folder.

# API server example

## Running method
```sh
Usage: kaptchowmin.exe [OPTIONS]

Options:
    --address <ADDRESS>
        Listening address. [default: 0.0.0.0:8000]
    --mcp
        mcp protocol support, mutually exclusive with only_mcp.
    --only-mcp
        Only enable mcp protocol, do not enable normal routing, mutually exclusive with mcp.
    --ocr
        Enable content recognition, mutually exclusive with old.
    --old
        Enable the old version of model content recognition, which is mutually exclusive with OCR.
    --det
        Enable object detection.
    --slide
        Enable slider and pit recognition.
    --ocr-charset-range <OCR_CHARSET_RANGE>
        Global default character set, used for probability recognition. If the API does not provide a character set, this parameter is used. When the value is 0~7, it means selecting the built-in character set. Other values indicate a custom character set, such as "0123456789+-x/=". If not set, the complete character set is used without restrictions.
    --ocr-path <OCR_PATH>
        Content recognition model and character set path. If you have enabled the inline-model option of features (enabled by default), you don't need to care about this option unless you want to use a custom model. The model and charset paths must share the same base name, e.g. model/stock_common.onnx + model/stock_common.json. [default: model/stock_common.onnx]
    --det-path <DET_PATH>
        Target detection model path. If you have enabled the `inline-model` option for features (enabled by default), you don't need to worry about this option unless you want to use a custom model. [default: model/stock_det.onnx]
    --acme <ACME>
        Enter your domain name to automatically obtain SSL certificates, i.e., HTTPS support.
    -h, --help
        Print help
```

## Running Examples
```sh
# Start all functions
kaptchowmin.exe --address 0.0.0.0:8000 --ocr --det --slide --mcp

# View all options
kaptchowmin.exe --help
```

## Docker

The repo ships a multi-stage `Dockerfile` (Alpine + musl static binary) that
enables all features and defaults to `balanced` (ft-small):

```sh
# Build (the build stage needs network access for ONNX Runtime)
docker build -t kaptchowmin .

# Run (defaults to --ocr --det --slide --manifest model/models.json --profile balanced)
docker run --rm -p 8000:8000 kaptchowmin

# Arguments replace the default CMD entirely, e.g. switch to the most accurate tier
docker run --rm -p 8000:8000 kaptchowmin --ocr --det --slide \
    --manifest model/models.json --profile accurate

# Mount your own model + manifest
docker run --rm -p 8000:8000 \
    -v "$PWD/my_rec.onnx:/app/model/my_rec.onnx:ro" \
    -v "$PWD/my_rec.json:/app/model/my_rec.json:ro" \
    -v "$PWD/my_manifest.json:/app/model/my_manifest.json:ro" \
    kaptchowmin --ocr --manifest model/my_manifest.json

# OCR + slider only, with an API key
docker run --rm -p 8000:8000 kaptchowmin --ocr --slide --api-key secret
```

The image contains `model/stock_*` and `model/finetuned/` (not
`model/ppocrv6/` — mount it yourself if needed); `HEALTHCHECK` probes
`/healthz`. For end-to-end browser workflows (retries, calibration), see
`playwright-cookbook.md`.

## API Documentation

| Endpoint            | Method | Description                 |
| ------------------- | ------ | --------------------------- |
| `/ocr`              | POST   | Perform OCR recognition     |
| `/solve`            | POST   | Text / math captcha solving (recognition + arithmetic answer) |
| `/solve/click`      | POST   | Click captcha solving (glyph boxes + prompt-ordered click points) |
| `/det`              | POST   | Perform object detection    |
| `/slide_match`      | POST   | Slider matching algorithm   |
| `/slide_comparison` | POST   | Slider comparison algorithm |
| `/slide/solve`      | POST   | Slider solving (match + drag distance / trajectory) |
| `/status`           | GET    | Get current service status  |
| `/docs`             | GET    | Swagger UI documentation    |

## API test examples, see the `test_api.py` file for complete tests

```python
--> 200 GET /status

curl -X GET "http://127.0.0.1:8000/status"

{"code":200,"msg":"success","data":{"service_status":"running","enabled_features":["ocr","det","slide"]}}

--> 200 POST /ocr

curl -X POST "http://127.0.0.1:8000/ocr"
-H "Content-Type: application/json"
-d '{"image": "base64 image"}'

{"code":200,"msg":"success","data":{"text":"What is nine times six?","probability":null,"charset":null,"confidence":null}}

--> 200 POST /solve

curl -X POST "http://127.0.0.1:8000/solve"
-H "Content-Type: application/json"
-d '{"image": "base64 image"}'

{"code":200,"msg":"success","data":{"text":"5+?=8","math":{"expression":"5+?=8","result":3.0,"answer":"3"},"probability":null,"charset":null,"confidence":null,"model":"ft-small"}}

--> 200 POST /det

curl -X POST "http://127.0.0.1:8000/det"
-H "Content-Type: application/json"
-d '{"image": "base64 image"}'

{"code":200,"msg":"success","data":{"bboxes":[[80,3,98,21],[56,6,76,25],[31,7,51,26],[2,2,21,22],[100,0,127,18]]}}

--> 200 POST /slide_match

curl -X POST "http://127.0.0.1:8000/slide_match"
-H "Content-Type: application/json"
-d '{"target_image": "base64 image", "background_image": "base64 image", "simple_target": true}'

{"code":200,"msg":"success","data":{"target":[215,45,261,91],"target_x":0,"target_y":45}}

--> 200 POST /slide/solve

curl -X POST "http://127.0.0.1:8000/slide/solve"
-H "Content-Type: application/json"
-d '{"target_image": "base64 image", "background_image": "base64 image", "handle_x": 20, "handle_y": 40, "seed": 7}'

{"code":200,"msg":"success","data":{"target":[140,69,200,124],"target_x":0,"target_y":0,"gap_x":140,"gap_y":69,"distance":120.0,"duration_ms":482.0,"trajectory":[{"x":0.06,"y":-0.02,"t":0.0},{"x":1.2,"y":0.3,"t":16.6}]}}

--> 200 POST /slide_comparison

curl -X POST "http://127.0.0.1:8000/slide_comparison"
-H "Content-Type: application/json"
-d '{"target_image": "base64 image", "background_image": "base64 image"}'

{"code":200,"msg":"success","data":{"x":144,"y":76}}
```
## Python client

`clients/python/` ships a zero-dependency (standard library only) synchronous
client wrapping `/solve`, `/slide/solve`, `/ocr`, `/det`:

```sh
pip install clients/python   # or editable: pip install -e clients/python
```

```python
from captchowmin import Client

client = Client("http://127.0.0.1:8000", timeout=10, retries=2)

answer = client.solve(open("captcha.png", "rb").read())
print(answer.text, answer.math.answer if answer.math else None)

slide = client.slide_solve(target_bytes, background_bytes, handle_x=12, handle_y=40)
for point in slide.trajectory:      # deltas from the handle start
    ...                             # mouse.move(handle_x + point.x, handle_y + point.y)
```

Pass `api_key=` when the server runs with `--api-key`; connection failures
raise `NetworkError`, HTTP / business errors raise `ServerError`, and 5xx / 429
are retried with exponential backoff honoring `Retry-After`. See
`clients/python/README.md`; for full browser workflows see
`playwright-cookbook.md`.

## MCP Protocol Support

This project supports the MCP (Model Context Protocol), enabling AI Agents to directly invoke the kaptchowmin service.

Version: 2025-03-26

Endpoint: `POST /mcp`

Methods: `initialize` `tools/list` `tools/call`

### Tool Invocation Request

```json
{
    "jsonrpc": "2.0",
    "id": 0,
    "method": "tools/call",
    "params": {
        "name": "ocr",
        "arguments": {"image": "image_b64", "color_filter": "green"},
    },
}
```
### Tool Call Response

```json
{
    "jsonrpc": "2.0",
    "id": 0,
    "result": {
    "content": [
        {
            "type": "text",
            "text": "{\"probability\":null,\"text\":\"Equals?\"}"
        }
    ],
    "isError": false
    }
}
```

# Difficult and complicated cases

It is strongly recommended to build with the provided `Dockerfile`.

Regarding the issue of using `cuda`.

Both `cuda` and `cuDNN` need to be installed.

`CUDA 12` builds require `cuDNN 9.x`.

`CUDA 11` builds require `cuDNN 8.x`.

It is uncertain whether `cuda 10` is effective.

Static linking is used by default, and the link library will be automatically downloaded during construction. Please set up the proxy. The `cuda` feature does not support static linking (it will download the dynamic link library itself).

If you want to specify the path of the static link library, you can set the environment variable `ORT_LIB_LOCATION`. After setting, the link library will not be downloaded automatically.

For example, if the library path is `onnxruntime\build\Windows\Release\Release\onnxruntime.lib`, then `ORT_LIB_LOCATION` is set to `onnxruntime\build\Windows\Release`.

The `download-binaries` feature is enabled by default to automatically download the link library.

Most of the download failures are network problems. After enabling the proxy, remember to restart vscode and restart the terminal so that the proxy can use the https_proxy environment variable.

The automatically downloaded link library is stored in `C:\Users\<username>\AppData\ort.pyke.io`.

Enable the dynamic linking feature `kaptchowmin = { path = "path/to/captchowmin", features = ["load-dynamic"] }`

After enabling the `load-dynamic` feature, you can use `Kaptchowmin::set_onnxruntime_path` to specify the path of the [onnxruntime](https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1) dynamic link library.

After enabling the `load-dynamic` feature, the [onnxruntime](https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1) link library will not be automatically downloaded during construction.

Please manually download the [onnxruntime](https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1) link library and place it in the program's running directory (or system API directory), so you don't need to call `Kaptchowmin::set_onnxruntime_path` again.

Windows static linking failed, please install vs2022.

Linux musl should be compiled with docker.

Linux x86-64 static linking failed, please install gcc11 and g++11, ubuntu ≥ 20.04.

Linux arm64 static linking failed, glibc ≥ 2.35 is required (Ubuntu ≥ 22.04).

macOS static linking failed, macOS ≥ 10.15 is required.

cuda may `painc (exit code: 0xc000007b)` when executing `cargo test`. This is because the automatically generated dynamic link library is in the `target/debug` directory and needs to be manually copied to the `target/debug/deps` directory (cuda does not currently support static linking).

Dynamic linking requires version 1.18.x of [onnxruntime](https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1).

For more complex issues, please visit [ort.pyke.io](https://ort.pyke.io/).