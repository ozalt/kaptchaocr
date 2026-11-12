# Third-party licenses

KaptchaOCR itself is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)).
It redistributes or derives from the following third-party works.

## ddddocr

- Upstream: <https://github.com/sml2h3/ddddocr>
- License: MIT
- Used for: portions of the original implementation this project is based on,
  and the stock models used for benchmark comparison. The stock model weights
  are not redistributed.

```text
The MIT License (MIT)
Copyright © 2022 <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## 86maid/ddddocr (Rust port)

- Upstream: <https://github.com/86maid/ddddocr>
- License: the repository README declares Apache-2.0, but no `LICENSE` file is
  present in the upstream repository at the time of writing.
- Used for: the Rust API shape and algorithms this project was adapted from
  (`classification_with_png_fix`, `slide_comparison`, charset ranges, MCP tool
  naming). No source file or model weight from this repository is
  redistributed.
- This is an attribution notice; it does not grant rights beyond those in the
  upstream project.

## PaddleOCR / PP-OCRv6

- Upstream: <https://github.com/PaddlePaddle/PaddleOCR>
- License: Apache-2.0
- Used for: base PP-OCRv6 models used for benchmark comparison, and the
  fine-tuned derivatives shipped in `model/finetuned/`.

The Apache-2.0 license text is included in the root [`LICENSE`](LICENSE) file.
