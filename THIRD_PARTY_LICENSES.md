# Third-party licenses

KaptchaOCR itself is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)).
It redistributes or derives from the following third-party works.

## ddddocr

- Upstream: <https://github.com/sml2h3/ddddocr>
- License: MIT
- Used for: the bundled stock recognition/detection models (`model/stock_*.onnx`)
  and portions of the original implementation this project is based on.

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

## PaddleOCR / PP-OCRv6

- Upstream: <https://github.com/PaddlePaddle/PaddleOCR>
- License: Apache-2.0
- Used for: base PP-OCRv6 detection/recognition models and the fine-tuned
  derivatives shipped in `model/finetuned/`.

The Apache-2.0 license text is included in the root [`LICENSE`](LICENSE) file.
