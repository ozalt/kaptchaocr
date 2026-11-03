# KaptchaOCR documentation

Start with the [project README](../README.md) for the overview and quick start.
These pages cover the details.

## Guides

| Page | What it covers |
| --- | --- |
| [HTTP API](api.md) | Every REST endpoint, request/response shapes, uploads, errors |
| [OCR](ocr.md) | Recognition, charsets, color filters, probability output |
| [Detection](detection.md) | Glyph/line detection and click-captcha boxes |
| [Slider](slider.md) | Template matching and gap comparison |
| [Solve API](solve.md) | `/solve`, `/solve/click`, `/slide/solve` |
| [Models](models.md) | Bundled models, manifest, profiles, custom ONNX import |
| [MCP](mcp.md) | MCP tools and JSON-RPC examples |
| [Docker](docker.md) | Image build, run, mounts, overrides |
| [Building](building.md) | Source builds, feature flags, linking |
| [CLI](cli.md) | Full server flag reference |
| [Troubleshooting](troubleshooting.md) | CUDA, static linking, ONNX Runtime issues |
| [Python client](python-client.md) | Zero-dependency client usage |
| [Benchmarks](../benchmarks/README.md) | Held-out accuracy/latency results and charts |

## Conventions

- Images in JSON requests are **standard base64 strings** (no `data:` prefix).
- The server returns `{"code": 200, "msg": "success", "data": ...}` on success.
- Feature-gated routes only exist when the matching flag is enabled:
  `/ocr` and `/solve` need `--ocr` (or `--old`), `/det` needs `--det`,
  `/slide/*` needs `--slide`, `/solve/click` needs both `--ocr` and `--det`,
  `/mcp` needs `--mcp` (or `--only-mcp`).

## Disclaimer

KaptchaOCR does not solve CAPTCHAs that ask you to pick pictures of cars,
trucks, traffic lights, or other objects. That is a different job and is still
work in progress. It is not included in this repo yet.
