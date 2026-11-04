# Docker

The repository ships a multi-stage `Dockerfile` (Alpine build stage, musl static
binary) with all features enabled. The image defaults to:

```sh
--ocr --det --slide --manifest model/models.json --profile balanced
```

## Build and run

The build stage compiles Rust inside the container and downloads ONNX Runtime,
so the host needs **only Docker** (plus network access during the build).

```sh
docker build -t kaptchaocr .
docker run --rm -p 8000:8000 kaptchaocr
```

The resulting image is ~711 MB. `HEALTHCHECK` probes `/healthz`.

## Override the defaults

Arguments replace the container `CMD` entirely:

```sh
# most accurate tier
docker run --rm -p 8000:8000 kaptchaocr \
  --ocr --det --slide --manifest model/models.json --profile accurate

# OCR + slider only, with an API key
docker run --rm -p 8000:8000 kaptchaocr \
  --ocr --slide --api-key secret
```

## Mount custom models

```sh
docker run --rm -p 8000:8000 \
  -v "$PWD/my_rec.onnx:/app/model/my_rec.onnx:ro" \
  -v "$PWD/my_rec.json:/app/model/my_rec.json:ro" \
  -v "$PWD/my_manifest.json:/app/model/my_manifest.json:ro" \
  kaptchaocr --ocr --manifest model/my_manifest.json
```

## What is inside

- `kaptchaocr` binary at `/usr/local/bin/kaptchaocr`
- `model/finetuned/` models
- `model/models.json` manifest

A GHCR image will be published with the first public release. Until then,
`docker build` is the supported path.

See [CLI](cli.md) for flags and [Building](building.md) for native builds.
