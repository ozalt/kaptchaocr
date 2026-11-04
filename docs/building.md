# Building from source

## Requirements

- Rust **stable** (edition 2021)
- Network access for the ONNX Runtime download (or a local ORT install)
- Docker only if you want the musl/static build

## Quick build

```sh
cargo build --release
make run          # ./target/release/kaptchaocr with the balanced profile
make test         # library tests, both feature configs
make check        # fmt + clippy, both feature configs
```

The binary embeds the fine-tuned `ft-small` and `ft-small-det` models by
default, so it runs without external model files.

## Feature flags

| Feature | Default | Effect |
| --- | --- | --- |
| `inline-model` | yes | Embed the fine-tuned recognition/detection models in the binary |
| `download-binaries` | yes | Download a matching ONNX Runtime during build |
| `load-dynamic` | no | Load a user-supplied ONNX Runtime at runtime |
| `cuda` | no | Build with CUDA support (dynamic library only) |

Without `inline-model`, the server reads `model/finetuned/small_rec.onnx` and
`model/finetuned/small_det.onnx` from disk.

## Linking

Static linking is the default for the `ort` dependency; the link library is
downloaded during the build. If you already have ONNX Runtime, set
`ORT_LIB_LOCATION` to the directory containing the library and the download is
skipped.

For dynamic linking:

```toml
kaptchaocr = { path = "path/to/kaptchaocr", features = ["load-dynamic"] }
```

Then call `KaptchaOcr::set_onnxruntime_path(path)` before creating a model, or
place the ONNX Runtime shared library where the loader can find it. Dynamic
linking expects ONNX Runtime **1.18.x**.

## Platform notes

| Target | Status | Notes |
| --- | --- | --- |
| Linux x86-64 | supported | static build needs gcc/g++ 11 (Ubuntu ≥ 20.04); musl via Docker |
| Linux arm64 | supported | static build needs glibc ≥ 2.35 (Ubuntu ≥ 22.04) |
| Linux MUSL | supported | statically linked; use the Dockerfile |
| macOS x86-64 / arm64 | supported | macOS ≥ 10.15 |
| Windows x86-64 | supported | install the Microsoft VC++ runtime; static linking needs VS 2022 |

CUDA builds are dynamic-only. See [Troubleshooting](troubleshooting.md) for the
CUDA/cuDNN matrix and common build failures.

## Tests and lint

```sh
cargo test --lib
cargo test --lib --no-default-features --features download-binaries
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --no-default-features --features download-binaries -- -D warnings
```

Tests and lint are listed above. Architecture and code live in `src/`.
