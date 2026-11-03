# Troubleshooting

## Build with Docker first

The supported build path is the repository `Dockerfile`; it pins the musl
toolchain and handles ONNX Runtime. Native builds are supported too, but the
platform/linking notes below apply.

## CUDA

- Both CUDA and cuDNN are required.
- CUDA 12 builds need cuDNN 9.x.
- CUDA 11 builds need cuDNN 8.x.
- CUDA 10 is not validated.

The `cuda` feature does **not** support static linking; it downloads the
dynamic ONNX Runtime library itself.

## ONNX Runtime download / linking

Static linking is the default and downloads the matching ONNX Runtime during
the build. If your network requires a proxy, set the usual `https_proxy` /
`http_proxy` environment variables for the build shell.

To use an existing ORT install instead:

```sh
export ORT_LIB_LOCATION=/path/to/onnxruntime
```

For example, if the library is
`onnxruntime/build/Windows/Release/Release/onnxruntime.lib`, point
`ORT_LIB_LOCATION` at `onnxruntime/build/Windows/Release`. With
`ORT_LIB_LOCATION` set, the build does not download a library.

The downloaded library is cached under the platform's ORT cache directory
(e.g. `%APPDATA%\ort.pyke.io` on Windows).

## Dynamic linking

Enable `load-dynamic` and provide ONNX Runtime **1.18.x**:

```toml
kaptchaocr = { path = "path/to/kaptchaocr", features = ["load-dynamic"] }
```

Use `KaptchaOcr::set_onnxruntime_path(path)` before model creation, or place
the shared library in the loader path. With `load-dynamic`, the build does not
download ORT.

## Platform notes

- **Windows:** install the Microsoft VC++ runtime. Static linking needs Visual
  Studio 2022 (`vs2022` build tools).
- **Linux x86-64 static:** use gcc/g++ 11 and Ubuntu ≥ 20.04.
- **Linux arm64 static:** glibc ≥ 2.35 (Ubuntu ≥ 22.04).
- **Linux musl:** build with the provided Dockerfile.
- **macOS static:** macOS ≥ 10.15.

## CUDA + `cargo test`

The dynamic CUDA library may end up in `target/debug` while the test binary
looks in `target/debug/deps`. Copy the library into `target/debug/deps` if a
test run cannot load it. CUDA does not support static linking.

## More

- ONNX Runtime: <https://onnxruntime.ai/>
- `ort` crate docs: <https://ort.pyke.io/>
