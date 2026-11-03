# CLI reference

```text
Usage: kaptchaocr [OPTIONS]

Options:
      --address <ADDRESS>
          Listen address [default: 0.0.0.0:8000]
      --mcp
          Enable MCP protocol support; mutually exclusive with only_mcp
      --only-mcp
          Enable the MCP protocol only, without the regular routes; mutually exclusive with mcp
      --ocr
          Enable content recognition; mutually exclusive with old
      --old
          Enable content recognition with the legacy model; mutually exclusive with ocr
      --det
          Enable object detection
      --slide
          Enable slider and gap recognition
      --ocr-charset-range <OCR_CHARSET_RANGE>
          Global default charset for probability-based recognition. If the API does not provide a charset, this value is used. Values 0~7 select a built-in charset; any other value is a custom charset, e.g. "0123456789+-x/=". If unset, the full charset is used without restriction
      --ocr-path <OCR_PATH>
          Path to the content recognition model and its charset. If the inline-model feature is enabled (the default), you can ignore this option unless you want a custom model. The model and charset paths must share the same base name, e.g. `model/stock_common.onnx` + `model/stock_common.json` [default: model/stock_common.onnx]
      --det-path <DET_PATH>
          Path to the object detection model. If the inline-model feature is enabled (the default), you can ignore this option unless you want a custom model [default: model/stock_det.onnx]
      --acme <ACME>
          Provide your domain to automatically obtain an SSL certificate, i.e. HTTPS support
      --max-concurrency <MAX_CONCURRENCY>
          Maximum number of concurrent ONNX inference tasks, defaults to the number of CPU cores. 0 uses the number of CPU cores [default: 0]
      --rate-limit <RATE_LIMIT>
          Allowed requests per IP per 10 seconds; 0 means unlimited [default: 0]
      --api-key <API_KEY>
          Optional API key; when set, all REST / MCP requests must carry an `Authorization: Bearer <key>` or `x-api-key: <key>` header
      --num-threads <NUM_THREADS>
          ORT `IntraOpNumThreads` (threads used per inference); 0 uses the ORT default [default: 0]
      --config <CONFIG>
          Optional TOML config file; can set model paths / charset / thread count, etc. Command-line arguments take precedence
      --manifest <MANIFEST>
          Optional model manifest file (`model/models.json`); loads rec / det models from the manifest, so any ONNX model can be plugged in without code changes. Takes precedence over --ocr-path / --det-path
      --profile <PROFILE>
          Named profile in the model manifest (the `profiles` field of `model/models.json`), e.g. fast / balanced / accurate. Must be used together with --manifest
      --rec-model <REC_MODEL>
          Select a rec model by name from the manifest (defaults to the first rec entry), e.g. --rec-model ft-small. Takes precedence over --profile
      --det-model <DET_MODEL>
          Select a det model by name from the manifest (defaults to the first det entry), e.g. --det-model ppocrv6-tiny-det
  -h, --help
          Print help
```

## Common combinations

```sh
# everything, balanced tier (Docker default)
kaptchaocr --ocr --det --slide --manifest model/models.json --profile balanced

# fastest / most accurate recognition
kaptchaocr --ocr --manifest model/models.json --profile fast
kaptchaocr --ocr --manifest model/models.json --profile accurate

# click captchas: glyph detector + recognizer
kaptchaocr --ocr --det --manifest model/models.json \
  --det-model ft-small-det --rec-model ft-small

# MCP only
kaptchaocr --ocr --det --slide --only-mcp
```

Without `--manifest`, the embedded stock models are used. Profiles and model
selection are covered in [Models](models.md).
