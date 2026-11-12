# Evaluation harness

End-to-end checks against a **running** KaptchaOCR server, driven through the
Python client. Each suite prints a JSON summary and writes it to
[`results/`](results/).

## Suites

| Suite | What it measures | Data |
| --- | --- | --- |
| `rec` | exact match / CER on the six held-out recognition splits | `zaltahir/kaptchaocr-rec` parquet |
| `synth` | accuracy on fresh generators never used for training | generated locally |
| `math` | `/solve` on PIL-drawn arithmetic CAPTCHAs | generated locally |
| `click` | `/solve/click` on the 400-image detection test split | `zaltahir/kaptchaocr-det` parquet |
| `slider` | `/slide_match` on procedurally generated jigsaw sliders | generated locally |

## Run

Start the server first:

```sh
cargo build --release
./target/release/kaptchaocr --ocr --det --slide
```

Then, in another shell:

```sh
python3 -m venv .venv
.venv/bin/pip install captcha pillow pandas pyarrow huggingface_hub

# Recognition and click suites need the parquet files:
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("zaltahir/kaptchaocr-rec", repo_type="dataset",
                  allow_patterns=["data/test_*"], local_dir="hf/rec")
snapshot_download("zaltahir/kaptchaocr-det", repo_type="dataset",
                  allow_patterns=["data/test-*"], local_dir="hf/det")
PY

.venv/bin/python evaluation/run_eval.py --data hf --suites synth,math,slider --limit 50
.venv/bin/python evaluation/run_eval.py --data hf --suites rec,click
```

`--limit N` samples N images per split. `--tag _name` suffixes the result
files. Results in this directory were produced on 2026-11-09 with `ft-small`
(and `ft-medium` for the synth/math comparison runs).
