# kaptchaocr — local convenience targets (Linux / macOS).
#
#   make run                    build (if needed) + start the server locally
#   make run PROFILE=accurate   pick a model tier: fast | balanced | accurate
#   make help                   list all targets

BIN     := target/release/kaptchaocr
PROFILE ?= balanced
ADDR    ?= 127.0.0.1:8000

.PHONY: help build run docker docker-run test check fmt eval clean

help: ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

build: ## build the release binary (fine-tuned models embedded)
	cargo build --release

run: build ## run the server locally (PROFILE=…, ADDR=host:port)
	$(BIN) --address $(ADDR) --ocr --det --slide --manifest model/models.json --profile $(PROFILE)

docker: ## build the Docker image (tag: kaptchaocr)
	docker build -t kaptchaocr .

docker-run: ## run the Docker image (port 8000)
	docker run --rm -p 8000:8000 kaptchaocr

test: ## run library tests (both feature configs)
	cargo test --lib
	cargo test --lib --no-default-features --features download-binaries

check: ## fmt check + clippy (both feature configs)
	cargo fmt --all --check
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --no-default-features --features download-binaries -- -D warnings

fmt: ## format the code
	cargo fmt --all

eval: ## run synth checks against a running server (ADDR=host:port)
	python3 evaluation/run_eval.py --url http://$(ADDR) --data . --suites synth --limit 20

clean: ## remove build artifacts
	cargo clean
