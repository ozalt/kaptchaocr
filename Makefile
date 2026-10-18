# kaptchowmin — local convenience targets (Linux / macOS).
#
#   make run                    build (if needed) + start the server locally
#   make run PROFILE=accurate   pick a model tier: fast | balanced | accurate
#   make help                   list all targets

BIN     := target/release/kaptchowmin
PROFILE ?= balanced
ADDR    ?= 127.0.0.1:8000

.PHONY: help build run docker docker-run test check fmt clean

help: ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

build: ## build the release binary (stock models embedded)
	cargo build --release

run: build ## run the server locally (PROFILE=…, ADDR=host:port)
	$(BIN) --address $(ADDR) --ocr --det --slide --manifest model/models.json --profile $(PROFILE)

docker: ## build the Docker image (tag: kaptchowmin)
	docker build -t kaptchowmin .

docker-run: ## run the Docker image (port 8000)
	docker run --rm -p 8000:8000 kaptchowmin

test: ## run library tests (both feature configs)
	cargo test --lib
	cargo test --lib --no-default-features --features download-binaries

check: ## fmt check + clippy (both feature configs)
	cargo fmt --all --check
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --no-default-features --features download-binaries -- -D warnings

fmt: ## format the code
	cargo fmt --all

clean: ## remove build artifacts
	cargo clean
