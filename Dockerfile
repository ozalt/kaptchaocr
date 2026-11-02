# syntax=docker/dockerfile:1

# ---------- build stage ----------
# Alpine toolchain for a musl-linked release binary. The inline-model feature
# embeds the stock models at compile time, so they must be present here.
FROM rust:1.90.0-alpine AS builder

RUN apk add --no-cache \
    build-base \
    musl-dev \
    perl \
    make \
    openssl-dev \
    openssl-libs-static \
    pkgconfig \
    git \
    cmake \
    linux-headers

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY initialize.json list.json ./
COPY model/stock_common.onnx model/stock_common.json model/stock_old.onnx model/stock_old.json model/stock_det.onnx model/models.json ./model/

RUN cargo build --release --locked

# ---------- runtime stage ----------
FROM alpine:3.20

RUN apk add --no-cache ca-certificates libgcc libstdc++

WORKDIR /app

COPY --from=builder /app/target/release/kaptchaocr /usr/local/bin/kaptchaocr
COPY --from=builder /app/model ./model
COPY model/finetuned ./model/finetuned

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD wget -qO- http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["kaptchaocr"]
# Default: all features, balanced fine-tuned recognition model (ft-small).
CMD ["--address", "0.0.0.0:8000", "--ocr", "--det", "--slide", \
     "--manifest", "model/models.json", "--profile", "balanced"]
