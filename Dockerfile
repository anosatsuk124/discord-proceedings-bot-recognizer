# レガシービルダー(BuildKit なし)でもビルドできるよう、BuildKit 専用の構文は使わない。
ARG RUST_VERSION=1.99.1

FROM rust:${RUST_VERSION}-slim-trixie AS build
# CTranslate2 / oneDNN / libopus を C++ / C からビルドする
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake g++ make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src

# 依存(特に CTranslate2 と oneDNN)のビルドは重いので、ソースより先にビルドしてレイヤーに残す。
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release --locked \
    && rm -rf src target/release/deps/discord_proceedings_bot_recognizer-* \
        target/release/discord-proceedings-bot-recognizer*

COPY src ./src
RUN cargo build --release --locked \
    && strip target/release/discord-proceedings-bot-recognizer

FROM debian:trixie-slim
# libgomp1: CTranslate2 と oneDNN が使う OpenMP
# ca-certificates: Hugging Face からモデルを取ってくるときの TLS
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --user-group --no-create-home --home-dir /nonexistent app \
    && mkdir -p /models /data \
    && chown app:app /models /data
COPY --from=build /src/target/release/discord-proceedings-bot-recognizer /usr/local/bin/proceedings-recognizer
USER app
# モデルのキャッシュ(Hugging Face の形式)
ENV HF_HOME=/models
VOLUME ["/models", "/data"]
ENTRYPOINT ["proceedings-recognizer"]
CMD ["watch", "/data"]
