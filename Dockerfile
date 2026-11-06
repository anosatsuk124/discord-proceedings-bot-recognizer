# レガシービルダー(BuildKit なし)でもビルドできるよう、BuildKit 専用の構文は使わない。
ARG RUST_VERSION=1.99.1

FROM rust:${RUST_VERSION}-slim-trixie AS build
# CTranslate2 / oneDNN / libopus を C++ / C からビルドする
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake g++ make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src

# 依存(特に CTranslate2 と oneDNN)のビルドは重いので、ソースより先にビルドしてレイヤーに残す。
# デプロイ先でほかのサービスと同居しながらビルドするので、優先度を最低にしておく。
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
    && nice -n 19 cargo build --release --locked \
    && rm -rf src target/release/deps/discord_proceedings_bot_recognizer-* \
        target/release/discord-proceedings-bot-recognizer*

COPY src ./src
# VAD のモデル(バイナリに埋め込む)
COPY assets ./assets
RUN nice -n 19 cargo build --release --locked \
    && strip target/release/discord-proceedings-bot-recognizer \
    # 実行用イメージに無い OpenMP のランタイム(CTranslate2 と oneDNN が使う)と、
    # uid 1000 が書けるモデルとデータの置き場所を用意しておく
    && mkdir -p /out/lib /out/models /out/data \
    && cp -L "$(gcc -print-file-name=libgomp.so.1)" /out/lib/

# 実行用はシェルやパッケージマネージャーを含まない distroless(glibc / libstdc++ / CA 証明書のみ)にする。
FROM gcr.io/distroless/cc-debian13
COPY --from=build /out/lib/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libgomp.so.1
COPY --from=build /src/target/release/discord-proceedings-bot-recognizer /usr/local/bin/proceedings-recognizer
COPY --from=build --chown=1000:1000 /out/models /models
COPY --from=build --chown=1000:1000 /out/data /data
# bot と同じ uid 1000 で動かす(共有するデータディレクトリをどちらも書けるように)
USER 1000:1000
# モデルのキャッシュ(Hugging Face の形式)
ENV HF_HOME=/models
VOLUME ["/models", "/data"]
ENTRYPOINT ["/usr/local/bin/proceedings-recognizer"]
CMD ["watch", "/data"]
