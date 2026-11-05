# discord-proceedings-bot-recognizer

[discord-proceedings-bot](https://github.com/anosatsuk124/discord-proceedings-bot) が録音した発言を文字起こしして、
議事録を書きます。同じ内容を次の 3 つの形で書きます。

| ファイル | 内容 |
| --- | --- |
| `proceedings.csv` | 表計算ソフト向けの CSV(`speaker,date,talk`) |
| `proceedings.html` | ブラウザで開く 1 ファイルで完結したページ。外部のファイルやスクリプトは読み込まない |
| `proceedings.md` | Markdown。メモアプリや GitHub にそのまま貼れる |

推論には [faster-whisper](https://github.com/SYSTRAN/faster-whisper) と同じ
[CTranslate2](https://github.com/OpenNMT/CTranslate2) を、Rust のバインディング [ct2rs](https://crates.io/crates/ct2rs)
から使います。faster-whisper の前処理(log-Mel 特徴量)とデコードの手順(区間ごとの逐次デコード、
温度を上げての再デコード、無音の判定、タイムスタンプでの分割)を移植しているので、同じモデルと設定なら
faster-whisper とほぼ同じ結果と速さになります。

既定のモデルは [kotoba-tech/kotoba-whisper-v2.0-faster](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) を CPU (int8) で使います。

## 使い方

```sh
# 録音が終わったセッションを 1 つ文字起こしする(<session_dir> に proceedings.csv・.html・.md を書く)
proceedings-recognizer transcribe <DATA_DIR>/sessions/<guild_id>/<session_id>

# DATA_DIR を見張り、録音が終わったセッション(manifest.json があり proceedings.csv が無いもの)を順に処理する
proceedings-recognizer watch <DATA_DIR> --interval 30

# 音声ファイル(Ogg Opus か 16kHz・16bit の WAV)を 1 つ文字起こしして標準出力に書く(動作確認用)
proceedings-recognizer file sample.ogg
```

議事録はどれも UTF-8 で、時刻順に並べます。表計算ソフトで開いたときに数式として実行されないよう、
CSV では `=` `+` `-` `@` などで始まるセル(表示名や発言)の先頭に `'` を付けます。HTML と Markdown では、
表示名や発言がタグや書式として解釈されないようにエスケープします。

watch は `proceedings.csv` の有無で処理済みかを判断します。bot は `proceedings.csv` ができたら
3 つを投稿するので、HTML と Markdown を先に、CSV を最後に書きます。

文字起こしに失敗したセッションには `transcription.error` を書き、watch はそのセッションを飛ばします。
原因を直したら `transcription.error` を消すと、次の巡回でやり直します。

## 設定(環境変数)

| 名前 | 既定値 | 内容 |
| --- | --- | --- |
| `WHISPER_MODEL` | `kotoba-tech/kotoba-whisper-v2.0-faster` | Hugging Face のリポジトリ ID、または CTranslate2 形式のモデルのディレクトリ |
| `WHISPER_MODEL_REVISION` | (最新) | モデルのリビジョン(コミット SHA など)。固定するとモデルが差し替えられても影響を受けない |
| `WHISPER_DEVICE` | `cpu` | `cpu` か `cuda`(`cuda` は `--features cuda` でビルドしたときだけ) |
| `WHISPER_COMPUTE_TYPE` | CPU では `int8` | CTranslate2 の計算の型(`int8`, `int8_float32`, `float32` など) |
| `WHISPER_CPU_THREADS` | `0` | CPU のスレッド数。`0` なら CTranslate2 の既定(4)。物理コア数にすると速い |
| `WHISPER_LANGUAGE` | `ja` | 言語。自動判定には対応していない |
| `WHISPER_BEAM_SIZE` | `5` | ビームサーチの幅 |
| `WHISPER_CHUNK_LENGTH` | `15` | 1 回にデコードする長さ(秒)。`none` ならモデルの設定(30 秒)。15 は kotoba-whisper の推奨 |
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `false` | 前の区間の文字起こしをプロンプトに使うか。kotoba-whisper の推奨は `false` |
| `PROCEEDINGS_TIMEZONE` | `UTC` | 議事録に書く時刻のタイムゾーン(例: `Asia/Tokyo`) |
| `HF_HOME` | `~/.cache/huggingface` | モデルのキャッシュの場所 |
| `HF_TOKEN` | | 非公開のモデルを使うときのトークン |
| `RUST_LOG` | `info` | ログの詳しさ(`debug` にするとデコードの様子も出る) |

モデルは初回だけ Hugging Face から取ってきます。必要なファイルがキャッシュにそろっていれば、ネットワークには出ません。

## Docker

bot のリポジトリの `compose.yaml` から bot と一緒に動かすのが基本です。単体で動かすなら:

```sh
docker build -t discord-proceedings-recognizer .
docker run --rm -v "$PWD/data:/data" -v models:/models \
  -e WHISPER_CPU_THREADS=12 -e PROCEEDINGS_TIMEZONE=Asia/Tokyo \
  discord-proceedings-recognizer watch /data
```

イメージは uid 1000 で動きます。`/data` はこのユーザーが書ける必要があります。

## 開発

ツールチェーン(Rust、CMake)は [mise](https://mise.jdx.dev/) で管理しています。CTranslate2 と oneDNN を
C++ からビルドするので、初回のビルドには時間がかかります。

```sh
mise install
mise run test     # cargo test --release(依存の C++ を二重にビルドしないよう release で統一)
mise run build    # cargo build --release
mise run lint     # cargo fmt --check と cargo clippy
```

CPU の行列演算には oneDNN を使います。faster-whisper の PyPI 版 CTranslate2 は Intel MKL と oneDNN の両方を
含みますが、MKL は Intel の CPU でしか選ばれないので、AMD の CPU ではこのビルドと同じく oneDNN で計算します。
Intel の CPU で faster-whisper と比べるときは、faster-whisper 側に `CT2_USE_MKL=0` を付けると計算経路がそろいます。
