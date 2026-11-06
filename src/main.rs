//! discord-proceedings-bot の録音を文字起こしして、議事録(CSV・HTML・Markdown)を書く。
//!
//! 推論は faster-whisper と同じ CTranslate2 で行い、前処理とデコードの手順も faster-whisper に合わせている。

mod audio;
mod config;
mod features;
mod manifest;
mod model;
mod proceedings;
mod report;
mod tokenizer;
mod transcribe;
mod vad;
mod watch;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::config::{WhisperSettings, output_timezone};
use crate::proceedings::{transcribe_session, write_proceedings};
use crate::transcribe::Transcriber;

/// モデルや言語などの設定は環境変数(WHISPER_*, PROCEEDINGS_TIMEZONE)で指定する。README を参照。
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 録音が終わったセッションを 1 つ文字起こしして、議事録を書く
    Transcribe {
        /// セッションのディレクトリ(manifest.json があるところ)
        session_dir: PathBuf,
        /// proceedings.csv・.html・.md を書くディレクトリ。省略するとセッションのディレクトリ
        #[arg(short, long)]
        output_dir: Option<PathBuf>,
    },
    /// DATA_DIR/sessions を見張り、録音が終わったセッションを順に文字起こしする
    Watch {
        /// bot と共有するデータディレクトリ
        data_dir: PathBuf,
        /// 新しいセッションを探す間隔(秒)
        #[arg(long, default_value_t = 30)]
        interval: u64,
    },
    /// 音声ファイル(Ogg Opus か 16kHz・16bit の WAV)を 1 つ文字起こしして、標準出力に書く(動作確認用)
    File { audio: PathBuf },
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            log::error!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    let settings = WhisperSettings::from_env()?;
    let timezone = output_timezone()?;
    match cli.command {
        Command::Transcribe {
            session_dir,
            output_dir,
        } => {
            let transcriber = Transcriber::load(&settings)?;
            let proceedings = transcribe_session(&session_dir, &transcriber)?;
            let output_dir = output_dir.unwrap_or_else(|| session_dir.clone());
            write_proceedings(&proceedings, &output_dir, &timezone)?;
            log::info!(
                "wrote {} line(s) of proceedings to {}",
                proceedings.rows.len(),
                output_dir.display()
            );
        }
        Command::Watch { data_dir, interval } => {
            let transcriber = Transcriber::load(&settings)?;
            watch::run(
                &data_dir,
                &transcriber,
                &timezone,
                Duration::from_secs(interval),
            );
        }
        Command::File { audio } => {
            let transcriber = Transcriber::load(&settings)?;
            let started = Instant::now();
            let samples = audio::decode_file(&audio)?;
            let decoded = started.elapsed();
            let segments = transcriber.transcribe(&samples)?;
            for segment in &segments {
                println!(
                    "[{:.2}s -> {:.2}s] {}",
                    segment.start, segment.end, segment.text
                );
            }
            log::info!(
                "{:.1}s of audio: decoded in {:.2}s, transcribed in {:.2}s",
                samples.len() as f32 / audio::SAMPLE_RATE as f32,
                decoded.as_secs_f32(),
                (started.elapsed() - decoded).as_secs_f32(),
            );
        }
    }
    Ok(())
}
