use std::path::PathBuf;
mod audio;
mod whisper;

use crate::whisper::{whisper, whisper_init};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the discord bot output directory
    #[arg(short, long)]
    path: PathBuf,
    //    /// Path to the output csv file
    //    #[arg(short, long)]
    //    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let params = whisper_init();

    let audio = crate::audio::wav_to_integer_mono(&args.path);

    whisper(params, &audio);
}
