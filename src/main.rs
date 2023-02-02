use std::{io::BufWriter, path::PathBuf};
mod audio;
mod parse_json;
mod serialize_to_csv;
mod whisper;

use crate::whisper::{whisper, whisper_init};
use clap::Parser;
use csv::WriterBuilder;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the discord bot output directory
    #[arg(short, long)]
    path: PathBuf,
    /// Path to the output csv file
    #[arg(short, long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();

    let talks = crate::parse_json::TalkList::read(&args.path.join("talks.json"));

    let csv_file = BufWriter::new(std::fs::File::create(&args.output).unwrap());

    let mut csv_file = WriterBuilder::new().from_writer(csv_file);

    for talk in talks.0.into_iter() {
        let audio =
            crate::audio::wav_to_integer_mono(&args.path.join(format!("{}.wav", talk.uuid)));
        let params = whisper_init();
        for line in whisper(params, &audio).iter() {
            println!(
                "speaker: {:?}, date: {:?}: {}",
                talk.discord_name, talk.date, line
            );
            let csv = crate::serialize_to_csv::Csv::new(
                talk.discord_name.to_string(),
                talk.date.to_string(),
                line.to_string(),
            );
            csv_file.serialize(csv).unwrap();
        }
    }
    csv_file.flush().unwrap();
}
