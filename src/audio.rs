use std::path::PathBuf;

pub fn wav_to_integer_mono(file: &PathBuf) -> Vec<f32> {
    let mut reader = hound::WavReader::open(file).expect("failed to open file");
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: _,
        ..
    } = reader.spec();

    let mut audio = whisper_rs::convert_integer_to_float_audio(
        &reader
            .samples::<i16>()
            .map(|s| s.expect("invalid sample"))
            .collect::<Vec<_>>(),
    );

    if sample_rate != 16000 {
        panic!("sample rate must be 16KHz");
    }

    if channels == 2 {
        audio = whisper_rs::convert_stereo_to_mono_audio(&audio);
    }

    audio
}
