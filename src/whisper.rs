use once_cell::sync::OnceCell;
use std::{env, path::PathBuf, sync::Mutex};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

const MODEL_PATH_ENV: &str = "MODEL_PATH";

pub static WHISPER_CTX: OnceCell<Mutex<WhisperContext>> = OnceCell::new();

pub fn whisper_init() -> FullParams<'static, 'static> {
    let model_path = PathBuf::from(env::var(MODEL_PATH_ENV).expect("Couldn't get the MODEL_PATH"))
        .canonicalize()
        .unwrap()
        .display()
        .to_string();
    let model_path = model_path.as_str();
    let _ctx = WHISPER_CTX
        .get_or_init(|| Mutex::new(WhisperContext::new(model_path).expect("failed to load model")))
        .lock()
        .unwrap();

    let mut params = FullParams::new(SamplingStrategy::Greedy { n_past: 0 });

    // Edit params as needed.
    // Set the number of threads to use to 1.
    params.set_n_threads(2);
    params.set_language("ja");
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);

    params
}

pub fn whisper(params: FullParams, audio: &[f32]) -> Vec<String> {
    let mut ctx = WHISPER_CTX.get().unwrap().lock().unwrap();

    ctx.full(params, audio).expect("failed to run model");

    let num_segments = ctx.full_n_segments();

    let mut segments = Vec::new();
    for i in 0..num_segments {
        // Get the transcribed text and timestamps for the current segment.
        let segment = ctx.full_get_segment_text(i).expect("failed to get segment");
        let _start_timestamp = ctx.full_get_segment_t0(i);
        let _end_timestamp = ctx.full_get_segment_t1(i);

        segments.push(segment);
    }
    segments
}
