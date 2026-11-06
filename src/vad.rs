//! 音声から無音を取り除く VAD(音声区間検出)。faster-whisper の `vad_filter=True` と同じことをする。
//!
//! Whisper は無音を渡されると、「ごめん」などの言っていない文字列を書き出すことがある。
//! そこで、発話のある区間だけをつないで Whisper に渡し、結果の時刻を元の音声の時刻に戻す。
//!
//! faster-whisper の `vad.py`(MIT License, Copyright (c) 2023 SYSTRAN。元は Silero VAD)から、
//! 次のものを移植した。
//! - `SileroVADModel`
//! - `get_speech_timestamps`
//! - `collect_chunks`
//! - `SpeechTimestampsMap`
//!
//! モデルは、faster-whisper が同梱している Silero VAD v6 の ONNX をそのまま使い、
//! 純 Rust の tract で推論する。

use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use tract_onnx::prelude::*;

use crate::audio::SAMPLE_RATE;

/// faster-whisper 1.2.1 が同梱している Silero VAD v6(入手元とライセンスは assets/README.md)。
const MODEL: &[u8] = include_bytes!("../assets/silero_vad_v6.onnx");
/// VAD の 1 窓のサンプル数(32ms)。
const WINDOW: usize = 512;
/// 各窓の前に付ける、直前の窓の末尾のサンプル数。
const CONTEXT: usize = 64;
/// モデルに渡す 1 行(直前の窓の末尾 + 窓)のサンプル数。
const ROW: usize = CONTEXT + WINDOW;
/// LSTM の状態の大きさ。
const STATE: usize = 128;
/// 1 回の推論でまとめて渡す窓の数(faster-whisper の `encoder_batch_size`)。
const BATCH_WINDOWS: usize = 10_000;

/// faster-whisper の `VadOptions`。既定値も同じ。
#[derive(Clone, Debug, PartialEq)]
pub struct VadOptions {
    /// 発話の確率がこれ以上の窓を、発話とみなす。
    pub threshold: f64,
    /// 発話の途中で確率がこれを下回ったら、無音とみなす。None なら `threshold - 0.15`(最小 0.01)。
    pub neg_threshold: Option<f64>,
    /// これ以下の長さの発話の区間は捨てる(ミリ秒)。
    pub min_speech_duration_ms: u32,
    /// 発話の区間をこれより長くしない(秒)。超えそうなら、途中の無音で区切る。
    pub max_speech_duration_s: f64,
    /// 無音がこれだけ続いたら、発話の区間を閉じる(ミリ秒)。
    pub min_silence_duration_ms: u32,
    /// 発話の区間の前後に足す余白(ミリ秒)。
    pub speech_pad_ms: u32,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: None,
            min_speech_duration_ms: 0,
            max_speech_duration_s: f64::INFINITY,
            min_silence_duration_ms: 2000,
            speech_pad_ms: 400,
        }
    }
}

/// 発話の区間。位置はサンプル単位で、`end` は区間に含まない。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeechChunk {
    /// 区間の始まり。
    pub start: usize,
    /// 区間の終わり(含まない)。
    pub end: usize,
}

/// Silero VAD のモデル。
pub struct SpeechDetector {
    model: Arc<TypedRunnableModel>,
}

impl SpeechDetector {
    /// バイナリに埋め込んだモデルを読み込む。
    pub fn load() -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut Cursor::new(MODEL))
            .and_then(|model| model.into_optimized())
            .and_then(|model| model.into_runnable())
            .context("cannot load the Silero VAD model")?;
        Ok(Self { model })
    }

    /// 発話の区間を求める。
    pub fn speech_chunks(&self, audio: &[f32], options: &VadOptions) -> Result<Vec<SpeechChunk>> {
        let probs = self.speech_probs(audio)?;
        Ok(speech_timestamps(&probs, audio.len(), options))
    }

    /// 512 サンプルの窓ごとに、発話の確率を求める(faster-whisper の `SileroVADModel.__call__`)。
    ///
    /// 末尾は 0 で埋めて 512 の倍数にする。faster-whisper と同じく、ちょうど倍数のときも 1 窓足す。
    /// 各窓には直前の窓の末尾 64 サンプルを前に付け(最初の窓には 0)、LSTM の状態は次の窓へ引き継ぐ。
    pub fn speech_probs(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let windows = audio.len() / WINDOW + 1;
        let mut padded = audio.to_vec();
        padded.resize(windows * WINDOW, 0.0);

        let mut h: TValue = Tensor::zero::<f32>(&[1, 1, STATE])?.into();
        let mut c: TValue = Tensor::zero::<f32>(&[1, 1, STATE])?.into();
        let mut probs = Vec::with_capacity(windows);
        for first in (0..windows).step_by(BATCH_WINDOWS) {
            let count = BATCH_WINDOWS.min(windows - first);
            let mut batch = vec![0.0f32; count * ROW];
            for (offset, row) in batch.as_chunks_mut::<ROW>().0.iter_mut().enumerate() {
                let start = (first + offset) * WINDOW;
                if start > 0 {
                    row[..CONTEXT].copy_from_slice(&padded[start - CONTEXT..start]);
                }
                row[CONTEXT..].copy_from_slice(&padded[start..start + WINDOW]);
            }
            let input = Tensor::from_shape(&[count, ROW], &batch)?;
            let outputs = self.model.run(tvec!(input.into(), h, c))?;
            ensure!(
                outputs.len() == 3,
                "the VAD model returned {} outputs",
                outputs.len()
            );
            probs.extend(outputs[0].to_plain_array_view::<f32>()?.iter().copied());
            h = outputs[1].clone();
            c = outputs[2].clone();
        }
        ensure!(
            probs.len() == windows,
            "the VAD model returned {} probabilities for {windows} windows",
            probs.len()
        );
        Ok(probs)
    }
}

/// 窓ごとの発話の確率から、発話の区間を求める(faster-whisper の `get_speech_timestamps`)。
///
/// 確率と閾値は f32 で比べる。faster-whisper は numpy の float32 と Python の float を比べており、
/// numpy 2 ではその比較が float32 で行われるため。
pub fn speech_timestamps(
    probs: &[f32],
    audio_length: usize,
    options: &VadOptions,
) -> Vec<SpeechChunk> {
    let rate = f64::from(SAMPLE_RATE);
    let window = WINDOW as i64;
    let audio_length = audio_length as i64;
    let threshold = options.threshold as f32;
    let neg_threshold = options
        .neg_threshold
        .unwrap_or_else(|| (options.threshold - 0.15).max(0.01)) as f32;
    let min_speech_samples = rate * f64::from(options.min_speech_duration_ms) / 1000.0;
    let speech_pad_samples = rate * f64::from(options.speech_pad_ms) / 1000.0;
    let max_speech_samples =
        rate * options.max_speech_duration_s - WINDOW as f64 - 2.0 * speech_pad_samples;
    let min_silence_samples = rate * f64::from(options.min_silence_duration_ms) / 1000.0;
    let min_silence_samples_at_max_speech = rate * 98.0 / 1000.0;

    // faster-whisper と同じく、0 を「まだ無い」の意味で使う。
    let mut triggered = false;
    let mut speeches: Vec<(i64, i64)> = Vec::new();
    let mut current_start: Option<i64> = None;
    let mut temp_end = 0; // 無音が始まった位置(区間の終わりの候補)
    let mut prev_end = 0; // 最長を超えたときに区切る位置の候補
    let mut next_start = 0; // 区切ったあとに次の区間を始める位置の候補

    for (index, &prob) in probs.iter().enumerate() {
        let position = window * index as i64;
        if prob >= threshold && temp_end != 0 {
            temp_end = 0;
            if next_start < prev_end {
                next_start = position;
            }
        }

        if prob >= threshold && !triggered {
            triggered = true;
            current_start = Some(position);
            continue;
        }

        if triggered
            && let Some(start) = current_start
            && (position - start) as f64 > max_speech_samples
        {
            if prev_end != 0 {
                speeches.push((start, prev_end));
                // 区切った位置より前に発話が戻っていなければ、いったん区間を閉じる。
                current_start = if next_start < prev_end {
                    triggered = false;
                    None
                } else {
                    Some(next_start)
                };
                (prev_end, next_start, temp_end) = (0, 0, 0);
            } else {
                speeches.push((start, position));
                current_start = None;
                (prev_end, next_start, temp_end) = (0, 0, 0);
                triggered = false;
                continue;
            }
        }

        if prob < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = position;
            }
            // ごく短い無音では区切らない
            if (position - temp_end) as f64 > min_silence_samples_at_max_speech {
                prev_end = temp_end;
            }
            if ((position - temp_end) as f64) < min_silence_samples {
                continue;
            }
            if let Some(start) = current_start
                && (temp_end - start) as f64 > min_speech_samples
            {
                speeches.push((start, temp_end));
            }
            current_start = None;
            (prev_end, next_start, temp_end) = (0, 0, 0);
            triggered = false;
        }
    }

    if let Some(start) = current_start
        && (audio_length - start) as f64 > min_speech_samples
    {
        speeches.push((start, audio_length));
    }

    pad_speeches(&mut speeches, audio_length, speech_pad_samples);
    speeches
        .into_iter()
        .map(|(start, end)| SpeechChunk {
            start: start.max(0) as usize,
            end: end.max(0) as usize,
        })
        .collect()
}

/// 区間の前後に余白を足す。隣の区間との間が余白 2 つ分より狭ければ、間を半分ずつ分ける。
fn pad_speeches(speeches: &mut [(i64, i64)], audio_length: i64, pad: f64) {
    let count = speeches.len();
    for index in 0..count {
        if index == 0 {
            speeches[0].0 = (speeches[0].0 as f64 - pad).max(0.0) as i64;
        }
        let padded_end = (audio_length as f64).min(speeches[index].1 as f64 + pad) as i64;
        if index + 1 == count {
            speeches[index].1 = padded_end;
            continue;
        }
        let silence = speeches[index + 1].0 - speeches[index].1;
        if (silence as f64) < 2.0 * pad {
            let half = silence.div_euclid(2);
            speeches[index].1 += half;
            speeches[index + 1].0 = (speeches[index + 1].0 - half).max(0);
        } else {
            speeches[index].1 = padded_end;
            speeches[index + 1].0 = (speeches[index + 1].0 as f64 - pad).max(0.0) as i64;
        }
    }
}

/// 発話の区間だけをつなげた音声を作る(faster-whisper の `collect_chunks`。長さの上限なし)。
pub fn collect_chunks(audio: &[f32], chunks: &[SpeechChunk]) -> Vec<f32> {
    let mut speech = Vec::with_capacity(
        chunks
            .iter()
            .map(|chunk| chunk.end.saturating_sub(chunk.start))
            .sum(),
    );
    for chunk in chunks {
        let end = chunk.end.min(audio.len());
        speech.extend_from_slice(&audio[chunk.start.min(end)..end]);
    }
    speech
}

/// つなげた音声の時刻を、元の音声の時刻に戻す(faster-whisper の `SpeechTimestampsMap`)。
pub struct TimestampMap {
    /// 各区間の終わりの、つなげた音声での位置(サンプル)。
    chunk_end_sample: Vec<i64>,
    /// 各区間より前に取り除いた無音の長さ(秒)。
    total_silence_before: Vec<f64>,
}

impl TimestampMap {
    /// 発話の区間から作る。
    pub fn new(chunks: &[SpeechChunk]) -> Self {
        let rate = f64::from(SAMPLE_RATE);
        let mut previous_end = 0;
        let mut silent_samples = 0;
        let mut chunk_end_sample = Vec::with_capacity(chunks.len());
        let mut total_silence_before = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            silent_samples += chunk.start as i64 - previous_end;
            previous_end = chunk.end as i64;
            chunk_end_sample.push(chunk.end as i64 - silent_samples);
            total_silence_before.push(silent_samples as f64 / rate);
        }
        Self {
            chunk_end_sample,
            total_silence_before,
        }
    }

    /// つなげた音声での時刻(秒)を、元の音声での時刻にする。小数第 2 位に丸める。
    ///
    /// 区間の終わりちょうどの時刻は、`is_end` なら前の区間、そうでなければ次の区間のものとみなす。
    pub fn original_time(&self, time: f32, is_end: bool) -> f32 {
        let Some(index) = self.chunk_index(time, is_end) else {
            return time;
        };
        let original = self.total_silence_before[index] + f64::from(time);
        ((original * 100.0).round_ties_even() / 100.0) as f32
    }

    fn chunk_index(&self, time: f32, is_end: bool) -> Option<usize> {
        let last = self.chunk_end_sample.len().checked_sub(1)?;
        let sample = (f64::from(time) * f64::from(SAMPLE_RATE)) as i64;
        if is_end && let Some(index) = self.chunk_end_sample.iter().position(|&end| end == sample) {
            return Some(index);
        }
        Some(
            self.chunk_end_sample
                .partition_point(|&end| end <= sample)
                .min(last),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    /// faster-whisper 1.2.1 で求めた参照値。求め方は tests/fixtures/README.md を参照。
    const REFERENCE: &str = include_str!("../tests/fixtures/vad-reference.json");

    #[derive(Deserialize)]
    struct Reference {
        model_probs: Vec<String>,
        speech_timestamps: Vec<TimestampsCase>,
        timestamp_map: MapCase,
    }

    #[derive(Deserialize)]
    struct TimestampsCase {
        name: String,
        options: OptionsOverride,
        audio_length: usize,
        probs: Vec<String>,
        expected: Vec<(usize, usize)>,
    }

    #[derive(Deserialize)]
    struct OptionsOverride {
        threshold: Option<f64>,
        neg_threshold: Option<f64>,
        min_speech_duration_ms: Option<u32>,
        max_speech_duration_s: Option<f64>,
        min_silence_duration_ms: Option<u32>,
        speech_pad_ms: Option<u32>,
    }

    impl OptionsOverride {
        fn apply(&self) -> VadOptions {
            let d = VadOptions::default();
            VadOptions {
                threshold: self.threshold.unwrap_or(d.threshold),
                neg_threshold: self.neg_threshold.or(d.neg_threshold),
                min_speech_duration_ms: self
                    .min_speech_duration_ms
                    .unwrap_or(d.min_speech_duration_ms),
                max_speech_duration_s: self
                    .max_speech_duration_s
                    .unwrap_or(d.max_speech_duration_s),
                min_silence_duration_ms: self
                    .min_silence_duration_ms
                    .unwrap_or(d.min_silence_duration_ms),
                speech_pad_ms: self.speech_pad_ms.unwrap_or(d.speech_pad_ms),
            }
        }
    }

    #[derive(Deserialize)]
    struct MapCase {
        chunks: Vec<(usize, usize)>,
        queries: Vec<(f32, bool, f32)>,
    }

    fn reference() -> Reference {
        serde_json::from_str(REFERENCE).expect("the fixture is valid JSON")
    }

    fn parse_f32(values: &[String]) -> Vec<f32> {
        values
            .iter()
            .map(|value| value.parse().expect("a float"))
            .collect()
    }

    /// 1.5 秒の合成信号のあとに 0.5 秒の無音(参照値を作ったスクリプトと同じ式)。
    fn model_signal() -> Vec<f32> {
        let rate = f64::from(SAMPLE_RATE);
        let voiced = (0..(1.5 * rate) as usize).map(|n| {
            let t = n as f64 / rate;
            let tau = std::f64::consts::TAU;
            (0.3 * (tau * 220.0 * t).sin() * (0.5 + 0.5 * (tau * 4.0 * t).sin())
                + 0.05 * (tau * 1800.0 * t).sin()) as f32
        });
        voiced
            .chain(std::iter::repeat_n(0.0, SAMPLE_RATE as usize / 2))
            .collect()
    }

    #[test]
    fn model_matches_faster_whisper() {
        let expected = parse_f32(&reference().model_probs);
        let detector = SpeechDetector::load().unwrap();
        let probs = detector.speech_probs(&model_signal()).unwrap();
        assert_eq!(probs.len(), expected.len());
        for (index, (got, want)) in probs.iter().zip(&expected).enumerate() {
            assert!((got - want).abs() < 1e-4, "window {index}: {got} != {want}");
        }
        // 参照値に発話と無音の両方が入っていること(確率がどれも同じような値では確かめにならない)
        assert!(expected.iter().any(|&p| p > 0.9));
        assert!(expected.iter().any(|&p| p < 0.1));
    }

    #[test]
    fn silence_has_no_speech() {
        let detector = SpeechDetector::load().unwrap();
        // 3 秒の完全な無音と、-90dBFS ほどのごく小さい雑音
        let silence = vec![0.0f32; 3 * SAMPLE_RATE as usize];
        let hiss: Vec<f32> = (0..3 * SAMPLE_RATE as usize)
            .map(|n| 3e-5 * ((n * 7919 % 101) as f32 / 50.0 - 1.0))
            .collect();
        for audio in [silence, hiss] {
            assert!(
                detector
                    .speech_chunks(&audio, &VadOptions::default())
                    .unwrap()
                    .is_empty()
            );
        }
        // 音声が空でも 1 窓分は推論する(faster-whisper と同じ)
        assert_eq!(detector.speech_probs(&[]).unwrap().len(), 1);
    }

    #[test]
    fn speech_timestamps_match_faster_whisper() {
        for case in reference().speech_timestamps {
            let chunks: Vec<(usize, usize)> = speech_timestamps(
                &parse_f32(&case.probs),
                case.audio_length,
                &case.options.apply(),
            )
            .into_iter()
            .map(|chunk| (chunk.start, chunk.end))
            .collect();
            assert_eq!(chunks, case.expected, "{}", case.name);
        }
    }

    #[test]
    fn timestamp_map_matches_faster_whisper() {
        let case = reference().timestamp_map;
        let chunks: Vec<SpeechChunk> = case
            .chunks
            .iter()
            .map(|&(start, end)| SpeechChunk { start, end })
            .collect();
        let map = TimestampMap::new(&chunks);
        for (time, is_end, expected) in case.queries {
            let got = map.original_time(time, is_end);
            assert!(
                (got - expected).abs() < 1e-6,
                "{time} (is_end={is_end}): {got} != {expected}"
            );
        }
        assert_eq!(TimestampMap::new(&[]).original_time(1.5, false), 1.5);
    }

    #[test]
    fn collect_chunks_concatenates_speech() {
        let audio: Vec<f32> = (0..10).map(|n| n as f32).collect();
        let chunks = [
            SpeechChunk { start: 1, end: 3 },
            SpeechChunk { start: 6, end: 8 },
            // 音声より長い区間は、音声の終わりで切る
            SpeechChunk { start: 9, end: 12 },
        ];
        assert_eq!(
            collect_chunks(&audio, &chunks),
            vec![1.0, 2.0, 6.0, 7.0, 9.0]
        );
        assert!(collect_chunks(&audio, &[]).is_empty());
    }
}
