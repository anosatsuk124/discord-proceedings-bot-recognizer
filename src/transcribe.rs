//! CTranslate2 の Whisper で音声を文字起こしする。
//!
//! faster-whisper の `WhisperModel.transcribe()`(MIT License, Copyright (c) 2023 SYSTRAN)の
//! 逐次デコード(`generate_segments` / `generate_with_fallback` /
//! `_split_segments_by_timestamps`)を移植したもの。推論は faster-whisper と同じ CTranslate2 で行う。
//!
//! 移植していないもの: 言語の自動判定、VAD フィルター、単語単位のタイムスタンプ、
//! hallucination_silence_threshold、initial_prompt / prefix / hotwords。

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ct2rs::sys::{StorageView, Whisper, WhisperGenerationResult, WhisperOptions};
use flate2::Compression;
use flate2::write::ZlibEncoder;

use crate::audio;
use crate::config::WhisperSettings;
use crate::features::{FeatureExtractor, Mel, PreprocessorConfig};
use crate::model;
use crate::tokenizer::WhisperTokenizer;

/// エンコーダーに渡すフレーム数(30 秒分)。faster-whisper の `pad_or_trim` の既定値。
const ENCODER_FRAMES: usize = 3000;
/// タイムスタンプトークン 1 つあたりの秒数。
const TIME_PRECISION: f32 = 0.02;
/// エンコーダーの畳み込みのストライド。タイムスタンプ 1 つ = 特徴量 2 フレーム。
const INPUT_STRIDE: usize = 2;
/// Whisper のデコーダーが扱える最大のトークン数。
const MAX_LENGTH: usize = 448;

/// 文字起こしの結果の 1 区間。時刻は音声の先頭からの秒数。
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// 始まり(秒)。
    pub start: f32,
    /// 終わり(秒)。
    pub end: f32,
    /// 文字起こしした文。
    pub text: String,
}

/// faster-whisper の `TranscriptionOptions` のうち使うもの。既定値は faster-whisper と同じ。
#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// 言語コード(`ja` など)。
    pub language: String,
    /// 温度 0 でデコードするときのビームサーチの幅。
    pub beam_size: usize,
    /// 温度が 0 より大きいときにサンプリングする候補の数。
    pub best_of: usize,
    /// ビームサーチの patience。
    pub patience: f32,
    /// 長さのペナルティ。
    pub length_penalty: f32,
    /// デコードに失敗したときに順に試す温度。
    pub temperatures: Vec<f32>,
    /// 圧縮率がこれより高い(繰り返しが多い)結果は失敗とみなし、次の温度でデコードし直す。
    pub compression_ratio_threshold: Option<f32>,
    /// 平均対数尤度がこれより低い結果は失敗とみなし、次の温度でデコードし直す。
    pub log_prob_threshold: Option<f32>,
    /// 無音の確率がこれより高く、平均対数尤度も低い区間は、無音として飛ばす。
    pub no_speech_threshold: Option<f32>,
    /// 前の区間の文字起こしを、次の区間のプロンプトに使うか。
    pub condition_on_previous_text: bool,
    /// この温度以上でデコードし直した区間の後は、プロンプトを空にする。
    pub prompt_reset_on_temperature: f32,
    /// 最初のタイムスタンプとして許す最大の秒数。
    pub max_initial_timestamp: f32,
}

impl DecodeOptions {
    /// 設定から作る。設定に無い値は faster-whisper の既定値にする。
    pub fn from_settings(settings: &WhisperSettings) -> Self {
        Self {
            language: settings.language.clone(),
            beam_size: settings.beam_size,
            best_of: 5,
            patience: 1.0,
            length_penalty: 1.0,
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: Some(2.4),
            log_prob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
            condition_on_previous_text: settings.condition_on_previous_text,
            prompt_reset_on_temperature: 0.5,
            max_initial_timestamp: 1.0,
        }
    }
}

/// モデルを読み込んで、音声を文字起こしするもの。
pub struct Transcriber {
    whisper: Whisper,
    tokenizer: WhisperTokenizer,
    features: FeatureExtractor,
    options: DecodeOptions,
    suppress_tokens: Vec<i32>,
    sot_sequence: Vec<u32>,
    /// 1 回のデコードで読み進める特徴量のフレーム数(chunk_length 秒分)。
    nb_max_frames: usize,
    /// 特徴量 1 フレームあたりの秒数。
    time_per_frame: f32,
}

impl Transcriber {
    /// モデルを読み込む。キャッシュに無ければ Hugging Face から取ってくる。
    pub fn load(settings: &WhisperSettings) -> Result<Self> {
        let started = Instant::now();
        let model_dir = model::resolve(&settings.model, settings.revision.as_deref())?;
        let preprocessor = PreprocessorConfig::load(&model_dir)?;
        let tokenizer = WhisperTokenizer::from_file(&model_dir.join("tokenizer.json"))?;
        let whisper = Whisper::new(
            &model_dir,
            ct2rs::Config {
                device: settings.device.to_ct2(),
                compute_type: settings.ct2_compute_type(),
                num_threads_per_replica: settings.cpu_threads,
                ..Default::default()
            },
        )
        .with_context(|| format!("cannot load the model from {}", model_dir.display()))?;

        let options = DecodeOptions::from_settings(settings);
        let sot_sequence = if whisper.is_multilingual() {
            let language_token = format!("<|{}|>", options.language);
            let language = tokenizer.token_to_id(&language_token).with_context(|| {
                format!("the model does not support language {:?}", options.language)
            })?;
            vec![tokenizer.sot, language, tokenizer.transcribe]
        } else {
            vec![tokenizer.sot]
        };
        let chunk_length = settings.chunk_length.unwrap_or(preprocessor.chunk_length);
        let nb_max_frames = chunk_length * preprocessor.sampling_rate / preprocessor.hop_length;
        if nb_max_frames == 0 || nb_max_frames > ENCODER_FRAMES {
            bail!("WHISPER_CHUNK_LENGTH must be between 1 and 30 seconds: {chunk_length}");
        }
        if preprocessor.sampling_rate != audio::SAMPLE_RATE as usize {
            bail!(
                "the model expects {}Hz audio, but only {}Hz is supported",
                preprocessor.sampling_rate,
                audio::SAMPLE_RATE
            );
        }

        log::info!(
            "loaded {} from {} in {:.1}s (compute_type={}, cpu_threads={}, chunk_length={chunk_length}s)",
            settings.model,
            model_dir.display(),
            started.elapsed().as_secs_f32(),
            settings.compute_type,
            settings.cpu_threads,
        );
        Ok(Self {
            suppress_tokens: tokenizer.suppressed_tokens()?,
            features: FeatureExtractor::new(&preprocessor),
            time_per_frame: preprocessor.hop_length as f32 / preprocessor.sampling_rate as f32,
            whisper,
            tokenizer,
            options,
            sot_sequence,
            nb_max_frames,
        })
    }

    /// 音声ファイルを読んで文字起こしする。
    pub fn transcribe_file(&self, path: &Path) -> Result<Vec<Segment>> {
        let samples = audio::decode_file(path)?;
        self.transcribe(&samples)
            .with_context(|| format!("cannot transcribe {}", path.display()))
    }

    /// 16kHz モノラルの音声を文字起こしする。
    pub fn transcribe(&self, samples: &[f32]) -> Result<Vec<Segment>> {
        let mel = self.features.compute(samples);
        let content_frames = mel.frames.saturating_sub(1);

        let mut segments = Vec::new();
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut prompt_reset_since = 0;
        let mut seek = 0;
        while seek < content_frames {
            let time_offset = seek as f32 * self.time_per_frame;
            let segment_size = self.nb_max_frames.min(content_frames - seek);
            let segment_duration = segment_size as f32 * self.time_per_frame;

            let encoder_output = self.encode(&mel, seek, segment_size)?;
            let prompt = self.prompt(&all_tokens[prompt_reset_since..]);
            let decoded = self.generate_with_fallback(&encoder_output, &prompt)?;

            if let Some(threshold) = self.options.no_speech_threshold {
                let silent = decoded.result.no_speech_prob > threshold
                    && !self
                        .options
                        .log_prob_threshold
                        .is_some_and(|lp| decoded.avg_logprob > lp);
                if silent {
                    log::debug!(
                        "no speech at {time_offset:.2}s (no_speech_prob={:.2})",
                        decoded.result.no_speech_prob
                    );
                    seek += segment_size;
                    continue;
                }
            }

            let tokens: Vec<u32> = decoded.result.sequences_ids[0]
                .iter()
                .map(|&id| id as u32)
                .collect();
            let (current, next_seek) = split_segments_by_timestamps(
                &tokens,
                self.tokenizer.timestamp_begin(),
                time_offset,
                segment_size,
                segment_duration,
                seek,
            );
            // 同じ位置を何度もデコードしないよう、少なくとも 1 区間分は進める(faster-whisper には無い保険)。
            seek = if next_seek > seek {
                next_seek
            } else {
                seek + segment_size
            };

            for segment in current {
                let text = self.tokenizer.decode(&segment.tokens)?;
                let text = text.trim();
                if segment.start == segment.end || text.is_empty() {
                    continue;
                }
                all_tokens.extend_from_slice(&segment.tokens);
                segments.push(Segment {
                    start: segment.start,
                    end: segment.end,
                    text: text.to_owned(),
                });
            }

            if !self.options.condition_on_previous_text
                || decoded.temperature > self.options.prompt_reset_on_temperature
            {
                prompt_reset_since = all_tokens.len();
            }
        }
        Ok(segments)
    }

    fn encode(&self, mel: &Mel, seek: usize, len: usize) -> Result<StorageView<'static>> {
        let mut window = mel.window(seek, len, ENCODER_FRAMES);
        let features = StorageView::new(
            &[1, mel.n_mels, ENCODER_FRAMES],
            &mut window,
            ct2rs::Device::default(),
        )?;
        self.whisper.encode(&features, false)
    }

    /// faster-whisper の `get_prompt`(prefix / hotwords なし、タイムスタンプあり)。
    fn prompt(&self, previous_tokens: &[u32]) -> Vec<u32> {
        let mut prompt = Vec::new();
        if !previous_tokens.is_empty() {
            prompt.push(self.tokenizer.sot_prev);
            let keep = MAX_LENGTH / 2 - 1;
            prompt
                .extend_from_slice(&previous_tokens[previous_tokens.len().saturating_sub(keep)..]);
        }
        prompt.extend_from_slice(&self.sot_sequence);
        prompt
    }

    /// 温度を上げながらデコードし直す(faster-whisper の `generate_with_fallback`)。
    fn generate_with_fallback(
        &self,
        encoder_output: &StorageView,
        prompt: &[u32],
    ) -> Result<Decoded> {
        let o = &self.options;
        let prompt_tokens = prompt
            .iter()
            .map(|&id| self.tokenizer.id_to_token(id))
            .collect::<Result<Vec<_>>>()?;
        let max_initial_timestamp_index =
            (o.max_initial_timestamp / TIME_PRECISION).round() as usize;

        let mut all_results: Vec<Decoded> = Vec::new();
        let mut below_threshold: Vec<usize> = Vec::new();
        for &temperature in &o.temperatures {
            let mut options = WhisperOptions {
                length_penalty: o.length_penalty,
                max_length: MAX_LENGTH,
                return_scores: true,
                return_no_speech_prob: true,
                suppress_blank: true,
                suppress_tokens: self.suppress_tokens.clone(),
                max_initial_timestamp_index,
                ..Default::default()
            };
            if temperature > 0.0 {
                options.beam_size = 1;
                options.num_hypotheses = o.best_of;
                options.sampling_topk = 0;
                options.sampling_temperature = temperature;
            } else {
                options.beam_size = o.beam_size;
                options.patience = o.patience;
            }

            let result = self
                .whisper
                .generate(
                    encoder_output,
                    std::slice::from_ref(&prompt_tokens),
                    &options,
                )?
                .into_iter()
                .next()
                .context("CTranslate2 returned no result")?;
            let tokens: Vec<u32> = result.sequences_ids[0]
                .iter()
                .map(|&id| id as u32)
                .collect();
            let seq_len = tokens.len() as f32;
            let score = result.scores.first().copied().unwrap_or(0.0);
            let avg_logprob = score * seq_len.powf(o.length_penalty) / (seq_len + 1.0);
            let text = self.tokenizer.decode(&tokens)?;
            let compression_ratio = compression_ratio(text.trim());

            let mut needs_fallback = false;
            if let Some(threshold) = o.compression_ratio_threshold {
                if compression_ratio > threshold {
                    needs_fallback = true; // 繰り返しが多すぎる
                    log::debug!(
                        "compression ratio {compression_ratio:.2} > {threshold} at temperature {temperature}"
                    );
                } else {
                    below_threshold.push(all_results.len());
                }
            }
            if o.log_prob_threshold.is_some_and(|lp| avg_logprob < lp) {
                needs_fallback = true; // 平均対数確率が低すぎる
                log::debug!(
                    "avg log prob {avg_logprob:.2} is too low at temperature {temperature}"
                );
            }
            if o.no_speech_threshold
                .is_some_and(|ns| result.no_speech_prob > ns)
                && o.log_prob_threshold.is_some_and(|lp| avg_logprob < lp)
            {
                needs_fallback = false; // 無音
            }

            let decoded = Decoded {
                result,
                avg_logprob,
                temperature,
            };
            if !needs_fallback {
                return Ok(decoded);
            }
            all_results.push(decoded);
        }

        // どの温度でも条件を満たさなかったときは、圧縮率の条件を満たしたもののうち
        // (無ければ全体のうち)平均対数確率が最も高いものを使う。温度は最後に試した値にする。
        let last_temperature = o.temperatures.last().copied().unwrap_or(0.0);
        let candidates: Vec<usize> = if below_threshold.is_empty() {
            (0..all_results.len()).collect()
        } else {
            below_threshold
        };
        let best = candidates
            .into_iter()
            .max_by(|&a, &b| {
                all_results[a]
                    .avg_logprob
                    .total_cmp(&all_results[b].avg_logprob)
            })
            .context("no temperatures configured")?;
        let mut decoded = all_results.swap_remove(best);
        decoded.temperature = last_temperature;
        Ok(decoded)
    }
}

struct Decoded {
    result: WhisperGenerationResult,
    avg_logprob: f32,
    temperature: f32,
}

/// faster-whisper の `get_compression_ratio`。繰り返しの多い(幻覚の疑いがある)出力ほど大きくなる。
fn compression_ratio(text: &str) -> f32 {
    let bytes = text.as_bytes();
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(bytes)
        .expect("writing to a Vec cannot fail");
    let compressed = encoder.finish().expect("writing to a Vec cannot fail");
    bytes.len() as f32 / compressed.len() as f32
}

#[derive(Debug, PartialEq)]
struct TokenSegment {
    start: f32,
    end: f32,
    tokens: Vec<u32>,
}

/// タイムスタンプトークンで区間に分け、次に読み始める位置(フレーム)を求める
/// (faster-whisper の `_split_segments_by_timestamps`)。
fn split_segments_by_timestamps(
    tokens: &[u32],
    timestamp_begin: u32,
    time_offset: f32,
    segment_size: usize,
    segment_duration: f32,
    seek: usize,
) -> (Vec<TokenSegment>, usize) {
    let is_timestamp = |token: u32| token >= timestamp_begin;
    let timestamp_time = |token: u32| (token as f32 - timestamp_begin as f32) * TIME_PRECISION;

    let single_timestamp_ending = tokens.len() >= 2
        && !is_timestamp(tokens[tokens.len() - 2])
        && is_timestamp(tokens[tokens.len() - 1]);
    let consecutive: Vec<usize> = (1..tokens.len())
        .filter(|&i| is_timestamp(tokens[i]) && is_timestamp(tokens[i - 1]))
        .collect();

    let mut segments = Vec::new();
    if consecutive.is_empty() {
        let mut duration = segment_duration;
        if let Some(&last) = tokens.iter().rev().find(|&&t| is_timestamp(t))
            && last != timestamp_begin
        {
            duration = timestamp_time(last);
        }
        segments.push(TokenSegment {
            start: time_offset,
            end: time_offset + duration,
            tokens: tokens.to_vec(),
        });
        return (segments, seek + segment_size);
    }

    let mut slices = consecutive;
    if single_timestamp_ending {
        slices.push(tokens.len());
    }
    let mut last_slice = 0;
    for &current in &slices {
        let sliced = &tokens[last_slice..current];
        segments.push(TokenSegment {
            start: time_offset + timestamp_time(sliced[0]),
            end: time_offset + timestamp_time(sliced[sliced.len() - 1]),
            tokens: sliced.to_vec(),
        });
        last_slice = current;
    }
    let next_seek = if single_timestamp_ending {
        seek + segment_size
    } else {
        let last_timestamp_position = (tokens[last_slice - 1] - timestamp_begin) as usize;
        seek + last_timestamp_position * INPUT_STRIDE
    };
    (segments, next_seek)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TB: u32 = 1000; // テスト用の timestamp_begin
    const fn ts(seconds_x50: u32) -> u32 {
        TB + seconds_x50
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn splits_on_consecutive_timestamps() {
        // <|0.00|> a b <|1.00|><|1.00|> c <|2.00|><|2.00|> d <|3.00|>
        let tokens = [ts(0), 1, 2, ts(50), ts(50), 3, ts(100), ts(100), 4, ts(150)];
        let (segments, seek) = split_segments_by_timestamps(&tokens, TB, 10.0, 1500, 15.0, 1000);
        assert_eq!(segments.len(), 3);
        assert!(approx(segments[0].start, 10.0) && approx(segments[0].end, 11.0));
        assert_eq!(segments[0].tokens, vec![ts(0), 1, 2, ts(50)]);
        assert!(approx(segments[1].start, 11.0) && approx(segments[1].end, 12.0));
        // 最後が単独のタイムスタンプで終わるので、最後の区間も含めて区間ぶん丸ごと進める。
        assert!(approx(segments[2].start, 12.0) && approx(segments[2].end, 13.0));
        assert_eq!(segments[2].tokens, vec![ts(100), 4, ts(150)]);
        assert_eq!(seek, 1000 + 1500);
    }

    #[test]
    fn seeks_to_the_last_complete_timestamp() {
        // <|0.00|> a <|1.00|><|1.00|> b(途中で切れている)
        let tokens = [ts(0), 1, ts(50), ts(50), 2];
        let (segments, seek) = split_segments_by_timestamps(&tokens, TB, 0.0, 1500, 15.0, 0);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].tokens, vec![ts(0), 1, ts(50)]);
        // 最後の完全な区間の終わり(1.00 秒 = タイムスタンプ 50 = 特徴量 100 フレーム)から読み直す。
        assert_eq!(seek, 100);
    }

    #[test]
    fn single_segment_uses_the_last_timestamp_as_duration() {
        // <|0.00|> a b <|2.40|>
        let tokens = [ts(0), 1, 2, ts(120)];
        let (segments, seek) = split_segments_by_timestamps(&tokens, TB, 5.0, 300, 3.0, 500);
        assert_eq!(segments.len(), 1);
        assert!(approx(segments[0].start, 5.0) && approx(segments[0].end, 7.4));
        assert_eq!(seek, 800);
    }

    #[test]
    fn without_timestamps_covers_the_whole_window() {
        let tokens = [1, 2, 3];
        let (segments, seek) = split_segments_by_timestamps(&tokens, TB, 0.0, 300, 3.0, 0);
        assert_eq!(segments.len(), 1);
        assert!(approx(segments[0].end, 3.0));
        assert_eq!(seek, 300);
    }

    #[test]
    fn compression_ratio_detects_repetition() {
        let normal = compression_ratio("今日はいい天気ですね。会議を始めます。");
        let repeated = compression_ratio(&"ありがとうございました。".repeat(20));
        assert!(normal < 2.4, "{normal}");
        assert!(repeated > 2.4, "{repeated}");
        assert_eq!(compression_ratio(""), 0.0);
    }
}
