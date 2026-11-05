//! Whisper の入力になる log-Mel スペクトログラム。
//!
//! faster-whisper の `FeatureExtractor`(MIT License, Copyright (c) 2023 SYSTRAN)を移植したもの。
//! 窓や正規化の仕方を合わせないと認識精度が落ちるので、細部まで同じにしている。

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use realfft::{RealFftPlanner, RealToComplex};
use serde::Deserialize;

/// モデルの preprocessor_config.json のうち、特徴量の計算に使う値。
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
pub struct PreprocessorConfig {
    pub feature_size: usize,
    pub sampling_rate: usize,
    pub hop_length: usize,
    pub chunk_length: usize,
    pub n_fft: usize,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        // faster-whisper の FeatureExtractor の既定値
        Self {
            feature_size: 80,
            sampling_rate: 16_000,
            hop_length: 160,
            chunk_length: 30,
            n_fft: 400,
        }
    }
}

impl PreprocessorConfig {
    /// モデルのディレクトリから読む。ファイルが無ければ既定値を使う。
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("preprocessor_config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("invalid {}", path.display()))
    }
}

/// `[n_mels][frames]` の行優先で並べた特徴量。
#[derive(Clone, Debug)]
pub struct Mel {
    pub n_mels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

impl Mel {
    /// `start` から `len` フレームを切り出し、`target` フレームになるまで 0 で埋める
    /// (faster-whisper の `pad_or_trim`)。
    pub fn window(&self, start: usize, len: usize, target: usize) -> Vec<f32> {
        let len = len.min(target).min(self.frames.saturating_sub(start));
        let mut out = vec![0.0f32; self.n_mels * target];
        for mel in 0..self.n_mels {
            let src = &self.data[mel * self.frames + start..][..len];
            out[mel * target..][..len].copy_from_slice(src);
        }
        out
    }
}

pub struct FeatureExtractor {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    /// `[n_mels][n_fft / 2 + 1]`
    mel_filters: Vec<f32>,
    /// 各 Mel フィルターが 0 でない周波数ビンの範囲。
    mel_ranges: Vec<std::ops::Range<usize>>,
    window: Vec<f64>,
    fft: Arc<dyn RealToComplex<f64>>,
}

impl FeatureExtractor {
    pub fn new(config: &PreprocessorConfig) -> Self {
        let n_fft = config.n_fft;
        let n_bins = n_fft / 2 + 1;
        let mel_filters = mel_filters(config.sampling_rate, n_fft, config.feature_size);
        let mel_ranges = (0..config.feature_size)
            .map(|mel| {
                let row = &mel_filters[mel * n_bins..][..n_bins];
                let first = row.iter().position(|&w| w != 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&w| w != 0.0).map_or(0, |i| i + 1);
                first..last.max(first)
            })
            .collect();
        // np.hanning(n_fft + 1)[:-1](周期的なハン窓)
        let window = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
            .collect();
        Self {
            n_fft,
            hop_length: config.hop_length,
            n_mels: config.feature_size,
            mel_filters,
            mel_ranges,
            window,
            fft: RealFftPlanner::<f64>::new().plan_fft_forward(n_fft),
        }
    }

    /// 音声全体の log-Mel スペクトログラムを計算する(faster-whisper の `FeatureExtractor.__call__`)。
    pub fn compute(&self, samples: &[f32]) -> Mel {
        // 末尾に hop_length 分の 0 を足し(padding=160)、両端を n_fft / 2 だけ鏡映で延ばす(center=True)。
        let mut waveform: Vec<f64> = samples.iter().map(|&s| f64::from(s)).collect();
        waveform.resize(samples.len() + self.hop_length, 0.0);
        let pad = self.n_fft / 2;
        let padded: Vec<f64> = (0..waveform.len() + 2 * pad)
            .map(|i| waveform[reflect_index(i as isize - pad as isize, waveform.len())])
            .collect();

        let n_bins = self.n_fft / 2 + 1;
        // STFT の最後のフレームは使わない(magnitudes = stft[..., :-1])。
        let frames = (1 + (padded.len() - self.n_fft) / self.hop_length).saturating_sub(1);

        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0.0f32; n_bins];
        let mut data = vec![0.0f32; self.n_mels * frames];
        for frame in 0..frames {
            let start = frame * self.hop_length;
            for ((x, &s), &w) in input
                .iter_mut()
                .zip(&padded[start..start + self.n_fft])
                .zip(&self.window)
            {
                *x = s * w;
            }
            self.fft
                .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                .expect("buffer sizes come from the same plan");
            for (p, c) in power.iter_mut().zip(&spectrum) {
                *p = c.norm_sqr() as f32;
            }
            for (mel, range) in self.mel_ranges.iter().enumerate() {
                let filter = &self.mel_filters[mel * n_bins..][range.clone()];
                let energy: f32 = filter
                    .iter()
                    .zip(&power[range.clone()])
                    .map(|(w, p)| w * p)
                    .sum();
                data[mel * frames + frame] = energy;
            }
        }

        // log10 を取り、全体の最大値から 8 (= 80dB) 以上小さい値を切り上げてから正規化する。
        let mut max = f32::NEG_INFINITY;
        for value in &mut data {
            *value = value.max(1e-10).log10();
            max = max.max(*value);
        }
        for value in &mut data {
            *value = (value.max(max - 8.0) + 4.0) / 4.0;
        }

        Mel {
            n_mels: self.n_mels,
            frames,
            data,
        }
    }
}

/// numpy の `np.pad(mode="reflect")` と同じ添字の折り返し。
fn reflect_index(index: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let i = index.rem_euclid(period);
    (if i < len as isize { i } else { period - i }) as usize
}

/// Slaney 形式の Mel フィルターバンク(librosa の `filters.mel` と同じ)。`[n_mels][n_fft / 2 + 1]`
fn mel_filters(sampling_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| i as f64 * sampling_rate as f64 / n_fft as f64)
        .collect();

    let min_mel = 0.0;
    let max_mel = 45.245_640_471_924_965;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let log_step = 6.4f64.ln() / 27.0;
    let freqs: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let mel = min_mel + (max_mel - min_mel) * i as f64 / (n_mels + 1) as f64;
            if mel >= min_log_mel {
                min_log_hz * (log_step * (mel - min_log_mel)).exp()
            } else {
                f_sp * mel
            }
        })
        .collect();

    let mut weights = vec![0.0f32; n_mels * n_bins];
    for mel in 0..n_mels {
        let enorm = 2.0 / (freqs[mel + 2] - freqs[mel]);
        for (bin, &fft_freq) in fft_freqs.iter().enumerate() {
            let lower = (fft_freq - freqs[mel]) / (freqs[mel + 1] - freqs[mel]);
            let upper = (freqs[mel + 2] - fft_freq) / (freqs[mel + 2] - freqs[mel + 1]);
            weights[mel * n_bins + bin] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    /// faster-whisper 1.2.1 の FeatureExtractor(feature_size=128)で求めた値。
    /// 入力は 0.5 秒の 16kHz 信号 `0.3 * sin(2π * 440 t) + 0.1 * sin(2π * 3000 t)`。
    /// 求め方は tests/fixtures/README.md を参照。
    const REFERENCE: &str = include_str!("../tests/fixtures/features-reference.json");

    #[derive(Deserialize)]
    struct Reference {
        frames: usize,
        mean: f64,
        min: f64,
        max: f64,
        /// (mel, frame, value)
        samples: Vec<(usize, usize, f64)>,
        filter_row_sums: Vec<f64>,
    }

    fn test_signal() -> Vec<f32> {
        (0..8000)
            .map(|i| {
                let t = i as f64 / 16000.0;
                (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.1 * (2.0 * std::f64::consts::PI * 3000.0 * t).sin()) as f32
            })
            .collect()
    }

    fn extractor() -> FeatureExtractor {
        FeatureExtractor::new(&PreprocessorConfig {
            feature_size: 128,
            ..Default::default()
        })
    }

    #[test]
    fn mel_filters_match_faster_whisper() {
        let reference: Reference = serde_json::from_str(REFERENCE).unwrap();
        let filters = mel_filters(16000, 400, 128);
        for (mel, expected) in reference.filter_row_sums.iter().enumerate() {
            let sum: f64 = filters[mel * 201..][..201]
                .iter()
                .map(|&w| f64::from(w))
                .sum();
            assert!(
                (sum - expected).abs() < 1e-6,
                "mel {mel}: {sum} vs {expected}"
            );
        }
    }

    #[test]
    fn features_match_faster_whisper() {
        let reference: Reference = serde_json::from_str(REFERENCE).unwrap();
        let mel = extractor().compute(&test_signal());
        assert_eq!(mel.n_mels, 128);
        assert_eq!(mel.frames, reference.frames);

        let mean = mel.data.iter().map(|&v| f64::from(v)).sum::<f64>() / mel.data.len() as f64;
        let min = mel.data.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        let max = mel.data.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        assert!(
            (mean - reference.mean).abs() < 1e-4,
            "mean {mean} vs {}",
            reference.mean
        );
        assert!((f64::from(min) - reference.min).abs() < 1e-4);
        assert!((f64::from(max) - reference.max).abs() < 1e-4);
        for &(m, f, expected) in &reference.samples {
            let value = f64::from(mel.data[m * mel.frames + f]);
            assert!(
                (value - expected).abs() < 1e-3,
                "[{m}][{f}] {value} vs {expected}"
            );
        }
    }

    #[test]
    fn window_pads_with_zeros() {
        let mel = Mel {
            n_mels: 2,
            frames: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        assert_eq!(
            mel.window(1, 2, 4),
            vec![2.0, 3.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0]
        );
        assert_eq!(mel.window(0, 3, 2), vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn reflect_matches_numpy() {
        // np.pad(np.arange(4), 3, mode="reflect") == [3 2 1 0 1 2 3 2 1 0]
        let indices: Vec<usize> = (-3..7).map(|i| reflect_index(i, 4)).collect();
        assert_eq!(indices, vec![3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
    }

    #[test]
    fn very_short_audio_does_not_panic() {
        let extractor = extractor();
        assert_eq!(extractor.compute(&[]).frames, 1);
        assert!(extractor.compute(&[0.1; 10]).frames >= 1);
    }
}
