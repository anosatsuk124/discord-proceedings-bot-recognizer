//! 環境変数から読む設定。

use anyhow::{Context, Result, anyhow, bail};
use ct2rs::{ComputeType, Device};
use jiff::tz::TimeZone;

/// `WHISPER_MODEL` を指定しないときに使うモデル。
pub const DEFAULT_MODEL: &str = "kotoba-tech/kotoba-whisper-v2.0-faster";

/// モデルと文字起こしの設定。
///
/// 既定値は kotoba-whisper-v2.0-faster のモデルカードの推奨
/// (`language="ja"`, `chunk_length=15`, `condition_on_previous_text=False`) に合わせ、
/// CPU では int8 で動かす。それ以外は faster-whisper の `transcribe()` の既定値と同じ。
#[derive(Clone, Debug, PartialEq)]
pub struct WhisperSettings {
    /// Hugging Face のリポジトリ ID、またはローカルの CTranslate2 モデルのディレクトリ。
    pub model: String,
    /// Hugging Face のリビジョン(コミット SHA など)。固定するとモデルの差し替えを防げる。
    pub revision: Option<String>,
    /// 推論に使うデバイス。
    pub device: DeviceKind,
    /// CTranslate2 の計算の型(`int8`, `float32` など)。読み込むときに検証済み。
    pub compute_type: String,
    /// 0 なら CTranslate2 の既定(4 スレッド)。
    pub cpu_threads: usize,
    /// 文字起こしする言語。多言語モデルでは必須(自動判定には対応していない)。
    pub language: String,
    /// ビームサーチの幅。
    pub beam_size: usize,
    /// 1 回にデコードする音声の長さ(秒)。None ならモデルの preprocessor_config.json に従う。
    pub chunk_length: Option<usize>,
    /// 前の区間の文字起こしを、次の区間のプロンプトに使うか。
    pub condition_on_previous_text: bool,
}

/// 推論に使うデバイス。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// CPU。
    Cpu,
    /// NVIDIA の GPU(`cuda` フィーチャーを付けてビルドしたときだけ使える)。
    Cuda,
}

impl DeviceKind {
    /// CTranslate2 のデバイスに変換する。
    pub fn to_ct2(self) -> Device {
        match self {
            Self::Cpu => Device::CPU,
            Self::Cuda => Device::CUDA,
        }
    }
}

impl WhisperSettings {
    /// 環境変数(`WHISPER_*`)から読む。
    pub fn from_env() -> Result<Self> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// 名前から値を引く関数を使って読む。テストでは環境変数の代わりに使う。
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let get = |name: &str| {
            lookup(name)
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        };

        let device = match get("WHISPER_DEVICE").as_deref() {
            None | Some("cpu") => DeviceKind::Cpu,
            Some("cuda") => DeviceKind::Cuda,
            Some(other) => bail!("WHISPER_DEVICE must be cpu or cuda: {other}"),
        };
        let compute_type = get("WHISPER_COMPUTE_TYPE").unwrap_or_else(|| {
            // GPU では保存されている精度(このモデルなら float16)のまま、CPU では int8 を既定にする。
            match device {
                DeviceKind::Cpu => "int8".to_owned(),
                DeviceKind::Cuda => "default".to_owned(),
            }
        });
        parse_compute_type(&compute_type)?;

        let chunk_length = match get("WHISPER_CHUNK_LENGTH") {
            Some(value) if value.eq_ignore_ascii_case("none") => None,
            Some(value) => Some(parse_number(&value, "WHISPER_CHUNK_LENGTH", 1)?),
            None => Some(15),
        };

        Ok(Self {
            model: get("WHISPER_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_owned()),
            revision: get("WHISPER_MODEL_REVISION"),
            device,
            compute_type,
            cpu_threads: get("WHISPER_CPU_THREADS")
                .map(|value| parse_number(&value, "WHISPER_CPU_THREADS", 0))
                .transpose()?
                .unwrap_or(0),
            language: get("WHISPER_LANGUAGE").unwrap_or_else(|| "ja".to_owned()),
            beam_size: get("WHISPER_BEAM_SIZE")
                .map(|value| parse_number(&value, "WHISPER_BEAM_SIZE", 1))
                .transpose()?
                .unwrap_or(5),
            chunk_length,
            condition_on_previous_text: get("WHISPER_CONDITION_ON_PREVIOUS_TEXT")
                .map(|value| parse_bool(&value, "WHISPER_CONDITION_ON_PREVIOUS_TEXT"))
                .transpose()?
                .unwrap_or(false),
        })
    }

    /// CTranslate2 の計算の型。
    pub fn ct2_compute_type(&self) -> ComputeType {
        parse_compute_type(&self.compute_type).expect("validated in from_lookup")
    }
}

/// 議事録の時刻のタイムゾーンを指定する環境変数。
const TIMEZONE_VARIABLE: &str = "PROCEEDINGS_TIMEZONE";

/// 議事録の時刻を書くタイムゾーン(PROCEEDINGS_TIMEZONE、既定は UTC)。
pub fn output_timezone() -> Result<TimeZone> {
    output_timezone_from(std::env::var(TIMEZONE_VARIABLE).ok().as_deref())
}

/// タイムゾーン名を解釈する。空か未指定なら UTC。
pub fn output_timezone_from(name: Option<&str>) -> Result<TimeZone> {
    match name.map(str::trim).filter(|name| !name.is_empty()) {
        None => Ok(TimeZone::UTC),
        Some(name) => TimeZone::get(name)
            .with_context(|| format!("PROCEEDINGS_TIMEZONE is not a valid time zone: {name}")),
    }
}

fn parse_compute_type(value: &str) -> Result<ComputeType> {
    Ok(match value.to_ascii_lowercase().as_str() {
        "default" => ComputeType::DEFAULT,
        "auto" => ComputeType::AUTO,
        "float32" => ComputeType::FLOAT32,
        "int8" => ComputeType::INT8,
        "int8_float32" => ComputeType::INT8_FLOAT32,
        "int8_float16" => ComputeType::INT8_FLOAT16,
        "int8_bfloat16" => ComputeType::INT8_BFLOAT16,
        "int16" => ComputeType::INT16,
        "float16" => ComputeType::FLOAT16,
        "bfloat16" => ComputeType::BFLOAT16,
        _ => bail!("WHISPER_COMPUTE_TYPE is not a CTranslate2 compute type: {value}"),
    })
}

fn parse_number(value: &str, name: &str, minimum: usize) -> Result<usize> {
    let number: usize = value
        .parse()
        .map_err(|_| anyhow!("{name} must be an integer: {value}"))?;
    if number < minimum {
        bail!("{name} must be >= {minimum}: {value}");
    }
    Ok(number)
}

fn parse_bool(value: &str, name: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{name} must be a boolean (true/false): {value}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn settings(vars: &[(&str, &str)]) -> Result<WhisperSettings> {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        WhisperSettings::from_lookup(|name| vars.get(name).cloned())
    }

    #[test]
    fn defaults_follow_the_kotoba_model_card_on_cpu() {
        let s = settings(&[]).unwrap();
        assert_eq!(s.model, DEFAULT_MODEL);
        assert_eq!(s.revision, None);
        assert_eq!(s.device, DeviceKind::Cpu);
        assert_eq!(s.compute_type, "int8");
        assert_eq!(s.cpu_threads, 0);
        assert_eq!(s.language, "ja");
        assert_eq!(s.beam_size, 5);
        assert_eq!(s.chunk_length, Some(15));
        assert!(!s.condition_on_previous_text);
    }

    #[test]
    fn values_can_be_overridden() {
        let s = settings(&[
            ("WHISPER_MODEL", "/models/large-v3"),
            ("WHISPER_MODEL_REVISION", "abc123"),
            ("WHISPER_COMPUTE_TYPE", "int8_float32"),
            ("WHISPER_CPU_THREADS", "12"),
            ("WHISPER_LANGUAGE", "en"),
            ("WHISPER_BEAM_SIZE", "1"),
            ("WHISPER_CHUNK_LENGTH", "none"),
            ("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "true"),
        ])
        .unwrap();
        assert_eq!(s.model, "/models/large-v3");
        assert_eq!(s.revision.as_deref(), Some("abc123"));
        assert_eq!(s.compute_type, "int8_float32");
        assert_eq!(s.cpu_threads, 12);
        assert_eq!(s.language, "en");
        assert_eq!(s.beam_size, 1);
        assert_eq!(s.chunk_length, None);
        assert!(s.condition_on_previous_text);
    }

    #[test]
    fn cuda_keeps_the_stored_precision_by_default() {
        let s = settings(&[("WHISPER_DEVICE", "cuda")]).unwrap();
        assert_eq!(s.compute_type, "default");
    }

    #[test]
    fn invalid_values_are_rejected() {
        for (name, value) in [
            ("WHISPER_DEVICE", "tpu"),
            ("WHISPER_COMPUTE_TYPE", "int4"),
            ("WHISPER_CPU_THREADS", "-1"),
            ("WHISPER_BEAM_SIZE", "0"),
            ("WHISPER_CHUNK_LENGTH", "0"),
            ("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "maybe"),
        ] {
            assert!(settings(&[(name, value)]).is_err(), "{name}={value}");
        }
    }

    #[test]
    fn empty_values_mean_default() {
        let s = settings(&[("WHISPER_MODEL", "  "), ("WHISPER_LANGUAGE", "")]).unwrap();
        assert_eq!(s.model, DEFAULT_MODEL);
        assert_eq!(s.language, "ja");
    }

    #[test]
    fn timezone() {
        assert_eq!(output_timezone_from(None).unwrap(), TimeZone::UTC);
        assert_eq!(
            output_timezone_from(Some("Asia/Tokyo"))
                .unwrap()
                .iana_name(),
            Some("Asia/Tokyo")
        );
        assert!(output_timezone_from(Some("Mars/Olympus")).is_err());
    }
}
