//! 音声ファイルを、Whisper に渡す 16kHz モノラルの f32 に変換する。
//!
//! bot が保存するのは Ogg Opus。libopus はデコード時に出力のサンプリング周波数とチャンネル数を
//! 指定できるので、48kHz ステレオの Opus を直接 16kHz モノラルで取り出す(別途のリサンプリングは要らない)。
//! 動作確認用に、16kHz の 16bit PCM の WAV も読める。

use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use ogg::PacketReader;
use opus::{Channels, Decoder};

/// Whisper が受け取る音声のサンプリングレート(Hz)。デコードした音声はすべてこれにそろえる。
pub const SAMPLE_RATE: u32 = 16_000;
const OPUS_RATE: u32 = 48_000;
/// Opus の 1 パケットは最大 120ms。
const MAX_FRAME_SAMPLES: usize = (SAMPLE_RATE as usize) * 120 / 1000;

/// 拡張子が .wav なら WAV、それ以外は Ogg Opus として読む。
pub fn decode_file(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let reader = BufReader::new(file);
    let is_wav = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));
    if is_wav {
        decode_wav(reader)
    } else {
        decode_ogg_opus(reader)
    }
    .with_context(|| format!("cannot decode {}", path.display()))
}

/// 16kHz・16bit PCM の WAV を読む。ステレオなら左右の平均を取る。
pub fn decode_wav<R: Read>(mut reader: R) -> Result<Vec<f32>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a RIFF/WAVE file"
    );
    let mut channels = None;
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
        let body = &bytes[offset + 8..(offset + 8 + size).min(bytes.len())];
        match id {
            b"fmt " => {
                ensure!(body.len() >= 16, "the fmt chunk is too short");
                let format = u16::from_le_bytes([body[0], body[1]]);
                let count = u16::from_le_bytes([body[2], body[3]]);
                let rate = u32::from_le_bytes(body[4..8].try_into()?);
                let bits = u16::from_le_bytes([body[14], body[15]]);
                ensure!(
                    format == 1 && bits == 16 && rate == SAMPLE_RATE && (1..=2).contains(&count),
                    "only 16-bit PCM WAV at {SAMPLE_RATE}Hz with 1 or 2 channels is supported \
                     (format={format}, bits={bits}, rate={rate}, channels={count})"
                );
                channels = Some(usize::from(count));
            }
            b"data" => {
                let channels = channels.context("the data chunk comes before the fmt chunk")?;
                let samples = body
                    .chunks_exact(2 * channels)
                    .map(|frame| {
                        frame
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|&s| f32::from(i16::from_le_bytes(s)) / 32768.0)
                            .sum::<f32>()
                            / channels as f32
                    })
                    .collect();
                return Ok(samples);
            }
            _ => {}
        }
        // チャンクは偶数バイトにそろえて並ぶ。
        offset += 8 + size + (size & 1);
    }
    bail!("the WAV file has no data chunk")
}

/// Ogg Opus を、libopus で 16kHz モノラルの f32 サンプルに直接デコードする。
pub fn decode_ogg_opus<R: Read + Seek>(reader: R) -> Result<Vec<f32>> {
    let mut packets = PacketReader::new(reader);

    let head = packets
        .read_packet()?
        .context("the file has no Ogg packets")?;
    let serial = head.stream_serial();
    let head = OpusHead::parse(&head.data)?;

    let mut decoder = Decoder::new(SAMPLE_RATE, Channels::Mono)?;
    let mut samples = Vec::new();
    let mut frame = vec![0.0f32; MAX_FRAME_SAMPLES];
    let mut header_packets = 1;
    let mut broken_packets = 0usize;
    while let Some(packet) = packets.read_packet()? {
        if packet.stream_serial() != serial {
            continue;
        }
        if header_packets < 2 {
            // 2 つ目のパケットは OpusTags(メタデータ)なので読み飛ばす。
            header_packets += 1;
            continue;
        }
        match decoder.decode_float(&packet.data, &mut frame, false) {
            Ok(count) => samples.extend_from_slice(&frame[..count]),
            Err(error) => {
                // 壊れたパケットが 1 つあっても、残りの音声は使う。
                broken_packets += 1;
                log::debug!("skipping an Opus packet that cannot be decoded: {error}");
            }
        }
    }
    if broken_packets > 0 {
        log::warn!("skipped {broken_packets} Opus packet(s) that could not be decoded");
    }

    // pre-skip は 48kHz でのサンプル数で書かれている。
    let pre_skip = usize::from(head.pre_skip) * SAMPLE_RATE as usize / OPUS_RATE as usize;
    samples.drain(..pre_skip.min(samples.len()));
    Ok(samples)
}

struct OpusHead {
    pre_skip: u16,
}

impl OpusHead {
    /// RFC 7845 5.1
    fn parse(data: &[u8]) -> Result<Self> {
        ensure!(
            data.len() >= 19 && &data[..8] == b"OpusHead",
            "the first Ogg packet is not an OpusHead"
        );
        let version = data[8];
        ensure!(version >> 4 == 0, "unsupported Ogg Opus version: {version}");
        let channels = data[9];
        let mapping_family = data[18];
        if mapping_family != 0 && channels > 2 {
            bail!("multichannel Opus ({channels} channels) is not supported");
        }
        Ok(Self {
            pre_skip: u16::from_le_bytes([data[10], data[11]]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// bot のテストと同じ、ffmpeg で作った 1.5 秒・440Hz のステレオ Ogg Opus。
    const SINE: &[u8] = include_bytes!("../tests/fixtures/sine-stereo.opus");

    #[test]
    fn decodes_to_16khz_mono() {
        let samples = decode_ogg_opus(Cursor::new(SINE)).unwrap();
        let seconds = samples.len() as f32 / SAMPLE_RATE as f32;
        assert!((seconds - 1.5).abs() < 0.05, "{seconds}s");

        // 440Hz の正弦波なので、ゼロ交差はおよそ 1 秒あたり 880 回になる。
        let middle = &samples[SAMPLE_RATE as usize / 4..SAMPLE_RATE as usize * 5 / 4];
        let crossings = middle
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        assert!((860..=900).contains(&crossings), "{crossings} crossings");
        let peak = middle.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.05 && peak <= 1.0, "peak {peak}");
    }

    fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 2 * u32::from(channels)).to_le_bytes());
        out.extend_from_slice(&(2 * channels).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn reads_16khz_pcm_wav() {
        let mono = decode_wav(Cursor::new(wav(16000, 1, &[0, 16384, -32768]))).unwrap();
        assert_eq!(mono, vec![0.0, 0.5, -1.0]);
        let stereo = decode_wav(Cursor::new(wav(16000, 2, &[16384, 0, -16384, -16384]))).unwrap();
        assert_eq!(stereo, vec![0.25, -0.5]);
    }

    #[test]
    fn rejects_other_wav_formats() {
        assert!(decode_wav(Cursor::new(wav(44100, 1, &[0, 1]))).is_err());
        assert!(decode_wav(Cursor::new(b"RIFF\0\0\0\0WAVE".to_vec())).is_err());
    }

    #[test]
    fn rejects_files_that_are_not_ogg_opus() {
        assert!(decode_ogg_opus(Cursor::new(b"RIFF....WAVEfmt ".to_vec())).is_err());
    }
}
