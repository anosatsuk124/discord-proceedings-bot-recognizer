//! bot が保存した Ogg Opus を、Whisper に渡す 16kHz モノラルの f32 に変換する。
//!
//! libopus はデコード時に出力のサンプリング周波数とチャンネル数を指定できるので、
//! 48kHz ステレオの Opus を直接 16kHz モノラルで取り出す(別途のリサンプリングは要らない)。

use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use ogg::PacketReader;
use opus::{Channels, Decoder};

pub const SAMPLE_RATE: u32 = 16_000;
const OPUS_RATE: u32 = 48_000;
/// Opus の 1 パケットは最大 120ms。
const MAX_FRAME_SAMPLES: usize = (SAMPLE_RATE as usize) * 120 / 1000;

pub fn decode_ogg_opus_file(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    decode_ogg_opus(BufReader::new(file))
        .with_context(|| format!("cannot decode {}", path.display()))
}

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

    #[test]
    fn rejects_files_that_are_not_ogg_opus() {
        assert!(decode_ogg_opus(Cursor::new(b"RIFF....WAVEfmt ".to_vec())).is_err());
    }
}
