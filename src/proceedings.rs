//! セッション 1 つ分の録音から議事録 CSV を作る。

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};

use crate::manifest::SessionManifest;
use crate::transcribe::{Segment, Transcriber};

pub const OUTPUT_FILE: &str = "proceedings.csv";

/// 音声ファイルを文字起こしするもの。テストでは偽物に差し替える。
pub trait SpeechToText {
    fn transcribe_file(&self, audio: &Path) -> Result<Vec<Segment>>;
}

impl SpeechToText for Transcriber {
    fn transcribe_file(&self, audio: &Path) -> Result<Vec<Segment>> {
        Transcriber::transcribe_file(self, audio)
    }
}

/// 議事録の 1 行。列は以前の recognizer と同じ speaker, date, talk。
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub speaker: String,
    pub date: Timestamp,
    pub talk: String,
}

/// セッションの発言をすべて文字起こしして、時刻順の行にする。
///
/// 読めない音声ファイルがあってもセッション全体は止めず、その発言だけを飛ばす。
pub fn transcribe_session(session_dir: &Path, stt: &dyn SpeechToText) -> Result<Vec<Row>> {
    let manifest = SessionManifest::load(session_dir)?;
    let mut rows = Vec::new();
    let mut skipped = 0;
    for utterance in &manifest.utterances {
        let audio = session_dir.join(&utterance.file);
        let segments = match stt.transcribe_file(&audio) {
            Ok(segments) => segments,
            Err(error) => {
                log::error!("skipping {}: {error:#}", audio.display());
                skipped += 1;
                continue;
            }
        };
        for segment in segments {
            let offset = SignedDuration::try_from_secs_f32(segment.start.max(0.0))
                .context("segment start is out of range")?;
            rows.push(Row {
                speaker: utterance.speaker.clone(),
                date: utterance.started_at.checked_add(offset)?,
                talk: segment.text,
            });
        }
    }
    if skipped > 0 {
        log::warn!(
            "{}: skipped {skipped} of {} utterance(s)",
            session_dir.display(),
            manifest.utterances.len()
        );
    }
    rows.sort_by_key(|row| row.date);
    Ok(rows)
}

/// 議事録を CSV で書く。watch は出力ファイルの有無で処理済みかを判断するので、
/// 一時ファイルに書いてから置き換える。
pub fn write_csv(rows: &[Row], output: &Path, timezone: &TimeZone) -> Result<()> {
    let mut temporary = output.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = Path::new(&temporary);

    let file = File::create(temporary)
        .with_context(|| format!("cannot create {}", temporary.display()))?;
    let mut writer = csv::Writer::from_writer(BufWriter::new(file));
    writer.write_record(["speaker", "date", "talk"])?;
    for row in rows {
        let date = row
            .date
            .to_zoned(timezone.clone())
            .strftime("%Y-%m-%dT%H:%M:%S%:z")
            .to_string();
        writer.write_record([row.speaker.as_str(), &date, &row.talk])?;
    }
    writer
        .into_inner()
        .map_err(|error| error.into_error())?
        .into_inner()
        .map_err(|error| error.into_error())?
        .sync_all()?;
    std::fs::rename(temporary, output)
        .with_context(|| format!("cannot write {}", output.display()))?;
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    /// ファイル名ごとに決まった結果を返す偽物。
    pub(crate) struct FakeStt(pub HashMap<String, Result<Vec<Segment>, String>>);

    impl SpeechToText for FakeStt {
        fn transcribe_file(&self, audio: &Path) -> Result<Vec<Segment>> {
            let name = audio.file_name().unwrap().to_str().unwrap();
            match self.0.get(name) {
                Some(Ok(segments)) => Ok(segments.clone()),
                Some(Err(message)) => Err(anyhow::anyhow!(message.clone())),
                None => Err(anyhow::anyhow!("no such file: {name}")),
            }
        }
    }

    pub(crate) fn segment(start: f32, text: &str) -> Segment {
        Segment {
            start,
            end: start + 1.0,
            text: text.to_owned(),
        }
    }

    pub(crate) fn write_manifest(dir: &Path) {
        std::fs::write(
            dir.join("manifest.json"),
            r#"{
                "version": 1,
                "session_id": "s",
                "guild_id": "g",
                "channel_id": "c",
                "started_at": "2026-11-05T00:00:00Z",
                "ended_at": "2026-11-05T01:00:00Z",
                "utterances": [
                    {"id": "a", "user_id": "1", "speaker": "山田", "started_at": "2026-11-05T00:00:10Z", "ended_at": "2026-11-05T00:00:20Z", "file": "a.ogg"},
                    {"id": "b", "user_id": "2", "speaker": "佐藤", "started_at": "2026-11-05T00:00:12Z", "ended_at": "2026-11-05T00:00:14Z", "file": "b.ogg"},
                    {"id": "c", "user_id": "3", "speaker": "鈴木", "started_at": "2026-11-05T00:00:30Z", "ended_at": "2026-11-05T00:00:31Z", "file": "c.ogg"}
                ]
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn rows_are_ordered_by_time_and_failed_utterances_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        write_manifest(dir.path());
        let stt = FakeStt(HashMap::from([
            (
                "a.ogg".to_owned(),
                Ok(vec![segment(0.0, "始めます。"), segment(5.5, "以上です。")]),
            ),
            ("b.ogg".to_owned(), Ok(vec![segment(0.5, "はい。")])),
            ("c.ogg".to_owned(), Err("broken".to_owned())),
        ]));

        let rows = transcribe_session(dir.path(), &stt).unwrap();
        let summary: Vec<(&str, String, &str)> = rows
            .iter()
            .map(|r| (r.speaker.as_str(), r.date.to_string(), r.talk.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("山田", "2026-11-05T00:00:10Z".to_owned(), "始めます。"),
                ("佐藤", "2026-11-05T00:00:12.5Z".to_owned(), "はい。"),
                ("山田", "2026-11-05T00:00:15.5Z".to_owned(), "以上です。"),
            ]
        );
    }

    #[test]
    fn csv_uses_the_configured_time_zone() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join(OUTPUT_FILE);
        let rows = vec![Row {
            speaker: "山田".to_owned(),
            date: "2026-11-05T00:00:10.7Z".parse().unwrap(),
            talk: "カンマ, と \"引用符\" を含む".to_owned(),
        }];
        write_csv(&rows, &output, &TimeZone::get("Asia/Tokyo").unwrap()).unwrap();

        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            "speaker,date,talk\n山田,2026-11-05T09:00:10+09:00,\"カンマ, と \"\"引用符\"\" を含む\"\n"
        );
        assert!(!dir.path().join("proceedings.csv.tmp").exists());
    }
}
