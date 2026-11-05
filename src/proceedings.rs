//! セッション 1 つ分の録音から議事録を作る。
//!
//! 議事録は同じ内容を 3 つの形で書く。CSV(表計算ソフト向け)、HTML(1 ファイルで完結する
//! ページ)、Markdown。HTML と Markdown の中身は [`crate::report`] で組み立てる。

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};

use crate::manifest::SessionManifest;
use crate::report;
use crate::transcribe::{Segment, Transcriber};

/// 議事録の CSV。watch はこれの有無で処理済みかを判断するので、3 つのうち最後に書く。
pub const CSV_FILE: &str = "proceedings.csv";
/// 議事録の HTML(1 ファイルで完結するページ)。
pub const HTML_FILE: &str = "proceedings.html";
/// 議事録の Markdown。
pub const MARKDOWN_FILE: &str = "proceedings.md";

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
    /// 発言者の表示名。
    pub speaker: String,
    /// 発言の時刻。
    pub date: Timestamp,
    /// 発言の内容。
    pub talk: String,
}

/// セッション 1 つ分の議事録。
#[derive(Clone, Debug, PartialEq)]
pub struct Proceedings {
    /// セッション ID。
    pub session_id: String,
    /// サーバー名。古い manifest には無い。
    pub guild_name: Option<String>,
    /// ボイスチャンネル名。古い manifest には無い。
    pub channel_name: Option<String>,
    /// 録音を始めた時刻。
    pub started_at: Timestamp,
    /// 録音を終えた時刻。
    pub ended_at: Timestamp,
    /// 時刻順の発言。
    pub rows: Vec<Row>,
}

/// セッションの発言をすべて文字起こしして、時刻順の議事録にする。
///
/// 読めない音声ファイルがあってもセッション全体は止めず、その発言だけを飛ばす。
pub fn transcribe_session(session_dir: &Path, stt: &dyn SpeechToText) -> Result<Proceedings> {
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
    Ok(Proceedings {
        session_id: manifest.session_id,
        guild_name: manifest.guild_name,
        channel_name: manifest.channel_name,
        started_at: manifest.started_at,
        ended_at: manifest.ended_at,
        rows,
    })
}

/// 議事録を CSV・HTML・Markdown で output_dir に書く。
///
/// watch は CSV の有無で処理済みかを判断し、bot は CSV ができたら 3 つを投稿するので、
/// HTML と Markdown を先に、CSV を最後に書く。どれも一時ファイルに書いてから置き換える。
pub fn write_proceedings(
    proceedings: &Proceedings,
    output_dir: &Path,
    timezone: &TimeZone,
) -> Result<()> {
    write_atomically(
        &output_dir.join(HTML_FILE),
        report::render_html(proceedings, timezone).as_bytes(),
    )?;
    write_atomically(
        &output_dir.join(MARKDOWN_FILE),
        report::render_markdown(proceedings, timezone).as_bytes(),
    )?;
    write_atomically(
        &output_dir.join(CSV_FILE),
        &render_csv(&proceedings.rows, timezone)?,
    )
}

fn write_atomically(output: &Path, contents: &[u8]) -> Result<()> {
    let mut temporary = output.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = Path::new(&temporary);

    let mut file = std::fs::File::create(temporary)
        .with_context(|| format!("cannot create {}", temporary.display()))?;
    file.write_all(contents)?;
    file.sync_all()?;
    std::fs::rename(temporary, output)
        .with_context(|| format!("cannot write {}", output.display()))?;
    Ok(())
}

/// 議事録の CSV。列は以前の recognizer と同じ speaker, date, talk。
fn render_csv(rows: &[Row], timezone: &TimeZone) -> Result<Vec<u8>> {
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record(["speaker", "date", "talk"])?;
    for row in rows {
        let date = row
            .date
            .to_zoned(timezone.clone())
            .strftime("%Y-%m-%dT%H:%M:%S%:z")
            .to_string();
        writer.write_record([
            neutralize_formula(&row.speaker).as_ref(),
            &date,
            neutralize_formula(&row.talk).as_ref(),
        ])?;
    }
    writer
        .into_inner()
        .map_err(|error| error.into_error().into())
}

/// 表計算ソフトで開いたときに数式として実行されないよう(CSV インジェクション対策)、
/// `=` `+` `-` `@` タブ・改行で始まるセルの先頭に `'` を付ける。
/// 表示名は Discord の参加者が自由に決められるので、`=HYPERLINK(...)` のような名前もありうる。
fn neutralize_formula(cell: &str) -> std::borrow::Cow<'_, str> {
    if cell.starts_with(['=', '+', '-', '@', '\t', '\r']) {
        format!("'{cell}").into()
    } else {
        cell.into()
    }
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

        let proceedings = transcribe_session(dir.path(), &stt).unwrap();
        let summary: Vec<(&str, String, &str)> = proceedings
            .rows
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
        let rows = vec![Row {
            speaker: "山田".to_owned(),
            date: "2026-11-05T00:00:10.7Z".parse().unwrap(),
            talk: "カンマ, と \"引用符\" を含む".to_owned(),
        }];
        let csv = render_csv(&rows, &TimeZone::get("Asia/Tokyo").unwrap()).unwrap();
        assert_eq!(
            String::from_utf8(csv).unwrap(),
            "speaker,date,talk\n山田,2026-11-05T09:00:10+09:00,\"カンマ, と \"\"引用符\"\" を含む\"\n"
        );
    }

    #[test]
    fn csv_cells_cannot_become_formulas() {
        let rows = vec![Row {
            speaker: "=HYPERLINK(\"https://example.com\")".to_owned(),
            date: "2026-11-05T00:00:00Z".parse().unwrap(),
            talk: "-1 は負の数".to_owned(),
        }];
        let csv = render_csv(&rows, &TimeZone::UTC).unwrap();
        let mut reader = csv::Reader::from_reader(csv.as_slice());
        let record = reader.records().next().unwrap().unwrap();
        assert_eq!(&record[0], "'=HYPERLINK(\"https://example.com\")");
        assert_eq!(&record[2], "'-1 は負の数");
    }

    #[test]
    fn writes_csv_html_and_markdown() {
        let dir = tempfile::tempdir().unwrap();
        write_manifest(dir.path());
        let stt = FakeStt(HashMap::from([
            ("a.ogg".to_owned(), Ok(vec![segment(0.0, "始めます。")])),
            ("b.ogg".to_owned(), Ok(vec![])),
            ("c.ogg".to_owned(), Ok(vec![])),
        ]));
        let proceedings = transcribe_session(dir.path(), &stt).unwrap();
        assert_eq!(proceedings.session_id, "s");
        write_proceedings(&proceedings, dir.path(), &TimeZone::UTC).unwrap();

        for file in [CSV_FILE, HTML_FILE, MARKDOWN_FILE] {
            let text = std::fs::read_to_string(dir.path().join(file)).unwrap();
            assert!(text.contains("始めます。"), "{file}: {text}");
            assert!(!dir.path().join(format!("{file}.tmp")).exists());
        }
    }
}
