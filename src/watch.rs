//! DATA_DIR を見張り、録音が終わったセッションを順に文字起こしする。

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use jiff::tz::TimeZone;

use crate::manifest::MANIFEST_FILE;
use crate::proceedings::{OUTPUT_FILE, SpeechToText, transcribe_session, write_csv};

/// 文字起こしに失敗したセッションに置く。消すと次の巡回でやり直す。
pub const ERROR_FILE: &str = "transcription.error";

/// manifest.json があり、議事録もエラーの記録もまだ無いセッションを、古い順に返す。
pub fn pending_sessions(data_dir: &Path) -> Result<Vec<PathBuf>> {
    let sessions_root = data_dir.join("sessions");
    if !sessions_root.exists() {
        return Ok(Vec::new());
    }
    let mut pending = Vec::new();
    for guild in read_dirs(&sessions_root)? {
        for session in read_dirs(&guild)? {
            if session.join(MANIFEST_FILE).is_file()
                && !session.join(OUTPUT_FILE).exists()
                && !session.join(ERROR_FILE).exists()
            {
                pending.push(session);
            }
        }
    }
    // セッション ID は開始時刻から始まるので、名前の順 = 時刻の順。
    pending.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(pending)
}

fn read_dirs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("cannot read {}", dir.display()))? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            dirs.push(entry.path());
        }
    }
    Ok(dirs)
}

/// セッションを 1 つ文字起こしして議事録を書く。失敗したら ERROR_FILE に理由を書く。
pub fn process_session(session: &Path, stt: &dyn SpeechToText, timezone: &TimeZone) -> bool {
    log::info!("transcribing {}", session.display());
    let started = std::time::Instant::now();
    let result = transcribe_session(session, stt).and_then(|rows| {
        write_csv(&rows, &session.join(OUTPUT_FILE), timezone).map(|()| rows.len())
    });
    match result {
        Ok(count) => {
            log::info!(
                "wrote {count} line(s) to {} in {:.1}s",
                session.join(OUTPUT_FILE).display(),
                started.elapsed().as_secs_f32()
            );
            true
        }
        Err(error) => {
            log::error!("failed to transcribe {}: {error:#}", session.display());
            if let Err(write_error) =
                std::fs::write(session.join(ERROR_FILE), format!("{error:#}\n"))
            {
                log::error!("cannot record the failure: {write_error}");
            }
            false
        }
    }
}

pub fn run(data_dir: &Path, stt: &dyn SpeechToText, timezone: &TimeZone, interval: Duration) -> ! {
    log::info!(
        "watching {} every {}s",
        data_dir.join("sessions").display(),
        interval.as_secs()
    );
    loop {
        match pending_sessions(data_dir) {
            Ok(sessions) => {
                for session in sessions {
                    process_session(&session, stt, timezone);
                }
            }
            Err(error) => log::error!("cannot list sessions: {error:#}"),
        }
        std::thread::sleep(interval);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proceedings::tests::{FakeStt, segment, write_manifest};
    use std::collections::HashMap;

    fn session(root: &Path, guild: &str, id: &str) -> PathBuf {
        let dir = root.join("sessions").join(guild).join(id);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn lists_only_finished_and_unprocessed_sessions_in_time_order() {
        let root = tempfile::tempdir().unwrap();
        let recording = session(root.path(), "g1", "20261105T030000Z-00000003");
        let done = session(root.path(), "g1", "20261105T010000Z-00000001");
        let failed = session(root.path(), "g2", "20261105T015000Z-00000005");
        let later = session(root.path(), "g1", "20261105T020000Z-00000002");
        let earlier = session(root.path(), "g2", "20261105T005000Z-00000004");
        for dir in [&done, &failed, &later, &earlier] {
            std::fs::write(dir.join(MANIFEST_FILE), "{}").unwrap();
        }
        std::fs::write(done.join(OUTPUT_FILE), "").unwrap();
        std::fs::write(failed.join(ERROR_FILE), "").unwrap();
        std::fs::write(recording.join("x.ogg"), "").unwrap();

        assert_eq!(pending_sessions(root.path()).unwrap(), vec![earlier, later]);
    }

    #[test]
    fn missing_sessions_directory_means_nothing_to_do() {
        let root = tempfile::tempdir().unwrap();
        assert!(pending_sessions(root.path()).unwrap().is_empty());
    }

    #[test]
    fn writes_proceedings_or_records_the_failure() {
        let root = tempfile::tempdir().unwrap();
        let good = session(root.path(), "g", "1");
        write_manifest(&good);
        let bad = session(root.path(), "g", "2");
        std::fs::write(bad.join(MANIFEST_FILE), "not json").unwrap();

        let stt = FakeStt(HashMap::from([
            ("a.ogg".to_owned(), Ok(vec![segment(0.0, "こんにちは")])),
            ("b.ogg".to_owned(), Ok(vec![])),
            ("c.ogg".to_owned(), Ok(vec![])),
        ]));
        assert!(process_session(&good, &stt, &TimeZone::UTC));
        assert!(!process_session(&bad, &stt, &TimeZone::UTC));

        let csv = std::fs::read_to_string(good.join(OUTPUT_FILE)).unwrap();
        assert_eq!(
            csv,
            "speaker,date,talk\n山田,2026-11-05T00:00:10+00:00,こんにちは\n"
        );
        assert!(
            std::fs::read_to_string(bad.join(ERROR_FILE))
                .unwrap()
                .contains("invalid")
        );
        assert!(pending_sessions(root.path()).unwrap().is_empty());
    }
}
