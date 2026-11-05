//! bot が書く manifest.json の読み込み。
//!
//! ディレクトリ構成(bot 側の src/manifest.ts と対になる):
//!
//! ```text
//! <DATA_DIR>/sessions/<guild_id>/<session_id>/manifest.json
//! <DATA_DIR>/sessions/<guild_id>/<session_id>/<utterance_id>.ogg
//! ```
//!
//! manifest.json は bot が録音をすべて閉じてから最後に書くので、これがあるディレクトリは
//! 録音が終わったセッションとして扱ってよい。

use std::path::Path;

use anyhow::{Context, Result, bail};
use jiff::Timestamp;
use serde::Deserialize;

pub const MANIFEST_FILE: &str = "manifest.json";
pub const SUPPORTED_VERSION: u32 = 1;

// bot との約束を丸ごと検証する(deny_unknown_fields)ため、今は読まないフィールドも持っておく。
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionManifest {
    pub version: u32,
    pub session_id: String,
    pub guild_id: String,
    pub channel_id: String,
    pub started_at: Timestamp,
    pub ended_at: Timestamp,
    pub utterances: Vec<Utterance>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Utterance {
    pub id: String,
    pub user_id: String,
    pub speaker: String,
    pub started_at: Timestamp,
    pub ended_at: Timestamp,
    /// セッションディレクトリ直下のファイル名。
    pub file: String,
}

impl SessionManifest {
    pub fn load(session_dir: &Path) -> Result<Self> {
        let path = session_dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("invalid {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_str(text)?;
        if manifest.version != SUPPORTED_VERSION {
            bail!("unsupported manifest version: {}", manifest.version);
        }
        for utterance in &manifest.utterances {
            ensure_plain_file_name(&utterance.file)?;
        }
        Ok(manifest)
    }
}

/// セッションディレクトリの外を指せないよう、ただのファイル名だけを受け付ける。
fn ensure_plain_file_name(name: &str) -> Result<()> {
    let is_plain =
        !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0']);
    if !is_plain {
        bail!("file must be a plain file name: {name:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_with_file(file: &str) -> String {
        format!(
            r#"{{
                "version": 1,
                "session_id": "20261105T080102Z-deadbeef",
                "guild_id": "1",
                "channel_id": "2",
                "started_at": "2026-11-05T08:01:02.345Z",
                "ended_at": "2026-11-05T09:00:00.000Z",
                "utterances": [{{
                    "id": "u1",
                    "user_id": "3",
                    "speaker": "山田",
                    "started_at": "2026-11-05T08:10:00.000Z",
                    "ended_at": "2026-11-05T08:10:05.000Z",
                    "file": {file:?}
                }}]
            }}"#
        )
    }

    #[test]
    fn parses_the_bot_manifest() {
        let manifest = SessionManifest::parse(&manifest_with_file("u1.ogg")).unwrap();
        assert_eq!(manifest.session_id, "20261105T080102Z-deadbeef");
        assert_eq!(manifest.utterances.len(), 1);
        let utterance = &manifest.utterances[0];
        assert_eq!(utterance.speaker, "山田");
        assert_eq!(utterance.file, "u1.ogg");
        assert_eq!(
            utterance.started_at,
            "2026-11-05T08:10:00Z".parse::<Timestamp>().unwrap()
        );
    }

    #[test]
    fn rejects_paths_outside_the_session() {
        for file in [
            "../x.ogg",
            "/etc/passwd",
            "a/b.ogg",
            "a\\b.ogg",
            "..",
            ".",
            "",
        ] {
            assert!(
                SessionManifest::parse(&manifest_with_file(file)).is_err(),
                "{file:?}"
            );
        }
    }

    #[test]
    fn rejects_other_versions() {
        let text = manifest_with_file("u1.ogg").replace("\"version\": 1", "\"version\": 2");
        let error = SessionManifest::parse(&text).unwrap_err();
        assert!(error.to_string().contains("unsupported manifest version"));
    }

    #[test]
    fn rejects_timestamps_without_offset() {
        let text = manifest_with_file("u1.ogg").replace("08:10:00.000Z", "08:10:00.000");
        assert!(SessionManifest::parse(&text).is_err());
    }
}
