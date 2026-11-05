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

/// bot が録音を終えたときに最後に書く manifest のファイル名。
pub const MANIFEST_FILE: &str = "manifest.json";
/// 読める manifest の版。
pub const SUPPORTED_VERSION: u32 = 1;

/// bot が書く manifest.json。1 回の録音セッションを表す。
// bot との約束にある必須のフィールドがそろっているかを検証するため、今は読まないフィールドも持っておく。
// 知らないフィールド(text_channel_id など bot だけが使うもの)は無視する。
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct SessionManifest {
    /// manifest の版。[`SUPPORTED_VERSION`] だけを受け付ける。
    pub version: u32,
    /// セッション ID(例: `20261105T080102Z-1a2b3c4d`)。
    pub session_id: String,
    /// サーバーの ID。
    pub guild_id: String,
    /// 録音したボイスチャンネルの ID。
    pub channel_id: String,
    /// 録音時点のサーバー名。古い bot の manifest には無い。
    pub guild_name: Option<String>,
    /// 録音したボイスチャンネルの名前。古い bot の manifest には無い。
    pub channel_name: Option<String>,
    /// 録音を始めた時刻。
    pub started_at: Timestamp,
    /// 録音を終えた時刻。
    pub ended_at: Timestamp,
    /// 発言。
    pub utterances: Vec<Utterance>,
}

/// 発言 1 つ(無音で区切られたひと続きの音声)。
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct Utterance {
    /// 発言の ID。
    pub id: String,
    /// 話した人の Discord のユーザー ID。
    pub user_id: String,
    /// 録音時点のサーバー内の表示名。
    pub speaker: String,
    /// 話し始めた時刻。
    pub started_at: Timestamp,
    /// 話し終えた時刻。
    pub ended_at: Timestamp,
    /// セッションディレクトリ直下のファイル名。
    pub file: String,
}

impl SessionManifest {
    /// セッションのディレクトリから manifest.json を読む。
    pub fn load(session_dir: &Path) -> Result<Self> {
        let path = session_dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("invalid {}", path.display()))
    }

    /// manifest.json の中身を読み、版とファイル名を検証する。
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
    fn reads_the_names_when_the_bot_writes_them() {
        let manifest = SessionManifest::parse(&manifest_with_file("u1.ogg")).unwrap();
        assert_eq!(manifest.guild_name, None);
        assert_eq!(manifest.channel_name, None);

        let text = manifest_with_file("u1.ogg").replace(
            "\"channel_id\": \"2\",",
            "\"channel_id\": \"2\", \"guild_name\": \"開発部\", \"channel_name\": \"定例\",",
        );
        let manifest = SessionManifest::parse(&text).unwrap();
        assert_eq!(manifest.guild_name.as_deref(), Some("開発部"));
        assert_eq!(manifest.channel_name.as_deref(), Some("定例"));
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
    fn ignores_fields_it_does_not_use() {
        let text = manifest_with_file("u1.ogg").replace(
            "\"channel_id\": \"2\",",
            "\"channel_id\": \"2\", \"text_channel_id\": \"9\", \"future\": [1, 2],",
        );
        assert!(text.contains("text_channel_id"));
        assert!(SessionManifest::parse(&text).is_ok());
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
