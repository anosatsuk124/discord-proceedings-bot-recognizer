//! CTranslate2 形式の Whisper モデルの場所を決める。
//!
//! WHISPER_MODEL がディレクトリならそれを使い、そうでなければ Hugging Face のリポジトリ ID とみなして
//! HF_HOME のキャッシュ(既定は ~/.cache/huggingface)に取ってくる。必要なファイルがキャッシュに
//! そろっていればネットワークには出ない。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo, RepoType};

/// faster-whisper の `download_model` と同じく、推論に要るファイルだけを、この順に取ってくる。
///
/// [`LAST_FILE`] を必ず最後に取るので、キャッシュに [`LAST_FILE`] があれば、それより前の
/// ファイル(リポジトリにあれば preprocessor_config.json も)は取り終えている。
/// 途中で止まったダウンロードを、ファイルの欠けたモデルとして使ってしまわないための順番。
const MODEL_FILES: &[&str] = &[
    "preprocessor_config.json",
    "vocabulary.json",
    "vocabulary.txt",
    "config.json",
    "model.bin",
    LAST_FILE,
];
const LAST_FILE: &str = "tokenizer.json";
/// 非公開のモデルを取ってくるときの Hugging Face のトークンを指定する環境変数。
const TOKEN_VARIABLE: &str = "HF_TOKEN";
const REQUIRED_FILES: &[&str] = &["config.json", "model.bin", LAST_FILE];

/// モデルのディレクトリを返す。`model` がディレクトリならそのまま使い、そうでなければ
/// Hugging Face のリポジトリ ID とみなして、キャッシュに無いファイルを取ってくる。
pub fn resolve(model: &str, revision: Option<&str>) -> Result<PathBuf> {
    let path = Path::new(model);
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    let repo = Repo::with_revision(
        model.to_owned(),
        RepoType::Model,
        revision.unwrap_or("main").to_owned(),
    );
    if let Some(dir) = cached(&Cache::from_env(), &repo) {
        log::info!("using the cached model in {}", dir.display());
        return Ok(dir);
    }
    download(repo).with_context(|| format!("cannot download the model {model}"))
}

/// 必要なファイルがすべてキャッシュにあれば、そのディレクトリ(snapshots/<commit>)を返す。
fn cached(cache: &Cache, repo: &Repo) -> Option<PathBuf> {
    let dir = if is_commit_hash(repo.revision()) {
        // コミットで固定しているときは refs を介さずに snapshots/<commit> を見る。
        // Python の huggingface_hub(faster-whisper)が作ったキャッシュには refs/<commit> が無いため。
        cache
            .path()
            .join(repo.folder_name())
            .join("snapshots")
            .join(repo.revision())
    } else {
        cache
            .repo(repo.clone())
            .get("config.json")?
            .parent()?
            .to_path_buf()
    };
    // snapshots のファイルは blobs へのシンボリックリンクなので、リンク先まで確かめる(is_file はたどる)。
    let complete = REQUIRED_FILES.iter().all(|file| dir.join(file).is_file())
        && ["vocabulary.json", "vocabulary.txt"]
            .iter()
            .any(|file| dir.join(file).is_file());
    complete.then_some(dir)
}

fn is_commit_hash(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

fn download(repo: Repo) -> Result<PathBuf> {
    log::info!("downloading {} (revision {})", repo.url(), repo.revision());
    let token = std::env::var(TOKEN_VARIABLE).ok().filter(|t| !t.is_empty());
    let mut builder = ApiBuilder::from_env().with_progress(false);
    if token.is_some() {
        builder = builder.with_token(token);
    }
    let api = builder.build()?.repo(repo);
    let info = api.info()?;

    let mut dir = None;
    for &file in MODEL_FILES {
        if !info
            .siblings
            .iter()
            .any(|sibling| sibling.rfilename == file)
        {
            continue;
        }
        let path = api.get(file)?;
        log::info!("fetched {file}");
        dir.get_or_insert_with(|| path.parent().map(Path::to_path_buf));
    }
    let dir = dir.flatten().context("the repository has no model files")?;
    for file in REQUIRED_FILES {
        if !dir.join(file).exists() {
            bail!("the repository has no {file}; is it a CTranslate2 Whisper model?");
        }
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc";
    const FILES: &[&str] = &[
        "config.json",
        "model.bin",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocabulary.json",
    ];

    fn snapshot(root: &Path, files: &[&str]) -> PathBuf {
        let dir = root
            .join("models--kotoba-tech--kotoba-whisper-v2.0-faster")
            .join("snapshots")
            .join(COMMIT);
        std::fs::create_dir_all(&dir).unwrap();
        for file in files {
            std::fs::write(dir.join(file), "").unwrap();
        }
        dir
    }

    fn repo(revision: &str) -> Repo {
        Repo::with_revision(
            "kotoba-tech/kotoba-whisper-v2.0-faster".to_owned(),
            RepoType::Model,
            revision.to_owned(),
        )
    }

    #[test]
    fn finds_a_pinned_commit_without_refs() {
        // Python の huggingface_hub が作るキャッシュと同じく refs/<commit> が無い。
        let root = tempfile::tempdir().unwrap();
        let dir = snapshot(root.path(), FILES);
        let cache = Cache::new(root.path().to_path_buf());
        assert_eq!(cached(&cache, &repo(COMMIT)), Some(dir));
    }

    #[test]
    fn follows_refs_for_branch_names() {
        let root = tempfile::tempdir().unwrap();
        let dir = snapshot(root.path(), FILES);
        let refs = root
            .path()
            .join("models--kotoba-tech--kotoba-whisper-v2.0-faster/refs");
        std::fs::create_dir_all(&refs).unwrap();
        std::fs::write(refs.join("main"), COMMIT).unwrap();
        let cache = Cache::new(root.path().to_path_buf());
        assert_eq!(cached(&cache, &repo("main")), Some(dir));
    }

    #[test]
    fn incomplete_snapshots_are_not_used() {
        for missing in ["vocabulary.json", LAST_FILE] {
            let root = tempfile::tempdir().unwrap();
            let files: Vec<&str> = FILES.iter().copied().filter(|f| *f != missing).collect();
            snapshot(root.path(), &files);
            let cache = Cache::new(root.path().to_path_buf());
            assert_eq!(cached(&cache, &repo(COMMIT)), None, "without {missing}");
        }
    }

    #[test]
    fn models_without_a_preprocessor_config_are_cached_too() {
        // Systran/faster-whisper-tiny などは preprocessor_config.json を持たない。
        let root = tempfile::tempdir().unwrap();
        let files: Vec<&str> = FILES
            .iter()
            .copied()
            .filter(|f| *f != "preprocessor_config.json")
            .collect();
        let dir = snapshot(root.path(), &files);
        let cache = Cache::new(root.path().to_path_buf());
        assert_eq!(cached(&cache, &repo(COMMIT)), Some(dir));
    }

    #[test]
    fn the_tokenizer_is_fetched_last() {
        assert_eq!(MODEL_FILES.last(), Some(&LAST_FILE));
        for file in REQUIRED_FILES.iter().chain(&["preprocessor_config.json"]) {
            assert!(MODEL_FILES.contains(file), "{file}");
        }
    }

    #[test]
    fn commit_hashes() {
        assert!(is_commit_hash(COMMIT));
        assert!(!is_commit_hash("main"));
        assert!(!is_commit_hash(&COMMIT[..39]));
    }
}
