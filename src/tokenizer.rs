//! Whisper のトークナイザー(tokenizer.json)と特殊トークン。
//!
//! faster-whisper の `Tokenizer`(MIT License, Copyright (c) 2023 SYSTRAN)を移植したもの。

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, anyhow};

/// Whisper のトークナイザーと、デコードに使う特殊トークンの ID。
pub struct WhisperTokenizer {
    inner: tokenizers::Tokenizer,
    /// `<|startoftranscript|>`
    pub sot: u32,
    /// `<|endoftext|>`
    pub eot: u32,
    /// `<|startofprev|>`。前の区間の文字起こしをプロンプトとして渡すときの先頭。
    pub sot_prev: u32,
    /// `<|startoflm|>`
    pub sot_lm: u32,
    /// `<|transcribe|>`
    pub transcribe: u32,
    /// `<|translate|>`
    pub translate: u32,
    /// `<|notimestamps|>`
    pub no_timestamps: u32,
    /// `<|nospeech|>`(古いモデルでは `<|nocaptions|>`)。無音の判定に使う。
    pub no_speech: Option<u32>,
}

impl WhisperTokenizer {
    /// tokenizer.json を読み、特殊トークンの ID を引いておく。
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|error| anyhow!("cannot load {}: {error}", path.display()))?;
        let id = |token: &str| {
            inner
                .token_to_id(token)
                .with_context(|| format!("{} has no {token} token", path.display()))
        };
        Ok(Self {
            sot: id("<|startoftranscript|>")?,
            eot: id("<|endoftext|>")?,
            sot_prev: id("<|startofprev|>")?,
            sot_lm: id("<|startoflm|>")?,
            transcribe: id("<|transcribe|>")?,
            translate: id("<|translate|>")?,
            no_timestamps: id("<|notimestamps|>")?,
            no_speech: inner
                .token_to_id("<|nospeech|>")
                .or_else(|| inner.token_to_id("<|nocaptions|>")),
            inner,
        })
    }

    /// これ以上の ID はタイムスタンプ(`<|0.00|>`, `<|0.02|>`, ...)。
    pub fn timestamp_begin(&self) -> u32 {
        self.no_timestamps + 1
    }

    /// トークン文字列の ID。
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// CTranslate2 にプロンプトを渡すときに使う、ID に対応するトークン文字列。
    pub fn id_to_token(&self, id: u32) -> Result<String> {
        self.inner
            .id_to_token(id)
            .with_context(|| format!("unknown token id: {id}"))
    }

    /// 文字列をトークン ID の列にする(特殊トークンは足さない)。
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| anyhow!("cannot encode {text:?}: {error}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// 本文のトークン(`<|endoftext|>` より前の ID)だけを文字列に戻す。
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let text: Vec<u32> = tokens.iter().copied().filter(|&t| t < self.eot).collect();
        self.inner
            .decode(&text, true)
            .map_err(|error| anyhow!("cannot decode tokens: {error}"))
    }

    /// 話していない内容(話者名の括弧書きや ♪ など)を出させないために抑えるトークン。
    pub fn non_speech_tokens(&self) -> Result<BTreeSet<u32>> {
        let mut symbols: Vec<String> = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』"
            .chars()
            .map(String::from)
            .collect();
        symbols.extend(
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪"
                .split_whitespace()
                .map(String::from),
        );
        let miscellaneous: Vec<String> = "♩♪♫♬♭♮♯".chars().map(String::from).collect();

        let mut result = BTreeSet::new();
        for text in [" -", " '"] {
            if let Some(&first) = self.encode(text)?.first() {
                result.insert(first);
            }
        }
        for symbol in symbols.iter().chain(&miscellaneous) {
            let is_misc = miscellaneous.contains(symbol);
            for tokens in [self.encode(symbol)?, self.encode(&format!(" {symbol}"))?] {
                if tokens.len() == 1 || (is_misc && !tokens.is_empty()) {
                    result.insert(tokens[0]);
                }
            }
        }
        Ok(result)
    }

    /// faster-whisper の `suppress_tokens=[-1]` のときに CTranslate2 に渡す ID の一覧。
    pub fn suppressed_tokens(&self) -> Result<Vec<i32>> {
        let mut tokens = self.non_speech_tokens()?;
        tokens.extend([
            self.transcribe,
            self.translate,
            self.sot,
            self.sot_prev,
            self.sot_lm,
        ]);
        tokens.extend(self.no_speech);
        Ok(tokens.into_iter().map(|t| t as i32).collect())
    }
}
