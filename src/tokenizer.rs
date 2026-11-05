//! Whisper のトークナイザー(tokenizer.json)と特殊トークン。
//!
//! faster-whisper の `Tokenizer`(MIT License, Copyright (c) 2023 SYSTRAN)を移植したもの。

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, anyhow};

pub struct WhisperTokenizer {
    inner: tokenizers::Tokenizer,
    pub sot: u32,
    pub eot: u32,
    pub sot_prev: u32,
    pub sot_lm: u32,
    pub transcribe: u32,
    pub translate: u32,
    pub no_timestamps: u32,
    pub no_speech: Option<u32>,
}

impl WhisperTokenizer {
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

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// CTranslate2 にプロンプトを渡すときに使う、ID に対応するトークン文字列。
    pub fn id_to_token(&self, id: u32) -> Result<String> {
        self.inner
            .id_to_token(id)
            .with_context(|| format!("unknown token id: {id}"))
    }

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
