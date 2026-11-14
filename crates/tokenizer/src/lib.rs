//! Turns text into the tokens a model counts in.
//!
//! A conversation's arithmetic is in tokens: how much of the context is used,
//! what may still be generated, what has to be forgotten. The kernel proves
//! that arithmetic, and it is only as true as the counts put to it. A count is
//! the model's own only if the text is split the way the model's tokeniser
//! splits it.
//!
//! So nothing here is invented. The vocabulary, the merges and the pattern
//! that splits text into pieces all come from the model's own
//! `tokenizer.json`, and a file this crate does not know how to follow is
//! refused by name rather than followed approximately. A tokeniser that was
//! nearly right would give counts that were nearly right, which is the one
//! thing a proof about those counts cannot survive.

#![forbid(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

mod bytes;
mod gguf;

/// Why a tokeniser could not be read, or could not be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokeniserError {
    /// `FL-20`
    NotJson,
    /// `FL-21`: the file describes a tokeniser this crate does not follow.
    /// What it describes is named, so a model that cannot be counted says why.
    Unfollowable(Unfollowable),
    /// `FL-22`
    UnknownId,
    /// `FL-23`
    TokenNotByteLevel,
    /// `FL-24`
    NotGguf,
    /// `FL-25`
    Truncated,
}

/// What about a tokeniser this crate does not follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unfollowable {
    /// The model is not byte-pair encoding: Unigram and WordPiece split text
    /// by other rules entirely.
    NotBytePair,
    /// The pre-tokeniser is not the byte-level one, so pieces would not be the
    /// model's pieces.
    NotByteLevel,
    /// The file carries a pattern this crate's regular expressions cannot run.
    PatternNotRunnable,
    /// A merge is not two pieces, so the rank it carries cannot be used.
    MergeNotAPair,
    /// The vocabulary is missing, empty, or not text to numbers.
    NoVocabulary,
    /// A name the file carries is not text.
    NameNotText,
    /// The file is of a version this crate does not read.
    VersionNotKnown,
    /// A metadata value is of a type the format does not define, so its end
    /// cannot be found.
    ValueTypeNotKnown,
    /// The file names its pre-tokeniser, and this crate has no cited pattern
    /// for that name. The name is all the file carries; the pattern it stands
    /// for lives in the reader, so guessing it would be inventing one.
    PreTokeniserNotKnown,
}

/// The pattern a `gpt2` pre-tokeniser splits by, as `llama.cpp` has it. It is
/// the published byte-level pattern without the final `\s+` alternative.
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// The pattern a `qwen2` pre-tokeniser splits by, as `llama.cpp` has it:
/// case-insensitive contractions, single digits, and runs of line endings kept
/// with what they follow.
const QWEN2_PATTERN: &str = concat!(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
    r"|[^
\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[
]*|\s*[
]+|\s+(?!\S)|\s+",
);

/// The rank of each merge: the lower the rank, the earlier it applies. A pair
/// written twice keeps the rank it was first given, which is the one the file
/// means.
///
/// Exactly two pieces. A space inside a piece is not possible in a byte-level
/// vocabulary, where a space is written as a character of the alphabet, so a
/// merge with two spaces in it is a merge the file means something else by.
fn ranked(merged: Vec<String>) -> Result<BTreeMap<(String, String), u32>, TokeniserError> {
    let mut ranks = BTreeMap::new();
    for (rank, merge) in merged.into_iter().enumerate() {
        let rank = u32::try_from(rank)
            .map_err(|_| TokeniserError::Unfollowable(Unfollowable::MergeNotAPair))?;
        let pair = match merge.split(' ').collect::<Vec<_>>()[..] {
            [left, right] => (left.to_string(), right.to_string()),
            _ => return Err(TokeniserError::Unfollowable(Unfollowable::MergeNotAPair)),
        };
        ranks.entry(pair).or_insert(rank);
    }
    Ok(ranks)
}

/// The pattern a pre-tokeniser's name stands for, for the names this crate has
/// a cited pattern for. A file carries the name and not the pattern, so a name
/// with no cited pattern is refused rather than guessed at.
///
/// The names and their patterns are `llama.cpp`'s, cited in
/// `model/authorities.toml`; the aliases are the ones it lists beside each.
fn pattern_named(name: Option<&str>) -> Result<fancy_regex::Regex, TokeniserError> {
    let pattern = match name {
        None | Some("default" | "gpt-2" | "gpt2" | "olmo" | "jais" | "mpt") => GPT2_PATTERN,
        Some("qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "stablelm2" | "hunyuan") => {
            QWEN2_PATTERN
        }
        Some(_) => {
            return Err(TokeniserError::Unfollowable(
                Unfollowable::PreTokeniserNotKnown,
            ))
        }
    };
    fancy_regex::Regex::new(pattern)
        .map_err(|_| TokeniserError::Unfollowable(Unfollowable::PatternNotRunnable))
}

/// The pattern the byte-level pre-tokeniser splits by when it is asked to.
/// It is the authority's own, written here because the files that use it do
/// not carry it: they say `use_regex`, and this is what that means.
const BYTE_LEVEL_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// A model's tokeniser: what text is split into, and what each piece is called.
#[derive(Clone, Debug)]
pub struct Tokeniser {
    ids: BTreeMap<String, u32>,
    tokens: BTreeMap<u32, String>,
    /// The rank of each merge: the lower the rank, the earlier it applies.
    ranks: BTreeMap<(String, String), u32>,
    /// The pattern the file carries for splitting text into pieces, if it
    /// carries one. Without it the whole text is one piece, which is what the
    /// published byte-level pre-tokeniser does on its own.
    pattern: Option<fancy_regex::Regex>,
    /// Whether a space is put before text that does not start with one, which
    /// is what makes the first word look like every other word to the model.
    prefix_space: bool,
    alphabet: [char; 256],
}

impl Tokeniser {
    /// Read a model's `tokenizer.json`.
    pub fn read(file: &[u8]) -> Result<Self, TokeniserError> {
        let read: serde_json::Value =
            serde_json::from_slice(file).map_err(|_| TokeniserError::NotJson)?;
        if !read.is_object() {
            return Err(TokeniserError::NotJson);
        }
        let model = &read["model"];
        if model["type"].as_str() != Some("BPE") {
            return Err(TokeniserError::Unfollowable(Unfollowable::NotBytePair));
        }

        let mut ids = BTreeMap::new();
        let mut tokens = BTreeMap::new();
        let vocabulary = model["vocab"]
            .as_object()
            .ok_or(TokeniserError::Unfollowable(Unfollowable::NoVocabulary))?;
        for (token, id) in vocabulary {
            let id = u32::try_from(
                id.as_u64()
                    .ok_or(TokeniserError::Unfollowable(Unfollowable::NoVocabulary))?,
            )
            .map_err(|_| TokeniserError::Unfollowable(Unfollowable::NoVocabulary))?;
            ids.insert(token.clone(), id);
            tokens.insert(id, token.clone());
        }
        if ids.is_empty() {
            return Err(TokeniserError::Unfollowable(Unfollowable::NoVocabulary));
        }
        // Tokens the file adds beside the vocabulary, such as the ones that
        // mark the ends of a turn. They are ids like any other to a counter.
        for added in read["added_tokens"].as_array().into_iter().flatten() {
            let (Some(content), Some(id)) = (added["content"].as_str(), added["id"].as_u64())
            else {
                continue;
            };
            if let Ok(id) = u32::try_from(id) {
                ids.insert(content.to_string(), id);
                tokens.insert(id, content.to_string());
            }
        }

        // Written either as one string of two pieces, or as the two. Both
        // become the same list, and the ranks are taken from it in one place.
        let mut merged = Vec::new();
        for merge in model["merges"].as_array().into_iter().flatten() {
            merged.push(match merge {
                serde_json::Value::String(text) => text.clone(),
                serde_json::Value::Array(two) => match two.as_slice() {
                    [left, right] => match (left.as_str(), right.as_str()) {
                        (Some(left), Some(right)) => format!("{left} {right}"),
                        _ => return Err(TokeniserError::Unfollowable(Unfollowable::MergeNotAPair)),
                    },
                    _ => return Err(TokeniserError::Unfollowable(Unfollowable::MergeNotAPair)),
                },
                _ => return Err(TokeniserError::Unfollowable(Unfollowable::MergeNotAPair)),
            });
        }
        let ranks = ranked(merged)?;

        let described = pre_tokeniser(&read["pre_tokenizer"])?;
        if !described.byte_level {
            return Err(TokeniserError::Unfollowable(Unfollowable::NotByteLevel));
        }

        Ok(Self {
            ids,
            tokens,
            ranks,
            pattern: described.pattern,
            prefix_space: described.prefix_space,
            alphabet: bytes::alphabet(),
        })
    }

    /// Build a tokeniser from a vocabulary in id order, merges in rank order,
    /// and how the text is split before any of it.
    ///
    /// Both readers end here: what a tokeniser is does not depend on which
    /// kind of file it was read from.
    fn assemble(
        tokens: Vec<String>,
        merged: Vec<String>,
        pattern: Option<fancy_regex::Regex>,
        prefix_space: bool,
    ) -> Result<Self, TokeniserError> {
        let mut ids = BTreeMap::new();
        let mut by_id = BTreeMap::new();
        for (id, token) in tokens.into_iter().enumerate() {
            let id = u32::try_from(id)
                .map_err(|_| TokeniserError::Unfollowable(Unfollowable::NoVocabulary))?;
            ids.insert(token.clone(), id);
            by_id.insert(id, token);
        }
        let ranks = ranked(merged)?;
        Ok(Self {
            ids,
            tokens: by_id,
            ranks,
            pattern,
            prefix_space,
            alphabet: bytes::alphabet(),
        })
    }

    /// Read a model's GGUF file, or as much of it as holds the header.
    ///
    /// A model of gigabytes carries its vocabulary in the first part of its
    /// own file; nothing past the metadata is read.
    pub fn from_gguf(file: &[u8]) -> Result<Self, TokeniserError> {
        gguf::read(file)
    }

    /// How many tokens this vocabulary holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Is the vocabulary empty? It never is: a file with no vocabulary is
    /// refused.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The ids of `text`.
    ///
    /// The text is split into pieces by the model's own pattern, each piece is
    /// written in the byte alphabet, and the merges are applied to it in rank
    /// order until none applies. Every byte has a character and every
    /// character of the alphabet is in a byte-level vocabulary, so there is no
    /// text this cannot encode and no unknown token to stand in.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        // A space before the first word, when the file asks for one, so that
        // the first word is the same token as the same word anywhere else.
        let spaced = if self.prefix_space && !text.starts_with(' ') && !text.is_empty() {
            format!(" {text}")
        } else {
            text.to_string()
        };
        for piece in self.pieces(&spaced) {
            let written: String = piece
                .bytes()
                .map(|byte| self.alphabet[usize::from(byte)])
                .collect();
            for token in self.merged(&written) {
                // A character of the alphabet that the vocabulary does not
                // name cannot be written as any id, so it is left out rather
                // than replaced by an id that means something else.
                if let Some(id) = self.ids.get(&token) {
                    out.push(*id);
                }
            }
        }
        out
    }

    /// How many tokens `text` is.
    #[must_use]
    pub fn count(&self, text: &str) -> u64 {
        self.encode(text).len() as u64
    }

    /// The bytes `ids` stand for.
    pub fn decode(&self, ids: &[u32]) -> Result<Vec<u8>, TokeniserError> {
        let mut out = Vec::new();
        for id in ids {
            let token = self.tokens.get(id).ok_or(TokeniserError::UnknownId)?;
            for character in token.chars() {
                let byte = self
                    .alphabet
                    .iter()
                    .position(|written| *written == character)
                    .ok_or(TokeniserError::TokenNotByteLevel)?;
                out.push(byte as u8);
            }
        }
        Ok(out)
    }

    /// The pieces the model's pattern splits `text` into, or the whole of it.
    fn pieces<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let Some(pattern) = &self.pattern else {
            return vec![text];
        };
        let mut out = Vec::new();
        let mut at = 0;
        for found in pattern.find_iter(text).flatten() {
            // Anything the pattern steps over is still text the model sees, so
            // it is kept as its own piece rather than dropped.
            if found.start() > at {
                out.push(&text[at..found.start()]);
            }
            out.push(found.as_str());
            at = found.end();
        }
        if at < text.len() {
            out.push(&text[at..]);
        }
        out.retain(|piece| !piece.is_empty());
        out
    }

    /// One piece, merged until no merge applies.
    fn merged(&self, written: &str) -> Vec<String> {
        let mut parts: Vec<String> = written.chars().map(String::from).collect();
        loop {
            // The merge with the lowest rank anywhere in the piece goes first:
            // the ranks are an order over the whole vocabulary, not a
            // left-to-right sweep.
            let mut best: Option<(u32, usize)> = None;
            for (at, pair) in parts.windows(2).enumerate() {
                let [left, right] = pair else { continue };
                if let Some(rank) = self.ranks.get(&(left.clone(), right.clone())) {
                    if best.is_none_or(|(lowest, _)| *rank < lowest) {
                        best = Some((*rank, at));
                    }
                }
            }
            let Some((_, at)) = best else { return parts };
            let joined = format!("{}{}", parts[at], parts[at + 1]);
            parts.splice(at..at + 2, [joined]);
        }
    }
}

/// What a file's pre-tokeniser says: whether the text ends up in the byte
/// alphabet, what it is split by, and whether a space is put in front.
struct PreTokeniser {
    byte_level: bool,
    pattern: Option<fancy_regex::Regex>,
    prefix_space: bool,
}

/// Read a pre-tokeniser. A sequence of them is read through: what matters is
/// that the text ends up in the byte alphabet and is split the file's way.
fn pre_tokeniser(described: &serde_json::Value) -> Result<PreTokeniser, TokeniserError> {
    match described["type"].as_str() {
        Some("ByteLevel") => {
            // The authority's defaults, for a file that leaves them out: it
            // splits by its own pattern and puts a space in front.
            let flag = |name: &str| described[name].as_bool().unwrap_or(true);
            let pattern =
                if flag("use_regex") {
                    Some(fancy_regex::Regex::new(BYTE_LEVEL_PATTERN).map_err(|_| {
                        TokeniserError::Unfollowable(Unfollowable::PatternNotRunnable)
                    })?)
                } else {
                    None
                };
            Ok(PreTokeniser {
                byte_level: true,
                pattern,
                prefix_space: flag("add_prefix_space"),
            })
        }
        Some("Split") => {
            let pattern = described["pattern"]["Regex"]
                .as_str()
                .ok_or(TokeniserError::Unfollowable(Unfollowable::NotByteLevel))?;
            let compiled = fancy_regex::Regex::new(pattern)
                .map_err(|_| TokeniserError::Unfollowable(Unfollowable::PatternNotRunnable))?;
            // A split on its own is not byte level; it is byte level when a
            // ByteLevel stands beside it in the sequence.
            Ok(PreTokeniser {
                byte_level: false,
                pattern: Some(compiled),
                prefix_space: false,
            })
        }
        Some("Sequence") => {
            let mut out = PreTokeniser {
                byte_level: false,
                pattern: None,
                prefix_space: false,
            };
            for part in described["pretokenizers"].as_array().into_iter().flatten() {
                let part = pre_tokeniser(part)?;
                out.byte_level |= part.byte_level;
                out.prefix_space |= part.prefix_space;
                // The first pattern in the sequence is the one that splits;
                // what follows it reads pieces, not the whole text.
                out.pattern = out.pattern.or(part.pattern);
            }
            Ok(out)
        }
        _ => Err(TokeniserError::Unfollowable(Unfollowable::NotByteLevel)),
    }
}
