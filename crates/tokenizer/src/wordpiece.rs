//! WordPiece: how BERT and the encoders built on it split text.
//!
//! An encoder's input is the ids its own `tokenizer.json` gives, and nothing
//! here is invented: the vocabulary, the unknown token, the subword prefix,
//! the longest word, what the normaliser does and the template that wraps a
//! sentence are all the file's. A file that asks for a step this crate does
//! not follow is refused by what it asks for.
//!
//! The steps are those of the authority, Hugging Face `tokenizers`
//! (`normalizers/bert.rs`, `pre_tokenizers/bert.rs`, `models/wordpiece`),
//! written again here and checked against it. Which characters are controls,
//! punctuation or nonspacing marks, and how a character decomposes, are read
//! from the same Unicode tables the authority reads, `unicode_categories` and
//! `unicode-normalization-alignments`, so that the two agree on every
//! character and not only on the common ones.

use std::collections::BTreeMap;

use unicode_categories::UnicodeCategories;
use unicode_normalization_alignments::UnicodeNormalization;

use crate::{TokeniserError, Unfollowable};

/// A WordPiece tokeniser read from a model's `tokenizer.json`.
#[derive(Clone, Debug)]
pub struct WordPiece {
    ids: BTreeMap<String, u32>,
    unknown: u32,
    prefix: String,
    longest_word: usize,
    clean_text: bool,
    chinese: bool,
    strip_accents: bool,
    lowercase: bool,
    /// The ids the template puts before and after a sentence: `[CLS]` and
    /// `[SEP]` in BERT's files.
    first: u32,
    last: u32,
    /// The most ids a sentence may be, its two wrapping ids included, when
    /// the file truncates.
    truncate_to: Option<usize>,
    /// Tokens the file adds, matched in the text as written, longest first.
    added: Vec<(String, u32)>,
}

const fn unfollowable(why: Unfollowable) -> TokeniserError {
    TokeniserError::Unfollowable(why)
}

/// Text the file gives as a string, or the reason named.
fn text(value: &serde_json::Value, why: Unfollowable) -> Result<&str, TokeniserError> {
    value.as_str().ok_or(unfollowable(why))
}

/// A control character, as the authority's BERT normaliser decides it: one
/// of the Unicode categories Cc, Cf or Co, except the tab and line endings,
/// which are whitespace.
fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && c.is_other()
}

/// Whitespace, as the authority's BERT normaliser decides it.
fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

/// The ideographs the BERT normaliser sets apart with spaces: the CJK Unified
/// Ideographs and their extensions A to E, and the compatibility ideographs.
const fn is_chinese(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x2_0000..=0x2_A6DF
            | 0x2_A700..=0x2_B73F
            | 0x2_B740..=0x2_B81F
            | 0x2_B920..=0x2_CEAF
            | 0xF900..=0xFAFF
            | 0x2_F800..=0x2_FA1F
    )
}

/// Punctuation, which the BERT pre-tokeniser keeps as a piece of its own.
fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || c.is_punctuation()
}

impl WordPiece {
    /// Read a model's `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// [`TokeniserError::NotJson`] for a file that is not a JSON object, and
    /// [`TokeniserError::Unfollowable`] naming the first step this crate does
    /// not follow.
    pub fn read(file: &[u8]) -> Result<Self, TokeniserError> {
        let read: serde_json::Value =
            serde_json::from_slice(file).map_err(|_| TokeniserError::NotJson)?;
        if !read.is_object() {
            return Err(TokeniserError::NotJson);
        }
        let model = &read["model"];
        if model["type"].as_str() != Some("WordPiece") {
            return Err(unfollowable(Unfollowable::NotWordPiece));
        }
        let mut ids = BTreeMap::new();
        for (token, id) in model["vocab"]
            .as_object()
            .ok_or(unfollowable(Unfollowable::NoVocabulary))?
        {
            let id = id
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or(unfollowable(Unfollowable::NoVocabulary))?;
            ids.insert(token.clone(), id);
        }
        let unknown_token = text(&model["unk_token"], Unfollowable::NoVocabulary)?;
        let unknown = *ids
            .get(unknown_token)
            .ok_or(unfollowable(Unfollowable::NoVocabulary))?;
        let prefix = text(
            &model["continuing_subword_prefix"],
            Unfollowable::NoVocabulary,
        )?
        .to_string();
        let longest_word = model["max_input_chars_per_word"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(unfollowable(Unfollowable::NoVocabulary))?;

        // BERT's normaliser and nothing else: another normaliser changes the
        // text before any of it is split, so every id after it could differ.
        let normaliser = &read["normalizer"];
        let flag = |key: &str| {
            normaliser[key]
                .as_bool()
                .ok_or(unfollowable(Unfollowable::NormaliserNotKnown))
        };
        if normaliser["type"].as_str() != Some("BertNormalizer") {
            return Err(unfollowable(Unfollowable::NormaliserNotKnown));
        }
        let (clean_text, chinese, lowercase) = (
            flag("clean_text")?,
            flag("handle_chinese_chars")?,
            flag("lowercase")?,
        );
        let strip_accents = match &normaliser["strip_accents"] {
            serde_json::Value::Null => lowercase,
            serde_json::Value::Bool(strip) => *strip,
            _ => return Err(unfollowable(Unfollowable::NormaliserNotKnown)),
        };
        if read["pre_tokenizer"]["type"].as_str() != Some("BertPreTokenizer") {
            return Err(unfollowable(Unfollowable::PreTokeniserNotKnown));
        }

        // The template for one sentence: a special token, the sentence, a
        // special token, all of type 0. The ids are the ones the template
        // itself lists for each.
        let template = &read["post_processor"];
        let special = |item: &serde_json::Value| -> Result<u32, TokeniserError> {
            let name = item["SpecialToken"]["id"]
                .as_str()
                .ok_or(unfollowable(Unfollowable::TemplateNotKnown))?;
            match template["special_tokens"][name]["ids"]
                .as_array()
                .map(Vec::as_slice)
            {
                Some([id]) => id
                    .as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .ok_or(unfollowable(Unfollowable::TemplateNotKnown)),
                _ => Err(unfollowable(Unfollowable::TemplateNotKnown)),
            }
        };
        let single = template["single"].as_array().map(Vec::as_slice);
        let (first, last) = match (template["type"].as_str(), single) {
            (Some("TemplateProcessing"), Some([before, sentence, after]))
                if sentence["Sequence"]["id"].as_str() == Some("A")
                    && [
                        before["SpecialToken"]["type_id"].as_u64(),
                        sentence["Sequence"]["type_id"].as_u64(),
                        after["SpecialToken"]["type_id"].as_u64(),
                    ] == [Some(0), Some(0), Some(0)] =>
            {
                (special(before)?, special(after)?)
            }
            _ => return Err(unfollowable(Unfollowable::TemplateNotKnown)),
        };

        // Truncation keeps the start of a sentence, its two wrapping ids
        // counted. Padding adds ids the model masks out; an encoder run on the
        // sentence alone gives the same states for it, so padding is not
        // applied here.
        let truncation = &read["truncation"];
        let truncate_to = if truncation.is_null() {
            None
        } else {
            let length = truncation["max_length"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok());
            match (
                length,
                truncation["direction"].as_str(),
                truncation["strategy"].as_str(),
                truncation["stride"].as_u64(),
            ) {
                (Some(length), Some("Right"), Some("LongestFirst"), Some(0)) if length >= 2 => {
                    Some(length)
                }
                _ => return Err(unfollowable(Unfollowable::TruncationNotKnown)),
            }
        };

        // Added tokens are matched in the text as written, so only those the
        // file marks as not normalised and not stripping or matching whole
        // words are followed.
        let mut added = Vec::new();
        for entry in read["added_tokens"].as_array().into_iter().flatten() {
            let plain = entry["normalized"].as_bool() == Some(false)
                && [&entry["lstrip"], &entry["rstrip"], &entry["single_word"]]
                    .iter()
                    .all(|flag| flag.as_bool() == Some(false));
            let content = entry["content"].as_str();
            let id = entry["id"].as_u64().and_then(|id| u32::try_from(id).ok());
            match (plain, content, id) {
                (true, Some(content), Some(id)) if !content.is_empty() => {
                    ids.insert(content.to_string(), id);
                    added.push((content.to_string(), id));
                }
                _ => return Err(unfollowable(Unfollowable::AddedTokenNotPlain)),
            }
        }
        added.sort_by(|left, right| right.0.len().cmp(&left.0.len()).then(left.0.cmp(&right.0)));

        Ok(Self {
            ids,
            unknown,
            prefix,
            longest_word,
            clean_text,
            chinese,
            strip_accents,
            lowercase,
            first,
            last,
            truncate_to,
            added,
        })
    }

    /// The ids of `text` as the model reads one sentence: wrapped in the
    /// template's two ids, and cut to the file's length when it truncates.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        while let Some((at, length, id)) = self.first_added(rest) {
            self.encode_text(&rest[..at], &mut ids);
            ids.push(id);
            rest = &rest[at + length..];
        }
        self.encode_text(rest, &mut ids);
        if let Some(most) = self.truncate_to {
            ids.truncate(most - 2);
        }
        let mut out = Vec::with_capacity(ids.len() + 2);
        out.push(self.first);
        out.extend(ids);
        out.push(self.last);
        out
    }

    /// The first added token in `text`: where, how many bytes, and its id.
    /// Earlier wins; at the same place the longer, tried first.
    fn first_added(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut found: Option<(usize, usize, u32)> = None;
        for (token, id) in &self.added {
            if let Some(at) = text.find(token.as_str()) {
                if found.is_none_or(|(seen, _, _)| at < seen) {
                    found = Some((at, token.len(), *id));
                }
            }
        }
        found
    }

    /// The text as the normaliser leaves it, in the authority's order.
    fn normalise(&self, text: &str) -> String {
        let mut out: String = if self.clean_text {
            text.chars()
                .filter(|&c| !(c == '\0' || c == '\u{FFFD}' || is_control(c)))
                .map(|c| if is_whitespace(c) { ' ' } else { c })
                .collect()
        } else {
            text.to_string()
        };
        if self.chinese {
            let mut spaced = String::with_capacity(out.len());
            for c in out.chars() {
                if is_chinese(c) {
                    spaced.push(' ');
                    spaced.push(c);
                    spaced.push(' ');
                } else {
                    spaced.push(c);
                }
            }
            out = spaced;
        }
        if self.strip_accents {
            out = out
                .nfd()
                .map(|(c, _)| c)
                .filter(|c| !c.is_mark_nonspacing())
                .collect();
        }
        if self.lowercase {
            out = out.chars().flat_map(char::to_lowercase).collect();
        }
        out
    }

    /// The ids of text with no added token in it.
    fn encode_text(&self, text: &str, out: &mut Vec<u32>) {
        let normal = self.normalise(text);
        for word in normal.split(char::is_whitespace) {
            // Punctuation is a word of its own.
            let mut start = 0;
            for (at, c) in word.char_indices() {
                if is_punctuation(c) {
                    self.word(&word[start..at], out);
                    self.word(&word[at..at + c.len_utf8()], out);
                    start = at + c.len_utf8();
                }
            }
            self.word(&word[start..], out);
        }
    }

    /// One word: the longest piece of the vocabulary from its start, then the
    /// longest continuing piece from where that ends, and so on. A word that
    /// cannot be covered so, or is longer than the file allows, is the
    /// unknown token once.
    fn word(&self, word: &str, out: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        if word.chars().count() > self.longest_word {
            out.push(self.unknown);
            return;
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                let piece = if start > 0 {
                    format!("{}{}", self.prefix, &word[start..end])
                } else {
                    word[start..end].to_string()
                };
                if let Some(id) = self.ids.get(&piece) {
                    found = Some(*id);
                    break;
                }
                end -= word[start..end].chars().last().map_or(1, char::len_utf8);
            }
            let Some(id) = found else {
                out.push(self.unknown);
                return;
            };
            pieces.push(id);
            start = end;
        }
        out.extend(pieces);
    }
}
