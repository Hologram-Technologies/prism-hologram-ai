//! The tokeniser a GGUF model carries in its own header.
//!
//! A GGUF file begins with a magic number, a version, the number of tensors
//! and the number of metadata entries, then the metadata itself: a key, a type
//! and a value, over and over. The tensors and their data come after all of
//! it, and nothing here reads them. A vocabulary of a hundred and fifty
//! thousand tokens sits in the header of a file of gigabytes, and the header is
//! all a count needs.
//!
//! What the keys mean is the format's own specification; which pattern a
//! pre-tokeniser's name stands for is `llama.cpp`'s, because the name is all
//! the file carries and the pattern lives in the reader. Both are cited in
//! `model/authorities.toml`. A name this crate has no cited pattern for is
//! refused rather than guessed at.

use crate::{Tokeniser, TokeniserError, Unfollowable};

/// `GGUF`, the four bytes a file of this format begins with.
const MAGIC: [u8; 4] = *b"GGUF";

/// The version this crate reads. Version 3 is the one the specification
/// describes; an older file is a different format with the same name.
const VERSION: u32 = 3;

/// The keys a tokeniser is built from.
mod key {
    pub(crate) const MODEL: &str = "tokenizer.ggml.model";
    pub(crate) const PRE: &str = "tokenizer.ggml.pre";
    pub(crate) const TOKENS: &str = "tokenizer.ggml.tokens";
    pub(crate) const MERGES: &str = "tokenizer.ggml.merges";
    pub(crate) const SPACE_PREFIX: &str = "tokenizer.ggml.add_space_prefix";
}

/// The value types the format defines, by the number that names them.
mod kind {
    pub(crate) const UINT8: u32 = 0;
    pub(crate) const INT8: u32 = 1;
    pub(crate) const UINT16: u32 = 2;
    pub(crate) const INT16: u32 = 3;
    pub(crate) const UINT32: u32 = 4;
    pub(crate) const INT32: u32 = 5;
    pub(crate) const FLOAT32: u32 = 6;
    pub(crate) const BOOL: u32 = 7;
    pub(crate) const STRING: u32 = 8;
    pub(crate) const ARRAY: u32 = 9;
    pub(crate) const UINT64: u32 = 10;
    pub(crate) const INT64: u32 = 11;
    pub(crate) const FLOAT64: u32 = 12;
}

/// How wide a value of each fixed-width type is.
const fn width(kind: u32) -> Option<usize> {
    Some(match kind {
        kind::UINT8 | kind::INT8 | kind::BOOL => 1,
        kind::UINT16 | kind::INT16 => 2,
        kind::UINT32 | kind::INT32 | kind::FLOAT32 => 4,
        kind::UINT64 | kind::INT64 | kind::FLOAT64 => 8,
        _ => return None,
    })
}

/// A position in the header.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], TokeniserError> {
        let end = self.at + N;
        let out = self
            .bytes
            .get(self.at..end)
            .and_then(|part| <[u8; N]>::try_from(part).ok())
            .ok_or(TokeniserError::Truncated)?;
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, TokeniserError> {
        Ok(u32::from_le_bytes(self.fixed::<4>()?))
    }

    fn u64(&mut self) -> Result<u64, TokeniserError> {
        Ok(u64::from_le_bytes(self.fixed::<8>()?))
    }

    /// A length and that many bytes of text.
    fn text(&mut self) -> Result<String, TokeniserError> {
        let length = usize::try_from(self.u64()?).map_err(|_| TokeniserError::Truncated)?;
        let end = self
            .at
            .checked_add(length)
            .ok_or(TokeniserError::Truncated)?;
        let part = self
            .bytes
            .get(self.at..end)
            .ok_or(TokeniserError::Truncated)?;
        self.at = end;
        // A token is text. A name that is not text is a file this crate cannot
        // follow, rather than bytes to be read some other way.
        core::str::from_utf8(part)
            .map(str::to_string)
            .map_err(|_| TokeniserError::Unfollowable(Unfollowable::NameNotText))
    }

    /// Step over a value of `kind` without keeping it.
    fn step_over(&mut self, kind: u32) -> Result<(), TokeniserError> {
        match kind {
            kind::STRING => {
                self.text()?;
            }
            kind::ARRAY => {
                let inner = self.u32()?;
                let count = usize::try_from(self.u64()?).map_err(|_| TokeniserError::Truncated)?;
                for _ in 0..count {
                    self.step_over(inner)?;
                }
            }
            _ => {
                let width = width(kind).ok_or(TokeniserError::Unfollowable(
                    Unfollowable::ValueTypeNotKnown,
                ))?;
                let end = self
                    .at
                    .checked_add(width)
                    .ok_or(TokeniserError::Truncated)?;
                if end > self.bytes.len() {
                    return Err(TokeniserError::Truncated);
                }
                self.at = end;
            }
        }
        Ok(())
    }

    /// An array of text, which is how a vocabulary and its merges are written.
    fn texts(&mut self) -> Result<Vec<String>, TokeniserError> {
        let inner = self.u32()?;
        let count = usize::try_from(self.u64()?).map_err(|_| TokeniserError::Truncated)?;
        if inner != kind::STRING {
            // Step over it anyway, so a file that writes one of these keys as
            // numbers is refused for what it is rather than read as text.
            for _ in 0..count {
                self.step_over(inner)?;
            }
            return Err(TokeniserError::Unfollowable(Unfollowable::NoVocabulary));
        }
        let mut out = Vec::new();
        for _ in 0..count {
            out.push(self.text()?);
        }
        Ok(out)
    }
}

/// Build a tokeniser from a GGUF file's header.
///
/// `file` may be the whole model or only as much of it as holds the header:
/// nothing past the metadata is read.
pub(crate) fn read(file: &[u8]) -> Result<Tokeniser, TokeniserError> {
    let mut reader = Reader { bytes: file, at: 0 };
    if reader.fixed::<4>()? != MAGIC {
        return Err(TokeniserError::NotGguf);
    }
    if reader.u32()? != VERSION {
        return Err(TokeniserError::Unfollowable(Unfollowable::VersionNotKnown));
    }
    // The tensor count, which this crate never uses: the tensors themselves
    // are past everything it reads.
    reader.u64()?;
    let entries = reader.u64()?;

    let mut model = None;
    let mut pre = None;
    let mut tokens = Vec::new();
    let mut merges = Vec::new();
    let mut space_prefix = false;
    for _ in 0..entries {
        let name = reader.text()?;
        let kind = reader.u32()?;
        match (name.as_str(), kind) {
            (key::MODEL, kind::STRING) => model = Some(reader.text()?),
            (key::PRE, kind::STRING) => pre = Some(reader.text()?),
            (key::TOKENS, kind::ARRAY) => tokens = reader.texts()?,
            (key::MERGES, kind::ARRAY) => merges = reader.texts()?,
            (key::SPACE_PREFIX, kind::BOOL) => space_prefix = reader.fixed::<1>()? != [0],
            _ => reader.step_over(kind)?,
        }
    }

    // Only one tokeniser model is byte-level byte-pair encoding. The others
    // are refused for what they are: a count from the wrong one would be a
    // number the model never meant.
    if model.as_deref() != Some("gpt2") {
        return Err(TokeniserError::Unfollowable(Unfollowable::NotBytePair));
    }
    if tokens.is_empty() {
        return Err(TokeniserError::Unfollowable(Unfollowable::NoVocabulary));
    }
    let pattern = crate::pattern_named(pre.as_deref())?;
    Tokeniser::assemble(tokens, merges, Some(pattern), space_prefix)
}
