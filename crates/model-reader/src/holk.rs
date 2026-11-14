//! Reads the header of a weight container into the questions the kernel
//! answers.
//!
//! A container carries a model's weights in one file: a magic, the length of a
//! header, that header as JSON, and then the buffers it describes. The header
//! names each buffer by a label and a span, says where the tokeniser the model
//! carries lies, and carries its manifest's own identity, which is not the
//! hash of the file it is written in.
//!
//! What this reads is the header and nothing else. The buffers themselves may
//! be hundreds of megabytes and arrive later, or never; whether each one lies
//! where it claims, and whether its bytes are the ones named, are the kernel's
//! decisions, asked here and answered there.
//!
//! The layout is not a published specification. It is measured from the one
//! file this engine has seen, and is written down here as what that file is:
//! six bytes of magic, a four-byte length, JSON, spaces to fill. Anything that
//! deviates is refused by name rather than read as a near one — a reader that
//! guessed would be guessing about a model's weights.

use hologram_ai_download::{Label, Root};
use hologram_ai_kernel_client::{admission, blocks, range};

/// The magic a container begins with, the newline included.
pub const MAGIC: &[u8; 6] = b"HOLK1\n";

/// The format its header names.
pub const FORMAT: &str = "holo-kappa/1";

/// Where the header's length is written, and how wide it is.
const LENGTH_AT: usize = MAGIC.len();
const LENGTH_BYTES: usize = 4;

/// Where the JSON begins.
const JSON_AT: usize = LENGTH_AT + LENGTH_BYTES;

/// Why a container could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HolkError {
    /// `FL-31`
    NotHolk,
    /// `FL-32`
    Truncated,
    /// `FL-33`
    NotJson,
    /// `FL-34`
    PaddedWithSomethingElse,
    /// `FL-35`: the header describes something this reader does not follow.
    /// What it describes is named, so a model that cannot be loaded says why.
    Unfollowable(Unfollowable),
}

/// What about a container this reader does not follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unfollowable {
    /// The header names a format other than [`FORMAT`].
    NotThisFormat,
    /// The header has no manifest, no directory, or no tokeniser.
    PartMissing,
    /// A number the header carries is not a whole number that fits.
    NotAWholeNumber,
    /// A name the header carries is not text.
    NotText,
    /// A buffer's label is not `blake3:` and sixty-four digits.
    LabelNotBlake3,
    /// The file's root is not `sha256:` and sixty-four digits.
    RootNotSha256,
    /// A quantisation format is not one the kernel knows.
    FormatNotKnown,
}

/// Where something lies in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// The byte it starts at.
    pub at: u64,
    /// How many bytes it is.
    pub len: u64,
}

impl Span {
    /// The byte after this span.
    #[must_use]
    pub const fn end(self) -> u64 {
        self.at.saturating_add(self.len)
    }
}

/// One buffer the container carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// What the model calls this tensor.
    pub name: String,
    /// The quantisation format its bytes are in, as the kernel numbers them.
    pub format: u8,
    /// How many rows it has.
    pub rows: u64,
    /// How many columns it has.
    pub columns: u64,
    /// The label its bytes must have.
    pub label: Label,
    /// Where its bytes lie.
    pub span: Span,
}

impl Entry {
    /// The kernel's question about where this buffer lies: whether it follows
    /// the one before it and ends inside a file of `file_length` bytes.
    #[must_use]
    pub fn range_question(&self, file_length: u64, previous_end: u64) -> [u8; 33] {
        range::request(file_length, previous_end, self.span.at, self.span.len)
    }

    /// The kernel's question about whether these bytes may be used, against
    /// the label the container names and the `observed` label of what arrived.
    #[must_use]
    pub fn admission_question(&self, observed: Label, received: u64) -> [u8; 81] {
        admission::request(&self.label.0, &observed.0, self.span.len, received)
    }

    /// The kernel's question about how long one block of this shape is.
    #[must_use]
    pub fn block_question(&self) -> [u8; 18] {
        blocks::request(self.format, self.rows, self.columns)
    }
}

/// What a container's manifest calls its own root.
///
/// Thirty-two bytes, written `sha256:` and sixty-four digits — and not the
/// hash of the file it is written in. For the published container it is the
/// identity its catalogue pinned, over the manifest the model was built from:
/// the header names `8932967d…`, while the file's own bytes hash to
/// `acf0b5ad…`, which is what a download arrives at and what the hub record
/// says. Nothing here recomputes what the identity covers.
///
/// It is its own type for that reason. A [`Root`] is what the bytes of a
/// download come to; this is what a manifest says about itself; and the two
/// being different types is what stops one being compared to the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity(pub [u8; 32]);

impl Identity {
    /// The identity a manifest names, read as strictly as a label is.
    pub fn from_hex(text: &str) -> Result<Self, HolkError> {
        Root::from_hex(text)
            .map(|read| Self(read.0))
            .map_err(|_| unfollowable(Unfollowable::RootNotSha256))
    }
}

/// What the header says about the model as a whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    /// What the manifest calls its root, which is not this file's hash.
    pub identity: Identity,
    /// How many tokens the model's vocabulary has.
    pub vocabulary: u64,
    /// How many layers it has.
    pub layers: u64,
}

/// A container's header, read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Container {
    /// What the header says about the model as a whole.
    pub manifest: Manifest,
    /// Where the tokeniser lies: a GGUF header, which the tokeniser crate
    /// reads.
    pub tokeniser: Span,
    /// The buffers it carries, in the order the header lists them.
    pub entries: Vec<Entry>,
}

impl Container {
    /// Where the header's JSON lies in a file that begins with `bytes`.
    ///
    /// Only the first ten bytes are needed to answer, which is what lets a
    /// reader ask for the header before it has the file.
    pub fn header_span(bytes: &[u8]) -> Result<Span, HolkError> {
        if bytes.len() < JSON_AT {
            return Err(HolkError::Truncated);
        }
        if bytes.get(..MAGIC.len()) != Some(MAGIC.as_slice()) {
            return Err(HolkError::NotHolk);
        }
        let mut length = [0u8; LENGTH_BYTES];
        for (byte, taken) in length
            .iter_mut()
            .zip(bytes.iter().skip(LENGTH_AT).take(LENGTH_BYTES))
        {
            *byte = *taken;
        }
        Ok(Span {
            at: JSON_AT as u64,
            len: u64::from(u32::from_le_bytes(length)),
        })
    }

    /// Read a container's header from the start of the file.
    ///
    /// `bytes` must reach the end of the header; what follows it need not be
    /// there, and is never read.
    pub fn read(bytes: &[u8]) -> Result<Self, HolkError> {
        let span = Self::header_span(bytes)?;
        let from = usize::try_from(span.at).map_err(|_| HolkError::Truncated)?;
        let to = usize::try_from(span.end()).map_err(|_| HolkError::Truncated)?;
        let header = bytes.get(from..to).ok_or(HolkError::Truncated)?;

        // The published file writes its JSON and fills the rest of the header
        // with spaces. Where the document ends is the parser's to say, not
        // something to look for backwards; anything but spaces after it is
        // something this reader has not seen and will not step over.
        let mut documents =
            serde_json::Deserializer::from_slice(header).into_iter::<serde_json::Value>();
        let Some(Ok(read)) = documents.next() else {
            return Err(HolkError::NotJson);
        };
        let after = documents.byte_offset();
        if header
            .get(after..)
            .is_none_or(|rest| rest.iter().any(|byte| *byte != b' '))
        {
            return Err(HolkError::PaddedWithSomethingElse);
        }

        if read["format"].as_str() != Some(FORMAT) {
            return Err(unfollowable(Unfollowable::NotThisFormat));
        }
        let manifest = &read["manifest"];
        let tokeniser = &read["tokenizer"];
        let listed = read["dir"]
            .as_array()
            .ok_or_else(|| unfollowable(Unfollowable::PartMissing))?;
        if manifest.is_null() || tokeniser.is_null() {
            return Err(unfollowable(Unfollowable::PartMissing));
        }

        // What the manifest says about the model as a whole, which one of the
        // names a buffer carries cannot be read without.
        let ternary = manifest["tern"].as_bool().unwrap_or(false);

        let mut entries = Vec::with_capacity(listed.len());
        for entry in listed {
            entries.push(Entry {
                name: text(&entry["name"])?.to_string(),
                format: format_of(text(&entry["fmt"])?, ternary)?,
                rows: whole(&entry["N"])?,
                columns: whole(&entry["K"])?,
                label: label_of(text(&entry["kappa"])?)?,
                span: span_of(entry)?,
            });
        }

        Ok(Self {
            manifest: Manifest {
                identity: Identity::from_hex(text(&manifest["root"])?)?,
                vocabulary: whole(&manifest["vocab"])?,
                layers: whole(&manifest["n_layers"])?,
            },
            tokeniser: span_of(tokeniser)?,
            entries,
        })
    }

    /// The byte after the last buffer the header describes.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.span.end())
            .max()
            .unwrap_or_else(|| self.tokeniser.end())
    }
}

/// A container describing something this reader does not follow.
fn unfollowable(what: Unfollowable) -> HolkError {
    HolkError::Unfollowable(what)
}

/// The text a field carries.
fn text(value: &serde_json::Value) -> Result<&str, HolkError> {
    value
        .as_str()
        .ok_or_else(|| unfollowable(Unfollowable::NotText))
}

/// The whole number a field carries.
fn whole(value: &serde_json::Value) -> Result<u64, HolkError> {
    value
        .as_u64()
        .ok_or_else(|| unfollowable(Unfollowable::NotAWholeNumber))
}

/// The span a field with an offset and a length describes.
fn span_of(value: &serde_json::Value) -> Result<Span, HolkError> {
    Ok(Span {
        at: whole(&value["off"])?,
        len: whole(&value["len"])?,
    })
}

/// The label a buffer is named by, written as the hash it is.
fn label_of(text: &str) -> Result<Label, HolkError> {
    let digits = text
        .strip_prefix("blake3:")
        .ok_or_else(|| unfollowable(Unfollowable::LabelNotBlake3))?;
    Label::from_hex(digits).map_err(|_| unfollowable(Unfollowable::LabelNotBlake3))
}

/// The number the kernel gives a quantisation format.
///
/// The names are the container's; the numbers are the kernel's, as its own
/// module lists them. A name with no number here is refused rather than read
/// as a near one.
///
/// The published container writes the vectors it does not quantise as `f32`:
/// the norms, which stay full width. The kernel answers for those too.
///
/// `ternary` says whether the manifest declares this model's weights ternary.
/// The published container writes its ternary matrices as `q1`, and the same
/// family writes a one-bit format under that same name with another layout
/// entirely, so `q1` is the kernel's ternary format only where the manifest
/// says the weights are ternary, and is refused where it does not.
fn format_of(name: &str, ternary: bool) -> Result<u8, HolkError> {
    match name {
        "q4" => Ok(0),
        "q3f" => Ok(1),
        "t2" => Ok(2),
        "t2r" => Ok(3),
        "ternary" => Ok(4),
        "q1" if ternary => Ok(4),
        "f32" => Ok(5),
        _ => Err(unfollowable(Unfollowable::FormatNotKnown)),
    }
}
