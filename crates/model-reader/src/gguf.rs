//! The tensor table of a GGUF file: what each tensor is, what shape it has,
//! and where its bytes lie.
//!
//! A model published as GGUF carries its weights in one file, after a header
//! that names every tensor in it. A converter turning such a file into a
//! weight container has to read that table — which tensor, what type, which
//! bytes — and nothing else about the file.
//!
//! This reads the table and stops. It does not read a tensor's bytes, decide
//! whether a type may be converted, or say anything about what a container
//! stores: a block here is as long as the *file* says it is, and what our own
//! container stores for the same weights is the kernel's thirteenth decision,
//! answered elsewhere. A reader that reported our length instead of the
//! file's would be a converter lying through its reader, and the first sign of
//! it would be a table that disagrees with the file by a few bytes a block,
//! several hundred million bytes in.
//!
//! The tokeniser reads the same files, and reads the part before this one: the
//! metadata, where a vocabulary lives. The two stop in different places on
//! purpose, and `EN-27` requires them to agree about where the metadata ends
//! by reading one file with both.

use std::collections::BTreeMap;

/// What a GGUF file begins with.
const MAGIC: [u8; 4] = *b"GGUF";

/// The version this reader follows.
const VERSION: u32 = 3;

/// How the file's own bytes are aligned when no `general.alignment` says
/// otherwise, as the format documents.
const ALIGNMENT: u64 = 32;

/// Why a GGUF file's tensor table could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufError {
    /// `FL-36`
    NotGguf,
    /// `FL-37`
    VersionNotKnown,
    /// `FL-38`
    Truncated,
    /// `FL-39`: the number the file writes for a type this reader does not
    /// follow. Named by its number, because that is what the file says and
    /// what someone adding it would look up.
    TypeNotFollowed(u32),
    /// `FL-40`
    ShapeNotFollowed,
}

/// How a tensor's numbers are stored, as the format numbers the types.
///
/// Only the ones this repository has seen in a published file. Everything
/// else is refused by its number rather than guessed at: a type read as the
/// wrong one would put every tensor after it in the wrong place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Four bytes a number.
    F32,
    /// Two bytes a number.
    F16,
    /// Thirty-two numbers in eighteen bytes: an f16 scale, then a nibble each.
    Q4_0,
    /// Thirty-two numbers in thirty-four bytes: an f16 scale, then a byte each.
    Q8_0,
}

impl Kind {
    /// The number the file writes for this type.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
        }
    }

    /// How many numbers one block holds, and how many bytes that block is.
    ///
    /// These are the file's own figures. What a weight container stores for
    /// the same numbers is a different question, and not this reader's.
    #[must_use]
    pub const fn block(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q8_0 => (32, 34),
        }
    }

    /// The type a file's number stands for, or the number itself if this
    /// reader does not follow it.
    fn of(code: u32) -> Result<Self, GgufError> {
        match code {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            other => Err(GgufError::TypeNotFollowed(other)),
        }
    }
}

/// One tensor, as the file's table describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// What the model calls it.
    pub name: String,
    /// Its shape, in the order the file writes it: the fastest-moving
    /// dimension first, which is the opposite of how a matrix is usually
    /// written down.
    pub shape: Vec<u64>,
    /// How its numbers are stored.
    pub kind: Kind,
    /// Where its bytes begin, counted from the start of the file.
    pub at: u64,
    /// How many bytes they are, from the shape and the type.
    pub bytes: u64,
}

impl Tensor {
    /// How many numbers it holds.
    #[must_use]
    pub fn numbers(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// A GGUF file's tensor table, read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensors {
    /// Where the tensor data begins, counted from the start of the file.
    pub data_at: u64,
    /// The tensors, in the order the file lists them.
    pub tensors: Vec<Tensor>,
}

impl Tensors {
    /// The tensor of that name, if the file has one.
    #[must_use]
    pub fn named(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

/// Read a GGUF file's tensor table.
///
/// `file` must reach the end of the table; the tensors themselves need not be
/// there, and are never read.
pub fn read(file: &[u8]) -> Result<Tensors, GgufError> {
    let mut at = Reader { bytes: file, at: 0 };
    if at.fixed::<4>()? != MAGIC {
        return Err(GgufError::NotGguf);
    }
    if at.u32()? != VERSION {
        return Err(GgufError::VersionNotKnown);
    }
    let count = at.u64()?;
    let entries = at.u64()?;

    // The metadata is stepped over, not read: a vocabulary is the tokeniser's
    // business. What is taken from it is the alignment, if it says one.
    let mut alignment = ALIGNMENT;
    for _ in 0..entries {
        let name = at.text()?;
        let kind = at.u32()?;
        if name == "general.alignment" && kind == kind::UINT32 {
            alignment = u64::from(at.u32()?);
        } else {
            at.step_over(kind)?;
        }
    }

    let mut tensors = Vec::new();
    let mut seen = BTreeMap::new();
    for _ in 0..count {
        let name = at.text()?;
        let dimensions = at.u32()?;
        let mut shape = Vec::new();
        for _ in 0..dimensions {
            shape.push(at.u64()?);
        }
        let kind = Kind::of(at.u32()?)?;
        let offset = at.u64()?;
        let bytes = measured(&shape, kind)?;
        seen.insert(name.clone(), offset);
        tensors.push(Tensor {
            name,
            shape,
            kind,
            at: offset,
            bytes,
        });
    }

    // Where the file's own bytes begin: the table's end, rounded up as the
    // file says. Every tensor's offset is from there, so this reader hands
    // back offsets from the start of the file and nobody has to add it twice.
    let after = u64::try_from(at.at).map_err(|_| GgufError::Truncated)?;
    let data_at = after
        .checked_next_multiple_of(alignment.max(1))
        .ok_or(GgufError::Truncated)?;
    for tensor in &mut tensors {
        tensor.at = tensor
            .at
            .checked_add(data_at)
            .ok_or(GgufError::ShapeNotFollowed)?;
        tensor
            .at
            .checked_add(tensor.bytes)
            .ok_or(GgufError::ShapeNotFollowed)?;
    }
    Ok(Tensors { data_at, tensors })
}

/// How many bytes a tensor of this shape and type is, as the file stores it.
fn measured(shape: &[u64], kind: Kind) -> Result<u64, GgufError> {
    if shape.is_empty() {
        return Err(GgufError::ShapeNotFollowed);
    }
    let mut numbers: u64 = 1;
    for side in shape {
        if *side == 0 {
            return Err(GgufError::ShapeNotFollowed);
        }
        numbers = numbers
            .checked_mul(*side)
            .ok_or(GgufError::ShapeNotFollowed)?;
    }
    let (per_block, bytes) = kind.block();
    // A block holds a whole number of numbers, so a tensor that does not fill
    // its last block is one this reader cannot measure rather than one it
    // rounds.
    if !numbers.is_multiple_of(per_block) {
        return Err(GgufError::ShapeNotFollowed);
    }
    (numbers / per_block)
        .checked_mul(bytes)
        .ok_or(GgufError::ShapeNotFollowed)
}

/// The value types the format numbers, as far as stepping over them needs.
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

/// How wide a value of that type is, where it is a number.
fn width(kind: u32) -> Option<usize> {
    match kind {
        kind::UINT8 | kind::INT8 | kind::BOOL => Some(1),
        kind::UINT16 | kind::INT16 => Some(2),
        kind::UINT32 | kind::INT32 | kind::FLOAT32 => Some(4),
        kind::UINT64 | kind::INT64 | kind::FLOAT64 => Some(8),
        _ => None,
    }
}

/// A place in the file, and the bytes from there on.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    /// The next `N` bytes.
    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let end = self.at.checked_add(N).ok_or(GgufError::Truncated)?;
        let taken = self
            .bytes
            .get(self.at..end)
            .ok_or(GgufError::Truncated)?
            .try_into()
            .map_err(|_| GgufError::Truncated)?;
        self.at = end;
        Ok(taken)
    }

    /// The next four bytes as a number.
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.fixed::<4>()?))
    }

    /// The next eight bytes as a number.
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.fixed::<8>()?))
    }

    /// The next string: how long it is, then that many bytes of text.
    fn text(&mut self) -> Result<String, GgufError> {
        let length = usize::try_from(self.u64()?).map_err(|_| GgufError::Truncated)?;
        let end = self.at.checked_add(length).ok_or(GgufError::Truncated)?;
        let taken = self.bytes.get(self.at..end).ok_or(GgufError::Truncated)?;
        self.at = end;
        // A name that is not text is a file this reader cannot follow, rather
        // than bytes to be read some other way.
        core::str::from_utf8(taken)
            .map(str::to_string)
            .map_err(|_| GgufError::Truncated)
    }

    /// Step over a value of `kind` without keeping it.
    fn step_over(&mut self, kind: u32) -> Result<(), GgufError> {
        match kind {
            kind::STRING => {
                self.text()?;
            }
            kind::ARRAY => {
                let inner = self.u32()?;
                let count = usize::try_from(self.u64()?).map_err(|_| GgufError::Truncated)?;
                for _ in 0..count {
                    self.step_over(inner)?;
                }
            }
            _ => {
                let width = width(kind).ok_or(GgufError::Truncated)?;
                let end = self.at.checked_add(width).ok_or(GgufError::Truncated)?;
                if end > self.bytes.len() {
                    return Err(GgufError::Truncated);
                }
                self.at = end;
            }
        }
        Ok(())
    }
}
