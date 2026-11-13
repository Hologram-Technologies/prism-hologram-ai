//! As much of the protocol-buffer wire format as a preflight needs.
//!
//! A message is a run of fields. Each field begins with a number holding its
//! field number and how it is encoded: a variable-length number, eight bytes,
//! four bytes, or a length and that many bytes. A reader that does not know a
//! field steps over it, which is how a file written against a newer schema
//! still reads.
//!
//! Numbers are variable-length, seven bits at a time, least significant first,
//! the top bit saying another byte follows. Ten such bytes carry the sixty-four
//! bits a number holds; an eleventh is not a number this format can mean.

use crate::OnnxError;

/// How a field is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Encoding {
    /// A variable-length number.
    Number,
    /// Eight bytes.
    Eight,
    /// Four bytes.
    Four,
    /// A length, then that many bytes.
    Bytes,
}

impl Encoding {
    fn of(key: u64) -> Result<Self, OnnxError> {
        match key & 7 {
            0 => Ok(Self::Number),
            1 => Ok(Self::Eight),
            2 => Ok(Self::Bytes),
            5 => Ok(Self::Four),
            // 3 and 4 delimited groups, which this format removed; 6 and 7 were
            // never defined. A file using one is not a file this reader can
            // step over, because it cannot know where the field ends.
            _ => Err(OnnxError::UnknownEncoding),
        }
    }
}

/// A position in a message.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// Is there another field?
    pub(crate) fn more(&self) -> bool {
        self.at < self.bytes.len()
    }

    /// The next field's number and encoding.
    pub(crate) fn field(&mut self) -> Result<(u64, Encoding), OnnxError> {
        let key = self.number()?;
        Ok((key >> 3, Encoding::of(key)?))
    }

    /// A variable-length number.
    pub(crate) fn number(&mut self) -> Result<u64, OnnxError> {
        let mut value = 0u64;
        for step in 0..10u32 {
            let byte = *self.bytes.get(self.at).ok_or(OnnxError::Truncated)?;
            self.at += 1;
            // The tenth byte carries the one bit left of the sixty-four, so
            // only its lowest bit is part of the number; the rest is the sign
            // extension this format writes for a negative number, and a
            // negative number here is read as the large number it stands for.
            value |= u64::from(byte & 0x7f).wrapping_shl(step * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(OnnxError::NumberTooLong)
    }

    /// The bytes of a length-delimited field.
    pub(crate) fn delimited(&mut self) -> Result<&'a [u8], OnnxError> {
        let length = usize::try_from(self.number()?).map_err(|_| OnnxError::Truncated)?;
        let end = self.at.checked_add(length).ok_or(OnnxError::Truncated)?;
        let out = self.bytes.get(self.at..end).ok_or(OnnxError::Truncated)?;
        self.at = end;
        Ok(out)
    }

    /// Step over a field this reader does not need.
    pub(crate) fn step_over(&mut self, encoding: Encoding) -> Result<(), OnnxError> {
        match encoding {
            Encoding::Number => {
                self.number()?;
            }
            Encoding::Eight => self.fixed(8)?,
            Encoding::Four => self.fixed(4)?,
            Encoding::Bytes => {
                self.delimited()?;
            }
        }
        Ok(())
    }

    fn fixed(&mut self, width: usize) -> Result<(), OnnxError> {
        // The sum cannot overflow: the position is inside a slice that exists,
        // and a fixed field is four bytes or eight. Checking it would be a
        // branch no file could take, which is a branch no test could reach.
        let end = self.at + width;
        if end > self.bytes.len() {
            return Err(OnnxError::Truncated);
        }
        self.at = end;
        Ok(())
    }
}

/// The text of a length-delimited field.
pub(crate) fn text(bytes: &[u8]) -> Result<String, OnnxError> {
    core::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|_| OnnxError::NameNotText)
}
