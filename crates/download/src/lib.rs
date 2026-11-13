//! Labels a download's bytes as they arrive, and asks the kernel whether the
//! buffer may be used.
//!
//! An archive names each buffer it carries by a label, the BLAKE3 hash of the
//! bytes, and by a length. A download is a stream of chunks from somewhere
//! else: the label is taken as the chunks arrive, so nothing is held twice and
//! nothing is read before it is answered for.
//!
//! Whether the buffer may be used is not decided here. The expected label and
//! length, and the observed ones, go to the kernel, which admits or refuses.
//! This crate fetches nothing: whoever holds the connection hands the chunks
//! in, which is also how it works in a browser, where fetching belongs to the
//! page.

#![forbid(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hologram_ai_kernel_client::admission;

/// How many bytes a label is.
pub const LABEL_BYTES: usize = 32;

/// The label of a buffer: the BLAKE3 hash of its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label(pub [u8; LABEL_BYTES]);

/// Why a download could not be admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadError {
    /// `FL-14`
    LabelNotHex,
    /// `FL-15`: the kernel's own answer, carried as it gave it.
    Refused(admission::Verdict),
}

impl Label {
    /// The label an archive names, written as sixty-four hexadecimal digits in
    /// lower case. Upper case, a shorter or longer name, and anything that is
    /// not a digit are each refused: a label read loosely is a label that can
    /// be made to match.
    pub fn from_hex(text: &str) -> Result<Self, DownloadError> {
        let digits = text.as_bytes();
        if digits.len() != LABEL_BYTES * 2 {
            return Err(DownloadError::LabelNotHex);
        }
        let mut out = [0; LABEL_BYTES];
        for (byte, pair) in out.iter_mut().zip(digits.chunks_exact(2)) {
            let mut value = 0u8;
            for digit in pair {
                let part = match digit {
                    b'0'..=b'9' => digit - b'0',
                    b'a'..=b'f' => digit - b'a' + 10,
                    _ => return Err(DownloadError::LabelNotHex),
                };
                value = value * 16 + part;
            }
            *byte = value;
        }
        Ok(Self(out))
    }

    /// The label as an archive writes it.
    #[must_use]
    pub fn to_hex(self) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(LABEL_BYTES * 2);
        for byte in self.0 {
            out.push(char::from(DIGITS[usize::from(byte >> 4)]));
            out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
        }
        out
    }
}

/// A download in progress: the label of what has arrived, and how much.
#[derive(Clone, Debug, Default)]
pub struct Arriving {
    hasher: blake3::Hasher,
    received: u64,
}

impl Arriving {
    /// A download of nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take in one chunk, as it arrives.
    pub fn take(&mut self, chunk: &[u8]) {
        self.hasher.update(chunk);
        self.received = self.received.saturating_add(chunk.len() as u64);
    }

    /// How many bytes have arrived.
    #[must_use]
    pub const fn received(&self) -> u64 {
        self.received
    }

    /// The label of everything that has arrived.
    #[must_use]
    pub fn label(&self) -> Label {
        Label(*self.hasher.finalize().as_bytes())
    }

    /// The kernel's question about this download, against the label and length
    /// the archive names.
    #[must_use]
    pub fn question(&self, expected: Label, declared: u64) -> [u8; 81] {
        admission::request(&expected.0, &self.label().0, declared, self.received)
    }

    /// Put the question to `kernel` and carry its answer.
    ///
    /// The buffer may be used only where this returns `Ok`. A refusal carries
    /// the kernel's own verdict, so what was decided is not restated here, and
    /// an answer the client does not know is a refusal like any other.
    pub fn admitted(
        &self,
        expected: Label,
        declared: u64,
        kernel: &mut dyn Kernel,
    ) -> Result<(), DownloadError> {
        let answer = kernel.ask(&self.question(expected, declared));
        match admission::answer(&answer) {
            admission::Verdict::Admit => Ok(()),
            refused => Err(DownloadError::Refused(refused)),
        }
    }
}

/// Whoever holds the verified kernel holo.
///
/// The same shape the rest of the engine uses: a request's bytes in, an
/// answer's bytes out.
pub trait Kernel {
    /// Put `request` to the kernel and return its answer.
    fn ask(&mut self, request: &[u8]) -> Vec<u8>;
}
