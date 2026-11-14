//! Labels a download's bytes as they arrive, and asks the kernel whether the
//! buffer may be used.
//!
//! An archive names each buffer it carries by a label, the BLAKE3 hash of the
//! bytes, and by a length; a container names the whole file it carries them in
//! by a root, the SHA-256 of all of it. A download is a stream of chunks from
//! somewhere else: both are taken as the chunks arrive, so nothing is held
//! twice and nothing is read before it is answered for.
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
    /// `FL-30`
    RootNotHex,
}

impl Label {
    /// The label an archive names, written as sixty-four hexadecimal digits in
    /// lower case. Upper case, a shorter or longer name, and anything that is
    /// not a digit are each refused: a label read loosely is a label that can
    /// be made to match.
    pub fn from_hex(text: &str) -> Result<Self, DownloadError> {
        thirty_two(text).map(Self).ok_or(DownloadError::LabelNotHex)
    }

    /// The label as an archive writes it.
    #[must_use]
    pub fn to_hex(self) -> String {
        written(&self.0)
    }
}

/// Sixty-four lower-case hexadecimal digits, as a hash of thirty-two bytes.
///
/// Upper case, a shorter or longer name, and anything that is not a digit are
/// each refused: a name read loosely is a name that can be made to match.
fn thirty_two(text: &str) -> Option<[u8; LABEL_BYTES]> {
    let digits = text.as_bytes();
    if digits.len() != LABEL_BYTES * 2 {
        return None;
    }
    let mut out = [0; LABEL_BYTES];
    for (byte, pair) in out.iter_mut().zip(digits.chunks_exact(2)) {
        let mut value = 0u8;
        for digit in pair {
            let part = match digit {
                b'0'..=b'9' => digit - b'0',
                b'a'..=b'f' => digit - b'a' + 10,
                _ => return None,
            };
            value = value * 16 + part;
        }
        *byte = value;
    }
    Some(out)
}

/// A hash, as an archive writes it.
fn written(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    out
}

/// How many bytes a root is.
pub const ROOT_BYTES: usize = 32;

/// The root a container names its whole file by: the SHA-256 of every byte of
/// it, in order.
///
/// A label answers for one buffer and the kernel decides on it; a root answers
/// for the file the buffers came in, which is one hash over the whole download
/// and is what a container's manifest names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Root(pub [u8; ROOT_BYTES]);

impl Root {
    /// The root a manifest names, written as sixty-four hexadecimal digits in
    /// lower case, with or without the `sha256:` a container writes before it.
    pub fn from_hex(text: &str) -> Result<Self, DownloadError> {
        let text = text.strip_prefix("sha256:").unwrap_or(text);
        thirty_two(text).map(Self).ok_or(DownloadError::RootNotHex)
    }

    /// The root as a manifest writes it, the `sha256:` included.
    #[must_use]
    pub fn to_hex(self) -> String {
        format!("sha256:{}", written(&self.0))
    }
}

/// The whole of a download, hashed as its bytes arrive.
///
/// The buffers inside a file are labelled one at a time, each by its own
/// [`Arriving`]; this is the one hash that answers for the file they arrived
/// in. Both take the same bytes, once, as they come.
#[derive(Clone, Default)]
pub struct Whole {
    digest: sha2::Sha256,
    received: u64,
}

impl Whole {
    /// A file of nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take in one chunk, as it arrives.
    pub fn take(&mut self, chunk: &[u8]) {
        sha2::Digest::update(&mut self.digest, chunk);
        self.received = self.received.saturating_add(chunk.len() as u64);
    }

    /// How many bytes have arrived.
    #[must_use]
    pub const fn received(&self) -> u64 {
        self.received
    }

    /// The root of everything that has arrived.
    ///
    /// Taking it does not end the download: more may arrive, and the root
    /// moves with it.
    #[must_use]
    pub fn root(&self) -> Root {
        Root(sha2::Digest::finalize(self.digest.clone()).into())
    }
}

impl core::fmt::Debug for Whole {
    /// What has arrived, without the state of the hash, which is not something
    /// to read.
    fn fmt(&self, out: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        out.debug_struct("Whole")
            .field("received", &self.received)
            .finish_non_exhaustive()
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
