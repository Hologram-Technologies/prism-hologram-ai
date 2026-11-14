//! The engine as one WebAssembly module, for the page to call.
//!
//! The page already runs the kernel: that is the holo's own guest, verified
//! before it is compiled. What the page could not do is everything around the
//! kernel — read the tokeniser a model carries, count the tokens of a turn,
//! label the bytes of a download as they arrive. Those are this repository's
//! crates, and this module is the one place they are offered to JavaScript, so
//! that the page parses no model file of its own and no rule is written twice.
//!
//! # The interface
//!
//! Bytes in linear memory, and nothing else. [`allocate`] reserves a buffer and
//! answers where it is; the page writes its bytes there and hands back the same
//! offset. A call reads that buffer and leaves its answer in one result buffer,
//! which [`result_at`] and [`result_len`] describe, until the next call
//! replaces it. [`release`] gives a buffer back.
//!
//! Every call answers `0` for yes and `1` for no. A call that answers no leaves
//! the failure in the result buffer, as the name of the registered failure it
//! is: this module's own, which [`CallError`] lists, or the one the crate
//! behind the call returned.
//!
//! ```text
//! const at = instance.exports.allocate(bytes.length);
//! new Uint8Array(instance.exports.memory.buffer).set(bytes, at);
//! const no = instance.exports.tokeniser_read(at, bytes.length);
//! instance.exports.release(at);
//! ```
//!
//! # Why there is no `unsafe` here
//!
//! A buffer is a `Vec<u8>` this module keeps, and the page is told where it
//! lives; nothing reconstructs a `Vec` from a pointer, and nothing reads memory
//! the module did not allocate — a call that names a buffer this module does
//! not hold is refused rather than followed. Exporting a symbol is the one
//! unsafe thing here, because a module that exports nothing cannot be called,
//! and each export says so. There is no `unsafe` block, function or trait
//! implementation in this crate.

#![deny(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::cell::RefCell;
use std::collections::BTreeMap;

use hologram_ai_download::{Arriving, Label, LABEL_BYTES};
use hologram_ai_tokenizer::Tokeniser;

/// The call was answered; the result buffer holds the answer.
const OK: u32 = 0;
/// The call was not answered; the result buffer holds the failure's name.
const NO: u32 = 1;

/// How many bytes one token's number takes in a list of tokens.
const TOKEN_BYTES: usize = 4;

/// Why a call into this module could not be answered by the module itself.
///
/// A call can also fail because the crate behind it refused; that failure is
/// the crate's own, reported by its own name, and is not one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallError {
    /// `FL-26`: the page named a buffer this module did not allocate. Reading
    /// memory this module does not hold is the one thing a module like this
    /// must never be talked into.
    NoBuffer,
    /// `FL-27`: the bytes handed in where text was asked for are not UTF-8.
    NotText,
    /// `FL-28`: a tokeniser or a download was used before one was started.
    NothingRead,
    /// `FL-29`: a label handed in is not thirty-two bytes.
    WrongSize,
}

/// What the module holds between calls.
#[derive(Default)]
struct State {
    /// The buffers the page has been given, by where they are.
    buffers: BTreeMap<u32, Vec<u8>>,
    /// The last answer, until the next call replaces it.
    result: Vec<u8>,
    /// The tokeniser last read, if one was.
    tokeniser: Option<Tokeniser>,
    /// The download being labelled, if one was started.
    arriving: Option<Arriving>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Where a buffer of `len` bytes is, for the page to write into.
///
/// The offset is this module's to give: a call may only name an offset that
/// came from here.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn allocate(len: u32) -> u32 {
    STATE.with_borrow_mut(|state| {
        let buffer = vec![0u8; len as usize];
        let at = address(&buffer);
        state.buffers.insert(at, buffer);
        at
    })
}

/// Give a buffer back.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn release(at: u32) -> u32 {
    // The last answer is left where it is: a page gives a buffer back as soon
    // as the call that read it is done, and the answer to that call is what it
    // is about to read.
    STATE.with_borrow_mut(|state| {
        if state.buffers.remove(&at).is_some() {
            OK
        } else {
            failed(state, &format!("{:?}", CallError::NoBuffer))
        }
    })
}

/// Where the last answer is.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn result_at() -> u32 {
    STATE.with_borrow(|state| address(&state.result))
}

/// How many bytes the last answer is.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn result_len() -> u32 {
    STATE.with_borrow(|state| state.result.len() as u32)
}

/// Read the tokeniser a model carries in its own `tokenizer.json`.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn tokeniser_read(at: u32, len: u32) -> u32 {
    read_tokeniser(at, len, false)
}

/// Read the tokeniser a model carries in its own GGUF header.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn tokeniser_read_gguf(at: u32, len: u32) -> u32 {
    read_tokeniser(at, len, true)
}

/// How many tokens the tokeniser read knows, or none if none was read.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn tokeniser_size() -> u32 {
    STATE.with_borrow_mut(|state| {
        state.result.clear();
        match &state.tokeniser {
            Some(tokeniser) => tokeniser.len() as u32,
            None => 0,
        }
    })
}

/// The tokens of a turn: each token's number, four bytes, least significant
/// first.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn tokeniser_encode(at: u32, len: u32) -> u32 {
    answer(|state| {
        let text = text_of(state, at, len)?;
        let tokeniser = state.tokeniser.as_ref().ok_or(CallError::NothingRead)?;
        let tokens = tokeniser.encode(&text);
        let mut out = Vec::with_capacity(tokens.len() * TOKEN_BYTES);
        for token in tokens {
            out.extend_from_slice(&token.to_le_bytes());
        }
        Ok(out)
    })
}

/// How many tokens a turn is, as eight bytes, least significant first.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn tokeniser_count(at: u32, len: u32) -> u32 {
    answer(|state| {
        let text = text_of(state, at, len)?;
        let tokeniser = state.tokeniser.as_ref().ok_or(CallError::NothingRead)?;
        Ok(tokeniser.count(&text).to_le_bytes().to_vec())
    })
}

/// The bytes a list of tokens stands for, each token four bytes, least
/// significant first.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn tokeniser_decode(at: u32, len: u32) -> u32 {
    STATE.with_borrow_mut(|state| {
        let ids = match taken(state, at, len) {
            Ok(bytes) if bytes.len() % TOKEN_BYTES == 0 => bytes
                .chunks_exact(TOKEN_BYTES)
                .map(|four| u32::from_le_bytes([four[0], four[1], four[2], four[3]]))
                .collect::<Vec<u32>>(),
            Ok(_) => return failed(state, &format!("{:?}", CallError::WrongSize)),
            Err(why) => return failed(state, &format!("{why:?}")),
        };
        let Some(tokeniser) = state.tokeniser.as_ref() else {
            return failed(state, &format!("{:?}", CallError::NothingRead));
        };
        match tokeniser.decode(&ids) {
            Ok(bytes) => {
                state.result = bytes;
                OK
            }
            Err(why) => failed(state, &format!("{why:?}")),
        }
    })
}

/// Start labelling a download.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn download_start() -> u32 {
    STATE.with_borrow_mut(|state| {
        state.result.clear();
        state.arriving = Some(Arriving::new());
        OK
    })
}

/// Take the next bytes of the download as they arrive.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn download_take(at: u32, len: u32) -> u32 {
    answer(|state| {
        let chunk = taken(state, at, len)?;
        let arriving = state.arriving.as_mut().ok_or(CallError::NothingRead)?;
        arriving.take(&chunk);
        Ok(Vec::new())
    })
}

/// The label of what has arrived so far: thirty-two bytes.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn download_label() -> u32 {
    answer(|state| {
        let arriving = state.arriving.as_ref().ok_or(CallError::NothingRead)?;
        Ok(arriving.label().0.to_vec())
    })
}

/// The request that asks the kernel whether this buffer may be used: the label
/// the archive names is the buffer handed in, thirty-two bytes, and the length
/// it declares is `declared`.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn download_question(at: u32, len: u32, declared: u64) -> u32 {
    answer(|state| {
        let expected = taken(state, at, len)?;
        let expected: [u8; LABEL_BYTES] = expected.try_into().map_err(|_| CallError::WrongSize)?;
        let arriving = state.arriving.as_ref().ok_or(CallError::NothingRead)?;
        Ok(arriving.question(Label(expected), declared).to_vec())
    })
}

/// Hand the module a buffer holding `bytes`, and say where it is.
///
/// What a page does instead is call [`allocate`] and write the bytes into
/// linear memory itself; a caller that is not a page cannot reach that memory
/// without a raw pointer, and nothing here takes one.
#[must_use]
pub fn given(bytes: &[u8]) -> u32 {
    let at = allocate(bytes.len() as u32);
    STATE.with_borrow_mut(|state| {
        if let Some(buffer) = state.buffers.get_mut(&at) {
            buffer.copy_from_slice(bytes);
        }
    });
    at
}

/// What the result buffer holds, for a caller that is not a page.
///
/// A page reads the bytes at [`result_at`]; this hands back a copy of the
/// same bytes.
#[must_use]
pub fn last_answer() -> Vec<u8> {
    STATE.with_borrow(|state| state.result.clone())
}

/// Read a tokeniser from one of the two files a model carries it in.
fn read_tokeniser(at: u32, len: u32, gguf: bool) -> u32 {
    STATE.with_borrow_mut(|state| {
        let bytes = match taken(state, at, len) {
            Ok(bytes) => bytes,
            Err(why) => return failed(state, &format!("{why:?}")),
        };
        let read = if gguf {
            Tokeniser::from_gguf(&bytes)
        } else {
            Tokeniser::read(&bytes)
        };
        match read {
            Ok(tokeniser) => {
                state.tokeniser = Some(tokeniser);
                state.result.clear();
                OK
            }
            Err(why) => failed(state, &format!("{why:?}")),
        }
    })
}

/// Where a buffer is, for the page to read and write.
///
/// Only meaningful inside this module's own memory, which is where it is used:
/// the page is handed the offset and nothing else.
fn address(buffer: &[u8]) -> u32 {
    buffer.as_ptr() as usize as u32
}

/// A copy of the bytes the page wrote into one of this module's buffers.
fn taken(state: &State, at: u32, len: u32) -> Result<Vec<u8>, CallError> {
    let buffer = state.buffers.get(&at).ok_or(CallError::NoBuffer)?;
    buffer
        .get(..len as usize)
        .map(<[u8]>::to_vec)
        .ok_or(CallError::WrongSize)
}

/// The text the page wrote into one of this module's buffers.
fn text_of(state: &State, at: u32, len: u32) -> Result<String, CallError> {
    let bytes = taken(state, at, len)?;
    String::from_utf8(bytes).map_err(|_| CallError::NotText)
}

/// Run a call that fails only as this module does, and keep its answer.
fn answer(call: impl FnOnce(&mut State) -> Result<Vec<u8>, CallError>) -> u32 {
    STATE.with_borrow_mut(|state| match call(state) {
        Ok(bytes) => {
            state.result = bytes;
            OK
        }
        Err(why) => failed(state, &format!("{why:?}")),
    })
}

/// Leave a failure's name where the page reads answers, and say no.
fn failed(state: &mut State, why: &str) -> u32 {
    state.result = why.as_bytes().to_vec();
    NO
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the last call answered.
    fn said() -> Vec<u8> {
        last_answer()
    }

    /// `FL-26`: a buffer this module did not allocate is not read.
    #[test]
    fn a_buffer_the_module_does_not_hold_is_refused() {
        let at = given(b"{}");
        release(at);
        assert_eq!(tokeniser_read(at, 2), NO);
        assert_eq!(said(), b"NoBuffer");
        assert_eq!(release(at), NO, "and giving it back twice is refused too");
    }

    /// `FL-27`: bytes that are not text are not counted as if they were.
    #[test]
    fn bytes_that_are_not_text_are_refused() {
        let at = given(&[0xff, 0xfe]);
        assert_eq!(tokeniser_count(at, 2), NO);
        assert_eq!(said(), b"NotText");
    }

    /// `FL-28`: nothing is counted or labelled before there is something to
    /// count or label with.
    #[test]
    fn a_tokeniser_or_a_download_is_started_before_it_is_used() {
        STATE.with_borrow_mut(|state| {
            state.tokeniser = None;
            state.arriving = None;
        });
        let at = given(b"hello");
        assert_eq!(tokeniser_count(at, 5), NO);
        assert_eq!(said(), b"NothingRead");
        assert_eq!(download_label(), NO);
        assert_eq!(said(), b"NothingRead");
    }

    /// `FL-29`: a label that is not thirty-two bytes is not read as one.
    #[test]
    fn a_label_of_the_wrong_size_is_refused() {
        download_start();
        let at = given(&[0u8; 31]);
        assert_eq!(download_question(at, 31, 0), NO);
        assert_eq!(said(), b"WrongSize");
    }
}
