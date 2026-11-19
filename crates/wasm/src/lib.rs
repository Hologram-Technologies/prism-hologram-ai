//! The engine as one WebAssembly module, for the page to call.
//!
//! The page already runs the kernel: that is the holo's own guest, verified
//! before it is compiled. What the page could not do is everything around the
//! kernel — read the header of a weight container, read the tokeniser a model
//! carries, count the tokens of a turn, label and root the bytes of a download
//! as they arrive. Those are this repository's
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
//! not hold is refused rather than followed. Exporting a symbol is unsafe,
//! because a module that exports nothing cannot be called, and each export
//! says so. There is one `unsafe` block: the call to the kernel the page runs,
//! `kernel.ask`, which is handed two buffers the calling function holds and
//! names no other memory. There is no `unsafe` function or trait
//! implementation in this crate.

#![deny(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::cell::RefCell;
use std::collections::BTreeMap;

use hologram_ai_cpu_encode as encode;
use hologram_ai_download::{Arriving, Label, Whole, LABEL_BYTES};
use hologram_ai_exact::{request_product, Dense, Layout, Q4Scales};
use hologram_ai_model_reader::holk::Container;
use hologram_ai_model_reader::onnx::{DataFile, File, Graph, ImportError, Imported, ReadFile};
use hologram_ai_model_reader::Kernel;
use hologram_ai_tokenizer::{Tokeniser, WordPiece};

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
    /// `FL-49`: the ONNX importer refused, named as `ImportError` names it.
    Import(ImportError),
    /// `FL-74`: an encoder could not be read from the weights handed over, or
    /// could not run, named as `EncodeError` names it.
    Encode(encode::EncodeError),
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
    /// The block being labelled, if a download was started.
    arriving: Option<Arriving>,
    /// The whole file being rooted, if a download was started.
    whole: Option<Whole>,
    /// The container whose header was read, if one was.
    container: Option<Container>,
    /// Files being named by their BLAKE3, by the number `hash_start` gave.
    hashing: BTreeMap<u32, blake3::Hasher>,
    /// The number the next `hash_start` gives.
    next_hash: u32,
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

/// The id the model stops on, as its own file names it: four bytes, least
/// significant first, or nothing at all where the file names none.
///
/// A page that stops on a list of its own is a page that keeps talking when
/// the list and the model disagree.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn tokeniser_stops_at() -> u32 {
    answer(|state| {
        let tokeniser = state.tokeniser.as_ref().ok_or(CallError::NothingRead)?;
        Ok(tokeniser
            .stops_at()
            .map(|id| id.to_le_bytes().to_vec())
            .unwrap_or_default())
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

/// Start a download: a file to root, and its first block to label.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn download_start() -> u32 {
    STATE.with_borrow_mut(|state| {
        state.result.clear();
        state.arriving = Some(Arriving::new());
        state.whole = Some(Whole::new());
        OK
    })
}

/// Start the next block of the same download.
///
/// A container names each buffer it carries by its own label and the file they
/// arrived in by one root, so the label begins again here and the root does
/// not.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn download_block() -> u32 {
    answer(|state| {
        if state.whole.is_none() {
            return Err(CallError::NothingRead);
        }
        state.arriving = Some(Arriving::new());
        Ok(Vec::new())
    })
}

/// A q4 matrix with `f32` scales times a vector, every group sum exact.
///
/// The buffer is the row count and the column count, four bytes each, least
/// significant first; the input, one little-endian `f32` per column; then the
/// tensor's bytes as the engine stores them: the codes, then the scales. The
/// answer is one little-endian `f32` per row, the same bits on every machine.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn exact_q4(at: u32, len: u32) -> u32 {
    exact_product(at, len, Layout::Q4F32)
}

/// As [`exact_q4`], with the scales stored as IEEE halves.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn exact_q4f16(at: u32, len: u32) -> u32 {
    exact_product(at, len, Layout::Q4F16)
}

/// As [`exact_q4`], for a ternary tensor: its sign plane, its nonzero plane,
/// then its `f32` scales.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn exact_ternary(at: u32, len: u32) -> u32 {
    exact_product(at, len, Layout::Ternary)
}

/// Take the next bytes of the download as they arrive, into both hashes.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn download_take(at: u32, len: u32) -> u32 {
    answer(|state| {
        let chunk = taken(state, at, len)?;
        let arriving = state.arriving.as_mut().ok_or(CallError::NothingRead)?;
        arriving.take(&chunk);
        let whole = state.whole.as_mut().ok_or(CallError::NothingRead)?;
        whole.take(&chunk);
        Ok(Vec::new())
    })
}

/// The label of the block so far: thirty-two bytes.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn download_label() -> u32 {
    answer(|state| {
        let arriving = state.arriving.as_ref().ok_or(CallError::NothingRead)?;
        Ok(arriving.label().0.to_vec())
    })
}

/// The root of the whole file so far: thirty-two bytes.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn download_root() -> u32 {
    answer(|state| {
        let whole = state.whole.as_ref().ok_or(CallError::NothingRead)?;
        Ok(whole.root().0.to_vec())
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

/// Read the header of a weight container.
///
/// The bytes must reach the end of the header, which the first ten of them
/// say: `container_header` answers that from those ten alone, so a page can
/// ask for the header before it has the file.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_read(at: u32, len: u32) -> u32 {
    STATE.with_borrow_mut(|state| {
        let bytes = match taken(state, at, len) {
            Ok(bytes) => bytes,
            Err(why) => return failed(state, &format!("{why:?}")),
        };
        match Container::read(&bytes) {
            Ok(container) => {
                state.container = Some(container);
                state.result.clear();
                OK
            }
            Err(why) => failed(state, &format!("{why:?}")),
        }
    })
}

/// Where a container's header lies, from the start of its file: two eight-byte
/// numbers, least significant first.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_header(at: u32, len: u32) -> u32 {
    STATE.with_borrow_mut(|state| {
        let bytes = match taken(state, at, len) {
            Ok(bytes) => bytes,
            Err(why) => return failed(state, &format!("{why:?}")),
        };
        match Container::header_span(&bytes) {
            Ok(span) => {
                state.result = [span.at.to_le_bytes(), span.len.to_le_bytes()].concat();
                OK
            }
            Err(why) => failed(state, &format!("{why:?}")),
        }
    })
}

/// How many buffers the container read carries, or none if none was read.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_entries() -> u32 {
    STATE.with_borrow_mut(|state| {
        state.result.clear();
        state
            .container
            .as_ref()
            .map_or(0, |container| container.entries.len() as u32)
    })
}

/// Where one buffer lies: two eight-byte numbers, least significant first.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_span(index: u32) -> u32 {
    answer(|state| {
        let entry = entry(state, index)?;
        Ok([entry.span.at.to_le_bytes(), entry.span.len.to_le_bytes()].concat())
    })
}

/// What that buffer is called.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_name(index: u32) -> u32 {
    answer(|state| Ok(entry(state, index)?.name.clone().into_bytes()))
}

/// Where the tokeniser the model carries lies: two eight-byte numbers, least
/// significant first. What lies there is a GGUF header.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_tokeniser() -> u32 {
    answer(|state| {
        let container = state.container.as_ref().ok_or(CallError::NothingRead)?;
        let span = container.tokeniser;
        Ok([span.at.to_le_bytes(), span.len.to_le_bytes()].concat())
    })
}

/// What the container's manifest calls its root: thirty-two bytes.
///
/// Not the hash of the file. `download_root` is that, and a page that
/// compares the two is comparing two different numbers about the same model.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_identity() -> u32 {
    answer(|state| {
        let container = state.container.as_ref().ok_or(CallError::NothingRead)?;
        Ok(container.manifest.identity.0.to_vec())
    })
}

/// The kernel's question about where one buffer lies.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_range_question(index: u32, file_length: u64, previous_end: u64) -> u32 {
    answer(|state| {
        Ok(entry(state, index)?
            .range_question(file_length, previous_end)
            .to_vec())
    })
}

/// The kernel's question about whether this buffer's bytes may be used,
/// against the label of the block being taken now.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_admission_question(index: u32) -> u32 {
    answer(|state| {
        let arriving = state.arriving.as_ref().ok_or(CallError::NothingRead)?;
        let (observed, received) = (arriving.label(), arriving.received());
        Ok(entry(state, index)?
            .admission_question(observed, received)
            .to_vec())
    })
}

/// The kernel's question about how long one block of this buffer is.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn container_block_question(index: u32) -> u32 {
    answer(|state| Ok(entry(state, index)?.block_question().to_vec()))
}

/// One buffer of the container read.
fn entry(state: &State, index: u32) -> Result<&hologram_ai_model_reader::holk::Entry, CallError> {
    let container = state.container.as_ref().ok_or(CallError::NothingRead)?;
    container
        .entries
        .get(index as usize)
        .ok_or(CallError::NothingRead)
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

/// Take an exact product and keep its answer.
fn exact_product(at: u32, len: u32, layout: Layout) -> u32 {
    STATE.with_borrow_mut(|state| {
        let bytes = match taken(state, at, len) {
            Ok(bytes) => bytes,
            Err(why) => return failed(state, &format!("{why:?}")),
        };
        match request_product(&bytes, layout) {
            Ok(rows) => {
                state.result = rows.iter().flat_map(|row| row.to_le_bytes()).collect();
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

// ─── An ONNX export, imported in the page ──────────────────────────────────
//
// The importer of `hologram-ai-model-reader` puts every span, operator and
// 4-bit weight of a model to the kernel as the bytes arrive. In the page the
// kernel is the holo's guest, verified before it is compiled, and it answers
// synchronously; this module asks it through one import the page provides,
// `kernel.ask`, and is handed the answer in a buffer of its own. Nothing here
// decides what the kernel would answer.

/// The most bytes a kernel answer is read as. The kernel answers in words and
/// numbers, a few dozen bytes; a longer answer is cut and then is not the one
/// the importer accepts, so it refuses.
const ANSWER_BYTES: usize = 256;

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "kernel")]
extern "C" {
    /// The page's kernel: the request at `request`, `length` bytes; the answer
    /// written at `answer`, at most `capacity` bytes; the answer's length.
    #[link_name = "ask"]
    fn page_ask(request: u32, length: u32, answer: u32, capacity: u32) -> u32;
}

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    /// Outside a browser there is no page; a test installs the kernel the
    /// importer asks.
    static NATIVE_KERNEL: RefCell<Option<Box<dyn Kernel>>> = RefCell::new(None);
}

/// Install the kernel the importer asks, outside a browser. In the page it is
/// the holo's guest, reached through the `kernel.ask` import.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_kernel(kernel: Box<dyn Kernel>) {
    NATIVE_KERNEL.with_borrow_mut(|installed| *installed = Some(kernel));
}

/// The kernel the page runs.
struct PageKernel;

impl Kernel for PageKernel {
    fn ask(&mut self, request: &[u8]) -> Vec<u8> {
        let mut answer = vec![0u8; ANSWER_BYTES];
        #[cfg(target_arch = "wasm32")]
        let length = {
            #[allow(
                unsafe_code,
                reason = "the one way a module reaches the kernel the page runs"
            )]
            // SAFETY: the import is called with the address and length of
            // `request` and of `answer`, two buffers this function holds for
            // the whole call; the page writes at most `capacity` bytes into
            // `answer` and reads nothing else. Nothing is reconstructed from a
            // pointer, and no other memory is named.
            let length = unsafe {
                page_ask(
                    address(request),
                    request.len() as u32,
                    address(&answer),
                    ANSWER_BYTES as u32,
                )
            };
            length as usize
        };
        #[cfg(not(target_arch = "wasm32"))]
        let length = NATIVE_KERNEL.with_borrow_mut(|installed| match installed {
            Some(kernel) => {
                let said = kernel.ask(request);
                let n = said.len().min(ANSWER_BYTES);
                answer[..n].copy_from_slice(&said[..n]);
                said.len()
            }
            None => 0,
        });
        answer.truncate(length.min(ANSWER_BYTES));
        answer
    }
}

/// An ONNX import in progress.
#[derive(Default)]
struct OnnxImport {
    graph: Option<Graph>,
    imported: Option<Imported>,
    data: Option<DataFile>,
    read: Vec<ReadFile>,
}

thread_local! {
    static ONNX: RefCell<OnnxImport> = RefCell::new(OnnxImport::default());
}

/// A model file as the record gives it: its name, its SHA-256, its length and
/// its place. A data file is named as the graph names it, since the importer
/// finds each tensor's file by that name.
fn onnx_file(
    state: &State,
    name: String,
    root: (u32, u32),
    index: u8,
    declared: u8,
    length: u64,
) -> Result<File, CallError> {
    let bytes = taken(state, root.0, root.1)?;
    let address: [u8; 32] = bytes.try_into().map_err(|_| CallError::WrongSize)?;
    Ok(File {
        name,
        index,
        declared,
        length,
        address: Label(address),
    })
}

/// Start importing a graph file of `length` bytes whose SHA-256 is the 32
/// bytes at `at`, with `declared` data files beside it.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn onnx_start(at: u32, len: u32, length: u64, declared: u32) -> u32 {
    answer(|state| {
        let declared = u8::try_from(declared).map_err(|_| CallError::WrongSize)?;
        let file = onnx_file(state, "graph".into(), (at, len), 0, declared, length)?;
        ONNX.with_borrow_mut(|onnx| {
            *onnx = OnnxImport {
                graph: Some(Graph::new(file)),
                ..OnnxImport::default()
            };
        });
        Ok(Vec::new())
    })
}

/// The next bytes of the graph file, in order.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn onnx_take(at: u32, len: u32) -> u32 {
    answer(|state| {
        let chunk = taken(state, at, len)?;
        ONNX.with_borrow_mut(|onnx| {
            let graph = onnx.graph.as_mut().ok_or(CallError::NothingRead)?;
            graph
                .take(&chunk, &mut PageKernel)
                .map_err(CallError::Import)
        })?;
        Ok(Vec::new())
    })
}

/// The graph file has ended: the data files its tensors lie in, by name, one
/// to a line, which the page reads next in that order.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn onnx_graph_finish() -> u32 {
    answer(|_| {
        ONNX.with_borrow_mut(|onnx| {
            let graph = onnx.graph.take().ok_or(CallError::NothingRead)?;
            let imported = graph.finish(&mut PageKernel).map_err(CallError::Import)?;
            let files = imported.data_files().join("\n").into_bytes();
            onnx.imported = Some(imported);
            Ok(files)
        })
    })
}

/// Start the data file at `index` (from 1), `length` bytes, SHA-256 at `at`.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn onnx_data_start(at: u32, len: u32, index: u32, length: u64) -> u32 {
    answer(|state| {
        let index = u8::try_from(index).map_err(|_| CallError::WrongSize)?;
        ONNX.with_borrow_mut(|onnx| {
            let imported = onnx.imported.as_ref().ok_or(CallError::NothingRead)?;
            let files = imported.data_files();
            let declared = u8::try_from(files.len()).map_err(|_| CallError::WrongSize)?;
            // The data files are read in the order the graph names them, from 1.
            let name = usize::from(index)
                .checked_sub(1)
                .and_then(|at| files.get(at))
                .cloned()
                .ok_or(CallError::NothingRead)?;
            let file = onnx_file(state, name, (at, len), index, declared, length)?;
            onnx.data = Some(
                imported
                    .data(file, &mut PageKernel)
                    .map_err(CallError::Import)?,
            );
            Ok(Vec::new())
        })
    })
}

/// The next bytes of the data file being read.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn onnx_data_take(at: u32, len: u32) -> u32 {
    answer(|state| {
        let chunk = taken(state, at, len)?;
        ONNX.with_borrow_mut(|onnx| {
            onnx.data
                .as_mut()
                .ok_or(CallError::NothingRead)?
                .take(&chunk);
            Ok(Vec::new())
        })
    })
}

/// The data file being read has ended.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn onnx_data_finish() -> u32 {
    answer(|_| {
        ONNX.with_borrow_mut(|onnx| {
            let data = onnx.data.take().ok_or(CallError::NothingRead)?;
            let read = data.finish(&mut PageKernel).map_err(CallError::Import)?;
            onnx.read.push(read);
            Ok(Vec::new())
        })
    })
}

/// The element type's name, as the role map reads it.
fn element(data_type: u64) -> String {
    match data_type {
        1 => "f32".into(),
        2 => "u8".into(),
        3 => "i8".into(),
        6 => "i32".into(),
        7 => "i64".into(),
        10 => "f16".into(),
        16 => "bf16".into(),
        other => format!("t{other}"),
    }
}

/// Every file has ended: the model, as JSON the page's role map reads. Its
/// nodes in file order, each an operator, a name, the inputs that are tensors
/// of the model (others null) and the whole-number and real attributes; and
/// every tensor's type and shape, and where its bytes lie: the file, the
/// offset, the length and the address the importer gave them.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn onnx_finish() -> u32 {
    answer(|_| {
        ONNX.with_borrow_mut(|onnx| {
            let imported = onnx.imported.take().ok_or(CallError::NothingRead)?;
            let nodes = imported.nodes().to_vec();
            let read = std::mem::take(&mut onnx.read);
            let model = imported
                .finish(read, &mut PageKernel)
                .map_err(CallError::Import)?;
            let held: BTreeMap<&str, &hologram_ai_model_reader::onnx::Tensor> =
                model.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
            let mut tensors = serde_json::Map::new();
            let mut places = serde_json::Map::new();
            for t in &model.tensors {
                tensors.insert(
                    t.name.clone(),
                    serde_json::json!([element(t.data_type), t.dims]),
                );
                let hex: String = t.address.0.iter().map(|b| format!("{b:02x}")).collect();
                places.insert(
                    t.name.clone(),
                    serde_json::json!([t.file, t.offset, t.length, format!("blake3:{hex}")]),
                );
            }
            let nodes: Vec<serde_json::Value> = nodes
                .iter()
                .map(|node| {
                    let inputs: Vec<serde_json::Value> = node
                        .inputs
                        .iter()
                        .map(|input| {
                            if held.contains_key(input.as_str()) {
                                serde_json::json!(input)
                            } else {
                                serde_json::Value::Null
                            }
                        })
                        .collect();
                    let mut attributes = serde_json::Map::new();
                    for (name, value) in &node.ints {
                        attributes.insert(name.clone(), serde_json::json!(value));
                    }
                    for (name, value) in &node.floats {
                        attributes.insert(name.clone(), serde_json::json!(value));
                    }
                    serde_json::json!([node.op_type, node.name, inputs, attributes])
                })
                .collect();
            let out = serde_json::json!({
                "nodes": nodes,
                "tensors": tensors,
                "places": places,
                "operators": model.operators,
                "fused": model.fused,
            });
            serde_json::to_vec(&out).map_err(|_| CallError::NotText)
        })
    })
}

/// Start naming a file by its BLAKE3, the address Hologram AI names
/// everything by. Answers the number the next calls name this file by, four
/// bytes little-endian, so the bytes of several files may arrive interleaved.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn hash_start() -> u32 {
    answer(|state| {
        let id = state.next_hash;
        state.next_hash = state.next_hash.wrapping_add(1);
        state.hashing.insert(id, blake3::Hasher::new());
        Ok(id.to_le_bytes().to_vec())
    })
}

/// The next bytes of the file numbered `id`, in order.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn hash_take(at: u32, len: u32, id: u32) -> u32 {
    answer(|state| {
        let chunk = taken(state, at, len)?;
        state
            .hashing
            .get_mut(&id)
            .ok_or(CallError::NothingRead)?
            .update(&chunk);
        Ok(Vec::new())
    })
}

/// The file numbered `id` has ended: its BLAKE3, thirty-two bytes.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn hash_finish(id: u32) -> u32 {
    answer(|state| {
        let hasher = state.hashing.remove(&id).ok_or(CallError::NothingRead)?;
        Ok(hasher.finalize().as_bytes().to_vec())
    })
}

// ─── An encoder, run in the page (row 9) ────────────────────────────────────

/// One weight as `encode_tensor` took it.
struct Held {
    kind: u8,
    rows: usize,
    columns: usize,
    /// The codes of a 4-bit matrix, or a dense weight's values.
    first: Vec<u8>,
    /// A 4-bit matrix's scales; empty for a dense weight.
    second: Vec<u8>,
}

/// An encoder's weights as the page hands them over, by role, and its config.
#[derive(Default)]
struct Encoder {
    tensors: BTreeMap<String, Held>,
    /// The encoder's WordPiece tokeniser, once `wordpiece_read` has read it.
    wordpiece: Option<WordPiece>,
    /// What the weights do not say, once `encode_open` has read the config.
    config: Option<encode::Config>,
}

thread_local! {
    static ENCODE: RefCell<Encoder> = RefCell::new(Encoder::default());
}

/// Start importing an encoder's graph file: as `onnx_start`, but its
/// operators are asked of decision 18's encoder style (KN-52), since an
/// encoder's 4-bit weights name the fused domain as a decoder's do.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn onnx_start_encoder(at: u32, len: u32, length: u64, declared: u32) -> u32 {
    answer(|state| {
        let declared = u8::try_from(declared).map_err(|_| CallError::WrongSize)?;
        let file = onnx_file(state, "graph".into(), (at, len), 0, declared, length)?;
        ONNX.with_borrow_mut(|onnx| {
            *onnx = OnnxImport {
                graph: Some(Graph::encoder(file)),
                ..OnnxImport::default()
            };
        });
        ENCODE.with_borrow_mut(|e| {
            e.tensors.clear();
            e.config = None;
        });
        Ok(Vec::new())
    })
}

/// One weight of an encoder, by the role the page placed it in
/// (`encoderRoles`, GP-17): four bytes of name length, the name, one byte of
/// kind (0 a 4-bit matrix with half scales, 1 with binary32 scales, 2 dense
/// halves, 3 dense binary32, 4 dense bfloat16), four bytes each of rows and
/// columns, four bytes of the first part's length, the first part (the codes,
/// or the values), then the scales; all little-endian.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn encode_tensor(at: u32, len: u32) -> u32 {
    answer(|state| {
        let given = taken(state, at, len)?;
        let word = |at: usize| -> Result<usize, CallError> {
            let bytes: [u8; 4] = given
                .get(at..at + 4)
                .and_then(|b| b.try_into().ok())
                .ok_or(CallError::WrongSize)?;
            Ok(u32::from_le_bytes(bytes) as usize)
        };
        let named = word(0)?;
        let name = given.get(4..4 + named).ok_or(CallError::WrongSize)?;
        let name = String::from_utf8(name.to_vec()).map_err(|_| CallError::NotText)?;
        let at = 4 + named;
        let kind = *given.get(at).ok_or(CallError::WrongSize)?;
        if kind > 4 {
            return Err(CallError::WrongSize);
        }
        let (rows, columns, first) = (word(at + 1)?, word(at + 5)?, word(at + 9)?);
        let first = given
            .get(at + 13..at + 13 + first)
            .ok_or(CallError::WrongSize)?
            .to_vec();
        let second = given
            .get(at + 13 + first.len()..)
            .unwrap_or_default()
            .to_vec();
        ENCODE.with_borrow_mut(|e| {
            e.tensors.insert(
                name,
                Held {
                    kind,
                    rows,
                    columns,
                    first,
                    second,
                },
            );
        });
        Ok(Vec::new())
    })
}

/// A number of `fields` that is a whole number, read exactly from its text.
fn whole_field(fields: &serde_json::Value, keys: &[&str]) -> Result<usize, CallError> {
    keys.iter()
        .find_map(|key| fields.get(*key))
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(CallError::NotText)
}

/// LayerNorm's epsilon in units of `2^-32`, rounded half to even, from the
/// decimal text of the config's number: `m * 10^e` is read as integers, so no
/// float lies on the way. A model whose config names none (DistilBERT's) is
/// at transformers' default, 1e-12, which is 0 in these units.
fn epsilon_units(text: Option<&str>) -> Result<u64, CallError> {
    let Some(text) = text else {
        return Ok(0);
    };
    let (mantissa, exponent) = match text.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().map_err(|_| CallError::NotText)?),
        None => (text, 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(CallError::NotText);
    }
    let m: u128 = digits.parse().map_err(|_| CallError::WrongSize)?;
    let shift = exponent - i32::try_from(fraction.len()).map_err(|_| CallError::WrongSize)?;
    let scaled = m
        .checked_shl(32)
        .filter(|s| s >> 32 == m)
        .ok_or(CallError::WrongSize)?;
    let units = if shift >= 0 {
        10u128
            .checked_pow(shift.unsigned_abs())
            .and_then(|p| scaled.checked_mul(p))
            .ok_or(CallError::WrongSize)?
    } else {
        match 10u128.checked_pow(shift.unsigned_abs()) {
            // Past 10^38 the value is below half a unit.
            None => 0,
            Some(d) => {
                let (q, r) = (scaled / d, scaled % d);
                let twice = r.checked_mul(2).ok_or(CallError::WrongSize)?;
                if twice > d || (twice == d && q % 2 == 1) {
                    q + 1
                } else {
                    q
                }
            }
        }
    };
    u64::try_from(units).map_err(|_| CallError::WrongSize)
}

/// Read the encoder the held weights make: `config` is the text of the
/// model's config.json. Its head and layer counts are whole numbers, read as
/// BERT (`num_attention_heads`, `num_hidden_layers`) or DistilBERT (`n_heads`,
/// `n_layers`) names them, and `layer_norm_eps` by `epsilon_units`. Answers
/// nothing; a role missing or misshapen is `Encode(Unrecognised(..))` (FL-74).
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn encode_open(at: u32, len: u32) -> u32 {
    answer(|state| {
        let text = text_of(state, at, len)?;
        let fields: serde_json::Value =
            serde_json::from_str(&text).map_err(|_| CallError::NotText)?;
        let eps: BTreeMap<String, Box<serde_json::value::RawValue>> =
            serde_json::from_str(&text).map_err(|_| CallError::NotText)?;
        let config = encode::Config {
            heads: whole_field(&fields, &["num_attention_heads", "n_heads"])?,
            layers: whole_field(&fields, &["num_hidden_layers", "n_layers"])?,
            epsilon: epsilon_units(eps.get("layer_norm_eps").map(|v| v.get()))?,
        };
        ENCODE.with_borrow_mut(|e| {
            with_encoder(e, config, |_| Ok(()))?;
            e.config = Some(config);
            Ok(Vec::new())
        })
    })
}

/// The encoder the held weights make under `config`, lent to `run`.
fn with_encoder<T>(
    e: &Encoder,
    config: encode::Config,
    run: impl FnOnce(&encode::Model<'_>) -> Result<T, CallError>,
) -> Result<T, CallError> {
    let tensors: BTreeMap<String, encode::Tensor<'_>> = e
        .tensors
        .iter()
        .map(|(name, held)| {
            let Held {
                kind,
                rows,
                columns,
                first,
                second,
            } = held;
            let (rows, columns) = (*rows, *columns);
            let tensor = match kind {
                0 | 1 => encode::Tensor::Q4 {
                    codes: first,
                    scales: if *kind == 0 {
                        Q4Scales::F16(second)
                    } else {
                        Q4Scales::F32(second)
                    },
                    rows,
                    columns,
                },
                2 => encode::Tensor::Dense {
                    values: Dense::F16(first),
                    rows,
                    columns,
                },
                3 => encode::Tensor::Dense {
                    values: Dense::F32(first),
                    rows,
                    columns,
                },
                _ => encode::Tensor::Dense {
                    values: Dense::Bf16(first),
                    rows,
                    columns,
                },
            };
            (name.clone(), tensor)
        })
        .collect();
    let model = encode::read(&tensors, config).map_err(CallError::Encode)?;
    run(&model)
}

/// Token ids, four bytes each little-endian, through the encoder `encode_open`
/// read, answering `outputs`' binary32 bits, four bytes each.
fn encode_ids(
    state: &State,
    at: u32,
    len: u32,
    outputs: impl FnOnce(&encode::Model<'_>, &[u32]) -> Result<Vec<u32>, CallError>,
) -> Result<Vec<u8>, CallError> {
    let given = taken(state, at, len)?;
    if given.len() % TOKEN_BYTES != 0 {
        return Err(CallError::WrongSize);
    }
    let ids: Vec<u32> = given
        .chunks_exact(TOKEN_BYTES)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    ENCODE.with_borrow(|e| {
        let config = e.config.ok_or(CallError::NothingRead)?;
        with_encoder(e, config, |model| {
            let bits = outputs(model, &ids)?;
            Ok(bits.iter().flat_map(|b| b.to_le_bytes()).collect())
        })
    })
}

/// Read an encoder's WordPiece tokeniser from its `tokenizer.json`: BERT's
/// and DistilBERT's split text by WordPiece, not by byte pairs, and wrap a
/// sentence in `[CLS]` and `[SEP]` as the file's template says.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn wordpiece_read(at: u32, len: u32) -> u32 {
    STATE.with_borrow_mut(|state| {
        let bytes = match taken(state, at, len) {
            Ok(bytes) => bytes,
            Err(why) => return failed(state, &format!("{why:?}")),
        };
        match WordPiece::read(&bytes) {
            Ok(wordpiece) => {
                ENCODE.with_borrow_mut(|e| e.wordpiece = Some(wordpiece));
                state.result.clear();
                OK
            }
            Err(why) => failed(state, &format!("{why:?}")),
        }
    })
}

/// A sentence's ids by the WordPiece tokeniser read, four bytes each,
/// little-endian.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn wordpiece_encode(at: u32, len: u32) -> u32 {
    answer(|state| {
        let text = text_of(state, at, len)?;
        ENCODE.with_borrow(|e| {
            let wordpiece = e.wordpiece.as_ref().ok_or(CallError::NothingRead)?;
            Ok(wordpiece
                .encode(&text)
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect())
        })
    })
}

/// A sentence's ids to its embedding at unit length (EN-62), as binary32 bits.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn encode_embed(at: u32, len: u32) -> u32 {
    answer(|state| {
        encode_ids(state, at, len, |model, ids| {
            encode::embed(model, ids).map_err(CallError::Encode)
        })
    })
}

/// A sentence's ids to a classifier's label: the index of the first largest
/// score, compared as integers (`hologram_ai_cpu_encode::label`), then every
/// label's score (EN-63) as binary32 bits, all four bytes each.
#[allow(unsafe_code, reason = "a module that exports nothing cannot be called")]
#[no_mangle]
pub extern "C" fn encode_classify(at: u32, len: u32) -> u32 {
    answer(|state| {
        encode_ids(state, at, len, |model, ids| {
            let scores = encode::classify(model, ids).map_err(CallError::Encode)?;
            let label = encode::label(&scores).unwrap_or(u32::MAX);
            Ok(std::iter::once(label).chain(scores).collect())
        })
    })
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
            state.whole = None;
            state.container = None;
        });
        let at = given(b"hello");
        assert_eq!(tokeniser_count(at, 5), NO);
        assert_eq!(said(), b"NothingRead");
        assert_eq!(download_label(), NO);
        assert_eq!(said(), b"NothingRead");
        assert_eq!(download_root(), NO);
        assert_eq!(said(), b"NothingRead");
        assert_eq!(download_block(), NO);
        assert_eq!(said(), b"NothingRead");
    }

    /// A file's BLAKE3 is the same however its bytes arrive, and two files
    /// hashed at once do not mix.
    #[test]
    fn a_file_is_named_by_its_blake3_however_it_arrives() {
        let bytes: Vec<u8> = (0..5000u32).map(|i| (i * 31 % 251) as u8).collect();
        let other = b"another file".to_vec();
        let started = |answer: u32| {
            assert_eq!(answer, OK);
            STATE.with_borrow(|state| {
                u32::from_le_bytes([
                    state.result[0],
                    state.result[1],
                    state.result[2],
                    state.result[3],
                ])
            })
        };
        let a = started(hash_start());
        let b = started(hash_start());
        for part in bytes.chunks(1999) {
            let at = given(part);
            assert_eq!(hash_take(at, part.len() as u32, a), OK);
            let at = given(&other);
            assert_eq!(hash_take(at, 0, b), OK);
        }
        let at = given(&other);
        assert_eq!(hash_take(at, other.len() as u32, b), OK);
        assert_eq!(hash_finish(a), OK);
        assert_eq!(said(), blake3::hash(&bytes).as_bytes());
        assert_eq!(hash_finish(b), OK);
        assert_eq!(said(), blake3::hash(&other).as_bytes());
        assert_eq!(hash_finish(a), NO, "a finished file is not finished twice");
        assert_eq!(said(), b"NothingRead");
        let at = given(&other);
        assert_eq!(hash_take(at, 1, 9_999), NO, "a file never started");
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
