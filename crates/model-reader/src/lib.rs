//! Reads a model's own files into the questions the kernel answers.
//!
//! Two documents arrive with every model on a hub: a safetensors file, whose
//! header says where each tensor's bytes lie, and a config, which says what
//! kind of model it is. Neither is trusted: both are bytes someone else wrote.
//!
//! This crate turns them into the kernel's questions and nothing else. It
//! counts, it measures, and it asks. Whether a model may run, and whether a
//! tensor's bytes lie where they claim, are the kernel's decisions, reached
//! through [`hologram_ai_kernel_client`]; no answer of the kernel's is restated
//! here.
//!
//! The kernel itself is the Prism holo built from `app/`. This crate does not
//! run it. Whoever holds the verified holo implements [`Kernel`] by passing the
//! request's bytes to it and handing the answer's bytes back.

#![forbid(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod config;
pub mod header;
pub mod holk;

/// Whoever holds the verified kernel holo.
///
/// One method, because the kernel has one interface: a request's bytes in, an
/// answer's bytes out. An implementation that answers anything other than what
/// the holo answers is not the kernel, and the reader will report its answers
/// as refusals rather than guess at them.
pub trait Kernel {
    /// Put `request` to the kernel and return its answer.
    fn ask(&mut self, request: &[u8]) -> Vec<u8>;
}
