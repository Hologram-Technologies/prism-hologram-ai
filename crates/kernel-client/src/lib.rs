//! Writes requests to the Hologram AI kernel and reads its answers.
//!
//! The kernel is the Prism holo built from `app/`: every decision the
//! application makes about what it may load and run. The engine and the page
//! never decide for themselves; they ask the kernel, through this crate.
//!
//! A request is a fixed binary layout. Byte 0 names the decision and each field
//! after it is an unsigned big-endian number of fixed width. An answer is short
//! text. One module per decision gives a `request` function, which cannot fail,
//! and an `answer` function, which is total: text the kernel is not known to
//! give reads as `Unrecognised`, so a caller must handle it and cannot mistake
//! it for a verdict.
//!
//! This crate does not run the kernel. Whoever holds the verified holo passes
//! the request bytes to it and hands the answer bytes back.

#![no_std]
#![forbid(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

/// What the kernel answers to a request it does not accept: the wrong length
/// for its decision, a field out of range, or a decision that does not exist.
pub const INVALID_REQUEST: &[u8] = b"invalid request";

/// Write `number` as eight big-endian bytes at `at`.
fn put(out: &mut [u8], at: usize, number: u64) {
    out[at..at + 8].copy_from_slice(&number.to_be_bytes());
}

/// The number a canonical decimal text names: digits only, no leading zero
/// unless the number is zero, and within 64 bits.
fn decimal(text: &[u8]) -> Option<u64> {
    if text.is_empty() || text.len() > 20 || (text.len() > 1 && text[0] == b'0') {
        return None;
    }
    let mut number: u64 = 0;
    for &byte in text {
        if !byte.is_ascii_digit() {
            return None;
        }
        number = number
            .checked_mul(10)?
            .checked_add(u64::from(byte - b'0'))?;
    }
    Some(number)
}

/// Decision 1: the next step of the journey.
pub mod journey {
    /// Where the application is.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum State {
        /// Nothing is loaded.
        Idle,
        /// A model is being checked before any weight is downloaded.
        Preflight,
        /// Weights are streaming into the store.
        Download,
        /// The archive is being compiled.
        Compile,
        /// Weights are being resolved and their labels verified.
        Materialize,
        /// A model is loaded and waiting.
        Ready,
        /// A turn is being generated.
        Generating,
        /// Something failed; only a reset leaves this state.
        Failed,
    }

    impl State {
        /// The byte the kernel knows this state by.
        #[must_use]
        pub const fn code(self) -> u8 {
            match self {
                Self::Idle => 0,
                Self::Preflight => 1,
                Self::Download => 2,
                Self::Compile => 3,
                Self::Materialize => 4,
                Self::Ready => 5,
                Self::Generating => 6,
                Self::Failed => 7,
            }
        }

        /// The state a byte names, if it names one.
        #[must_use]
        pub const fn from_code(code: u8) -> Option<Self> {
            Some(match code {
                0 => Self::Idle,
                1 => Self::Preflight,
                2 => Self::Download,
                3 => Self::Compile,
                4 => Self::Materialize,
                5 => Self::Ready,
                6 => Self::Generating,
                7 => Self::Failed,
                _ => return None,
            })
        }
    }

    /// What happened.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Event {
        /// The user named a model.
        Start,
        /// The preflight verdict allowed the model.
        PreflightPassed,
        /// Every weight has streamed in.
        DownloadDone,
        /// The archive is compiled.
        CompileDone,
        /// Every label matched.
        LabelsVerified,
        /// The user sent a message.
        Send,
        /// The turn ended by itself.
        TurnDone,
        /// The user cancelled the turn.
        CancelTurn,
        /// Something failed, or the preflight rejected the model.
        Fail,
        /// Start over: clear a failure, or choose another model.
        Reset,
    }

    impl Event {
        /// The byte the kernel knows this event by.
        #[must_use]
        pub const fn code(self) -> u8 {
            match self {
                Self::Start => 1,
                Self::PreflightPassed => 2,
                Self::DownloadDone => 3,
                Self::CompileDone => 4,
                Self::LabelsVerified => 5,
                Self::Send => 6,
                Self::TurnDone => 7,
                Self::CancelTurn => 8,
                Self::Fail => 9,
                Self::Reset => 10,
            }
        }

        /// The event a byte names, if it names one.
        #[must_use]
        pub const fn from_code(code: u8) -> Option<Self> {
            Some(match code {
                1 => Self::Start,
                2 => Self::PreflightPassed,
                3 => Self::DownloadDone,
                4 => Self::CompileDone,
                5 => Self::LabelsVerified,
                6 => Self::Send,
                7 => Self::TurnDone,
                8 => Self::CancelTurn,
                9 => Self::Fail,
                10 => Self::Reset,
                _ => return None,
            })
        }
    }

    /// The kernel's answer to a step.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Step {
        /// The step is allowed and the application is now in this state.
        To(State),
        /// The step is not allowed; the state does not change.
        Refused,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request that asks what `event` does in `state`.
    #[must_use]
    pub const fn request(state: State, event: Event) -> [u8; 3] {
        [1, state.code(), event.code()]
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Step {
        match text {
            b"idle" => Step::To(State::Idle),
            b"preflight" => Step::To(State::Preflight),
            b"download" => Step::To(State::Download),
            b"compile" => Step::To(State::Compile),
            b"materialize" => Step::To(State::Materialize),
            b"ready" => Step::To(State::Ready),
            b"generating" => Step::To(State::Generating),
            b"failed" => Step::To(State::Failed),
            b"refused" => Step::Refused,
            crate::INVALID_REQUEST => Step::Invalid,
            _ => Step::Unrecognised,
        }
    }
}

/// Decision 2: whether a model may proceed.
pub mod preflight {
    /// The kernel's verdict on a model.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Verdict {
        /// Runnable, and its family is recognised.
        Supported,
        /// Runnable, built from the generic recipe.
        Derived,
        /// Rejected: a required config key is missing.
        MissingConfigKey,
        /// Rejected: the model uses a feature that is not supported.
        UnsupportedFeature,
        /// Rejected: the weights do not match the recipe.
        ManifestDoesNotFit,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    impl Verdict {
        /// Whether weights may be downloaded. True for the two runnable
        /// verdicts and for nothing else.
        #[must_use]
        pub const fn authorises_download(self) -> bool {
            matches!(self, Self::Supported | Self::Derived)
        }
    }

    /// The request for a model with these four findings.
    #[must_use]
    pub const fn request(
        family_recognised: bool,
        missing_keys: u8,
        manifest_fits: bool,
        unsupported_features: u8,
    ) -> [u8; 5] {
        [
            2,
            family_recognised as u8,
            missing_keys,
            manifest_fits as u8,
            unsupported_features,
        ]
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Verdict {
        match text {
            b"supported" => Verdict::Supported,
            b"derived" => Verdict::Derived,
            b"rejected: missing config key" => Verdict::MissingConfigKey,
            b"rejected: unsupported feature" => Verdict::UnsupportedFeature,
            b"rejected: manifest does not fit" => Verdict::ManifestDoesNotFit,
            crate::INVALID_REQUEST => Verdict::Invalid,
            _ => Verdict::Unrecognised,
        }
    }
}

/// Decision 3: whether a proposed stage plan is right.
pub mod plan {
    /// The kernel's verdict on a proposed plan.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Verdict {
        /// The plan is right.
        Plan,
        /// One layer does not fit beside the context; no plan can.
        CannotRun,
        /// The proposed stage is larger than the budget allows.
        StageDoesNotFit,
        /// A larger stage would fit.
        StageTooSmall,
        /// The stages leave a layer out, or one of them is empty.
        StagesDoNotCover,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request that proposes `stage_layers` layers in each of
    /// `stage_count` stages.
    #[must_use]
    pub fn request(
        budget: u64,
        context_bytes: u64,
        layer_bytes: u64,
        layers: u64,
        stage_layers: u64,
        stage_count: u64,
    ) -> [u8; 49] {
        let mut out = [0; 49];
        out[0] = 3;
        crate::put(&mut out, 1, budget);
        crate::put(&mut out, 9, context_bytes);
        crate::put(&mut out, 17, layer_bytes);
        crate::put(&mut out, 25, layers);
        crate::put(&mut out, 33, stage_layers);
        crate::put(&mut out, 41, stage_count);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Verdict {
        match text {
            b"plan" => Verdict::Plan,
            b"cannot run" => Verdict::CannotRun,
            b"stage does not fit" => Verdict::StageDoesNotFit,
            b"stage too small" => Verdict::StageTooSmall,
            b"stages do not cover" => Verdict::StagesDoNotCover,
            crate::INVALID_REQUEST => Verdict::Invalid,
            _ => Verdict::Unrecognised,
        }
    }
}

/// Decision 4: the quantisation tier.
pub mod tier {
    /// The tier the kernel chose.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Tier {
        /// Eight-bit weights, held in memory.
        Int8,
        /// Four-bit weights, held in memory.
        Int4,
        /// Eight-bit weights, paged in as needed.
        Int8Paged,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a model of `parameters` in `budget` bytes of memory.
    #[must_use]
    pub fn request(parameters: u64, budget: u64) -> [u8; 17] {
        let mut out = [0; 17];
        out[0] = 4;
        crate::put(&mut out, 1, parameters);
        crate::put(&mut out, 9, budget);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Tier {
        match text {
            b"int8" => Tier::Int8,
            b"int4" => Tier::Int4,
            b"int8 paged" => Tier::Int8Paged,
            crate::INVALID_REQUEST => Tier::Invalid,
            _ => Tier::Unrecognised,
        }
    }
}

/// Decision 5: whether a buffer is admitted.
pub mod admission {
    /// The kernel's verdict on a buffer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Verdict {
        /// The label and the length are the ones expected.
        Admit,
        /// The label or the length differs.
        Refuse,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a buffer whose BLAKE3 label should be `expected` and
    /// was observed to be `observed`, and whose length should be
    /// `declared_length` and was `observed_length`.
    #[must_use]
    pub fn request(
        expected: &[u8; 32],
        observed: &[u8; 32],
        declared_length: u64,
        observed_length: u64,
    ) -> [u8; 81] {
        let mut out = [0; 81];
        out[0] = 5;
        out[1..33].copy_from_slice(expected);
        out[33..65].copy_from_slice(observed);
        crate::put(&mut out, 65, declared_length);
        crate::put(&mut out, 73, observed_length);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Verdict {
        match text {
            b"admit" => Verdict::Admit,
            b"refuse" => Verdict::Refuse,
            crate::INVALID_REQUEST => Verdict::Invalid,
            _ => Verdict::Unrecognised,
        }
    }
}

/// Decision 6: whether a tensor's byte range is valid.
pub mod range {
    /// The kernel's verdict on a range.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Verdict {
        /// The range follows the previous tensor and ends within the file.
        Accept,
        /// The range overlaps the previous tensor or passes the end of the file.
        Reject,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a tensor at `offset` of `length` bytes, in a file of
    /// `file_length` bytes whose previous tensor ended at `previous_end`.
    #[must_use]
    pub fn request(file_length: u64, previous_end: u64, offset: u64, length: u64) -> [u8; 33] {
        let mut out = [0; 33];
        out[0] = 6;
        crate::put(&mut out, 1, file_length);
        crate::put(&mut out, 9, previous_end);
        crate::put(&mut out, 17, offset);
        crate::put(&mut out, 25, length);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Verdict {
        match text {
            b"accept" => Verdict::Accept,
            b"reject" => Verdict::Reject,
            crate::INVALID_REQUEST => Verdict::Invalid,
            _ => Verdict::Unrecognised,
        }
    }
}

/// Decision 7: whether a proposed trim of the conversation is right.
pub mod trim {
    /// The kernel's verdict on a proposed trim.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Verdict {
        /// The trim is right: what remains fits, and no fewer pairs would do.
        Trim,
        /// The part that is never dropped is already larger than the context.
        TooLong,
        /// What remains does not fit the context.
        DoesNotFit,
        /// The newest dropped pair could have been kept.
        DropsTooMany,
        /// More was reported dropped than there was.
        DroppedExceedsTotal,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request that proposes dropping `dropped` tokens of pairs, the
    /// newest of them `newest_dropped` tokens, from pairs totalling `total`,
    /// beside a part of `fixed` tokens that is never dropped.
    #[must_use]
    pub fn request(
        context: u64,
        fixed: u64,
        total: u64,
        dropped: u64,
        newest_dropped: u64,
    ) -> [u8; 41] {
        let mut out = [0; 41];
        out[0] = 7;
        crate::put(&mut out, 1, context);
        crate::put(&mut out, 9, fixed);
        crate::put(&mut out, 17, total);
        crate::put(&mut out, 25, dropped);
        crate::put(&mut out, 33, newest_dropped);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Verdict {
        match text {
            b"trim" => Verdict::Trim,
            b"too long" => Verdict::TooLong,
            b"does not fit" => Verdict::DoesNotFit,
            b"drops too many" => Verdict::DropsTooMany,
            b"dropped exceeds total" => Verdict::DroppedExceedsTotal,
            crate::INVALID_REQUEST => Verdict::Invalid,
            _ => Verdict::Unrecognised,
        }
    }
}

/// Decision 8: the next size of the attention cache.
pub mod bucket {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// The cache should now hold this many positions.
        Bucket(u64),
        /// More is needed than the model's maximum context.
        ContextExceeded,
        /// The current cache is already larger than the maximum.
        BucketExceedsMaximum,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a cache that holds `current` positions, needs
    /// `needed`, and may never hold more than `maximum`.
    #[must_use]
    pub fn request(needed: u64, current: u64, maximum: u64) -> [u8; 25] {
        let mut out = [0; 25];
        out[0] = 8;
        crate::put(&mut out, 1, needed);
        crate::put(&mut out, 9, current);
        crate::put(&mut out, 17, maximum);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"context exceeded" => Answer::ContextExceeded,
            b"bucket exceeds maximum" => Answer::BucketExceedsMaximum,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => crate::decimal(text).map_or(Answer::Unrecognised, Answer::Bucket),
        }
    }
}

/// Decision 9: a turn's token budget.
pub mod generation {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// The turn may generate this many tokens.
        Budget(u64),
        /// There is no room left in the context.
        ContextFull,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a turn in a context of `context` tokens with `used`
    /// already taken. A `requested` of zero asks for no cap.
    #[must_use]
    pub fn request(context: u64, used: u64, requested: u64) -> [u8; 25] {
        let mut out = [0; 25];
        out[0] = 9;
        crate::put(&mut out, 1, context);
        crate::put(&mut out, 9, used);
        crate::put(&mut out, 17, requested);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"context full" => Answer::ContextFull,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => crate::decimal(text).map_or(Answer::Unrecognised, Answer::Budget),
        }
    }
}

/// Decision 10: may this ONNX graph be imported?
///
/// A graph is importable when the opset it declares lies within the window of
/// opsets the importer implements, and every operator in it is one the importer
/// builds. The window and the count of operators outside it are the importer's
/// to report; whether the graph may be built is the kernel's to answer.
pub mod operators {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Verdict {
        /// Every operator is implemented and the opset is within the window.
        Importable,
        /// The graph declares an opset older than the window.
        OpsetBelowMinimum,
        /// The graph declares an opset newer than the window.
        OpsetAboveMaximum,
        /// The graph uses an operator the importer does not build.
        UnsupportedOperator,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    impl Verdict {
        /// Whether a graph may be built. Only one answer says so, and a
        /// verdict the client does not know never does.
        #[must_use]
        pub const fn authorises_import(self) -> bool {
            matches!(self, Self::Importable)
        }
    }

    /// The request about a graph declaring opset `declared`, against an
    /// importer that implements opsets `lowest` to `highest` and finds
    /// `unimplemented` of the graph's operators outside what it builds.
    #[must_use]
    pub fn request(declared: u64, lowest: u64, highest: u64, unimplemented: u64) -> [u8; 33] {
        let mut out = [0; 33];
        out[0] = 10;
        crate::put(&mut out, 1, declared);
        crate::put(&mut out, 9, lowest);
        crate::put(&mut out, 17, highest);
        crate::put(&mut out, 25, unimplemented);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Verdict {
        match text {
            b"importable" => Verdict::Importable,
            b"opset below minimum" => Verdict::OpsetBelowMinimum,
            b"opset above maximum" => Verdict::OpsetAboveMaximum,
            b"unsupported operator" => Verdict::UnsupportedOperator,
            crate::INVALID_REQUEST => Verdict::Invalid,
            _ => Verdict::Unrecognised,
        }
    }
}

/// Decision 11: the size a dimension is bound to.
pub mod shape {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// The dimension is bound to this size.
        Size(u64),
        /// The request is below the lower bound.
        BelowMinimum,
        /// The request is above the upper bound.
        AboveMaximum,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request to bind a dimension to `requested`, rounded up to a
    /// multiple of `step`, within `lower` and `upper`. A `step` of zero or
    /// one asks for the exact size.
    #[must_use]
    pub fn request(lower: u64, upper: u64, requested: u64, step: u64) -> [u8; 33] {
        let mut out = [0; 33];
        out[0] = 11;
        crate::put(&mut out, 1, lower);
        crate::put(&mut out, 9, upper);
        crate::put(&mut out, 17, requested);
        crate::put(&mut out, 25, step);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"below minimum" => Answer::BelowMinimum,
            b"above maximum" => Answer::AboveMaximum,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => crate::decimal(text).map_or(Answer::Unrecognised, Answer::Size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::decimal;

    #[test]
    fn decimal_reads_canonical_numbers() {
        assert_eq!(decimal(b"0"), Some(0));
        assert_eq!(decimal(b"128"), Some(128));
        assert_eq!(decimal(b"18446744073709551615"), Some(u64::MAX));
    }

    #[test]
    fn decimal_refuses_what_the_kernel_never_writes() {
        // One past the widest number, a leading zero, a sign, a space, nothing.
        assert_eq!(decimal(b"18446744073709551616"), None);
        assert_eq!(decimal(b"99999999999999999999"), None);
        assert_eq!(decimal(b"007"), None);
        assert_eq!(decimal(b"00"), None);
        assert_eq!(decimal(b"+5"), None);
        assert_eq!(decimal(b"5 "), None);
        assert_eq!(decimal(b""), None);
        assert_eq!(decimal(b"123456789012345678901"), None);
    }
}
