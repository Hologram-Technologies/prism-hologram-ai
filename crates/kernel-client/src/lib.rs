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

/// How long one block of weights is, which is the kernel's thirteenth
/// decision.
pub mod blocks {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// One block of this shape is this many bytes.
        Bytes(u64),
        /// The shape is not one this format can be stored in.
        BadShape,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for the length of one block of a tensor of `rows` by
    /// `columns` stored in quantisation format `format`.
    #[must_use]
    pub fn request(format: u8, rows: u64, columns: u64) -> [u8; 18] {
        let mut out = [0; 18];
        out[0] = 13;
        out[1] = format;
        crate::put(&mut out, 2, rows);
        crate::put(&mut out, 10, columns);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"bad shape" => Answer::BadShape,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => crate::decimal(text).map_or(Answer::Unrecognised, Answer::Bytes),
        }
    }
}

/// Write `number`'s low `width` bytes, big-endian, at `at`.
fn put_width(out: &mut [u8], at: usize, width: usize, number: u64) {
    out[at..at + width].copy_from_slice(&number.to_be_bytes()[8 - width..]);
}

/// Decision 18: whether an ONNX operator, by its name, is one the engine runs.
pub mod opname {
    /// Which export style the graph is in.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Style {
        /// Normalisation, rotation and attention written out in standard operators.
        Decomposed,
        /// Those written as the runtime's fused extension operators.
        Fused,
        /// A BERT-style encoder: layer norms written out, GELU by `Erf`, and
        /// the classifier's dense layers as `Gemm`.
        Encoder,
    }

    /// The kernel's verdict on an operator.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Verdict {
        /// The name is on its style's list.
        Supported,
        /// It is not.
        Unsupported,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// An operator's name as the 32 bytes the kernel compares, zero after it;
    /// `None` for a name longer than 32 bytes, which is no listed name.
    #[must_use]
    pub fn name(text: &str) -> Option<[u8; 32]> {
        let bytes = text.as_bytes();
        let mut out = [0; 32];
        out.get_mut(..bytes.len())?.copy_from_slice(bytes);
        Some(out)
    }

    /// The request for operator `name`, written by [`name`], in a graph of `style`.
    #[must_use]
    pub fn request(style: Style, name: &[u8; 32]) -> [u8; 34] {
        let mut out = [0; 34];
        out[0] = 18;
        out[1] = match style {
            Style::Decomposed => 0,
            Style::Fused => 1,
            Style::Encoder => 2,
        };
        out[2..34].copy_from_slice(name);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Verdict {
        match text {
            b"supported" => Verdict::Supported,
            b"unsupported operator" => Verdict::Unsupported,
            crate::INVALID_REQUEST => Verdict::Invalid,
            _ => Verdict::Unrecognised,
        }
    }
}

/// Decision 19: whether a model's family and shape are ones the engine runs.
pub mod family {
    /// What a model's configuration says about its shape.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Shape {
        /// Attention heads.
        pub heads: u16,
        /// Key-value heads.
        pub groups: u16,
        /// The size of one head.
        pub head_size: u16,
        /// Layers.
        pub layers: u16,
        /// The hidden width.
        pub width: u32,
        /// The vocabulary.
        pub vocabulary: u32,
    }

    /// The kernel's verdict on a configuration.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Verdict {
        /// A listed family, the gated activation, a shape the engine runs.
        Supported,
        /// The family is not listed.
        UnknownFamily,
        /// Listed, but its activation or shape is not one the engine runs.
        UnsupportedShape,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// A configuration's `model_type` or activation as the eight bytes the
    /// kernel compares; `None` for one longer than eight, which is unlisted.
    #[must_use]
    pub fn word(text: &str) -> Option<[u8; 8]> {
        let bytes = text.as_bytes();
        let mut out = [0; 8];
        out.get_mut(..bytes.len())?.copy_from_slice(bytes);
        Some(out)
    }

    /// The request for a model of type `kind` and activation `activation`,
    /// each written by [`word`], with `shape`.
    #[must_use]
    pub fn request(kind: &[u8; 8], activation: &[u8; 8], shape: Shape) -> [u8; 33] {
        let mut out = [0; 33];
        out[0] = 19;
        out[1..9].copy_from_slice(kind);
        out[9..17].copy_from_slice(activation);
        crate::put_width(&mut out, 17, 2, u64::from(shape.heads));
        crate::put_width(&mut out, 19, 2, u64::from(shape.groups));
        crate::put_width(&mut out, 21, 2, u64::from(shape.head_size));
        crate::put_width(&mut out, 23, 2, u64::from(shape.layers));
        crate::put_width(&mut out, 25, 4, u64::from(shape.width));
        crate::put_width(&mut out, 29, 4, u64::from(shape.vocabulary));
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Verdict {
        match text {
            b"supported" => Verdict::Supported,
            b"rejected: unknown family" => Verdict::UnknownFamily,
            b"rejected: unsupported shape" => Verdict::UnsupportedShape,
            crate::INVALID_REQUEST => Verdict::Invalid,
            _ => Verdict::Unrecognised,
        }
    }
}

/// Decision 20: how many bytes one part of a MatMulNBits weight is.
pub mod nbits {
    /// Which part.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Part {
        /// The packed 4-bit codes.
        Codes,
        /// One f32 scale for each block.
        ScalesF32,
        /// One f16 scale for each block.
        ScalesF16,
    }

    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// This part is this many bytes.
        Bytes(u64),
        /// The shape or block size is not one this layout takes.
        BadShape,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for `part` of a weight of `rows` by `columns` in blocks of `block`.
    #[must_use]
    pub fn request(part: Part, rows: u32, columns: u32, block: u16) -> [u8; 12] {
        let mut out = [0; 12];
        out[0] = 20;
        out[1] = match part {
            Part::Codes => 0,
            Part::ScalesF32 => 1,
            Part::ScalesF16 => 2,
        };
        crate::put_width(&mut out, 2, 4, u64::from(rows));
        crate::put_width(&mut out, 6, 4, u64::from(columns));
        crate::put_width(&mut out, 10, 2, u64::from(block));
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"bad shape" => Answer::BadShape,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => crate::decimal(text).map_or(Answer::Unrecognised, Answer::Bytes),
        }
    }
}

/// Decision 21: which offered precision fits.
pub mod precision {
    /// What one precision costs, if the repository offers it.
    pub type Offer = Option<u64>;

    /// The kernel's choice.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Choice {
        /// 4-bit weights, f16 scales and activations.
        Q4F16,
        /// 4-bit weights, f32 scales and activations.
        Q4,
        /// f16 weights.
        Fp16,
        /// No offered precision fits.
        NoneFits,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a `budget` of bytes, with what q4f16, q4 and fp16 cost
    /// where the repository offers them.
    #[must_use]
    pub fn request(budget: u64, q4f16: Offer, q4: Offer, fp16: Offer) -> [u8; 36] {
        let mut out = [0; 36];
        out[0] = 21;
        for (i, offer) in [q4f16, q4, fp16].into_iter().enumerate() {
            out[1 + i] = u8::from(offer.is_some());
            crate::put(&mut out, 12 + 8 * i, offer.unwrap_or(0));
        }
        crate::put(&mut out, 4, budget);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Choice {
        match text {
            b"q4f16" => Choice::Q4F16,
            b"q4" => Choice::Q4,
            b"fp16" => Choice::Fp16,
            b"none fits" => Choice::NoneFits,
            crate::INVALID_REQUEST => Choice::Invalid,
            _ => Choice::Unrecognised,
        }
    }
}

/// Decision 22: WebGPU, the CPU, or neither.
pub mod device {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Device {
        /// The adapter can hold the model.
        WebGpu,
        /// It cannot, and the model is within the CPU ceiling.
        Cpu,
        /// Neither.
        Refused,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a model of `model` bytes whose largest buffer is
    /// `largest`, on an adapter that is `present` with binding limit `limit`,
    /// with the CPU taking models up to `ceiling` bytes.
    #[must_use]
    pub fn request(present: bool, limit: u64, largest: u64, model: u64, ceiling: u64) -> [u8; 34] {
        let mut out = [0; 34];
        out[0] = 22;
        out[1] = u8::from(present);
        crate::put(&mut out, 2, limit);
        crate::put(&mut out, 10, largest);
        crate::put(&mut out, 18, model);
        crate::put(&mut out, 26, ceiling);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Device {
        match text {
            b"webgpu" => Device::WebGpu,
            b"cpu" => Device::Cpu,
            b"refused: too large for cpu" => Device::Refused,
            crate::INVALID_REQUEST => Device::Invalid,
            _ => Device::Unrecognised,
        }
    }
}

/// Decision 23: whether a span of one of a model's files may be read.
pub mod chunk {
    /// The kernel's verdict.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Verdict {
        /// The span lies inside the file, after the previous one.
        Accept,
        /// It does not.
        Reject,
        /// The file is not one the model declares.
        Undeclared,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// Which file, and what the model declares.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct File {
        /// How many data files the model declares beside its graph.
        pub declared: u8,
        /// This file: 0 for the graph, then 1 onward for the data files.
        pub index: u8,
        /// Its length in bytes.
        pub length: u64,
    }

    /// The request for `length` bytes at `offset` of `file`, after a span that ended at `previous_end`.
    #[must_use]
    pub fn request(file: File, offset: u64, length: u64, previous_end: u64) -> [u8; 35] {
        let mut out = [0; 35];
        out[0] = 23;
        out[1] = file.declared;
        out[2] = file.index;
        crate::put(&mut out, 3, file.length);
        crate::put(&mut out, 11, offset);
        crate::put(&mut out, 19, length);
        crate::put(&mut out, 27, previous_end);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Verdict {
        match text {
            b"accept" => Verdict::Accept,
            b"reject" => Verdict::Reject,
            b"undeclared file" => Verdict::Undeclared,
            crate::INVALID_REQUEST => Verdict::Invalid,
            _ => Verdict::Unrecognised,
        }
    }
}

/// Decision 24: what each of a model's files is called.
pub mod source {
    /// The precision a file is for.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Precision {
        /// `_q4f16`.
        Q4F16,
        /// `_q4`.
        Q4,
        /// `_fp16`.
        Fp16,
        /// No suffix.
        Fp32,
    }

    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer<'a> {
        /// The file's path in the repository.
        Path(&'a [u8]),
        /// The part is beyond the data files declared.
        Undeclared,
        /// More data files are declared than a model may have.
        TooMany,
        /// The kernel has no files for this precision.
        UnsupportedPrecision,
        /// The kernel has no files for this task.
        UnsupportedTask,
        /// The kernel did not accept the request.
        Invalid,
    }

    /// The task the files are for, in decision 24's code.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Task {
        /// `text-generation`.
        TextGeneration,
        /// `feature-extraction`: a sentence embedding.
        FeatureExtraction,
        /// `text-classification`: label scores.
        TextClassification,
    }

    /// The request for part `part` (0 the graph, then the data files) of the
    /// text-generation files at `precision`, with `declared` data files.
    #[must_use]
    pub fn request(precision: Precision, declared: u8, part: u8) -> [u8; 5] {
        request_for(Task::TextGeneration, precision, declared, part)
    }

    /// The request for part `part` of `task`'s files at `precision`, with
    /// `declared` data files.
    #[must_use]
    pub fn request_for(task: Task, precision: Precision, declared: u8, part: u8) -> [u8; 5] {
        [
            24,
            match task {
                Task::TextGeneration => 0,
                Task::FeatureExtraction => 1,
                Task::TextClassification => 2,
            },
            match precision {
                Precision::Q4F16 => 0,
                Precision::Q4 => 1,
                Precision::Fp16 => 2,
                Precision::Fp32 => 3,
            },
            declared,
            part,
        ]
    }

    /// Read the kernel's answer. A path is any other text, which is why this
    /// answer has no `Unrecognised`: the kernel's names are its own to choose.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer<'_> {
        match text {
            b"undeclared file" => Answer::Undeclared,
            b"too many files" => Answer::TooMany,
            b"unsupported precision" => Answer::UnsupportedPrecision,
            b"unsupported task" => Answer::UnsupportedTask,
            crate::INVALID_REQUEST => Answer::Invalid,
            path => Answer::Path(path),
        }
    }
}

/// Decision 25: whether generation stops.
pub mod stop {
    /// The most stop ids one request carries.
    pub const MOST: usize = 16;

    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// The token is one of the model's stop ids.
        EndToken,
        /// The budget is spent.
        Budget,
        /// Neither.
        Continue,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// A request: its bytes, of which the first `length` are sent.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Request {
        bytes: [u8; 14 + 4 * MOST],
        length: usize,
    }

    impl Request {
        /// The bytes to send.
        #[must_use]
        pub fn bytes(&self) -> &[u8] {
            &self.bytes[..self.length]
        }
    }

    /// The request for `token`, the `generated`-th of a turn of `budget`, with
    /// the model's stop ids `ids`. More than [`MOST`] ids are written as a
    /// count the length does not match, which the kernel refuses as invalid,
    /// rather than cut short, which it would answer.
    #[must_use]
    pub fn request(token: u32, generated: u32, budget: u32, ids: &[u32]) -> Request {
        let mut bytes = [0; 14 + 4 * MOST];
        bytes[0] = 25;
        bytes[1] = u8::try_from(ids.len()).unwrap_or(u8::MAX);
        crate::put_width(&mut bytes, 2, 4, u64::from(token));
        crate::put_width(&mut bytes, 6, 4, u64::from(generated));
        crate::put_width(&mut bytes, 10, 4, u64::from(budget));
        for (i, id) in ids.iter().take(MOST).enumerate() {
            crate::put_width(&mut bytes, 14 + 4 * i, 4, u64::from(*id));
        }
        Request {
            bytes,
            length: 14 + 4 * ids.len().min(MOST),
        }
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"stop: end token" => Answer::EndToken,
            b"stop: budget" => Answer::Budget,
            b"continue" => Answer::Continue,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => Answer::Unrecognised,
        }
    }
}

/// Decision 26: whether a cached block is dropped to make room.
pub mod evict {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// There is room already; drop nothing more.
        Enough,
        /// Drop this block.
        Evict,
        /// Keep it: it is pinned.
        Keep,
        /// No candidate is left and there is still no room.
        DoesNotFit,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a block that is `pinned` or not, with `remaining`
    /// candidates including it, when `needed` bytes are wanted, `free` are free
    /// and `freed` have been freed so far.
    #[must_use]
    pub fn request(pinned: bool, remaining: u32, needed: u64, free: u64, freed: u64) -> [u8; 30] {
        let mut out = [0; 30];
        out[0] = 26;
        out[1] = u8::from(pinned);
        crate::put_width(&mut out, 2, 4, u64::from(remaining));
        crate::put(&mut out, 6, needed);
        crate::put(&mut out, 14, free);
        crate::put(&mut out, 22, freed);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"enough" => Answer::Enough,
            b"evict" => Answer::Evict,
            b"keep" => Answer::Keep,
            b"does not fit" => Answer::DoesNotFit,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => Answer::Unrecognised,
        }
    }
}

/// Decision 27: whether a computed prefix may be reused.
pub mod reuse {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// Reuse it.
        Reuse,
        /// Compute the prompt from the start.
        Recompute,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a prefix held under `held`, `held_length` positions
    /// long, for a prompt of `prompt_length` whose prefix label is `asked`.
    #[must_use]
    pub fn request(
        asked: &[u8; 32],
        held: &[u8; 32],
        held_length: u32,
        prompt_length: u32,
    ) -> [u8; 73] {
        let mut out = [0; 73];
        out[0] = 27;
        out[1..33].copy_from_slice(asked);
        out[33..65].copy_from_slice(held);
        crate::put_width(&mut out, 65, 4, u64::from(held_length));
        crate::put_width(&mut out, 69, 4, u64::from(prompt_length));
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"reuse" => Answer::Reuse,
            b"recompute" => Answer::Recompute,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => Answer::Unrecognised,
        }
    }
}

/// Decision 28: whether a model's manifest is satisfied.
pub mod manifest {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// Every listed tensor is held and admitted under the manifest's label.
        Ready,
        /// The label matches; some tensors are not yet admitted.
        Incomplete,
        /// The addresses admitted are not the manifest's.
        Refused,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a manifest whose label is `expected`, when the label
    /// of the addresses admitted is `observed`, `admitted` of `listed` tensors.
    #[must_use]
    pub fn request(
        expected: &[u8; 32],
        observed: &[u8; 32],
        listed: u64,
        admitted: u64,
    ) -> [u8; 81] {
        let mut out = [0; 81];
        out[0] = 28;
        out[1..33].copy_from_slice(expected);
        out[33..65].copy_from_slice(observed);
        crate::put(&mut out, 65, listed);
        crate::put(&mut out, 73, admitted);
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"ready" => Answer::Ready,
            b"incomplete" => Answer::Incomplete,
            b"refused" => Answer::Refused,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => Answer::Unrecognised,
        }
    }
}

/// Decision 34: whether a claimed sum is the exact sum of one 32-wide q4
/// block times int8 activations.
pub mod block_check {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// The claim is the block's sum.
        Exact,
        /// The claim is not the block's sum.
        Differs,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for one block as the engine stores it, weight `g` in bits
    /// `4 * (g % 2)` of `codes[g / 2]` as its code less eight, times
    /// `activations`, with `claim` its sum.
    ///
    /// The kernel reads weight `16 * h + i` from the high half of byte
    /// `8 * h + 8 - i / 2` when `i` is even and from the low half when it is
    /// odd, and its activation from byte `16 * h + 32 - i`. The claim is
    /// written plus 32768 in four bytes, wrapping, which no two `i32` claims
    /// share.
    #[must_use]
    pub fn request(codes: &[u8; 16], activations: &[i8; 32], claim: i32) -> [u8; 53] {
        let mut out = [0; 53];
        out[0] = 34;
        for g in 0..32 {
            let code = (codes[g / 2] >> (4 * (g % 2))) & 0x0f;
            let (h, i) = (g / 16, g % 16);
            out[8 * h + 8 - i / 2] |= if i % 2 == 0 { code << 4 } else { code };
            out[16 * h + 32 - i] = activations[g].to_be_bytes()[0];
        }
        out[49..53].copy_from_slice(&claim.wrapping_add(32768).to_be_bytes());
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"exact" => Answer::Exact,
            b"differs" => Answer::Differs,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => Answer::Unrecognised,
        }
    }
}

/// Decision 42: whether a claimed binary32 is the nearest, ties to even, to an
/// exact value `magnitude * 2^exponent`.
pub mod round_check {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// The claim is the nearest binary32.
        Exact,
        /// The claim is not.
        Differs,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for the value `magnitude * 2^exponent`, negative or not,
    /// with `claimed` the binary32 bits said to be nearest: the sign in one
    /// byte, the magnitude in eight, the exponent plus 32768 in two, wrapping,
    /// and the claimed bits in four.
    #[must_use]
    pub fn request(negative: bool, magnitude: u64, exponent: i32, claimed: u32) -> [u8; 16] {
        let mut out = [0; 16];
        out[0] = 42;
        out[1] = u8::from(negative);
        out[2..10].copy_from_slice(&magnitude.to_be_bytes());
        out[10..12].copy_from_slice(&(exponent.wrapping_add(32768) as u16).to_be_bytes());
        out[12..16].copy_from_slice(&claimed.to_be_bytes());
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"exact" => Answer::Exact,
            b"differs" => Answer::Differs,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => Answer::Unrecognised,
        }
    }
}

/// Decision 37: whether a claimed product is one group's sum times its half
/// scale, exactly, in units of `2^-24`.
pub mod scale_check {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// The claim is the product.
        Exact,
        /// The claim is not the product.
        Differs,
        /// The scale is an infinity or a NaN.
        NotFinite,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for a group whose sum is `sum` and whose scale is the half
    /// `bits`, with `claim` the product: the sum plus 32768 in four bytes,
    /// wrapping, the half in two, then the claim's sign, 1 for negative, and
    /// its magnitude in eight, wrapping past 64 bits.
    #[must_use]
    pub fn request(sum: i32, bits: u16, claim: i128) -> [u8; 16] {
        let mut out = [0; 16];
        out[0] = 37;
        out[1..5].copy_from_slice(&sum.wrapping_add(32768).to_be_bytes());
        out[5..7].copy_from_slice(&bits.to_be_bytes());
        out[7] = u8::from(claim < 0);
        out[8..16].copy_from_slice(&(claim.unsigned_abs() as u64).to_be_bytes());
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"exact" => Answer::Exact,
            b"differs" => Answer::Differs,
            b"not finite" => Answer::NotFinite,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => Answer::Unrecognised,
        }
    }
}

/// Decision 49: whether a claimed binary32, or a claimed whole number, is the
/// one a config number's decimal text names.
pub mod config_check {
    /// What the claim is said to be.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Mode {
        /// The nearest binary32, ties to even.
        Binary32,
        /// The whole number, exactly.
        Whole,
    }

    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// The claim is the number the text names.
        Exact,
        /// The claim is not.
        Differs,
        /// The text is not a decimal the kernel reads.
        NotDecimal,
        /// The value is past what the kernel converts.
        OutOfRange,
        /// A whole number was asked for and the text names a fraction.
        NotWhole,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for `text` with `claimed` the answer: the mode in one byte,
    /// the claim in eight, then the text in forty, padded with zero bytes,
    /// which end it. `None` for text longer than 40 bytes or holding a zero
    /// byte, which the kernel does not read.
    #[must_use]
    pub fn request(mode: Mode, text: &[u8], claimed: u64) -> Option<[u8; 50]> {
        if text.len() > 40 || text.contains(&0) {
            return None;
        }
        let mut out = [0; 50];
        out[0] = 49;
        out[1] = u8::from(mode == Mode::Whole);
        out[2..10].copy_from_slice(&claimed.to_be_bytes());
        out[10..10 + text.len()].copy_from_slice(text);
        Some(out)
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"exact" => Answer::Exact,
            b"differs" => Answer::Differs,
            b"not a decimal" => Answer::NotDecimal,
            b"out of range" => Answer::OutOfRange,
            b"not whole" => Answer::NotWhole,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => Answer::Unrecognised,
        }
    }
}

/// Decision 50: whether an f16 cache entry lies within a tolerance of a
/// reference rotary value, both in units of `2^-30`.
pub mod rotary_check {
    /// The kernel's answer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Answer {
        /// The entry is within the tolerance.
        Agrees,
        /// It is not.
        Disagrees,
        /// The entry is an infinity or a NaN.
        NotFinite,
        /// The kernel did not accept the request.
        Invalid,
        /// Text the kernel is not known to give.
        Unrecognised,
    }

    /// The request for the f16 `half` against `reference` with `tolerance`:
    /// the half in two bytes, the reference's sign (1 for negative) in one and
    /// its magnitude in four, then the tolerance in four, each at most 2^30.
    #[must_use]
    pub fn request(half: u16, reference: i32, tolerance: u32) -> [u8; 12] {
        let mut out = [0; 12];
        out[0] = 50;
        out[1..3].copy_from_slice(&half.to_be_bytes());
        out[3] = u8::from(reference < 0);
        out[4..8].copy_from_slice(&reference.unsigned_abs().to_be_bytes());
        out[8..12].copy_from_slice(&tolerance.to_be_bytes());
        out
    }

    /// Read the kernel's answer.
    #[must_use]
    pub fn answer(text: &[u8]) -> Answer {
        match text {
            b"agrees" => Answer::Agrees,
            b"disagrees" => Answer::Disagrees,
            b"not finite" => Answer::NotFinite,
            crate::INVALID_REQUEST => Answer::Invalid,
            _ => Answer::Unrecognised,
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
