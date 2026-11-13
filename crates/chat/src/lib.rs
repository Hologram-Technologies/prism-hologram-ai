//! A conversation: where it is, what it holds, and what it may do next.
//!
//! Four of the kernel's decisions are about a conversation. Which step may be
//! taken is decision 1. Whether a proposed trim is the right one is decision 7.
//! How large the decode bucket must be is decision 8, and how many tokens a
//! turn may generate is decision 9.
//!
//! None of them is decided here. This crate keeps the counts a conversation
//! has — the context it was compiled for, the part that is never dropped, and
//! each question-and-answer pair in turn — proposes what it believes should
//! happen, and asks. A refusal carries the kernel's own verdict.

#![forbid(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hologram_ai_kernel_client::{bucket, generation, journey, trim};
use hologram_ai_tokenizer::Tokeniser;

/// Whoever holds the verified kernel holo: a request's bytes in, an answer's
/// bytes out.
pub trait Kernel {
    /// Put `request` to the kernel and return its answer.
    fn ask(&mut self, request: &[u8]) -> Vec<u8>;
}

/// Why a conversation could not do what was asked of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatError {
    /// `FL-16`: the kernel's own answer, carried as it gave it.
    StepRefused(journey::Step),
    /// `FL-17`
    TrimRefused(trim::Verdict),
    /// `FL-18`
    NoBucket(bucket::Answer),
    /// `FL-19`
    NoBudget(generation::Answer),
}

/// A trim: how much of the oldest conversation to drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Trim {
    /// How many whole pairs, counted from the oldest.
    pub pairs: usize,
    /// How many tokens those pairs hold altogether.
    pub dropped: u64,
    /// How many tokens the newest of them holds.
    pub newest_dropped: u64,
}

/// One conversation.
#[derive(Clone, Debug)]
pub struct Conversation {
    state: journey::State,
    context: u64,
    fixed: u64,
    pairs: Vec<u64>,
}

impl Conversation {
    /// A conversation at the beginning, for a model compiled with `context`
    /// tokens of room, of which `fixed` is the part that is never dropped: the
    /// system prompt, the newest message, and whatever is reserved for the
    /// answer.
    #[must_use]
    pub fn new(context: u64, fixed: u64) -> Self {
        Self {
            state: journey::State::Idle,
            context,
            fixed,
            pairs: Vec::new(),
        }
    }

    /// Where the conversation is.
    #[must_use]
    pub const fn state(&self) -> journey::State {
        self.state
    }

    /// Every question-and-answer pair, oldest first, as token counts.
    #[must_use]
    pub fn pairs(&self) -> &[u64] {
        &self.pairs
    }

    /// How many tokens every pair holds altogether.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.pairs
            .iter()
            .fold(0u64, |sum, pair| sum.saturating_add(*pair))
    }

    /// Remember a finished pair of `tokens`.
    ///
    /// The count is the caller's. [`Self::remember_turn`] takes the text
    /// instead and counts it with the model's own tokeniser, which is what the
    /// kernel's arithmetic is about.
    pub fn remember(&mut self, tokens: u64) {
        self.pairs.push(tokens);
    }

    /// Remember a finished turn: what was asked and what was answered, counted
    /// with the model's own tokeniser.
    ///
    /// A pair is a question and its answer together, because that is what a
    /// trim drops: half a pair is a question whose answer is gone.
    pub fn remember_turn(&mut self, tokeniser: &Tokeniser, question: &str, answer: &str) {
        let tokens = tokeniser
            .count(question)
            .saturating_add(tokeniser.count(answer));
        self.remember(tokens);
    }

    /// How many tokens are never dropped: the system prompt, the newest
    /// message, and whatever is reserved for the answer.
    #[must_use]
    pub const fn fixed(&self) -> u64 {
        self.fixed
    }

    /// Set that part from the text it is, counted with the model's own
    /// tokeniser, plus the tokens reserved for the answer.
    pub fn fix(&mut self, tokeniser: &Tokeniser, system: &str, newest: &str, reserve: u64) {
        self.fixed = tokeniser
            .count(system)
            .saturating_add(tokeniser.count(newest))
            .saturating_add(reserve);
    }

    /// Take one step of the journey.
    ///
    /// The step the kernel gives is the new state. A step it refuses leaves
    /// the conversation exactly where it was: a state the kernel did not give
    /// is not a state to be in.
    pub fn step(
        &mut self,
        event: journey::Event,
        kernel: &mut dyn Kernel,
    ) -> Result<journey::State, ChatError> {
        let answer = kernel.ask(&journey::request(self.state, event));
        match journey::answer(&answer) {
            journey::Step::To(next) => {
                self.state = next;
                Ok(next)
            }
            refused => Err(ChatError::StepRefused(refused)),
        }
    }

    /// The trim this conversation proposes: the fewest whole pairs, from the
    /// oldest, that leave the rest fitting beside the part that is never
    /// dropped.
    ///
    /// Whole pairs, because half a pair is a question whose answer is gone.
    /// Oldest first, because that is the order a conversation is forgotten in.
    /// The fewest, because the kernel accepts a trim only when putting the
    /// newest dropped pair back would not fit, and anything more forgets what
    /// it did not have to.
    #[must_use]
    pub fn proposed_trim(&self) -> Trim {
        let mut trim = Trim::default();
        let mut remaining = self.total();
        while self.fixed.saturating_add(remaining) > self.context && trim.pairs < self.pairs.len() {
            let pair = self.pairs.get(trim.pairs).copied().unwrap_or_default();
            remaining = remaining.saturating_sub(pair);
            trim.dropped = trim.dropped.saturating_add(pair);
            trim.newest_dropped = pair;
            trim.pairs += 1;
        }
        trim
    }

    /// The kernel's question about the trim this conversation proposes.
    #[must_use]
    pub fn trim_question(&self) -> [u8; 41] {
        let proposed = self.proposed_trim();
        trim::request(
            self.context,
            self.fixed,
            self.total(),
            proposed.dropped,
            proposed.newest_dropped,
        )
    }

    /// Propose the trim and, if the kernel accepts it, forget those pairs.
    ///
    /// Nothing is forgotten unless the kernel accepts: a refusal leaves the
    /// conversation whole, carrying the verdict that refused it.
    pub fn trim(&mut self, kernel: &mut dyn Kernel) -> Result<Trim, ChatError> {
        let proposed = self.proposed_trim();
        let answer = kernel.ask(&self.trim_question());
        match trim::answer(&answer) {
            trim::Verdict::Trim => {
                self.pairs.drain(..proposed.pairs);
                Ok(proposed)
            }
            refused => Err(ChatError::TrimRefused(refused)),
        }
    }

    /// The decode bucket for a sequence of `needed` tokens, grown from
    /// `current`, within the model's `maximum` context.
    pub fn bucket(
        &self,
        needed: u64,
        current: u64,
        maximum: u64,
        kernel: &mut dyn Kernel,
    ) -> Result<u64, ChatError> {
        let answer = kernel.ask(&bucket::request(needed, current, maximum));
        match bucket::answer(&answer) {
            bucket::Answer::Bucket(size) => Ok(size),
            refused => Err(ChatError::NoBucket(refused)),
        }
    }

    /// How many tokens this turn may generate, having used `used` of the
    /// context, when the caller asks for `requested` or for as many as there
    /// is room for, which is `requested` of zero.
    pub fn budget(
        &self,
        used: u64,
        requested: u64,
        kernel: &mut dyn Kernel,
    ) -> Result<u64, ChatError> {
        let answer = kernel.ask(&generation::request(self.context, used, requested));
        match generation::answer(&answer) {
            generation::Answer::Budget(tokens) => Ok(tokens),
            refused => Err(ChatError::NoBudget(refused)),
        }
    }
}
