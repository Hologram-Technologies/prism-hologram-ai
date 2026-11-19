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

/// The chat templates the engine defines (KN-47): each family's template, for
/// the turns its page runs, written out rather than interpreted, and checked
/// equal to the kernel's own definition (decision 46) and to the text
/// Hugging Face's transformers renders.
///
/// Why: a template is code the model's files carry, and running it would mean
/// trusting an interpreter with it. Decided by Ilya on 2026-11-18: the subset
/// the supported families use is defined and proved in LexLean, and any other
/// template is refused by name (decision 45). No Jinja library.
///
/// The subset: system, user and assistant turns, with or without the
/// generation prompt, Qwen3 with its thinking on or off, and Llama 3.2's date.
/// Tools are not in it: they cannot be written as a [`Role`]. A Qwen3
/// assistant turn that carries `</think>` reasoning is refused by name.
pub mod template {
    /// A family whose template is defined here.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Family {
        /// Qwen2.5-1.5B-Instruct's template.
        Qwen25,
        /// Qwen3-0.6B's template.
        Qwen3,
        /// Llama 3.2 1B Instruct's template.
        Llama32,
    }

    /// Who said a turn.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Role {
        /// The system's instructions.
        System,
        /// The person.
        User,
        /// The model.
        Assistant,
    }

    impl Role {
        fn name(self) -> &'static str {
            match self {
                Self::System => "system",
                Self::User => "user",
                Self::Assistant => "assistant",
            }
        }

        fn code(self) -> u8 {
            match self {
                Self::System => 0,
                Self::User => 1,
                Self::Assistant => 2,
            }
        }
    }

    /// One turn.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Message<'a> {
        /// Who said it.
        pub role: Role,
        /// What was said.
        pub content: &'a str,
    }

    /// How a conversation is rendered.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Options {
        /// Open the assistant's turn after the last message.
        pub add_generation_prompt: bool,
        /// Qwen3's `enable_thinking=false`: an empty thinking block opens the
        /// assistant's turn.
        pub thinking_off: bool,
        /// Llama 3.2's date line, as `strftime("%d %b %Y")` writes it.
        pub date: [u8; 11],
    }

    /// Why a conversation is not rendered.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum TemplateError {
        /// `FL-60`: a Qwen3 assistant turn carries `</think>` reasoning, which
        /// the defined subset leaves out.
        ReasoningInTurn,
    }

    /// Python's `str.isspace`, which Jinja's `trim` strips by.
    fn python_space(c: char) -> bool {
        matches!(
            c,
            '\u{9}'..='\u{d}'
                | '\u{1c}'..='\u{20}'
                | '\u{85}'
                | '\u{a0}'
                | '\u{1680}'
                | '\u{2000}'..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
        )
    }

    fn qwen_turn(out: &mut String, role: Role, content: &str) {
        out.push_str("<|im_start|>");
        out.push_str(role.name());
        out.push('\n');
        out.push_str(content);
        out.push_str("<|im_end|>\n");
    }

    /// `messages` rendered as `family`'s template renders them.
    ///
    /// # Errors
    ///
    /// [`TemplateError::ReasoningInTurn`] for a Qwen3 assistant turn that
    /// carries `</think>`.
    pub fn render(
        family: Family,
        messages: &[Message<'_>],
        options: Options,
    ) -> Result<String, TemplateError> {
        let mut out = String::new();
        let first_system = messages.first().is_some_and(|m| m.role == Role::System);
        match family {
            Family::Llama32 => {
                out.push_str("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n");
                out.push_str("Cutting Knowledge Date: December 2023\nToday Date: ");
                out.push_str(&String::from_utf8_lossy(&options.date));
                out.push_str("\n\n");
                if let (true, Some(first)) = (first_system, messages.first()) {
                    out.push_str(first.content.trim_matches(python_space));
                }
                out.push_str("<|eot_id|>");
                for (i, m) in messages.iter().enumerate() {
                    if i == 0 && first_system {
                        continue;
                    }
                    out.push_str("<|start_header_id|>");
                    out.push_str(m.role.name());
                    out.push_str("<|end_header_id|>\n\n");
                    out.push_str(m.content.trim_matches(python_space));
                    out.push_str("<|eot_id|>");
                }
                if options.add_generation_prompt {
                    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
            }
            Family::Qwen25 | Family::Qwen3 => {
                let qwen3 = family == Family::Qwen3;
                if qwen3
                    && messages
                        .iter()
                        .any(|m| m.role == Role::Assistant && m.content.contains("</think>"))
                {
                    return Err(TemplateError::ReasoningInTurn);
                }
                match (first_system, messages.first()) {
                    (true, Some(first)) => qwen_turn(&mut out, Role::System, first.content),
                    _ if !qwen3 => out.push_str(
                        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n",
                    ),
                    _ => {}
                }
                let last = messages.len().saturating_sub(1);
                let query = messages
                    .iter()
                    .rposition(|m| m.role == Role::User)
                    .unwrap_or(last);
                for (i, m) in messages.iter().enumerate() {
                    if i == 0 && first_system {
                        continue;
                    }
                    if qwen3 && m.role == Role::Assistant && i > query && i == last {
                        out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
                        out.push_str(m.content.trim_start_matches('\n'));
                        out.push_str("<|im_end|>\n");
                    } else {
                        qwen_turn(&mut out, m.role, m.content);
                    }
                }
                if options.add_generation_prompt {
                    out.push_str("<|im_start|>assistant\n");
                    if qwen3 && options.thinking_off {
                        out.push_str("<think>\n\n</think>\n\n");
                    }
                }
            }
        }
        Ok(out)
    }

    /// Decision 46's request for the same conversation, which the kernel
    /// renders by its own definition: seventy-nine bytes, the family, the
    /// flags, the date and the count, then four roles, four lengths and four
    /// texts of up to fourteen bytes, a turn past the count all zero. `None`
    /// past what one request holds.
    #[must_use]
    pub fn request(family: Family, messages: &[Message<'_>], options: Options) -> Option<Vec<u8>> {
        if messages.len() > 4 {
            return None;
        }
        let mut out = vec![0u8; 79];
        out[0] = 46;
        out[1] = match family {
            Family::Qwen25 => 1,
            Family::Qwen3 => 2,
            Family::Llama32 => 3,
        };
        out[2] = u8::from(options.add_generation_prompt) + 2 * u8::from(options.thinking_off);
        out[3..14].copy_from_slice(&options.date);
        out[14] = u8::try_from(messages.len()).ok()?;
        for (i, m) in messages.iter().enumerate() {
            let text = m.content.as_bytes();
            if text.len() > 14 {
                return None;
            }
            out[15 + i] = m.role.code();
            out[19 + i] = u8::try_from(text.len()).ok()?;
            out[23 + 14 * i..23 + 14 * i + text.len()].copy_from_slice(text);
        }
        Some(out)
    }
}
