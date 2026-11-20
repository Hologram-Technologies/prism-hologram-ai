//! Token ids to logits on the CPU, in integers only.
//!
//! Why: Hologram AI's inference uses no floating-point operation between the
//! admitted bytes and the token ids (Q_ENGINE_PLAN.md, "Integer only"), so a
//! run gives the same bits on every machine and anyone can recompute it.
//!
//! How: the residual stream is binary32 bit patterns, held as `u32` and
//! touched only by integer code (decided by Ilya on 2026-11-18). Each stage
//! reads them exactly, computes on integers, and rounds its result back once,
//! half to even (KN-42): the 4-bit products of `hologram-ai-exact` (EN-34 to
//! EN-41), RMSNorm by the integer square root (EN-47), rotary positions on
//! the certified sine table (EN-45, EN-48), softmax on the certified exponent
//! table (EN-43, EN-44) and SiLU (EN-46). A stage that takes fixed point
//! enters by one declared narrowing to units of `2^-16`, rounded half to even.
//!
//! What: [`read`] finds a Qwen2-, Qwen3- or Llama-shaped model's weights in
//! its ONNX export, fused or decomposed, by the nodes that read them; [`step`] runs one token through it and
//! answers the logits; [`greedy`] picks the largest.

#![forbid(unsafe_code)]
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use hologram_ai_exact::{
    self as exact, Dense, DenseLayout, Dyadic, ExactError, Grouping, Q4Matrix, Q4Scales, Q4Zeros,
    RopeScaling, Row, Scheme, Width,
};
use hologram_ai_onnx_preflight::stream::Node;

/// Why a forward pass could not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// `FL-55`: the graph has no node for a role this pass reads, or the
    /// weight that node reads is missing or not the shape the role needs.
    Unrecognised(Role),
    /// `FL-56`: an exact stage refused its operands.
    Exact(ExactError),
    /// `FL-57`: a row is past RMSNorm's bounds (EN-47), or its root is 0.
    NormRefused,
    /// `FL-58`: a token id at or past the vocabulary.
    TokenOutOfRange,
    /// `FL-80`: a product answered outside the model, by the engine's
    /// helpers, came back with another number of rows than the matrix has, or
    /// a slice sent to one did not read back.
    Elsewhere,
    /// `FL-110`: a layer, the final norm or the head was asked of a model
    /// read before its weights arrived (EN-110), or a prefill was asked out of
    /// order: a step on a model with a layer still missing, a layer the
    /// prefill is not at, or an empty prompt.
    NotHeld,
}

impl From<ExactError> for DecodeError {
    fn from(error: ExactError) -> Self {
        Self::Exact(error)
    }
}

/// A role in the model, which a graph that lacks it is refused by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The token embedding, read by `/model/embed_tokens/Gather`.
    Embedding,
    /// A layer's attention: head counts the config and the graph disagree on.
    Attention,
    /// A layer's norm before attention.
    InputNorm,
    /// The query, key, value and output projections.
    Query,
    /// See [`Role::Query`].
    Key,
    /// See [`Role::Query`].
    Value,
    /// See [`Role::Query`].
    Output,
    /// The per-head norms of queries and keys.
    QueryNorm,
    /// See [`Role::QueryNorm`].
    KeyNorm,
    /// A layer's norm before its feed-forward.
    PostNorm,
    /// The feed-forward's gate, up and down projections.
    Gate,
    /// See [`Role::Gate`].
    Up,
    /// See [`Role::Gate`].
    Down,
    /// The norm after the last layer.
    FinalNorm,
}

/// A tensor whose data lies in the graph file, as the importer found it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Initializer<'a> {
    /// Its element type, as `onnx.proto` numbers them.
    pub data_type: u64,
    /// Its bytes.
    pub bytes: &'a [u8],
}

/// What a model's config gives that its graph may not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// `num_attention_heads`; a fused graph's attention node must agree.
    pub heads: usize,
    /// `num_key_value_heads`; likewise.
    pub kv_heads: usize,
    /// `rope_theta`, a whole number.
    pub rope_theta: u64,
    /// `rms_norm_eps`, as binary32 bits.
    pub rms_norm_eps: u32,
    /// `rope_scaling`.
    pub scaling: RopeScaling,
}

/// A model's sizes, read from its graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    /// The residual stream's width.
    pub hidden: usize,
    /// Layers.
    pub layers: usize,
    /// Query heads.
    pub heads: usize,
    /// Key and value heads.
    pub kv_heads: usize,
    /// One head's width.
    pub head_dim: usize,
    /// The feed-forward's width.
    pub ff: usize,
    /// Token ids.
    pub vocab: usize,
}

struct Layer<'a> {
    input_norm: Dense<'a>,
    query: Q4Matrix<'a>,
    key: Q4Matrix<'a>,
    value: Q4Matrix<'a>,
    biases: Option<[Dense<'a>; 3]>,
    output: Q4Matrix<'a>,
    query_norm: Option<Dense<'a>>,
    key_norm: Option<Dense<'a>>,
    post_norm: Dense<'a>,
    gate: Q4Matrix<'a>,
    up: Q4Matrix<'a>,
    down: Q4Matrix<'a>,
}

/// A model read from its graph, its weights borrowed from the file's bytes.
pub struct Model<'a> {
    /// Its sizes.
    pub dims: Dims,
    embedding: Dense<'a>,
    layers: Vec<Layer<'a>>,
    /// `None` while a layer is still missing (EN-110).
    final_norm: Option<Dense<'a>>,
    frequencies: Vec<u64>,
    epsilon: u64,
    root: i128,
}

/// The activation scheme every product takes its input on: 16-bit codes, one
/// power of two per group of 32, as measured on milestone 2's models.
const SCHEME: Scheme = Scheme {
    width: Width::Sixteen,
    grouping: Grouping::Groups,
};

/// The input of `node` that is an initializer, the `nth` of them.
fn weight<'a>(
    node: &Node,
    initializers: &BTreeMap<String, Initializer<'a>>,
    nth: usize,
) -> Option<Initializer<'a>> {
    node.inputs
        .iter()
        .filter_map(|input| initializers.get(input))
        .nth(nth)
        .copied()
}

fn node<'n>(nodes: &'n [Node], name: &str, role: Role) -> Result<&'n Node, DecodeError> {
    nodes
        .iter()
        .find(|n| n.name == name)
        .ok_or(DecodeError::Unrecognised(role))
}

fn int(node: &Node, key: &str) -> Option<i64> {
    node.ints.iter().find(|(k, _)| k == key).map(|&(_, v)| v)
}

/// A vector of `length` values read by the node `name`: halves, as a q4f16
/// export stores them, or binary32, as a q4 export does.
fn vector<'a>(
    nodes: &[Node],
    initializers: &BTreeMap<String, Initializer<'a>>,
    name: &str,
    role: Role,
    length: usize,
) -> Result<Dense<'a>, DecodeError> {
    let w =
        weight(node(nodes, name, role)?, initializers, 0).ok_or(DecodeError::Unrecognised(role))?;
    match w.data_type {
        10 if w.bytes.len() == 2 * length => Ok(Dense::F16(w.bytes)),
        1 if w.bytes.len() == 4 * length => Ok(Dense::F32(w.bytes)),
        _ => Err(DecodeError::Unrecognised(role)),
    }
}

/// A `MatMulNBits` node's matrix, at four bits with an exact codec.
fn matrix<'a>(
    nodes: &[Node],
    initializers: &BTreeMap<String, Initializer<'a>>,
    name: &str,
    role: Role,
) -> Result<Q4Matrix<'a>, DecodeError> {
    let n = node(nodes, name, role)?;
    let refused = DecodeError::Unrecognised(role);
    let (Some(columns), Some(rows), Some(bits), Some(block)) = (
        int(n, "K"),
        int(n, "N"),
        int(n, "bits"),
        int(n, "block_size"),
    ) else {
        return Err(refused);
    };
    let codes = weight(n, initializers, 0).ok_or(refused)?;
    let scales = weight(n, initializers, 1).ok_or(refused)?;
    let zeros = weight(n, initializers, 2);
    let scales_type = i32::try_from(scales.data_type).map_err(|_| refused)?;
    let zeros_type = zeros.map(|z| i32::try_from(z.data_type).unwrap_or(-1));
    exact::matmulnbits_codec(bits, block, scales_type, zeros_type)?;
    Ok(Q4Matrix {
        codes: codes.bytes,
        scales: if scales_type == 10 {
            Q4Scales::F16(scales.bytes)
        } else {
            Q4Scales::F32(scales.bytes)
        },
        zeros: zeros.map_or(Q4Zeros::Eight, |z| Q4Zeros::Packed(z.bytes)),
        rows: usize::try_from(rows).map_err(|_| refused)?,
        columns: usize::try_from(columns).map_err(|_| refused)?,
        block: usize::try_from(block).map_err(|_| refused)?,
    })
}

/// `value * 2^shift`, rounded half to even when `shift` is negative.
fn shift_even(value: i128, shift: i32) -> i128 {
    if shift >= 0 {
        return value
            .checked_shl(shift.unsigned_abs())
            .unwrap_or(if value < 0 { i128::MIN } else { i128::MAX });
    }
    let by = shift.unsigned_abs();
    if by >= 127 {
        return 0;
    }
    let magnitude = value.unsigned_abs();
    let (whole, rest, half) = (
        magnitude >> by,
        magnitude & ((1u128 << by) - 1),
        1u128 << (by - 1),
    );
    let up = rest > half || (rest == half && whole & 1 == 1);
    let rounded = (whole + u128::from(up)) as i128;
    if value < 0 {
        -rounded
    } else {
        rounded
    }
}

/// The units of `2^-32` nearest a binary32's value, half to even.
fn units32(bits: u32) -> Result<u64, DecodeError> {
    let d = Dyadic::from_binary32(bits).ok_or(ExactError::InputNotFinite)?;
    u64::try_from(shift_even(d.mantissa, d.exponent + 32)).map_err(|_| DecodeError::NormRefused)
}

/// How an export names the nodes this pass reads: the fused style (Qwen3's
/// and Llama's exports, `SkipSimplifiedLayerNormalization` and
/// `GroupQueryAttention`) or the decomposed one (Qwen2.5's, norms as `Mul_1`
/// and biases as `Add`).
struct Style {
    attention: &'static str,
    fused: bool,
}

impl Style {
    fn input_norm(&self, l: usize) -> String {
        match (self.fused, l) {
            (true, 0) => "/model/layers.0/input_layernorm/LayerNorm".to_string(),
            (true, _) => format!("/model/layers.{l}/input_layernorm/SkipLayerNorm"),
            (false, _) => format!("/model/layers.{l}/input_layernorm/Mul_1"),
        }
    }

    fn post_norm(&self, l: usize) -> String {
        let last = if self.fused { "SkipLayerNorm" } else { "Mul_1" };
        format!("/model/layers.{l}/post_attention_layernorm/{last}")
    }

    fn final_norm(&self, layers: usize) -> String {
        if self.fused {
            format!("/model/layers.{layers}/final_norm_layernorm/SkipLayerNorm")
        } else {
            "/model/norm/Mul_1".to_string()
        }
    }

    fn projection(&self, l: usize, p: &str) -> String {
        format!("/model/layers.{l}/{}/{p}_proj/MatMul_Q4", self.attention)
    }
}

/// Reads a Qwen2-, Qwen3- or Llama-shaped model from its ONNX export, fused
/// or decomposed: each weight from the node that reads it, the sizes from the
/// weights and `config`, the rotary frequencies from `config` (EN-48). Query
/// and key norms and query, key and value biases are read where the graph
/// has them.
///
/// # Errors
///
/// [`DecodeError::Unrecognised`] for a role the graph does not have in the
/// shape this pass reads, and [`DecodeError::Exact`] for a 4-bit matrix with
/// no exact codec.
pub fn read<'a>(
    nodes: &[Node],
    initializers: &BTreeMap<String, Initializer<'a>>,
    config: Config,
) -> Result<Model<'a>, DecodeError> {
    read_first(nodes, initializers, config, usize::MAX)
}

/// The model as far as its weights have arrived (EN-110): the embedding,
/// the sizes the graph states, and layers `0..held` (all of them when `held`
/// is at least their number), the final norm only once every layer is held.
/// What a prefill reads layer by layer while the rest of the file streams;
/// [`step`] and [`advance`] refuse it with [`DecodeError::NotHeld`] until it
/// is whole.
///
/// # Errors
///
/// As [`read`], for the embedding and the layers asked.
pub fn read_first<'a>(
    nodes: &[Node],
    initializers: &BTreeMap<String, Initializer<'a>>,
    config: Config,
    held: usize,
) -> Result<Model<'a>, DecodeError> {
    let gather = node(nodes, "/model/embed_tokens/Gather", Role::Embedding)?;
    let embedding =
        weight(gather, initializers, 0).ok_or(DecodeError::Unrecognised(Role::Embedding))?;
    let has = |name: &str| nodes.iter().any(|n| n.name == name);
    let style = if has("/model/layers.0/attn/q_proj/MatMul_Q4") {
        Style {
            attention: "attn",
            fused: true,
        }
    } else if has("/model/layers.0/self_attn/q_proj/MatMul_Q4") {
        Style {
            attention: "self_attn",
            fused: false,
        }
    } else {
        return Err(DecodeError::Unrecognised(Role::Query));
    };
    let layers = (0..)
        .take_while(|&l| has(&style.projection(l, "q")))
        .count();
    let refused = DecodeError::Unrecognised(Role::Attention);
    let (heads, kv_heads) = (config.heads, config.kv_heads);
    if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
        return Err(refused);
    }
    if style.fused {
        // The attention node states the head counts too; they must agree.
        let attention = node(
            nodes,
            "/model/layers.0/attn/GroupQueryAttention",
            Role::Attention,
        )?;
        let stated = |key: &str| int(attention, key).and_then(|v| usize::try_from(v).ok());
        if stated("num_heads") != Some(heads) || stated("kv_num_heads") != Some(kv_heads) {
            return Err(refused);
        }
    }
    let query = matrix(nodes, initializers, &style.projection(0, "q"), Role::Query)?;
    let gate = matrix(
        nodes,
        initializers,
        "/model/layers.0/mlp/gate_proj/MatMul_Q4",
        Role::Gate,
    )?;
    let hidden = query.columns;
    let head_dim = query.rows / heads;
    // The embedding, and the head that reads it, as halves or binary32.
    let width = match embedding.data_type {
        10 => 2,
        1 => 4,
        _ => return Err(DecodeError::Unrecognised(Role::Embedding)),
    };
    let vocab = embedding.bytes.len() / width / hidden.max(1);
    if embedding.bytes.len() != width * vocab * hidden || head_dim * heads != query.rows {
        return Err(DecodeError::Unrecognised(Role::Embedding));
    }
    let dims = Dims {
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        ff: gate.rows,
        vocab,
    };
    let shaped = |m: Q4Matrix<'a>, rows: usize, columns: usize, role: Role| {
        if m.rows == rows && m.columns == columns {
            Ok(m)
        } else {
            Err(DecodeError::Unrecognised(role))
        }
    };
    let mut read_layers = Vec::with_capacity(layers);
    // Query and key norms are Qwen3's; biases are Qwen2's. A model has both
    // or neither in every layer, as layer 0 has them.
    let normed = has("/model/layers.0/attn/q_norm/SimplifiedLayerNormalization");
    let biased = has(&format!("/model/layers.0/{}/q_proj/Add", style.attention));
    for l in 0..layers.min(held) {
        let at = |rest: &str| format!("/model/layers.{l}/{rest}");
        let kv = kv_heads * head_dim;
        let head_norm = |p: &str, role: Role| -> Result<Option<Dense<'a>>, DecodeError> {
            if normed {
                Ok(Some(vector(
                    nodes,
                    initializers,
                    &at(&format!("attn/{p}_norm/SimplifiedLayerNormalization")),
                    role,
                    head_dim,
                )?))
            } else {
                Ok(None)
            }
        };
        let biases = if biased {
            let bias = |p: &str, role: Role, width: usize| {
                vector(
                    nodes,
                    initializers,
                    &at(&format!("{}/{p}_proj/Add", style.attention)),
                    role,
                    width,
                )
            };
            Some([
                bias("q", Role::Query, heads * head_dim)?,
                bias("k", Role::Key, kv)?,
                bias("v", Role::Value, kv)?,
            ])
        } else {
            None
        };
        read_layers.push(Layer {
            input_norm: vector(
                nodes,
                initializers,
                &style.input_norm(l),
                Role::InputNorm,
                hidden,
            )?,
            query: shaped(
                matrix(nodes, initializers, &style.projection(l, "q"), Role::Query)?,
                heads * head_dim,
                hidden,
                Role::Query,
            )?,
            key: shaped(
                matrix(nodes, initializers, &style.projection(l, "k"), Role::Key)?,
                kv,
                hidden,
                Role::Key,
            )?,
            value: shaped(
                matrix(nodes, initializers, &style.projection(l, "v"), Role::Value)?,
                kv,
                hidden,
                Role::Value,
            )?,
            biases,
            output: shaped(
                matrix(nodes, initializers, &style.projection(l, "o"), Role::Output)?,
                hidden,
                heads * head_dim,
                Role::Output,
            )?,
            query_norm: head_norm("q", Role::QueryNorm)?,
            key_norm: head_norm("k", Role::KeyNorm)?,
            post_norm: vector(
                nodes,
                initializers,
                &style.post_norm(l),
                Role::PostNorm,
                hidden,
            )?,
            gate: shaped(
                matrix(
                    nodes,
                    initializers,
                    &at("mlp/gate_proj/MatMul_Q4"),
                    Role::Gate,
                )?,
                dims.ff,
                hidden,
                Role::Gate,
            )?,
            up: shaped(
                matrix(nodes, initializers, &at("mlp/up_proj/MatMul_Q4"), Role::Up)?,
                dims.ff,
                hidden,
                Role::Up,
            )?,
            down: shaped(
                matrix(
                    nodes,
                    initializers,
                    &at("mlp/down_proj/MatMul_Q4"),
                    Role::Down,
                )?,
                hidden,
                dims.ff,
                Role::Down,
            )?,
        });
    }
    let final_norm = if held >= layers {
        Some(vector(
            nodes,
            initializers,
            &style.final_norm(layers),
            Role::FinalNorm,
            hidden,
        )?)
    } else {
        None
    };
    let head = u32::try_from(head_dim).map_err(|_| refused)?;
    let frequencies =
        exact::rope_frequencies(head, config.rope_theta, config.scaling).ok_or(refused)?;
    // floor(2^40 / sqrt(head_dim)): the floor root of the floored quotient.
    let root = i128::try_from(exact::integer_root((1u128 << 80) / u128::from(head)))
        .map_err(|_| refused)?;
    Ok(Model {
        dims,
        embedding: if width == 2 {
            Dense::F16(embedding.bytes)
        } else {
            Dense::F32(embedding.bytes)
        },
        layers: read_layers,
        final_norm,
        frequencies,
        epsilon: units32(config.rms_norm_eps)?,
        root,
    })
}

/// One component of a layer a probe reads (EN-97), each binary32 bit
/// patterns of the residual stream's width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    /// The residual stream after the layer: what the next layer reads.
    Residual,
    /// What attention adds to the residual stream: its output projection.
    Attention,
    /// What the feed-forward adds: its down projection.
    FeedForward,
}

impl Component {
    /// Components by number, in the order above; how many there are is
    /// [`Component::COUNT`].
    #[must_use]
    pub fn numbered(number: u32) -> Option<Self> {
        match number {
            0 => Some(Self::Residual),
            1 => Some(Self::Attention),
            2 => Some(Self::FeedForward),
            _ => None,
        }
    }

    /// How many components a layer has.
    pub const COUNT: u32 = 3;
}

/// What a run holds between steps: each layer's keys and values, binary32
/// bit patterns, position after position; and a probe, if one is set, with
/// what it read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Session {
    keys: Vec<Vec<u32>>,
    values: Vec<Vec<u32>>,
    position: u32,
    probe: Option<(usize, Component)>,
    probed: Option<Vec<u32>>,
}

impl Session {
    /// A run that has seen no token.
    #[must_use]
    pub fn new(model: &Model<'_>) -> Self {
        Self {
            keys: vec![Vec::new(); model.dims.layers],
            values: vec![Vec::new(); model.dims.layers],
            position: 0,
            probe: None,
            probed: None,
        }
    }

    /// Read `component` of `layer` in the next step (EN-97). Why on the
    /// session: a probe is the run itself, recomputed to that point, so what
    /// it reads is the value that produced the answer, not a copy kept aside.
    pub fn probe(&mut self, layer: usize, component: Component) {
        self.probe = Some((layer, component));
        self.probed = None;
    }

    /// What the probe read in the last step, once; `None` when no probe was
    /// set or its layer is past the model's.
    pub fn probed(&mut self) -> Option<Vec<u32>> {
        self.probe = None;
        self.probed.take()
    }

    /// How many tokens it has seen.
    #[must_use]
    pub fn position(&self) -> u32 {
        self.position
    }
    /// The run as it was after its first `length` tokens, which a later turn
    /// that opens the same way carries on from; `None` past what it has seen.
    #[must_use]
    pub fn prefix(&self, length: u32) -> Option<Self> {
        if length > self.position {
            return None;
        }
        let cut = |rows: &Vec<Vec<u32>>| {
            rows.iter()
                .map(|layer| {
                    let width = layer.len() / usize::try_from(self.position).unwrap_or(1).max(1);
                    layer[..width * length as usize].to_vec()
                })
                .collect()
        };
        Some(Self {
            probe: None,
            probed: None,
            keys: cut(&self.keys),
            values: cut(&self.values),
            position: length,
        })
    }
}

fn half(weights: Dense<'_>, i: usize) -> Result<Dyadic, DecodeError> {
    weights.at(i).map_err(DecodeError::Exact)
}

fn value(bits: u32) -> Result<Dyadic, DecodeError> {
    Dyadic::from_binary32(bits).ok_or(DecodeError::Exact(ExactError::InputNotFinite))
}

fn add_to(row: &mut Row, product: i128, exponent: i32) {
    row.add_scaled(product.unsigned_abs(), exponent, product < 0);
}

/// RMSNorm of binary32 `x` with half weights: each value entering EN-47 in
/// units of `2^-16`, the normalised value times its weight rounded once.
fn norm(x: &[u32], weights: Dense<'_>, epsilon: u64) -> Result<Vec<u32>, DecodeError> {
    let fixed = x
        .iter()
        .map(|&b| {
            let d = value(b)?;
            i64::try_from(shift_even(d.mantissa, d.exponent + 16))
                .map_err(|_| DecodeError::NormRefused)
        })
        .collect::<Result<Vec<i64>, DecodeError>>()?;
    let ones = vec![1i64 << 16; fixed.len()];
    let normalised = exact::rms_norm(&fixed, &ones, epsilon).ok_or(DecodeError::NormRefused)?;
    normalised
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let g = half(weights, i)?;
            Ok(exact::round_exact(
                i128::from(n) * g.mantissa,
                g.exponent - 16,
            ))
        })
        .collect()
}

/// Rotates one head in place at `position`: pair `i` with `i + d/2`, on
/// EN-48's angle and EN-45's table, each output rounded once.
fn rotate(head: &mut [u32], position: u32, frequencies: &[u64]) -> Result<(), DecodeError> {
    let half_width = head.len() / 2;
    for (i, &f) in frequencies.iter().enumerate().take(half_width) {
        let angle = exact::rope_angle(position, f);
        let (c, s) = (
            i128::from(exact::cosine_turn(angle)),
            i128::from(exact::sine_turn(angle)),
        );
        let (x, y) = (value(head[i])?, value(head[i + half_width])?);
        let mut first = Row::ZERO;
        add_to(&mut first, x.mantissa * c, x.exponent - 30);
        add_to(&mut first, -(y.mantissa * s), y.exponent - 30);
        let mut second = Row::ZERO;
        add_to(&mut second, y.mantissa * c, y.exponent - 30);
        add_to(&mut second, x.mantissa * s, x.exponent - 30);
        head[i] = exact::round_to_binary32(&first);
        head[i + half_width] = exact::round_to_binary32(&second);
    }
    Ok(())
}

/// `a + b`, both binary32, rounded once.
fn add(a: u32, b: u32) -> Result<u32, DecodeError> {
    let (a, b) = (value(a)?, value(b)?);
    let mut row = Row::ZERO;
    add_to(&mut row, a.mantissa, a.exponent);
    add_to(&mut row, b.mantissa, b.exponent);
    Ok(exact::round_to_binary32(&row))
}

/// The softmax exponent of a score, in units of `2^-8` of a base-two
/// exponent: `score log2(e) / sqrt(d) * 2^8`, from the score rounded once to
/// binary32, `LOG2_E` and `floor(2^40 / sqrt(d))`, rounded half to even.
fn exponent(score: u32, root: i128) -> Result<i64, DecodeError> {
    let s = value(score)?;
    let product = s.mantissa * i128::from(exact::LOG2_E) * root;
    let scaled = shift_even(product, s.exponent + 8 - 32 - 40);
    Ok(i64::try_from(scaled).unwrap_or(if scaled < 0 { i64::MIN } else { i64::MAX }))
}

/// One token through the model: its logits, binary32 bit patterns, one per
/// token id.
///
/// # Errors
///
/// [`DecodeError::TokenOutOfRange`] for an id past the vocabulary,
/// [`DecodeError::NormRefused`] for a row RMSNorm does not take, and
/// [`DecodeError::Exact`] when a product refuses its input.
pub fn step(model: &Model<'_>, session: &mut Session, token: u32) -> Result<Vec<u32>, DecodeError> {
    step_with(model, session, token, &mut Whole)
}

/// One token through the model without its logits (EN-91): the session is
/// left exactly as [`step`] leaves it, every key and value in the cache the
/// same bits, and only the output head, a quarter of a token's work, is not
/// computed. For every prompt id but the last, whose logits nothing reads.
///
/// # Errors
///
/// As [`step`], less the head's own product.
pub fn advance(model: &Model<'_>, session: &mut Session, token: u32) -> Result<(), DecodeError> {
    advance_with(model, session, token, &mut Whole)
}

/// [`step`], with every product answered by `products` (EN-92).
///
/// # Errors
///
/// As [`step`], and [`DecodeError::Elsewhere`] for an answer that is not
/// one bit pattern per row.
pub fn step_with(
    model: &Model<'_>,
    session: &mut Session,
    token: u32,
    products: &mut dyn Products,
) -> Result<Vec<u32>, DecodeError> {
    let h = through(model, session, token, products)?;
    let logits = answered(model, products, Product::Head, &h)?;
    session.position += 1;
    Ok(logits)
}

/// [`advance`], with every product answered by `products` (EN-92).
///
/// # Errors
///
/// As [`step_with`], less the head's own product.
pub fn advance_with(
    model: &Model<'_>,
    session: &mut Session,
    token: u32,
    products: &mut dyn Products,
) -> Result<(), DecodeError> {
    through(model, session, token, products)?;
    session.position += 1;
    Ok(())
}

/// A prompt on its way through the model a layer at a time (EN-110): every
/// prompt row through layer `l` before any row enters layer `l + 1`, so a
/// layer runs as soon as its weights arrive, while later layers are still on
/// their way.
///
/// Why the same bits as [`step`]: row `i`'s work in layer `l` reads its own
/// row and the keys and values of layer `l` at positions up to its own, and
/// every one of those is in the cache, the same bits, whichever order the
/// rows and layers are taken in. The session's position moves only when the
/// prefill finishes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prefill {
    rows: Vec<Vec<u32>>,
    start: u32,
    layer: usize,
}

impl Prefill {
    /// How many layers every prompt row has been through.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.layer
    }
}

/// Start a prefill of `prompt` after what `session` has seen: each id's
/// embedding row, which only the embedding has to have arrived for.
///
/// # Errors
///
/// [`DecodeError::NotHeld`] for an empty prompt, and
/// [`DecodeError::TokenOutOfRange`] for an id past the vocabulary.
pub fn prefill_start(
    model: &Model<'_>,
    session: &Session,
    prompt: &[u32],
) -> Result<Prefill, DecodeError> {
    if prompt.is_empty() {
        return Err(DecodeError::NotHeld);
    }
    Ok(Prefill {
        rows: prompt
            .iter()
            .map(|&t| embed(model, t))
            .collect::<Result<Vec<Vec<u32>>, DecodeError>>()?,
        start: session.position,
        layer: 0,
    })
}

/// The next layer for every prompt row, its keys and values added to the
/// cache, with every product answered by `products`.
///
/// # Errors
///
/// [`DecodeError::NotHeld`] when the model was read without that layer, or
/// the cache holds another number of positions for it than the prefill
/// started after; and as [`step_with`].
pub fn prefill_layer_with(
    model: &Model<'_>,
    session: &mut Session,
    prefill: &mut Prefill,
    products: &mut dyn Products,
) -> Result<(), DecodeError> {
    let l = prefill.layer;
    let width = model.dims.kv_heads * model.dims.head_dim;
    let cached = session
        .keys
        .get(l)
        .map(Vec::len)
        .ok_or(DecodeError::NotHeld)?;
    if l >= model.layers.len() || cached != width * prefill.start as usize {
        return Err(DecodeError::NotHeld);
    }
    let start = prefill.start;
    for (i, row) in prefill.rows.iter_mut().enumerate() {
        let position = start + u32::try_from(i).map_err(|_| DecodeError::NotHeld)?;
        *row = layer_through(model, session, l, row, position, products)?;
    }
    prefill.layer += 1;
    Ok(())
}

/// [`prefill_layer_with`], every product computed on the model.
///
/// # Errors
///
/// As [`prefill_layer_with`].
pub fn prefill_layer(
    model: &Model<'_>,
    session: &mut Session,
    prefill: &mut Prefill,
) -> Result<(), DecodeError> {
    prefill_layer_with(model, session, prefill, &mut Whole)
}

/// Every layer done: the final norm and the head for the last prompt row, the
/// logits [`step`] gives for it, and the session moved past the prompt.
///
/// # Errors
///
/// [`DecodeError::NotHeld`] before every layer has run or while the model
/// lacks its final norm; and as [`step_with`].
pub fn prefill_finish_with(
    model: &Model<'_>,
    session: &mut Session,
    prefill: &Prefill,
    products: &mut dyn Products,
) -> Result<Vec<u32>, DecodeError> {
    if prefill.layer != model.dims.layers || model.layers.len() != model.dims.layers {
        return Err(DecodeError::NotHeld);
    }
    let last = prefill.rows.last().ok_or(DecodeError::NotHeld)?;
    let h = finish(model, last)?;
    let logits = answered(model, products, Product::Head, &h)?;
    let taken = u32::try_from(prefill.rows.len()).map_err(|_| DecodeError::NotHeld)?;
    session.position = prefill.start + taken;
    Ok(logits)
}

/// [`prefill_finish_with`], the head computed on the model.
///
/// # Errors
///
/// As [`prefill_finish_with`].
pub fn prefill_finish(
    model: &Model<'_>,
    session: &mut Session,
    prefill: &Prefill,
) -> Result<Vec<u32>, DecodeError> {
    prefill_finish_with(model, session, prefill, &mut Whole)
}

/// One of the products a step takes: a layer's projection, or the head.
///
/// Why named: a product answered elsewhere (the engine's helpers, GP-52) is
/// asked for by name and its rows by number, the weights already held there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Product {
    /// A layer's query projection.
    Query(usize),
    /// A layer's key projection.
    Key(usize),
    /// A layer's value projection.
    Value(usize),
    /// A layer's output projection.
    Output(usize),
    /// A layer's gate projection.
    Gate(usize),
    /// A layer's up projection.
    Up(usize),
    /// A layer's down projection.
    Down(usize),
    /// The output head, the embedding read as rows of logits.
    Head,
}

impl Product {
    /// Its number: seven per layer, in the order above, then the head.
    #[must_use]
    pub fn number(self, layers: usize) -> usize {
        match self {
            Self::Query(l) => 7 * l,
            Self::Key(l) => 7 * l + 1,
            Self::Value(l) => 7 * l + 2,
            Self::Output(l) => 7 * l + 3,
            Self::Gate(l) => 7 * l + 4,
            Self::Up(l) => 7 * l + 5,
            Self::Down(l) => 7 * l + 6,
            Self::Head => 7 * layers,
        }
    }

    /// The product numbered `number` in a model of `layers` layers.
    #[must_use]
    pub fn numbered(number: usize, layers: usize) -> Option<Self> {
        let l = number / 7;
        match (l < layers, number % 7) {
            (true, 0) => Some(Self::Query(l)),
            (true, 1) => Some(Self::Key(l)),
            (true, 2) => Some(Self::Value(l)),
            (true, 3) => Some(Self::Output(l)),
            (true, 4) => Some(Self::Gate(l)),
            (true, 5) => Some(Self::Up(l)),
            (true, 6) => Some(Self::Down(l)),
            _ if number == 7 * layers => Some(Self::Head),
            _ => None,
        }
    }
}

/// What answers a step's products. [`Whole`] computes each on the model; the
/// engine's helpers each hold a slice of every matrix's rows (GP-52).
pub trait Products {
    /// `which` times `input`, one bit pattern per row of the matrix.
    ///
    /// # Errors
    ///
    /// The product's own refusal, or [`DecodeError::Elsewhere`].
    fn product(
        &mut self,
        model: &Model<'_>,
        which: Product,
        input: &[u32],
    ) -> Result<Vec<u32>, DecodeError>;
}

/// Every product computed whole, on the model, as [`step`] does.
#[derive(Clone, Copy, Debug, Default)]
pub struct Whole;

impl Products for Whole {
    fn product(
        &mut self,
        model: &Model<'_>,
        which: Product,
        input: &[u32],
    ) -> Result<Vec<u32>, DecodeError> {
        model.slice(which, 0..model.height(which)).product(input)
    }
}

/// Products answered elsewhere (GP-52): `ask` is given a product's number,
/// its input as little-endian bytes and room for its rows, and says whether
/// they were written. The input's own refusal is given here first, as
/// [`Whole`] would give it.
pub struct Asked<F>(pub F);

impl<F: FnMut(u32, &[u8], &mut [u8]) -> bool> Products for Asked<F> {
    fn product(
        &mut self,
        model: &Model<'_>,
        which: Product,
        input: &[u32],
    ) -> Result<Vec<u32>, DecodeError> {
        exact::quantise_with(input, SCHEME)?;
        let given: Vec<u8> = input.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut answer = vec![0u8; 4 * model.height(which)];
        let number =
            u32::try_from(which.number(model.dims.layers)).map_err(|_| DecodeError::Elsewhere)?;
        if !(self.0)(number, &given, &mut answer) {
            return Err(DecodeError::Elsewhere);
        }
        Ok(answer
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect())
    }
}

/// `which` answered by `products`, held to one bit pattern per row.
fn answered(
    model: &Model<'_>,
    products: &mut dyn Products,
    which: Product,
    input: &[u32],
) -> Result<Vec<u32>, DecodeError> {
    let out = products.product(model, which, input)?;
    if out.len() == model.height(which) {
        Ok(out)
    } else {
        Err(DecodeError::Elsewhere)
    }
}

/// Some of a matrix's output rows, as the model borrows them.
///
/// Why this is exact: each output row of a q4 or a dense product is that
/// row's weights times the input, the input put on its integers the same way
/// for every row; so rows computed apart are the same bits as rows computed
/// together (EN-92).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slice<'a> {
    /// Rows of a q4 matrix.
    Q4(Q4Matrix<'a>),
    /// Rows of the head: each `columns` values, halves or binary32, row
    /// after row.
    Head {
        /// The rows' values.
        weights: Dense<'a>,
        /// The input's length.
        columns: usize,
        /// How many rows.
        rows: usize,
    },
}

impl Slice<'_> {
    /// These rows times `input`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Exact`] when the product refuses its input.
    pub fn product(&self, input: &[u32]) -> Result<Vec<u32>, DecodeError> {
        Ok(match *self {
            Self::Q4(matrix) => exact::q4_product_with(input, &matrix, SCHEME)?,
            Self::Head {
                weights,
                columns,
                rows,
            } => exact::dense_product_with(
                input,
                weights,
                DenseLayout::OutputMajor,
                columns,
                rows,
                SCHEME,
            )?,
        })
    }

    /// The slice as bytes another instance holds with [`Held::read`]: four
    /// little-endian words of kind and shape, the lengths, then the bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut word = |n: usize| out.extend_from_slice(&(n as u32).to_le_bytes());
        match *self {
            Self::Q4(m) => {
                let (scale_kind, scales) = match m.scales {
                    Q4Scales::F32(b) => (1, b),
                    Q4Scales::F16(b) => (2, b),
                };
                let zeros: &[u8] = match m.zeros {
                    Q4Zeros::Eight => &[],
                    Q4Zeros::Packed(b) => b,
                };
                for n in [
                    1,
                    m.rows,
                    m.columns,
                    m.block,
                    scale_kind,
                    usize::from(matches!(m.zeros, Q4Zeros::Packed(_))),
                    m.codes.len(),
                    scales.len(),
                    zeros.len(),
                ] {
                    word(n);
                }
                out.extend_from_slice(m.codes);
                out.extend_from_slice(scales);
                out.extend_from_slice(zeros);
            }
            Self::Head {
                weights,
                columns,
                rows,
            } => {
                // Kind 2 holds halves, kind 3 binary32.
                let (kind, bytes) = match weights {
                    Dense::F32(b) => (3, b),
                    Dense::F16(b) | Dense::Bf16(b) => (2, b),
                };
                for n in [kind, rows, columns, bytes.len()] {
                    word(n);
                }
                out.extend_from_slice(bytes);
            }
        }
        out
    }
}

/// A slice held apart from the model, as [`Slice::to_bytes`] wrote it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Held {
    bytes: Vec<u8>,
    shape: Vec<usize>,
}

impl Held {
    /// The slice `bytes` hold.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Elsewhere`] when they are not one.
    pub fn read(bytes: &[u8]) -> Result<Self, DecodeError> {
        let word = |i: usize| {
            bytes
                .get(4 * i..4 * i + 4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize)
                .ok_or(DecodeError::Elsewhere)
        };
        let words = match word(0)? {
            1 => 9,
            2 | 3 => 4,
            _ => return Err(DecodeError::Elsewhere),
        };
        let shape = (0..words)
            .map(word)
            .collect::<Result<Vec<usize>, DecodeError>>()?;
        let body: usize = if words == 9 {
            shape[6] + shape[7] + shape[8]
        } else {
            shape[3]
        };
        if bytes.len() != 4 * words + body {
            return Err(DecodeError::Elsewhere);
        }
        Ok(Self {
            bytes: bytes[4 * words..].to_vec(),
            shape,
        })
    }

    /// The rows it holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.shape[1]
    }

    /// Its rows times `input`, as [`Slice::product`].
    ///
    /// # Errors
    ///
    /// As [`Slice::product`].
    pub fn product(&self, input: &[u32]) -> Result<Vec<u32>, DecodeError> {
        let s = &self.shape;
        let slice = if s[0] == 1 {
            let (codes, rest) = self.bytes.split_at(s[6]);
            let (scales, zeros) = rest.split_at(s[7]);
            Slice::Q4(Q4Matrix {
                codes,
                scales: if s[4] == 1 {
                    Q4Scales::F32(scales)
                } else {
                    Q4Scales::F16(scales)
                },
                zeros: if s[5] == 1 {
                    Q4Zeros::Packed(zeros)
                } else {
                    Q4Zeros::Eight
                },
                rows: s[1],
                columns: s[2],
                block: s[3],
            })
        } else {
            Slice::Head {
                weights: if s[0] == 3 {
                    Dense::F32(&self.bytes)
                } else {
                    Dense::F16(&self.bytes)
                },
                columns: s[2],
                rows: s[1],
            }
        };
        slice.product(input)
    }
}

impl<'a> Model<'a> {
    /// Every product a step takes, in order of [`Product::number`]: on a model
    /// read before every layer arrived (EN-110), those of the layers it holds
    /// and the head.
    #[must_use]
    pub fn products(&self) -> Vec<Product> {
        (0..=7 * self.dims.layers)
            .filter_map(|n| Product::numbered(n, self.dims.layers))
            .filter(|&p| self.holds(p))
            .collect()
    }

    /// Whether the model holds `which`'s weights: the head always, a layer's
    /// products once that layer is read.
    #[must_use]
    pub fn holds(&self, which: Product) -> bool {
        which == Product::Head || self.matrix(which).is_some()
    }

    fn matrix(&self, which: Product) -> Option<&Q4Matrix<'a>> {
        match which {
            Product::Query(l) => self.layers.get(l).map(|x| &x.query),
            Product::Key(l) => self.layers.get(l).map(|x| &x.key),
            Product::Value(l) => self.layers.get(l).map(|x| &x.value),
            Product::Output(l) => self.layers.get(l).map(|x| &x.output),
            Product::Gate(l) => self.layers.get(l).map(|x| &x.gate),
            Product::Up(l) => self.layers.get(l).map(|x| &x.up),
            Product::Down(l) => self.layers.get(l).map(|x| &x.down),
            Product::Head => None,
        }
    }

    /// The rows `which` answers.
    #[must_use]
    pub fn height(&self, which: Product) -> usize {
        self.matrix(which).map_or(self.dims.vocab, |m| m.rows)
    }

    /// Rows `rows` of `which`, borrowed from the model. A range past the
    /// matrix is cut to it.
    #[must_use]
    pub fn slice(&self, which: Product, rows: std::ops::Range<usize>) -> Slice<'a> {
        let end = rows.end.min(self.height(which));
        let start = rows.start.min(end);
        match self.matrix(which) {
            Some(m) => {
                let blocks = m.columns.div_ceil(m.block);
                let half_row = blocks * m.block / 2;
                let scales = match m.scales {
                    Q4Scales::F32(b) => Q4Scales::F32(&b[start * blocks * 4..end * blocks * 4]),
                    Q4Scales::F16(b) => Q4Scales::F16(&b[start * blocks * 2..end * blocks * 2]),
                };
                let zeros = match m.zeros {
                    Q4Zeros::Eight => Q4Zeros::Eight,
                    Q4Zeros::Packed(b) => {
                        let per = blocks.div_ceil(2);
                        Q4Zeros::Packed(&b[start * per..end * per])
                    }
                };
                Slice::Q4(Q4Matrix {
                    codes: &m.codes[start * half_row..end * half_row],
                    scales,
                    zeros,
                    rows: end - start,
                    columns: m.columns,
                    block: m.block,
                })
            }
            None => {
                let hidden = self.dims.hidden;
                let rows =
                    |width: usize, b: &'a [u8]| &b[start * width * hidden..end * width * hidden];
                Slice::Head {
                    weights: match self.embedding {
                        Dense::F32(b) => Dense::F32(rows(4, b)),
                        Dense::F16(b) => Dense::F16(rows(2, b)),
                        Dense::Bf16(b) => Dense::Bf16(rows(2, b)),
                    },
                    columns: hidden,
                    rows: end - start,
                }
            }
        }
    }
}

/// Every layer and the final norm for one token, its keys and values added to
/// the cache: the row the head reads.
fn through(
    model: &Model<'_>,
    session: &mut Session,
    token: u32,
    products: &mut dyn Products,
) -> Result<Vec<u32>, DecodeError> {
    if model.layers.len() != model.dims.layers {
        return Err(DecodeError::NotHeld);
    }
    let position = session.position;
    let mut x = embed(model, token)?;
    for l in 0..model.layers.len() {
        x = layer_through(model, session, l, &x, position, products)?;
    }
    finish(model, &x)
}

/// A token's embedding row, each half read exactly as binary32.
fn embed(model: &Model<'_>, token: u32) -> Result<Vec<u32>, DecodeError> {
    let hidden = model.dims.hidden;
    let id = usize::try_from(token).map_err(|_| DecodeError::TokenOutOfRange)?;
    if id >= model.dims.vocab {
        return Err(DecodeError::TokenOutOfRange);
    }
    (0..hidden)
        .map(|k| {
            let e = half(model.embedding, id * hidden + k)?;
            Ok(exact::round_exact(e.mantissa, e.exponent))
        })
        .collect()
}

/// The final norm of the last layer's row: the row the head reads.
fn finish(model: &Model<'_>, x: &[u32]) -> Result<Vec<u32>, DecodeError> {
    norm(
        x,
        model.final_norm.ok_or(DecodeError::NotHeld)?,
        model.epsilon,
    )
}

/// Layer `l` for one row `x` at `position`, its key and value added to the
/// cache: the row the next layer reads.
fn layer_through(
    model: &Model<'_>,
    session: &mut Session,
    l: usize,
    x: &[u32],
    position: u32,
    products: &mut dyn Products,
) -> Result<Vec<u32>, DecodeError> {
    let Dims {
        heads,
        kv_heads,
        head_dim,
        ..
    } = model.dims;
    let layer = model.layers.get(l).ok_or(DecodeError::NotHeld)?;
    let group = heads / kv_heads;
    let mut x = x.to_vec();
    let mut times = |which: Product, input: &[u32]| answered(model, products, which, input);
    let h = norm(&x, layer.input_norm, model.epsilon)?;
    let mut q = times(Product::Query(l), &h)?;
    let mut k = times(Product::Key(l), &h)?;
    let mut v = times(Product::Value(l), &h)?;
    if let Some(biases) = layer.biases {
        // As the export adds them: to the product, rounded once more.
        for (projected, bias) in [
            (&mut q, biases[0]),
            (&mut k, biases[1]),
            (&mut v, biases[2]),
        ] {
            for (i, x) in projected.iter_mut().enumerate() {
                let b = half(bias, i)?;
                *x = add(*x, exact::round_exact(b.mantissa, b.exponent))?;
            }
        }
    }
    for (projected, head_norm) in [(&mut q, layer.query_norm), (&mut k, layer.key_norm)] {
        for head in projected.chunks_mut(head_dim) {
            if let Some(weights) = head_norm {
                let normed = norm(head, weights, model.epsilon)?;
                head.copy_from_slice(&normed);
            }
            rotate(head, position, &model.frequencies)?;
        }
    }
    session.keys[l].extend_from_slice(&k);
    session.values[l].extend_from_slice(&v);
    let seen = session.keys[l].len() / (kv_heads * head_dim);
    let mut attended = Vec::with_capacity(heads * head_dim);
    for (hq, query) in q.chunks(head_dim).enumerate() {
        let kv = hq / group;
        let at = |t: usize| (t * kv_heads + kv) * head_dim;
        let exponents = (0..seen)
            .map(|t| {
                let key = &session.keys[l][at(t)..at(t) + head_dim];
                let mut row = Row::ZERO;
                for (&a, &b) in query.iter().zip(key) {
                    let (a, b) = (value(a)?, value(b)?);
                    add_to(&mut row, a.mantissa * b.mantissa, a.exponent + b.exponent);
                }
                exponent(exact::round_to_binary32(&row), model.root)
            })
            .collect::<Result<Vec<i64>, DecodeError>>()?;
        let weights = exact::softmax_exp2(&exponents);
        for j in 0..head_dim {
            let mut row = Row::ZERO;
            for (t, &p) in weights.iter().enumerate() {
                let v = value(session.values[l][at(t) + j])?;
                add_to(&mut row, i128::from(p) * v.mantissa, v.exponent - 30);
            }
            attended.push(exact::round_to_binary32(&row));
        }
    }
    let o = times(Product::Output(l), &attended)?;
    if session.probe == Some((l, Component::Attention)) {
        session.probed = Some(o.clone());
    }
    x = x
        .iter()
        .zip(&o)
        .map(|(&a, &b)| add(a, b))
        .collect::<Result<Vec<u32>, DecodeError>>()?;
    let h = norm(&x, layer.post_norm, model.epsilon)?;
    let g = times(Product::Gate(l), &h)?;
    let u = times(Product::Up(l), &h)?;
    let gated = g
        .iter()
        .zip(&u)
        .map(|(&g, &u)| {
            let g = value(g)?;
            let fixed = i64::try_from(shift_even(g.mantissa, g.exponent + 16))
                .map_err(|_| DecodeError::NormRefused)?;
            let a = exact::silu(fixed);
            let u = value(u)?;
            Ok(exact::round_exact(
                i128::from(a) * u.mantissa,
                u.exponent - 16,
            ))
        })
        .collect::<Result<Vec<u32>, DecodeError>>()?;
    let d = times(Product::Down(l), &gated)?;
    if session.probe == Some((l, Component::FeedForward)) {
        session.probed = Some(d.clone());
    }
    x = x
        .iter()
        .zip(&d)
        .map(|(&a, &b)| add(a, b))
        .collect::<Result<Vec<u32>, DecodeError>>()?;
    if session.probe == Some((l, Component::Residual)) {
        session.probed = Some(x.clone());
    }
    Ok(x)
}

/// The token id of the largest logit, the first on a tie, by the order of
/// binary32 values read from their bits.
#[must_use]
pub fn greedy(logits: &[u32]) -> Option<u32> {
    let key = |bits: u32| {
        if bits >> 31 == 1 {
            -i64::from(bits & 0x7fff_ffff)
        } else {
            i64::from(bits)
        }
    };
    let mut best: Option<(usize, i64)> = None;
    for (i, &bits) in logits.iter().enumerate() {
        if best.is_none_or(|(_, b)| key(bits) > b) {
            best = Some((i, key(bits)));
        }
    }
    best.and_then(|(i, _)| u32::try_from(i).ok())
}
