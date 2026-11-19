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
//! What: [`read`] finds a Qwen3-shaped model's weights in its fused ONNX
//! export by the nodes that read them; [`step`] runs one token through it and
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
    /// A layer's attention, whose node gives the head counts.
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

/// What a model's config gives that its graph does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
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
    input_norm: &'a [u8],
    query: Q4Matrix<'a>,
    key: Q4Matrix<'a>,
    value: Q4Matrix<'a>,
    output: Q4Matrix<'a>,
    query_norm: &'a [u8],
    key_norm: &'a [u8],
    post_norm: &'a [u8],
    gate: Q4Matrix<'a>,
    up: Q4Matrix<'a>,
    down: Q4Matrix<'a>,
}

/// A model read from its graph, its weights borrowed from the file's bytes.
pub struct Model<'a> {
    /// Its sizes.
    pub dims: Dims,
    embedding: &'a [u8],
    layers: Vec<Layer<'a>>,
    final_norm: &'a [u8],
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

/// A vector of `length` halves read by the node `name`.
fn halves<'a>(
    nodes: &[Node],
    initializers: &BTreeMap<String, Initializer<'a>>,
    name: &str,
    role: Role,
    length: usize,
) -> Result<&'a [u8], DecodeError> {
    let w =
        weight(node(nodes, name, role)?, initializers, 0).ok_or(DecodeError::Unrecognised(role))?;
    if w.data_type != 10 || w.bytes.len() != 2 * length {
        return Err(DecodeError::Unrecognised(role));
    }
    Ok(w.bytes)
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

/// Reads a Qwen3-shaped model from its fused ONNX export: each weight from
/// the node that reads it, the sizes from the weights and the attention
/// node, the rotary frequencies from `config` (EN-48).
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
    let gather = node(nodes, "/model/embed_tokens/Gather", Role::Embedding)?;
    let embedding =
        weight(gather, initializers, 0).ok_or(DecodeError::Unrecognised(Role::Embedding))?;
    let layers = (0..)
        .take_while(|l| {
            nodes
                .iter()
                .any(|n| n.name == format!("/model/layers.{l}/attn/GroupQueryAttention"))
        })
        .count();
    let attention = node(
        nodes,
        "/model/layers.0/attn/GroupQueryAttention",
        Role::Attention,
    )?;
    let refused = DecodeError::Unrecognised(Role::Attention);
    let heads =
        usize::try_from(int(attention, "num_heads").ok_or(refused)?).map_err(|_| refused)?;
    let kv_heads =
        usize::try_from(int(attention, "kv_num_heads").ok_or(refused)?).map_err(|_| refused)?;
    if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
        return Err(refused);
    }
    let query = matrix(
        nodes,
        initializers,
        "/model/layers.0/attn/q_proj/MatMul_Q4",
        Role::Query,
    )?;
    let gate = matrix(
        nodes,
        initializers,
        "/model/layers.0/mlp/gate_proj/MatMul_Q4",
        Role::Gate,
    )?;
    let hidden = query.columns;
    let head_dim = query.rows / heads;
    let vocab = embedding.bytes.len() / 2 / hidden.max(1);
    if embedding.data_type != 10
        || embedding.bytes.len() != 2 * vocab * hidden
        || head_dim * heads != query.rows
    {
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
    for l in 0..layers {
        let at = |rest: &str| format!("/model/layers.{l}/{rest}");
        let input_norm_name = if l == 0 {
            at("input_layernorm/LayerNorm")
        } else {
            at("input_layernorm/SkipLayerNorm")
        };
        let kv = kv_heads * head_dim;
        read_layers.push(Layer {
            input_norm: halves(
                nodes,
                initializers,
                &input_norm_name,
                Role::InputNorm,
                hidden,
            )?,
            query: shaped(
                matrix(
                    nodes,
                    initializers,
                    &at("attn/q_proj/MatMul_Q4"),
                    Role::Query,
                )?,
                heads * head_dim,
                hidden,
                Role::Query,
            )?,
            key: shaped(
                matrix(nodes, initializers, &at("attn/k_proj/MatMul_Q4"), Role::Key)?,
                kv,
                hidden,
                Role::Key,
            )?,
            value: shaped(
                matrix(
                    nodes,
                    initializers,
                    &at("attn/v_proj/MatMul_Q4"),
                    Role::Value,
                )?,
                kv,
                hidden,
                Role::Value,
            )?,
            output: shaped(
                matrix(
                    nodes,
                    initializers,
                    &at("attn/o_proj/MatMul_Q4"),
                    Role::Output,
                )?,
                hidden,
                heads * head_dim,
                Role::Output,
            )?,
            query_norm: halves(
                nodes,
                initializers,
                &at("attn/q_norm/SimplifiedLayerNormalization"),
                Role::QueryNorm,
                head_dim,
            )?,
            key_norm: halves(
                nodes,
                initializers,
                &at("attn/k_norm/SimplifiedLayerNormalization"),
                Role::KeyNorm,
                head_dim,
            )?,
            post_norm: halves(
                nodes,
                initializers,
                &at("post_attention_layernorm/SkipLayerNorm"),
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
    let final_norm = halves(
        nodes,
        initializers,
        &format!("/model/layers.{layers}/final_norm_layernorm/SkipLayerNorm"),
        Role::FinalNorm,
        hidden,
    )?;
    let head = u32::try_from(head_dim).map_err(|_| refused)?;
    let frequencies =
        exact::rope_frequencies(head, config.rope_theta, config.scaling).ok_or(refused)?;
    // floor(2^40 / sqrt(head_dim)): the floor root of the floored quotient.
    let root = i128::try_from(exact::integer_root((1u128 << 80) / u128::from(head)))
        .map_err(|_| refused)?;
    Ok(Model {
        dims,
        embedding: embedding.bytes,
        layers: read_layers,
        final_norm,
        frequencies,
        epsilon: units32(config.rms_norm_eps)?,
        root,
    })
}

/// What a run holds between steps: each layer's keys and values, binary32
/// bit patterns, position after position.
pub struct Session {
    keys: Vec<Vec<u32>>,
    values: Vec<Vec<u32>>,
    position: u32,
}

impl Session {
    /// A run that has seen no token.
    #[must_use]
    pub fn new(model: &Model<'_>) -> Self {
        Self {
            keys: vec![Vec::new(); model.dims.layers],
            values: vec![Vec::new(); model.dims.layers],
            position: 0,
        }
    }

    /// How many tokens it has seen.
    #[must_use]
    pub fn position(&self) -> u32 {
        self.position
    }
}

fn half(bytes: &[u8], i: usize) -> Result<Dyadic, DecodeError> {
    Dyadic::from_binary16(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
        .ok_or(DecodeError::Exact(ExactError::ScaleNotFinite))
}

fn value(bits: u32) -> Result<Dyadic, DecodeError> {
    Dyadic::from_binary32(bits).ok_or(DecodeError::Exact(ExactError::InputNotFinite))
}

fn add_to(row: &mut Row, product: i128, exponent: i32) {
    row.add_scaled(product.unsigned_abs(), exponent, product < 0);
}

/// RMSNorm of binary32 `x` with half weights: each value entering EN-47 in
/// units of `2^-16`, the normalised value times its weight rounded once.
fn norm(x: &[u32], weights: &[u8], epsilon: u64) -> Result<Vec<u32>, DecodeError> {
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
    let Dims {
        hidden,
        heads,
        kv_heads,
        head_dim,
        vocab,
        ..
    } = model.dims;
    let id = usize::try_from(token).map_err(|_| DecodeError::TokenOutOfRange)?;
    if id >= vocab {
        return Err(DecodeError::TokenOutOfRange);
    }
    let position = session.position;
    let mut x = (0..hidden)
        .map(|k| {
            let e = half(model.embedding, id * hidden + k)?;
            Ok(exact::round_exact(e.mantissa, e.exponent))
        })
        .collect::<Result<Vec<u32>, DecodeError>>()?;
    let group = heads / kv_heads;
    for (l, layer) in model.layers.iter().enumerate() {
        let h = norm(&x, layer.input_norm, model.epsilon)?;
        let mut q = exact::q4_product_with(&h, &layer.query, SCHEME)?;
        let mut k = exact::q4_product_with(&h, &layer.key, SCHEME)?;
        let v = exact::q4_product_with(&h, &layer.value, SCHEME)?;
        for head in q.chunks_mut(head_dim) {
            let normed = norm(head, layer.query_norm, model.epsilon)?;
            head.copy_from_slice(&normed);
            rotate(head, position, &model.frequencies)?;
        }
        for head in k.chunks_mut(head_dim) {
            let normed = norm(head, layer.key_norm, model.epsilon)?;
            head.copy_from_slice(&normed);
            rotate(head, position, &model.frequencies)?;
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
        let o = exact::q4_product_with(&attended, &layer.output, SCHEME)?;
        x = x
            .iter()
            .zip(&o)
            .map(|(&a, &b)| add(a, b))
            .collect::<Result<Vec<u32>, DecodeError>>()?;
        let h = norm(&x, layer.post_norm, model.epsilon)?;
        let g = exact::q4_product_with(&h, &layer.gate, SCHEME)?;
        let u = exact::q4_product_with(&h, &layer.up, SCHEME)?;
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
        let d = exact::q4_product_with(&gated, &layer.down, SCHEME)?;
        x = x
            .iter()
            .zip(&d)
            .map(|(&a, &b)| add(a, b))
            .collect::<Result<Vec<u32>, DecodeError>>()?;
    }
    let h = norm(&x, model.final_norm, model.epsilon)?;
    let logits = exact::dense_product_with(
        &h,
        Dense::F16(model.embedding),
        DenseLayout::OutputMajor,
        hidden,
        vocab,
        SCHEME,
    )?;
    session.position += 1;
    Ok(logits)
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
