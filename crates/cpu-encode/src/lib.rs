//! Token ids to a sentence embedding or label scores on the CPU, in integers
//! only.
//!
//! Why: row 9 of milestone 2 runs one embedding model (all-MiniLM-L6-v2) and
//! one classifier (DistilBERT SST-2) on the integer engine, so an encoder's
//! answer is the same bits on every machine, as a decoder's is
//! (Q_ENGINE_PLAN.md, "Integer only").
//!
//! How: as `hologram-ai-cpu-decode` holds a decoder's residual stream, as
//! binary32 bit patterns in `u32` touched only by integer code. Each stage
//! reads them exactly, computes on integers and rounds its result back once,
//! half to even: the 4-bit and dense products of `hologram-ai-exact` (EN-34 to
//! EN-41), softmax on the certified exponent table (EN-43, EN-44), LayerNorm
//! (EN-60) and GELU (EN-61). A stage that takes fixed point enters by one
//! declared narrowing to units of `2^-16`, rounded half to even.
//!
//! What: [`read`] takes the weights by the roles the page places them in
//! (`site/app/engine/onnx-roles.mjs`, GP-17); [`hidden`] runs a sentence
//! through every layer; [`embed`] pools it to unit length; [`classify`]
//! answers a classifier's label scores and [`label`] the first largest.

#![forbid(unsafe_code)]
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use hologram_ai_exact::{
    self as exact, Dense, DenseLayout, Dyadic, ExactError, Grouping, Q4Matrix, Q4Scales, Q4Zeros,
    Row, Scheme, Width,
};

/// Why an encoder could not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// `FL-70`: a role this pass reads is missing, or its weight is not the
    /// shape or format the role needs.
    Unrecognised(Role),
    /// `FL-71`: an exact stage refused its operands.
    Exact(ExactError),
    /// `FL-72`: a row is past LayerNorm's bounds (EN-60) or its root is 0, or
    /// a pooled sentence has length 0.
    NormRefused,
    /// `FL-73`: an empty sentence, a token id past the vocabulary, or more
    /// tokens than the position table holds.
    TokenOutOfRange,
}

impl From<ExactError> for EncodeError {
    fn from(error: ExactError) -> Self {
        Self::Exact(error)
    }
}

/// A role in the encoder, which a set of weights that lacks it is refused by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The word, position and token-type tables.
    Embedding,
    /// The LayerNorm after the embeddings.
    EmbeddingNorm,
    /// A layer's query, key, value and output projections and their biases.
    Attention,
    /// A layer's LayerNorm after attention.
    AttentionNorm,
    /// A layer's two feed-forward projections and their biases.
    FeedForward,
    /// A layer's LayerNorm after the feed-forward.
    OutputNorm,
    /// A classifier's two dense layers and their biases.
    Head,
}

/// A weight as the page hands it over: a `MatMulNBits` matrix at four bits in
/// blocks of 32, or a dense table, matrix or vector of `rows` by `columns`.
#[derive(Clone, Copy, Debug)]
pub enum Tensor<'a> {
    /// The packed codes and one scale per block; zero points are the default 8.
    Q4 {
        /// The codes, two to a byte, low half first.
        codes: &'a [u8],
        /// One scale per block of 32.
        scales: Q4Scales<'a>,
        /// Outputs (`N`).
        rows: usize,
        /// Inputs (`K`).
        columns: usize,
    },
    /// Values row after row.
    Dense {
        /// The values.
        values: Dense<'a>,
        /// Rows.
        rows: usize,
        /// Values per row.
        columns: usize,
    },
}

/// What the weights do not say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Attention heads.
    pub heads: usize,
    /// Layers.
    pub layers: usize,
    /// LayerNorm's epsilon in units of `2^-32`, rounded half to even from the
    /// config's number; the encoders' `1e-12` is 0 in these units.
    pub epsilon: u64,
}

/// A vector of `length` values.
#[derive(Clone, Copy, Debug)]
struct Vector<'a> {
    values: Dense<'a>,
}

#[derive(Clone, Copy, Debug)]
struct Norm<'a> {
    weight: Vector<'a>,
    bias: Vector<'a>,
}

#[derive(Clone, Copy, Debug)]
struct Linear<'a> {
    matrix: Q4Matrix<'a>,
    bias: Vector<'a>,
}

#[derive(Clone, Copy, Debug)]
struct Layer<'a> {
    query: Linear<'a>,
    key: Linear<'a>,
    value: Linear<'a>,
    output: Linear<'a>,
    attention_norm: Norm<'a>,
    up: Linear<'a>,
    down: Linear<'a>,
    output_norm: Norm<'a>,
}

#[derive(Clone, Copy, Debug)]
struct Head<'a> {
    pre: Dense<'a>,
    pre_bias: Vector<'a>,
    classifier: Dense<'a>,
    classifier_bias: Vector<'a>,
    labels: usize,
}

/// An encoder read from its weights.
#[derive(Clone, Debug)]
pub struct Model<'a> {
    word: Dense<'a>,
    position: Dense<'a>,
    token_type: Option<Dense<'a>>,
    embedding_norm: Norm<'a>,
    layers: Vec<Layer<'a>>,
    head: Option<Head<'a>>,
    width: usize,
    heads: usize,
    vocabulary: usize,
    positions: usize,
    epsilon: u64,
    root: i128,
}

/// The activation scheme every product takes its input on, as the decoder's:
/// 16-bit codes, one power of two per group of 32.
const SCHEME: Scheme = Scheme {
    width: Width::Sixteen,
    grouping: Grouping::Groups,
};

/// The largest `r` with `r^2 <= value`.
fn floor_root(value: u128) -> u128 {
    let (mut low, mut high) = (0u128, 1u128 << 64);
    while high - low > 1 {
        let middle = (low + high) / 2;
        if middle
            .checked_mul(middle)
            .is_some_and(|square| square <= value)
        {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn dense<'a>(
    tensors: &BTreeMap<String, Tensor<'a>>,
    name: &str,
    role: Role,
    rows: Option<usize>,
    columns: usize,
) -> Result<(Dense<'a>, usize), EncodeError> {
    match tensors.get(name) {
        Some(&Tensor::Dense {
            values,
            rows: r,
            columns: c,
        }) if c == columns && rows.is_none_or(|want| want == r) && r > 0 => Ok((values, r)),
        _ => Err(EncodeError::Unrecognised(role)),
    }
}

fn vector<'a>(
    tensors: &BTreeMap<String, Tensor<'a>>,
    name: &str,
    role: Role,
    length: usize,
) -> Result<Vector<'a>, EncodeError> {
    Ok(Vector {
        values: dense(tensors, name, role, Some(1), length)?.0,
    })
}

fn linear<'a>(
    tensors: &BTreeMap<String, Tensor<'a>>,
    prefix: &str,
    weight: &str,
    bias: &str,
    role: Role,
    (rows, columns): (usize, usize),
) -> Result<Linear<'a>, EncodeError> {
    let Some(&Tensor::Q4 {
        codes,
        scales,
        rows: r,
        columns: c,
    }) = tensors.get(&format!("{prefix}{weight}"))
    else {
        return Err(EncodeError::Unrecognised(role));
    };
    if r != rows || c != columns {
        return Err(EncodeError::Unrecognised(role));
    }
    Ok(Linear {
        matrix: Q4Matrix {
            codes,
            scales,
            zeros: Q4Zeros::Eight,
            rows,
            columns,
            block: 32,
        },
        bias: vector(tensors, &format!("{prefix}{bias}"), role, rows)?,
    })
}

fn norm<'a>(
    tensors: &BTreeMap<String, Tensor<'a>>,
    name: &str,
    role: Role,
    width: usize,
) -> Result<Norm<'a>, EncodeError> {
    Ok(Norm {
        weight: vector(tensors, name, role, width)?,
        bias: vector(tensors, &format!("{name}_b"), role, width)?,
    })
}

/// Reads an encoder from its weights, named by the roles the page places them
/// in (GP-17): `word`, `position`, `type` (BERT's, optional), `emb_norm(_b)`;
/// per layer `l<i>.w{q,k,v,o}`, `l<i>.b{q,k,v,o}`, `l<i>.attn_norm(_b)`,
/// `l<i>.w1`, `l<i>.b1`, `l<i>.w2`, `l<i>.b2`, `l<i>.out_norm(_b)`; and a
/// classifier's `pre`, `pre_b`, `cls`, `cls_b` (optional). The width,
/// vocabulary, feed-forward size and labels are the weights' own.
///
/// # Errors
///
/// [`EncodeError::Unrecognised`] for a role that is missing or not the shape
/// or format it needs, including a width that is not a whole number of heads.
pub fn read<'a>(
    tensors: &BTreeMap<String, Tensor<'a>>,
    config: Config,
) -> Result<Model<'a>, EncodeError> {
    let Some(&Tensor::Dense {
        values: word,
        rows: vocabulary,
        columns: width,
    }) = tensors.get("word")
    else {
        return Err(EncodeError::Unrecognised(Role::Embedding));
    };
    if width == 0 || config.heads == 0 || width % config.heads != 0 || config.layers == 0 {
        return Err(EncodeError::Unrecognised(Role::Attention));
    }
    let (position, positions) = dense(tensors, "position", Role::Embedding, None, width)?;
    let token_type = if tensors.contains_key("type") {
        Some(dense(tensors, "type", Role::Embedding, None, width)?.0)
    } else {
        None
    };
    let embedding_norm = norm(tensors, "emb_norm", Role::EmbeddingNorm, width)?;
    let feed = match tensors.get("l0.w1") {
        Some(&Tensor::Q4 { rows, .. }) => rows,
        _ => return Err(EncodeError::Unrecognised(Role::FeedForward)),
    };
    let layers = (0..config.layers)
        .map(|l| {
            let p = format!("l{l}.");
            let square = (width, width);
            Ok(Layer {
                query: linear(tensors, &p, "wq", "bq", Role::Attention, square)?,
                key: linear(tensors, &p, "wk", "bk", Role::Attention, square)?,
                value: linear(tensors, &p, "wv", "bv", Role::Attention, square)?,
                output: linear(tensors, &p, "wo", "bo", Role::Attention, square)?,
                attention_norm: norm(
                    tensors,
                    &format!("{p}attn_norm"),
                    Role::AttentionNorm,
                    width,
                )?,
                up: linear(tensors, &p, "w1", "b1", Role::FeedForward, (feed, width))?,
                down: linear(tensors, &p, "w2", "b2", Role::FeedForward, (width, feed))?,
                output_norm: norm(tensors, &format!("{p}out_norm"), Role::OutputNorm, width)?,
            })
        })
        .collect::<Result<Vec<Layer<'a>>, EncodeError>>()?;
    let head = if tensors.contains_key("cls") {
        let (pre, _) = dense(tensors, "pre", Role::Head, Some(width), width)?;
        let (classifier, labels) = dense(tensors, "cls", Role::Head, None, width)?;
        Some(Head {
            pre,
            pre_bias: vector(tensors, "pre_b", Role::Head, width)?,
            classifier,
            classifier_bias: vector(tensors, "cls_b", Role::Head, labels)?,
            labels,
        })
    } else {
        None
    };
    // floor(2^40 / sqrt(d)) for a head of d values: the floor root of the
    // floored quotient, one narrowing.
    let head_width = (width / config.heads) as u128;
    let root = i128::try_from(floor_root((1u128 << 80) / head_width))
        .map_err(|_| EncodeError::Unrecognised(Role::Attention))?;
    Ok(Model {
        word,
        position,
        token_type,
        embedding_norm,
        layers,
        head,
        width,
        heads: config.heads,
        vocabulary,
        positions,
        epsilon: config.epsilon,
        root,
    })
}

/// Value `i` of a dense table, exactly.
fn element(values: Dense<'_>, i: usize) -> Result<Dyadic, EncodeError> {
    let refused = EncodeError::Exact(ExactError::ScaleNotFinite);
    match values {
        Dense::F16(bytes) => bytes
            .get(2 * i..2 * i + 2)
            .and_then(|b| Dyadic::from_binary16(u16::from_le_bytes([b[0], b[1]])))
            .ok_or(refused),
        Dense::Bf16(bytes) => bytes
            .get(2 * i..2 * i + 2)
            .and_then(|b| Dyadic::from_bfloat16(u16::from_le_bytes([b[0], b[1]])))
            .ok_or(refused),
        Dense::F32(bytes) => bytes
            .get(4 * i..4 * i + 4)
            .and_then(|b| Dyadic::from_binary32(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .ok_or(refused),
    }
}

fn value(bits: u32) -> Result<Dyadic, EncodeError> {
    Dyadic::from_binary32(bits).ok_or(EncodeError::Exact(ExactError::InputNotFinite))
}

fn add_to(row: &mut Row, d: Dyadic) {
    row.add_scaled(d.mantissa.unsigned_abs(), d.exponent, d.mantissa < 0);
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

/// A binary32 value in units of `2^-16`, rounded half to even: the one
/// narrowing by which a value enters a fixed-point stage.
fn fixed(bits: u32) -> Result<i64, EncodeError> {
    let d = value(bits)?;
    i64::try_from(shift_even(d.mantissa, d.exponent + 16)).map_err(|_| EncodeError::NormRefused)
}

/// A product's outputs with the bias added, each rounded once more.
fn with_bias(product: &[u32], bias: Vector<'_>) -> Result<Vec<u32>, EncodeError> {
    product
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let mut row = Row::ZERO;
            add_to(&mut row, value(p)?);
            add_to(&mut row, element(bias.values, i)?);
            Ok(exact::round_to_binary32(&row))
        })
        .collect()
}

fn project(x: &[u32], linear: &Linear<'_>) -> Result<Vec<u32>, EncodeError> {
    with_bias(
        &exact::q4_product_with(x, &linear.matrix, SCHEME)?,
        linear.bias,
    )
}

/// LayerNorm of binary32 `x` (EN-60), each value entering in units of `2^-16`;
/// the normalised value times its weight plus its bias, rounded once.
fn layer_norm(x: &[u32], norm: &Norm<'_>, epsilon: u64) -> Result<Vec<u32>, EncodeError> {
    let fixed = x
        .iter()
        .map(|&b| fixed(b))
        .collect::<Result<Vec<i64>, EncodeError>>()?;
    let ones = vec![1i64 << 16; fixed.len()];
    let zeros = vec![0i64; fixed.len()];
    let normalised =
        exact::layer_norm(&fixed, &ones, &zeros, epsilon).ok_or(EncodeError::NormRefused)?;
    normalised
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let g = element(norm.weight.values, i)?;
            let mut row = Row::ZERO;
            add_to(
                &mut row,
                Dyadic {
                    mantissa: i128::from(n) * g.mantissa,
                    exponent: g.exponent - 16,
                },
            );
            add_to(&mut row, element(norm.bias.values, i)?);
            Ok(exact::round_to_binary32(&row))
        })
        .collect()
}

/// `a + b`, both binary32, rounded once.
fn add(a: u32, b: u32) -> Result<u32, EncodeError> {
    let mut row = Row::ZERO;
    add_to(&mut row, value(a)?);
    add_to(&mut row, value(b)?);
    Ok(exact::round_to_binary32(&row))
}

/// The softmax exponent of a score, in units of `2^-8` of a base-two
/// exponent, as the decoder takes it: `score log2(e) / sqrt(d) * 2^8`, from
/// the score rounded once to binary32, rounded half to even.
fn exponent(score: u32, root: i128) -> Result<i64, EncodeError> {
    let s = value(score)?;
    let product = s.mantissa * i128::from(exact::LOG2_E) * root;
    let scaled = shift_even(product, s.exponent + 8 - 32 - 40);
    Ok(i64::try_from(scaled).unwrap_or(if scaled < 0 { i64::MIN } else { i64::MAX }))
}

/// Every token of a sentence through every layer: its hidden state, binary32
/// bit patterns, one row of `width` per token. Every token attends to every
/// other: an encoder has no causal mask, and one sentence has no padding.
///
/// # Errors
///
/// [`EncodeError::TokenOutOfRange`] for an empty sentence, an id past the
/// vocabulary or more tokens than the position table holds;
/// [`EncodeError::NormRefused`] for a row LayerNorm does not take; and
/// [`EncodeError::Exact`] when a product refuses its input.
pub fn hidden(model: &Model<'_>, ids: &[u32]) -> Result<Vec<Vec<u32>>, EncodeError> {
    let d = model.width;
    if ids.is_empty() || ids.len() > model.positions {
        return Err(EncodeError::TokenOutOfRange);
    }
    let mut x = ids
        .iter()
        .enumerate()
        .map(|(t, &id)| {
            let id = usize::try_from(id).map_err(|_| EncodeError::TokenOutOfRange)?;
            if id >= model.vocabulary {
                return Err(EncodeError::TokenOutOfRange);
            }
            let row = (0..d)
                .map(|k| {
                    let mut sum = Row::ZERO;
                    add_to(&mut sum, element(model.word, id * d + k)?);
                    add_to(&mut sum, element(model.position, t * d + k)?);
                    if let Some(types) = model.token_type {
                        // One sentence: every token is of type 0.
                        add_to(&mut sum, element(types, k)?);
                    }
                    Ok(exact::round_to_binary32(&sum))
                })
                .collect::<Result<Vec<u32>, EncodeError>>()?;
            layer_norm(&row, &model.embedding_norm, model.epsilon)
        })
        .collect::<Result<Vec<Vec<u32>>, EncodeError>>()?;
    let head_width = d / model.heads;
    for layer in &model.layers {
        let q = x
            .iter()
            .map(|row| project(row, &layer.query))
            .collect::<Result<Vec<Vec<u32>>, EncodeError>>()?;
        let k = x
            .iter()
            .map(|row| project(row, &layer.key))
            .collect::<Result<Vec<Vec<u32>>, EncodeError>>()?;
        let v = x
            .iter()
            .map(|row| project(row, &layer.value))
            .collect::<Result<Vec<Vec<u32>>, EncodeError>>()?;
        let mut next = Vec::with_capacity(x.len());
        for (m, row) in x.iter().enumerate() {
            let mut attended = Vec::with_capacity(d);
            for h in 0..model.heads {
                let span = h * head_width..(h + 1) * head_width;
                let exponents = k
                    .iter()
                    .map(|key| {
                        let mut score = Row::ZERO;
                        for (&a, &b) in q[m][span.clone()].iter().zip(&key[span.clone()]) {
                            let (a, b) = (value(a)?, value(b)?);
                            add_to(
                                &mut score,
                                Dyadic {
                                    mantissa: a.mantissa * b.mantissa,
                                    exponent: a.exponent + b.exponent,
                                },
                            );
                        }
                        exponent(exact::round_to_binary32(&score), model.root)
                    })
                    .collect::<Result<Vec<i64>, EncodeError>>()?;
                let weights = exact::softmax_exp2(&exponents);
                for j in span.clone() {
                    let mut sum = Row::ZERO;
                    for (t, &p) in weights.iter().enumerate() {
                        let v = value(v[t][j])?;
                        add_to(
                            &mut sum,
                            Dyadic {
                                mantissa: i128::from(p) * v.mantissa,
                                exponent: v.exponent - 30,
                            },
                        );
                    }
                    attended.push(exact::round_to_binary32(&sum));
                }
            }
            let o = project(&attended, &layer.output)?;
            let residual = row
                .iter()
                .zip(&o)
                .map(|(&a, &b)| add(a, b))
                .collect::<Result<Vec<u32>, EncodeError>>()?;
            let after = layer_norm(&residual, &layer.attention_norm, model.epsilon)?;
            let up = project(&after, &layer.up)?
                .iter()
                .map(|&u| Ok(exact::round_exact(i128::from(exact::gelu(fixed(u)?)), -16)))
                .collect::<Result<Vec<u32>, EncodeError>>()?;
            let down = project(&up, &layer.down)?;
            let residual = after
                .iter()
                .zip(&down)
                .map(|(&a, &b)| add(a, b))
                .collect::<Result<Vec<u32>, EncodeError>>()?;
            next.push(layer_norm(&residual, &layer.output_norm, model.epsilon)?);
        }
        x = next;
    }
    Ok(x)
}

/// A sentence's embedding: its tokens' hidden states summed, each sum rounded
/// once to binary32 and entering in units of `2^-16`, then divided by the
/// sum's length, the integer square root of its sum of squares, and rounded
/// half to even to `2^-40` before binary32. The sum stands for the mean: unit
/// length cancels the count.
///
/// # Errors
///
/// As [`hidden`], and [`EncodeError::NormRefused`] for a sum of length 0.
pub fn embed(model: &Model<'_>, ids: &[u32]) -> Result<Vec<u32>, EncodeError> {
    let rows = hidden(model, ids)?;
    let sums = (0..model.width)
        .map(|k| {
            let mut sum = Row::ZERO;
            for row in &rows {
                add_to(&mut sum, value(row[k])?);
            }
            fixed(exact::round_to_binary32(&sum))
        })
        .collect::<Result<Vec<i64>, EncodeError>>()?;
    let squares: u128 = sums
        .iter()
        .map(|&s| u128::from(s.unsigned_abs()).pow(2))
        .sum();
    let length = floor_root(squares);
    if length == 0 {
        return Err(EncodeError::NormRefused);
    }
    Ok(sums
        .iter()
        .map(|&s| {
            let scaled = i128::from(s) << 40;
            let magnitude = scaled.unsigned_abs();
            let (quotient, rest) = (magnitude / length, magnitude % length);
            let up = 2 * rest > length || (2 * rest == length && quotient & 1 == 1);
            let rounded = (quotient + u128::from(up)) as i128;
            exact::round_exact(if s < 0 { -rounded } else { rounded }, -40)
        })
        .collect())
}

/// A classifier's label scores: the first token's hidden state through the
/// dense `pre` layer and its bias, ReLU, then the dense `cls` layer and its
/// bias, each product and each bias rounded once.
///
/// # Errors
///
/// As [`hidden`], and [`EncodeError::Unrecognised`] with [`Role::Head`] for
/// a model with no classifier.
pub fn classify(model: &Model<'_>, ids: &[u32]) -> Result<Vec<u32>, EncodeError> {
    let head = model.head.ok_or(EncodeError::Unrecognised(Role::Head))?;
    let rows = hidden(model, ids)?;
    let first = rows.first().ok_or(EncodeError::TokenOutOfRange)?;
    let d = model.width;
    let pre = exact::dense_product_with(first, head.pre, DenseLayout::OutputMajor, d, d, SCHEME)?;
    let pre = with_bias(&pre, head.pre_bias)?
        .iter()
        .map(|&bits| if bits >> 31 == 1 { 0 } else { bits })
        .collect::<Vec<u32>>();
    let scores = exact::dense_product_with(
        &pre,
        head.classifier,
        DenseLayout::OutputMajor,
        d,
        head.labels,
        SCHEME,
    )?;
    with_bias(&scores, head.classifier_bias)
}

/// The label of the largest score, the first on a tie, by the order of
/// binary32 values read from their bits.
#[must_use]
pub fn label(scores: &[u32]) -> Option<u32> {
    let key = |bits: u32| {
        if bits >> 31 == 1 {
            -i64::from(bits & 0x7fff_ffff)
        } else {
            i64::from(bits)
        }
    };
    let mut best: Option<(usize, i64)> = None;
    for (i, &bits) in scores.iter().enumerate() {
        if best.is_none_or(|(_, b)| key(bits) > b) {
            best = Some((i, key(bits)));
        }
    }
    best.and_then(|(i, _)| u32::try_from(i).ok())
}
