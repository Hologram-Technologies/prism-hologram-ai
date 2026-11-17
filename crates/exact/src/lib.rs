//! Matrix-vector products whose every group sum is exact.
//!
//! A floating-point product of the same weights and the same input differs
//! between machines: the order of a sum and the width of each step are the
//! hardware's. That is why a re-run on another device proves nothing about an
//! answer. Here the input is scaled by a power of two onto eight-bit integers,
//! each group of weights is decoded to the integers its format stores, and the
//! group's sum of products is taken by `uor-matmul`, whose accumulator is wide
//! enough that it cannot overflow. That sum has one value on every machine.
//!
//! What is left in floating point is applying each group's scale and adding
//! the groups, which is done in `f64`, one group after another, in the stored
//! order, with no fused operations. IEEE 754 fixes the result of each such
//! step, so the output has the same bits wherever it is computed: natively,
//! and in the WebAssembly module the page loads. Rounding the input onto eight
//! bits is the one approximation, and its bound is stated on [`quantise`].
//!
//! The weight layouts are the engine's own, read with no repack:
//!
//! - q4: four-bit codes, weight `g` at bits `4 * (g % 2)` of byte `g / 2`, a
//!   weight being its code less eight, then one scale per 32 weights, `f32` or
//!   `f16`. This is ONNX's `MatMulNBits` at four bits, block 32, no zero point.
//! - ternary: a sign plane and a nonzero plane of one bit per weight (bit `g`
//!   least significant first, sign 1 meaning +1), then one `f32` scale per 128
//!   weights. This is Bonsai's layout.

#![forbid(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use uor_matmul::{dot_ref, Alphabet, Codec, Full, Grid};

/// One weight or input value: an `i8`, admitted whole.
type Value = Alphabet<i8, Full<i8>>;

/// The largest magnitude an input is scaled onto.
pub const INPUT_LIMIT: i8 = 127;

/// Weights in one q4 scale group.
pub const Q4_GROUP: usize = 32;

/// Weights in one ternary scale group.
pub const TERNARY_GROUP: usize = 128;

/// The four-bit codes 0 to 15 stand for -8 to 7.
const Q4_TABLE: [Value; 16] = [
    Value::of(-8),
    Value::of(-7),
    Value::of(-6),
    Value::of(-5),
    Value::of(-4),
    Value::of(-3),
    Value::of(-2),
    Value::of(-1),
    Value::of(0),
    Value::of(1),
    Value::of(2),
    Value::of(3),
    Value::of(4),
    Value::of(5),
    Value::of(6),
    Value::of(7),
];

/// A ternary weight's two bits as one code, `nonzero + 2 * sign`.
const TERNARY_TABLE: [Value; 4] = [Value::of(0), Value::of(-1), Value::of(0), Value::of(1)];

/// Why a product could not be taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactError {
    /// `FL-49`: an input is not a finite number.
    InputNotFinite,
    /// `FL-50`: the row length is not a whole number of scale groups.
    GroupMisaligned,
    /// `FL-51`: a buffer is not the length the shape implies.
    LengthMismatch,
}

/// An input vector on eight-bit integers: `codes[i] * 2^-exponent` stands for
/// the `i`-th input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quantised {
    codes: Vec<i8>,
    exponent: i32,
}

impl Quantised {
    /// The integer codes.
    #[must_use]
    pub fn codes(&self) -> &[i8] {
        &self.codes
    }

    /// The power of two the codes are scaled by: a code `c` stands for
    /// `c * 2^-exponent`.
    #[must_use]
    pub fn exponent(&self) -> i32 {
        self.exponent
    }
}

/// `2^power` as an `f64`, built from its bits, so it is exactly that power
/// wherever it is computed. The powers asked for here lie in the normal range.
fn power_of_two(power: i32) -> f64 {
    let biased = (power.clamp(-1022, 1023) + 1023) as u64;
    f64::from_bits(biased << 52)
}

/// The largest exponent `e` with `largest * 2^e <= 127`, for `largest > 0`.
fn exponent_for(largest: f64) -> i32 {
    // The float's own exponent puts `e` within one; the two loops settle it,
    // each step an exact multiplication by a power of two.
    let bits = largest.to_bits();
    let own = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut e = 6 - own;
    let limit = f64::from(INPUT_LIMIT);
    while e > -1022 && largest * power_of_two(e) > limit {
        e -= 1;
    }
    while e < 1023 && largest * power_of_two(e + 1) <= limit {
        e += 1;
    }
    e
}

/// Scales `input` by one power of two onto the integers -127 to 127, each
/// rounded to the nearest, ties to even.
///
/// The power is the largest that keeps the largest input within 127, so each
/// input is off by at most half of `2^-exponent`, which is at most the largest
/// input divided by 127 (and by 254 when the largest input sets the power
/// exactly). An input of all zeros is all zero codes at exponent 0.
pub fn quantise(input: &[f32]) -> Result<Quantised, ExactError> {
    let mut largest = 0.0_f64;
    for &x in input {
        if !x.is_finite() {
            return Err(ExactError::InputNotFinite);
        }
        largest = largest.max(f64::from(x).abs());
    }
    if largest == 0.0 {
        return Ok(Quantised {
            codes: vec![0; input.len()],
            exponent: 0,
        });
    }
    let exponent = exponent_for(largest);
    let scale = power_of_two(exponent);
    let codes = input
        .iter()
        .map(|&x| {
            let scaled = (f64::from(x) * scale).round_ties_even();
            scaled.clamp(-f64::from(INPUT_LIMIT), f64::from(INPUT_LIMIT)) as i8
        })
        .collect();
    Ok(Quantised { codes, exponent })
}

/// How a q4 matrix stores its scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4Scales<'a> {
    /// One little-endian `f32` per group.
    F32(&'a [u8]),
    /// One little-endian IEEE half per group, widened exactly.
    F16(&'a [u8]),
}

/// An IEEE half as the `f32` it stands for; every half is one exactly.
fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    let magnitude = match exponent {
        0 => fraction * (power_of_two(-24) as f32),
        31 if fraction == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + fraction / 1024.0) * (power_of_two(exponent - 15) as f32),
    };
    sign * magnitude
}

impl Q4Scales<'_> {
    fn bytes_per_scale(self) -> usize {
        match self {
            Self::F32(_) => 4,
            Self::F16(_) => 2,
        }
    }

    fn len(self) -> usize {
        match self {
            Self::F32(bytes) | Self::F16(bytes) => bytes.len(),
        }
    }

    fn at(self, group: usize) -> f64 {
        match self {
            Self::F32(bytes) => {
                let at = group * 4;
                let word = [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]];
                f64::from(f32::from_le_bytes(word))
            }
            Self::F16(bytes) => {
                let at = group * 2;
                f64::from(half(u16::from_le_bytes([bytes[at], bytes[at + 1]])))
            }
        }
    }
}

/// The exact sum of products of each group of a q4 row with the input codes,
/// row after row, group after group.
pub fn q4_group_sums(
    input: &Quantised,
    codes: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<i64>, ExactError> {
    if !columns.is_multiple_of(Q4_GROUP) {
        return Err(ExactError::GroupMisaligned);
    }
    if input.codes.len() != columns || codes.len() != rows * columns / 2 {
        return Err(ExactError::LengthMismatch);
    }
    let grid = Grid::<'_, i8, Full<i8>, 16>::new(&Q4_TABLE);
    let activations = uor_matmul::as_alphabet_full(&input.codes);
    let mut weights = [Value::of(0); Q4_GROUP];
    let mut sums = Vec::with_capacity(rows * columns / Q4_GROUP);
    for row in 0..rows {
        for group in 0..columns / Q4_GROUP {
            let first = row * columns + group * Q4_GROUP;
            for (i, weight) in weights.iter_mut().enumerate() {
                let g = first + i;
                // Low nibble first, as `uor-matmul`'s `Packed` reads a byte.
                let nibble = (codes[g / 2] >> (4 * (g % 2))) & 0x0f;
                *weight = grid.decode_element(u16::from(nibble), 0);
            }
            let start = group * Q4_GROUP;
            let sum = dot_ref(&activations[start..start + Q4_GROUP], &weights);
            sums.push(sum as i64);
        }
    }
    Ok(sums)
}

/// The exact sum of products of each group of a ternary row with the input
/// codes, row after row, group after group.
pub fn ternary_group_sums(
    input: &Quantised,
    signs: &[u8],
    nonzeros: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<i64>, ExactError> {
    if !columns.is_multiple_of(TERNARY_GROUP) {
        return Err(ExactError::GroupMisaligned);
    }
    let plane = rows * columns / 8;
    if input.codes.len() != columns || signs.len() != plane || nonzeros.len() != plane {
        return Err(ExactError::LengthMismatch);
    }
    let grid = Grid::<'_, i8, Full<i8>, 4>::new(&TERNARY_TABLE);
    let activations = uor_matmul::as_alphabet_full(&input.codes);
    let mut weights = [Value::of(0); TERNARY_GROUP];
    let mut sums = Vec::with_capacity(rows * columns / TERNARY_GROUP);
    for row in 0..rows {
        for group in 0..columns / TERNARY_GROUP {
            let first = row * columns + group * TERNARY_GROUP;
            for (i, weight) in weights.iter_mut().enumerate() {
                let g = first + i;
                let bit = |plane: &[u8]| u16::from((plane[g / 8] >> (g % 8)) & 1);
                *weight = grid.decode_element(bit(nonzeros) + 2 * bit(signs), 0);
            }
            let start = group * TERNARY_GROUP;
            let sum = dot_ref(&activations[start..start + TERNARY_GROUP], &weights);
            sums.push(sum as i64);
        }
    }
    Ok(sums)
}

/// Each row's output from its exact group sums: each sum times its group's
/// scale, added in `f64` group after group, then scaled back by the input's
/// power of two and rounded once to `f32`.
fn rows_from(sums: &[i64], scale: impl Fn(usize) -> f64, groups: usize, exponent: i32) -> Vec<f32> {
    let back = power_of_two(-exponent);
    sums.chunks(groups.max(1))
        .enumerate()
        .map(|(row, row_sums)| {
            let mut total = 0.0_f64;
            for (group, &sum) in row_sums.iter().enumerate() {
                total += sum as f64 * scale(row * groups + group);
            }
            (total * back) as f32
        })
        .collect()
}

/// A q4 matrix of `rows` by `columns` times `input`: one output per row.
pub fn q4_matvec(
    input: &[f32],
    codes: &[u8],
    scales: Q4Scales<'_>,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, ExactError> {
    let quantised = quantise(input)?;
    let sums = q4_group_sums(&quantised, codes, rows, columns)?;
    let groups = columns / Q4_GROUP;
    if scales.len() != rows * groups * scales.bytes_per_scale() {
        return Err(ExactError::LengthMismatch);
    }
    Ok(rows_from(
        &sums,
        |g| scales.at(g),
        groups,
        quantised.exponent,
    ))
}

/// A ternary matrix of `rows` by `columns`, in its three planes, times
/// `input`: one output per row.
pub fn ternary_matvec(
    input: &[f32],
    signs: &[u8],
    nonzeros: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, ExactError> {
    let quantised = quantise(input)?;
    let sums = ternary_group_sums(&quantised, signs, nonzeros, rows, columns)?;
    let groups = columns / TERNARY_GROUP;
    if scales.len() != rows * groups * 4 {
        return Err(ExactError::LengthMismatch);
    }
    Ok(rows_from(
        &sums,
        |g| Q4Scales::F32(scales).at(g),
        groups,
        quantised.exponent,
    ))
}

/// Which stored layout a request's tensor is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// q4 codes, then `f32` scales.
    Q4F32,
    /// q4 codes, then IEEE-half scales.
    Q4F16,
    /// A sign plane, a nonzero plane, then `f32` scales.
    Ternary,
}

/// A product asked for as one buffer, the way the page's module is handed it:
/// the row count and the column count, four bytes each, least significant
/// first; the input, one little-endian `f32` per column; then the tensor's
/// bytes as the engine stores them. Every length is checked before any slice
/// is taken, so a short or misshapen buffer is a registered failure.
pub fn request_product(bytes: &[u8], layout: Layout) -> Result<Vec<f32>, ExactError> {
    let word = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize)
            .ok_or(ExactError::LengthMismatch)
    };
    let rows = word(0)?;
    let columns = word(4)?;
    let input_end = columns
        .checked_mul(4)
        .and_then(|n| n.checked_add(8))
        .ok_or(ExactError::LengthMismatch)?;
    let input: Vec<f32> = bytes
        .get(8..input_end)
        .ok_or(ExactError::LengthMismatch)?
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let tensor = bytes.get(input_end..).ok_or(ExactError::LengthMismatch)?;
    let weights = rows
        .checked_mul(columns)
        .ok_or(ExactError::LengthMismatch)?;
    match layout {
        Layout::Q4F32 | Layout::Q4F16 => {
            if !columns.is_multiple_of(Q4_GROUP) {
                return Err(ExactError::GroupMisaligned);
            }
            let (codes, scales) = tensor
                .split_at_checked(weights / 2)
                .ok_or(ExactError::LengthMismatch)?;
            let scales = if layout == Layout::Q4F16 {
                Q4Scales::F16(scales)
            } else {
                Q4Scales::F32(scales)
            };
            q4_matvec(&input, codes, scales, rows, columns)
        }
        Layout::Ternary => {
            if !columns.is_multiple_of(TERNARY_GROUP) {
                return Err(ExactError::GroupMisaligned);
            }
            let plane = weights / 8;
            let (signs, rest) = tensor
                .split_at_checked(plane)
                .ok_or(ExactError::LengthMismatch)?;
            let (nonzeros, scales) = rest
                .split_at_checked(plane)
                .ok_or(ExactError::LengthMismatch)?;
            ternary_matvec(&input, signs, nonzeros, scales, rows, columns)
        }
    }
}
