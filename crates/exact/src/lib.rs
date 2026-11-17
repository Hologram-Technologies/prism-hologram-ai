//! Matrix-vector products computed in integers from the stored bytes to one
//! final rounding.
//!
//! A floating-point product of the same weights and the same input differs
//! between machines: the order of a sum and the width of each step are the
//! hardware's. That is why a re-run on another device proves nothing about an
//! answer. Here no floating-point operation is used at all:
//!
//! 1. Every stored number is read as the exact integer pair it stands for:
//!    an IEEE `f32` or half is `mantissa * 2^exponent` with both integers, read
//!    from its bits ([`Dyadic`]). Nothing is rounded on reading.
//! 2. The input is scaled by one power of two onto the integers -127 to 127,
//!    each rounded to the nearest, ties to even, by integer shifts
//!    ([`quantise`]). This is the one approximation, and its bound is stated
//!    there.
//! 3. Each group's sum of products is taken by `uor-matmul`, an `i32` at most
//!    `32 * 8 * 127` (q4) or `128 * 127` (ternary) in magnitude.
//! 4. A row's groups are combined exactly: each sum times its scale's
//!    mantissa, shifted to the row's smallest scale exponent, added in `i128`.
//!    The row's value is that integer times one power of two. A row whose
//!    scales are too far apart for `i128` to hold the sum is refused
//!    ([`ExactError::ScaleSpanTooWide`]), never approximated.
//! 5. That value is rounded once to the nearest `f32`, ties to even, by
//!    building the `f32`'s bits from integers ([`Dyadic::to_f32`]).
//!
//! Every step is integer arithmetic with one stated result, so the output has
//! the same bits on every machine: natively and in the WebAssembly module the
//! page loads.
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

/// The bits a row's exact sum may use, its sign aside: `i128` less one, so
/// that adding one more term to a sum within the bound cannot overflow.
const ROW_BITS: u32 = 126;

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
    /// `FL-50`: an input is not a finite number.
    InputNotFinite,
    /// `FL-51`: the row length is not a whole number of scale groups.
    GroupMisaligned,
    /// `FL-52`: a buffer is not the length the shape implies.
    LengthMismatch,
    /// `FL-53`: a stored scale is not a finite number.
    ScaleNotFinite,
    /// `FL-54`: a row's scales lie too many powers of two apart for its sum to
    /// be held exactly in `i128`.
    ScaleSpanTooWide,
}

/// A number that is exactly `mantissa * 2^exponent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dyadic {
    /// The integer the power of two multiplies.
    pub mantissa: i128,
    /// The power of two.
    pub exponent: i32,
}

/// A float's sign, significand and exponent, as `width` describes it.
struct Format {
    fraction_bits: u32,
    exponent_bits: u32,
    bias: i32,
}

const BINARY32: Format = Format {
    fraction_bits: 23,
    exponent_bits: 8,
    bias: 127,
};

const BINARY16: Format = Format {
    fraction_bits: 10,
    exponent_bits: 5,
    bias: 15,
};

impl Format {
    /// The exact number `bits` stands for, or `None` for an infinity or NaN.
    fn read(&self, bits: u32) -> Option<Dyadic> {
        let fraction = bits & ((1 << self.fraction_bits) - 1);
        let biased = (bits >> self.fraction_bits) & ((1 << self.exponent_bits) - 1);
        let negative = (bits >> (self.fraction_bits + self.exponent_bits)) & 1 == 1;
        let lowest = 1 - self.bias - self.fraction_bits as i32;
        let (magnitude, exponent) = if biased == (1 << self.exponent_bits) - 1 {
            return None;
        } else if biased == 0 {
            // Subnormal: the fraction counts units of the smallest step.
            (fraction, lowest)
        } else {
            (
                fraction | (1 << self.fraction_bits),
                lowest + biased as i32 - 1,
            )
        };
        let magnitude = i128::from(magnitude);
        Some(Dyadic {
            mantissa: if negative { -magnitude } else { magnitude },
            exponent,
        })
    }
}

/// `magnitude / 2^shift`, rounded to the nearest integer, ties to even.
fn shift_right_even(magnitude: u128, shift: u32) -> u128 {
    if shift == 0 {
        return magnitude;
    }
    if shift > 128 {
        return 0;
    }
    let kept = if shift == 128 { 0 } else { magnitude >> shift };
    let dropped = magnitude - if shift == 128 { 0 } else { kept << shift };
    let half = 1u128 << (shift - 1);
    if dropped > half || (dropped == half && kept & 1 == 1) {
        kept + 1
    } else {
        kept
    }
}

/// How many bits `magnitude` needs.
fn bits_of(magnitude: u128) -> u32 {
    128 - magnitude.leading_zeros()
}

impl Dyadic {
    /// Reads a little-endian IEEE `f32`.
    #[must_use]
    pub fn from_f32_bits(bits: u32) -> Option<Self> {
        BINARY32.read(bits)
    }

    /// Reads an IEEE half.
    #[must_use]
    pub fn from_f16_bits(bits: u16) -> Option<Self> {
        BINARY16.read(u32::from(bits))
    }

    /// The nearest `f32`, ties to even, with infinity past the largest.
    ///
    /// The bits are put together from integers: the exponent field, then the
    /// significand rounded once. A result of zero is `+0.0`.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        let sign = if self.mantissa < 0 { 1u32 << 31 } else { 0 };
        let magnitude = self.mantissa.unsigned_abs();
        if magnitude == 0 {
            return 0.0;
        }
        let width = bits_of(magnitude) as i32;
        // The value lies in [2^top, 2^(top + 1)).
        let top = self.exponent + width - 1;
        const INFINITY: u32 = 0x7f80_0000;
        if top > 127 {
            return f32::from_bits(sign | INFINITY);
        }
        // The power of two the last significand bit stands for: 2^(top - 23)
        // for a normal result, 2^-149 below the normal range.
        let unit = (top - 23).max(-149);
        let significand = if unit >= self.exponent {
            shift_right_even(magnitude, (unit - self.exponent) as u32)
        } else {
            // Exact: the value has fewer bits than a significand holds.
            magnitude << (self.exponent - unit) as u32
        };
        // A subnormal's bits are its significand (field 0). A normal
        // significand carries its leading 2^23, which adds one to the field,
        // and one that rounded up to 2^24 carries once more: both are this
        // addition, and a carry past the largest field is infinity.
        let field = (unit + 149) as u32;
        let bits = (field << 23) + significand as u32;
        if bits >= INFINITY {
            return f32::from_bits(sign | INFINITY);
        }
        f32::from_bits(sign | bits)
    }
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

/// Whether `a` is larger in magnitude than `b`. Both are read from an `f32`,
/// so each mantissa is below `2^24`.
fn larger(a: Dyadic, b: Dyadic) -> bool {
    let (ma, mb) = (a.mantissa.unsigned_abs(), b.mantissa.unsigned_abs());
    if mb == 0 {
        return ma != 0;
    }
    if ma == 0 {
        return false;
    }
    let (ta, tb) = (
        a.exponent + bits_of(ma) as i32,
        b.exponent + bits_of(mb) as i32,
    );
    if ta != tb {
        return ta > tb;
    }
    // Same leading power: their exponents differ by fewer than 24.
    let low = a.exponent.min(b.exponent);
    (ma << (a.exponent - low) as u32) > (mb << (b.exponent - low) as u32)
}

/// Scales `input` by one power of two onto the integers -127 to 127, each
/// rounded to the nearest, ties to even.
///
/// The power is the largest that keeps the largest input within 127, so each
/// input is off by at most half of `2^-exponent`, which is at most the largest
/// input divided by 127 (and by 254 when the largest input sets the power
/// exactly). An input of all zeros is all zero codes at exponent 0.
pub fn quantise(input: &[f32]) -> Result<Quantised, ExactError> {
    let read: Vec<Dyadic> = input
        .iter()
        .map(|x| Dyadic::from_f32_bits(x.to_bits()).ok_or(ExactError::InputNotFinite))
        .collect::<Result<_, _>>()?;
    let zero = Dyadic {
        mantissa: 0,
        exponent: 0,
    };
    let largest = read
        .iter()
        .fold(zero, |m, &x| if larger(x, m) { x } else { m });
    let top = largest.mantissa.unsigned_abs();
    if top == 0 {
        return Ok(Quantised {
            codes: vec![0; input.len()],
            exponent: 0,
        });
    }
    // The largest `k` with `top * 2^k <= 127`: `top * 2^k` in [64, 128) at
    // `k = 7 - bits`, one step down if that is past 127.
    let mut k = 7 - bits_of(top) as i32;
    let past = if k >= 0 {
        top << k as u32 > 127
    } else {
        top > 127u128 << (-k) as u32
    };
    if past {
        k -= 1;
    }
    let exponent = k - largest.exponent;
    let codes = read
        .iter()
        .map(|x| {
            let magnitude = x.mantissa.unsigned_abs();
            let shift = x.exponent + exponent;
            let code = if shift >= 0 {
                magnitude << shift as u32
            } else {
                shift_right_even(magnitude, (-shift) as u32)
            };
            // Within 127 by the choice of exponent; the bound is restated.
            let code = code.min(INPUT_LIMIT as u128) as i8;
            if x.mantissa < 0 {
                -code
            } else {
                code
            }
        })
        .collect();
    Ok(Quantised { codes, exponent })
}

/// How a q4 matrix stores its scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4Scales<'a> {
    /// One little-endian `f32` per group.
    F32(&'a [u8]),
    /// One little-endian IEEE half per group.
    F16(&'a [u8]),
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

    /// Group `group`'s scale, exactly. The caller has checked the length.
    fn at(self, group: usize) -> Result<Dyadic, ExactError> {
        let read = match self {
            Self::F32(bytes) => {
                let at = group * 4;
                let word = [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]];
                Dyadic::from_f32_bits(u32::from_le_bytes(word))
            }
            Self::F16(bytes) => {
                let at = group * 2;
                Dyadic::from_f16_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
            }
        };
        read.ok_or(ExactError::ScaleNotFinite)
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

/// One row's exact value from its group sums and its groups' scales: the sum
/// of each group sum times its scale, times `2^-input_exponent`.
///
/// Each product is shifted to the row's smallest scale exponent and added in
/// `i128`. Before any addition the row is checked to fit: every product needs
/// at most its own bits plus its shift, and the sum of `n` of them at most
/// `ceil(log2 n)` bits more.
fn exact_row(sums: &[i64], scales: &[Dyadic], input_exponent: i32) -> Result<Dyadic, ExactError> {
    if sums.len() != scales.len() {
        return Err(ExactError::LengthMismatch);
    }
    let terms: Vec<(i128, i32)> = sums
        .iter()
        .zip(scales)
        .filter(|(&sum, scale)| sum != 0 && scale.mantissa != 0)
        .map(|(&sum, scale)| {
            i128::from(sum)
                .checked_mul(scale.mantissa)
                .map(|m| (m, scale.exponent))
                .ok_or(ExactError::ScaleSpanTooWide)
        })
        .collect::<Result<_, _>>()?;
    let Some(lowest) = terms.iter().map(|&(_, e)| e).min() else {
        return Ok(Dyadic {
            mantissa: 0,
            exponent: 0,
        });
    };
    let widest = terms
        .iter()
        .map(|&(m, e)| bits_of(m.unsigned_abs()) as i64 + i64::from(e - lowest))
        .max()
        .unwrap_or(0);
    let carry = i64::from(bits_of(terms.len() as u128));
    if widest + carry > i64::from(ROW_BITS) {
        return Err(ExactError::ScaleSpanTooWide);
    }
    let total = terms
        .iter()
        .map(|&(m, e)| m << (e - lowest) as u32)
        .sum::<i128>();
    Ok(Dyadic {
        mantissa: total,
        exponent: lowest - input_exponent,
    })
}

/// Each row's output from its exact group sums, rounded once to `f32`.
fn rows_from(
    sums: &[i64],
    scale: impl Fn(usize) -> Result<Dyadic, ExactError>,
    groups: usize,
    exponent: i32,
) -> Result<Vec<f32>, ExactError> {
    sums.chunks(groups.max(1))
        .enumerate()
        .map(|(row, row_sums)| {
            let scales = (0..row_sums.len())
                .map(|group| scale(row * groups + group))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(exact_row(row_sums, &scales, exponent)?.to_f32())
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
    rows_from(&sums, |g| scales.at(g), groups, quantised.exponent)
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
    rows_from(
        &sums,
        |g| Q4Scales::F32(scales).at(g),
        groups,
        quantised.exponent,
    )
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
