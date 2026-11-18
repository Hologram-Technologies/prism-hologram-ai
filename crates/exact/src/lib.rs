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
//! 4. A row's groups are combined exactly in `uor-matmul`'s complete
//!    register (`Complete`): each sum times its scale's mantissa is placed at
//!    its scale's exponent less the input's, with every bit kept ([`Row`]).
//! 5. The row is rounded once to the nearest binary32, ties to even, by reading
//!    the register's leading bits and its sticky bit and building the binary32
//!    bit pattern ([`round_to_binary32`]).
//!
//! The crate has no floating-point type at all: it takes the input as binary32
//! bit patterns and returns the outputs as binary32 bit patterns, and works in
//! integers between. Every step has one stated result, so the output has the
//! same bits on every machine: natively and in the WebAssembly module the page
//! loads.
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

use uor_matmul::core_types::Complete;
use uor_matmul::{dot_ref, Alphabet, Codec, Full, Grid};

/// One weight or input value: an `i8`, admitted whole.
type Value = Alphabet<i8, Full<i8>>;

/// The largest magnitude an input is scaled onto.
pub const INPUT_LIMIT: i8 = 127;

/// Weights in one q4 scale group.
pub const Q4_GROUP: usize = 32;

/// Weights in one ternary scale group.
pub const TERNARY_GROUP: usize = 128;

/// The lowest power of two a row's register holds. A row's terms are a group
/// sum (below `2^15`) times a scale's mantissa (below `2^24`), placed at the
/// scale's exponent (`-149` to `127`) less the input's power (`-121` to
/// `155`, from the largest and the smallest finite binary32 input). Every bit
/// of every term, and of their sum over fewer than `2^32` groups, therefore
/// lies between `2^-304` and `2^345`.
const ROW_MIN_EXP: i32 = -320;

/// The register's 64-bit limbs: 768 bits from `2^-320`, past `2^345`.
const ROW_LIMBS: usize = 12;

/// One row's exact value: `uor-matmul`'s complete register at the width above.
pub type Row = Complete<ROW_LIMBS, ROW_MIN_EXP>;

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
}

/// A number that is exactly `mantissa * 2^exponent`. Read from a float, the
/// mantissa is odd, or zero with exponent 0.
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
        // Odd, or zero: a row's span is then counted from the bits each
        // scale really has, not from its stored width.
        let (magnitude, exponent) = if magnitude == 0 {
            (0, 0)
        } else {
            let zeros = magnitude.trailing_zeros();
            (magnitude >> zeros, exponent + zeros as i32)
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
    /// Reads an IEEE binary32 bit pattern.
    #[must_use]
    pub fn from_binary32(bits: u32) -> Option<Self> {
        BINARY32.read(bits)
    }

    /// Reads an IEEE half bit pattern.
    #[must_use]
    pub fn from_binary16(bits: u16) -> Option<Self> {
        BINARY16.read(u32::from(bits))
    }
}

/// The binary32 bit pattern nearest a row's exact value, ties to even, with
/// infinity past the largest finite one and `+0` for zero.
///
/// The leading one fixes the exponent; the 24 bits from it (fewer below the
/// normal range, whose last bit stands for `2^-149`) are the significand; the
/// bit after them and whether any bit below that is set decide the one
/// rounding.
#[must_use]
pub fn round_to_binary32(row: &Row) -> u32 {
    const INFINITY: u32 = 0x7f80_0000;
    let Some(high) = row.magnitude_high_bit() else {
        return 0;
    };
    let sign = if row.is_negative() { 1u32 << 31 } else { 0 };
    // The value lies in [2^top, 2^(top + 1)).
    let top = ROW_MIN_EXP + high as i32;
    if top > 127 {
        return sign | INFINITY;
    }
    // The power of two the last significand bit stands for, and its bit.
    let unit = (top - 23).max(-149);
    let last = (unit - ROW_MIN_EXP) as u32;
    let mut significand = 0u32;
    for bit in (last..=high).rev() {
        significand = significand * 2 + u32::from(row.magnitude_bit(bit));
    }
    // Below the register's floor nothing is set, so `last` is at least 1 here.
    let guard = last > 0 && row.magnitude_bit(last - 1);
    let sticky = last > 1 && row.magnitude_any_below(last - 1);
    if guard && (sticky || significand % 2 == 1) {
        significand += 1;
    }
    // A subnormal's bits are its significand (field 0). A normal significand
    // carries its leading 2^23, which adds one to the field, and one that
    // rounded up to 2^24 carries once more: both are this addition, and a
    // carry past the largest field is infinity.
    let bits = (((unit + 149) as u32) << 23) + significand;
    if bits >= INFINITY {
        return sign | INFINITY;
    }
    sign | bits
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

/// Whether `a` is larger in magnitude than `b`. Both are read from a binary32,
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
pub fn quantise(input: &[u32]) -> Result<Quantised, ExactError> {
    let read: Vec<Dyadic> = input
        .iter()
        .map(|&bits| Dyadic::from_binary32(bits).ok_or(ExactError::InputNotFinite))
        .collect::<Result<Vec<Dyadic>, ExactError>>()?;
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
                Dyadic::from_binary32(u32::from_le_bytes(word))
            }
            Self::F16(bytes) => {
                let at = group * 2;
                Dyadic::from_binary16(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
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
/// of each group sum times its scale, times `2^-input_exponent`, held in the
/// complete register with every bit.
#[must_use]
pub fn exact_row(sums: &[i64], scales: &[Dyadic], input_exponent: i32) -> Row {
    let mut row = Row::ZERO;
    for (&sum, scale) in sums.iter().zip(scales) {
        // A group sum is below 2^15 and a mantissa below 2^24: the product
        // is far inside i128.
        let product = i128::from(sum) * scale.mantissa;
        row.add_scaled(
            product.unsigned_abs(),
            scale.exponent - input_exponent,
            product < 0,
        );
    }
    row
}

/// Each row's output from its exact group sums, rounded once to binary32.
fn rows_from(
    sums: &[i64],
    scale: impl Fn(usize) -> Result<Dyadic, ExactError>,
    groups: usize,
    exponent: i32,
) -> Result<Vec<u32>, ExactError> {
    sums.chunks(groups.max(1))
        .enumerate()
        .map(|(row, row_sums)| {
            let scales = (0..row_sums.len())
                .map(|group| scale(row * groups + group))
                .collect::<Result<Vec<Dyadic>, ExactError>>()?;
            Ok(round_to_binary32(&exact_row(row_sums, &scales, exponent)))
        })
        .collect()
}

/// A q4 matrix of `rows` by `columns` times `input`, both as binary32 bit
/// patterns: one output bit pattern per row.
pub fn q4_matvec(
    input: &[u32],
    codes: &[u8],
    scales: Q4Scales<'_>,
    rows: usize,
    columns: usize,
) -> Result<Vec<u32>, ExactError> {
    let quantised = quantise(input)?;
    let sums = q4_group_sums(&quantised, codes, rows, columns)?;
    let groups = columns / Q4_GROUP;
    if scales.len() != rows * groups * scales.bytes_per_scale() {
        return Err(ExactError::LengthMismatch);
    }
    rows_from(&sums, |g| scales.at(g), groups, quantised.exponent)
}

/// A ternary matrix of `rows` by `columns`, in its three planes, times
/// `input`, as binary32 bit patterns: one output bit pattern per row.
pub fn ternary_matvec(
    input: &[u32],
    signs: &[u8],
    nonzeros: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<u32>, ExactError> {
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
pub fn request_product(bytes: &[u8], layout: Layout) -> Result<Vec<u32>, ExactError> {
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
    let input: Vec<u32> = bytes
        .get(8..input_end)
        .ok_or(ExactError::LengthMismatch)?
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
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
