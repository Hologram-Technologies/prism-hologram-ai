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
//! - `MatMulNBits` at four bits ([`Q4Matrix`]): any block in [`Q4_BLOCKS`], the
//!   last block of a row padded, binary32 or half scales, and zero points
//!   absent (8) or packed four bits to a block.
//! - dense binary32, half and bfloat16 weights, as ONNX `MatMul` stores its
//!   second input ([`Dense`]).
//!
//! A format with no codec here is refused by name ([`ExactError::NoExactCodec`],
//! [`initializer_codec`], [`matmulnbits_codec`]), never run approximately.

#![forbid(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use uor_matmul::codec::Offset;
use uor_matmul::core_types::{Bnd, Complete};
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
    /// `FL-53`: a stored scale or weight is not a finite number.
    ScaleNotFinite,
    /// `FL-54`: the stored format has no exact codec here, so it is refused
    /// rather than run approximately.
    NoExactCodec,
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

const BFLOAT16: Format = Format {
    fraction_bits: 7,
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

    /// Reads a bfloat16 bit pattern.
    #[must_use]
    pub fn from_bfloat16(bits: u16) -> Option<Self> {
        BFLOAT16.read(u32::from(bits))
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
///
/// When every term, shifted to the row's smallest scale exponent, fits so that
/// their sum stays inside `i128`, the terms are added there and the total is
/// placed once; otherwise each term is placed on its own. The value is the
/// same either way.
#[must_use]
pub fn exact_row(sums: &[i64], scales: &[Dyadic], input_exponent: i32) -> Row {
    let terms: Vec<(i128, i32)> = sums
        .iter()
        .zip(scales)
        // A group sum is below 2^20 and a mantissa below 2^24: inside i128.
        .map(|(&sum, scale)| (i128::from(sum) * scale.mantissa, scale.exponent))
        .filter(|&(product, _)| product != 0)
        .collect();
    let mut row = Row::ZERO;
    let Some(lowest) = terms.iter().map(|&(_, e)| e).min() else {
        return row;
    };
    let carry = i64::from(bits_of(terms.len() as u128));
    let fits = terms.iter().all(|&(product, e)| {
        bits_of(product.unsigned_abs()) as i64 + i64::from(e - lowest) + carry <= 126
    });
    if fits {
        let total: i128 = terms
            .iter()
            .map(|&(product, e)| product << (e - lowest) as u32)
            .sum();
        row.add_scaled(total.unsigned_abs(), lowest - input_exponent, total < 0);
    } else {
        for &(product, e) in &terms {
            row.add_scaled(product.unsigned_abs(), e - input_exponent, product < 0);
        }
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

/// A four-bit code's unsigned value.
type Nibble = Alphabet<i8, Bnd<15>>;

/// The smallest and largest q4 block sizes with an exact codec here: the
/// powers of two `MatMulNBits` admits, up to the widest block whose sum of
/// 127 times 15 per weight stays far inside `i64`.
pub const Q4_BLOCKS: [usize; 5] = [16, 32, 64, 128, 256];

/// How a q4 matrix's zero points are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q4Zeros<'a> {
    /// None stored: every block's zero point is 8, as `MatMulNBits` defines.
    Eight,
    /// `MatMulNBits`' packed `uint8` zero points: per row, one four-bit zero
    /// point per block, two to a byte, low half first, a row padded to a whole
    /// byte.
    Packed(&'a [u8]),
}

/// A q4 matrix as `MatMulNBits` stores it at four bits: `rows` (its `N`) by
/// `columns` (its `K`), each row cut into blocks of `block` weights, the last
/// padded; the codes, two to a byte, low half first; one scale per block; and
/// the zero points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q4Matrix<'a> {
    /// The packed codes, `rows * blocks * block / 2` bytes.
    pub codes: &'a [u8],
    /// One scale per block, row after row.
    pub scales: Q4Scales<'a>,
    /// The zero points.
    pub zeros: Q4Zeros<'a>,
    /// Output rows (`N`).
    pub rows: usize,
    /// Inputs (`K`).
    pub columns: usize,
    /// Weights per block.
    pub block: usize,
}

impl Q4Matrix<'_> {
    /// Blocks in one row, the last padded.
    fn blocks(&self) -> usize {
        self.columns.div_ceil(self.block)
    }

    /// Every buffer the length the shape implies, and a block with a codec.
    fn check(&self, input: usize) -> Result<(), ExactError> {
        if !Q4_BLOCKS.contains(&self.block) {
            return Err(ExactError::NoExactCodec);
        }
        let blocks = self.blocks();
        let groups = self
            .rows
            .checked_mul(blocks)
            .ok_or(ExactError::LengthMismatch)?;
        let codes = groups
            .checked_mul(self.block / 2)
            .ok_or(ExactError::LengthMismatch)?;
        let zeros_fit = match self.zeros {
            Q4Zeros::Eight => true,
            Q4Zeros::Packed(bytes) => {
                Some(bytes.len()) == self.rows.checked_mul(blocks.div_ceil(2))
            }
        };
        if input != self.columns
            || self.codes.len() != codes
            || Some(self.scales.len()) != groups.checked_mul(self.scales.bytes_per_scale())
            || !zeros_fit
        {
            return Err(ExactError::LengthMismatch);
        }
        Ok(())
    }

    /// The zero point of block `group` of row `row`.
    fn zero(&self, row: usize, group: usize) -> i8 {
        match self.zeros {
            Q4Zeros::Eight => 8,
            Q4Zeros::Packed(bytes) => {
                let byte = bytes[row * self.blocks().div_ceil(2) + group / 2];
                ((byte >> (4 * (group % 2))) & 0x0f) as i8
            }
        }
    }
}

/// The exact sum of products of each block of a q4 matrix with the input
/// codes, row after row, block after block. Each weight is its code less its
/// block's zero point, decoded by `uor-matmul`'s `Offset` over a table of the
/// sixteen codes.
pub fn q4_block_sums(input: &Quantised, matrix: &Q4Matrix<'_>) -> Result<Vec<i64>, ExactError> {
    matrix.check(input.codes.len())?;
    // A q4 code as its unsigned value, 0 to 15, before the zero point is
    // taken. Each entry is within its bound, so no entry can be refused.
    let mut table = [Nibble::new(0).map_err(|_| ExactError::NoExactCodec)?; 16];
    for (code, entry) in (0i8..).zip(table.iter_mut()) {
        *entry = Nibble::new(code).map_err(|_| ExactError::NoExactCodec)?;
    }
    let grid = Grid::<'_, i8, Bnd<15>, 16>::new(&table);
    let activations = uor_matmul::as_alphabet_full(&input.codes);
    let blocks = matrix.blocks();
    let mut weights = vec![Value::of(0); matrix.block];
    let mut sums = Vec::with_capacity(matrix.rows * blocks);
    for row in 0..matrix.rows {
        for group in 0..blocks {
            let codec = Offset::<i8, Bnd<15>, Bnd<30>, _>::new(grid, matrix.zero(row, group))
                .ok_or(ExactError::NoExactCodec)?;
            let start = group * matrix.block;
            let used = matrix.block.min(matrix.columns - start);
            let first = (row * blocks + group) * matrix.block;
            for (i, weight) in weights.iter_mut().take(used).enumerate() {
                let g = first + i;
                let nibble = (matrix.codes[g / 2] >> (4 * (g % 2))) & 0x0f;
                *weight = Value::of(codec.decode_element(u16::from(nibble), 0).get());
            }
            let sum = dot_ref(&activations[start..start + used], &weights[..used]);
            sums.push(sum as i64);
        }
    }
    Ok(sums)
}

/// A q4 matrix times `input`, both as binary32 bit patterns: one output bit
/// pattern per row.
pub fn q4_product(input: &[u32], matrix: &Q4Matrix<'_>) -> Result<Vec<u32>, ExactError> {
    let quantised = quantise(input)?;
    let sums = q4_block_sums(&quantised, matrix)?;
    rows_from(
        &sums,
        |g| matrix.scales.at(g),
        matrix.blocks(),
        quantised.exponent,
    )
}

/// A dense weight matrix as ONNX `MatMul`'s second input stores it: `columns`
/// (its `K`) rows of `rows` (its `N`) values, row after row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dense<'a> {
    /// Little-endian binary32.
    F32(&'a [u8]),
    /// Little-endian IEEE half.
    F16(&'a [u8]),
    /// Little-endian bfloat16.
    Bf16(&'a [u8]),
}

impl Dense<'_> {
    fn bytes(self) -> (usize, usize) {
        match self {
            Self::F32(bytes) => (bytes.len(), 4),
            Self::F16(bytes) | Self::Bf16(bytes) => (bytes.len(), 2),
        }
    }

    /// Weight `at`, exactly.
    fn at(self, at: usize) -> Result<Dyadic, ExactError> {
        let read = match self {
            Self::F32(bytes) => {
                let b = at * 4;
                Dyadic::from_binary32(u32::from_le_bytes([
                    bytes[b],
                    bytes[b + 1],
                    bytes[b + 2],
                    bytes[b + 3],
                ]))
            }
            Self::F16(bytes) => {
                Dyadic::from_binary16(u16::from_le_bytes([bytes[2 * at], bytes[2 * at + 1]]))
            }
            Self::Bf16(bytes) => {
                Dyadic::from_bfloat16(u16::from_le_bytes([bytes[2 * at], bytes[2 * at + 1]]))
            }
        };
        read.ok_or(ExactError::ScaleNotFinite)
    }
}

/// How a dense matrix's values lie in its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseLayout {
    /// As ONNX `MatMul` stores its second input: `K` rows of `N`, so weight
    /// `(k, n)` is value `k * N + n`.
    InputMajor,
    /// `N` rows of `K`, as an embedding table is stored and as `MatMul` reads
    /// it through a `Transpose`: weight `(k, n)` is value `n * K + k`. Read in
    /// place, with no transposed copy.
    OutputMajor,
}

/// `input` times a dense weight matrix of `columns` (`K`) inputs and `rows`
/// (`N`) outputs: output `n` is the sum over `k` of input `k` times weight
/// `(k, n)`, each product an input code times the weight's exact value, held
/// in the complete register and rounded once.
pub fn dense_product(
    input: &[u32],
    weights: Dense<'_>,
    layout: DenseLayout,
    columns: usize,
    rows: usize,
) -> Result<Vec<u32>, ExactError> {
    let quantised = quantise(input)?;
    let (length, width) = weights.bytes();
    if input.len() != columns
        || Some(length) != columns.checked_mul(rows).and_then(|n| n.checked_mul(width))
    {
        return Err(ExactError::LengthMismatch);
    }
    let place = |n: usize, k: usize| match layout {
        DenseLayout::InputMajor => k * rows + n,
        DenseLayout::OutputMajor => n * columns + k,
    };
    (0..rows)
        .map(|n| {
            let row = match weights {
                Dense::F16(bytes) => half_row(
                    &quantised.codes,
                    bytes,
                    |k| place(n, k),
                    -quantised.exponent,
                )?,
                _ => {
                    let values = (0..columns)
                        .map(|k| weights.at(place(n, k)))
                        .collect::<Result<Vec<Dyadic>, ExactError>>()?;
                    wide_row(&quantised.codes, &values, -quantised.exponent)
                }
            };
            Ok(round_to_binary32(&row))
        })
        .collect()
}

/// The power of two a half's integer form counts: every finite half is an
/// integer times `2^-24`, of magnitude below `2^40`.
const HALF_UNIT: i32 = -24;

/// Products summed in `i64` before they join the `i128` total: each is below
/// `127 * 2^40 = 2^47`, so 4096 of them stay below `2^59`.
const HALF_CHUNK: usize = 4096;

/// A half as the integer it is in units of `2^-24`, or `None` for an infinity
/// or NaN. Exact for every finite half.
fn half_fixed(bits: u16) -> Option<i64> {
    let fraction = i64::from(bits & 0x3ff);
    let biased = (bits >> 10) & 0x1f;
    let magnitude = match biased {
        31 => return None,
        0 => fraction,
        _ => (fraction | 0x400) << (biased - 1),
    };
    Some(if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    })
}

/// A row of half weights times the input codes, exactly: each product summed
/// in integers in units of `2^-24`, then placed once in the register at that
/// unit times `2^shift`. The
/// value is the one each product placed on its own would give.
fn half_row(
    codes: &[i8],
    bytes: &[u8],
    at: impl Fn(usize) -> usize,
    shift: i32,
) -> Result<Row, ExactError> {
    let mut total: i128 = 0;
    for (chunk, start) in codes.chunks(HALF_CHUNK).zip((0..).step_by(HALF_CHUNK)) {
        let mut part: i64 = 0;
        for (offset, &code) in chunk.iter().enumerate() {
            let b = 2 * at(start + offset);
            let weight = half_fixed(u16::from_le_bytes([bytes[b], bytes[b + 1]]))
                .ok_or(ExactError::ScaleNotFinite)?;
            part += i64::from(code) * weight;
        }
        total += i128::from(part);
    }
    let mut row = Row::ZERO;
    row.add_scaled(total.unsigned_abs(), HALF_UNIT + shift, total < 0);
    Ok(row)
}

/// A row of weights read exactly times the input codes. When every weight,
/// shifted to the row's smallest exponent, fits in 62 bits, the products are
/// summed in `i128` and placed once; otherwise each product is placed in the
/// register on its own. Either way the value is exact and the same.
fn wide_row(codes: &[i8], values: &[Dyadic], shift: i32) -> Row {
    let lowest = values
        .iter()
        .filter(|v| v.mantissa != 0)
        .map(|v| v.exponent)
        .min()
        .unwrap_or(0);
    let fits = values.iter().all(|v| {
        v.mantissa == 0
            || bits_of(v.mantissa.unsigned_abs()) as i64 + i64::from(v.exponent - lowest) <= 62
    }) && codes.len() < 1 << 24;
    let mut row = Row::ZERO;
    if fits {
        // Each product below 2^69, fewer than 2^24 of them: below 2^93.
        let total: i128 = codes
            .iter()
            .zip(values)
            .map(|(&c, v)| i128::from(c) * (v.mantissa << (v.exponent - lowest) as u32))
            .sum();
        row.add_scaled(total.unsigned_abs(), lowest + shift, total < 0);
    } else {
        for (&code, v) in codes.iter().zip(values) {
            let product = i128::from(code) * v.mantissa;
            row.add_scaled(product.unsigned_abs(), v.exponent + shift, product < 0);
        }
    }
    row
}

/// How many bits an input code has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Width {
    /// -127 to 127.
    Eight,
    /// -32,767 to 32,767.
    Sixteen,
}

impl Width {
    fn limit(self) -> u128 {
        match self {
            Self::Eight => 127,
            Self::Sixteen => 32_767,
        }
    }
}

/// Which inputs share one power of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grouping {
    /// The whole vector, as [`quantise`] does.
    Vector,
    /// Each run of [`ACTIVATION_GROUP`] inputs, the last shorter if the vector
    /// is: one outlier then sets the step only for its own group.
    Groups,
}

/// Inputs in one activation group under [`Grouping::Groups`].
pub const ACTIVATION_GROUP: usize = 32;

/// How an input vector is put on integers: the one approximation in a
/// product, measured per model (brief step 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scheme {
    /// Bits per code.
    pub width: Width,
    /// Which inputs share a power of two.
    pub grouping: Grouping,
}

/// An input vector on integers under a [`Scheme`]: input `i` stands for
/// `codes[i] * 2^-exponents[i / group]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Activations {
    codes: Vec<i16>,
    exponents: Vec<i32>,
    group: usize,
}

impl Activations {
    /// The integer codes.
    #[must_use]
    pub fn codes(&self) -> &[i16] {
        &self.codes
    }

    /// One power of two per group.
    #[must_use]
    pub fn exponents(&self) -> &[i32] {
        &self.exponents
    }

    /// Inputs per group: the vector's length under [`Grouping::Vector`].
    #[must_use]
    pub fn group(&self) -> usize {
        self.group
    }

    /// The power of two input `i` is scaled by.
    fn exponent_at(&self, i: usize) -> i32 {
        self.exponents[i / self.group.max(1)]
    }
}

/// Scales `read` by the largest power of two that keeps every value within
/// `limit`, each rounded to the nearest, ties to even: the codes and the
/// power. All zeros are zero codes at power 0.
fn scale_onto(read: &[Dyadic], limit: u128) -> (Vec<i16>, i32) {
    let zero = Dyadic {
        mantissa: 0,
        exponent: 0,
    };
    let largest = read
        .iter()
        .fold(zero, |m, &x| if larger(x, m) { x } else { m });
    let top = largest.mantissa.unsigned_abs();
    if top == 0 {
        return (vec![0; read.len()], 0);
    }
    // The largest `k` with `top * 2^k <= limit`: one step down from the `k`
    // that gives `top * 2^k` the bits of `limit`, if that is past it.
    let mut k = bits_of(limit) as i32 - bits_of(top) as i32;
    let past = if k >= 0 {
        top << k as u32 > limit
    } else {
        top > limit << (-k) as u32
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
            // Within the limit by the choice of exponent; restated.
            let code = code.min(limit) as i16;
            if x.mantissa < 0 {
                -code
            } else {
                code
            }
        })
        .collect();
    (codes, exponent)
}

/// Puts `input`, binary32 bit patterns, on integers under `scheme`: each group
/// scaled by the largest power of two that keeps it within the width's limit,
/// each value rounded to the nearest, ties to even. Each input is then off by
/// at most half its group's step, which is at most the group's largest input
/// divided by the limit.
pub fn quantise_with(input: &[u32], scheme: Scheme) -> Result<Activations, ExactError> {
    let read: Vec<Dyadic> = input
        .iter()
        .map(|&bits| Dyadic::from_binary32(bits).ok_or(ExactError::InputNotFinite))
        .collect::<Result<Vec<Dyadic>, ExactError>>()?;
    let group = match scheme.grouping {
        Grouping::Vector => read.len().max(1),
        Grouping::Groups => ACTIVATION_GROUP,
    };
    let mut codes = Vec::with_capacity(read.len());
    let mut exponents = Vec::with_capacity(read.len().div_ceil(group));
    for part in read.chunks(group) {
        let (part_codes, exponent) = scale_onto(part, scheme.width.limit());
        codes.extend(part_codes);
        exponents.push(exponent);
    }
    Ok(Activations {
        codes,
        exponents,
        group,
    })
}

/// A row's exact value from terms `(product, exponent)`, each `product *
/// 2^exponent`: summed in `i128` at the smallest exponent when every shifted
/// term fits with room for the carries, and placed once; placed term by term
/// otherwise. The value is the same either way.
fn place_terms(terms: &[(i128, i32)]) -> Row {
    let mut row = Row::ZERO;
    let Some(lowest) = terms
        .iter()
        .filter(|&&(p, _)| p != 0)
        .map(|&(_, e)| e)
        .min()
    else {
        return row;
    };
    let carry = i64::from(bits_of(terms.len() as u128));
    let fits = terms.iter().all(|&(p, e)| {
        p == 0 || bits_of(p.unsigned_abs()) as i64 + i64::from(e - lowest) + carry <= 126
    });
    if fits {
        let total: i128 = terms
            .iter()
            .filter(|&&(p, _)| p != 0)
            .map(|&(p, e)| p << (e - lowest) as u32)
            .sum();
        row.add_scaled(total.unsigned_abs(), lowest, total < 0);
    } else {
        for &(p, e) in terms {
            row.add_scaled(p.unsigned_abs(), e, p < 0);
        }
    }
    row
}

/// One wide input or weight value, admitted whole.
type Wide = Alphabet<i16, Full<i16>>;

/// The exact sums of a q4 matrix's rows with `input`, one per unit: a unit is
/// the smaller of a weight block and an input group, so each unit has one
/// weight scale and one input power. Row after row, unit after unit.
pub fn q4_unit_sums(input: &Activations, matrix: &Q4Matrix<'_>) -> Result<Vec<i64>, ExactError> {
    matrix.check(input.codes.len())?;
    let mut table = [Nibble::new(0).map_err(|_| ExactError::NoExactCodec)?; 16];
    for (code, entry) in (0i8..).zip(table.iter_mut()) {
        *entry = Nibble::new(code).map_err(|_| ExactError::NoExactCodec)?;
    }
    let grid = Grid::<'_, i8, Bnd<15>, 16>::new(&table);
    let activations = uor_matmul::as_alphabet_full(&input.codes);
    let unit = matrix.block.min(input.group);
    let blocks = matrix.blocks();
    let padded = blocks * matrix.block;
    let mut weights = vec![Wide::of(0); unit];
    let mut sums = Vec::with_capacity(matrix.rows * padded.div_ceil(unit));
    for row in 0..matrix.rows {
        for start in (0..matrix.columns).step_by(unit) {
            let group = start / matrix.block;
            let codec = Offset::<i8, Bnd<15>, Bnd<30>, _>::new(grid, matrix.zero(row, group))
                .ok_or(ExactError::NoExactCodec)?;
            let used = unit.min(matrix.columns - start);
            let first = row * padded + start;
            for (i, weight) in weights.iter_mut().take(used).enumerate() {
                let g = first + i;
                let nibble = (matrix.codes[g / 2] >> (4 * (g % 2))) & 0x0f;
                *weight = Wide::of(i16::from(codec.decode_element(u16::from(nibble), 0).get()));
            }
            let sum = dot_ref(&activations[start..start + used], &weights[..used]);
            sums.push(sum as i64);
        }
    }
    Ok(sums)
}

/// A q4 matrix times `input` under `scheme`, both as binary32 bit patterns:
/// one output bit pattern per row.
pub fn q4_product_with(
    input: &[u32],
    matrix: &Q4Matrix<'_>,
    scheme: Scheme,
) -> Result<Vec<u32>, ExactError> {
    let activations = quantise_with(input, scheme)?;
    let sums = q4_unit_sums(&activations, matrix)?;
    let unit = matrix.block.min(activations.group);
    let units = matrix.columns.div_ceil(unit);
    let blocks = matrix.blocks();
    (0..matrix.rows)
        .map(|row| {
            let terms = (0..units)
                .map(|u| {
                    let scale = matrix.scales.at(row * blocks + u * unit / matrix.block)?;
                    let sum = i128::from(sums[row * units + u]);
                    Ok((
                        sum * scale.mantissa,
                        scale.exponent - activations.exponent_at(u * unit),
                    ))
                })
                .collect::<Result<Vec<(i128, i32)>, ExactError>>()?;
            Ok(round_to_binary32(&place_terms(&terms)))
        })
        .collect()
}

/// `input` times a dense weight matrix under `scheme`, as [`dense_product`]
/// with the input on the scheme's integers: each input group's products are
/// summed exactly and the groups placed at their own powers.
pub fn dense_product_with(
    input: &[u32],
    weights: Dense<'_>,
    layout: DenseLayout,
    columns: usize,
    rows: usize,
    scheme: Scheme,
) -> Result<Vec<u32>, ExactError> {
    let activations = quantise_with(input, scheme)?;
    let (length, width) = weights.bytes();
    if input.len() != columns
        || Some(length) != columns.checked_mul(rows).and_then(|n| n.checked_mul(width))
    {
        return Err(ExactError::LengthMismatch);
    }
    let place = |n: usize, k: usize| match layout {
        DenseLayout::InputMajor => k * rows + n,
        DenseLayout::OutputMajor => n * columns + k,
    };
    // Sixteen-bit codes times a half below 2^40 are below 2^55; 32 of them
    // stay below 2^60, inside i64.
    let chunk = ACTIVATION_GROUP.min(activations.group);
    (0..rows)
        .map(|n| {
            let mut terms = Vec::with_capacity(columns.div_ceil(chunk));
            for start in (0..columns).step_by(chunk) {
                let shift = -activations.exponent_at(start);
                let end = (start + chunk).min(columns);
                match weights {
                    Dense::F16(bytes) => {
                        let mut part: i64 = 0;
                        for k in start..end {
                            let b = 2 * place(n, k);
                            let w = half_fixed(u16::from_le_bytes([bytes[b], bytes[b + 1]]))
                                .ok_or(ExactError::ScaleNotFinite)?;
                            part += i64::from(activations.codes[k]) * w;
                        }
                        terms.push((i128::from(part), HALF_UNIT + shift));
                    }
                    _ => {
                        for k in start..end {
                            let w = weights.at(place(n, k))?;
                            terms.push((
                                i128::from(activations.codes[k]) * w.mantissa,
                                w.exponent + shift,
                            ));
                        }
                    }
                }
            }
            Ok(round_to_binary32(&place_terms(&terms)))
        })
        .collect()
}

/// A ternary matrix of `rows` by `columns`, in its three planes, times
/// `input` under `scheme`: one output bit pattern per row.
pub fn ternary_product_with(
    input: &[u32],
    signs: &[u8],
    nonzeros: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
    scheme: Scheme,
) -> Result<Vec<u32>, ExactError> {
    let activations = quantise_with(input, scheme)?;
    if !columns.is_multiple_of(TERNARY_GROUP) {
        return Err(ExactError::GroupMisaligned);
    }
    let plane = rows * columns / 8;
    if input.len() != columns
        || signs.len() != plane
        || nonzeros.len() != plane
        || scales.len() != rows * (columns / TERNARY_GROUP) * 4
    {
        return Err(ExactError::LengthMismatch);
    }
    let grid = Grid::<'_, i8, Full<i8>, 4>::new(&TERNARY_TABLE);
    let codes = uor_matmul::as_alphabet_full(&activations.codes);
    let unit = TERNARY_GROUP.min(activations.group);
    let mut weights = vec![Wide::of(0); unit];
    (0..rows)
        .map(|row| {
            let mut terms = Vec::with_capacity(columns / unit);
            for start in (0..columns).step_by(unit) {
                let used = unit.min(columns - start);
                for (i, weight) in weights.iter_mut().take(used).enumerate() {
                    let g = row * columns + start + i;
                    let bit = |plane: &[u8]| u16::from((plane[g / 8] >> (g % 8)) & 1);
                    *weight = Wide::of(i16::from(
                        grid.decode_element(bit(nonzeros) + 2 * bit(signs), 0).get(),
                    ));
                }
                let sum = dot_ref(&codes[start..start + used], &weights[..used]) as i64;
                let scale = Q4Scales::F32(scales)
                    .at(row * (columns / TERNARY_GROUP) + start / TERNARY_GROUP)?;
                terms.push((
                    i128::from(sum) * scale.mantissa,
                    scale.exponent - activations.exponent_at(start),
                ));
            }
            Ok(round_to_binary32(&place_terms(&terms)))
        })
        .collect()
}

/// One term of a row's exact recombination: a unit sum times a scale's
/// mantissa, at the scale's power less the input group's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowTerm {
    /// The unit's exact sum of products.
    pub sum: i64,
    /// The scale's mantissa, odd or zero.
    pub mantissa: i128,
    /// The power of two the product stands at.
    pub exponent: i32,
}

/// Terms per link of decision 40.
pub const LINK_TERMS: usize = 6;

/// Bytes of one decision-40 request.
pub const LINK_BYTES: usize = 79;

/// Each row's terms for a q4 matrix times `input` under `scheme`: the terms
/// [`q4_product_with`] places, in the order it places them.
pub fn q4_row_terms(
    input: &[u32],
    matrix: &Q4Matrix<'_>,
    scheme: Scheme,
) -> Result<Vec<Vec<RowTerm>>, ExactError> {
    let activations = quantise_with(input, scheme)?;
    let sums = q4_unit_sums(&activations, matrix)?;
    let unit = matrix.block.min(activations.group);
    let units = matrix.columns.div_ceil(unit);
    let blocks = matrix.blocks();
    (0..matrix.rows)
        .map(|row| {
            (0..units)
                .map(|u| {
                    let scale = matrix.scales.at(row * blocks + u * unit / matrix.block)?;
                    Ok(RowTerm {
                        sum: sums[row * units + u],
                        mantissa: scale.mantissa,
                        exponent: scale.exponent - activations.exponent_at(u * unit),
                    })
                })
                .collect::<Result<Vec<RowTerm>, ExactError>>()
        })
        .collect()
}

/// One signed number as decision 40 reads it: a sign byte, then the
/// magnitude in `width` bytes, most significant first.
fn put_signed(out: &mut Vec<u8>, negative: bool, magnitude: u128, width: usize) {
    out.push(u8::from(negative));
    out.extend_from_slice(&magnitude.to_be_bytes()[16 - width..]);
}

/// A row's terms as decision 40's chain: the requests, each the partial so
/// far, up to [`LINK_TERMS`] terms and the partial after them, and the exact
/// row it reaches, as an integer and the power it stands at. `None` when a
/// term or a partial is too wide for the guest's 64-bit numbers: a partial at
/// or past `2^62`, or a term whose sum or mantissa reaches `2^24` or whose
/// value reaches `2^58` at the chain's lowest power.
#[must_use]
pub fn row_links(terms: &[RowTerm]) -> Option<(Vec<[u8; LINK_BYTES]>, i128, i32)> {
    let base = terms
        .iter()
        .filter(|t| t.sum != 0 && t.mantissa != 0)
        .map(|t| t.exponent)
        .min()
        .unwrap_or(0);
    let mut links = Vec::new();
    let mut partial: i128 = 0;
    for chunk in terms.chunks(LINK_TERMS) {
        let mut request = Vec::with_capacity(LINK_BYTES);
        request.push(40);
        let mut next = partial;
        let mut fields = Vec::with_capacity(10 * LINK_TERMS);
        for t in chunk {
            let (s, m) = (t.sum.unsigned_abs(), t.mantissa.unsigned_abs());
            let zero = s == 0 || m == 0;
            let shift = if zero {
                0
            } else {
                u32::try_from(t.exponent - base).ok()?
            };
            if s >= 1 << 24
                || m >= 1 << 24
                || shift > 58
                || u128::from(s) * m >= 1u128 << (58 - shift)
            {
                return None;
            }
            let value = (u128::from(s) * m) << shift;
            let negative = !zero && ((t.sum < 0) != (t.mantissa < 0));
            next += if negative {
                -(value as i128)
            } else {
                value as i128
            };
            fields.push(u8::from(negative));
            fields.extend_from_slice(&(s as u32).to_be_bytes());
            fields.extend_from_slice(&(m as u32).to_be_bytes());
            fields.push(shift as u8);
        }
        fields.resize(10 * LINK_TERMS, 0);
        if partial.unsigned_abs() >= 1 << 62 || next.unsigned_abs() >= 1 << 62 {
            return None;
        }
        put_signed(&mut request, partial < 0, partial.unsigned_abs(), 8);
        put_signed(&mut request, next < 0, next.unsigned_abs(), 8);
        request.extend_from_slice(&fields);
        links.push(request.try_into().ok()?);
        partial = next;
    }
    Some((links, partial, base))
}

/// The binary32 bit pattern nearest `value * 2^power`, ties to even.
#[must_use]
pub fn round_exact(value: i128, power: i32) -> u32 {
    let mut row = Row::ZERO;
    row.add_scaled(value.unsigned_abs(), power, value < 0);
    round_to_binary32(&row)
}

/// Bytes of one decision-41 request.
pub const REQUANT_BYTES: usize = 62;

/// Inputs per decision-41 request.
pub const REQUANT_INPUTS: usize = 5;

/// One input's magnitude as decision 41 reads it: its mantissa, whether its
/// group's power multiplies it up (1) or divides it down (0), and by how many
/// powers of two.
fn requant_place(value: Dyadic, exponent: i32) -> (u32, u8, u32) {
    let mantissa = value.mantissa.unsigned_abs() as u32;
    let shift = value.exponent + exponent;
    if shift >= 0 {
        (mantissa, 1, shift as u32)
    } else {
        (mantissa, 0, shift.unsigned_abs())
    }
}

/// Decision 41's requests for `input` put on sixteen-bit codes per group of
/// [`ACTIVATION_GROUP`]: each request names its group's largest input, then up
/// to [`REQUANT_INPUTS`] of the group's inputs, each with its quotient witness
/// (`2 * half + odd`, the input divided down and rounded toward zero) and its
/// code's magnitude. The codes are [`quantise_with`]'s.
pub fn requant_requests(input: &[u32]) -> Result<Vec<[u8; REQUANT_BYTES]>, ExactError> {
    let scheme = Scheme {
        width: Width::Sixteen,
        grouping: Grouping::Groups,
    };
    let activations = quantise_with(input, scheme)?;
    let read: Vec<Dyadic> = input
        .iter()
        .map(|&bits| Dyadic::from_binary32(bits).ok_or(ExactError::InputNotFinite))
        .collect::<Result<Vec<Dyadic>, ExactError>>()?;
    let zero = Dyadic {
        mantissa: 0,
        exponent: 0,
    };
    let mut requests = Vec::new();
    for (g, group) in read.chunks(ACTIVATION_GROUP).enumerate() {
        let exponent = activations.exponents[g];
        let largest = group
            .iter()
            .fold(zero, |m, &x| if larger(x, m) { x } else { m });
        let (lm, ld, ls) = requant_place(largest, exponent);
        let start = g * ACTIVATION_GROUP;
        for (c, chunk) in group.chunks(REQUANT_INPUTS).enumerate() {
            let mut request = Vec::with_capacity(REQUANT_BYTES);
            request.push(41);
            request.extend_from_slice(&lm.to_be_bytes());
            request.push(ld);
            request.push(u8::try_from(ls).unwrap_or(u8::MAX));
            for (i, &x) in chunk.iter().enumerate() {
                let (m, d, s) = requant_place(x, exponent);
                let code = activations.codes[start + c * REQUANT_INPUTS + i].unsigned_abs();
                let quotient = if d == 1 || s > 39 {
                    0
                } else {
                    u64::from(m) >> s
                };
                request.extend_from_slice(&m.to_be_bytes());
                request.push(d);
                request.push(u8::try_from(s).unwrap_or(u8::MAX));
                request.extend_from_slice(
                    &u16::try_from(quotient / 2)
                        .unwrap_or(u16::MAX)
                        .to_be_bytes(),
                );
                request.push((quotient % 2) as u8);
                request.extend_from_slice(&code.to_be_bytes());
            }
            // A short last request repeats nothing: its unused inputs are
            // zero, which decision 41 reads as code 0 for value 0.
            request.resize(REQUANT_BYTES, 0);
            requests.push(request.try_into().map_err(|_| ExactError::LengthMismatch)?);
        }
    }
    Ok(requests)
}

/// An ONNX initializer of `data_type` as dense weights, or
/// [`ExactError::NoExactCodec`]: binary32 (1), half (10) and bfloat16 (16) have
/// exact codecs; no other type does yet.
pub fn initializer_codec(data_type: i32, bytes: &[u8]) -> Result<Dense<'_>, ExactError> {
    match data_type {
        1 => Ok(Dense::F32(bytes)),
        10 => Ok(Dense::F16(bytes)),
        16 => Ok(Dense::Bf16(bytes)),
        _ => Err(ExactError::NoExactCodec),
    }
}

/// Whether a `MatMulNBits` node has an exact codec here: four bits, a block
/// in [`Q4_BLOCKS`], binary32 (1) or half (10) scales, and zero points absent
/// or packed `uint8` (2). Anything else is [`ExactError::NoExactCodec`].
pub fn matmulnbits_codec(
    bits: i64,
    block: i64,
    scales_type: i32,
    zeros_type: Option<i32>,
) -> Result<(), ExactError> {
    let block_ok = usize::try_from(block).is_ok_and(|b| Q4_BLOCKS.contains(&b));
    let zeros_ok = matches!(zeros_type, None | Some(2));
    if bits == 4 && block_ok && matches!(scales_type, 1 | 10) && zeros_ok {
        Ok(())
    } else {
        Err(ExactError::NoExactCodec)
    }
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

/// The fractional part of softmax's base-two exponent: entry `j` is the
/// integer nearest `2^(30 - j/256)`, so `2^30` at `j = 0`. Generated with
/// integers of any width; EN-43 certifies each entry exactly, as
/// `(2t - 1)^256 < 2^(7936 - j) < (2t + 1)^256`.
pub const EXP2_FRACTION: [u32; 256] = [
    1073741824, 1070838486, 1067942999, 1065055341, 1062175491, 1059303428, 1056439131, 1053582579,
    1050733751, 1047892626, 1045059183, 1042233401, 1039415261, 1036604740, 1033801819, 1031006477,
    1028218693, 1025438448, 1022665720, 1019900489, 1017142735, 1014392438, 1011649578, 1008914134,
    1006186087, 1003465416, 1000752102, 998046124, 995347464, 992656100, 989972014, 987295185,
    984625594, 981963222, 979308048, 976660054, 974019220, 971385527, 968758955, 966139485,
    963527098, 960921775, 958323496, 955732243, 953147997, 950570738, 948000448, 945437108,
    942880699, 940331203, 937788600, 935252872, 932724001, 930201967, 927686753, 925178340,
    922676710, 920181844, 917693724, 915212331, 912737649, 910269657, 907808339, 905353676,
    902905651, 900464244, 898029440, 895601218, 893179563, 890764456, 888355878, 885953814,
    883558244, 881169153, 878786521, 876410331, 874040567, 871677210, 869320244, 866969651,
    864625413, 862287515, 859955938, 857630665, 855311680, 852998965, 850692504, 848392279,
    846098274, 843810471, 841528855, 839253408, 836984114, 834720956, 832463917, 830212982,
    827968132, 825729353, 823496627, 821269938, 819049271, 816834607, 814625932, 812423229,
    810226483, 808035676, 805850792, 803671817, 801498734, 799331526, 797170178, 795014675,
    792865000, 790721137, 788583072, 786450787, 784324269, 782203500, 780088465, 777979150,
    775875538, 773777614, 771685363, 769598769, 767517817, 765442492, 763372778, 761308661,
    759250125, 757197155, 755149737, 753107854, 751071493, 749040637, 747015274, 744995386,
    742980960, 740971982, 738968435, 736970306, 734977579, 732990241, 731008277, 729031671,
    727060411, 725094480, 723133865, 721178552, 719228525, 717283772, 715344277, 713410026,
    711481005, 709557200, 707638598, 705725183, 703816941, 701913860, 700015924, 698123120,
    696235434, 694352853, 692475362, 690602947, 688735596, 686873293, 685016026, 683163781,
    681316545, 679474303, 677637043, 675804750, 673977412, 672155015, 670337545, 668524990,
    666717336, 664914570, 663116678, 661323648, 659535466, 657752119, 655973594, 654199878,
    652430958, 650666822, 648907455, 647152846, 645402981, 643657847, 641917433, 640181724,
    638450708, 636724373, 635002706, 633285695, 631573326, 629865587, 628162466, 626463950,
    624770026, 623080683, 621395908, 619715688, 618040012, 616368866, 614702239, 613040119,
    611382493, 609729349, 608080675, 606436459, 604796689, 603161352, 601530438, 599903933,
    598281827, 596664106, 595050760, 593441776, 591837143, 590236848, 588640881, 587049229,
    585461881, 583878825, 582300049, 580725543, 579155293, 577589290, 576027521, 574469975,
    572916640, 571367506, 569822560, 568281792, 566745190, 565212742, 563684439, 562160268,
    560640218, 559124278, 557612438, 556104685, 554601009, 553101399, 551605844, 550114332,
    548626854, 547143398, 545663953, 544188508, 542717053, 541249576, 539786068, 538326517,
];

/// `2^(30 - d/256)` for `d` in units of `2^-8`: the fractional entry of `d`,
/// shifted down by `d`'s whole part and rounded half to even, the one rounding
/// rule of this crate. Within one unit of the exact value (EN-43); 0 from
/// `d = 7936`, where the exact value is at most half a unit.
pub fn exp2_neg(d: u32) -> u32 {
    let entry = u64::from(EXP2_FRACTION[(d & 255) as usize]);
    let shift = d >> 8;
    if shift == 0 {
        return entry as u32;
    }
    if shift > 32 {
        return 0;
    }
    let quotient = entry >> shift;
    let rest = entry & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    let up = rest > half || (rest == half && quotient & 1 == 1);
    (quotient + u64::from(up)) as u32
}

/// Softmax in integers. `exponents` are scores already in units of `2^-8` of
/// a base-two exponent (the model's `log2(e)` and scale folded in by the
/// caller). Each output is `exp2_neg(top - x)` times `2^30`, over the exact sum
/// of those weights, rounded half to even: units of `2^-30`. The top score's
/// weight is `2^30`, so the sum is at least `2^30` and no output passes it
/// (EN-44). An empty row gives an empty row.
pub fn softmax_exp2(exponents: &[i64]) -> Vec<u32> {
    let Some(&top) = exponents.iter().max() else {
        return Vec::new();
    };
    let weights: Vec<u128> = exponents
        .iter()
        .map(|&x| {
            let d = u32::try_from(i128::from(top) - i128::from(x)).unwrap_or(u32::MAX);
            u128::from(exp2_neg(d))
        })
        .collect();
    let sum: u128 = weights.iter().sum();
    weights
        .iter()
        .map(|&e| {
            let scaled = e << 30;
            let (quotient, rest) = (scaled / sum, scaled % sum);
            let up = 2 * rest > sum || (2 * rest == sum && quotient & 1 == 1);
            (quotient + u128::from(up)) as u32
        })
        .collect()
}

/// Steps in a quarter turn of the sine table: entry `r` is the sine at
/// `r / 65536` of a turn, in units of `2^-30`.
pub const SINE_QUARTER: usize = 16384;

/// `a * b / 2^120`, cut down: the fixed point the sine table is generated in.
/// Both factors stay below `2^123`, so the result fits.
fn fixed_mul(a: u128, b: u128) -> u128 {
    let low_mask = u128::from(u64::MAX);
    let (a1, a0) = (a >> 64, a & low_mask);
    let (b1, b0) = (b >> 64, b & low_mask);
    let (middle, middle_carry) = (a1 * b0).overflowing_add(a0 * b1);
    let (low, low_carry) = (a0 * b0).overflowing_add(middle << 64);
    let high = a1 * b1 + (middle >> 64) + (u128::from(middle_carry) << 64) + u128::from(low_carry);
    (high << 8) | (low >> 120)
}

/// `2^120 atan(1 / m)`, cut down, by its alternating series.
fn atan_inverse(m: u128) -> u128 {
    let mut power = (1u128 << 120) / m;
    let (mut sum, mut k) = (0u128, 0u128);
    while power > 0 {
        let term = power / (2 * k + 1);
        sum = if k.is_multiple_of(2) {
            sum + term
        } else {
            sum - term
        };
        power /= m * m;
        k += 1;
    }
    sum
}

/// `2^30 sin(pi r / 32768)` for `r <= 16384`, rounded half to even: `pi` by
/// Machin's formula and the sine by its Taylor series, in integers over
/// `2^120`, so every machine generates the same table.
fn sine_entry(pi: u128, r: u128) -> u32 {
    let x = (pi >> 15) * r + (((pi & 0x7fff) * r) >> 15);
    let square = fixed_mul(x, x);
    let (mut sum, mut term, mut k) = (x, x, 1u128);
    loop {
        term = fixed_mul(term, square) / ((2 * k) * (2 * k + 1));
        if term == 0 {
            break;
        }
        sum = if k % 2 == 1 { sum - term } else { sum + term };
        k += 1;
    }
    let (quotient, rest, half) = (sum >> 90, sum & ((1u128 << 90) - 1), 1u128 << 89);
    let up = rest > half || (rest == half && quotient & 1 == 1);
    (quotient + u128::from(up)) as u32
}

/// The quarter-wave sine table: 16,385 entries, entry `r` the integer nearest
/// `2^30 sin(pi r / 32768)`, generated once by exact integer arithmetic.
pub fn sine_quarter() -> &'static [u32] {
    static TABLE: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let pi = 16 * atan_inverse(5) - 4 * atan_inverse(239);
        (0..=SINE_QUARTER as u128)
            .map(|r| sine_entry(pi, r))
            .collect()
    })
}

/// The sine of `angle` units of `2^-32` of a turn, in units of `2^-30`: the
/// angle rounded half to even to a step of `2^-16` of a turn, read from the
/// quarter-wave table by quadrant.
pub fn sine_turn(angle: u32) -> i32 {
    let (whole, rest) = (angle >> 16, angle & 0xffff);
    let up = rest > 0x8000 || (rest == 0x8000 && whole & 1 == 1);
    let step = (whole + u32::from(up)) & 0xffff;
    let table = sine_quarter();
    let r = (step & 0x3fff) as usize;
    let quarter = SINE_QUARTER;
    match step >> 14 {
        0 => table[r] as i32,
        1 => table[quarter - r] as i32,
        2 => -(table[r] as i32),
        _ => -(table[quarter - r] as i32),
    }
}

/// The cosine of `angle` units of `2^-32` of a turn: the sine a quarter turn on.
pub fn cosine_turn(angle: u32) -> i32 {
    sine_turn(angle.wrapping_add(1 << 30))
}

/// The integer nearest `2^32 log2(e)`: SiLU's exponent `e^-x` is taken in base
/// two as `2^(-x log2(e))`, on EN-43's table.
pub const LOG2_E: u64 = 6_196_328_019;

/// SiLU's input and output unit: `2^-16`.
pub const SILU_FRACTION: u32 = 16;

/// `x / (1 + e^-x)` for `x` in units of `2^-16` below `2^47` in size, in the
/// same units: the exponent `|x| log2(e)` in units of `2^-8`, rounded half to
/// even, gives `w = exp2_neg`, and the quotient `x 2^30 / (2^30 + w)` (or
/// `x w / (2^30 + w)` below 0) is rounded half to even.
pub fn silu(x: i64) -> i64 {
    let scaled = u128::from(x.unsigned_abs()) * u128::from(LOG2_E);
    let (whole, rest, half) = (scaled >> 40, scaled & ((1u128 << 40) - 1), 1u128 << 39);
    let d = whole + u128::from(rest > half || (rest == half && whole & 1 == 1));
    let w = u128::from(exp2_neg(u32::try_from(d).unwrap_or(u32::MAX)));
    let denominator = (1u128 << 30) + w;
    let numerator = u128::from(x.unsigned_abs()) * if x >= 0 { 1u128 << 30 } else { w };
    let (quotient, rest) = (numerator / denominator, numerator % denominator);
    let up = 2 * rest > denominator || (2 * rest == denominator && quotient & 1 == 1);
    let magnitude = (quotient + u128::from(up)) as i64;
    if x >= 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// `n / d` rounded half to even, for `d > 0`: the rule of every narrowing.
fn divide_even(n: u128, d: u128) -> u128 {
    let (quotient, rest) = (n / d, n % d);
    quotient + u128::from(2 * rest > d || (2 * rest == d && quotient & 1 == 1))
}

/// The integer square root, `r^2 <= value < (r + 1)^2`, as decision 43 decides it,
/// digit by digit.
fn integer_root(value: u128) -> u128 {
    let (mut rest, mut root, mut bit) = (value, 0u128, 1u128 << 126);
    while bit > value {
        bit >>= 2;
    }
    while bit != 0 {
        if rest >= root + bit {
            rest -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

/// RMSNorm in integers. Values and weights are in units of `2^-16`, below
/// `2^31` in size, at most `2^16` of them; `epsilon` is in units of `2^-32`,
/// below `2^32`. The sum of squares plus `n epsilon` is exact; `r` is the
/// largest integer with `n r^2 <= (that sum) 2^32`; each output is `x g 2^16 / r` rounded half to even, in units of
/// `2^-16`. `None` for a row past the bounds or whose root is 0.
pub fn rms_norm(values: &[i64], weights: &[i64], epsilon: u64) -> Option<Vec<i64>> {
    let limit = 1u64 << 31;
    if values.is_empty()
        || values.len() > 1 << 16
        || values.len() != weights.len()
        || epsilon >= 1 << 32
        || values
            .iter()
            .chain(weights)
            .any(|v| v.unsigned_abs() >= limit)
    {
        return None;
    }
    let n = values.len() as u128;
    let squares: u128 = values
        .iter()
        .map(|&x| u128::from(x.unsigned_abs()).pow(2))
        .sum();
    // The floor root of the floored mean is the floor root of the mean: one narrowing.
    let r = integer_root(((squares + n * u128::from(epsilon)) << 32) / n);
    if r == 0 {
        return None;
    }
    Some(
        values
            .iter()
            .zip(weights)
            .map(|(&x, &g)| {
                let numerator = i128::from(x) * i128::from(g) * (1i128 << 16);
                let magnitude = divide_even(numerator.unsigned_abs(), r) as i64;
                if numerator < 0 {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect(),
    )
}

/// One in the fixed point rotary frequencies are generated in: `2^-62`.
const ROPE_ONE: u128 = 1 << 62;

/// `log2(x)` for `x >= 1` in units of `2^-62`, `x` too: the whole part by
/// halving, then one bit per squaring, cut down at each step.
fn log2_fixed(x: u128) -> u128 {
    let (mut m, mut result) = (x, 0u128);
    while m >= 2 * ROPE_ONE {
        m >>= 1;
        result += ROPE_ONE;
    }
    for bit in (0..62).rev() {
        m = (m * m) >> 62;
        if m >= 2 * ROPE_ONE {
            m >>= 1;
            result |= 1 << bit;
        }
    }
    result
}

/// `2^e` for `e` in units of `2^-62`, `e` below `2^64` and `2^e` below
/// `2^64`: the fraction as a product of `2^(2^-j)`, each the integer square
/// root of the one before, then shifted by the whole part.
fn pow2_fixed(e: u128) -> Option<u64> {
    let (whole, fraction) = (e >> 62, e & (ROPE_ONE - 1));
    let (mut power, mut product) = (2 * ROPE_ONE, ROPE_ONE);
    for bit in (0..62).rev() {
        power = integer_root(power << 62);
        if (fraction >> bit) & 1 == 1 {
            product = (product * power) >> 62;
        }
    }
    let shifted = if whole >= 62 {
        product.checked_shl((whole - 62) as u32)?
    } else {
        product >> (62 - whole)
    };
    u64::try_from(shifted).ok()
}

/// How a model scales its rotary frequencies, as its config names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeScaling {
    /// The frequencies as `theta` gives them.
    None,
    /// Llama 3's: kept above `high / original` turns per position, divided by
    /// `factor` below `low / original`, and blended linearly between.
    Llama3 {
        /// `factor`, a whole number.
        factor: u32,
        /// `low_freq_factor`, a whole number.
        low: u32,
        /// `high_freq_factor`, a whole number above `low`.
        high: u32,
        /// `original_max_position_embeddings`.
        original: u32,
    },
}

/// Llama 3's scaling of one frequency `f`, in units of `2^-64` of a turn per
/// position, rounded half to even where it divides.
fn llama3_scale(f: u64, factor: u32, low: u32, high: u32, original: u32) -> Option<u64> {
    let (f, factor) = (u128::from(f), u128::from(factor));
    let reach = f * u128::from(original);
    if reach > u128::from(high) << 64 {
        return u64::try_from(f).ok();
    }
    if reach < u128::from(low) << 64 {
        return u64::try_from(divide_even(f, factor)).ok();
    }
    // s = (original f - low 2^64) / ((high - low) 2^64), and
    // f' = f ((1 - s) / factor + s) = f (D - N + N factor) / (factor D).
    let numerator = reach - (u128::from(low) << 64);
    let denominator = u128::from(high - low) << 64;
    let blend = denominator.checked_add(numerator.checked_mul(factor - 1)?)?;
    let scaled = f.checked_mul(blend)?;
    u64::try_from(divide_even(scaled, factor.checked_mul(denominator)?)).ok()
}

/// The rotary frequency of each of a head's `head_dim / 2` pairs, in units of
/// `2^-64` of a turn per position: `theta^(-2i / head_dim) / (2 pi)`, generated
/// in integers over `2^62` (logarithms by squaring, powers by square roots,
/// `pi` by Machin's formula), then scaled as the config names. `None` for an
/// odd or zero head size, a `theta` below 2, or a scaling past its bounds.
pub fn rope_frequencies(head_dim: u32, theta: u64, scaling: RopeScaling) -> Option<Vec<u64>> {
    if head_dim == 0 || head_dim % 2 == 1 || head_dim > 1024 || theta < 2 {
        return None;
    }
    if let RopeScaling::Llama3 {
        factor,
        low,
        high,
        original,
    } = scaling
    {
        if factor == 0 || low == 0 || high <= low || original == 0 {
            return None;
        }
    }
    let pi = (16 * atan_inverse(5) - 4 * atan_inverse(239)) >> 58;
    let (log_theta, log_pi) = (log2_fixed(u128::from(theta) << 62), log2_fixed(pi));
    let half = u128::from(head_dim / 2);
    (0..half)
        .map(|i| {
            // 2^64 theta^(-i / half) / (2 pi) = 2^(63 - (i / half) log2 theta - log2 pi).
            let exponent = (63 * ROPE_ONE).checked_sub(i * log_theta / half + log_pi)?;
            let f = pow2_fixed(exponent)?;
            match scaling {
                RopeScaling::None => Some(f),
                RopeScaling::Llama3 {
                    factor,
                    low,
                    high,
                    original,
                } => llama3_scale(f, factor, low, high, original),
            }
        })
        .collect()
}

/// The rotary angle of `position` at frequency `frequency`, in units of
/// `2^-32` of a turn: `position * frequency` modulo a whole turn, rounded half
/// to even to `2^-32` of a turn, which EN-45's `sine_turn` and `cosine_turn` read.
pub fn rope_angle(position: u32, frequency: u64) -> u32 {
    let turns = (u128::from(position) * u128::from(frequency)) as u64;
    let (whole, rest) = (turns >> 32, turns & 0xffff_ffff);
    let up = rest > 1 << 31 || (rest == 1 << 31 && whole & 1 == 1);
    (whole + u64::from(up)) as u32
}

/// LayerNorm in integers. Values, weights and biases are in units of `2^-16`,
/// below `2^31` in size, at most `2^16` of them; `epsilon` is in units of
/// `2^-32`, below `2^32`. The centred values `n x - sum(x)` and the variance's
/// numerator `V = n sum(x^2) - sum(x)^2` are exact; `r` is the largest integer
/// with `n^2 r^2 <= (V + n^2 epsilon) 2^32`; each output is
/// `(n x - sum(x)) g 2^16 / (n r)` rounded half to even, plus `b`, in units of
/// `2^-16`. `None` for a row past the bounds or whose root is 0.
pub fn layer_norm(
    values: &[i64],
    weights: &[i64],
    biases: &[i64],
    epsilon: u64,
) -> Option<Vec<i64>> {
    let limit = 1u64 << 31;
    if values.is_empty()
        || values.len() > 1 << 16
        || values.len() != weights.len()
        || values.len() != biases.len()
        || epsilon >= 1 << 32
        || values
            .iter()
            .chain(weights)
            .chain(biases)
            .any(|v| v.unsigned_abs() >= limit)
    {
        return None;
    }
    let n = values.len() as i128;
    let sum: i128 = values.iter().map(|&x| i128::from(x)).sum();
    let squares: u128 = values
        .iter()
        .map(|&x| u128::from(x.unsigned_abs()).pow(2))
        .sum();
    // n sum(x^2) >= sum(x)^2 (Cauchy-Schwarz), so V is never negative.
    let v = (n as u128) * squares - sum.unsigned_abs().pow(2);
    let nn = (n * n) as u128;
    // The floor root of the floored quotient is the floor root of the quotient: one narrowing.
    let r = integer_root(((v + nn * u128::from(epsilon)) << 32) / nn);
    if r == 0 {
        return None;
    }
    let denominator = n as u128 * r;
    Some(
        values
            .iter()
            .zip(weights)
            .zip(biases)
            .map(|((&x, &g), &b)| {
                let numerator = (n * i128::from(x) - sum) * i128::from(g) * (1i128 << 16);
                let magnitude = divide_even(numerator.unsigned_abs(), denominator) as i64;
                b + if numerator < 0 { -magnitude } else { magnitude }
            })
            .collect(),
    )
}

/// The fraction GELU's series are carried in: `2^-40`.
const GELU_FRACTION: u32 = 40;

/// `sqrt(2 / pi)` in units of `2^-40`: the integer square root of
/// `2^121 / floor(2^40 pi)`, `pi` by Machin's formula.
fn gelu_constant() -> u128 {
    let pi = (16 * atan_inverse(5) - 4 * atan_inverse(239)) >> 80;
    integer_root((1u128 << 121) / pi)
}

/// GELU, `x Phi(x)`, for `x` in units of `2^-16`, in the same units. At or
/// above 8 it is `x` and at or below -8 it is 0, each within `2^-40` of the
/// true value. Between, with `X = |x|` and `Y = X^2`,
/// `erf(X / sqrt 2) = sqrt(2 / pi) X A / B` for the positive series
/// `A = sum Y^k / (2k+1)!!` and `B = e^(Y/2) = sum (Y/2)^k / k!`, every term cut
/// down to `2^-40`, so nothing cancels; the output `(x + X erf) / 2` is rounded
/// half to even once.
pub fn gelu(x: i64) -> i64 {
    let eight = 8i64 << SILU_FRACTION;
    if x >= eight {
        return x;
    }
    if x <= -eight {
        return 0;
    }
    let one = 1u128 << GELU_FRACTION;
    let magnitude = u128::from(x.unsigned_abs());
    // Y in units of 2^-32, below 2^38.
    let y = magnitude * magnitude;
    let (mut a, mut b) = (one, one);
    let (mut sum_a, mut sum_b) = (one, one);
    let mut k = 1u128;
    while a > 0 || b > 0 {
        a = a * y / ((2 * k + 1) << 32);
        b = b * y / (k << 33);
        sum_a += a;
        sum_b += b;
        k += 1;
    }
    let ratio = (sum_a << GELU_FRACTION) / sum_b;
    // erf(X / sqrt 2) in units of 2^-40.
    let erf = (gelu_constant() * magnitude * ratio) >> (GELU_FRACTION + SILU_FRACTION);
    // (x + X erf) / 2, in units of 2^-56 before the one narrowing.
    let doubled = (i128::from(x) << GELU_FRACTION) + (magnitude * erf) as i128;
    let rounded = divide_even(doubled.unsigned_abs(), 1u128 << (GELU_FRACTION + 1)) as i64;
    if doubled < 0 {
        -rounded
    } else {
        rounded
    }
}
