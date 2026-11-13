//! The safetensors header: where each tensor's bytes lie.
//!
//! A safetensors file begins with eight bytes holding the header's length as a
//! little-endian number, then that many bytes of JSON, then the tensor data.
//! Each entry in the JSON names a tensor and gives its dtype, its shape and the
//! pair of offsets bounding its bytes, counted from the start of the data.
//!
//! Nothing here decides whether a tensor's bytes lie where they claim. The
//! reader turns each entry into the kernel's tensor-range question and folds
//! the answers in file order, passing each accepted tensor's end as the next
//! tensor's previous end, which is how the kernel sees that the tensors run
//! forward and stay inside the file.

use hologram_ai_kernel_client::range;
use serde_json::Value;

use crate::Kernel;

/// The key the format reserves for its own metadata. It is not a tensor.
const METADATA: &str = "__metadata__";

/// One tensor, as the header describes it and the file places it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// The name the checkpoint gives it, such as `model.embed_tokens.weight`.
    pub name: String,
    /// The element type the header names, such as `F32` or `BF16`.
    pub dtype: String,
    /// The dimensions, outermost first.
    pub shape: Vec<u64>,
    /// Where its bytes start, counted from the start of the file, not from the
    /// start of the data. This is what the kernel is asked about.
    pub offset: u64,
    /// How many bytes it occupies.
    pub length: u64,
}

/// Why a safetensors header could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderError {
    /// `FL-01`
    TooShortForLength,
    /// `FL-02`
    HeaderExceedsFile,
    /// `FL-03`
    NotJson,
    /// `FL-04`
    TensorNotDescribed,
    /// `FL-05`
    OffsetsDoNotRunForward,
    /// `FL-06`: the kernel's own answer, carried as it gave it.
    RangeRefused(range::Verdict),
}

/// Read `file`'s header into its tensors, in file order, each one's byte range
/// accepted by `kernel`.
///
/// `file` is the whole file. The reader needs its length to ask the kernel
/// anything, and a header that describes bytes the file does not have is the
/// first thing a corrupt or truncated download looks like.
pub fn read(file: &[u8], kernel: &mut dyn Kernel) -> Result<Vec<Tensor>, HeaderError> {
    let length: [u8; 8] = match file.get(..8).map(<[u8; 8]>::try_from) {
        Some(Ok(bytes)) => bytes,
        _ => return Err(HeaderError::TooShortForLength),
    };
    let file_length = file.len() as u64;
    // Where the data begins: the eight bytes of length, then the header. It is
    // also the first tensor's previous end, so a tensor that starts inside the
    // header is one the kernel sees reaching backwards.
    // A declared length near what a number holds saturates, and a length that
    // saturates is refused below like any other that reaches past the file.
    // The sum is never wrong in a way a caller could see: the header has to lie
    // inside the file, and no file is that long.
    let data_start = 8u64.saturating_add(u64::from_le_bytes(length));
    let text = usize::try_from(data_start)
        .ok()
        .and_then(|end| file.get(8..end))
        .ok_or(HeaderError::HeaderExceedsFile)?;

    let entries = match serde_json::from_slice(text) {
        Ok(Value::Object(entries)) => entries,
        _ => return Err(HeaderError::NotJson),
    };

    let mut tensors = Vec::new();
    for (name, value) in entries {
        if name == METADATA {
            continue;
        }
        tensors.push(describe(name, &value, data_start)?);
    }
    // The header is a JSON object, and an object's entries carry no order the
    // file has to respect. The tensors are put in the order their bytes lie,
    // which is the order the kernel's question is about.
    tensors.sort_by_key(|tensor| (tensor.offset, tensor.length));

    let mut previous_end = data_start;
    for tensor in &tensors {
        let answer = kernel.ask(&range::request(
            file_length,
            previous_end,
            tensor.offset,
            tensor.length,
        ));
        match range::answer(&answer) {
            range::Verdict::Accept => {}
            refused => return Err(HeaderError::RangeRefused(refused)),
        }
        // Reached only on an accepted range, which the kernel accepts only when
        // it ends within the file; the sum is therefore at most the file's
        // length and the saturation never happens.
        previous_end = tensor.offset.saturating_add(tensor.length);
    }
    Ok(tensors)
}

/// One header entry as a tensor placed in the file.
fn describe(name: String, value: &Value, data_start: u64) -> Result<Tensor, HeaderError> {
    let entry = value.as_object().ok_or(HeaderError::TensorNotDescribed)?;
    let dtype = entry
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or(HeaderError::TensorNotDescribed)?
        .to_string();
    let shape = entry
        .get("shape")
        .and_then(Value::as_array)
        .ok_or(HeaderError::TensorNotDescribed)?
        .iter()
        .map(Value::as_u64)
        .collect::<Option<Vec<u64>>>()
        .ok_or(HeaderError::TensorNotDescribed)?;
    let offsets = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or(HeaderError::TensorNotDescribed)?;
    let (first, last) = match offsets.as_slice() {
        [first, last] => (
            first.as_u64().ok_or(HeaderError::TensorNotDescribed)?,
            last.as_u64().ok_or(HeaderError::TensorNotDescribed)?,
        ),
        _ => return Err(HeaderError::TensorNotDescribed),
    };
    let length = last
        .checked_sub(first)
        .ok_or(HeaderError::OffsetsDoNotRunForward)?;
    Ok(Tensor {
        name,
        dtype,
        shape,
        // A start past what a number can hold is not refused here: it saturates
        // to a place no file reaches, and the kernel refuses it as a range that
        // ends outside the file.
        offset: data_start.saturating_add(first),
        length,
    })
}
