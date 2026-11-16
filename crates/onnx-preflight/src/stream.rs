//! Reads an ONNX file as its bytes arrive.
//!
//! A model's weights are most of its file, and a browser cannot hold a file of
//! a gigabyte to learn what is in it. This reader is fed the file in pieces,
//! in whatever sizes the network hands them over, and says what it passes: the
//! opsets, each node, and each tensor's name, shape, type and where its data
//! lies. It never holds a tensor's data. When it reaches one it steps over it,
//! and [`Stream::position`] then names the byte it needs next, so a caller
//! reading by ranges can ask for that byte and one reading a stream can let
//! the bytes between go by, hashing them as they pass.
//!
//! What it holds is what it cannot yet read: a field whose bytes have not all
//! arrived. A node is held whole, because its parts are read together; that is
//! bounded by [`RECORD_LIMIT`], and a node larger than it is refused, not
//! buffered without end.

use crate::wire::{text, Encoding, Reader};
use crate::{field, OnnxError};

/// The most bytes one node or one opset record may be. The largest node in the
/// exports this engine reads is a few kilobytes; a constant folded into a node
/// can be megabytes. Past this the file is asking the reader to hold weights.
pub const RECORD_LIMIT: u64 = 16 << 20;

/// Field numbers of `NodeProto`, `AttributeProto`, `TensorProto` and
/// `StringStringEntryProto`, from `onnx.proto`.
mod number {
    pub(super) const NODE_INPUT: u64 = 1;
    pub(super) const NODE_OUTPUT: u64 = 2;
    pub(super) const NODE_ATTRIBUTE: u64 = 5;
    pub(super) const ATTRIBUTE_NAME: u64 = 1;
    pub(super) const ATTRIBUTE_FLOAT: u64 = 2;
    pub(super) const ATTRIBUTE_INT: u64 = 3;
    pub(super) const TENSOR_DIMS: u64 = 1;
    pub(super) const TENSOR_DATA_TYPE: u64 = 2;
    pub(super) const TENSOR_NAME: u64 = 8;
    pub(super) const TENSOR_RAW_DATA: u64 = 9;
    pub(super) const TENSOR_EXTERNAL_DATA: u64 = 13;
    pub(super) const TENSOR_DATA_LOCATION: u64 = 14;
    pub(super) const ENTRY_KEY: u64 = 1;
    pub(super) const ENTRY_VALUE: u64 = 2;
    /// `float_data`, `int32_data`, `string_data`, `int64_data`, `double_data`
    /// and `uint64_data`: a tensor's values written as numbers, not as bytes.
    pub(super) const TENSOR_TYPED: [u64; 6] = [4, 5, 6, 7, 10, 11];
}

/// One node of the graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// Its operator's name, such as `MatMul`.
    pub op_type: String,
    /// The domain the operator belongs to; empty for the ONNX operators.
    pub domain: String,
    /// The names of the values it reads, in order.
    pub inputs: Vec<String>,
    /// The names of the values it writes, in order.
    pub outputs: Vec<String>,
    /// Its whole-number attributes, such as a block size, in file order.
    pub ints: Vec<(String, i64)>,
    /// Its single-precision attributes, such as an epsilon, in file order.
    pub floats: Vec<(String, f32)>,
}

/// Where a tensor's data lies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Data {
    /// The tensor carries none.
    None,
    /// Written as numbers inside the tensor's record, which small constants
    /// are. The reader steps over them and does not say where they are.
    Typed,
    /// Bytes inside this file.
    Inline {
        /// The file offset of the first byte.
        offset: u64,
        /// How many bytes.
        length: u64,
    },
    /// Bytes in another file beside this one.
    External {
        /// That file's name, relative to this one.
        location: String,
        /// The offset of the first byte in it.
        offset: u64,
        /// How many bytes; `None` when the record leaves it to the type and shape.
        length: Option<u64>,
    },
}

/// One tensor the graph carries: everything about it but its data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// Its name, which nodes read it by.
    pub name: String,
    /// Its shape.
    pub dims: Vec<u64>,
    /// Its element type, as `onnx.proto` numbers them: 1 is f32, 2 u8, 7 i64, 10 f16.
    pub data_type: u64,
    /// Where its data lies.
    pub data: Data,
}

/// What the reader passed.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// An opset the model declares, for any domain.
    Opset {
        /// The domain; empty for the ONNX operators.
        domain: String,
        /// The version.
        version: u64,
    },
    /// A node, in file order.
    Node(Node),
    /// A tensor, once its record has been read to its end.
    Tensor(Tensor),
    /// The reader is about to step over a tensor's data inside the file: the
    /// bytes from `offset` for `length`. Reported as the data begins, before
    /// the tensor's own event, so a caller can take those bytes as they pass;
    /// the [`Event::Tensor`] that follows names the same span.
    Data {
        /// The file offset of the first byte.
        offset: u64,
        /// How many bytes.
        length: u64,
    },
}

#[derive(Default)]
struct Building {
    name: String,
    dims: Vec<u64>,
    data_type: u64,
    inline: Option<(u64, u64)>,
    typed: bool,
    external: bool,
    location: Option<String>,
    offset: Option<u64>,
    length: Option<u64>,
}

/// Which record the cursor is in, without what is being built there.
#[derive(Clone, Copy)]
enum Inside {
    Model,
    Graph(u64),
    Tensor,
}

enum Place {
    Model,
    Graph {
        end: u64,
    },
    Tensor {
        graph_end: u64,
        end: u64,
        building: Building,
    },
}

/// The reader. Feed it the file from its first byte; see the module's words.
pub struct Stream {
    /// Bytes fed and not yet read.
    held: Vec<u8>,
    /// How many of `held` have been read.
    cursor: usize,
    /// The file offset of `held[0]`.
    at: u64,
    place: Place,
    saw_graph: bool,
    saw_default_opset: bool,
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}

/// A variable-length number at the start of `bytes`, and how many bytes it
/// took; `None` when the bytes end before the number does.
fn number(bytes: &[u8]) -> Result<Option<(u64, usize)>, OnnxError> {
    let mut value = 0u64;
    for (step, (shift, byte)) in (0u32..).step_by(7).zip(bytes.iter().take(10)).enumerate() {
        // As `wire::Reader::number`: the tenth byte's high bits are a sign
        // extension, and a negative number is read as the large one it stands for.
        value |= u64::from(byte & 0x7f).wrapping_shl(shift);
        if byte & 0x80 == 0 {
            return Ok(Some((value, step + 1)));
        }
    }
    if bytes.len() >= 10 {
        Err(OnnxError::NumberTooLong)
    } else {
        Ok(None)
    }
}

impl Stream {
    /// A reader at the start of a file.
    #[must_use]
    pub fn new() -> Self {
        Self {
            held: Vec::new(),
            cursor: 0,
            at: 0,
            place: Place::Model,
            saw_graph: false,
            saw_default_opset: false,
        }
    }

    /// The file offset of the next byte this reader must be fed. After a
    /// tensor's data it is past that data: the bytes between are not wanted.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.at + self.held.len() as u64
    }

    /// How many bytes it has been fed and not yet read: the part of a field
    /// that has not all arrived. Never a tensor's data.
    #[must_use]
    pub fn held(&self) -> usize {
        self.held.len()
    }

    /// Whether the reader is inside the graph, where the weights are.
    #[must_use]
    pub fn in_graph(&self) -> bool {
        !matches!(self.place, Place::Model)
    }

    /// Leave the graph without reading the rest of it: [`Stream::position`]
    /// becomes the first byte after the graph, where the opsets are. For a
    /// caller that wants the nodes, which come first, and not the tensors. A
    /// tensor whose record was under way is not reported.
    pub fn leave_graph(&mut self) {
        let end = match self.place {
            Place::Model => return,
            Place::Graph { end } | Place::Tensor { graph_end: end, .. } => end,
        };
        self.held.clear();
        self.cursor = 0;
        self.at = end;
        self.place = Place::Model;
    }

    /// Feed the bytes that begin at [`Stream::position`]; get what they completed.
    ///
    /// # Errors
    ///
    /// The file is not an ONNX model this reader follows; see [`OnnxError`].
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<Event>, OnnxError> {
        self.held.extend_from_slice(bytes);
        let mut events = Vec::new();
        while self.step(&mut events)? {}
        self.held.drain(..self.cursor);
        self.at += self.cursor as u64;
        self.cursor = 0;
        Ok(events)
    }

    /// The file has ended, `length` bytes long.
    ///
    /// The length is needed because the reader is not fed the bytes it steps
    /// over: a file that ends inside a tensor's data, or inside any field the
    /// reader passes by, leaves nothing held to show it.
    ///
    /// # Errors
    ///
    /// [`OnnxError::Truncated`] if it ended inside a field, the graph or a
    /// tensor; [`OnnxError::NoGraph`] and [`OnnxError::NoDefaultOpset`] as
    /// [`crate::read`] gives them.
    pub fn finish(&self, length: u64) -> Result<(), OnnxError> {
        if !matches!(self.place, Place::Model)
            || self.cursor < self.held.len()
            || self.position() != length
        {
            return Err(OnnxError::Truncated);
        }
        if !self.saw_graph {
            return Err(OnnxError::NoGraph);
        }
        if !self.saw_default_opset {
            return Err(OnnxError::NoDefaultOpset);
        }
        Ok(())
    }

    fn offset(&self) -> u64 {
        self.at + self.cursor as u64
    }

    fn rest(&self) -> &[u8] {
        self.held.get(self.cursor..).unwrap_or_default()
    }

    /// Step over `count` bytes from the cursor, held or not.
    fn pass(&mut self, count: u64) {
        let held = self.held.len() - self.cursor;
        if let Some(count) = usize::try_from(count).ok().filter(|count| *count <= held) {
            self.cursor += count;
        } else {
            self.at = self.offset() + count;
            self.held.clear();
            self.cursor = 0;
        }
    }

    /// Read one field. `Ok(false)` when the bytes held do not finish one.
    fn step(&mut self, events: &mut Vec<Event>) -> Result<bool, OnnxError> {
        // The end of the record the cursor is in, if it is in one.
        let end = match &self.place {
            Place::Model => None,
            Place::Graph { end } | Place::Tensor { end, .. } => Some(*end),
        };
        if let Some(end) = end {
            if self.offset() == end {
                self.close(events)?;
                return Ok(true);
            }
        }
        if self.cursor == self.held.len() {
            return Ok(false);
        }

        let Some((key, key_width)) = number(self.rest())? else {
            return Ok(false);
        };
        let field = key >> 3;
        let encoding = Encoding::of(key)?;
        let after_key = self.rest().get(key_width..).unwrap_or_default();

        // How far the field runs, so a field that leaves its record is caught
        // before any of it is believed. `scalar` is a number field's value.
        let (head, payload, scalar) = match encoding {
            Encoding::Number => match number(after_key)? {
                Some((value, width)) => (key_width + width, 0, value),
                None => return Ok(false),
            },
            Encoding::Eight => (key_width, 8, 0),
            Encoding::Four => (key_width, 4, 0),
            Encoding::Bytes => match number(after_key)? {
                Some((length, width)) => (key_width + width, length, 0),
                None => return Ok(false),
            },
        };
        let payload_at = self.offset() + head as u64;
        let field_end = payload_at
            .checked_add(payload)
            .ok_or(OnnxError::Truncated)?;
        if end.is_some_and(|end| field_end > end) {
            return Err(OnnxError::Truncated);
        }

        let inside = match self.place {
            Place::Model => Inside::Model,
            Place::Graph { end } => Inside::Graph(end),
            Place::Tensor { .. } => Inside::Tensor,
        };
        match (inside, field, encoding) {
            (Inside::Model, field::MODEL_GRAPH, Encoding::Bytes) => {
                self.saw_graph = true;
                self.place = Place::Graph { end: field_end };
                self.cursor += head;
            }
            (Inside::Model, field::MODEL_OPSET, Encoding::Bytes) => {
                let Some(record) = self.whole(head, payload)? else {
                    return Ok(false);
                };
                let (domain, version) = opset(&record)?;
                self.saw_default_opset |= domain.is_empty();
                events.push(Event::Opset { domain, version });
                self.cursor += head + record.len();
            }
            (Inside::Graph(graph_end), field::GRAPH_INITIALIZER, Encoding::Bytes) => {
                self.place = Place::Tensor {
                    graph_end,
                    end: field_end,
                    building: Building::default(),
                };
                self.cursor += head;
            }
            (Inside::Graph(_), field::GRAPH_NODE, Encoding::Bytes) => {
                let Some(record) = self.whole(head, payload)? else {
                    return Ok(false);
                };
                events.push(Event::Node(node(&record)?));
                self.cursor += head + record.len();
            }
            (Inside::Tensor, number::TENSOR_DIMS, Encoding::Number) => {
                self.building(|building| building.dims.push(scalar));
                self.cursor += head;
            }
            (Inside::Tensor, number::TENSOR_DIMS, Encoding::Bytes) => {
                let Some(record) = self.whole(head, payload)? else {
                    return Ok(false);
                };
                let mut reader = Reader::new(&record);
                let mut dims = Vec::new();
                while reader.more() {
                    dims.push(reader.number()?);
                }
                self.building(|building| building.dims.extend(dims));
                self.cursor += head + record.len();
            }
            (Inside::Tensor, number::TENSOR_DATA_TYPE, Encoding::Number) => {
                self.building(|building| building.data_type = scalar);
                self.cursor += head;
            }
            (Inside::Tensor, number::TENSOR_DATA_LOCATION, Encoding::Number) => {
                self.building(|building| building.external |= scalar == 1);
                self.cursor += head;
            }
            (Inside::Tensor, number::TENSOR_NAME, Encoding::Bytes) => {
                let Some(record) = self.whole(head, payload)? else {
                    return Ok(false);
                };
                let name = text(&record)?;
                self.building(|building| building.name = name);
                self.cursor += head + record.len();
            }
            (Inside::Tensor, number::TENSOR_EXTERNAL_DATA, Encoding::Bytes) => {
                let Some(record) = self.whole(head, payload)? else {
                    return Ok(false);
                };
                let (key, value) = entry(&record)?;
                let count = |value: &str| {
                    value
                        .parse::<u64>()
                        .map_err(|_| OnnxError::ExternalNotFollowed)
                };
                let (offset, length) = match key.as_str() {
                    "offset" => (Some(count(&value)?), None),
                    "length" => (None, Some(count(&value)?)),
                    _ => (None, None),
                };
                self.building(|building| {
                    building.external = true;
                    match key.as_str() {
                        "location" => building.location = Some(value),
                        "offset" => building.offset = offset,
                        "length" => building.length = length,
                        // A checksum, or a key a later schema adds: not where the data lies.
                        _ => {}
                    }
                });
                self.cursor += head + record.len();
            }
            (Inside::Tensor, number::TENSOR_RAW_DATA, Encoding::Bytes) => {
                self.building(|building| building.inline = Some((payload_at, payload)));
                events.push(Event::Data {
                    offset: payload_at,
                    length: payload,
                });
                self.cursor += head;
                self.pass(payload);
            }
            (inside, field, _) => {
                if matches!(inside, Inside::Tensor) {
                    self.building(|building| {
                        building.typed |= number::TENSOR_TYPED.contains(&field);
                    });
                }
                self.cursor += head;
                self.pass(payload);
            }
        }
        Ok(true)
    }

    /// The bytes of a field read whole, once they are all held.
    fn whole(&self, head: usize, payload: u64) -> Result<Option<Vec<u8>>, OnnxError> {
        if payload > RECORD_LIMIT {
            return Err(OnnxError::RecordTooLarge);
        }
        let length = usize::try_from(payload).map_err(|_| OnnxError::RecordTooLarge)?;
        let from = self.cursor + head;
        Ok(self.held.get(from..from + length).map(<[u8]>::to_vec))
    }

    /// Change the tensor being read, if one is.
    fn building(&mut self, change: impl FnOnce(&mut Building)) {
        if let Place::Tensor { building, .. } = &mut self.place {
            change(building);
        }
    }

    /// The cursor is at the end of the record it is in: report it and step out.
    fn close(&mut self, events: &mut Vec<Event>) -> Result<(), OnnxError> {
        match core::mem::replace(&mut self.place, Place::Model) {
            Place::Model | Place::Graph { .. } => {}
            Place::Tensor {
                graph_end,
                building,
                ..
            } => {
                self.place = Place::Graph { end: graph_end };
                let data = if building.external {
                    Data::External {
                        location: building.location.ok_or(OnnxError::ExternalNotFollowed)?,
                        offset: building.offset.unwrap_or(0),
                        length: building.length,
                    }
                } else if let Some((offset, length)) = building.inline {
                    Data::Inline { offset, length }
                } else if building.typed {
                    Data::Typed
                } else {
                    Data::None
                };
                events.push(Event::Tensor(Tensor {
                    name: building.name,
                    dims: building.dims,
                    data_type: building.data_type,
                    data,
                }));
            }
        }
        Ok(())
    }
}

fn opset(record: &[u8]) -> Result<(String, u64), OnnxError> {
    let mut domain = String::new();
    let mut version = 0;
    let mut reader = Reader::new(record);
    while reader.more() {
        match reader.field()? {
            (field::OPSET_DOMAIN, Encoding::Bytes) => domain = text(reader.delimited()?)?,
            (field::OPSET_VERSION, Encoding::Number) => version = reader.number()?,
            (_, encoding) => reader.step_over(encoding)?,
        }
    }
    Ok((domain, version))
}

fn entry(record: &[u8]) -> Result<(String, String), OnnxError> {
    let mut key = String::new();
    let mut value = String::new();
    let mut reader = Reader::new(record);
    while reader.more() {
        match reader.field()? {
            (number::ENTRY_KEY, Encoding::Bytes) => key = text(reader.delimited()?)?,
            (number::ENTRY_VALUE, Encoding::Bytes) => value = text(reader.delimited()?)?,
            (_, encoding) => reader.step_over(encoding)?,
        }
    }
    Ok((key, value))
}

fn node(record: &[u8]) -> Result<Node, OnnxError> {
    let mut out = Node {
        op_type: String::new(),
        domain: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        ints: Vec::new(),
        floats: Vec::new(),
    };
    let mut reader = Reader::new(record);
    while reader.more() {
        match reader.field()? {
            (number::NODE_INPUT, Encoding::Bytes) => out.inputs.push(text(reader.delimited()?)?),
            (number::NODE_OUTPUT, Encoding::Bytes) => out.outputs.push(text(reader.delimited()?)?),
            (field::NODE_OP_TYPE, Encoding::Bytes) => out.op_type = text(reader.delimited()?)?,
            (field::NODE_DOMAIN, Encoding::Bytes) => out.domain = text(reader.delimited()?)?,
            (number::NODE_ATTRIBUTE, Encoding::Bytes) => {
                let mut name = String::new();
                let mut int = None;
                let mut float = None;
                let mut attribute = Reader::new(reader.delimited()?);
                while attribute.more() {
                    match attribute.field()? {
                        (number::ATTRIBUTE_NAME, Encoding::Bytes) => {
                            name = text(attribute.delimited()?)?
                        }
                        // Two's complement in sixty-four bits, as the format writes a negative number.
                        (number::ATTRIBUTE_INT, Encoding::Number) => {
                            int = Some(i64::from_le_bytes(attribute.number()?.to_le_bytes()));
                        }
                        (number::ATTRIBUTE_FLOAT, Encoding::Four) => {
                            float = Some(f32::from_le_bytes(attribute.four()?))
                        }
                        (_, encoding) => attribute.step_over(encoding)?,
                    }
                }
                if let Some(int) = int {
                    out.ints.push((name, int));
                } else if let Some(float) = float {
                    out.floats.push((name, float));
                }
            }
            (_, encoding) => reader.step_over(encoding)?,
        }
    }
    Ok(out)
}
