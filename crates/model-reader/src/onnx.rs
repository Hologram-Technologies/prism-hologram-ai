//! Imports an ONNX model from its own files, as their bytes arrive.
//!
//! A model on the hub is a graph file and, for a large one, data files beside
//! it. Every byte of each is taken once, as it comes: the whole file is hashed
//! for the BLAKE3 its record gives, and each tensor's data is hashed on its
//! own for its address, the BLAKE3 of its raw bytes. The graph is read by
//! [`hologram_ai_onnx_preflight::stream::Stream`], which never holds a
//! tensor's data; this module hashes those bytes as they pass it.
//!
//! Nothing is decided here. Each tensor's span is put to the kernel (decision
//! 23), each operator the graph names (decision 18), the length of each 4-bit
//! weight's codes and scales (decision 20), and each whole file against its
//! record (decision 5). A refusal stops the import with the kernel's own
//! verdict.
//!
//! What it produces is the model's tensor table, each tensor with its address:
//! the row a store keyed by address files the weights under.

use std::collections::{BTreeMap, BTreeSet};

use hologram_ai_download::{Arriving, Label};
use hologram_ai_kernel_client::{admission, chunk, nbits, opname};
use hologram_ai_onnx_preflight::stream::{Data, Event, Node, Stream};
use hologram_ai_onnx_preflight::OnnxError;

use crate::Kernel;

/// The domain of the runtime's fused extension operators. A graph that names
/// any of them is in the fused export style, except `MatMulNBits`: that is how
/// 4-bit weights are stored in either style, and Qwen2.5's decomposed export
/// carries it beside a `Pow`, `ReduceMean` and `Softmax` that only the
/// decomposed list holds.
const FUSED_DOMAIN: &str = "com.microsoft";
const EITHER_STYLE: &str = "MatMulNBits";

/// One file of a model, as its record gives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct File {
    /// Its name beside the graph; the graph's own name for the graph.
    pub name: String,
    /// 0 for the graph, then 1 onward for the data files, in record order.
    pub index: u8,
    /// How many data files the model declares beside the graph.
    pub declared: u8,
    /// Its length in bytes.
    pub length: u64,
    /// The BLAKE3 the record gives for the whole file: its address.
    pub address: Label,
}

/// A tensor of the model, with its address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// Its name, which the graph's nodes read it by.
    pub name: String,
    /// Its shape.
    pub dims: Vec<u64>,
    /// Its element type, as `onnx.proto` numbers them.
    pub data_type: u64,
    /// The file its data lies in, by index.
    pub file: u8,
    /// The offset of its first byte in that file.
    pub offset: u64,
    /// How many bytes.
    pub length: u64,
    /// The BLAKE3 of those bytes.
    pub address: Label,
    /// Its position among the graph's tensor records.
    pub at: TensorAt,
}

/// What an import reads from a model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    /// Every tensor whose data is bytes, in the order the graph names them.
    pub tensors: Vec<Tensor>,
    /// The distinct operators the graph names, in the order they first appear.
    pub operators: Vec<String>,
    /// Whether the graph is in the fused export style.
    pub fused: bool,
}

/// Why a model could not be imported.
///
/// A failure names what failed by its place, not by text the file wrote: a
/// tensor by its position among the graph's tensor records, an operator by its
/// position among the distinct operators in the order they first appear, a
/// file by its index in the record. The caller holds the graph and can say
/// what each place is called.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportError {
    /// `FL-43`
    Unreadable(OnnxError),
    /// `FL-44`: the tensor, and the kernel's verdict on its span.
    SpanRefused(TensorAt, chunk::Verdict),
    /// `FL-45`: the file, and the kernel's verdict on it.
    FileRefused(u8, admission::Verdict),
    /// `FL-46`: the operator, and the kernel's verdict on it.
    OperatorRefused(OperatorAt, opname::Verdict),
    /// `FL-47`: the weight tensor, its length, and the kernel's answer.
    WeightRefused(TensorAt, u64, nbits::Answer),
    /// `FL-48`: the tensor whose data is not where the import can read it.
    DataMissing(TensorAt),
}

/// A tensor's position among the graph's tensor records, from 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorAt(pub u32);

/// An operator's position among the distinct operators the graph names, in
/// the order they first appear, from 0: its place in [`Model::operators`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperatorAt(pub u32);

/// The place of the `n`-th item, saturated: no graph has four billion.
fn place(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// A span being hashed as its bytes pass.
struct Open {
    offset: u64,
    length: u64,
    hasher: Arriving,
}

impl Open {
    fn new(offset: u64, length: u64) -> Self {
        Self {
            offset,
            length,
            hasher: Arriving::new(),
        }
    }

    /// Take the part of `chunk`, which begins at file offset `at`, that lies
    /// in this span. Whether the span is now whole.
    fn take(&mut self, at: u64, chunk: &[u8]) -> bool {
        let end = self.offset + self.length;
        let from = self.offset.max(at);
        let to = end.min(at + chunk.len() as u64);
        if from < to {
            // Both are inside the chunk, so they are below its length.
            let (from, to) = ((from - at) as usize, (to - at) as usize);
            self.hasher.take(&chunk[from..to]);
        }
        self.hasher.received() == self.length
    }
}

/// Bytes taken in, hashed whole and span by span, each by BLAKE3.
struct Taking {
    file: File,
    whole: Arriving,
    at: u64,
    open: Vec<Open>,
    done: BTreeMap<u64, Label>,
}

impl Taking {
    fn new(file: File) -> Self {
        Self {
            file,
            whole: Arriving::new(),
            at: 0,
            open: Vec::new(),
            done: BTreeMap::new(),
        }
    }

    /// Start hashing a span whose bytes have not all passed yet.
    fn watch(&mut self, offset: u64, length: u64) {
        if length == 0 {
            self.done
                .insert(offset, Label(*blake3::hash(&[]).as_bytes()));
        } else {
            self.open.push(Open::new(offset, length));
        }
    }

    /// Hash `chunk` into the spans it covers. Spans are watched before the
    /// chunk that holds their first byte is taken.
    fn take(&mut self, chunk: &[u8]) {
        self.whole.take(chunk);
        let at = self.at;
        let mut still = Vec::with_capacity(self.open.len());
        for mut span in self.open.drain(..) {
            if span.take(at, chunk) {
                self.done.insert(span.offset, span.hasher.label());
            } else {
                still.push(span);
            }
        }
        self.open = still;
        self.at += chunk.len() as u64;
    }

    /// Ask the kernel about the whole file.
    fn finish(&self, kernel: &mut dyn Kernel) -> Result<(), ImportError> {
        let question = admission::request(
            &self.file.address.0,
            &self.whole.label().0,
            self.file.length,
            self.whole.received(),
        );
        match admission::answer(&kernel.ask(&question)) {
            admission::Verdict::Admit => Ok(()),
            refused => Err(ImportError::FileRefused(self.file.index, refused)),
        }
    }
}

/// Ask the kernel about one tensor's span; the end of the span it accepted.
fn span(
    kernel: &mut dyn Kernel,
    file: &File,
    tensor: TensorAt,
    offset: u64,
    length: u64,
    previous_end: u64,
) -> Result<u64, ImportError> {
    let place = chunk::File {
        declared: file.declared,
        index: file.index,
        length: file.length,
    };
    match chunk::answer(&kernel.ask(&chunk::request(place, offset, length, previous_end))) {
        chunk::Verdict::Accept => Ok(offset + length),
        refused => Err(ImportError::SpanRefused(tensor, refused)),
    }
}

/// A model's graph file, imported as it arrives.
pub struct Graph {
    taking: Taking,
    stream: Stream,
    nodes: Vec<Node>,
    tensors: Vec<(
        hologram_ai_onnx_preflight::stream::Tensor,
        TensorAt,
        u64,
        u64,
    )>,
    previous_end: u64,
    external: Vec<(hologram_ai_onnx_preflight::stream::Tensor, TensorAt)>,
    records: usize,
    encoder: bool,
}

impl Graph {
    /// An import of the graph file `file`, from its first byte.
    #[must_use]
    pub fn new(file: File) -> Self {
        Self {
            taking: Taking::new(file),
            stream: Stream::new(),
            nodes: Vec::new(),
            tensors: Vec::new(),
            previous_end: 0,
            external: Vec::new(),
            records: 0,
            encoder: false,
        }
    }

    /// An import of the graph file `file` of an encoder, whose operators are
    /// asked of decision 18's encoder style (KN-52). The caller knows the
    /// task; the graph alone does not say it, since an encoder's 4-bit
    /// weights name the fused domain too.
    #[must_use]
    pub fn encoder(file: File) -> Self {
        Self {
            encoder: true,
            ..Self::new(file)
        }
    }

    /// Take the next bytes of the file, in order from its first.
    ///
    /// # Errors
    ///
    /// [`ImportError::Unreadable`] when the graph is not one the reader
    /// follows; [`ImportError::SpanRefused`] when the kernel refuses a
    /// tensor's span.
    pub fn take(&mut self, chunk: &[u8], kernel: &mut dyn Kernel) -> Result<(), ImportError> {
        // The reader is fed only from the byte it asks for: the data it steps
        // over is hashed here and not handed to it.
        let at = self.taking.at;
        let wanted = self.stream.position().max(at);
        let end = at + chunk.len() as u64;
        let events = if wanted < end {
            // Inside this chunk, so below its length.
            let from = (wanted - at) as usize;
            self.stream
                .feed(&chunk[from..])
                .map_err(ImportError::Unreadable)?
        } else {
            Vec::new()
        };
        for event in &events {
            if let Event::Data { offset, length } = event {
                self.taking.watch(*offset, *length);
            }
        }
        self.taking.take(chunk);
        for event in events {
            match event {
                Event::Node(node) => self.nodes.push(node),
                Event::Tensor(tensor) => {
                    let place = TensorAt(place(self.records));
                    self.records += 1;
                    match tensor.data {
                        Data::Inline { offset, length } => {
                            let file = self.taking.file.clone();
                            self.previous_end =
                                span(kernel, &file, place, offset, length, self.previous_end)?;
                            self.tensors.push((tensor, place, offset, length));
                        }
                        Data::External { .. } => self.external.push((tensor, place)),
                        Data::Typed | Data::None => {}
                    }
                }
                Event::Opset { .. } | Event::Data { .. } => {}
            }
        }
        Ok(())
    }

    /// The file has ended. The graph's tensors with their data inside it, and
    /// what still has to be read from the data files beside it.
    ///
    /// # Errors
    ///
    /// [`ImportError::Unreadable`] when the file ended early;
    /// [`ImportError::FileRefused`], [`ImportError::OperatorRefused`] and
    /// [`ImportError::WeightRefused`] when the kernel refuses; and
    /// [`ImportError::DataMissing`] when a tensor's data never passed.
    pub fn finish(self, kernel: &mut dyn Kernel) -> Result<Imported, ImportError> {
        self.stream
            .finish(self.taking.file.length)
            .map_err(ImportError::Unreadable)?;
        self.taking.finish(kernel)?;

        let fused = self
            .nodes
            .iter()
            .any(|node| node.domain == FUSED_DOMAIN && node.op_type != EITHER_STYLE);
        let style = if self.encoder {
            opname::Style::Encoder
        } else if fused {
            opname::Style::Fused
        } else {
            opname::Style::Decomposed
        };
        let mut operators: Vec<String> = Vec::new();
        let mut seen = BTreeSet::new();
        for node in &self.nodes {
            if seen.insert(node.op_type.clone()) {
                operators.push(node.op_type.clone());
            }
        }
        for (index, operator) in operators.iter().enumerate() {
            // A name longer than any listed one is asked as the empty name,
            // which no list holds, so the kernel refuses it in its own words.
            let name = opname::name(operator).unwrap_or([0; 32]);
            match opname::answer(&kernel.ask(&opname::request(style, &name))) {
                opname::Verdict::Supported => {}
                refused => {
                    return Err(ImportError::OperatorRefused(
                        OperatorAt(place(index)),
                        refused,
                    ))
                }
            }
        }

        let mut tensors = Vec::with_capacity(self.tensors.len());
        for (tensor, place, offset, length) in self.tensors {
            let address = *self
                .taking
                .done
                .get(&offset)
                .ok_or(ImportError::DataMissing(place))?;
            tensors.push(Tensor {
                name: tensor.name,
                dims: tensor.dims,
                data_type: tensor.data_type,
                file: 0,
                offset,
                length,
                address,
                at: place,
            });
        }
        Ok(Imported {
            model: Model {
                tensors,
                operators,
                fused,
            },
            nodes: self.nodes,
            external: self.external,
        })
    }
}

/// A graph read to its end: the tensors inside it, and those whose data is
/// still to be read from the files beside it.
pub struct Imported {
    model: Model,
    nodes: Vec<Node>,
    external: Vec<(hologram_ai_onnx_preflight::stream::Tensor, TensorAt)>,
}

impl Imported {
    /// The graph's nodes in file order: each one's name, operator, the
    /// values it reads and its attributes. What a weight is to a model is
    /// read from the node that reads it.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The data files the graph's tensors lie in, by name, in the order the
    /// graph first names them.
    #[must_use]
    pub fn data_files(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (tensor, _) in &self.external {
            if let Data::External { location, .. } = &tensor.data {
                if !out.contains(location) {
                    out.push(location.clone());
                }
            }
        }
        out
    }

    /// Start reading the data file `file`, whose name the graph's tensors give
    /// as their location.
    ///
    /// # Errors
    ///
    /// [`ImportError::DataMissing`] when a tensor in it names no length, or
    /// [`ImportError::SpanRefused`] when the kernel refuses its span.
    pub fn data(&self, file: File, kernel: &mut dyn Kernel) -> Result<DataFile, ImportError> {
        let mut spans: Vec<(TensorAt, u64, u64)> = Vec::new();
        for (tensor, place) in &self.external {
            if let Data::External {
                location,
                offset,
                length,
            } = &tensor.data
            {
                if *location == file.name {
                    let length = length.ok_or(ImportError::DataMissing(*place))?;
                    spans.push((*place, *offset, length));
                }
            }
        }
        spans.sort_by_key(|(_, offset, _)| *offset);
        let mut taking = Taking::new(file);
        let mut previous_end = 0;
        let mut places = BTreeMap::new();
        for (place, offset, length) in &spans {
            previous_end = span(kernel, &taking.file, *place, *offset, *length, previous_end)?;
            taking.watch(*offset, *length);
            places.insert(*offset, *place);
        }
        Ok(DataFile { taking, places })
    }

    /// The model, once every data file has been read: each external tensor
    /// with the address its file gave it.
    ///
    /// # Errors
    ///
    /// [`ImportError::DataMissing`] when a tensor lies in a file not read,
    /// or past the end of the one it names; [`ImportError::WeightRefused`]
    /// when the kernel refuses a 4-bit weight's length.
    pub fn finish(
        self,
        files: Vec<ReadFile>,
        kernel: &mut dyn Kernel,
    ) -> Result<Model, ImportError> {
        let mut model = self.model;
        for (tensor, place) in self.external {
            let Data::External {
                location,
                offset,
                length,
            } = tensor.data
            else {
                continue;
            };
            let read = files
                .iter()
                .find(|file| file.name == location)
                .ok_or(ImportError::DataMissing(place))?;
            let address = *read
                .done
                .get(&offset)
                .ok_or(ImportError::DataMissing(place))?;
            model.tensors.push(Tensor {
                name: tensor.name,
                dims: tensor.dims,
                data_type: tensor.data_type,
                file: read.index,
                offset,
                length: length.unwrap_or(0),
                address,
                at: place,
            });
        }
        weights(&self.nodes, &model.tensors, kernel)?;
        Ok(model)
    }
}

/// A data file being read, every span already accepted by the kernel.
pub struct DataFile {
    taking: Taking,
    places: BTreeMap<u64, TensorAt>,
}

impl DataFile {
    /// Take the next bytes of the file, in order from its first.
    pub fn take(&mut self, chunk: &[u8]) {
        self.taking.take(chunk);
    }

    /// The file has ended.
    ///
    /// # Errors
    ///
    /// [`ImportError::FileRefused`] when the kernel refuses the whole file,
    /// and [`ImportError::DataMissing`] when it ended before a span did.
    pub fn finish(self, kernel: &mut dyn Kernel) -> Result<ReadFile, ImportError> {
        self.taking.finish(kernel)?;
        if let Some(open) = self.taking.open.first() {
            // Every open span was watched under its tensor's place.
            let place = self
                .places
                .get(&open.offset)
                .copied()
                .unwrap_or(TensorAt(u32::MAX));
            return Err(ImportError::DataMissing(place));
        }
        Ok(ReadFile {
            name: self.taking.file.name,
            index: self.taking.file.index,
            done: self.taking.done,
        })
    }
}

/// A data file read to its end: the address of each span in it.
pub struct ReadFile {
    name: String,
    index: u8,
    done: BTreeMap<u64, Label>,
}

/// Put each 4-bit weight's codes and scales to the kernel against its shape.
fn weights(nodes: &[Node], tensors: &[Tensor], kernel: &mut dyn Kernel) -> Result<(), ImportError> {
    let by_name: BTreeMap<&str, &Tensor> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    for node in nodes.iter().filter(|node| node.op_type == "MatMulNBits") {
        let int = |key: &str| {
            node.ints
                .iter()
                .find(|(name, _)| name == key)
                .and_then(|(_, value)| u64::try_from(*value).ok())
        };
        let (Some(columns), Some(rows), Some(block)) = (int("K"), int("N"), int("block_size"))
        else {
            continue;
        };
        // Codes are the node's second input and scales its third.
        for (input, codes) in [(1usize, true), (2, false)] {
            let Some(tensor) = node
                .inputs
                .get(input)
                .and_then(|name| by_name.get(name.as_str()))
            else {
                continue;
            };
            let part = match (codes, tensor.data_type) {
                (true, _) => nbits::Part::Codes,
                (false, 10) => nbits::Part::ScalesF16,
                (false, _) => nbits::Part::ScalesF32,
            };
            // A dimension past the widths the kernel reads is asked as zero,
            // which it refuses as a bad shape.
            let narrow = |value: u64| u32::try_from(value).unwrap_or(0);
            let question = nbits::request(
                part,
                narrow(rows),
                narrow(columns),
                u16::try_from(block).unwrap_or(0),
            );
            let answer = nbits::answer(&kernel.ask(&question));
            if answer != nbits::Answer::Bytes(tensor.length) {
                return Err(ImportError::WeightRefused(tensor.at, tensor.length, answer));
            }
        }
    }
    Ok(())
}

/// A model with no data files: the graph is the whole of it.
///
/// # Errors
///
/// As [`Imported::finish`].
pub fn whole(imported: Imported, kernel: &mut dyn Kernel) -> Result<Model, ImportError> {
    imported.finish(Vec::new(), kernel)
}
