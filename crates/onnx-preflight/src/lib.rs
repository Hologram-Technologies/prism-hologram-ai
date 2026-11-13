//! Reads an ONNX model's opset and operators, and asks the kernel whether its
//! graph may be imported.
//!
//! An ONNX file is a protocol-buffer message. A preflight needs four things
//! from it: the opset the model declares for the default domain, the operator
//! each node of its graph names, the domain each node belongs to, and how many
//! tensors the graph carries. Everything else is stepped over, so a file
//! written against a newer schema still reads.
//!
//! Nothing here decides whether a graph may be built. Which operators a builder
//! implements is the builder's to say; this crate counts the ones outside that
//! set and puts the kernel's question. The kernel answers.

#![forbid(unsafe_code)]
// R5: this crate fails only as `model/failures.toml` lists, and never by
// `unwrap`, `expect` or `panic!`. `audit-limits` checks these stay; `just lint`
// enforces them.
#![forbid(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeSet;

use hologram_ai_kernel_client::operators;

mod wire;

use wire::{text, Encoding, Reader};

/// The default operator domain: the empty name, which the ONNX operators
/// themselves belong to. A node in any other domain is somebody's extension.
pub const DEFAULT_DOMAIN: &str = "";

/// Field numbers, from `onnx.proto`. A number read from the wrong field is a
/// mistake nothing else would catch, so each is named where it is used.
mod field {
    pub(crate) const MODEL_GRAPH: u64 = 7;
    pub(crate) const MODEL_OPSET: u64 = 8;
    pub(crate) const OPSET_DOMAIN: u64 = 1;
    pub(crate) const OPSET_VERSION: u64 = 2;
    pub(crate) const GRAPH_NODE: u64 = 1;
    pub(crate) const GRAPH_INITIALIZER: u64 = 5;
    pub(crate) const NODE_OP_TYPE: u64 = 4;
    pub(crate) const NODE_DOMAIN: u64 = 7;
}

/// One node's operator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operator {
    /// The domain it belongs to; [`DEFAULT_DOMAIN`] for the ONNX operators.
    pub domain: String,
    /// Its name, such as `MatMul`.
    pub op_type: String,
}

/// What a preflight reads from an ONNX file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    /// The opset version the model declares for the default domain.
    pub opset: u64,
    /// The operator of each node, in the order the file has them.
    pub operators: Vec<Operator>,
    /// How many tensors the graph carries with it.
    pub initializers: u64,
}

/// Why an ONNX model could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnnxError {
    /// `FL-08`
    Truncated,
    /// `FL-09`
    NumberTooLong,
    /// `FL-10`
    UnknownEncoding,
    /// `FL-11`
    NameNotText,
    /// `FL-12`
    NoGraph,
    /// `FL-13`
    NoDefaultOpset,
}

impl Model {
    /// The kernel's question about this model, for a builder that implements
    /// opsets `lowest` to `highest` and the operators `implemented` names.
    ///
    /// An operator counts as implemented when it belongs to the default domain
    /// and `implemented` names it. A node in another domain is somebody's
    /// extension, and a builder that was not told about it does not build it.
    #[must_use]
    pub fn question(&self, implemented: &[&str], lowest: u64, highest: u64) -> [u8; 33] {
        let unimplemented = self
            .operators
            .iter()
            .filter(|operator| {
                operator.domain != DEFAULT_DOMAIN
                    || !implemented.contains(&operator.op_type.as_str())
            })
            .count();
        operators::request(self.opset, lowest, highest, unimplemented as u64)
    }

    /// The operators this model names that `implemented` does not, each once,
    /// in the order they first appear. What to tell someone whose model was
    /// refused.
    #[must_use]
    pub fn missing(&self, implemented: &[&str]) -> Vec<Operator> {
        // Seen in a set, reported in a list: a graph with tens of thousands of
        // nodes is ordinary, and asking the list whether it already holds each
        // one would read the whole list once per node.
        let mut seen = BTreeSet::new();
        let mut out: Vec<Operator> = Vec::new();
        for operator in &self.operators {
            let known = operator.domain == DEFAULT_DOMAIN
                && implemented.contains(&operator.op_type.as_str());
            if !known && seen.insert((operator.domain.as_str(), operator.op_type.as_str())) {
                out.push(operator.clone());
            }
        }
        out
    }
}

/// Read `file`, an ONNX model.
pub fn read(file: &[u8]) -> Result<Model, OnnxError> {
    let mut graph = None;
    let mut opset = None;
    let mut model = Reader::new(file);
    while model.more() {
        match model.field()? {
            (field::MODEL_GRAPH, Encoding::Bytes) => graph = Some(model.delimited()?),
            (field::MODEL_OPSET, Encoding::Bytes) => {
                // The model declares one opset for each domain it uses. The
                // default domain's is the one the operators are versioned by;
                // a later declaration of the same domain replaces an earlier,
                // which is what reading them in order does.
                let declared = model.delimited()?;
                if let Some(version) = default_domain_opset(declared)? {
                    opset = Some(version);
                }
            }
            (_, encoding) => model.step_over(encoding)?,
        }
    }

    let graph = graph.ok_or(OnnxError::NoGraph)?;
    let opset = opset.ok_or(OnnxError::NoDefaultOpset)?;
    let (operators, initializers) = read_graph(graph)?;
    Ok(Model {
        opset,
        operators,
        initializers,
    })
}

/// The version in one opset declaration, when it is the default domain's.
fn default_domain_opset(declared: &[u8]) -> Result<Option<u64>, OnnxError> {
    // Both fields may be absent: a declaration that names no domain is the
    // default domain's, and one that names no version declares version zero,
    // which no opset is.
    let mut domain = String::new();
    let mut version = 0;
    let mut reader = Reader::new(declared);
    while reader.more() {
        match reader.field()? {
            (field::OPSET_DOMAIN, Encoding::Bytes) => domain = text(reader.delimited()?)?,
            (field::OPSET_VERSION, Encoding::Number) => version = reader.number()?,
            (_, encoding) => reader.step_over(encoding)?,
        }
    }
    Ok((domain == DEFAULT_DOMAIN).then_some(version))
}

/// The operators of a graph's nodes, and how many tensors it carries.
fn read_graph(graph: &[u8]) -> Result<(Vec<Operator>, u64), OnnxError> {
    let mut operators = Vec::new();
    let mut initializers = 0u64;
    let mut reader = Reader::new(graph);
    while reader.more() {
        match reader.field()? {
            (field::GRAPH_NODE, Encoding::Bytes) => operators.push(read_node(reader.delimited()?)?),
            (field::GRAPH_INITIALIZER, Encoding::Bytes) => {
                reader.delimited()?;
                initializers = initializers.saturating_add(1);
            }
            (_, encoding) => reader.step_over(encoding)?,
        }
    }
    Ok((operators, initializers))
}

/// One node's operator and domain.
fn read_node(node: &[u8]) -> Result<Operator, OnnxError> {
    // A node that names no operator reads as one whose name is empty, which no
    // builder implements and which therefore counts against the graph rather
    // than passing unnoticed.
    let mut operator = Operator {
        domain: String::new(),
        op_type: String::new(),
    };
    let mut reader = Reader::new(node);
    while reader.more() {
        match reader.field()? {
            (field::NODE_OP_TYPE, Encoding::Bytes) => operator.op_type = text(reader.delimited()?)?,
            (field::NODE_DOMAIN, Encoding::Bytes) => operator.domain = text(reader.delimited()?)?,
            (_, encoding) => reader.step_over(encoding)?,
        }
    }
    Ok(operator)
}
