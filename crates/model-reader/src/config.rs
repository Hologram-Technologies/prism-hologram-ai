//! The model's config: what kind of model it is.
//!
//! A checkpoint on a hub carries a `config.json` beside its weights. Three of
//! the four things the kernel's preflight question asks about are counted from
//! it: whether the architecture is one the registry names, how many of the keys
//! a decoder must carry are missing, and whether it asks for something the
//! recipe does not implement.
//!
//! The fourth, whether the tensor manifest fits the family's recipe, is a
//! property of the weights rather than of this document, and is the caller's
//! input to [`Reading::question`].
//!
//! Nothing here decides whether the model may run. The counts go to the kernel,
//! which answers.

use hologram_ai_kernel_client::preflight;
use serde_json::Value;

/// The architectures the registry names, taken from the family registry of
/// `Hologram-Technologies/hologram-ai`.
///
/// A name outside this list is not a rejection. The kernel answers `derived`
/// for a model whose config is otherwise whole: an unnamed decoder runs when
/// its manifest fits the generic recipe.
pub const NAMED_FAMILIES: [&str; 5] = [
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
];

/// The config keys a decoder must carry, each a positive whole number.
///
/// Keys with a library default, such as `rms_norm_eps` and `rope_theta`, are
/// absent from this list on purpose: a config that omits them is a whole model,
/// not a model missing something.
pub const REQUIRED_KEYS: [&str; 6] = [
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "num_attention_heads",
    "num_hidden_layers",
    "vocab_size",
];

/// The gate activations the recipe implements. A checkpoint that gates its MLP
/// differently computes something else, so it is flagged rather than run as if
/// it were one of these.
const IMPLEMENTED_ACTIVATIONS: [&str; 2] = ["silu", "swish"];

/// What the config says, counted as the kernel's preflight question asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reading {
    /// The first architecture named is one of [`NAMED_FAMILIES`].
    pub family_recognised: bool,
    /// How many of [`REQUIRED_KEYS`] are absent or are not a positive whole
    /// number.
    pub missing_keys: u8,
    /// How many features the config asks for that the recipe does not
    /// implement.
    pub unsupported_features: u8,
}

/// Why a config could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// `FL-07`
    NotJson,
}

impl Reading {
    /// The kernel's preflight question about this model.
    ///
    /// `manifest_fits` is read from the weights, not from the config: it says
    /// whether the tensors the checkpoint carries are the ones the family's
    /// recipe builds from.
    #[must_use]
    pub fn question(self, manifest_fits: bool) -> [u8; 5] {
        preflight::request(
            self.family_recognised,
            self.missing_keys,
            manifest_fits,
            self.unsupported_features,
        )
    }
}

/// Count what `bytes`, a model's `config.json`, says.
pub fn read(bytes: &[u8]) -> Result<Reading, ConfigError> {
    let config = match serde_json::from_slice(bytes) {
        Ok(Value::Object(config)) => config,
        _ => return Err(ConfigError::NotJson),
    };

    let family_recognised = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|names| names.first())
        .and_then(Value::as_str)
        .is_some_and(|name| NAMED_FAMILIES.contains(&name));

    let missing_keys = REQUIRED_KEYS
        .iter()
        .filter(|key| {
            !config
                .get(**key)
                .and_then(Value::as_u64)
                .is_some_and(|value| value > 0)
        })
        .count();

    // A value that is present and is not one of the activations the recipe
    // implements is flagged, whatever it is. A config that omits the key relies
    // on the family default, which the recipe does implement.
    let activation = config.get("hidden_act").filter(|value| !value.is_null());
    let unsupported_features = usize::from(activation.is_some_and(|value| {
        !value
            .as_str()
            .is_some_and(|name| IMPLEMENTED_ACTIVATIONS.contains(&name))
    }));

    Ok(Reading {
        family_recognised,
        // Both counts are bounded by lists this crate declares: six keys and
        // one feature, far inside what the question's single byte holds.
        missing_keys: u8::try_from(missing_keys).unwrap_or(u8::MAX),
        unsupported_features: u8::try_from(unsupported_features).unwrap_or(u8::MAX),
    })
}
