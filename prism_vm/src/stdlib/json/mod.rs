//! Python `json` module implementation.
//!
//! Provides high-performance JSON encoding and decoding.
//! All functions are designed for efficiency with zero-copy parsing where possible.
//!
//! # Functions
//!
//! - `loads(s)` - Parse JSON string to Python object
//! - `dumps(obj)` - Serialize Python object to JSON string
//! - `load(fp)` - Parse JSON from file-like object (future)
//! - `dump(obj, fp)` - Serialize to file-like object (future)
//!
//! # Performance Characteristics
//!
//! | Function | Time Complexity | Notes |
//! |----------|-----------------|-------|
//! | `loads()` | O(n) | Zero-copy string parsing |
//! | `dumps()` | O(n) | Single allocation for output |

mod decode;
mod encode;

#[cfg(test)]
mod tests;

pub use decode::{JsonDecodeError, loads};
pub use encode::{JsonEncodeError, dumps};

use super::{Module, ModuleError, ModuleResult};
use std::sync::Arc;

// =============================================================================
// JSON Module
// =============================================================================

/// The `json` module implementation.
#[derive(Debug, Clone)]
pub struct JsonModule {
    /// Cached attribute names for fast lookup.
    attrs: Vec<Arc<str>>,
}

impl JsonModule {
    /// Create a new json module instance.
    pub fn new() -> Self {
        let attrs = vec![
            Arc::from("loads"),
            Arc::from("dumps"),
            Arc::from("load"),
            Arc::from("dump"),
            Arc::from("JSONDecodeError"),
            Arc::from("JSONEncoder"),
            Arc::from("JSONDecoder"),
        ];

        Self { attrs }
    }
}

impl Default for JsonModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for JsonModule {
    fn name(&self) -> &str {
        "json"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            // Core functions - return error until callable system is ready
            "loads" | "dumps" | "load" | "dump" => {
                // TODO: Return actual function objects when callable system is ready
                Err(ModuleError::AttributeError(format!(
                    "json.{} is not yet callable as an object",
                    name
                )))
            }

            // Classes
            "JSONDecodeError" | "JSONEncoder" | "JSONDecoder" => {
                // TODO: Return class objects when type system is ready
                Err(ModuleError::AttributeError(format!(
                    "json.{} is not yet accessible as a type",
                    name
                )))
            }

            _ => Err(ModuleError::AttributeError(format!(
                "module 'json' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}
