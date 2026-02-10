//! Python `re` module implementation.
//!
//! High-performance regular expression matching with:
//! - O(m*n) guaranteed linear time complexity (via Rust `regex` crate)
//! - Optional backreference/lookaround support (via `fancy-regex`)
//! - LRU pattern cache for repeated compilations
//! - Full Python `re` API compatibility
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                     Public API                          │
//! │  match, search, findall, finditer, sub, split, compile  │
//! └─────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │                   Pattern Cache                         │
//! │             LRU(256) with FxHash lookup                │
//! └─────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │                   Regex Engine                          │
//! │     StandardEngine (regex) │ FancyEngine (fancy-regex)  │
//! └─────────────────────────────────────────────────────────┘
//! ```

mod cache;
mod engine;
mod flags;
mod functions;
mod match_obj;
mod pattern;

#[cfg(test)]
mod tests;

pub use cache::PatternCache;
pub use engine::{Engine, EngineKind};
pub use flags::RegexFlags;
pub use functions::*;
pub use match_obj::Match;
pub use pattern::CompiledPattern;

use super::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use std::sync::Arc;

// =============================================================================
// Module Constants
// =============================================================================

/// Maximum size of pattern cache.
pub const CACHE_SIZE: usize = 256;

/// Default flags (none set).
pub const DEFAULT_FLAGS: u32 = 0;

// =============================================================================
// Re Module
// =============================================================================

/// The `re` module implementation.
#[derive(Debug, Clone)]
pub struct ReModule {
    /// Cached attribute names.
    attrs: Vec<Arc<str>>,
}

impl ReModule {
    /// Create a new re module instance.
    pub fn new() -> Self {
        let attrs = vec![
            // Functions
            Arc::from("compile"),
            Arc::from("match"),
            Arc::from("search"),
            Arc::from("findall"),
            Arc::from("finditer"),
            Arc::from("sub"),
            Arc::from("subn"),
            Arc::from("split"),
            Arc::from("fullmatch"),
            Arc::from("escape"),
            Arc::from("purge"),
            // Flags
            Arc::from("IGNORECASE"),
            Arc::from("I"),
            Arc::from("MULTILINE"),
            Arc::from("M"),
            Arc::from("DOTALL"),
            Arc::from("S"),
            Arc::from("VERBOSE"),
            Arc::from("X"),
            Arc::from("ASCII"),
            Arc::from("A"),
            Arc::from("UNICODE"),
            Arc::from("U"),
            Arc::from("LOCALE"),
            Arc::from("L"),
            // Types
            Arc::from("Pattern"),
            Arc::from("Match"),
            // Error
            Arc::from("error"),
        ];

        Self { attrs }
    }
}

impl Default for ReModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReModule {
    fn name(&self) -> &str {
        "re"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            // Functions - return error until callable system is ready
            "compile" | "match" | "search" | "findall" | "finditer" | "sub" | "subn" | "split"
            | "fullmatch" | "escape" | "purge" => Err(ModuleError::AttributeError(format!(
                "re.{} is not yet callable as an object",
                name
            ))),

            // Flag constants
            "IGNORECASE" | "I" => Ok(Value::int_unchecked(RegexFlags::IGNORECASE as i64)),
            "MULTILINE" | "M" => Ok(Value::int_unchecked(RegexFlags::MULTILINE as i64)),
            "DOTALL" | "S" => Ok(Value::int_unchecked(RegexFlags::DOTALL as i64)),
            "VERBOSE" | "X" => Ok(Value::int_unchecked(RegexFlags::VERBOSE as i64)),
            "ASCII" | "A" => Ok(Value::int_unchecked(RegexFlags::ASCII as i64)),
            "UNICODE" | "U" => Ok(Value::int_unchecked(RegexFlags::UNICODE as i64)),
            "LOCALE" | "L" => Ok(Value::int_unchecked(RegexFlags::LOCALE as i64)),

            // Types/Error
            "Pattern" | "Match" | "error" => Err(ModuleError::AttributeError(format!(
                "re.{} is not yet accessible (type support pending)",
                name
            ))),

            _ => Err(ModuleError::AttributeError(format!(
                "module 're' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}
