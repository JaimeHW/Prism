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
mod python_api;

pub use cache::PatternCache;
pub use engine::{Engine, EngineKind};
pub use flags::RegexFlags;
pub use functions::*;
pub use match_obj::Match;
pub use pattern::CompiledPattern;
pub use python_api::{
    RegexMatchObject, RegexPatternObject, builtin_compile, builtin_escape, builtin_findall,
    builtin_finditer, builtin_fullmatch, builtin_match, builtin_match_end, builtin_match_getitem,
    builtin_match_group, builtin_match_groupdict, builtin_match_groups, builtin_match_span,
    builtin_match_start, builtin_pattern_findall, builtin_pattern_finditer,
    builtin_pattern_fullmatch, builtin_pattern_match, builtin_pattern_search,
    builtin_pattern_split, builtin_pattern_sub, builtin_pattern_subn, builtin_purge,
    builtin_search, builtin_split, builtin_sub, builtin_subn, match_attr_value, pattern_attr_value,
};

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::BuiltinFunctionObject;
use crate::builtins::VALUE_ERROR;
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use std::sync::{Arc, LazyLock};

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
            Arc::from("TEMPLATE"),
            Arc::from("T"),
            Arc::from("MULTILINE"),
            Arc::from("M"),
            Arc::from("DOTALL"),
            Arc::from("S"),
            Arc::from("VERBOSE"),
            Arc::from("X"),
            Arc::from("DEBUG"),
            Arc::from("ASCII"),
            Arc::from("A"),
            Arc::from("UNICODE"),
            Arc::from("U"),
            Arc::from("LOCALE"),
            Arc::from("L"),
            Arc::from("NOFLAG"),
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

static COMPILE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.compile"), builtin_compile));
static MATCH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.match"), builtin_match));
static SEARCH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.search"), builtin_search));
static FULLMATCH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.fullmatch"), builtin_fullmatch));
static FINDALL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.findall"), builtin_findall));
static FINDITER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.finditer"), builtin_finditer));
static SUB_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.sub"), builtin_sub));
static SUBN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.subn"), builtin_subn));
static SPLIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.split"), builtin_split));
static ESCAPE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("re.escape"), builtin_escape));
static PURGE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("re.purge"), builtin_purge));

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

impl Module for ReModule {
    fn name(&self) -> &str {
        "re"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "compile" => Ok(builtin_value(&COMPILE_FUNCTION)),
            "match" => Ok(builtin_value(&MATCH_FUNCTION)),
            "search" => Ok(builtin_value(&SEARCH_FUNCTION)),
            "fullmatch" => Ok(builtin_value(&FULLMATCH_FUNCTION)),
            "findall" => Ok(builtin_value(&FINDALL_FUNCTION)),
            "finditer" => Ok(builtin_value(&FINDITER_FUNCTION)),
            "sub" => Ok(builtin_value(&SUB_FUNCTION)),
            "subn" => Ok(builtin_value(&SUBN_FUNCTION)),
            "split" => Ok(builtin_value(&SPLIT_FUNCTION)),
            "escape" => Ok(builtin_value(&ESCAPE_FUNCTION)),
            "purge" => Ok(builtin_value(&PURGE_FUNCTION)),

            // Flag constants
            "NOFLAG" => Ok(Value::int_unchecked(0)),
            "IGNORECASE" | "I" => Ok(Value::int_unchecked(RegexFlags::IGNORECASE as i64)),
            "TEMPLATE" | "T" => Ok(Value::int_unchecked(RegexFlags::TEMPLATE as i64)),
            "MULTILINE" | "M" => Ok(Value::int_unchecked(RegexFlags::MULTILINE as i64)),
            "DOTALL" | "S" => Ok(Value::int_unchecked(RegexFlags::DOTALL as i64)),
            "VERBOSE" | "X" => Ok(Value::int_unchecked(RegexFlags::VERBOSE as i64)),
            "DEBUG" => Ok(Value::int_unchecked(RegexFlags::DEBUG as i64)),
            "ASCII" | "A" => Ok(Value::int_unchecked(RegexFlags::ASCII as i64)),
            "UNICODE" | "U" => Ok(Value::int_unchecked(RegexFlags::UNICODE as i64)),
            "LOCALE" | "L" => Ok(Value::int_unchecked(RegexFlags::LOCALE as i64)),

            "Pattern" => Ok(crate::builtins::builtin_type_object_for_type_id(
                TypeId::REGEX_PATTERN,
            )),
            "Match" => Ok(crate::builtins::builtin_type_object_for_type_id(
                TypeId::REGEX_MATCH,
            )),
            "error" => Ok(Value::object_ptr(
                &*VALUE_ERROR as *const crate::builtins::ExceptionTypeObject as *const (),
            )),

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
