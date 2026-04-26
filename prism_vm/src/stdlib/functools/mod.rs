//! Python `functools` module implementation.
//!
//! Higher-order functions and operations on callable objects.
//!
//! # Components
//!
//! | Function | Description | Module |
//! |----------|-------------|--------|
//! | `reduce` | Cumulative reduction | `reduce.rs` |
//! | `partial` | Partial function application | `partial.rs` |
//! | `lru_cache` | LRU memoization cache | `lru_cache.rs` |
//! | `cmp_to_key` | Comparison → key function adapter | `cmp.rs` |
//! | `total_ordering` | Derive rich comparisons | `cmp.rs` |
//! | `wraps` / `update_wrapper` | Copy function metadata | `wraps.rs` |
//! | `cached_property` | Lazy-computed cached descriptor | `cached_property.rs` |
//!
//! # Performance Highlights
//!
//! - **`reduce`**: Single-pass O(n), zero intermediate allocations
//! - **`partial`**: SmallVec<[Value; 8]> inline storage for ≤8 frozen args
//! - **`lru_cache`**: Arena-backed intrusive linked list for O(1) operations
//! - **`cmp_to_key`**: Zero-cost Ord wrapper via generics
//! - **`cached_property`**: O(1) cached access after first computation

pub mod cached_property;
pub mod cmp;
pub mod lru_cache;
pub mod partial;
pub mod reduce;
pub mod wraps;

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::calls::value_supports_call_protocol;
use crate::python_numeric::int_like_value;
use std::sync::Arc;
use std::sync::LazyLock;

use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::tuple::TupleObject;

// Re-export core types
pub use cached_property::CachedProperty;
pub use cmp::{CmpKey, ComparisonBase, TotalOrdering};
pub use lru_cache::{CacheInfo, LruCache};
pub use partial::Partial;
pub use reduce::{ReduceError, accumulate, reduce, reduce_fallible};
pub use wraps::{WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES, WrapperMetadata};

static WRAPS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("functools.wraps"), builtin_wraps));
static UPDATE_WRAPPER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("functools.update_wrapper"),
        builtin_update_wrapper,
    )
});
static LRU_CACHE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("functools.lru_cache"), builtin_lru_cache)
});
static CACHE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("functools.cache"), builtin_cache));
static WRAPS_DECORATOR: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("functools._identity_decorator"),
        builtin_identity_decorator,
    )
});
static WRAPPER_ASSIGNMENTS_VALUE: LazyLock<TupleObject> = LazyLock::new(|| {
    TupleObject::from_iter(
        WRAPPER_ASSIGNMENTS
            .iter()
            .map(|name| Value::string(intern(name))),
    )
});
static WRAPPER_UPDATES_VALUE: LazyLock<TupleObject> = LazyLock::new(|| {
    TupleObject::from_iter(
        WRAPPER_UPDATES
            .iter()
            .map(|name| Value::string(intern(name))),
    )
});

// =============================================================================
// Functools Module
// =============================================================================

/// The functools module implementation.
pub struct FunctoolsModule {
    attrs: Vec<Arc<str>>,
}

impl FunctoolsModule {
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("reduce"),
                Arc::from("partial"),
                Arc::from("partialmethod"),
                Arc::from("lru_cache"),
                Arc::from("cache"),
                Arc::from("cached_property"),
                Arc::from("cmp_to_key"),
                Arc::from("total_ordering"),
                Arc::from("update_wrapper"),
                Arc::from("wraps"),
                Arc::from("WRAPPER_ASSIGNMENTS"),
                Arc::from("WRAPPER_UPDATES"),
                Arc::from("singledispatch"),
                Arc::from("singledispatchmethod"),
            ],
        }
    }
}

impl Default for FunctoolsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for FunctoolsModule {
    fn name(&self) -> &str {
        "functools"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "wraps" => Ok(builtin_value(&WRAPS_FUNCTION)),
            "update_wrapper" => Ok(builtin_value(&UPDATE_WRAPPER_FUNCTION)),
            "lru_cache" => Ok(builtin_value(&LRU_CACHE_FUNCTION)),
            "cache" => Ok(builtin_value(&CACHE_FUNCTION)),
            "partial" => Ok(crate::stdlib::_functools::partial_class_value()),
            "WRAPPER_ASSIGNMENTS" => Ok(tuple_value(&WRAPPER_ASSIGNMENTS_VALUE)),
            "WRAPPER_UPDATES" => Ok(tuple_value(&WRAPPER_UPDATES_VALUE)),
            "reduce"
            | "partialmethod"
            | "cached_property"
            | "cmp_to_key"
            | "total_ordering"
            | "singledispatch"
            | "singledispatchmethod" => Err(ModuleError::AttributeError(format!(
                "functools.{} is not yet available as a callable object",
                name
            ))),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'functools' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Create a new functools module instance.
pub fn create_module() -> FunctoolsModule {
    FunctoolsModule::new()
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn tuple_value(tuple: &'static TupleObject) -> Value {
    Value::object_ptr(tuple as *const TupleObject as *const ())
}

fn builtin_wraps(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "wraps() takes from 1 to 3 arguments ({} given)",
            args.len()
        )));
    }

    Ok(builtin_value(&WRAPS_DECORATOR))
}

fn builtin_update_wrapper(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "update_wrapper() takes from 2 to 4 arguments ({} given)",
            args.len()
        )));
    }

    Ok(args[0])
}

fn builtin_lru_cache(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "lru_cache() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }

    let mut maxsize = args.first().copied();
    let mut typed = args.get(1).copied();
    for &(name, value) in keywords {
        match name {
            "maxsize" => {
                if maxsize.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "lru_cache() got multiple values for argument 'maxsize'".to_string(),
                    ));
                }
            }
            "typed" => {
                if typed.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "lru_cache() got multiple values for argument 'typed'".to_string(),
                    ));
                }
            }
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "lru_cache() got an unexpected keyword argument '{}'",
                    name
                )));
            }
        }
    }

    if let Some(candidate) = maxsize
        && value_supports_call_protocol(candidate)
        && typed.is_none()
    {
        return Ok(candidate);
    }

    if let Some(value) = maxsize
        && !value.is_none()
        && int_like_value(value).is_none()
    {
        return Err(BuiltinError::TypeError(
            "Expected first argument to be an integer, a callable, or None".to_string(),
        ));
    }

    Ok(builtin_value(&WRAPS_DECORATOR))
}

fn builtin_cache(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "cache() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(args[0])
}

fn builtin_identity_decorator(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "decorator() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    Ok(args[0])
}
