//! # Prism Core
//!
//! Core types, traits, and primitives for the Prism Python runtime.
//!
//! This crate provides the foundational building blocks shared across all Prism components:
//!
//! - **Value System**: Tagged union representation of Python values with NaN-boxing
//! - **Object Model**: GC-managed object references and type descriptors
//! - **Interning**: String and identifier interning for O(1) equality checks
//! - **Error Handling**: Result types and error definitions
//! - **Memory**: Arena allocators and allocation primitives

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod aot;
pub mod error;
pub mod intern;
pub mod small_int_cache;
pub mod span;
pub mod speculation;
pub mod value;

pub use aot::{
    AOT_IMPORT_FROM_SYMBOL, AOT_IMPORT_MODULE_SYMBOL, AOT_NATIVE_INIT_TABLE_END_SYMBOL,
    AOT_NATIVE_INIT_TABLE_START_SYMBOL, AOT_STORE_EXPR_SYMBOL, AotImmediate, AotImmediateKind,
    AotImportBinding, AotImportFromOp, AotImportModuleOp, AotNativeModuleInitEntry, AotOpStatus,
    AotOperand, AotOperandKind, AotStoreExprKind, AotStoreExprOp, AotStringRef,
};
pub use error::{PrismError, PrismResult};
pub use intern::{InternedString, StringInterner};
pub use small_int_cache::SmallIntCache;
pub use span::Span;
pub use speculation::{SpeculationProvider, TypeHint};
pub use value::{INT_TAG_PATTERN, STRING_TAG_PATTERN, TYPE_TAG_MASK, VALUE_PAYLOAD_MASK, Value};

/// Prism runtime version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Python language version this runtime targets.
pub const PYTHON_VERSION: (u8, u8, u8) = (3, 12, 0);
