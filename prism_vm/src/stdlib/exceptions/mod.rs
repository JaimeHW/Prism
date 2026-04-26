//! Python exception system.
//!
//! This module provides a high-performance exception handling infrastructure
//! for the Prism Python runtime. It is designed for zero-cost exception handling
//! when exceptions are not thrown, with fast paths for common cases.
//!
//! # Architecture
//!
//! The exception system is built on several key components:
//!
//! - **ExceptionTypeId**: u8-packed type identifier for O(1) JIT comparisons
//! - **ExceptionFlags**: Compact bitfield tracking exception state
//! - **ExceptionTypeSet**: u64 bitset for O(1) subclass checking
//! - **HandlerTable**: Binary-searchable table of exception handlers
//! - **Flyweight Pool**: Pre-allocated singletons for control-flow exceptions
//! - **ExceptionObject**: Full exception with lazy args/traceback
//!
//! # Performance Design
//!
//! ## Zero-Cost Happy Path
//!
//! When exceptions are not thrown, there is no runtime overhead. Exception
//! handlers are recorded in compile-time metadata tables, not runtime stacks.
//!
//! ## Fast Type Checking
//!
//! Exception type checking uses u8 comparisons and u64 bitsets:
//! - `except TypeError:` compiles to `cmp al, 52; je handler`
//! - `except (TypeError, ValueError):` uses bitset intersection
//!
//! ## Lazy Allocation
//!
//! Exception arguments and tracebacks are allocated lazily:
//! - Flyweight exceptions for StopIteration, GeneratorExit require zero allocation
//! - Full traceback is only computed when accessed
//!
//! # Modules
//!
//! - [`types`]: Exception type identifiers (ExceptionTypeId enum)
//! - [`flags`]: Exception state flags (ExceptionFlags bitfield)
//! - [`hierarchy`]: Exception inheritance support (ExceptionTypeSet)
//! - [`traceback`]: Traceback and frame information
//! - [`object`]: Exception object implementation
//! - [`flyweight`]: Pre-allocated singleton exceptions
//! - [`table`]: Handler table for zero-cost exception handling

pub mod flags;
pub mod flyweight;
pub mod hierarchy;
pub mod object;
pub mod table;
pub mod traceback;
pub mod types;

// Re-export commonly used items
pub use flags::ExceptionFlags;
pub use flyweight::{FlyweightPool, raise_generator_exit, raise_stop_iteration};
pub use hierarchy::{ExceptionTypeSet, common_ancestor, descendants, is_subclass};
pub use object::{ExceptionArgs, ExceptionObject, ExceptionRef};
pub use table::{CATCH_ALL, HandlerEntry, HandlerFlags, HandlerTable, HandlerTableBuilder};
pub use traceback::{FrameInfo, TracebackObject};
pub use types::ExceptionTypeId;
