//! Opcode handler modules.
//!
//! Organized by category for maintainability. Each handler takes
//! a reference to the VM and an instruction, returning a ControlFlow.

pub mod arithmetic;
pub mod attribute;
pub mod calls;
pub mod class;
pub mod comparison;
pub mod containers;
pub mod context;
pub mod control;
pub mod coroutine;
pub mod dict_access;
pub mod exception;
pub mod generator;
pub mod iteration;
pub mod kw_binding;
pub mod load_store;
pub mod r#match;
pub mod method_dispatch;
pub mod objects;
pub mod protocols;
pub mod set_access;
pub mod subscript;
pub mod unpack;

// Re-export common types for convenience
pub use super::dispatch::ControlFlow;
