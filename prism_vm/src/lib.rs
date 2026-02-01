//! Virtual machine for Prism with tiered execution.
#![deny(unsafe_op_in_unsafe_fn)]
pub mod frame;
pub mod interpreter;
pub use interpreter::Interpreter;
