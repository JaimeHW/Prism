//! Register-based bytecode system.
//!
//! This module provides the bytecode representation for Prism's register-based VM.
//! Key components:
//!
//! - [`Instruction`] - 32-bit packed instruction format
//! - [`Opcode`] - Enumeration of all bytecode operations
//! - [`CodeObject`] - Compiled function representation
//! - [`FunctionBuilder`] - High-level API for bytecode construction

mod builder;
mod code_object;
mod instruction;

pub use builder::{FunctionBuilder, Label};
pub use code_object::{CodeFlags, CodeObject, ExceptionEntry, LineTableEntry, disassemble};
pub use instruction::{ConstIndex, Instruction, InstructionFormat, LocalSlot, Opcode, Register};
