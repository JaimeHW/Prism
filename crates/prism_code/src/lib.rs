//! Shared bytecode and code object representation for Prism.
//!
//! This crate defines the executable bytecode format that is shared across the
//! compiler, VM, runtime, JIT, and AOT layers.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod builder;
pub mod code_object;
pub mod instruction;

pub use builder::{FunctionBuilder, KwNamesTuple, Label};
pub use code_object::{
    CodeFlags, CodeObject, Constant, ExceptionEntry, LineTableEntry, disassemble,
};
pub use instruction::{
    CLASS_META_DYNAMIC_BASES_FLAG, CLASS_META_DYNAMIC_KEYWORDS_FLAG, ConstIndex, Instruction,
    InstructionFormat, LocalSlot, Opcode, Register,
};
