//! Compatibility re-exports for Prism's shared bytecode representation.

/// Compatibility re-exports for bytecode construction helpers.
pub mod builder {
    pub use prism_code::builder::*;
}

/// Compatibility re-exports for code-object metadata types.
pub mod code_object {
    pub use prism_code::code_object::*;
}

/// Compatibility re-exports for instruction encoding types.
pub mod instruction {
    pub use prism_code::instruction::*;
}

pub use prism_code::{
    CLASS_META_DYNAMIC_BASES_FLAG, CLASS_META_DYNAMIC_KEYWORDS_FLAG, CodeFlags, CodeObject,
    ConstIndex, Constant, ExceptionEntry, FunctionBuilder, Instruction, InstructionFormat,
    KwNamesTuple, Label, LineTableEntry, LocalSlot, Opcode, Register, disassemble,
};
