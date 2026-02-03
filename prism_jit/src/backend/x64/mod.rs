//! x64 backend modules.
//!
//! This module provides complete x64 code generation infrastructure:
//! - `registers`: GPR/XMM register definitions and calling conventions
//! - `memory`: Executable memory allocation and management
//! - `encoder`: Low-level instruction encoding
//! - `assembler`: High-level code emission with labels
//! - `cpuid`: CPU feature detection for runtime optimization

pub mod assembler;
pub mod cpuid;
pub mod encoder;
pub mod memory;
pub mod registers;

// Re-export commonly used types
pub use assembler::{
    Assembler, ConstantPool, ConstantPoolEntry, Label, Relocation, RelocationType,
};
pub use cpuid::{CpuFeatureFlags, CpuFeatures, CpuLevel, CpuVendor};
pub use encoder::{Condition, EncodedInst, Mod, Rex};
pub use memory::{CodeCacheStats, CompiledCode, ExecutableBuffer, PAGE_SIZE};
pub use registers::{
    AllocatableRegs, CallingConvention, Gpr, GprSet, MemOperand, Scale, Xmm, XmmSet,
};
