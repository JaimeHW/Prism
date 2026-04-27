//! High-performance virtual machine for Prism with tiered execution.
//!
//! This crate provides the bytecode interpreter (Tier 0) for the Prism
//! Python runtime. It executes register-based bytecode with:
//!
//! - **Dispatch table**: Static function pointer table for O(1) opcode dispatch
//! - **Stack registers**: 256 registers per frame, inline in stack (2KB L1 fit)
//! - **Inline caching**: Monomorphic/polymorphic caches for attribute access
//! - **Profiling**: Call counts and type feedback for JIT tier-up decisions
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                 VirtualMachine                   │
//! ├─────────────────────────────────────────────────┤
//! │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
//! │  │ Frame 0 │  │ Frame 1 │  │ Frame N (curr)  │  │
//! │  │ 256 reg │→ │ 256 reg │→ │ 256 registers   │  │
//! │  └─────────┘  └─────────┘  └─────────────────┘  │
//! │                                                  │
//! │  ┌──────────────┐  ┌────────────────────────┐   │
//! │  │ GlobalScope  │  │ BuiltinRegistry        │   │
//! │  │ (FxHashMap)  │  │ (print, len, range...) │   │
//! │  └──────────────┘  └────────────────────────┘   │
//! │                                                  │
//! │  ┌──────────────┐  ┌────────────────────────┐   │
//! │  │ InlineCache  │  │ Profiler               │   │
//! │  │ (attr, call) │  │ (call counts, types)   │   │
//! │  └──────────────┘  └────────────────────────┘   │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use prism_vm::VirtualMachine;
//! use prism_compiler::compile;
//!
//! let code = compile("1 + 2")?;
//! let mut vm = VirtualMachine::new();
//! let result = vm.execute(code)?;
//! assert_eq!(result.as_int(), Some(3));
//! ```

#![allow(clippy::new_without_default)]

// Core modules
mod allocator;
mod aot_abi;
mod error;
mod finalizers;
mod frame;
mod gc_integration;
mod globals;
mod vm;

// Public facade modules
pub mod exceptions;
pub mod imports;

// Execution infrastructure
mod builtins;
mod dispatch;
mod ic_manager;
mod inline_cache;
mod osr_trigger;
mod profiler;
mod python_numeric;
mod source;
mod speculative;
mod threading_runtime;
mod truthiness;
mod type_feedback;

// JIT integration
mod compilation_queue;
mod deopt;
mod jit_bridge;
mod jit_context;
mod jit_dispatch;
mod jit_executor;
mod tier1_lowering;

// Opcode handlers (organized by category)
mod ops;

// Exception and generator runtime
mod exception;

// Standard library modules
mod stdlib;

// Import system
mod import;

// Re-exports
pub use error::{RuntimeError, RuntimeErrorKind, TracebackEntry, VmResult};
pub use exceptions::{ExceptionTypeId, RuntimeException, exception_type_id, runtime_exception};
pub use imports::{FrozenModuleSource, ModuleObject};
pub use jit_context::{JitConfig, JitStats};
pub(crate) use prism_runtime::allocation_context::alloc_value_in_current_heap_or_box as alloc_managed_value;
pub use source::SourceOptimization;
pub use vm::VirtualMachine;

/// Convenience function to create and run a VM.
pub fn run(
    code: std::sync::Arc<prism_code::CodeObject>,
) -> prism_core::PrismResult<prism_core::Value> {
    let mut vm = VirtualMachine::new();
    vm.execute(code)
}
