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

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::new_without_default)]

// Core modules
pub mod allocator;
pub mod aot_abi;
pub mod error;
pub mod frame;
pub mod gc_integration;
pub mod globals;
pub mod vm;

// Execution infrastructure
pub mod builtins;
pub mod dispatch;
pub mod ic_manager;
pub mod inline_cache;
pub mod osr_trigger;
pub mod profiler;
pub mod speculative;
pub mod truthiness;
pub mod type_feedback;

// JIT integration
pub mod compilation_queue;
pub mod deopt;
pub mod jit_bridge;
pub mod jit_context;
pub mod jit_dispatch;
pub mod jit_executor;
mod tier1_lowering;

// Opcode handlers (organized by category)
pub mod ops;

// Exception and generator runtime
pub mod exception;

// Standard library modules
pub mod stdlib;

// Import system
pub mod import;

// Re-exports
pub use allocator::{AllocResult, GcAllocator, HeapAllocExt};
pub use builtins::{BuiltinError, BuiltinFn, BuiltinRegistry};
pub use dispatch::ControlFlow;
pub use error::{RuntimeError, RuntimeErrorKind, VmResult};
pub use frame::{ClosureEnv, Frame, MAX_RECURSION_DEPTH};
pub use gc_integration::{ManagedHeap, RootProvider, SafePoint, StackRoots};
pub use globals::GlobalScope;
pub use ic_manager::{ICAccessResult, ICClassification, ICEntry, ICManager, ICSiteId, ICStats};
pub use inline_cache::{CallIC, InlineCacheStore, MonoIC, PolyIC};
pub use jit_bridge::{BridgeConfig, JitBridge};
pub use jit_context::{JitConfig, JitContext, JitStats, ProcessedResult};
pub use jit_dispatch::{DispatchResult, DispatchStats};
pub use jit_executor::{DeoptReason, ExecutionResult, JitExecutor};
pub use osr_trigger::{LoopInfo, OsrDecision, OsrTrigger};
pub use profiler::{CodeId, Profiler, TierUpDecision};
pub use vm::VirtualMachine;

/// Convenience function to create and run a VM.
pub fn run(
    code: std::sync::Arc<prism_compiler::bytecode::CodeObject>,
) -> prism_core::PrismResult<prism_core::Value> {
    let mut vm = VirtualMachine::new();
    vm.execute(code)
}
