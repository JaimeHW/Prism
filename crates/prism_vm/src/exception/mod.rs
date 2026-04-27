//! Exception runtime infrastructure for the Prism VM.
//!
//! This module provides the runtime components for exception handling,
//! complementing the data structures in `stdlib::exceptions`.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Exception System                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌───────────────────┐   ┌────────────────────────────────────┐│
//! │  │  stdlib/exceptions │   │        exception/ (this module)    ││
//! │  │  ─────────────────  │   │  ──────────────────────────────────││
//! │  │  • ExceptionObject  │   │  • ExceptionRef (flyweight)        ││
//! │  │  • ExceptionTypeId  │   │  • InlineHandlerCache              ││
//! │  │  • HandlerTable     │   │  • HandlerStack (runtime)          ││
//! │  │  • TracebackObject  │   │  • ExceptionState (FSM)            ││
//! │  │  • Flyweight pool   │   │  • Unwind logic                    ││
//! │  └───────────────────┘   └────────────────────────────────────┘│
//! │           │                           │                         │
//! │           └─────────── Used by ───────┘                         │
//! │                           ↓                                     │
//! │  ┌─────────────────────────────────────────────────────────────┐│
//! │  │                    VM Integration                            ││
//! │  │  • ControlFlow::Exception                                    ││
//! │  │  • ops/exception.rs opcodes                                  ││
//! │  │  • Frame exception state                                     ││
//! │  └─────────────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Targets
//!
//! | Metric | Target |
//! |--------|--------|
//! | try block overhead | 0 cycles (table-driven) |
//! | StopIteration raise | 8-12 cycles (flyweight + cache) |
//! | TypeError raise | 40-60 cycles (lazy traceback) |
//! | Handler lookup (cached) | 3-5 cycles |
//! | Handler lookup (miss) | O(N) linear scan |
//!
//! # Module Organization
//!
//! - [`exception_ref`]: Flyweight `ExceptionRef` for zero-copy propagation
//! - [`handler_cache`]: `InlineHandlerCache` for O(1) repeated lookups
//! - [`handler_stack`]: Runtime handler stack for nested try blocks
//! - [`unwind`]: Stack unwinding logic
//! - [`exc_info_stack`]: CPython 3.11+ exception info stack

mod exc_info_stack;
mod exception_ref;
mod handler_cache;
mod handler_stack;
mod state;
mod unwind;

// Re-exports
pub use exc_info_stack::{EntryFlags, ExcInfoEntry, ExcInfoStack, ExcInfoStackStats};
pub use exception_ref::{FlyweightExceptionRef, OwnedExceptionRef};
pub use handler_cache::{
    HandlerCacheStats, InlineHandlerCache, MultiLevelCache, NO_CACHED_HANDLER, NO_CACHED_PC,
};
pub use handler_stack::{
    HandlerFrame, HandlerSearchResult, HandlerStack, HandlerStackStats, NO_FRAME, NO_HANDLER,
};
pub use state::{
    ExceptionContext, ExceptionContextFlags, ExceptionState, ExceptionStateMachine, ExceptionStats,
    TransitionResult,
};
pub use unwind::{FinallyEntry, UnwindAction, UnwindInfo, UnwindResult, Unwinder, UnwinderStats};
