//! Generator opcode handlers and VM integration.
//!
//! This module provides the opcode handlers for Python generator operations,
//! including yield, yield from, send, throw, and close. It also manages the
//! generator execution context within the virtual machine.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         Generator Operations Module                          │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │  ┌────────────── Opcode Handlers ──────────────┐                            │
//! │  │                                              │                            │
//! │  │  yield_value    ─────▶  Suspend + yield     │                            │
//! │  │  yield_from     ─────▶  Delegate to sub-gen │                            │
//! │  │  send_value     ─────▶  Resume with value   │                            │
//! │  │  throw_into     ─────▶  Inject exception    │                            │
//! │  │  close_generator ────▶  Cleanup + close     │                            │
//! │  └──────────────────────────────────────────────┘                            │
//! │                         │                                                    │
//! │                         ▼                                                    │
//! │  ┌────────────── Frame Management ─────────────┐                            │
//! │  │                                              │                            │
//! │  │  GeneratorFramePool   ─── Arena allocation   │                            │
//! │  │  suspend.rs           ─── Register capture   │                            │
//! │  │  resume.rs            ─── State restoration  │                            │
//! │  └──────────────────────────────────────────────┘                            │
//! │                         │                                                    │
//! │                         ▼                                                    │
//! │  ┌────────────── Dispatch Caching ─────────────┐                            │
//! │  │                                              │                            │
//! │  │  ResumeTableCache     ─── Per-code tables    │                            │
//! │  │  InlineResumeCache    ─── 4-entry inline     │                            │
//! │  │  YieldPointMap        ─── PC → resume index  │                            │
//! │  └──────────────────────────────────────────────┘                            │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | Time Complexity | Space Overhead |
//! |-----------|-----------------|----------------|
//! | Yield | O(R) where R = live registers | 0 heap alloc (pooled) |
//! | Resume | O(R) register restore | 0 heap alloc |
//! | Lookup | O(1) inline cache hit | 32 bytes inline |
//! | Fallback | O(log N) table lookup | HashMap overhead |
//!
//! # Module Structure
//!
//! - `mod.rs` - Module exports, integration tests
//! - `opcodes.rs` - Opcode handlers (yield, yield_from, send, throw, close)
//! - `frame_pool.rs` - GeneratorFramePool for frame suspension/resumption
//! - `resume_cache.rs` - ResumeTableCache for computed-goto dispatch
//! - `suspend.rs` - Suspension logic (capture_frame, save_registers)
//! - `resume.rs` - Resumption logic (restore_frame, dispatch_to_yield_point)
//! - `context.rs` - GeneratorContext for VM integration
//!
//! # Usage
//!
//! The generator opcodes integrate with the VM dispatch loop:
//!
//! ```text
//! VM run_loop()
//!     │
//!     ├── Opcode::Yield
//!     │       └── yield_value() → ControlFlow::Yield { value, resume_point }
//!     │
//!     ├── Opcode::Send
//!     │       └── send_value() → ControlFlow::Resume { send_value }
//!     │
//!     └── ControlFlow handling
//!             └── VM suspends/resumes generator state
//! ```

pub mod context;
pub mod frame_pool;
pub mod opcodes;
pub mod resume;
pub mod resume_cache;
pub mod suspend;

// Re-exports for convenient access
pub use context::{GeneratorContext, GeneratorExecutionState};
pub use frame_pool::{GeneratorFramePool, PoolStats, PooledFrame};
pub use opcodes::{
    close_generator, get_yield_value, send_value, throw_into, yield_from, yield_value,
};
pub use resume::{ResumeDispatcher, dispatch_to_resume_point, restore_generator_frame};
pub use resume_cache::{InlineResumeCache, ResumeTableCache, YieldPointEntry};
pub use suspend::{SuspendResult, capture_generator_frame, save_live_registers};

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of yield points per code object.
/// Beyond this, we use a HashMap fallback instead of a flat table.
pub const MAX_INLINE_YIELD_POINTS: usize = 256;

/// Size of the inline resume cache (number of entries).
/// Optimized for common case of few hot generators.
pub const INLINE_CACHE_SIZE: usize = 4;

/// Initial capacity for the frame pool.
pub const INITIAL_POOL_CAPACITY: usize = 16;

/// Maximum frames to keep in pool (prevents unbounded growth).
pub const MAX_POOL_SIZE: usize = 64;
