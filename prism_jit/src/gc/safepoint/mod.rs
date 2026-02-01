//! GC Safepoint Infrastructure
//!
//! Page-protection based safepoints for zero-overhead GC triggering.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SafepointPage                            │
//! │  ┌──────────────────────────────────────────────────────┐  │
//! │  │  4KB page (mmap PROT_READ)                           │  │
//! │  │  Poll: test [r15], al (3 bytes, ~1 cycle)           │  │
//! │  │  Trap: mprotect PROT_NONE → SIGSEGV                 │  │
//! │  └──────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  SafepointCoordinator                                       │
//! │  - request_gc() → arm page, wait for all threads            │
//! │  - resume() → disarm page, wake threads                     │
//! │  - Uses futex/WaitOnAddress for optimal waiting             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation          | Latency       |
//! |--------------------|---------------|
//! | Poll (no GC)       | ~1 cycle      |
//! | Trap (GC needed)   | ~5-10μs       |
//! | Stop-the-world     | Σ(thread_wait)|

mod coordinator;
mod handler;
mod page;
mod poll;
mod stats;

pub use coordinator::{MutatorState, SafepointCoordinator, SafepointGuard};
pub use handler::{SafepointHandler, TrapContext, install_handler};
pub use page::{SafepointPage, SafepointState};
pub use poll::{SAFEPOINT_POLL_SIZE, emit_safepoint_poll, should_elide_safepoints};
pub use stats::SafepointStats;

/// Reserved register for safepoint page address (R15 on x64).
///
/// By dedicating a register, we save 4 bytes per poll instruction:
/// - With R15: `test [r15], al` = 3 bytes
/// - Without:  `test [rip+disp], al` = 7 bytes
///
/// The register is loaded once in the function prologue and remains
/// constant throughout execution.
pub const SAFEPOINT_REGISTER: u8 = 15; // R15

/// Size of the safepoint page region (64KB for guard zone).
pub const SAFEPOINT_REGION_SIZE: usize = 64 * 1024;

/// Size of the actual safepoint page (4KB).
pub const SAFEPOINT_PAGE_SIZE: usize = 4096;
