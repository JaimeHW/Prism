//! Deoptimization Runtime Module.
//!
//! This module provides the infrastructure for bailout from JIT code back to
//! the interpreter. It uses return-address hijacking for zero-cost deopt on
//! the success path and lazy state capture for minimal overhead.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Guard Failure Path                              │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  1. Guard fails → hijack return address                                 │
//! │     ┌──────────────────┐                                                │
//! │     │ JIT Stack Frame  │                                                │
//! │     │ ┌──────────────┐ │                                                │
//! │     │ │ Return Addr  │─┼──▶ DeoptTrampoline[deopt_id]                   │
//! │     │ └──────────────┘ │                                                │
//! │     └──────────────────┘                                                │
//! │                                                                         │
//! │  2. Trampoline captures delta state (modified registers only)           │
//! │     DeoptState { bc_offset, reason, delta: [(slot, value), ...] }       │
//! │                                                                         │
//! │  3. Reconstruct full interpreter frame from delta                       │
//! │     Frame::registers[slot] = delta.get(slot) ?? jit_frame[slot]         │
//! │                                                                         │
//! │  4. Resume interpreter at bc_offset                                     │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

pub mod recovery;
pub mod state;
pub mod stats;
pub mod trampoline;

pub use recovery::{DeoptRecovery, RecoveryResult};
pub use state::{DeoptDelta, DeoptReason, DeoptState, MAX_DELTA_ENTRIES};
pub use stats::{DeoptSite, DeoptSiteKey, DeoptStats};
pub use trampoline::{DeoptTrampoline, DeoptTrampolineEntry, TRAMPOLINE_ENTRY_SIZE};
