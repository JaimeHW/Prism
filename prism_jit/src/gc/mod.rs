//! GC integration for JIT-compiled code.
//!
//! This module provides the infrastructure needed for precise garbage
//! collection during JIT execution:
//!
//! - **Stack maps**: Record which stack slots/registers hold pointers at safepoints
//! - **JIT roots**: Walk JIT frames to find live references
//! - **Write barriers**: Track cross-generation references for generational GC
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    StackMapRegistry                         │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
//! │  │  StackMap A  │  │  StackMap B  │  │  StackMap C  │ ...  │
//! │  │  (fn addr)   │  │  (fn addr)   │  │  (fn addr)   │      │
//! │  └──────────────┘  └──────────────┘  └──────────────┘      │
//! │         │                 │                 │               │
//! │         ▼                 ▼                 ▼               │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              SafePoints (sorted by offset)          │   │
//! │  │  [offset: 0x10, regs: 0b0011, stack: 0b00001111]   │   │
//! │  │  [offset: 0x20, regs: 0b0001, stack: 0b00000011]   │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//!
//!                         GC Safepoint
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  1. Get return address from JIT frame                       │
//! │  2. Binary search StackMapRegistry by code range            │
//! │  3. Binary search SafePoints by code offset                 │
//! │  4. Use bitmaps to identify pointer slots/registers         │
//! │  5. Trace each live reference                               │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! - O(log n) lookup by return address via sorted code ranges
//! - Compact bitmap encoding: 64 stack slots + 16 registers per safepoint
//! - Lock-free read path for concurrent GC
//! - Cache-line aligned data structures

pub mod jit_roots;
pub mod stackmap;
pub mod write_barrier;

pub use jit_roots::{JitFrameWalker, JitRoots};
pub use stackmap::{SafePoint, StackMap, StackMapBuilder, StackMapRegistry};
pub use write_barrier::{CARD_SHIFT, CARD_SIZE, CardTable, WriteBarrier};
