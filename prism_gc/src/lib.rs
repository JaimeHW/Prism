//! Prism Garbage Collector
//!
//! A high-performance generational garbage collector optimized for Python/JIT workloads.
//!
//! # Architecture
//!
//! The GC uses a generational design with three spaces:
//!
//! - **Nursery (Young Generation)**: Bump-pointer allocation with copying collection.
//!   Most objects die here, so collection is fast (<1ms typically).
//!
//! - **Tenured (Old Generation)**: Mark-sweep with optional compaction.
//!   Objects that survive multiple nursery collections are promoted here.
//!
//! - **Large Object Space**: Objects larger than `LARGE_OBJECT_THRESHOLD` are
//!   allocated directly here to avoid copying overhead.
//!
//! # Write Barriers
//!
//! The GC uses card-table write barriers to track old→young references.
//! This allows minor collections to scan only dirty cards instead of the
//! entire old generation.
//!
//! # Usage
//!
//! ```ignore
//! use prism_gc::{GcHeap, GcRef, Trace};
//!
//! let mut heap = GcHeap::new(GcConfig::default());
//!
//! // Allocate an object
//! let obj = heap.alloc(MyObject::new());
//!
//! // Trigger collection if needed
//! heap.collect_if_needed();
//! ```
//!
//! # Safety
//!
//! The GC requires that:
//! - All GC-managed types implement the `Trace` trait
//! - All object references are properly traced during collection
//! - Write barriers are executed after pointer stores to old-gen objects

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod barrier;
pub mod collector;
pub mod config;
pub mod heap;
pub mod roots;
pub mod trace;

mod alloc;
mod stats;

// Re-exports for convenient access
pub use alloc::GcRef;
pub use config::GcConfig;
pub use heap::GcHeap;
pub use roots::{GcHandle, HandleScope};
pub use stats::GcStats;
pub use trace::{NoopObjectTracer, ObjectTracer, Trace, Tracer};

/// GC color for tri-color marking algorithm.
///
/// The tri-color invariant states that no black object may point
/// directly to a white object. This is maintained by:
/// - Marking objects gray when discovered
/// - Marking objects black after all children are processed
/// - Write barriers that re-gray objects when necessary
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcColor {
    /// Not yet visited (potentially dead).
    White = 0,
    /// In the work queue (reachable, children not yet scanned).
    Gray = 1,
    /// Fully scanned (reachable, all children processed).
    Black = 2,
}

/// Generation identifier for generational collection.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Generation {
    /// Young generation (nursery) - bump allocation, copying collection.
    Nursery = 0,
    /// Survivor space - objects that survived one minor GC.
    Survivor = 1,
    /// Old generation (tenured) - mark-sweep collection.
    Tenured = 2,
    /// Large object space - direct allocation, mark-sweep.
    LargeObject = 3,
}

impl Generation {
    /// Check if this generation is in the young space.
    #[inline]
    pub fn is_young(self) -> bool {
        matches!(self, Generation::Nursery | Generation::Survivor)
    }

    /// Check if this generation is in the old space.
    #[inline]
    pub fn is_old(self) -> bool {
        matches!(self, Generation::Tenured | Generation::LargeObject)
    }
}
