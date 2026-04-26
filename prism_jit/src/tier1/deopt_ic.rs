//! Deoptimization IC Integration
//!
//! Handles IC updates when deoptimization occurs due to an IC miss.
//! When the JIT deopts due to a cache miss, we capture the observed shape
//! and slot information, then update the IC so future executions hit the fast path.
//!
//! # Architecture
//!
//! ```text
//! JIT Code → IC Miss → Deopt Stub → DeoptHandler
//!                                       ↓
//!                                  IcDeoptContext
//!                                       ↓
//!                                  handle_ic_deopt()
//!                                       ↓
//!                                  IC Updated
//! ```
//!
//! # Performance
//!
//! - `IcDeoptContext` is 16 bytes, cache-line aligned
//! - Lock-free IC state transitions via atomic operations
//! - Statistics tracked with relaxed ordering (no sync overhead)

use crate::ic::{
    IcKind, IcManager, IcState, ShapeVersion, ic_call_miss, ic_get_property_miss,
    ic_set_property_miss,
};
use prism_runtime::object::shape::{PropertyFlags, ShapeId};
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// IC Deopt Context
// =============================================================================

/// Context for an IC miss deoptimization.
///
/// Captured when JIT code deopts due to an IC guard failure.
/// Contains all information needed to update the IC cache.
///
/// # Layout (16 bytes, cache-aligned)
///
/// ```text
/// Offset  Size  Field
/// 0       4     ic_site_idx
/// 4       4     observed_shape (raw u32)
/// 8       2     slot_offset
/// 10      1     flags (property flags)
/// 11      1     ic_kind
/// 12      1     is_inline
/// 13      3     padding
/// ```
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct IcDeoptContext {
    /// Index of the IC site in the IcManager.
    pub ic_site_idx: u32,

    /// Shape ID that was actually observed (caused the miss).
    pub observed_shape: ShapeId,

    /// Resolved slot offset for the property.
    pub slot_offset: u16,

    /// Property flags (writable, enumerable, etc.).
    pub flags: PropertyFlags,

    /// Kind of IC operation (GetAttr, SetAttr, Call).
    pub kind: IcKind,

    /// True if property is in inline storage, false if dictionary.
    pub is_inline: bool,

    /// Padding for alignment.
    _padding: [u8; 3],
}

impl IcDeoptContext {
    /// Create a new IC deopt context for a property operation.
    #[inline]
    pub fn new_property(
        ic_site_idx: u32,
        observed_shape: ShapeId,
        slot_offset: u16,
        flags: PropertyFlags,
        kind: IcKind,
        is_inline: bool,
    ) -> Self {
        debug_assert!(
            kind == IcKind::GetProperty || kind == IcKind::SetProperty,
            "new_property requires GetAttr or SetAttr kind"
        );
        Self {
            ic_site_idx,
            observed_shape,
            slot_offset,
            flags,
            kind,
            is_inline,
            _padding: [0; 3],
        }
    }

    /// Create a new IC deopt context for a call operation.
    #[inline]
    pub fn new_call(ic_site_idx: u32, observed_shape: ShapeId) -> Self {
        Self {
            ic_site_idx,
            observed_shape,
            slot_offset: 0,
            flags: PropertyFlags::empty(),
            kind: IcKind::Call,
            is_inline: false,
            _padding: [0; 3],
        }
    }

    /// Create a context with default values (for testing).
    #[inline]
    pub const fn empty() -> Self {
        Self {
            ic_site_idx: 0,
            observed_shape: ShapeId::EMPTY,
            slot_offset: 0,
            flags: PropertyFlags::empty(),
            kind: IcKind::GetProperty,
            is_inline: true,
            _padding: [0; 3],
        }
    }

    /// Check if this context is for a property operation.
    #[inline]
    pub fn is_property_op(&self) -> bool {
        self.kind.is_property_op()
    }

    /// Check if this context is for a call operation.
    #[inline]
    pub fn is_call_op(&self) -> bool {
        self.kind == IcKind::Call
    }
}

impl Default for IcDeoptContext {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// IC Deopt Update Result
// =============================================================================

/// Result of handling an IC deopt update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcUpdateResult {
    /// IC was successfully updated to a new state.
    Updated {
        /// Previous IC state before update.
        old_state: IcState,
        /// New IC state after update.
        new_state: IcState,
    },

    /// IC site was not found (invalid index).
    SiteNotFound,

    /// IC is already megamorphic, no update needed.
    AlreadyMegamorphic,

    /// IC kind mismatch (GetAttr context for Call IC, etc.).
    KindMismatch,
}

impl IcUpdateResult {
    /// Check if the update was successful.
    #[inline]
    pub fn is_success(&self) -> bool {
        matches!(self, IcUpdateResult::Updated { .. })
    }

    /// Check if a state transition occurred.
    #[inline]
    pub fn transitioned(&self) -> bool {
        match self {
            IcUpdateResult::Updated {
                old_state,
                new_state,
            } => old_state != new_state,
            _ => false,
        }
    }
}

// =============================================================================
// IC Deopt Statistics
// =============================================================================

/// Statistics for IC deopt handling.
///
/// Tracks deopt counts by IC kind and transition type for profiling.
#[derive(Debug, Default)]
pub struct IcDeoptStats {
    /// Total IC deopts handled.
    total_deopts: AtomicU64,

    /// GetAttr IC deopts.
    get_attr_deopts: AtomicU64,

    /// SetAttr IC deopts.
    set_attr_deopts: AtomicU64,

    /// Call IC deopts.
    call_deopts: AtomicU64,

    /// Deopts that caused state transitions.
    transitions: AtomicU64,

    /// Deopts to already-megamorphic ICs.
    already_mega: AtomicU64,

    /// Invalid site index deopts.
    invalid_sites: AtomicU64,
}

impl IcDeoptStats {
    /// Create new empty statistics.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a deopt event.
    fn record(&self, kind: IcKind, result: &IcUpdateResult) {
        self.total_deopts.fetch_add(1, Ordering::Relaxed);

        match kind {
            IcKind::GetProperty => self.get_attr_deopts.fetch_add(1, Ordering::Relaxed),
            IcKind::SetProperty => self.set_attr_deopts.fetch_add(1, Ordering::Relaxed),
            IcKind::Call => self.call_deopts.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };

        match result {
            IcUpdateResult::Updated {
                old_state,
                new_state,
            } if old_state != new_state => {
                self.transitions.fetch_add(1, Ordering::Relaxed);
            }
            IcUpdateResult::AlreadyMegamorphic => {
                self.already_mega.fetch_add(1, Ordering::Relaxed);
            }
            IcUpdateResult::SiteNotFound => {
                self.invalid_sites.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Get total deopt count.
    #[inline]
    pub fn total(&self) -> u64 {
        self.total_deopts.load(Ordering::Relaxed)
    }

    /// Get GetAttr deopt count.
    #[inline]
    pub fn get_attr_count(&self) -> u64 {
        self.get_attr_deopts.load(Ordering::Relaxed)
    }

    /// Get SetAttr deopt count.
    #[inline]
    pub fn set_attr_count(&self) -> u64 {
        self.set_attr_deopts.load(Ordering::Relaxed)
    }

    /// Get Call deopt count.
    #[inline]
    pub fn call_count(&self) -> u64 {
        self.call_deopts.load(Ordering::Relaxed)
    }

    /// Get state transition count.
    #[inline]
    pub fn transition_count(&self) -> u64 {
        self.transitions.load(Ordering::Relaxed)
    }

    /// Get already-megamorphic count.
    #[inline]
    pub fn already_mega_count(&self) -> u64 {
        self.already_mega.load(Ordering::Relaxed)
    }

    /// Get invalid site count.
    #[inline]
    pub fn invalid_site_count(&self) -> u64 {
        self.invalid_sites.load(Ordering::Relaxed)
    }

    /// Get a snapshot of all statistics.
    pub fn snapshot(&self) -> IcDeoptStatsSnapshot {
        IcDeoptStatsSnapshot {
            total: self.total(),
            get_attr: self.get_attr_count(),
            set_attr: self.set_attr_count(),
            call: self.call_count(),
            transitions: self.transition_count(),
            already_mega: self.already_mega_count(),
            invalid_sites: self.invalid_site_count(),
        }
    }
}

/// A point-in-time snapshot of IC deopt statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct IcDeoptStatsSnapshot {
    /// Total IC deopts handled.
    pub total: u64,
    /// GetAttr IC deopts.
    pub get_attr: u64,
    /// SetAttr IC deopts.
    pub set_attr: u64,
    /// Call IC deopts.
    pub call: u64,
    /// State transitions.
    pub transitions: u64,
    /// Already megamorphic.
    pub already_mega: u64,
    /// Invalid site indices.
    pub invalid_sites: u64,
}

// =============================================================================
// IC Deopt Handler
// =============================================================================

/// Handle an IC deoptimization by updating the IC cache.
///
/// This is the main entry point called from the deopt handler when
/// an IC miss causes deoptimization.
///
/// # Arguments
///
/// * `ic_manager` - The IC manager containing the IC sites
/// * `context` - The deopt context with observed shape/slot info
/// * `stats` - Optional statistics tracker
///
/// # Returns
///
/// Result indicating whether the IC was successfully updated.
#[inline(never)] // Keep out of hot path - this is the slow path
pub fn handle_ic_deopt(
    ic_manager: &mut IcManager,
    context: &IcDeoptContext,
    stats: Option<&IcDeoptStats>,
) -> IcUpdateResult {
    let result = handle_ic_deopt_inner(ic_manager, context);

    // Record statistics if provided
    if let Some(s) = stats {
        s.record(context.kind, &result);
    }

    result
}

/// Inner IC deopt handler (no statistics).
fn handle_ic_deopt_inner(ic_manager: &mut IcManager, context: &IcDeoptContext) -> IcUpdateResult {
    // Look up the IC site
    let site = match ic_manager.get_mut(context.ic_site_idx) {
        Some(s) => s,
        None => return IcUpdateResult::SiteNotFound,
    };

    // Handle based on IC kind
    match context.kind {
        IcKind::GetProperty | IcKind::SetProperty => {
            let property_ic = match site.property_ic_mut() {
                Some(ic) => ic,
                None => return IcUpdateResult::KindMismatch,
            };

            // Check if already megamorphic using PropertyIc's state
            let old_state = property_ic.state();
            if old_state == IcState::Megamorphic {
                return IcUpdateResult::AlreadyMegamorphic;
            }

            // Call the appropriate miss handler
            if context.kind == IcKind::GetProperty {
                ic_get_property_miss(
                    property_ic,
                    context.observed_shape,
                    context.slot_offset,
                    context.flags,
                );
            } else {
                ic_set_property_miss(
                    property_ic,
                    context.observed_shape,
                    context.slot_offset,
                    context.flags,
                );
            }

            let new_state = property_ic.state();
            IcUpdateResult::Updated {
                old_state,
                new_state,
            }
        }

        IcKind::Call => {
            // Check site header state first, before mutable borrow
            let old_state = site.header.state();
            if old_state == IcState::Megamorphic {
                return IcUpdateResult::AlreadyMegamorphic;
            }

            let call_ic = match site.call_ic_mut() {
                Some(ic) => ic,
                None => return IcUpdateResult::KindMismatch,
            };

            // For call IC, we update with minimal info from context
            // Full call resolution happens in the runtime
            ic_call_miss(
                call_ic,
                context.observed_shape.raw() as u64, // callee_id
                0,                                   // code_ptr (resolved later)
                0,                                   // argc (resolved later)
            );

            // Call IC state is internal, just report a transition occurred
            IcUpdateResult::Updated {
                old_state,
                new_state: IcState::Monomorphic, // Approximate - actual state is internal
            }
        }

        _ => IcUpdateResult::KindMismatch,
    }
}

/// Batch handle multiple IC deopts.
///
/// Useful when multiple IC sites need updating from a single deopt event.
#[inline]
pub fn handle_ic_deopts_batch(
    ic_manager: &mut IcManager,
    contexts: &[IcDeoptContext],
    stats: Option<&IcDeoptStats>,
) -> Vec<IcUpdateResult> {
    contexts
        .iter()
        .map(|ctx| handle_ic_deopt(ic_manager, ctx, stats))
        .collect()
}

// =============================================================================
// Deopt IC Context Builder
// =============================================================================

/// Builder for creating `IcDeoptContext` instances.
///
/// Provides a fluent API for constructing deopt contexts.
#[derive(Debug)]
pub struct IcDeoptContextBuilder {
    ic_site_idx: u32,
    observed_shape: ShapeId,
    slot_offset: u16,
    flags: PropertyFlags,
    kind: IcKind,
    is_inline: bool,
}

impl Default for IcDeoptContextBuilder {
    fn default() -> Self {
        Self {
            ic_site_idx: 0,
            observed_shape: ShapeId::EMPTY,
            slot_offset: 0,
            flags: PropertyFlags::empty(),
            kind: IcKind::GetProperty,
            is_inline: true,
        }
    }
}

impl IcDeoptContextBuilder {
    /// Create a new builder.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the IC site index.
    #[inline]
    pub fn ic_site(mut self, idx: u32) -> Self {
        self.ic_site_idx = idx;
        self
    }

    /// Set the observed shape.
    #[inline]
    pub fn shape(mut self, shape: ShapeId) -> Self {
        self.observed_shape = shape;
        self
    }

    /// Set the slot offset.
    #[inline]
    pub fn offset(mut self, offset: u16) -> Self {
        self.slot_offset = offset;
        self
    }

    /// Set the property flags.
    #[inline]
    pub fn flags(mut self, flags: PropertyFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Set the IC kind.
    #[inline]
    pub fn kind(mut self, kind: IcKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set whether the property is inline.
    #[inline]
    pub fn inline(mut self, is_inline: bool) -> Self {
        self.is_inline = is_inline;
        self
    }

    /// Build the context.
    #[inline]
    pub fn build(self) -> IcDeoptContext {
        IcDeoptContext {
            ic_site_idx: self.ic_site_idx,
            observed_shape: self.observed_shape,
            slot_offset: self.slot_offset,
            flags: self.flags,
            kind: self.kind,
            is_inline: self.is_inline,
            _padding: [0; 3],
        }
    }
}
