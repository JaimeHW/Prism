//! IC Runtime Handlers
//!
//! Runtime functions called on IC misses to update the cache and perform
//! the actual operation. These are the slow paths that handle cache updates
//! and state transitions.
//!
//! # Design
//!
//! Each handler:
//! 1. Performs the actual operation (property lookup, call dispatch)
//! 2. Updates the IC with the new type feedback
//! 3. Returns the result for the caller
//!
//! Handlers are designed to be called from JIT-compiled code with minimal
//! register spilling.

use super::{CallIc, IcState, PropertyIc, mega_ic::global_mega_cache, property_ic::SlotInfo};
use prism_runtime::object::shape::ShapeId;

// =============================================================================
// IC Miss Handler Trait
// =============================================================================

/// Trait for IC miss handlers.
///
/// Implemented by the runtime to provide actual property/call resolution.
pub trait IcMissHandler {
    /// Resolve a property get operation.
    ///
    /// Returns (slot_offset, flags, is_inline) if found.
    fn resolve_get_property(&self, object_shape: ShapeId, property_name: &str) -> Option<SlotInfo>;

    /// Resolve a property set operation.
    fn resolve_set_property(&self, object_shape: ShapeId, property_name: &str) -> Option<SlotInfo>;

    /// Resolve a function/method call.
    ///
    /// Returns (callee_id, code_ptr, argc).
    fn resolve_call(&self, callee_id: u64) -> Option<(u64, usize, u16)>;

    /// Resolve a method call on an object.
    fn resolve_method_call(
        &self,
        receiver_shape: ShapeId,
        method_name: &str,
    ) -> Option<(u64, usize, u16)>;
}

// =============================================================================
// IC Miss Functions
// =============================================================================

/// Handle a GetProperty IC miss.
///
/// Called when the IC cache doesn't have the expected shape.
/// Updates the IC and returns the slot info for the property.
#[inline(never)] // Keep out of hot path
pub fn ic_get_property_miss(
    ic: &mut PropertyIc,
    object_shape: ShapeId,
    slot_offset: u16,
    flags: prism_runtime::object::shape::PropertyFlags,
) -> SlotInfo {
    // Update the IC with new type feedback
    ic.update(object_shape, slot_offset, flags);

    // If megamorphic, also update global cache
    if ic.state() == IcState::Megamorphic {
        // We'd need bytecode offset here in real impl
        // global_mega_cache().insert(bytecode_offset, object_shape, slot_offset, flags.bits() as u16);
    }

    SlotInfo {
        offset: slot_offset,
        flags,
        is_inline: true,
    }
}

/// Handle a SetProperty IC miss.
#[inline(never)]
pub fn ic_set_property_miss(
    ic: &mut PropertyIc,
    object_shape: ShapeId,
    slot_offset: u16,
    flags: prism_runtime::object::shape::PropertyFlags,
) -> SlotInfo {
    // Update the IC
    ic.update(object_shape, slot_offset, flags);

    SlotInfo {
        offset: slot_offset,
        flags,
        is_inline: true,
    }
}

/// Handle a Call IC miss.
#[inline(never)]
pub fn ic_call_miss(ic: &mut CallIc, callee_id: u64, code_ptr: usize, argc: u16) {
    ic.update_function(callee_id, code_ptr, argc);
}

/// Handle a method call IC miss.
#[inline(never)]
pub fn ic_method_call_miss(
    ic: &mut CallIc,
    callee_id: u64,
    code_ptr: usize,
    argc: u16,
    receiver_shape: ShapeId,
) {
    ic.update_method(callee_id, code_ptr, argc, receiver_shape);
}

// =============================================================================
// IC Fast Path Helpers
// =============================================================================

/// Result of an IC lookup.
#[derive(Debug, Clone, Copy)]
pub enum IcLookupResult {
    /// Cache hit with slot info.
    Hit(SlotInfo),
    /// Cache miss - need to call miss handler.
    Miss,
    /// Megamorphic - use global cache or full lookup.
    Megamorphic,
}

/// Perform a fast property IC lookup.
///
/// This is the entry point called from JIT code.
#[inline(always)]
pub fn ic_property_lookup(ic: &mut PropertyIc, object_shape: ShapeId) -> IcLookupResult {
    match ic.lookup(object_shape) {
        Some(info) => IcLookupResult::Hit(info),
        None => {
            if ic.state() == IcState::Megamorphic {
                IcLookupResult::Megamorphic
            } else {
                IcLookupResult::Miss
            }
        }
    }
}

/// Perform a megamorphic cache lookup.
#[inline]
pub fn mega_cache_lookup(bytecode_offset: u32, shape_id: ShapeId) -> Option<SlotInfo> {
    let cache = global_mega_cache();
    cache
        .lookup(bytecode_offset, shape_id)
        .map(|(offset, flags)| SlotInfo {
            offset,
            flags: prism_runtime::object::shape::PropertyFlags::from_bits_truncate(flags as u8),
            is_inline: true,
        })
}

// =============================================================================
// IC Statistics
// =============================================================================

/// Aggregated IC statistics across all sites.
#[derive(Debug, Clone, Copy, Default)]
pub struct IcRuntimeStats {
    /// Total property IC lookups.
    pub property_lookups: u64,
    /// Property IC hits.
    pub property_hits: u64,
    /// Property IC misses.
    pub property_misses: u64,

    /// Total call IC lookups.
    pub call_lookups: u64,
    /// Call IC hits.
    pub call_hits: u64,
    /// Call IC misses.
    pub call_misses: u64,

    /// Megamorphic cache hits.
    pub mega_hits: u64,
    /// Megamorphic cache misses.
    pub mega_misses: u64,
}

impl IcRuntimeStats {
    /// Calculate overall hit rate.
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.property_hits + self.call_hits + self.mega_hits;
        let total_misses = self.property_misses + self.call_misses + self.mega_misses;
        let total = total_hits + total_misses;
        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }

    /// Calculate property IC hit rate.
    pub fn property_hit_rate(&self) -> f64 {
        let total = self.property_hits + self.property_misses;
        if total == 0 {
            0.0
        } else {
            self.property_hits as f64 / total as f64
        }
    }

    /// Calculate call IC hit rate.
    pub fn call_hit_rate(&self) -> f64 {
        let total = self.call_hits + self.call_misses;
        if total == 0 {
            0.0
        } else {
            self.call_hits as f64 / total as f64
        }
    }
}
