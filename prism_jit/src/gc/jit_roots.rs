//! JIT frame walking for garbage collection.
//!
//! This module provides the infrastructure to walk JIT stack frames during
//! garbage collection, identifying live references that need to be traced.
//!
//! # Frame Layout (x64)
//!
//! ```text
//! ┌────────────────────────┐  High addresses
//! │     Return Address     │  ← Points into caller's code
//! ├────────────────────────┤
//! │      Saved RBP         │  ← Frame pointer chain
//! ├────────────────────────┤  ← RBP points here
//! │      Local Slot 0      │  [RBP - 8]
//! │      Local Slot 1      │  [RBP - 16]
//! │         ...            │
//! │      Local Slot N      │  [RBP - 8*(N+1)]
//! ├────────────────────────┤
//! │    Spill Slot 0        │  [RBP - frame_size + spill_offset]
//! │    Spill Slot 1        │
//! │         ...            │
//! ├────────────────────────┤
//! │   Callee-Saved Regs    │
//! ├────────────────────────┤  ← RSP points here
//! │     (Stack grows)      │
//! └────────────────────────┘  Low addresses
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use prism_jit::gc::{JitFrameWalker, StackMapRegistry};
//!
//! // During GC, walk JIT frames
//! let walker = JitFrameWalker::new(registry);
//! walker.walk_frames(rbp, rsp, return_addr, |value, location| {
//!     tracer.trace_value(value);
//! });
//! ```

use super::stackmap::{CompactStackMapArray, StackMapRef, StackMapRegistry};
use prism_core::Value;
use std::ptr;

// =============================================================================
// JitRoots
// =============================================================================

/// Collection of roots found during JIT frame walking.
#[derive(Debug, Default)]
pub struct JitRoots {
    /// Live values from stack slots.
    pub stack_values: Vec<(Value, RootLocation)>,
    /// Live values from registers (saved on stack during GC).
    pub register_values: Vec<(Value, RootLocation)>,
}

impl JitRoots {
    /// Create empty root collection.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            stack_values: Vec::with_capacity(capacity),
            register_values: Vec::with_capacity(capacity),
        }
    }

    /// Total number of roots.
    #[inline]
    pub fn len(&self) -> usize {
        self.stack_values.len() + self.register_values.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.stack_values.is_empty() && self.register_values.is_empty()
    }

    /// Clear all roots.
    #[inline]
    pub fn clear(&mut self) {
        self.stack_values.clear();
        self.register_values.clear();
    }

    /// Iterate over all roots.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &(Value, RootLocation)> {
        self.stack_values.iter().chain(self.register_values.iter())
    }

    /// Update a root value after GC relocation.
    pub fn update_root(&mut self, location: RootLocation, new_value: Value) {
        for (value, loc) in self.stack_values.iter_mut() {
            if *loc == location {
                *value = new_value;
                return;
            }
        }
        for (value, loc) in self.register_values.iter_mut() {
            if *loc == location {
                *value = new_value;
                return;
            }
        }
    }
}

// =============================================================================
// RootLocation
// =============================================================================

/// Location of a root in a JIT frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootLocation {
    /// Stack slot at RBP-relative offset.
    Stack { frame_base: usize, offset: i32 },
    /// Register saved to stack (register index + stack location).
    Register { reg_index: u8, saved_at: usize },
}

impl RootLocation {
    /// Get the memory address where this root is stored.
    #[inline]
    pub fn address(&self) -> usize {
        match self {
            RootLocation::Stack { frame_base, offset } => {
                (*frame_base as isize + *offset as isize) as usize
            }
            RootLocation::Register { saved_at, .. } => *saved_at,
        }
    }
}

// =============================================================================
// JitFrameWalker
// =============================================================================

/// Walker for JIT stack frames.
///
/// Uses stack maps to precisely identify live references.
pub struct JitFrameWalker<'a> {
    /// Stack map registry for lookups.
    registry: &'a StackMapRegistry,
    /// Optional compact array for faster lookup.
    compact: Option<&'a CompactStackMapArray>,
}

impl<'a> JitFrameWalker<'a> {
    /// Create a new frame walker with the given registry.
    #[inline]
    pub fn new(registry: &'a StackMapRegistry) -> Self {
        Self {
            registry,
            compact: None,
        }
    }

    /// Create a frame walker with a compact array (faster lookup).
    #[inline]
    pub fn with_compact(registry: &'a StackMapRegistry, compact: &'a CompactStackMapArray) -> Self {
        Self {
            registry,
            compact: Some(compact),
        }
    }

    /// Look up safepoint by return address.
    #[inline]
    fn lookup(&self, return_addr: usize) -> Option<StackMapRef> {
        if let Some(compact) = self.compact {
            compact.lookup(return_addr)
        } else {
            self.registry.lookup(return_addr)
        }
    }

    /// Walk JIT frames starting from the given frame.
    ///
    /// # Arguments
    ///
    /// * `rbp` - Base pointer of the current frame
    /// * `rsp` - Stack pointer
    /// * `return_addr` - Return address (points into JIT code)
    /// * `saved_registers` - Saved register values (if available)
    /// * `visitor` - Callback for each live root
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - rbp and rsp point to valid stack memory
    /// - return_addr is a valid return address
    /// - saved_registers contains all callee-saved registers if provided
    pub unsafe fn walk_frames<F>(
        &self,
        mut rbp: *const u8,
        mut _rsp: *const u8,
        mut return_addr: *const u8,
        saved_registers: Option<&SavedRegisters>,
        mut visitor: F,
    ) where
        F: FnMut(Value, RootLocation),
    {
        // Walk frame chain
        loop {
            // Check for null/invalid frame pointer (end of chain)
            if rbp.is_null() || (rbp as usize) < 0x1000 {
                break;
            }

            // Look up stack map for this return address
            if let Some(stack_map_ref) = self.lookup(return_addr as usize) {
                let safepoint = stack_map_ref.safepoint;
                let frame_base = rbp as usize;

                // Trace live stack slots
                for slot_idx in safepoint.live_stack_slots() {
                    // Slot offset: [RBP - 8 * (slot_idx + 1)]
                    let offset = -8 * (slot_idx as i32 + 1);
                    let slot_addr = (frame_base as isize + offset as isize) as *const Value;

                    if !slot_addr.is_null() {
                        // SAFETY: We verified slot_addr is not null and the caller guarantees
                        // that rbp points to valid stack memory.
                        let value = unsafe { ptr::read(slot_addr) };
                        let location = RootLocation::Stack { frame_base, offset };
                        visitor(value, location);
                    }
                }

                // Trace live registers (if we have saved register state)
                if let Some(regs) = saved_registers {
                    for reg_idx in safepoint.live_registers() {
                        if let Some(value) = regs.get(reg_idx) {
                            let location = RootLocation::Register {
                                reg_index: reg_idx,
                                saved_at: regs.register_save_area as usize + (reg_idx as usize * 8),
                            };
                            visitor(value, location);
                        }
                    }
                }
            }

            // Move to next frame in chain
            // Previous RBP is at [RBP]
            // Return address is at [RBP + 8]
            // SAFETY: The caller guarantees rbp points to valid stack memory.
            let prev_rbp = unsafe { ptr::read(rbp as *const *const u8) };
            // SAFETY: RBP + 8 is the return address location on x64.
            return_addr = unsafe { ptr::read((rbp as *const *const u8).add(1)) };
            _rsp = rbp;
            rbp = prev_rbp;
        }
    }

    /// Walk a single frame and collect roots.
    ///
    /// # Safety
    ///
    /// Same requirements as `walk_frames`.
    pub unsafe fn walk_single_frame(
        &self,
        rbp: *const u8,
        return_addr: *const u8,
        saved_registers: Option<&SavedRegisters>,
    ) -> JitRoots {
        let mut roots = JitRoots::with_capacity(16);

        if let Some(stack_map_ref) = self.lookup(return_addr as usize) {
            let safepoint = stack_map_ref.safepoint;
            let frame_base = rbp as usize;

            // Collect stack roots
            for slot_idx in safepoint.live_stack_slots() {
                let offset = -8 * (slot_idx as i32 + 1);
                let slot_addr = (frame_base as isize + offset as isize) as *const Value;

                if !slot_addr.is_null() {
                    // SAFETY: slot_addr is verified non-null and caller guarantees valid stack memory.
                    let value = unsafe { ptr::read(slot_addr) };
                    let location = RootLocation::Stack { frame_base, offset };
                    roots.stack_values.push((value, location));
                }
            }

            // Collect register roots
            if let Some(regs) = saved_registers {
                for reg_idx in safepoint.live_registers() {
                    if let Some(value) = regs.get(reg_idx) {
                        let location = RootLocation::Register {
                            reg_index: reg_idx,
                            saved_at: regs.register_save_area as usize + (reg_idx as usize * 8),
                        };
                        roots.register_values.push((value, location));
                    }
                }
            }
        }

        roots
    }
}

// =============================================================================
// SavedRegisters
// =============================================================================

/// Saved register state for GC.
///
/// When entering a GC safepoint, we save all potentially-live registers
/// to a known memory location for scanning.
#[repr(C, align(16))]
pub struct SavedRegisters {
    /// Saved GPR values (RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8-R15).
    pub gprs: [u64; 16],
    /// Address where registers are saved (for root location tracking).
    pub register_save_area: *const u8,
}

impl SavedRegisters {
    /// Create a new SavedRegisters instance.
    #[inline]
    pub fn new() -> Self {
        Self {
            gprs: [0; 16],
            register_save_area: ptr::null(),
        }
    }

    /// Get a register value as a Value.
    #[inline]
    pub fn get(&self, reg_index: u8) -> Option<Value> {
        if (reg_index as usize) < self.gprs.len() {
            // Reinterpret u64 as Value (NaN-boxing)
            Some(Value::from_bits(self.gprs[reg_index as usize]))
        } else {
            None
        }
    }

    /// Set a register value.
    #[inline]
    pub fn set(&mut self, reg_index: u8, value: Value) {
        if (reg_index as usize) < self.gprs.len() {
            self.gprs[reg_index as usize] = value.to_bits();
        }
    }

    /// Set from raw bits.
    #[inline]
    pub fn set_raw(&mut self, reg_index: u8, bits: u64) {
        if (reg_index as usize) < self.gprs.len() {
            self.gprs[reg_index as usize] = bits;
        }
    }
}

impl Default for SavedRegisters {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Conservative Stack Scanning (Fallback)
// =============================================================================

/// Conservative stack scanner for when precise stack maps are unavailable.
///
/// This is a fallback for:
/// - Interpreter frames mixed with JIT frames
/// - Frames without stack maps (before compilation completes)
/// - Emergency GC when metadata is unavailable
pub struct ConservativeScanner {
    /// Minimum valid heap address.
    heap_start: usize,
    /// Maximum valid heap address.
    heap_end: usize,
}

impl ConservativeScanner {
    /// Create a new conservative scanner with heap bounds.
    #[inline]
    pub fn new(heap_start: usize, heap_end: usize) -> Self {
        Self {
            heap_start,
            heap_end,
        }
    }

    /// Check if a value looks like it could be a heap pointer.
    #[inline]
    pub fn is_potential_pointer(&self, bits: u64) -> bool {
        let addr = bits as usize;
        // Check alignment (8-byte aligned) and heap bounds
        (addr & 0x7) == 0 && addr >= self.heap_start && addr < self.heap_end
    }

    /// Scan a memory range for potential pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory range is valid and readable.
    pub unsafe fn scan_range<F>(&self, start: *const u8, end: *const u8, mut visitor: F)
    where
        F: FnMut(usize, u64),
    {
        let mut ptr = start as *const u64;
        let end_ptr = end as *const u64;

        while ptr < end_ptr {
            // SAFETY: Caller guarantees the memory range is valid and readable.
            let bits = unsafe { ptr::read(ptr) };
            if self.is_potential_pointer(bits) {
                visitor(ptr as usize, bits);
            }
            // SAFETY: We're iterating through a valid memory range.
            ptr = unsafe { ptr.add(1) };
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_roots_basics() {
        let mut roots = JitRoots::new();
        assert!(roots.is_empty());

        roots.stack_values.push((
            Value::int(42).unwrap(),
            RootLocation::Stack {
                frame_base: 0x1000,
                offset: -8,
            },
        ));
        assert_eq!(roots.len(), 1);
        assert!(!roots.is_empty());

        roots.clear();
        assert!(roots.is_empty());
    }

    #[test]
    fn test_root_location_address() {
        let stack_loc = RootLocation::Stack {
            frame_base: 0x1000,
            offset: -16,
        };
        assert_eq!(stack_loc.address(), 0x1000 - 16);

        let reg_loc = RootLocation::Register {
            reg_index: 0,
            saved_at: 0x2000,
        };
        assert_eq!(reg_loc.address(), 0x2000);
    }

    #[test]
    fn test_saved_registers() {
        let mut regs = SavedRegisters::new();

        regs.set(0, Value::int(42).unwrap());
        assert_eq!(regs.get(0).unwrap().as_int(), Some(42));

        regs.set_raw(1, 0x12345678);
        assert_eq!(regs.gprs[1], 0x12345678);

        // Out of bounds
        assert!(regs.get(20).is_none());
    }

    #[test]
    fn test_conservative_scanner() {
        let scanner = ConservativeScanner::new(0x10000, 0x20000);

        // Valid pointer
        assert!(scanner.is_potential_pointer(0x15000));

        // Unaligned
        assert!(!scanner.is_potential_pointer(0x15001));

        // Below heap
        assert!(!scanner.is_potential_pointer(0x5000));

        // Above heap
        assert!(!scanner.is_potential_pointer(0x25000));
    }
}
