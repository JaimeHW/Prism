//! State points for deoptimization and debugging.
//!
//! State points record the mapping from JIT locations (registers, spill slots)
//! back to interpreter state (locals, stack) at specific program points.
//! This enables:
//!
//! - **Deoptimization**: Reconstructing interpreter state when JIT code
//!   needs to bail out
//! - **Debugging**: Allowing debuggers to inspect local variable values
//! - **OSR Exit**: Mapping JIT state back to interpreter for seamless transition
//!
//! # Design
//!
//! ```text
//! State Point at call site:
//! ┌─────────────────────────────────────────────────────────┐
//! │  BC Offset: 42                                          │
//! │  JIT Offset: 0x100                                      │
//! │  ──────────────────────────────────────────────────────│
//! │  Local 0: Register(RAX)                                 │
//! │  Local 1: Stack(-8)                                     │
//! │  Local 2: Constant(0)                                   │
//! │  Local 3: Dead                                          │
//! │  ──────────────────────────────────────────────────────│
//! │  Eval Stack [0]: Register(RBX)                          │
//! │  Eval Stack [1]: Stack(-16)                             │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! State points are stored in a compact format optimized for:
//! - O(log n) lookup by JIT offset
//! - Cache-friendly iteration during deoptimization
//! - Minimal memory overhead (typically 16-32 bytes per state point)

use super::osr::ValueLocation;

// =============================================================================
// State Point
// =============================================================================

/// A state point recording the mapping at a specific JIT code location.
///
/// State points are inserted at:
/// - Call sites (before function calls)
/// - Loop back-edges (for potential OSR exit)
/// - Guard failures (type checks, bounds checks, etc.)
/// - Any point where we might need to deoptimize
#[derive(Debug, Clone)]
pub struct StatePoint {
    /// Offset in JIT code where this state point is valid.
    pub jit_offset: u32,
    /// Corresponding bytecode offset.
    pub bc_offset: u32,
    /// Locations of local variables.
    locals: Vec<ValueLocation>,
    /// Locations of evaluation stack values (if any are live).
    stack: Vec<ValueLocation>,
    /// Frame info for reconstructing interpreter frame.
    frame_info: StatePointFrameInfo,
}

impl StatePoint {
    /// Create a new state point.
    #[inline]
    pub fn new(jit_offset: u32, bc_offset: u32) -> Self {
        Self {
            jit_offset,
            bc_offset,
            locals: Vec::new(),
            stack: Vec::new(),
            frame_info: StatePointFrameInfo::default(),
        }
    }

    /// Create with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(
        jit_offset: u32,
        bc_offset: u32,
        local_cap: usize,
        stack_cap: usize,
    ) -> Self {
        Self {
            jit_offset,
            bc_offset,
            locals: Vec::with_capacity(local_cap),
            stack: Vec::with_capacity(stack_cap),
            frame_info: StatePointFrameInfo::default(),
        }
    }

    /// Add a local variable location.
    #[inline]
    pub fn add_local(&mut self, location: ValueLocation) {
        self.locals.push(location);
    }

    /// Add an evaluation stack location.
    #[inline]
    pub fn add_stack(&mut self, location: ValueLocation) {
        self.stack.push(location);
    }

    /// Set frame info.
    #[inline]
    pub fn set_frame_info(&mut self, info: StatePointFrameInfo) {
        self.frame_info = info;
    }

    /// Get local variable locations.
    #[inline]
    pub fn locals(&self) -> &[ValueLocation] {
        &self.locals
    }

    /// Get evaluation stack locations.
    #[inline]
    pub fn stack(&self) -> &[ValueLocation] {
        &self.stack
    }

    /// Get frame info.
    #[inline]
    pub fn frame_info(&self) -> &StatePointFrameInfo {
        &self.frame_info
    }

    /// Get location of a specific local.
    #[inline]
    pub fn local_location(&self, index: usize) -> Option<&ValueLocation> {
        self.locals.get(index)
    }

    /// Get location of a specific stack slot.
    #[inline]
    pub fn stack_location(&self, index: usize) -> Option<&ValueLocation> {
        self.stack.get(index)
    }

    /// Number of locals.
    #[inline]
    pub fn local_count(&self) -> usize {
        self.locals.len()
    }

    /// Number of stack values.
    #[inline]
    pub fn stack_count(&self) -> usize {
        self.stack.len()
    }

    /// Count of live locals (non-Dead).
    #[inline]
    pub fn live_local_count(&self) -> usize {
        self.locals.iter().filter(|l| l.is_live()).count()
    }

    /// Count of live stack values.
    #[inline]
    pub fn live_stack_count(&self) -> usize {
        self.stack.iter().filter(|l| l.is_live()).count()
    }
}

// =============================================================================
// State Point Frame Info
// =============================================================================

/// Additional frame information for state point reconstruction.
#[derive(Debug, Clone, Copy, Default)]
pub struct StatePointFrameInfo {
    /// JIT frame size in bytes.
    pub jit_frame_size: u32,
    /// Interpreter frame size in bytes.
    pub interp_frame_size: u32,
    /// Number of callee-saved registers that were pushed.
    pub callee_saved_count: u8,
    /// Whether the frame has spill slots.
    pub has_spills: bool,
}

impl StatePointFrameInfo {
    /// Create new frame info.
    #[inline]
    pub fn new(jit_frame_size: u32, interp_frame_size: u32) -> Self {
        Self {
            jit_frame_size,
            interp_frame_size,
            callee_saved_count: 0,
            has_spills: false,
        }
    }

    /// Set callee-saved count.
    #[inline]
    pub fn with_callee_saved(mut self, count: u8) -> Self {
        self.callee_saved_count = count;
        self
    }

    /// Mark as having spills.
    #[inline]
    pub fn with_spills(mut self) -> Self {
        self.has_spills = true;
        self
    }
}

// =============================================================================
// State Point Table
// =============================================================================

/// Table of state points for a compiled function.
///
/// Stored sorted by JIT offset for efficient binary search lookup.
#[derive(Debug, Default)]
pub struct StatePointTable {
    /// Sorted state points.
    points: Vec<StatePoint>,
}

impl StatePointTable {
    /// Create a new empty table.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    /// Add a state point (maintains sorted order).
    #[inline]
    pub fn add(&mut self, point: StatePoint) {
        let idx = self
            .points
            .binary_search_by_key(&point.jit_offset, |p| p.jit_offset)
            .unwrap_or_else(|i| i);
        self.points.insert(idx, point);
    }

    /// Lookup state point by exact JIT offset.
    #[inline]
    pub fn lookup_exact(&self, jit_offset: u32) -> Option<&StatePoint> {
        self.points
            .binary_search_by_key(&jit_offset, |p| p.jit_offset)
            .ok()
            .map(|i| &self.points[i])
    }

    /// Lookup state point at or before JIT offset.
    #[inline]
    pub fn lookup_at_or_before(&self, jit_offset: u32) -> Option<&StatePoint> {
        match self
            .points
            .binary_search_by_key(&jit_offset, |p| p.jit_offset)
        {
            Ok(i) => Some(&self.points[i]),
            Err(0) => None,
            Err(i) => Some(&self.points[i - 1]),
        }
    }

    /// Get all state points.
    #[inline]
    pub fn points(&self) -> &[StatePoint] {
        &self.points
    }

    /// Number of state points.
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Iterate over state points.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &StatePoint> {
        self.points.iter()
    }
}

// =============================================================================
// State Point Builder
// =============================================================================

/// Builder for constructing state points during compilation.
#[derive(Debug)]
pub struct StatePointBuilder {
    /// JIT offset of the state point.
    jit_offset: u32,
    /// Bytecode offset.
    bc_offset: u32,
    /// Local locations accumulated.
    locals: Vec<ValueLocation>,
    /// Stack locations accumulated.
    stack: Vec<ValueLocation>,
    /// Frame info.
    frame_info: Option<StatePointFrameInfo>,
}

impl StatePointBuilder {
    /// Create a new builder.
    #[inline]
    pub fn new(jit_offset: u32, bc_offset: u32) -> Self {
        Self {
            jit_offset,
            bc_offset,
            locals: Vec::new(),
            stack: Vec::new(),
            frame_info: None,
        }
    }

    /// Add a local in a register.
    #[inline]
    pub fn local_in_reg(&mut self, reg: u8) -> &mut Self {
        self.locals.push(ValueLocation::register(reg));
        self
    }

    /// Add a local on the stack.
    #[inline]
    pub fn local_on_stack(&mut self, offset: i32) -> &mut Self {
        self.locals.push(ValueLocation::stack(offset));
        self
    }

    /// Add a constant local.
    #[inline]
    pub fn local_constant(&mut self, value: u64) -> &mut Self {
        self.locals.push(ValueLocation::constant(value));
        self
    }

    /// Add a dead local.
    #[inline]
    pub fn local_dead(&mut self) -> &mut Self {
        self.locals.push(ValueLocation::dead());
        self
    }

    /// Add a stack value in a register.
    #[inline]
    pub fn stack_in_reg(&mut self, reg: u8) -> &mut Self {
        self.stack.push(ValueLocation::register(reg));
        self
    }

    /// Add a stack value on the stack.
    #[inline]
    pub fn stack_on_stack(&mut self, offset: i32) -> &mut Self {
        self.stack.push(ValueLocation::stack(offset));
        self
    }

    /// Set frame info.
    #[inline]
    pub fn frame_info(&mut self, info: StatePointFrameInfo) -> &mut Self {
        self.frame_info = Some(info);
        self
    }

    /// Build the state point.
    #[inline]
    pub fn build(self) -> StatePoint {
        let mut sp = StatePoint::new(self.jit_offset, self.bc_offset);
        for local in self.locals {
            sp.add_local(local);
        }
        for stack in self.stack {
            sp.add_stack(stack);
        }
        if let Some(info) = self.frame_info {
            sp.set_frame_info(info);
        }
        sp
    }
}

// =============================================================================
// State Reconstructor
// =============================================================================

/// Reconstructs interpreter state from JIT state at a state point.
#[derive(Debug)]
pub struct StateReconstructor<'a> {
    /// The state point to reconstruct from.
    state_point: &'a StatePoint,
    /// Saved register values from the JIT frame.
    saved_registers: &'a [u64],
    /// JIT frame base pointer.
    frame_base: usize,
}

impl<'a> StateReconstructor<'a> {
    /// Create a new reconstructor.
    #[inline]
    pub fn new(state_point: &'a StatePoint, saved_registers: &'a [u64], frame_base: usize) -> Self {
        Self {
            state_point,
            saved_registers,
            frame_base,
        }
    }

    /// Read a value from the given location.
    ///
    /// # Safety
    ///
    /// The caller must ensure frame_base points to valid memory.
    pub unsafe fn read_location(&self, location: &ValueLocation) -> Option<u64> {
        match *location {
            ValueLocation::Register(reg) => self.saved_registers.get(reg as usize).copied(),
            ValueLocation::Stack(offset) => {
                let addr = (self.frame_base as isize + offset as isize) as *const u64;
                // SAFETY: Caller guarantees frame_base is valid
                Some(unsafe { std::ptr::read(addr) })
            }
            ValueLocation::Constant(value) => Some(value),
            ValueLocation::Dead => None,
        }
    }

    /// Reconstruct all local variable values.
    ///
    /// # Safety
    ///
    /// The caller must ensure frame_base points to valid memory.
    pub unsafe fn reconstruct_locals(&self) -> Vec<Option<u64>> {
        self.state_point
            .locals()
            .iter()
            // SAFETY: Caller provided valid frame_base
            .map(|loc| unsafe { self.read_location(loc) })
            .collect()
    }

    /// Reconstruct all evaluation stack values.
    ///
    /// # Safety
    ///
    /// The caller must ensure frame_base points to valid memory.
    pub unsafe fn reconstruct_stack(&self) -> Vec<Option<u64>> {
        self.state_point
            .stack()
            .iter()
            // SAFETY: Caller provided valid frame_base
            .map(|loc| unsafe { self.read_location(loc) })
            .collect()
    }

    /// Get the bytecode offset to resume at.
    #[inline]
    pub fn resume_bc_offset(&self) -> u32 {
        self.state_point.bc_offset
    }
}
