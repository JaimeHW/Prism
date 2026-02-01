//! On-Stack Replacement (OSR) support for JIT compilation.
//!
//! OSR allows the VM to transition between execution tiers (interpreter → JIT
//! or JIT → higher-tier JIT) while a function is executing. This is essential
//! for optimizing hot loops that are entered once but iterate many times.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   OSR Loop Entry                        │
//! ├─────────────────────────────────────────────────────────┤
//! │  1. Check if OSR entry point exists at loop header      │
//! │  2. If yes, capture current state (locals, stack)       │
//! │  3. Reconstruct register/stack state for JIT frame      │
//! │  4. Jump into compiled code at OSR entry point          │
//! └─────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────┐
//! │                   OSR Exit (Deoptimization)             │
//! ├─────────────────────────────────────────────────────────┤
//! │  1. At deopt point, save all live values from regs      │
//! │  2. Reconstruct interpreter state from saved values     │
//! │  3. Resume execution in interpreter                     │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use prism_jit::tier2::osr::{OsrEntry, OsrState};
//!
//! // During compilation, create OSR entry for loop header
//! let entry = OsrEntry::new(
//!     loop_header_bc_offset,
//!     jit_entry_offset,
//!     state_descriptor,
//! );
//!
//! // At runtime, when loop count exceeds threshold
//! if let Some(jit_code) = code_cache.lookup_osr(code_id, bc_offset) {
//!     let state = OsrState::capture_interpreter_state(&frame);
//!     jit_code.enter_from_osr(state);
//! }
//! ```

use std::collections::BTreeMap;

// =============================================================================
// OSR Entry Point
// =============================================================================

/// An OSR entry point in compiled code.
///
/// OSR entries are placed at loop headers to allow hot loops to be
/// entered from the interpreter mid-execution.
#[derive(Debug, Clone)]
pub struct OsrEntry {
    /// Bytecode offset of the loop header (OSR entry point).
    pub bc_offset: u32,
    /// Offset in JIT code where OSR entry begins.
    pub jit_offset: u32,
    /// State descriptor defining value mapping.
    pub state_descriptor: OsrStateDescriptor,
}

impl OsrEntry {
    /// Create a new OSR entry point.
    #[inline]
    pub fn new(bc_offset: u32, jit_offset: u32, state_descriptor: OsrStateDescriptor) -> Self {
        Self {
            bc_offset,
            jit_offset,
            state_descriptor,
        }
    }
}

// =============================================================================
// OSR State Descriptor
// =============================================================================

/// Describes how to map interpreter state to JIT state at OSR entry.
///
/// At each OSR entry point, we need to know:
/// - Which interpreter locals map to which JIT locations
/// - How to initialize JIT frame from interpreter state
#[derive(Debug, Clone, Default)]
pub struct OsrStateDescriptor {
    /// Mapping from interpreter local index to JIT location.
    local_mappings: Vec<ValueLocation>,
    /// Frame size for JIT code (for stack allocation).
    frame_size: u32,
    /// Number of callee-saved registers to preserve.
    callee_saved_count: u8,
}

impl OsrStateDescriptor {
    /// Create a new empty state descriptor.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            local_mappings: Vec::with_capacity(capacity),
            frame_size: 0,
            callee_saved_count: 0,
        }
    }

    /// Set the frame size.
    #[inline]
    pub fn set_frame_size(&mut self, size: u32) {
        self.frame_size = size;
    }

    /// Get the frame size.
    #[inline]
    pub fn frame_size(&self) -> u32 {
        self.frame_size
    }

    /// Set callee-saved register count.
    #[inline]
    pub fn set_callee_saved_count(&mut self, count: u8) {
        self.callee_saved_count = count;
    }

    /// Get callee-saved register count.
    #[inline]
    pub fn callee_saved_count(&self) -> u8 {
        self.callee_saved_count
    }

    /// Add a local mapping.
    #[inline]
    pub fn add_local_mapping(&mut self, location: ValueLocation) {
        self.local_mappings.push(location);
    }

    /// Get the location for a local variable.
    #[inline]
    pub fn local_location(&self, index: usize) -> Option<&ValueLocation> {
        self.local_mappings.get(index)
    }

    /// Get all local mappings.
    #[inline]
    pub fn local_mappings(&self) -> &[ValueLocation] {
        &self.local_mappings
    }

    /// Number of locals.
    #[inline]
    pub fn local_count(&self) -> usize {
        self.local_mappings.len()
    }
}

// =============================================================================
// Value Location
// =============================================================================

/// Location of a value in JIT-compiled code.
///
/// Used to describe where interpreter values should be placed when
/// entering JIT code via OSR, or where to find values when deoptimizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueLocation {
    /// Value is in a register.
    Register(u8),
    /// Value is on the stack at RBP-relative offset.
    Stack(i32),
    /// Value is a constant (materialized at runtime).
    Constant(u64),
    /// Value is not live at this point.
    Dead,
}

impl ValueLocation {
    /// Create a register location.
    #[inline]
    pub const fn register(reg: u8) -> Self {
        ValueLocation::Register(reg)
    }

    /// Create a stack location.
    #[inline]
    pub const fn stack(offset: i32) -> Self {
        ValueLocation::Stack(offset)
    }

    /// Create a constant location.
    #[inline]
    pub const fn constant(value: u64) -> Self {
        ValueLocation::Constant(value)
    }

    /// Create a dead location.
    #[inline]
    pub const fn dead() -> Self {
        ValueLocation::Dead
    }

    /// Check if this is a register location.
    #[inline]
    pub const fn is_register(&self) -> bool {
        matches!(self, ValueLocation::Register(_))
    }

    /// Check if this is a stack location.
    #[inline]
    pub const fn is_stack(&self) -> bool {
        matches!(self, ValueLocation::Stack(_))
    }

    /// Check if this value is live.
    #[inline]
    pub const fn is_live(&self) -> bool {
        !matches!(self, ValueLocation::Dead)
    }
}

// =============================================================================
// OSR Compiled Code
// =============================================================================

/// Compiled code with OSR entry points.
#[derive(Debug)]
pub struct OsrCompiledCode {
    /// Base address of the compiled code.
    code_base: usize,
    /// Size of the compiled code.
    code_size: u32,
    /// OSR entry points indexed by bytecode offset.
    entries: BTreeMap<u32, OsrEntry>,
    /// Deoptimization metadata.
    deopt_info: Vec<DeoptInfo>,
}

impl OsrCompiledCode {
    /// Create a new OSR-enabled compiled code block.
    #[inline]
    pub fn new(code_base: usize, code_size: u32) -> Self {
        Self {
            code_base,
            code_size,
            entries: BTreeMap::new(),
            deopt_info: Vec::new(),
        }
    }

    /// Add an OSR entry point.
    #[inline]
    pub fn add_entry(&mut self, entry: OsrEntry) {
        self.entries.insert(entry.bc_offset, entry);
    }

    /// Add deoptimization info.
    #[inline]
    pub fn add_deopt_info(&mut self, info: DeoptInfo) {
        self.deopt_info.push(info);
    }

    /// Look up OSR entry by bytecode offset.
    #[inline]
    pub fn lookup_entry(&self, bc_offset: u32) -> Option<&OsrEntry> {
        self.entries.get(&bc_offset)
    }

    /// Get the JIT entry address for an OSR entry point.
    #[inline]
    pub fn entry_address(&self, bc_offset: u32) -> Option<usize> {
        self.entries
            .get(&bc_offset)
            .map(|e| self.code_base + e.jit_offset as usize)
    }

    /// Get the base address.
    #[inline]
    pub fn code_base(&self) -> usize {
        self.code_base
    }

    /// Get the code size.
    #[inline]
    pub fn code_size(&self) -> u32 {
        self.code_size
    }

    /// Check if address is within this code block.
    #[inline]
    pub fn contains_address(&self, addr: usize) -> bool {
        addr >= self.code_base && addr < self.code_base + self.code_size as usize
    }

    /// Get deopt info by JIT offset.
    #[inline]
    pub fn deopt_info(&self, jit_offset: u32) -> Option<&DeoptInfo> {
        self.deopt_info
            .iter()
            .find(|info| info.jit_offset == jit_offset)
    }

    /// Get all OSR entries.
    #[inline]
    pub fn entries(&self) -> impl Iterator<Item = &OsrEntry> {
        self.entries.values()
    }

    /// Get number of OSR entry points.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

// =============================================================================
// Deoptimization Info
// =============================================================================

/// Information needed to deoptimize from a specific JIT code point.
///
/// When the JIT code encounters an uncommon case (type guard failure,
/// exception, etc.), it needs to transfer control back to the interpreter.
#[derive(Debug, Clone)]
pub struct DeoptInfo {
    /// Offset in JIT code where deopt can occur.
    pub jit_offset: u32,
    /// Bytecode offset to resume at in interpreter.
    pub bc_offset: u32,
    /// State descriptor for reconstructing interpreter state.
    pub state_descriptor: OsrStateDescriptor,
    /// Reason for deoptimization.
    pub reason: DeoptReason,
}

impl DeoptInfo {
    /// Create new deopt info.
    #[inline]
    pub fn new(
        jit_offset: u32,
        bc_offset: u32,
        state_descriptor: OsrStateDescriptor,
        reason: DeoptReason,
    ) -> Self {
        Self {
            jit_offset,
            bc_offset,
            state_descriptor,
            reason,
        }
    }
}

/// Reason for deoptimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeoptReason {
    /// Type guard failed (value wasn't expected type).
    TypeGuard,
    /// Overflow in arithmetic operation.
    Overflow,
    /// Division by zero.
    DivisionByZero,
    /// Out-of-bounds array access.
    BoundsCheck,
    /// Null/undefined access.
    NullCheck,
    /// Called with wrong number of arguments.
    ArityMismatch,
    /// Generic speculation failure.
    Speculation,
    /// Explicit deoptimization requested.
    Explicit,
    /// Unknown deopt reason.
    Unknown,
}

impl DeoptReason {
    /// Get a human-readable description.
    #[inline]
    pub const fn description(&self) -> &'static str {
        match self {
            DeoptReason::TypeGuard => "type guard failed",
            DeoptReason::Overflow => "arithmetic overflow",
            DeoptReason::DivisionByZero => "division by zero",
            DeoptReason::BoundsCheck => "array bounds check failed",
            DeoptReason::NullCheck => "null/undefined access",
            DeoptReason::ArityMismatch => "wrong number of arguments",
            DeoptReason::Speculation => "speculation failed",
            DeoptReason::Explicit => "explicit deoptimization",
            DeoptReason::Unknown => "unknown reason",
        }
    }
}

// =============================================================================
// OSR Entry State Builder
// =============================================================================

/// Builder for constructing OSR entry state from interpreter frame.
#[derive(Debug, Default)]
pub struct OsrStateBuilder {
    /// Values to place in registers.
    register_values: Vec<(u8, u64)>,
    /// Values to place on stack.
    stack_values: Vec<(i32, u64)>,
}

impl OsrStateBuilder {
    /// Create a new state builder.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a register value.
    #[inline]
    pub fn set_register(&mut self, reg: u8, value: u64) {
        // Remove any existing entry for this register
        self.register_values.retain(|(r, _)| *r != reg);
        self.register_values.push((reg, value));
    }

    /// Set a stack slot value.
    #[inline]
    pub fn set_stack(&mut self, offset: i32, value: u64) {
        // Remove any existing entry for this offset
        self.stack_values.retain(|(o, _)| *o != offset);
        self.stack_values.push((offset, value));
    }

    /// Get register values.
    #[inline]
    pub fn register_values(&self) -> &[(u8, u64)] {
        &self.register_values
    }

    /// Get stack values.
    #[inline]
    pub fn stack_values(&self) -> &[(i32, u64)] {
        &self.stack_values
    }

    /// Number of register values.
    #[inline]
    pub fn register_count(&self) -> usize {
        self.register_values.len()
    }

    /// Number of stack values.
    #[inline]
    pub fn stack_count(&self) -> usize {
        self.stack_values.len()
    }
}

// =============================================================================
// OSR Statistics
// =============================================================================

/// Statistics for OSR activity.
#[derive(Debug, Default, Clone)]
pub struct OsrStats {
    /// Total OSR entries.
    pub entry_count: u64,
    /// Total deoptimizations.
    pub deopt_count: u64,
    /// Deopts by reason.
    pub deopt_reasons: [u64; 9],
}

impl OsrStats {
    /// Create new stats.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an OSR entry.
    #[inline]
    pub fn record_entry(&mut self) {
        self.entry_count += 1;
    }

    /// Record a deoptimization.
    #[inline]
    pub fn record_deopt(&mut self, reason: DeoptReason) {
        self.deopt_count += 1;
        self.deopt_reasons[reason as usize] += 1;
    }

    /// Get deopt count for a specific reason.
    #[inline]
    pub fn deopt_count_for(&self, reason: DeoptReason) -> u64 {
        self.deopt_reasons[reason as usize]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_location_creation() {
        let reg = ValueLocation::register(0);
        assert!(reg.is_register());
        assert!(reg.is_live());

        let stack = ValueLocation::stack(-8);
        assert!(stack.is_stack());
        assert!(stack.is_live());

        let dead = ValueLocation::dead();
        assert!(!dead.is_live());
    }

    #[test]
    fn test_osr_state_descriptor() {
        let mut desc = OsrStateDescriptor::new();
        desc.set_frame_size(64);
        desc.set_callee_saved_count(4);
        desc.add_local_mapping(ValueLocation::register(0));
        desc.add_local_mapping(ValueLocation::stack(-8));
        desc.add_local_mapping(ValueLocation::dead());

        assert_eq!(desc.frame_size(), 64);
        assert_eq!(desc.callee_saved_count(), 4);
        assert_eq!(desc.local_count(), 3);
        assert!(desc.local_location(0).unwrap().is_register());
        assert!(desc.local_location(1).unwrap().is_stack());
        assert!(!desc.local_location(2).unwrap().is_live());
    }

    #[test]
    fn test_osr_entry() {
        let mut desc = OsrStateDescriptor::new();
        desc.set_frame_size(48);

        let entry = OsrEntry::new(100, 200, desc);
        assert_eq!(entry.bc_offset, 100);
        assert_eq!(entry.jit_offset, 200);
        assert_eq!(entry.state_descriptor.frame_size(), 48);
    }

    #[test]
    fn test_osr_compiled_code() {
        let mut code = OsrCompiledCode::new(0x10000, 0x1000);

        let desc = OsrStateDescriptor::new();
        code.add_entry(OsrEntry::new(0, 100, desc.clone()));
        code.add_entry(OsrEntry::new(50, 200, desc));

        assert_eq!(code.entry_count(), 2);
        assert_eq!(code.entry_address(0), Some(0x10000 + 100));
        assert_eq!(code.entry_address(50), Some(0x10000 + 200));
        assert!(code.entry_address(25).is_none());

        assert!(code.contains_address(0x10500));
        assert!(!code.contains_address(0x20000));
    }

    #[test]
    fn test_deopt_info() {
        let desc = OsrStateDescriptor::new();
        let info = DeoptInfo::new(100, 50, desc, DeoptReason::TypeGuard);

        assert_eq!(info.jit_offset, 100);
        assert_eq!(info.bc_offset, 50);
        assert_eq!(info.reason, DeoptReason::TypeGuard);
        assert_eq!(info.reason.description(), "type guard failed");
    }

    #[test]
    fn test_osr_state_builder() {
        let mut builder = OsrStateBuilder::new();
        builder.set_register(0, 42);
        builder.set_register(1, 100);
        builder.set_stack(-8, 0xDEADBEEF);
        builder.set_stack(-16, 0xCAFEBABE);

        assert_eq!(builder.register_count(), 2);
        assert_eq!(builder.stack_count(), 2);

        // Overwrite should work
        builder.set_register(0, 99);
        assert_eq!(builder.register_count(), 2);
    }

    #[test]
    fn test_osr_stats() {
        let mut stats = OsrStats::new();
        stats.record_entry();
        stats.record_entry();
        stats.record_deopt(DeoptReason::TypeGuard);
        stats.record_deopt(DeoptReason::TypeGuard);
        stats.record_deopt(DeoptReason::Overflow);

        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.deopt_count, 3);
        assert_eq!(stats.deopt_count_for(DeoptReason::TypeGuard), 2);
        assert_eq!(stats.deopt_count_for(DeoptReason::Overflow), 1);
        assert_eq!(stats.deopt_count_for(DeoptReason::BoundsCheck), 0);
    }

    #[test]
    fn test_deopt_reasons() {
        let reasons = [
            DeoptReason::TypeGuard,
            DeoptReason::Overflow,
            DeoptReason::DivisionByZero,
            DeoptReason::BoundsCheck,
            DeoptReason::NullCheck,
            DeoptReason::ArityMismatch,
            DeoptReason::Speculation,
            DeoptReason::Explicit,
            DeoptReason::Unknown,
        ];

        for reason in reasons {
            // Ensure description doesn't panic
            let _ = reason.description();
        }
    }
}
