//! JIT code emission for safepoint polling.
//!
//! Emits efficient safepoint poll instructions using the dedicated R15 register.
//! Each poll is only 3 bytes: `test [r15], al`.
//!
//! # Safepoint Elision
//!
//! Small leaf functions that meet all criteria can skip safepoint polls:
//! - No allocations (no heap pressure)
//! - No calls (no recursion risk)
//! - No loops (bounded execution time)
//! - Small bytecode (<32 instructions)

use crate::backend::x64::assembler::Assembler;

// =============================================================================
// Constants
// =============================================================================

/// Size of the safepoint poll instruction in bytes.
///
/// `test byte ptr [r15], 0x01` = 4 bytes (41 F6 07 01)
/// Alternative with AL: `test byte ptr [r15], al` = 3 bytes (41 84 07)
pub const SAFEPOINT_POLL_SIZE: usize = 3;

/// Maximum instruction count for safepoint elision.
pub const ELISION_MAX_INSTRUCTIONS: usize = 32;

/// Maximum bytecode size for safepoint elision.
pub const ELISION_MAX_BYTECODE_SIZE: usize = 64;

// =============================================================================
// Poll Emission
// =============================================================================

/// Emit a safepoint poll instruction.
///
/// Generates: `test byte ptr [r15], al` (3 bytes)
///
/// # Performance
///
/// - Cost when armed: ~1 cycle (L1 cache hit)
/// - Cost when triggered: ~5-10μs (SIGSEGV/VEH handling)
///
/// # Arguments
///
/// * `asm` - The assembler to emit to
///
/// # Encoding
///
/// ```text
/// 41 84 07    test byte ptr [r15], al
/// │  │  └── ModR/M: [r15] (mod=00, reg=al, rm=111)
/// │  └───── Opcode: TEST r/m8, r8
/// └──────── REX.B prefix (R15 needs REX)
/// ```
#[inline]
pub fn emit_safepoint_poll(asm: &mut Assembler) {
    // REX.B (0x41) + TEST r/m8, r8 (0x84) + ModR/M [r15], al (0x07)
    asm.emit_bytes(&[0x41, 0x84, 0x07]);
}

/// Emit safepoint poll with explicit page address (RIP-relative).
///
/// This is the fallback when R15 is not available.
/// Generates: `test byte ptr [rip+disp32], 0x01` (7 bytes)
///
/// # Arguments
///
/// * `asm` - The assembler to emit to
/// * `page_addr` - Absolute address of the safepoint page
#[inline]
pub fn emit_safepoint_poll_rip_relative(asm: &mut Assembler, page_addr: usize) {
    // test byte ptr [rip + disp32], 0x01
    // F6 05 <disp32> 01
    let current_rip = asm.offset() + 7; // Size of this instruction
    let disp = (page_addr as isize - current_rip as isize) as i32;

    asm.emit_bytes(&[0xF6, 0x05]); // TEST r/m8, imm8 with RIP-relative
    asm.emit_u32(disp as u32);
    asm.emit_u8(0x01); // Immediate value
}

/// Load the safepoint page address into R15.
///
/// This should be called once in the function prologue.
/// Generates: `mov r15, imm64` (10 bytes)
///
/// # Arguments
///
/// * `asm` - The assembler to emit to
/// * `page_addr` - Absolute address of the safepoint page
#[inline]
pub fn emit_load_safepoint_register(asm: &mut Assembler, page_addr: usize) {
    // REX.WB (0x49) + MOV r64, imm64 (0xBF for R15)
    asm.emit_bytes(&[0x49, 0xBF]);
    asm.emit_u64(page_addr as u64);
}

// =============================================================================
// Safepoint Elision Analysis
// =============================================================================

/// Function characteristics for safepoint elision analysis.
#[derive(Debug, Clone, Default)]
pub struct FunctionTraits {
    /// Number of bytecode instructions.
    pub instruction_count: usize,
    /// Whether the function contains any loops.
    pub has_loops: bool,
    /// Whether the function contains any calls.
    pub has_calls: bool,
    /// Whether the function contains any allocations.
    pub has_allocations: bool,
    /// Whether the function can throw exceptions.
    pub can_throw: bool,
    /// Bytecode size in bytes.
    pub bytecode_size: usize,
}

impl FunctionTraits {
    /// Create a new FunctionTraits.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark that the function has a loop.
    #[inline]
    pub fn mark_loop(&mut self) {
        self.has_loops = true;
    }

    /// Mark that the function has a call.
    #[inline]
    pub fn mark_call(&mut self) {
        self.has_calls = true;
    }

    /// Mark that the function has an allocation.
    #[inline]
    pub fn mark_allocation(&mut self) {
        self.has_allocations = true;
    }

    /// Mark that the function can throw.
    #[inline]
    pub fn mark_throws(&mut self) {
        self.can_throw = true;
    }

    /// Check if this is a leaf function (no calls).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        !self.has_calls
    }
}

/// Determine if safepoint polls can be elided for a function.
///
/// Safepoints can be elided if ALL of the following are true:
/// - Function is a leaf (no calls)
/// - No allocations (no heap pressure)
/// - No loops (bounded execution time)
/// - Small bytecode (<32 instructions, <64 bytes)
///
/// # Arguments
///
/// * `traits` - The function characteristics
///
/// # Returns
///
/// `true` if safepoint polls can be safely elided.
#[inline]
pub fn should_elide_safepoints(traits: &FunctionTraits) -> bool {
    traits.is_leaf()
        && !traits.has_allocations
        && !traits.has_loops
        && traits.instruction_count < ELISION_MAX_INSTRUCTIONS
        && traits.bytecode_size < ELISION_MAX_BYTECODE_SIZE
}

/// Determine optimal safepoint placement for a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafepointPlacement {
    /// No safepoints needed (elided).
    None,
    /// Safepoint only at function return.
    ReturnOnly,
    /// Safepoints at loop back-edges and return.
    BackEdgesAndReturn,
    /// Safepoints at all allocation sites, back-edges, and return.
    Full,
}

/// Analyze a function and determine optimal safepoint placement.
///
/// This implements the strategic placement optimization:
/// - Leaf functions with no loops: None or ReturnOnly
/// - Functions with loops but no allocations: BackEdgesAndReturn
/// - Functions with allocations: Full
#[inline]
pub fn analyze_safepoint_placement(traits: &FunctionTraits) -> SafepointPlacement {
    if should_elide_safepoints(traits) {
        return SafepointPlacement::None;
    }

    if traits.is_leaf() && !traits.has_loops {
        return SafepointPlacement::ReturnOnly;
    }

    if !traits.has_allocations {
        return SafepointPlacement::BackEdgesAndReturn;
    }

    SafepointPlacement::Full
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poll_size() {
        assert_eq!(SAFEPOINT_POLL_SIZE, 3);
    }

    #[test]
    fn test_elision_leaf_no_loops() {
        let traits = FunctionTraits {
            instruction_count: 10,
            has_loops: false,
            has_calls: false,
            has_allocations: false,
            can_throw: false,
            bytecode_size: 30,
        };
        assert!(should_elide_safepoints(&traits));
    }

    #[test]
    fn test_no_elision_has_loops() {
        let traits = FunctionTraits {
            instruction_count: 10,
            has_loops: true,
            has_calls: false,
            has_allocations: false,
            can_throw: false,
            bytecode_size: 30,
        };
        assert!(!should_elide_safepoints(&traits));
    }

    #[test]
    fn test_no_elision_has_calls() {
        let traits = FunctionTraits {
            instruction_count: 10,
            has_loops: false,
            has_calls: true,
            has_allocations: false,
            can_throw: false,
            bytecode_size: 30,
        };
        assert!(!should_elide_safepoints(&traits));
    }

    #[test]
    fn test_no_elision_has_allocations() {
        let traits = FunctionTraits {
            instruction_count: 10,
            has_loops: false,
            has_calls: false,
            has_allocations: true,
            can_throw: false,
            bytecode_size: 30,
        };
        assert!(!should_elide_safepoints(&traits));
    }

    #[test]
    fn test_no_elision_too_large() {
        let traits = FunctionTraits {
            instruction_count: 100,
            has_loops: false,
            has_calls: false,
            has_allocations: false,
            can_throw: false,
            bytecode_size: 200,
        };
        assert!(!should_elide_safepoints(&traits));
    }

    #[test]
    fn test_placement_none() {
        let traits = FunctionTraits {
            instruction_count: 10,
            has_loops: false,
            has_calls: false,
            has_allocations: false,
            can_throw: false,
            bytecode_size: 30,
        };
        assert_eq!(
            analyze_safepoint_placement(&traits),
            SafepointPlacement::None
        );
    }

    #[test]
    fn test_placement_return_only() {
        let traits = FunctionTraits {
            instruction_count: 100, // Too large for elision
            has_loops: false,
            has_calls: false,
            has_allocations: false,
            can_throw: false,
            bytecode_size: 200,
        };
        assert_eq!(
            analyze_safepoint_placement(&traits),
            SafepointPlacement::ReturnOnly
        );
    }

    #[test]
    fn test_placement_back_edges() {
        let traits = FunctionTraits {
            instruction_count: 50,
            has_loops: true,
            has_calls: false,
            has_allocations: false,
            can_throw: false,
            bytecode_size: 100,
        };
        assert_eq!(
            analyze_safepoint_placement(&traits),
            SafepointPlacement::BackEdgesAndReturn
        );
    }

    #[test]
    fn test_placement_full() {
        let traits = FunctionTraits {
            instruction_count: 50,
            has_loops: true,
            has_calls: true,
            has_allocations: true,
            can_throw: false,
            bytecode_size: 100,
        };
        assert_eq!(
            analyze_safepoint_placement(&traits),
            SafepointPlacement::Full
        );
    }
}
