//! Float-specialized JIT templates for high-performance floating-point arithmetic.
//!
//! Provides type-specialized native code generation for:
//! - **Float addition** (`a + b`) — fully inline SSE2 `ADDSD`
//! - **Float subtraction** (`a - b`) — fully inline SSE2 `SUBSD`
//! - **Float multiplication** (`a * b`) — fully inline SSE2 `MULSD`
//! - **Float division** (`a / b`) — fully inline SSE2 `DIVSD` with zero-div guard
//! - **Float negation** (`-a`) — fully inline SSE2 `XORPD` sign-bit flip
//! - **Float absolute value** (`abs(a)`) — fully inline bit-mask clear sign bit
//! - **Float comparison** (`a < b`, etc.) — fully inline `UCOMISD` + `SETcc`
//!
//! # NaN-Boxing: Float Guard Strategy
//!
//! In Prism's NaN-boxing scheme, a float is any 64-bit value whose upper 16 bits
//! are *below* the QNAN threshold. Specifically:
//!
//! ```text
//! value >> 48  <  (QNAN_BITS >> 48)  →  it's a float (raw IEEE 754 bits)
//! value >> 48  >= (QNAN_BITS >> 48)  →  it's a tagged value (int, bool, etc.)
//! ```
//!
//! This means floats are stored as their raw IEEE 754 double-precision bits.
//! No boxing/unboxing is needed — the value in the frame slot IS the float.
//!
//! # SSE2 Strategy
//!
//! Since values live in memory (frame slots), we can directly load them into
//! XMM registers via `MOVSD xmm, [mem]` and store results back with
//! `MOVSD [mem], xmm`. No GPR↔XMM register moves required.
//!
//! The code generation pattern for binary ops:
//! 1. Float-guard LHS (GPR: load, SHR 48, CMP, JB ok / JMP deopt)
//! 2. Float-guard RHS (same)
//! 3. Load LHS into XMM0 from frame slot
//! 4. Load RHS into XMM1 from frame slot
//! 5. Perform SSE2 op (ADDSD/SUBSD/MULSD/DIVSD)
//! 6. Store result from XMM0 to destination frame slot

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::Condition;
use crate::backend::x64::Gpr;
use crate::backend::x64::registers::Xmm;

// =============================================================================
// Float Type Guard Helper
// =============================================================================

/// Emit code to verify a value is a float (not a NaN-boxed tagged value).
///
/// Performs:
/// 1. Load value into GPR
/// 2. Shift right by 48 to isolate upper 16 bits
/// 3. Compare with QNAN threshold
/// 4. If below (unsigned), it's a valid float → continue
/// 5. If at or above, it's a tagged value → deopt
///
/// After this function, the value in the original frame slot is confirmed
/// to be a raw IEEE 754 float.
fn emit_float_guard(ctx: &mut TemplateContext, value_reg: Gpr, scratch: Gpr, deopt_idx: usize) {
    ctx.asm.mov_rr(scratch, value_reg);
    ctx.asm.shr_ri(scratch, 48);

    let qnan_check = (value_tags::QNAN_BITS >> 48) as i32;
    ctx.asm.cmp_ri(scratch, qnan_check);

    // If upper bits < QNAN pattern → valid float
    let ok_label = ctx.asm.create_label();
    ctx.asm.jb(ok_label);

    // Not a float → deopt
    ctx.asm.jmp(ctx.deopt_label(deopt_idx));

    ctx.asm.bind_label(ok_label);
}

// =============================================================================
// Float Binary Arithmetic Templates
// =============================================================================

/// Template for float addition (`a + b`).
///
/// # Strategy
///
/// 1. Guard both operands as float
/// 2. Load into XMM0, XMM1 via MOVSD from frame slots
/// 3. ADDSD XMM0, XMM1
/// 4. MOVSD result to destination frame slot
///
/// # Estimated Size: ~100 bytes
pub struct FloatAddTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl FloatAddTemplate {
    #[inline]
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatAddTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard LHS as float
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Guard RHS as float
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Load operands into XMM
        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);

        // ADDSD
        ctx.asm.addsd(Xmm::Xmm0, Xmm::Xmm1);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.movsd_mr(&dst_slot, Xmm::Xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for float subtraction (`a - b`).
///
/// Same structure as FloatAdd but with SUBSD.
pub struct FloatSubTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl FloatSubTemplate {
    #[inline]
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatSubTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);
        ctx.asm.subsd(Xmm::Xmm0, Xmm::Xmm1);

        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.movsd_mr(&dst_slot, Xmm::Xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for float multiplication (`a * b`).
///
/// Same structure as FloatAdd but with MULSD.
pub struct FloatMulTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl FloatMulTemplate {
    #[inline]
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatMulTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);
        ctx.asm.mulsd(Xmm::Xmm0, Xmm::Xmm1);

        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.movsd_mr(&dst_slot, Xmm::Xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for float division (`a / b`).
///
/// Includes an additional zero-division guard: if RHS == 0.0, deopt
/// to the interpreter which will raise ZeroDivisionError.
///
/// # Strategy
///
/// 1. Guard both as float
/// 2. Load RHS into XMM1
/// 3. XORPD XMM2, XMM2 (create +0.0)
/// 4. UCOMISD XMM1, XMM2 (compare RHS with 0.0)
/// 5. JE → deopt (ZeroDivisionError)
/// 6. Load LHS into XMM0
/// 7. DIVSD XMM0, XMM1
/// 8. Store result
///
/// # Estimated Size: ~120 bytes (includes zero guard)
pub struct FloatDivTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl FloatDivTemplate {
    #[inline]
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatDivTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard LHS as float
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Guard RHS as float
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Load RHS first to check for zero
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);

        // Zero-division guard: XMM2 = 0.0, compare RHS with 0.0
        ctx.asm.zero_xmm(Xmm::Xmm2);
        ctx.asm.ucomisd(Xmm::Xmm1, Xmm::Xmm2);
        // UCOMISD sets ZF=1, PF=0 for equal; JP would catch unordered (NaN)
        // Use JE + JP pattern: deopt on exact zero
        ctx.asm.je(ctx.deopt_label(self.deopt_idx));

        // Load LHS
        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);

        // DIVSD
        ctx.asm.divsd(Xmm::Xmm0, Xmm::Xmm1);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.movsd_mr(&dst_slot, Xmm::Xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// Float Unary Templates
// =============================================================================

/// Template for float negation (`-a`).
///
/// # Strategy
///
/// Uses XORPD with a sign-bit mask (0x8000_0000_0000_0000) to flip the sign.
/// Since we don't have ANDPD in the assembler, we use a GPR-based approach:
///
/// 1. Guard operand as float
/// 2. Load value into GPR
/// 3. XOR with sign-bit constant (0x8000_0000_0000_0000)
/// 4. Store result
///
/// This is branchless and uses a single 64-bit XOR instruction.
///
/// # Estimated Size: ~70 bytes
pub struct FloatNegTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl FloatNegTemplate {
    #[inline]
    pub fn new(dst_reg: u8, src_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            src_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatNegTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard and load value
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(acc, &src_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // XOR with sign bit: flip bit 63
        // sign_bit = 0x8000_0000_0000_0000
        let sign_bit: i64 = i64::MIN; // 0x8000_0000_0000_0000
        ctx.asm.mov_ri64(scratch1, sign_bit);
        ctx.asm.xor_rr(acc, scratch1);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        70
    }
}

/// Template for float absolute value (`abs(a)`).
///
/// # Strategy
///
/// Clears the sign bit using AND with ~(1 << 63) = 0x7FFF_FFFF_FFFF_FFFF.
/// This is branchless and uses a single 64-bit AND instruction.
///
/// # Estimated Size: ~70 bytes
pub struct FloatAbsTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl FloatAbsTemplate {
    #[inline]
    pub fn new(dst_reg: u8, src_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            src_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatAbsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard and load value
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(acc, &src_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // AND with clear-sign-bit mask: 0x7FFF_FFFF_FFFF_FFFF
        let clear_sign: i64 = i64::MAX; // 0x7FFF_FFFF_FFFF_FFFF
        ctx.asm.mov_ri64(scratch1, clear_sign);
        ctx.asm.and_rr(acc, scratch1);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        70
    }
}

// =============================================================================
// Float Comparison Template
// =============================================================================

/// Comparison operation for float comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatCmpOp {
    /// `a < b`
    Lt,
    /// `a <= b`
    Le,
    /// `a == b`
    Eq,
    /// `a != b`
    Ne,
    /// `a > b`
    Gt,
    /// `a >= b`
    Ge,
}

/// Template for float comparison (`a <op> b`).
///
/// # Strategy
///
/// 1. Guard both operands as float
/// 2. Load into XMM0, XMM1
/// 3. UCOMISD XMM0, XMM1
/// 4. Emit branchless boolean result using conditional moves or SETcc
///
/// UCOMISD sets:
/// - ZF=1, PF=0 for equal
/// - CF=1 for less-than
/// - ZF=0, PF=0, CF=0 for greater-than
/// - PF=1 for unordered (NaN)
///
/// For NaN comparisons: Python's `nan < x` and `nan == x` both return False,
/// and `nan != x` returns True. All comparisons with NaN are unordered, so
/// PF=1. We deopt to the interpreter for unordered results to handle this
/// correctly.
///
/// # Estimated Size: ~120 bytes
pub struct FloatCompareTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub op: FloatCmpOp,
    pub deopt_idx: usize,
}

impl FloatCompareTemplate {
    #[inline]
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, op: FloatCmpOp, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            op,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatCompareTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard LHS as float
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Guard RHS as float
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Load into XMM for comparison
        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);

        // UCOMISD: compare LHS with RHS
        ctx.asm.ucomisd(Xmm::Xmm0, Xmm::Xmm1);

        // Deopt on unordered (NaN) — PF=1 means unordered
        ctx.asm
            .jcc(Condition::Parity, ctx.deopt_label(self.deopt_idx));

        // Branch based on comparison operator
        // After UCOMISD (non-NaN):
        //   LHS <  RHS → CF=1
        //   LHS == RHS → ZF=1
        //   LHS >  RHS → CF=0, ZF=0
        let true_label = ctx.asm.create_label();
        let done_label = ctx.asm.create_label();

        match self.op {
            FloatCmpOp::Lt => ctx.asm.jb(true_label),  // CF=1
            FloatCmpOp::Le => ctx.asm.jbe(true_label), // CF=1 or ZF=1
            FloatCmpOp::Eq => ctx.asm.je(true_label),  // ZF=1
            FloatCmpOp::Ne => ctx.asm.jne(true_label), // ZF=0
            FloatCmpOp::Gt => ctx.asm.ja(true_label),  // CF=0 and ZF=0
            FloatCmpOp::Ge => ctx.asm.jae(true_label), // CF=0
        }

        // False path
        ctx.asm.mov_ri64(acc, value_tags::false_value() as i64);
        ctx.asm.jmp(done_label);

        // True path
        ctx.asm.bind_label(true_label);
        ctx.asm.mov_ri64(acc, value_tags::true_value() as i64);

        ctx.asm.bind_label(done_label);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        140
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
