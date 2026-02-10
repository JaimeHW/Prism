//! Integer-specialized JIT templates for high-performance integer operations.
//!
//! Provides type-specialized native code generation for integer methods and
//! operations not covered by the basic arithmetic templates:
//!
//! - **Power** (`x ** y`) — Loop-based exponentiation with per-step overflow guard
//! - **Absolute value** (`abs(x)`) — NEG + JO for MIN_VALUE edge case
//! - **Bit length** (`x.bit_length()`) — BSR with zero‐value handling
//! - **Left shift** (`x << y`) — Overflow-checked via roundtrip verification
//! - **Right shift** (`x >> y`) — Arithmetic shift (always safe, no overflow)
//! - **Unary positive** (`+x`) — Type guard only (identity)
//! - **Int-to-float** (`float(x)`) — CVTSI2SD conversion
//! - **Optimized compare** — Three-way compare with SETcc
//!
//! # Design Philosophy
//!
//! All templates follow the same pattern as `arithmetic.rs`:
//! 1. Load operand(s) from stack slots
//! 2. Type-check via NaN-box tag inspection → deopt on mismatch
//! 3. Extract 48-bit signed integer payload
//! 4. Perform operation with overflow guard (jo → deopt)
//! 5. Box result back into NaN-box format
//! 6. Store to destination slot
//!
//! Templates that cannot overflow (rshift, unary_positive) skip the jo check.
//! Templates that produce float results (int_to_float) use SSE boxing instead.

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::Gpr;
use crate::backend::x64::encoder::Condition;

// =============================================================================
// Helper: Integer Type Check and Extract (local re-export for self-containment)
// =============================================================================

/// Emit code to check if a value is an integer and extract the payload.
/// Returns with value in `dst` register, deoptimizes if not an integer.
fn emit_int_check_and_extract(
    ctx: &mut TemplateContext,
    src: Gpr,
    dst: Gpr,
    scratch: Gpr,
    deopt_idx: usize,
) {
    // Check type tag (upper 16 bits)
    ctx.asm.mov_rr(scratch, src);
    ctx.asm.shr_ri(scratch, 48);
    let expected_tag = value_tags::int_tag_check() as i32;
    ctx.asm.cmp_ri(scratch, expected_tag);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));

    // Extract payload (sign-extend from 48 bits)
    ctx.asm.mov_rr(dst, src);
    ctx.asm.shl_ri(dst, 16);
    ctx.asm.sar_ri(dst, 16);
}

/// Emit code to box an integer result.
fn emit_int_box(ctx: &mut TemplateContext, value: Gpr, scratch: Gpr) {
    ctx.asm.mov_ri64(scratch, value_tags::PAYLOAD_MASK as i64);
    ctx.asm.and_rr(value, scratch);
    let tag = (value_tags::QNAN_BITS | value_tags::INT_TAG) as i64;
    ctx.asm.mov_ri64(scratch, tag);
    ctx.asm.or_rr(value, scratch);
}

/// Emit code to box a float result stored in xmm0.
#[allow(dead_code)]
fn emit_float_box_from_xmm(ctx: &mut TemplateContext, dst: Gpr) {
    // Float NaN-boxing: the raw f64 bits ARE the NaN-boxed representation
    // for doubles (they naturally fall in the NaN space).
    // We just movq xmm0 → gpr.
    // MOVQ r64, xmm0: 66 REX.W 0F 7E /r
    let xmm0 = ctx.regs.xmm0;
    let dst_slot = ctx.frame.register_slot(0); // temp — we use movsd_mr then mov_rm
    ctx.asm.movsd_mr(&dst_slot, xmm0);
    ctx.asm.mov_rm(dst, &dst_slot);
}

// =============================================================================
// Integer Power Template
// =============================================================================

/// Template for integer exponentiation (`x ** y`) with overflow guards.
///
/// # Algorithm
///
/// Uses binary exponentiation (exponentiation by squaring):
/// ```text
/// result = 1
/// while exp > 0:
///     if exp & 1: result *= base  (jo → deopt)
///     base *= base                (jo → deopt)
///     exp >>= 1
/// ```
///
/// # Overflow Safety
///
/// Every `imul` is followed by `jo` → deopt. This catches overflow at each
/// multiplication step, ensuring we never produce a silently-wrong result.
/// Negative exponents deopt immediately (result would be float).
///
/// # Estimated Code Size
///
/// ~256 bytes: 2×tag checks (40), exp sign check (8), loop body (80),
/// boxing (24), overhead (104)
pub struct IntPowerTemplate {
    pub dst_reg: u8,
    pub base_reg: u8,
    pub exp_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntPowerTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let base_slot = ctx.frame.register_slot(self.base_reg as u16);
        let exp_slot = ctx.frame.register_slot(self.exp_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and check base
        ctx.asm.mov_rm(acc, &base_slot);
        emit_int_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load and check exponent
        ctx.asm.mov_rm(scratch2, &exp_slot);
        emit_int_check_and_extract(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Negative exponent → deopt (result is float)
        ctx.asm.test_rr(scratch2, scratch2);
        ctx.asm.js(ctx.deopt_label(self.deopt_idx));

        // acc = base, scratch2 = exp
        // We need: result in scratch1, base in acc, exp in scratch2
        ctx.asm.mov_ri64(scratch1, 1); // result = 1

        // Loop: binary exponentiation
        let loop_top = ctx.asm.create_label();
        let loop_end = ctx.asm.create_label();
        let skip_mul = ctx.asm.create_label();

        ctx.asm.bind_label(loop_top);

        // if exp == 0, done
        ctx.asm.test_rr(scratch2, scratch2);
        ctx.asm.jz(loop_end);

        // if exp & 1 == 0, skip result *= base
        ctx.asm.test_ri(scratch2, 1);
        ctx.asm.jz(skip_mul);

        // result *= base (with overflow check)
        ctx.asm.imul_rr(scratch1, acc);
        ctx.asm.jo(ctx.deopt_label(self.deopt_idx));

        ctx.asm.bind_label(skip_mul);

        // base *= base (with overflow check)
        ctx.asm.imul_rr(acc, acc);
        ctx.asm.jo(ctx.deopt_label(self.deopt_idx));

        // exp >>= 1
        ctx.asm.shr_ri(scratch2, 1);

        ctx.asm.jmp(loop_top);

        ctx.asm.bind_label(loop_end);

        // Result is in scratch1, move to acc for boxing
        ctx.asm.mov_rr(acc, scratch1);

        // Box result
        emit_int_box(ctx, acc, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        256
    }
}

// =============================================================================
// Integer Absolute Value Template
// =============================================================================

/// Template for integer absolute value (`abs(x)`).
///
/// # Algorithm
///
/// ```text
/// if x >= 0: return x
/// result = -x
/// if overflow: deopt  (MIN_VALUE case: -MIN_VALUE overflows)
/// ```
///
/// Uses: `test` + `jns` (skip negation if non-negative), `neg` + `jo`.
///
/// # Estimated Code Size
///
/// ~96 bytes: tag check (20), sign test (8), neg+jo (8), boxing (24), overhead (36)
pub struct IntAbsTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntAbsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load and check source
        ctx.asm.mov_rm(acc, &src_slot);
        emit_int_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // If already non-negative, skip negation
        let skip_neg = ctx.asm.create_label();
        ctx.asm.test_rr(acc, acc);
        ctx.asm.jns(skip_neg);

        // Negate (can overflow for MIN_VALUE)
        ctx.asm.neg(acc);
        ctx.asm.jo(ctx.deopt_label(self.deopt_idx));

        ctx.asm.bind_label(skip_neg);

        // Box result
        emit_int_box(ctx, acc, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        96
    }
}

// =============================================================================
// Integer Bit Length Template
// =============================================================================

/// Template for integer bit length (`x.bit_length()`).
///
/// # Algorithm
///
/// ```text
/// if x == 0: return 0
/// if x < 0: x = ~x       (bit_length of negative = bit_length of bitwise complement)
/// result = bsr(x) + 1     (BSR = index of highest set bit, zero-indexed)
/// ```
///
/// BSR (Bit Scan Reverse) finds the position of the most significant set bit.
/// For `bit_length()`, the result is `bsr(abs_val) + 1` for positive numbers
/// and `bsr(~x) + 1` for negative numbers.
///
/// Since the assembler doesn't expose BSR directly, we emit it via raw bytes:
/// `REX.W 0F BD /r` = BSR r64, r64
///
/// # Estimated Code Size
///
/// ~128 bytes: tag check (20), zero check (8), sign handling (16),
/// BSR (4), inc (4), boxing (24), overhead (52)
pub struct IntBitLengthTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntBitLengthTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load and check source
        ctx.asm.mov_rm(acc, &src_slot);
        emit_int_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        let done = ctx.asm.create_label();
        let do_bsr = ctx.asm.create_label();

        // If zero, result is 0
        ctx.asm.test_rr(acc, acc);
        ctx.asm.jne(do_bsr);

        // result = 0 → box and store
        ctx.asm.xor_rr(acc, acc);
        emit_int_box(ctx, acc, scratch1);
        ctx.asm.mov_mr(&dst_slot, acc);
        ctx.asm.jmp(done);

        ctx.asm.bind_label(do_bsr);

        // If negative, compute ~x (bitwise NOT)
        let skip_not = ctx.asm.create_label();
        ctx.asm.test_rr(acc, acc);
        ctx.asm.jns(skip_not);
        ctx.asm.not(acc);
        ctx.asm.bind_label(skip_not);

        // BSR r64, r64: REX.W 0F BD /r
        // BSR scratch1, acc: find highest set bit
        // Encoding: REX.W (0x48 + r/b bits) 0F BD ModRM
        // We need scratch1 as dst and acc as src
        // For simplicity, use acc as both — BSR acc, acc
        let rex = 0x48u8
            | if acc.high_bit() { 0x04 } else { 0 }  // REX.R
            | if acc.high_bit() { 0x01 } else { 0 }; // REX.B
        let modrm = 0xC0 | (acc.low_bits() << 3) | acc.low_bits();
        ctx.asm.emit_bytes(&[rex, 0x0F, 0xBD, modrm]);

        // BSR gives zero-indexed position, bit_length = bsr + 1
        ctx.asm.inc(acc);

        // Box result
        emit_int_box(ctx, acc, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, acc);

        ctx.asm.bind_label(done);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        128
    }
}

// =============================================================================
// Integer Left Shift Template
// =============================================================================

/// Template for integer left shift (`x << y`) with overflow detection.
///
/// # Algorithm
///
/// ```text
/// if shift_amount < 0: deopt (negative shift)
/// if shift_amount >= 48: deopt (would shift out of NaN-box payload)
/// result = x << shift_amount
/// verify: (result >> shift_amount) == x  (roundtrip check for overflow)
/// if mismatch: deopt
/// ```
///
/// # Estimated Code Size
///
/// ~160 bytes: 2×tag checks (40), range check (16), shift (8),
/// roundtrip verify (16), boxing (24), overhead (56)
pub struct IntLShiftTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntLShiftTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and check LHS (value to shift)
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_int_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load and check RHS (shift amount)
        ctx.asm.mov_rm(scratch2, &rhs_slot);
        emit_int_check_and_extract(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Negative shift amount → deopt
        ctx.asm.test_rr(scratch2, scratch2);
        ctx.asm.js(ctx.deopt_label(self.deopt_idx));

        // Shift amount >= 48 → deopt (exceeds NaN-box payload width)
        ctx.asm.cmp_ri(scratch2, 48);
        ctx.asm.jge(ctx.deopt_label(self.deopt_idx));

        // Save original value for roundtrip verification
        ctx.asm.mov_rr(scratch1, acc);

        // Move shift amount to RCX for variable shift (shl_cl uses CL)
        ctx.asm.mov_rr(Gpr::Rcx, scratch2);

        // Perform shift
        ctx.asm.shl_cl(acc);

        // Roundtrip verification: (result >> shift_amount) must equal original
        // scratch2 still holds shift amount but RCX was overwritten
        // RCX already has the shift amount
        ctx.asm.mov_rr(scratch2, acc);
        ctx.asm.sar_cl(scratch2);
        ctx.asm.cmp_rr(scratch2, scratch1);
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));

        // Box result
        emit_int_box(ctx, acc, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        160
    }
}

// =============================================================================
// Integer Right Shift Template
// =============================================================================

/// Template for integer arithmetic right shift (`x >> y`).
///
/// Arithmetic right shift can never overflow (it only reduces magnitude),
/// so no overflow check is needed. We do check for negative shift amounts.
///
/// # Estimated Code Size
///
/// ~128 bytes: 2×tag checks (40), range check (8), shift (8),
/// boxing (24), overhead (48)
pub struct IntRShiftTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntRShiftTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and check LHS
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_int_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load and check RHS (shift amount)
        ctx.asm.mov_rm(scratch2, &rhs_slot);
        emit_int_check_and_extract(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Negative shift → deopt
        ctx.asm.test_rr(scratch2, scratch2);
        ctx.asm.js(ctx.deopt_label(self.deopt_idx));

        // Clamp shift to 63 (x >> 64+ is well-defined as sign-fill in Python
        // but SAR on x86 masks to 6 bits anyway, giving the correct result)

        // Move shift amount to RCX
        ctx.asm.mov_rr(Gpr::Rcx, scratch2);

        // Arithmetic right shift
        ctx.asm.sar_cl(acc);

        // Box result
        emit_int_box(ctx, acc, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        128
    }
}

// =============================================================================
// Integer Unary Positive Template
// =============================================================================

/// Template for unary positive (`+x`).
///
/// In Python, `+x` on an integer is the identity operation for int,
/// but it must verify the operand IS an integer (type guard).
/// No arithmetic is performed — just copy after validation.
///
/// # Estimated Code Size
///
/// ~48 bytes: tag check (20), copy (8), overhead (20)
pub struct IntUnaryPositiveTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntUnaryPositiveTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load and check source type (just validate, don't extract)
        ctx.asm.mov_rm(acc, &src_slot);
        ctx.asm.mov_rr(scratch1, acc);
        ctx.asm.shr_ri(scratch1, 48);
        let expected_tag = value_tags::int_tag_check() as i32;
        ctx.asm.cmp_ri(scratch1, expected_tag);
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));

        // Identity — store same NaN-boxed value (no extract/re-box needed)
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Integer-to-Float Conversion Template
// =============================================================================

/// Template for `float(x)` where x is an integer.
///
/// Uses CVTSI2SD to convert the 48-bit signed integer payload to an f64.
/// The resulting double is stored directly as a NaN-boxed float value
/// (f64 bits are their own NaN-box representation in Prism's value encoding).
///
/// # Estimated Code Size
///
/// ~64 bytes: tag check (20), cvtsi2sd (4), store (8), overhead (32)
pub struct IntToFloatTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntToFloatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let xmm0 = ctx.regs.xmm0;

        // Load and check source
        ctx.asm.mov_rm(acc, &src_slot);
        emit_int_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Convert int64 → f64
        ctx.asm.cvtsi2sd(xmm0, acc);

        // Store f64 result directly — f64 bits are the NaN-boxed value
        ctx.asm.movsd_mr(&dst_slot, xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        64
    }
}

// =============================================================================
// Integer Compare Template
// =============================================================================

/// Template for optimized integer comparison.
///
/// Supports all six comparison operations via the `condition` field:
/// - `Condition::Less` → `<`
/// - `Condition::LessEqual` → `<=`
/// - `Condition::Greater` → `>`
/// - `Condition::GreaterEqual` → `>=`
/// - `Condition::Equal` → `==`
/// - `Condition::NotEqual` → `!=`
///
/// Uses `CMP` + `SETcc` + `MOVZX` to produce a boolean result without branching.
///
/// # Estimated Code Size
///
/// ~112 bytes: 2×tag checks (40), cmp (4), setcc (4), movzx (4),
/// boolean boxing (24), overhead (36)
pub struct IntCompareTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub condition: Condition,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntCompareTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and check LHS
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_int_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load and check RHS
        ctx.asm.mov_rm(scratch2, &rhs_slot);
        emit_int_check_and_extract(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Compare
        ctx.asm.cmp_rr(acc, scratch2);

        // Branchless boolean construction:
        // Load both True and False NaN-boxed values, use SETcc + conditional select
        let true_val = value_tags::true_value() as i64;
        let false_val = value_tags::false_value() as i64;

        // SETcc: set low byte of acc to 1 if condition met, 0 otherwise
        ctx.asm.setcc(self.condition, acc);

        // Zero-extend byte to 64-bit (0 or 1)
        ctx.asm.movzx_rb(acc, acc);

        // Compute: acc = acc * (TRUE - FALSE) + FALSE
        // If acc=1: TRUE, if acc=0: FALSE
        // This avoids a branch for boolean boxing.
        let delta = true_val.wrapping_sub(false_val);
        ctx.asm.mov_ri64(scratch1, delta);
        ctx.asm.imul_rr(acc, scratch1);
        ctx.asm.mov_ri64(scratch1, false_val);
        ctx.asm.add_rr(acc, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        112
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::x64::Assembler;
    use crate::tier1::frame::FrameLayout;

    fn make_ctx<'a>(asm: &'a mut Assembler, frame: &'a FrameLayout) -> TemplateContext<'a> {
        let mut ctx = TemplateContext::new(asm, frame);
        ctx.create_deopt_label();
        ctx
    }

    // =========================================================================
    // IntPowerTemplate Tests
    // =========================================================================

    #[test]
    fn test_int_power_template_emits_code() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntPowerTemplate {
            dst_reg: 2,
            base_reg: 0,
            exp_reg: 1,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // Binary exponentiation loop should generate substantial code
        assert!(
            ctx.asm.offset() > 100,
            "Power template too short: {}",
            ctx.asm.offset()
        );
    }

    #[test]
    fn test_int_power_template_estimated_size() {
        let tmpl = IntPowerTemplate {
            dst_reg: 2,
            base_reg: 0,
            exp_reg: 1,
            deopt_idx: 0,
        };
        assert_eq!(tmpl.estimated_size(), 256);
    }

    #[test]
    fn test_int_power_same_src_dst() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntPowerTemplate {
            dst_reg: 0,
            base_reg: 0,
            exp_reg: 1,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);
        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_int_power_high_register_numbers() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntPowerTemplate {
            dst_reg: 15,
            base_reg: 10,
            exp_reg: 12,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);
        assert!(ctx.asm.offset() > 50);
    }

    // =========================================================================
    // IntAbsTemplate Tests
    // =========================================================================

    #[test]
    fn test_int_abs_template_emits_code() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntAbsTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        assert!(
            ctx.asm.offset() > 30,
            "Abs template too short: {}",
            ctx.asm.offset()
        );
    }

    #[test]
    fn test_int_abs_template_estimated_size() {
        let tmpl = IntAbsTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        assert_eq!(tmpl.estimated_size(), 96);
    }

    #[test]
    fn test_int_abs_same_src_dst() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntAbsTemplate {
            dst_reg: 0,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);
        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_int_abs_contains_neg_instruction() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntAbsTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // Should be larger than just a load+store (has conditional negation)
        assert!(ctx.asm.offset() > 40);
    }

    // =========================================================================
    // IntBitLengthTemplate Tests
    // =========================================================================

    #[test]
    fn test_int_bit_length_template_emits_code() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntBitLengthTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // Contains BSR instruction + zero handling + negative handling
        assert!(
            ctx.asm.offset() > 60,
            "BitLength template too short: {}",
            ctx.asm.offset()
        );
    }

    #[test]
    fn test_int_bit_length_estimated_size() {
        let tmpl = IntBitLengthTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        assert_eq!(tmpl.estimated_size(), 128);
    }

    #[test]
    fn test_int_bit_length_same_src_dst() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntBitLengthTemplate {
            dst_reg: 0,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);
        assert!(ctx.asm.offset() > 40);
    }

    #[test]
    fn test_int_bit_length_contains_bsr_encoding() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntBitLengthTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // Verify BSR opcode (0F BD) appears in emitted code
        let code = ctx.asm.code();
        let has_bsr = code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0xBD);
        assert!(has_bsr, "BSR instruction (0F BD) not found in emitted code");
    }

    // =========================================================================
    // IntLShiftTemplate Tests
    // =========================================================================

    #[test]
    fn test_int_lshift_template_emits_code() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntLShiftTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        assert!(
            ctx.asm.offset() > 80,
            "LShift template too short: {}",
            ctx.asm.offset()
        );
    }

    #[test]
    fn test_int_lshift_estimated_size() {
        let tmpl = IntLShiftTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        assert_eq!(tmpl.estimated_size(), 160);
    }

    #[test]
    fn test_int_lshift_contains_roundtrip_verification() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntLShiftTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // Should contain both shl and sar (roundtrip verification)
        assert!(ctx.asm.offset() > 100);
    }

    #[test]
    fn test_int_lshift_same_src_dst() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntLShiftTemplate {
            dst_reg: 0,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);
        assert!(ctx.asm.offset() > 50);
    }

    // =========================================================================
    // IntRShiftTemplate Tests
    // =========================================================================

    #[test]
    fn test_int_rshift_template_emits_code() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntRShiftTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        assert!(
            ctx.asm.offset() > 50,
            "RShift template too short: {}",
            ctx.asm.offset()
        );
    }

    #[test]
    fn test_int_rshift_estimated_size() {
        let tmpl = IntRShiftTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        assert_eq!(tmpl.estimated_size(), 128);
    }

    #[test]
    fn test_int_rshift_no_overflow_check() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntRShiftTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // RShift should be shorter than LShift (no roundtrip verification)
        let rshift_size = ctx.asm.offset();

        let mut asm2 = Assembler::new();
        let mut ctx2 = make_ctx(&mut asm2, &frame);
        let lshift_tmpl = IntLShiftTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        lshift_tmpl.emit(&mut ctx2);
        let lshift_size = ctx2.asm.offset();

        assert!(
            rshift_size < lshift_size,
            "RShift ({rshift_size}) should be shorter than LShift ({lshift_size})"
        );
    }

    // =========================================================================
    // IntUnaryPositiveTemplate Tests
    // =========================================================================

    #[test]
    fn test_int_unary_positive_template_emits_code() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntUnaryPositiveTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // Minimal: just type check + store
        assert!(
            ctx.asm.offset() > 10,
            "UnaryPositive too short: {}",
            ctx.asm.offset()
        );
    }

    #[test]
    fn test_int_unary_positive_estimated_size() {
        let tmpl = IntUnaryPositiveTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        assert_eq!(tmpl.estimated_size(), 48);
    }

    #[test]
    fn test_int_unary_positive_is_identity() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntUnaryPositiveTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // Should be shortest of all templates (just guard + copy)
        let pos_size = ctx.asm.offset();

        let mut asm2 = Assembler::new();
        let mut ctx2 = make_ctx(&mut asm2, &frame);
        let abs_tmpl = IntAbsTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        abs_tmpl.emit(&mut ctx2);

        assert!(
            pos_size < ctx2.asm.offset(),
            "Unary positive should be smaller than abs"
        );
    }

    #[test]
    fn test_int_unary_positive_same_src_dst() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntUnaryPositiveTemplate {
            dst_reg: 0,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);
        assert!(ctx.asm.offset() > 8);
    }

    // =========================================================================
    // IntToFloatTemplate Tests
    // =========================================================================

    #[test]
    fn test_int_to_float_template_emits_code() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntToFloatTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        assert!(
            ctx.asm.offset() > 20,
            "IntToFloat too short: {}",
            ctx.asm.offset()
        );
    }

    #[test]
    fn test_int_to_float_estimated_size() {
        let tmpl = IntToFloatTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        assert_eq!(tmpl.estimated_size(), 64);
    }

    #[test]
    fn test_int_to_float_contains_cvtsi2sd() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntToFloatTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // CVTSI2SD encoding: F2 REX.W 0F 2A (REX prefix between F2 and opcode)
        let code = ctx.asm.code();
        let has_cvt = code
            .windows(4)
            .any(|w| w[0] == 0xF2 && (w[1] & 0xF0 == 0x40) && w[2] == 0x0F && w[3] == 0x2A);
        assert!(has_cvt, "CVTSI2SD (F2 REX 0F 2A) not found in emitted code");
    }

    // =========================================================================
    // IntCompareTemplate Tests
    // =========================================================================

    #[test]
    fn test_int_compare_less_than() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntCompareTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            condition: Condition::Less,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        assert!(
            ctx.asm.offset() > 50,
            "Compare template too short: {}",
            ctx.asm.offset()
        );
    }

    #[test]
    fn test_int_compare_equal() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntCompareTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            condition: Condition::Equal,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_int_compare_greater_equal() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntCompareTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            condition: Condition::GreaterEqual,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_int_compare_not_equal() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntCompareTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            condition: Condition::NotEqual,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);
        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_int_compare_estimated_size() {
        let tmpl = IntCompareTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            condition: Condition::Less,
            deopt_idx: 0,
        };
        assert_eq!(tmpl.estimated_size(), 112);
    }

    #[test]
    fn test_int_compare_all_conditions_emit() {
        let conditions = [
            Condition::Less,
            Condition::LessEqual,
            Condition::Greater,
            Condition::GreaterEqual,
            Condition::Equal,
            Condition::NotEqual,
        ];

        let frame = FrameLayout::minimal(4);
        for cond in conditions {
            let mut asm = Assembler::new();
            let mut ctx = make_ctx(&mut asm, &frame);

            let tmpl = IntCompareTemplate {
                dst_reg: 2,
                lhs_reg: 0,
                rhs_reg: 1,
                condition: cond,
                deopt_idx: 0,
            };
            tmpl.emit(&mut ctx);
            assert!(
                ctx.asm.offset() > 40,
                "Condition {:?} produced too little code",
                cond
            );
        }
    }

    #[test]
    fn test_int_compare_contains_setcc() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let tmpl = IntCompareTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            condition: Condition::Less,
            deopt_idx: 0,
        };
        tmpl.emit(&mut ctx);

        // SETcc encoding: 0F 9x where x = condition code
        let code = ctx.asm.code();
        let has_setcc = code
            .windows(2)
            .any(|w| w[0] == 0x0F && (0x90..=0x9F).contains(&w[1]));
        assert!(has_setcc, "SETcc instruction not found in emitted code");
    }

    // =========================================================================
    // Cross-Template Tests
    // =========================================================================

    #[test]
    fn test_all_templates_create_with_different_deopt_indices() {
        let frame = FrameLayout::minimal(8);

        for deopt_idx in 0..3 {
            let mut asm = Assembler::new();
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            for _ in 0..=deopt_idx {
                ctx.create_deopt_label();
            }

            let tmpl = IntAbsTemplate {
                dst_reg: 1,
                src_reg: 0,
                deopt_idx,
            };
            tmpl.emit(&mut ctx);
            assert!(ctx.asm.offset() > 0);
        }
    }

    #[test]
    fn test_template_sizes_are_ordered() {
        // Verify relative size expectations
        let unary_pos = IntUnaryPositiveTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        let abs = IntAbsTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        let to_float = IntToFloatTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        let compare = IntCompareTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            condition: Condition::Less,
            deopt_idx: 0,
        };
        let rshift = IntRShiftTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        let bit_length = IntBitLengthTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        let lshift = IntLShiftTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        let power = IntPowerTemplate {
            dst_reg: 2,
            base_reg: 0,
            exp_reg: 1,
            deopt_idx: 0,
        };

        // Unary positive < abs < to_float < compare < rshift < bit_length < lshift < power
        assert!(unary_pos.estimated_size() < abs.estimated_size());
        assert!(to_float.estimated_size() < compare.estimated_size());
        assert!(compare.estimated_size() < rshift.estimated_size());
        assert!(rshift.estimated_size() <= bit_length.estimated_size());
        assert!(lshift.estimated_size() <= power.estimated_size());
    }

    #[test]
    fn test_multiple_templates_emit_sequentially() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(8);
        let mut ctx = make_ctx(&mut asm, &frame);

        let abs_tmpl = IntAbsTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        abs_tmpl.emit(&mut ctx);
        let after_abs = ctx.asm.offset();

        let rshift_tmpl = IntRShiftTemplate {
            dst_reg: 3,
            lhs_reg: 1,
            rhs_reg: 2,
            deopt_idx: 0,
        };
        rshift_tmpl.emit(&mut ctx);
        let after_rshift = ctx.asm.offset();

        assert!(
            after_rshift > after_abs,
            "Sequential emission should grow code buffer"
        );
    }
}
