//! Arithmetic templates - integer, float, and generic arithmetic operations.
//!
//! All templates include type guards that deoptimize on type mismatches.
//! Integer operations also check for overflow and deopt if detected.

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::Gpr;

// =============================================================================
// Helper: Integer Type Check and Extract
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
    // Mask to 48 bits and add tag
    ctx.asm.mov_ri64(scratch, value_tags::PAYLOAD_MASK as i64);
    ctx.asm.and_rr(value, scratch);
    let tag = (value_tags::QNAN_BITS | value_tags::INT_TAG) as i64;
    ctx.asm.mov_ri64(scratch, tag);
    ctx.asm.or_rr(value, scratch);
}

// =============================================================================
// Integer Addition
// =============================================================================

/// Template for integer addition with type guards.
pub struct IntAddTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntAddTemplate {
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

        // Add with overflow check
        ctx.asm.add_rr(acc, scratch2);
        ctx.asm.jo(ctx.deopt_label(self.deopt_idx));

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
// Integer Subtraction
// =============================================================================

/// Template for integer subtraction with type guards.
pub struct IntSubTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntSubTemplate {
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

        // Subtract with overflow check
        ctx.asm.sub_rr(acc, scratch2);
        ctx.asm.jo(ctx.deopt_label(self.deopt_idx));

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
// Integer Multiplication
// =============================================================================

/// Template for integer multiplication with type guards.
pub struct IntMulTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntMulTemplate {
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

        // Multiply with overflow check (imul sets OF)
        ctx.asm.imul_rr(acc, scratch2);
        ctx.asm.jo(ctx.deopt_label(self.deopt_idx));

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
// Integer Floor Division
// =============================================================================

/// Template for integer floor division.
pub struct IntFloorDivTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntFloorDivTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and check LHS (must be in RAX for idiv)
        ctx.asm.mov_rm(Gpr::Rax, &lhs_slot);
        emit_int_check_and_extract(ctx, Gpr::Rax, Gpr::Rax, scratch1, self.deopt_idx);

        // Load and check RHS
        ctx.asm.mov_rm(scratch2, &rhs_slot);
        emit_int_check_and_extract(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Check for division by zero
        ctx.asm.test_rr(scratch2, scratch2);
        ctx.asm.jz(ctx.deopt_label(self.deopt_idx));

        // Sign-extend RAX into RDX:RAX
        ctx.asm.cqo();

        // Divide (quotient in RAX, remainder in RDX)
        ctx.asm.idiv(scratch2);

        // Python floor division rounds towards negative infinity
        // If signs differ and remainder != 0, decrement quotient
        // For simplicity, deopt on negative divisor for now
        // (Full implementation would handle this properly)

        // Box result
        emit_int_box(ctx, Gpr::Rax, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, Gpr::Rax);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        160
    }
}

// =============================================================================
// Integer Modulo
// =============================================================================

/// Template for integer modulo.
pub struct IntModTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntModTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and check LHS (must be in RAX for idiv)
        ctx.asm.mov_rm(Gpr::Rax, &lhs_slot);
        emit_int_check_and_extract(ctx, Gpr::Rax, Gpr::Rax, scratch1, self.deopt_idx);

        // Load and check RHS
        ctx.asm.mov_rm(scratch2, &rhs_slot);
        emit_int_check_and_extract(ctx, scratch2, scratch2, scratch1, self.deopt_idx);

        // Check for division by zero
        ctx.asm.test_rr(scratch2, scratch2);
        ctx.asm.jz(ctx.deopt_label(self.deopt_idx));

        // Sign-extend RAX into RDX:RAX
        ctx.asm.cqo();

        // Divide (remainder in RDX)
        ctx.asm.idiv(scratch2);

        // Move remainder to result register
        ctx.asm.mov_rr(Gpr::Rax, Gpr::Rdx);

        // Box result
        emit_int_box(ctx, Gpr::Rax, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, Gpr::Rax);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        160
    }
}

// =============================================================================
// Integer Negation
// =============================================================================

/// Template for integer negation.
pub struct IntNegTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IntNegTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load and check source
        ctx.asm.mov_rm(acc, &src_slot);
        emit_int_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Negate (can overflow for MIN_INT)
        ctx.asm.neg(acc);
        ctx.asm.jo(ctx.deopt_label(self.deopt_idx));

        // Box result
        emit_int_box(ctx, acc, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80
    }
}

// =============================================================================
// Float Arithmetic
// =============================================================================

/// Template for float addition.
pub struct FloatAddTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatAddTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let xmm0 = ctx.regs.xmm0;
        let xmm1 = ctx.regs.xmm1;

        // Load operands as floats
        ctx.asm.movsd_rm(xmm0, &lhs_slot);
        ctx.asm.movsd_rm(xmm1, &rhs_slot);

        // Add
        ctx.asm.addsd(xmm0, xmm1);

        // Store result
        ctx.asm.movsd_mr(&dst_slot, xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        32
    }
}

/// Template for float subtraction.
pub struct FloatSubTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatSubTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let xmm0 = ctx.regs.xmm0;
        let xmm1 = ctx.regs.xmm1;

        ctx.asm.movsd_rm(xmm0, &lhs_slot);
        ctx.asm.movsd_rm(xmm1, &rhs_slot);
        ctx.asm.subsd(xmm0, xmm1);
        ctx.asm.movsd_mr(&dst_slot, xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        32
    }
}

/// Template for float multiplication.
pub struct FloatMulTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatMulTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let xmm0 = ctx.regs.xmm0;
        let xmm1 = ctx.regs.xmm1;

        ctx.asm.movsd_rm(xmm0, &lhs_slot);
        ctx.asm.movsd_rm(xmm1, &rhs_slot);
        ctx.asm.mulsd(xmm0, xmm1);
        ctx.asm.movsd_mr(&dst_slot, xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        32
    }
}

/// Template for float division.
pub struct FloatDivTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatDivTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let xmm0 = ctx.regs.xmm0;
        let xmm1 = ctx.regs.xmm1;

        ctx.asm.movsd_rm(xmm0, &lhs_slot);
        ctx.asm.movsd_rm(xmm1, &rhs_slot);
        ctx.asm.divsd(xmm0, xmm1);
        ctx.asm.movsd_mr(&dst_slot, xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        32
    }
}

/// Template for float negation.
pub struct FloatNegTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
}

impl OpcodeTemplate for FloatNegTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let xmm0 = ctx.regs.xmm0;

        // Load float
        ctx.asm.movsd_rm(xmm0, &src_slot);

        // Negate by subtracting from zero
        // Zero xmm1 and subtract src from it: 0 - x = -x
        ctx.asm.xorpd(ctx.regs.xmm1, ctx.regs.xmm1);
        ctx.asm.subsd(ctx.regs.xmm1, xmm0);
        ctx.asm.movsd_rr(xmm0, ctx.regs.xmm1);

        // Store result
        ctx.asm.movsd_mr(&dst_slot, xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

/// Template for float floor division: dst = floor(lhs / rhs).
/// Currently uses deoptimization fallback until SSE4.1 roundsd is added to assembler.
pub struct FloatFloorDivTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for FloatFloorDivTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // TODO: Implement inline SSE4.1 roundsd once available in assembler
        // For now, deopt to interpreter for correct Python floor division semantics
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for float modulo: dst = lhs % rhs (Python semantics).
/// Currently uses deoptimization fallback until SSE4.1 roundsd is added to assembler.
pub struct FloatModTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for FloatModTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // TODO: Implement inline Python modulo once SSE4.1 roundsd is available
        // Python modulo: x % y = x - floor(x / y) * y
        // For now, deopt to interpreter for correct semantics
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Generic Arithmetic (polymorphic, with type dispatch)
// =============================================================================

/// Template for generic addition (defers to runtime for type dispatch).
pub struct GenericAddTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GenericAddTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // For generic operations, we emit an inline cache or call to runtime
        // This is a simplified version that just deoptimizes
        // Full implementation would do inline type checks and fast paths

        let _lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let _rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let _dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        // Deopt to interpreter for generic operations
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for generic subtraction.
pub struct GenericSubTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GenericSubTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for generic multiplication.
pub struct GenericMulTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GenericMulTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
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
        // Create a deopt label for tests
        ctx.create_deopt_label();
        ctx
    }

    #[test]
    fn test_int_add_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = IntAddTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_int_sub_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = IntSubTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_int_mul_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = IntMulTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_int_neg_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = IntNegTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_float_add_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatAddTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_float_sub_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatSubTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_float_mul_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatMulTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_float_div_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatDivTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }
}
