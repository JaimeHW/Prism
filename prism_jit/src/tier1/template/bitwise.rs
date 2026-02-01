//! Bitwise and logical operation templates.
//!
//! Templates for bitwise operations (AND, OR, XOR, NOT, shifts)
//! and logical negation.

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::{Condition, Gpr};

// =============================================================================
// Helper: Emit bitwise binary operation
// =============================================================================

/// Operation type for bitwise operations.
#[derive(Clone, Copy)]
pub enum BitwiseOp {
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

fn emit_bitwise_binary(
    ctx: &mut TemplateContext,
    dst_reg: u8,
    lhs_reg: u8,
    rhs_reg: u8,
    op: BitwiseOp,
    deopt_idx: usize,
) {
    let lhs_slot = ctx.frame.register_slot(lhs_reg as u16);
    let rhs_slot = ctx.frame.register_slot(rhs_reg as u16);
    let dst_slot = ctx.frame.register_slot(dst_reg as u16);

    let acc = ctx.regs.accumulator;
    let scratch1 = ctx.regs.scratch1;
    let scratch2 = ctx.regs.scratch2;

    // Load and check LHS type
    ctx.asm.mov_rm(acc, &lhs_slot);
    ctx.asm.mov_rr(scratch1, acc);
    ctx.asm.shr_ri(scratch1, 48);
    let expected_tag = value_tags::int_tag_check() as i32;
    ctx.asm.cmp_ri(scratch1, expected_tag);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));

    // Load and check RHS type
    ctx.asm.mov_rm(scratch2, &rhs_slot);
    ctx.asm.mov_rr(scratch1, scratch2);
    ctx.asm.shr_ri(scratch1, 48);
    ctx.asm.cmp_ri(scratch1, expected_tag);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));

    // Extract payloads
    ctx.asm.shl_ri(acc, 16);
    ctx.asm.sar_ri(acc, 16);
    ctx.asm.shl_ri(scratch2, 16);
    ctx.asm.sar_ri(scratch2, 16);

    // Apply operation
    match op {
        BitwiseOp::And => ctx.asm.and_rr(acc, scratch2),
        BitwiseOp::Or => ctx.asm.or_rr(acc, scratch2),
        BitwiseOp::Xor => ctx.asm.xor_rr(acc, scratch2),
        BitwiseOp::Shl => {
            // Shift amount in CL
            ctx.asm.mov_rr(Gpr::Rcx, scratch2);
            ctx.asm.shl_cl(acc);
        }
        BitwiseOp::Shr => {
            ctx.asm.mov_rr(Gpr::Rcx, scratch2);
            ctx.asm.sar_cl(acc);
        }
    }

    // Box result
    ctx.asm.mov_ri64(scratch1, value_tags::PAYLOAD_MASK as i64);
    ctx.asm.and_rr(acc, scratch1);
    let tag = (value_tags::QNAN_BITS | value_tags::INT_TAG) as i64;
    ctx.asm.mov_ri64(scratch1, tag);
    ctx.asm.or_rr(acc, scratch1);

    // Store result
    ctx.asm.mov_mr(&dst_slot, acc);
}

// =============================================================================
// Bitwise AND
// =============================================================================

/// Template for bitwise AND.
pub struct BitwiseAndTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BitwiseAndTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_bitwise_binary(
            ctx,
            self.dst_reg,
            self.lhs_reg,
            self.rhs_reg,
            BitwiseOp::And,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// Bitwise OR
// =============================================================================

/// Template for bitwise OR.
pub struct BitwiseOrTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BitwiseOrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_bitwise_binary(
            ctx,
            self.dst_reg,
            self.lhs_reg,
            self.rhs_reg,
            BitwiseOp::Or,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// Bitwise XOR
// =============================================================================

/// Template for bitwise XOR.
pub struct BitwiseXorTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BitwiseXorTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_bitwise_binary(
            ctx,
            self.dst_reg,
            self.lhs_reg,
            self.rhs_reg,
            BitwiseOp::Xor,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// Bitwise NOT
// =============================================================================

/// Template for bitwise NOT.
pub struct BitwiseNotTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BitwiseNotTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load and check type
        ctx.asm.mov_rm(acc, &src_slot);
        ctx.asm.mov_rr(scratch1, acc);
        ctx.asm.shr_ri(scratch1, 48);
        let expected_tag = value_tags::int_tag_check() as i32;
        ctx.asm.cmp_ri(scratch1, expected_tag);
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));

        // Extract payload
        ctx.asm.shl_ri(acc, 16);
        ctx.asm.sar_ri(acc, 16);

        // NOT
        ctx.asm.not(acc);

        // Box result
        ctx.asm.mov_ri64(scratch1, value_tags::PAYLOAD_MASK as i64);
        ctx.asm.and_rr(acc, scratch1);
        let tag = (value_tags::QNAN_BITS | value_tags::INT_TAG) as i64;
        ctx.asm.mov_ri64(scratch1, tag);
        ctx.asm.or_rr(acc, scratch1);

        // Store result
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80
    }
}

// =============================================================================
// Left Shift
// =============================================================================

/// Template for left shift.
pub struct ShlTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for ShlTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_bitwise_binary(
            ctx,
            self.dst_reg,
            self.lhs_reg,
            self.rhs_reg,
            BitwiseOp::Shl,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        128
    }
}

// =============================================================================
// Right Shift
// =============================================================================

/// Template for arithmetic right shift.
pub struct ShrTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for ShrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_bitwise_binary(
            ctx,
            self.dst_reg,
            self.lhs_reg,
            self.rhs_reg,
            BitwiseOp::Shr,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        128
    }
}

// =============================================================================
// Logical NOT
// =============================================================================

/// Template for logical NOT.
pub struct NotTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for NotTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load value
        ctx.asm.mov_rm(acc, &src_slot);

        // Check for True
        let true_val = value_tags::true_value() as i64;
        ctx.asm.mov_ri64(scratch1, true_val);
        ctx.asm.cmp_rr(acc, scratch1);

        // If True, result is False
        let false_val = value_tags::false_value() as i64;
        let done_label = ctx.asm.create_label();
        ctx.asm.mov_ri64(acc, true_val); // Default to True
        ctx.asm.jne(done_label); // If not equal to True, skip to done
        ctx.asm.mov_ri64(acc, false_val); // If was True, set to False
        ctx.asm.bind_label(done_label);

        // Check for False
        let skip_label = ctx.asm.create_label();
        ctx.asm.mov_ri64(scratch1, true_val);
        ctx.asm.cmp_rr(acc, scratch1);
        ctx.asm.je(skip_label);

        // If neither True nor False, deopt
        ctx.asm.mov_rm(acc, &src_slot);
        ctx.asm.mov_ri64(scratch1, false_val);
        ctx.asm.cmp_rr(acc, scratch1);
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));

        // Was False, result is True
        ctx.asm.mov_ri64(acc, true_val);

        ctx.asm.bind_label(skip_label);

        // Store result
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        64
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

    #[test]
    fn test_bitwise_and_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = BitwiseAndTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_bitwise_or_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = BitwiseOrTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_bitwise_not_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = BitwiseNotTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 30);
    }

    #[test]
    fn test_shl_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = ShlTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }
}
