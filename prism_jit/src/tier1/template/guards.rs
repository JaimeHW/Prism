//! Type guard templates - verify types at runtime and deoptimize on mismatch.
//!
//! Guards check that values have expected types before specialized operations.
//! On type mismatch, they branch to deoptimization stubs.

use super::{OpcodeTemplate, TemplateContext, value_tags};

// =============================================================================
// Type Guards
// =============================================================================

/// Template for integer type guard.
///
/// Checks that the value in the given register is an integer.
/// Deoptimizes if the type tag doesn't match.
pub struct GuardIntTemplate {
    pub reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GuardIntTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let slot = ctx.frame.register_slot(self.reg as u16);
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load value
        ctx.asm.mov_rm(acc, &slot);

        // Extract type tag (upper 16 bits)
        ctx.asm.mov_rr(scratch1, acc);
        ctx.asm.shr_ri(scratch1, 48);

        // Compare to INT_TAG_CHECK
        let expected_tag = value_tags::int_tag_check() as i32;
        ctx.asm.cmp_ri(scratch1, expected_tag);

        // Deoptimize if not equal
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        32
    }
}

/// Template for float type guard.
///
/// Checks that the value is a valid float (not NaN-boxed special value).
/// For floats, we check that either:
/// - Upper bits are NOT the quiet NaN pattern (raw float)
/// - OR upper bits match the QNAN pattern but lower bits are all zero (infinity)
pub struct GuardFloatTemplate {
    pub reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GuardFloatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let slot = ctx.frame.register_slot(self.reg as u16);
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load value
        ctx.asm.mov_rm(acc, &slot);

        // Check if this is a NaN-boxed value (has QNAN_BITS in upper 16)
        ctx.asm.mov_rr(scratch1, acc);
        ctx.asm.shr_ri(scratch1, 48);

        // If upper 16 bits don't have the QNAN pattern, it's a valid float
        let qnan_check = (value_tags::QNAN_BITS >> 48) as i32;
        ctx.asm.cmp_ri(scratch1, qnan_check);

        // If upper bits < QNAN pattern (unsigned comparison), it's a float
        // Use jb (jump if below) to check CF=1
        let ok_label = ctx.asm.create_label();
        ctx.asm.jb(ok_label);

        // It might be a NaN-boxed value - check if it's not a special tag
        // We need to reject int, bool, none, and other tagged values
        // For now, deopt if we hit the QNAN pattern (conservative)
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));

        ctx.asm.bind_label(ok_label);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        40
    }
}

/// Template for boolean type guard.
///
/// Checks that the value is either True or False.
pub struct GuardBoolTemplate {
    pub reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GuardBoolTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let slot = ctx.frame.register_slot(self.reg as u16);
        let acc = ctx.regs.accumulator;

        // Load value
        ctx.asm.mov_rm(acc, &slot);

        // Check if equal to True
        let true_val = value_tags::true_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.scratch1, true_val);
        ctx.asm.cmp_rr(acc, ctx.regs.scratch1);

        let ok_label = ctx.asm.create_label();
        ctx.asm.je(ok_label);

        // Check if equal to False
        let false_val = value_tags::false_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.scratch1, false_val);
        ctx.asm.cmp_rr(acc, ctx.regs.scratch1);
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));

        ctx.asm.bind_label(ok_label);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        64
    }
}

/// Template for None type guard.
///
/// Checks that the value is None.
pub struct GuardNoneTemplate {
    pub reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GuardNoneTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let slot = ctx.frame.register_slot(self.reg as u16);
        let acc = ctx.regs.accumulator;

        // Load value
        ctx.asm.mov_rm(acc, &slot);

        // Check if equal to None
        let none_val = value_tags::none_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.scratch1, none_val);
        ctx.asm.cmp_rr(acc, ctx.regs.scratch1);
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        32
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
    fn test_guard_int_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(2);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GuardIntTemplate {
            reg: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        // Should generate code
        assert!(ctx.asm.offset() > 10);
    }

    #[test]
    fn test_guard_float_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(2);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GuardFloatTemplate {
            reg: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 10);
    }

    #[test]
    fn test_guard_bool_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(2);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GuardBoolTemplate {
            reg: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_guard_none_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(2);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GuardNoneTemplate {
            reg: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 10);
    }
}
