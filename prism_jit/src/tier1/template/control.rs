//! Control flow templates - jumps, branches, returns, exceptions.
//!
//! These templates handle control flow between basic blocks in JIT code.

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::{Gpr, Label};

// =============================================================================
// Unconditional Jump
// =============================================================================

/// Template for unconditional jump.
pub struct JumpTemplate {
    pub target: Label,
}

impl OpcodeTemplate for JumpTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(self.target);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        5 // jmp rel32
    }
}

// =============================================================================
// Conditional Branches
// =============================================================================

/// Template for branch if value is truthy.
pub struct BranchIfTrueTemplate {
    pub condition_reg: u8,
    pub target: Label,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BranchIfTrueTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let cond_slot = ctx.frame.register_slot(self.condition_reg as u16);
        let acc = ctx.regs.accumulator;

        // Load condition value
        ctx.asm.mov_rm(acc, &cond_slot);

        // Check for True
        let true_val = value_tags::true_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.scratch1, true_val);
        ctx.asm.cmp_rr(acc, ctx.regs.scratch1);
        ctx.asm.je(self.target);

        // Check for False (if not True and not False, deopt)
        let false_val = value_tags::false_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.scratch1, false_val);
        ctx.asm.cmp_rr(acc, ctx.regs.scratch1);
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));

        // Fall through if False
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

/// Template for branch if value is falsy.
pub struct BranchIfFalseTemplate {
    pub condition_reg: u8,
    pub target: Label,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for BranchIfFalseTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let cond_slot = ctx.frame.register_slot(self.condition_reg as u16);
        let acc = ctx.regs.accumulator;

        ctx.asm.mov_rm(acc, &cond_slot);

        // Check for False
        let false_val = value_tags::false_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.scratch1, false_val);
        ctx.asm.cmp_rr(acc, ctx.regs.scratch1);
        ctx.asm.je(self.target);

        // Check for True (if not True and not False, deopt)
        let true_val = value_tags::true_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.scratch1, true_val);
        ctx.asm.cmp_rr(acc, ctx.regs.scratch1);
        ctx.asm.jne(ctx.deopt_label(self.deopt_idx));

        // Fall through if True
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

/// Template for branch if value is None.
pub struct BranchIfNoneTemplate {
    pub condition_reg: u8,
    pub target: Label,
}

impl OpcodeTemplate for BranchIfNoneTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let cond_slot = ctx.frame.register_slot(self.condition_reg as u16);
        let acc = ctx.regs.accumulator;

        ctx.asm.mov_rm(acc, &cond_slot);

        // Check for None
        let none_val = value_tags::none_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.scratch1, none_val);
        ctx.asm.cmp_rr(acc, ctx.regs.scratch1);
        ctx.asm.je(self.target);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        28
    }
}

/// Template for branch if value is not None.
pub struct BranchIfNotNoneTemplate {
    pub condition_reg: u8,
    pub target: Label,
}

impl OpcodeTemplate for BranchIfNotNoneTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let cond_slot = ctx.frame.register_slot(self.condition_reg as u16);
        let acc = ctx.regs.accumulator;

        ctx.asm.mov_rm(acc, &cond_slot);

        // Check for not None
        let none_val = value_tags::none_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.scratch1, none_val);
        ctx.asm.cmp_rr(acc, ctx.regs.scratch1);
        ctx.asm.jne(self.target);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        28
    }
}

// =============================================================================
// Return Templates
// =============================================================================

/// Template for returning a value.
pub struct ReturnTemplate {
    pub value_reg: u8,
}

impl OpcodeTemplate for ReturnTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let value_slot = ctx.frame.register_slot(self.value_reg as u16);

        // Load return value into RAX (calling convention)
        ctx.asm.mov_rm(Gpr::Rax, &value_slot);

        // Epilogue handled by caller
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for returning None.
pub struct ReturnNoneTemplate;

impl OpcodeTemplate for ReturnNoneTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Load None into RAX
        let none_val = value_tags::none_value() as i64;
        ctx.asm.mov_ri64(Gpr::Rax, none_val);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        10
    }
}

// =============================================================================
// No Operation
// =============================================================================

/// Template for no operation.
pub struct NopTemplate;

impl OpcodeTemplate for NopTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.nop();
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        1
    }
}

// =============================================================================
// Exception Handling (Placeholder - would need runtime support)
// =============================================================================

/// Template for raising an exception.
pub struct RaiseTemplate {
    pub exception_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for RaiseTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Exceptions always deopt to interpreter for proper handling
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for re-raising the current exception.
pub struct ReraiseTemplate {
    pub deopt_idx: usize,
}

impl OpcodeTemplate for ReraiseTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for pop exception handler.
pub struct PopExceptHandlerTemplate {
    pub deopt_idx: usize,
}

impl OpcodeTemplate for PopExceptHandlerTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Exception handling in JIT requires complex runtime support
        // Deopt for now
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for end finally block.
pub struct EndFinallyTemplate {
    pub deopt_idx: usize,
}

impl OpcodeTemplate for EndFinallyTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Generator Templates (Placeholder)
// =============================================================================

/// Template for yield.
pub struct YieldTemplate {
    pub value_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for YieldTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Generators require complex state management - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for yield from.
pub struct YieldFromTemplate {
    pub iterator_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for YieldFromTemplate {
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
        ctx.create_deopt_label();
        ctx
    }

    #[test]
    fn test_jump_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let target = ctx.asm.create_label();
        ctx.asm.bind_label(target);

        let template = JumpTemplate { target };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_branch_if_true_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let target = ctx.asm.create_label();

        let template = BranchIfTrueTemplate {
            condition_reg: 0,
            target,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_branch_if_false_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let target = ctx.asm.create_label();

        let template = BranchIfFalseTemplate {
            condition_reg: 0,
            target,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_branch_if_none_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let target = ctx.asm.create_label();

        let template = BranchIfNoneTemplate {
            condition_reg: 0,
            target,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_return_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = ReturnTemplate { value_reg: 0 };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_return_none_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = ReturnNoneTemplate;
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_nop_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = NopTemplate;
        template.emit(&mut ctx);

        assert_eq!(ctx.asm.offset(), 1);
    }
}
