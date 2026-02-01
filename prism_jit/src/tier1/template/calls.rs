//! Function call templates - call, make function, make closure.
//!
//! These templates handle Python function calls. Most require runtime
//! support for argument handling and frame management.

use super::{OpcodeTemplate, TemplateContext};

// =============================================================================
// Function Calls
// =============================================================================

/// Template for calling a function.
pub struct CallTemplate {
    pub dst_reg: u8,
    pub func_reg: u8,
    pub arg_count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for CallTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Function calls require frame management - deopt
        // Full implementation would inline simple cases
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for calling with keyword arguments.
pub struct CallKwTemplate {
    pub dst_reg: u8,
    pub func_reg: u8,
    pub arg_count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for CallKwTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for calling a method.
pub struct CallMethodTemplate {
    pub dst_reg: u8,
    pub obj_reg: u8,
    pub name_idx: u16,
    pub arg_count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for CallMethodTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for tail call optimization.
pub struct TailCallTemplate {
    pub func_reg: u8,
    pub arg_count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for TailCallTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Tail call requires stack manipulation - deopt for now
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Function Creation
// =============================================================================

/// Template for making a function from a code object.
pub struct MakeFunctionTemplate {
    pub dst_reg: u8,
    pub code_idx: u16,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for MakeFunctionTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Function creation requires allocation - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for making a closure.
pub struct MakeClosureTemplate {
    pub dst_reg: u8,
    pub code_idx: u16,
    pub capture_count: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for MakeClosureTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Closure creation requires allocation and capture - deopt
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
    fn test_call_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = CallTemplate {
            dst_reg: 0,
            func_reg: 1,
            arg_count: 2,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_make_function_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = MakeFunctionTemplate {
            dst_reg: 0,
            code_idx: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_make_closure_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = MakeClosureTemplate {
            dst_reg: 0,
            code_idx: 0,
            capture_count: 2,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }
}
