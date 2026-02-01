//! Object operation templates - attribute/item access, iterators, len.
//!
//! These templates handle Python object operations. Most require runtime
//! support and will deoptimize to the interpreter.

use super::{OpcodeTemplate, TemplateContext};

// =============================================================================
// Attribute Access
// =============================================================================

/// Template for getting an attribute.
pub struct GetAttrTemplate {
    pub dst_reg: u8,
    pub obj_reg: u8,
    pub name_idx: u16,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GetAttrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Attribute access requires runtime lookup - deopt for now
        // Full implementation would use inline cache
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for setting an attribute.
pub struct SetAttrTemplate {
    pub obj_reg: u8,
    pub name_idx: u16,
    pub value_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for SetAttrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for deleting an attribute.
pub struct DelAttrTemplate {
    pub obj_reg: u8,
    pub name_idx: u16,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for DelAttrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Item Access
// =============================================================================

/// Template for getting an item (indexing).
pub struct GetItemTemplate {
    pub dst_reg: u8,
    pub obj_reg: u8,
    pub key_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GetItemTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Item access requires type dispatch - deopt
        // Full impl would have fast paths for list/dict with int/str keys
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for setting an item.
pub struct SetItemTemplate {
    pub obj_reg: u8,
    pub key_reg: u8,
    pub value_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for SetItemTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for deleting an item.
pub struct DelItemTemplate {
    pub obj_reg: u8,
    pub key_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for DelItemTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Iterator Operations
// =============================================================================

/// Template for getting an iterator.
pub struct GetIterTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GetIterTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Iterator protocol requires runtime - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for for-loop iteration.
pub struct ForIterTemplate {
    pub dst_reg: u8,
    pub iter_reg: u8,
    pub end_label: crate::backend::x64::Label,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for ForIterTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // For iteration is complex - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Length and Callable Check
// =============================================================================

/// Template for getting length.
pub struct LenTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for LenTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // len() requires type dispatch - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for checking if callable.
pub struct IsCallableTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IsCallableTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Callable check requires type inspection - deopt
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
    fn test_get_attr_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GetAttrTemplate {
            dst_reg: 1,
            obj_reg: 0,
            name_idx: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_get_item_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GetItemTemplate {
            dst_reg: 2,
            obj_reg: 0,
            key_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_get_iter_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GetIterTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }
}
