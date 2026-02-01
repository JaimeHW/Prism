//! Value loading templates - constants, None, True, False.
//!
//! These templates handle loading immediate values into bytecode registers.
//! All values are NaN-boxed according to the prism_core value representation.

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::{Gpr, MemOperand};

// =============================================================================
// Load Integer
// =============================================================================

/// Template for loading an immediate integer.
pub struct LoadIntTemplate {
    pub dst_reg: u8,
    pub value: i64,
}

impl OpcodeTemplate for LoadIntTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Create NaN-boxed integer
        let boxed = value_tags::box_int(self.value);
        ctx.asm.mov_ri64(ctx.regs.accumulator, boxed as i64);

        let slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        18 // mov imm64 + mov [rbp+off], r64
    }
}

// =============================================================================
// Load Float
// =============================================================================

/// Template for loading a floating-point constant.
///
/// Float values are stored directly in NaN-boxed format.
/// Regular floats are stored as-is (they're already valid NaN-box values).
pub struct LoadFloatTemplate {
    pub dst_reg: u8,
    pub value: f64,
}

impl OpcodeTemplate for LoadFloatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Floats are stored directly (NaN-boxing uses the float representation)
        let bits = self.value.to_bits();
        ctx.asm.mov_ri64(ctx.regs.accumulator, bits as i64);

        let slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        18
    }
}

// =============================================================================
// Load None
// =============================================================================

/// Template for loading None.
pub struct LoadNoneTemplate {
    pub dst_reg: u8,
}

impl OpcodeTemplate for LoadNoneTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let none_val = value_tags::none_value() as i64;
        ctx.asm.mov_ri64(ctx.regs.accumulator, none_val);

        let slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        18
    }
}

// =============================================================================
// Load Boolean
// =============================================================================

/// Template for loading True or False.
pub struct LoadBoolTemplate {
    pub dst_reg: u8,
    pub value: bool,
}

impl OpcodeTemplate for LoadBoolTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let bool_value = if self.value {
            value_tags::true_value()
        } else {
            value_tags::false_value()
        };
        ctx.asm.mov_ri64(ctx.regs.accumulator, bool_value as i64);

        let slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        18
    }
}

// =============================================================================
// Load Constant from Pool
// =============================================================================

/// Template for loading a constant from the constant pool.
///
/// The constant pool is accessed via a base pointer stored in the frame.
pub struct LoadConstTemplate {
    pub dst_reg: u8,
    pub const_idx: u16,
}

impl OpcodeTemplate for LoadConstTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Load constant pool base
        let const_pool_slot = ctx.frame.const_pool_slot();
        ctx.asm.mov_rm(ctx.regs.scratch1, &const_pool_slot);

        // Load constant at index (each constant is 8 bytes)
        let offset = (self.const_idx as i32) * 8;
        let const_mem = MemOperand::base_disp(ctx.regs.scratch1, offset);
        ctx.asm.mov_rm(ctx.regs.accumulator, &const_mem);

        // Store to destination register
        let slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&slot, ctx.regs.accumulator);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        24 // Load base + load indexed + store
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

    #[test]
    fn test_load_int_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = LoadIntTemplate {
            dst_reg: 0,
            value: 42,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_load_float_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = LoadFloatTemplate {
            dst_reg: 0,
            value: 3.14,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_load_none_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = LoadNoneTemplate { dst_reg: 1 };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_load_bool_true_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = LoadBoolTemplate {
            dst_reg: 0,
            value: true,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_load_bool_false_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = LoadBoolTemplate {
            dst_reg: 0,
            value: false,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_load_const_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = LoadConstTemplate {
            dst_reg: 0,
            const_idx: 5,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }
}
