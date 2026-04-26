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
