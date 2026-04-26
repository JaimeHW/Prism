use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;

fn make_ctx<'a>(asm: &'a mut Assembler, frame: &'a FrameLayout) -> TemplateContext<'a> {
    let mut ctx = TemplateContext::new(asm, frame);
    ctx.create_deopt_label();
    ctx
}

#[test]
fn test_lt_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let template = LtTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 50);
}

#[test]
fn test_eq_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let template = EqTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
        deopt_idx: 0,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 50);
}

#[test]
fn test_is_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = IsTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 20);
}

#[test]
fn test_is_not_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = IsNotTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 20);
}

#[test]
fn test_float_lt_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = FloatLtTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 20);
}

#[test]
fn test_float_le_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = FloatLeTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 20);
}

#[test]
fn test_float_gt_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = FloatGtTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 20);
}

#[test]
fn test_float_ge_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = FloatGeTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 20);
}

#[test]
fn test_float_eq_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = FloatEqTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
    };
    template.emit(&mut ctx);

    // Equality has extra NaN handling - should be larger
    assert!(ctx.asm.offset() > 30);
}

#[test]
fn test_float_ne_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = FloatNeTemplate {
        dst_reg: 2,
        lhs_reg: 0,
        rhs_reg: 1,
    };
    template.emit(&mut ctx);

    // Not-equal has extra NaN handling - should be larger
    assert!(ctx.asm.offset() > 30);
}
