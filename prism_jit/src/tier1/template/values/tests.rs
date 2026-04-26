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
        value: 3.125,
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
