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
