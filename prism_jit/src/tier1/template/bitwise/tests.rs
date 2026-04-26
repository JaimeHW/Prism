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
