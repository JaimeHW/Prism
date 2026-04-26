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
