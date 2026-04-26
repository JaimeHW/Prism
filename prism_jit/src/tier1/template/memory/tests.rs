use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;

#[test]
fn test_move_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = MoveTemplate {
        src_reg: 0,
        dst_reg: 1,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_move_self_noop() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = MoveTemplate {
        src_reg: 0,
        dst_reg: 0,
    };
    template.emit(&mut ctx);

    // Self-move should produce no code
    assert_eq!(ctx.asm.offset(), 0);
}

#[test]
fn test_load_local_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = LoadLocalTemplate {
        dst_reg: 0,
        local_idx: 0,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_store_local_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = StoreLocalTemplate {
        src_reg: 0,
        local_idx: 1,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_load_closure_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = LoadClosureTemplate {
        dst_reg: 0,
        closure_idx: 2,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_store_closure_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let template = StoreClosureTemplate {
        src_reg: 1,
        closure_idx: 0,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 0);
}
