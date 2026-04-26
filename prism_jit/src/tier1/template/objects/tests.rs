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
        ic_site_offset: None, // No IC for basic test
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
