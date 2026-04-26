use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;

fn make_ctx<'a>(asm: &'a mut Assembler, frame: &'a FrameLayout) -> TemplateContext<'a> {
    let mut ctx = TemplateContext::new(asm, frame);
    ctx.create_deopt_label();
    ctx
}

#[test]
fn test_build_list_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let template = BuildListTemplate {
        dst_reg: 0,
        start_reg: 1,
        count: 3,
        deopt_idx: 0,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_build_tuple_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let template = BuildTupleTemplate {
        dst_reg: 0,
        start_reg: 1,
        count: 2,
        deopt_idx: 0,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_build_dict_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let template = BuildDictTemplate {
        dst_reg: 0,
        start_reg: 1,
        count: 2,
        deopt_idx: 0,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_unpack_sequence_template() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = make_ctx(&mut asm, &frame);

    let template = UnpackSequenceTemplate {
        src_reg: 0,
        dst_start_reg: 1,
        count: 3,
        deopt_idx: 0,
    };
    template.emit(&mut ctx);

    assert!(ctx.asm.offset() > 0);
}
