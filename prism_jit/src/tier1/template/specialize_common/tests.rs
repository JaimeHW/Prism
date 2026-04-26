use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;

fn make_ctx_with_deopt<'a>(asm: &'a mut Assembler, frame: &'a FrameLayout) -> TemplateContext<'a> {
    let mut ctx = TemplateContext::new(asm, frame);
    ctx.create_deopt_label();
    ctx
}

// =========================================================================
// Object Layout Constants
// =========================================================================

#[test]
fn test_type_id_offset_is_zero() {
    assert_eq!(object_layout::TYPE_ID_OFFSET, 0);
}

#[test]
fn test_header_size() {
    assert_eq!(object_layout::HEADER_SIZE, 16);
}

// =========================================================================
// Type ID Constants
// =========================================================================

#[test]
fn test_type_id_list() {
    assert_eq!(type_ids::LIST, 6);
}

#[test]
fn test_type_id_tuple() {
    assert_eq!(type_ids::TUPLE, 7);
}

#[test]
fn test_type_id_dict() {
    assert_eq!(type_ids::DICT, 8);
}

#[test]
fn test_type_id_set() {
    assert_eq!(type_ids::SET, 9);
}

#[test]
fn test_type_ids_are_distinct() {
    let ids = [
        type_ids::LIST,
        type_ids::TUPLE,
        type_ids::DICT,
        type_ids::SET,
    ];
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(
                ids[i], ids[j],
                "Type IDs at {} and {} must be distinct",
                i, j
            );
        }
    }
}

// =========================================================================
// Tag Check Constants
// =========================================================================

#[test]
fn test_object_tag_check_value() {
    let expected = ((value_tags::QNAN_BITS | value_tags::OBJECT_TAG) >> 48) as u16;
    assert_eq!(object_tag_check(), expected);
}

#[test]
fn test_object_tag_check_nonzero() {
    assert_ne!(object_tag_check(), 0);
}

#[test]
fn test_string_tag_check_value() {
    let expected = ((value_tags::QNAN_BITS | value_tags::STRING_TAG) >> 48) as u16;
    assert_eq!(string_tag_check(), expected);
}

#[test]
fn test_string_tag_check_nonzero() {
    assert_ne!(string_tag_check(), 0);
}

#[test]
fn test_object_tag_differs_from_string_tag() {
    assert_ne!(object_tag_check(), string_tag_check());
}

#[test]
fn test_object_tag_differs_from_int_tag() {
    assert_ne!(
        object_tag_check() as u32,
        value_tags::int_tag_check() as u32,
    );
}

#[test]
fn test_string_tag_differs_from_int_tag() {
    assert_ne!(
        string_tag_check() as u32,
        value_tags::int_tag_check() as u32,
    );
}

// =========================================================================
// Code Emission: Object Type Guards
// =========================================================================

#[test]
fn test_emit_object_check_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    emit_object_check_and_extract(&mut ctx, acc, acc, s1, 0);
    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_emit_object_check_different_src_dst() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    let s2 = ctx.regs.scratch2;
    emit_object_check_and_extract(&mut ctx, acc, s1, s2, 0);
    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_emit_type_id_guard_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    emit_type_id_guard(&mut ctx, acc, s1, type_ids::DICT, 0);
    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_emit_typed_object_check_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    emit_typed_object_check_and_extract(&mut ctx, acc, acc, s1, type_ids::LIST, 0);
    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_emit_typed_object_check_different_type_ids_same_size() {
    // Different type IDs should generate same-sized code (just different immediates)
    let mut asm1 = Assembler::new();
    let mut asm2 = Assembler::new();
    let frame = FrameLayout::minimal(8);

    {
        let mut ctx1 = make_ctx_with_deopt(&mut asm1, &frame);
        let acc = ctx1.regs.accumulator;
        let s1 = ctx1.regs.scratch1;
        emit_typed_object_check_and_extract(&mut ctx1, acc, acc, s1, type_ids::LIST, 0);
        for l in &ctx1.deopt_labels {
            ctx1.asm.bind_label(*l);
        }
    }

    {
        let mut ctx2 = make_ctx_with_deopt(&mut asm2, &frame);
        let acc = ctx2.regs.accumulator;
        let s1 = ctx2.regs.scratch1;
        emit_typed_object_check_and_extract(&mut ctx2, acc, acc, s1, type_ids::DICT, 0);
        for l in &ctx2.deopt_labels {
            ctx2.asm.bind_label(*l);
        }
    }

    let code1 = asm1.finalize().unwrap();
    let code2 = asm2.finalize().unwrap();
    assert_eq!(code1.len(), code2.len());
}

// =========================================================================
// Code Emission: Integer Guard
// =========================================================================

#[test]
fn test_emit_int_check_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    let s2 = ctx.regs.scratch2;
    emit_int_check_and_extract(&mut ctx, acc, acc, s2, 0);
    let _ = s1; // intentionally unused
    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_emit_int_check_different_src_dst() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    let s2 = ctx.regs.scratch2;
    emit_int_check_and_extract(&mut ctx, acc, s1, s2, 0);
    assert!(ctx.asm.offset() > 0);
}

// =========================================================================
// Code Emission: String Guard
// =========================================================================

#[test]
fn test_emit_string_check_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    emit_string_check_and_extract(&mut ctx, acc, acc, s1, 0);
    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_emit_string_check_different_src_dst() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    let s2 = ctx.regs.scratch2;
    emit_string_check_and_extract(&mut ctx, acc, s1, s2, 0);
    assert!(ctx.asm.offset() > 0);
}

// =========================================================================
// Code Emission: Boxing Helpers
// =========================================================================

#[test]
fn test_emit_object_box_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    emit_object_box(&mut ctx, acc, s1);
    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_emit_string_box_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    let s1 = ctx.regs.scratch1;
    emit_string_box(&mut ctx, acc, s1);
    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_emit_bool_box_true_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    emit_bool_box_true(&mut ctx, acc);
    assert!(ctx.asm.offset() > 0);
}

#[test]
fn test_emit_bool_box_false_emits_code() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
    let acc = ctx.regs.accumulator;
    emit_bool_box_false(&mut ctx, acc);
    assert!(ctx.asm.offset() > 0);
}

// =========================================================================
// Guard Size Consistency
// =========================================================================

#[test]
fn test_object_and_string_guards_similar_size() {
    let mut asm1 = Assembler::new();
    let mut asm2 = Assembler::new();
    let frame = FrameLayout::minimal(8);

    {
        let mut ctx1 = make_ctx_with_deopt(&mut asm1, &frame);
        let acc = ctx1.regs.accumulator;
        let s1 = ctx1.regs.scratch1;
        emit_object_check_and_extract(&mut ctx1, acc, acc, s1, 0);
        for l in &ctx1.deopt_labels {
            ctx1.asm.bind_label(*l);
        }
    }

    {
        let mut ctx2 = make_ctx_with_deopt(&mut asm2, &frame);
        let acc = ctx2.regs.accumulator;
        let s1 = ctx2.regs.scratch1;
        emit_string_check_and_extract(&mut ctx2, acc, acc, s1, 0);
        for l in &ctx2.deopt_labels {
            ctx2.asm.bind_label(*l);
        }
    }

    let code1 = asm1.finalize().unwrap();
    let code2 = asm2.finalize().unwrap();
    // Same structure, just different immediate values
    assert_eq!(code1.len(), code2.len());
}

#[test]
fn test_int_and_object_guards_similar_size() {
    let mut asm1 = Assembler::new();
    let mut asm2 = Assembler::new();
    let frame = FrameLayout::minimal(8);

    {
        let mut ctx1 = make_ctx_with_deopt(&mut asm1, &frame);
        let acc = ctx1.regs.accumulator;
        let s1 = ctx1.regs.scratch1;
        emit_object_check_and_extract(&mut ctx1, acc, acc, s1, 0);
        for l in &ctx1.deopt_labels {
            ctx1.asm.bind_label(*l);
        }
    }

    {
        let mut ctx2 = make_ctx_with_deopt(&mut asm2, &frame);
        let acc = ctx2.regs.accumulator;
        let s2 = ctx2.regs.scratch2;
        emit_int_check_and_extract(&mut ctx2, acc, acc, s2, 0);
        for l in &ctx2.deopt_labels {
            ctx2.asm.bind_label(*l);
        }
    }

    let code1 = asm1.finalize().unwrap();
    let code2 = asm2.finalize().unwrap();
    // Both have: mov, shr, cmp, jne, (optional mov), shl/shr or shl/sar
    // Should be same or very close in size
    let diff = (code1.len() as i64 - code2.len() as i64).unsigned_abs();
    assert!(diff <= 4, "Guard size difference {} too large", diff);
}

// =========================================================================
// Combined Guard Sequences
// =========================================================================

#[test]
fn test_emit_dict_guard_sequence() {
    // Full dict type check: object tag + TypeId::DICT
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    {
        let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
        let acc = ctx.regs.accumulator;
        let s1 = ctx.regs.scratch1;
        emit_typed_object_check_and_extract(&mut ctx, acc, acc, s1, type_ids::DICT, 0);
        for l in &ctx.deopt_labels {
            ctx.asm.bind_label(*l);
        }
    }
    let code = asm.finalize().unwrap();
    // Object check (~20 bytes) + type_id check (~12 bytes)
    assert!(
        code.len() >= 20,
        "Dict guard too small: {} bytes",
        code.len()
    );
    assert!(
        code.len() <= 80,
        "Dict guard too large: {} bytes",
        code.len()
    );
}

#[test]
fn test_emit_set_guard_sequence() {
    // Full set type check
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    {
        let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
        let acc = ctx.regs.accumulator;
        let s1 = ctx.regs.scratch1;
        emit_typed_object_check_and_extract(&mut ctx, acc, acc, s1, type_ids::SET, 0);
        for l in &ctx.deopt_labels {
            ctx.asm.bind_label(*l);
        }
    }
    let code = asm.finalize().unwrap();
    assert!(
        code.len() >= 20,
        "Set guard too small: {} bytes",
        code.len()
    );
    assert!(
        code.len() <= 80,
        "Set guard too large: {} bytes",
        code.len()
    );
}

#[test]
fn test_sequential_guards_emit_correctly() {
    // Object guard followed by int guard (common for dict[int_key])
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    {
        let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
        let acc = ctx.regs.accumulator;
        let s1 = ctx.regs.scratch1;
        let s2 = ctx.regs.scratch2;
        emit_typed_object_check_and_extract(&mut ctx, acc, acc, s1, type_ids::DICT, 0);
        emit_int_check_and_extract(&mut ctx, s1, s1, s2, 0);
        for l in &ctx.deopt_labels {
            ctx.asm.bind_label(*l);
        }
    }
    let code = asm.finalize().unwrap();
    assert!(
        code.len() >= 40,
        "Sequential guards too small: {} bytes",
        code.len()
    );
}

#[test]
fn test_dual_object_guards_emit_correctly() {
    // Two object guards (common for dict merge: dict | dict)
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    {
        let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
        let acc = ctx.regs.accumulator;
        let s1 = ctx.regs.scratch1;

        // First dict check
        emit_typed_object_check_and_extract(&mut ctx, acc, acc, s1, type_ids::DICT, 0);

        // Second dict check (on a different value)
        let slot = ctx.frame.register_slot(1);
        ctx.asm.mov_rm(s1, &slot);
        emit_typed_object_check_and_extract(&mut ctx, s1, s1, acc, type_ids::DICT, 0);

        for l in &ctx.deopt_labels {
            ctx.asm.bind_label(*l);
        }
    }
    let code = asm.finalize().unwrap();
    assert!(
        code.len() >= 50,
        "Dual dict guards too small: {} bytes",
        code.len()
    );
}
