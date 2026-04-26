use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;

// =========================================================================
// Test Helper
// =========================================================================

/// Emit a template into an assembler, bind all deopt labels, and finalize.
fn emit_and_finalize(template: &dyn OpcodeTemplate) -> Vec<u8> {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let _deopt = ctx.create_deopt_label();
        template.emit(&mut ctx);
        // Bind all deopt labels to current position (required by assembler)
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
    }
    asm.finalize()
        .expect("assembler finalization should succeed")
}

// =========================================================================
// Layout Constant Tests
// =========================================================================

#[test]
fn test_layout_type_id_offset() {
    assert_eq!(list_layout::TYPE_ID_OFFSET, 0);
}

#[test]
fn test_layout_items_ptr_offset() {
    // ObjectHeader is 16 bytes, then Vec.ptr starts immediately
    assert_eq!(list_layout::ITEMS_PTR_OFFSET, 16);
}

#[test]
fn test_layout_items_len_offset() {
    // Vec.len is 8 bytes after Vec.ptr
    assert_eq!(list_layout::ITEMS_LEN_OFFSET, 24);
}

#[test]
fn test_layout_items_cap_offset() {
    // Vec.cap is 8 bytes after Vec.len
    assert_eq!(list_layout::ITEMS_CAP_OFFSET, 32);
}

#[test]
fn test_layout_value_size() {
    assert_eq!(list_layout::VALUE_SIZE, 8);
}

#[test]
fn test_layout_list_type_id() {
    assert_eq!(list_layout::LIST_TYPE_ID, 6);
}

// =========================================================================
// Tag Constant Tests
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
fn test_object_tag_differs_from_int_tag() {
    assert_ne!(
        object_tag_check() as u32,
        value_tags::int_tag_check() as u32,
    );
}

#[test]
fn test_object_tag_check_matches_nan_bits() {
    // OBJECT_TAG must reside in the payload bits region
    assert_eq!(value_tags::OBJECT_TAG & value_tags::QNAN_BITS, 0);
}

// =========================================================================
// ListIndexTemplate Tests
// =========================================================================

#[test]
fn test_list_index_creates() {
    let t = ListIndexTemplate::new(0, 1, 2, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.list_reg, 1);
    assert_eq!(t.index_reg, 2);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_list_index_estimated_size() {
    let t = ListIndexTemplate::new(0, 1, 2, 0);
    assert!(t.estimated_size() >= 100);
    assert!(t.estimated_size() <= 300);
}

#[test]
fn test_list_index_emits_code() {
    let code = emit_and_finalize(&ListIndexTemplate::new(0, 1, 2, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_index_code_size_within_estimate() {
    let t = ListIndexTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size() * 2,
        "Code size {} exceeds 2× estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_list_index_different_registers() {
    let code1 = emit_and_finalize(&ListIndexTemplate::new(0, 1, 2, 0));
    let code2 = emit_and_finalize(&ListIndexTemplate::new(3, 4, 5, 0));
    // Different registers should produce code of similar size
    assert!(!code1.is_empty());
    assert!(!code2.is_empty());
}

#[test]
fn test_list_index_self_alias_dst_list() {
    // dst == list: should still compile
    let code = emit_and_finalize(&ListIndexTemplate::new(1, 1, 2, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_index_self_alias_dst_index() {
    // dst == index: should still compile
    let code = emit_and_finalize(&ListIndexTemplate::new(2, 1, 2, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_index_high_register() {
    let code = emit_and_finalize(&ListIndexTemplate::new(14, 13, 12, 0));
    assert!(!code.is_empty());
}

// =========================================================================
// ListStoreTemplate Tests
// =========================================================================

#[test]
fn test_list_store_creates() {
    let t = ListStoreTemplate::new(0, 1, 2, 0);
    assert_eq!(t.list_reg, 0);
    assert_eq!(t.index_reg, 1);
    assert_eq!(t.value_reg, 2);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_list_store_estimated_size() {
    let t = ListStoreTemplate::new(0, 1, 2, 0);
    assert!(t.estimated_size() >= 100);
}

#[test]
fn test_list_store_emits_code() {
    let code = emit_and_finalize(&ListStoreTemplate::new(0, 1, 2, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_store_code_size_within_estimate() {
    let t = ListStoreTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size() * 2,
        "Code size {} exceeds 2× estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_list_store_different_registers() {
    let code1 = emit_and_finalize(&ListStoreTemplate::new(0, 1, 2, 0));
    let code2 = emit_and_finalize(&ListStoreTemplate::new(3, 4, 5, 0));
    assert!(!code1.is_empty());
    assert!(!code2.is_empty());
}

#[test]
fn test_list_store_self_alias() {
    let code = emit_and_finalize(&ListStoreTemplate::new(1, 1, 2, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_store_high_registers() {
    let code = emit_and_finalize(&ListStoreTemplate::new(14, 13, 12, 0));
    assert!(!code.is_empty());
}

// =========================================================================
// ListAppendFastTemplate Tests
// =========================================================================

#[test]
fn test_list_append_fast_creates() {
    let t = ListAppendFastTemplate::new(0, 1, 0);
    assert_eq!(t.list_reg, 0);
    assert_eq!(t.item_reg, 1);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_list_append_fast_estimated_size() {
    let t = ListAppendFastTemplate::new(0, 1, 0);
    assert!(t.estimated_size() >= 100);
}

#[test]
fn test_list_append_fast_emits_code() {
    let code = emit_and_finalize(&ListAppendFastTemplate::new(0, 1, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_append_fast_code_size_within_estimate() {
    let t = ListAppendFastTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size() * 2,
        "Code size {} exceeds 2× estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_list_append_fast_different_registers() {
    let code1 = emit_and_finalize(&ListAppendFastTemplate::new(0, 1, 0));
    let code2 = emit_and_finalize(&ListAppendFastTemplate::new(3, 4, 0));
    assert!(!code1.is_empty());
    assert!(!code2.is_empty());
}

#[test]
fn test_list_append_fast_self_alias() {
    // list_reg == item_reg: technically valid (append list to itself)
    let code = emit_and_finalize(&ListAppendFastTemplate::new(1, 1, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_append_fast_high_registers() {
    let code = emit_and_finalize(&ListAppendFastTemplate::new(14, 13, 0));
    assert!(!code.is_empty());
}

// =========================================================================
// ListConcatTemplate Tests
// =========================================================================

#[test]
fn test_list_concat_creates() {
    let t = ListConcatTemplate::new(0, 1, 2, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.lhs_reg, 1);
    assert_eq!(t.rhs_reg, 2);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_list_concat_estimated_size() {
    let t = ListConcatTemplate::new(0, 1, 2, 0);
    assert!(t.estimated_size() >= 80);
}

#[test]
fn test_list_concat_emits_code() {
    let code = emit_and_finalize(&ListConcatTemplate::new(0, 1, 2, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_concat_code_size_within_estimate() {
    let t = ListConcatTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size() * 2,
        "Code size {} exceeds 2× estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_list_concat_different_registers() {
    let code1 = emit_and_finalize(&ListConcatTemplate::new(0, 1, 2, 0));
    let code2 = emit_and_finalize(&ListConcatTemplate::new(5, 6, 7, 0));
    assert!(!code1.is_empty());
    assert!(!code2.is_empty());
}

#[test]
fn test_list_concat_self_alias() {
    let code = emit_and_finalize(&ListConcatTemplate::new(1, 1, 2, 0));
    assert!(!code.is_empty());
}

// =========================================================================
// ListRepeatTemplate Tests
// =========================================================================

#[test]
fn test_list_repeat_creates() {
    let t = ListRepeatTemplate::new(0, 1, 2, true, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.list_reg, 1);
    assert_eq!(t.count_reg, 2);
    assert!(t.list_first);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_list_repeat_estimated_size() {
    let t = ListRepeatTemplate::new(0, 1, 2, true, 0);
    assert!(t.estimated_size() >= 100);
}

#[test]
fn test_list_repeat_list_first_emits_code() {
    let code = emit_and_finalize(&ListRepeatTemplate::new(0, 1, 2, true, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_repeat_int_first_emits_code() {
    let code = emit_and_finalize(&ListRepeatTemplate::new(0, 1, 2, false, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_list_repeat_code_size_within_estimate() {
    let t = ListRepeatTemplate::new(0, 1, 2, true, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size() * 2,
        "Code size {} exceeds 2× estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_list_repeat_both_directions_compile() {
    let code_list_first = emit_and_finalize(&ListRepeatTemplate::new(0, 1, 2, true, 0));
    let code_int_first = emit_and_finalize(&ListRepeatTemplate::new(0, 1, 2, false, 0));
    assert!(!code_list_first.is_empty());
    assert!(!code_int_first.is_empty());
}

#[test]
fn test_list_repeat_max_constant() {
    assert_eq!(ListRepeatTemplate::MAX_REPEAT_INLINE, 1_000_000);
}

#[test]
fn test_list_repeat_high_registers() {
    let code = emit_and_finalize(&ListRepeatTemplate::new(14, 13, 12, true, 0));
    assert!(!code.is_empty());
}

// =========================================================================
// Helper Function Tests
// =========================================================================

#[test]
fn test_emit_object_check_and_extract() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let deopt = ctx.create_deopt_label();
        let acc = ctx.regs.accumulator;
        let scratch = ctx.regs.scratch1;
        emit_object_check_and_extract(&mut ctx, acc, acc, scratch, deopt);
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

#[test]
fn test_emit_list_type_guard() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let deopt = ctx.create_deopt_label();
        let acc = ctx.regs.accumulator;
        let scratch = ctx.regs.scratch1;
        emit_list_type_guard(&mut ctx, acc, scratch, deopt);
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

#[test]
fn test_emit_list_check_and_extract() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let deopt = ctx.create_deopt_label();
        let acc = ctx.regs.accumulator;
        let scratch = ctx.regs.scratch1;
        emit_list_check_and_extract(&mut ctx, acc, acc, scratch, deopt);
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

#[test]
fn test_emit_int_check_for_list() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let deopt = ctx.create_deopt_label();
        let acc = ctx.regs.accumulator;
        let scratch = ctx.regs.scratch1;
        emit_int_check_and_extract_for_list(&mut ctx, acc, acc, scratch, deopt);
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

#[test]
fn test_emit_object_box() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let acc = ctx.regs.accumulator;
        let scratch = ctx.regs.scratch1;
        emit_object_box(&mut ctx, acc, scratch);
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_multiple_deopt_labels() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let deopt0 = ctx.create_deopt_label();
        let deopt1 = ctx.create_deopt_label();
        let deopt2 = ctx.create_deopt_label();

        ListIndexTemplate::new(0, 1, 2, deopt0).emit(&mut ctx);
        ListStoreTemplate::new(3, 4, 5, deopt1).emit(&mut ctx);
        ListConcatTemplate::new(6, 7, 8, deopt2).emit(&mut ctx);
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

#[test]
fn test_sequential_template_emission() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let deopt = ctx.create_deopt_label();

        for i in 0..5u8 {
            ListIndexTemplate::new(i, i.wrapping_add(1), i.wrapping_add(2), deopt).emit(&mut ctx);
        }
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

#[test]
fn test_index_generates_more_code_than_concat() {
    // Index has inline bounds check and indexed load; concat just type-checks + deopts
    let index_code = emit_and_finalize(&ListIndexTemplate::new(0, 1, 2, 0));
    let concat_code = emit_and_finalize(&ListConcatTemplate::new(0, 1, 2, 0));
    assert!(
        index_code.len() > concat_code.len(),
        "Index ({} bytes) should generate more code than Concat ({} bytes)",
        index_code.len(),
        concat_code.len()
    );
}

#[test]
fn test_append_emits_substantial_code() {
    // Append has type check + len/cap check + indexed store + len update
    let append_code = emit_and_finalize(&ListAppendFastTemplate::new(0, 1, 0));
    // Should be at least 50 bytes for all the inline operations
    assert!(
        append_code.len() >= 50,
        "Append should emit substantial code, got {} bytes",
        append_code.len()
    );
}

#[test]
fn test_adjacent_register_indices() {
    let code = emit_and_finalize(&ListIndexTemplate::new(0, 1, 2, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_zero_register_indices() {
    let code = emit_and_finalize(&ListStoreTemplate::new(0, 0, 0, 0));
    assert!(!code.is_empty());
}
