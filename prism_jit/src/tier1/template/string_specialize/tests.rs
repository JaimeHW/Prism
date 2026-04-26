use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;

// =========================================================================
// Helper: Emit template and finalize
// =========================================================================

/// Helper to emit a template and get the generated code bytes.
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
// Tag Constants Tests
// =========================================================================

#[test]
fn test_string_tag_check_value() {
    let expected = ((value_tags::QNAN_BITS | value_tags::STRING_TAG) >> 48) as u16;
    assert_eq!(string_tag_check(), expected);
}

#[test]
fn test_string_tag_check_distinct_from_int() {
    assert_ne!(string_tag_check(), value_tags::int_tag_check());
}

#[test]
fn test_string_tag_is_nonzero() {
    assert_ne!(value_tags::STRING_TAG, 0);
}

#[test]
fn test_string_tag_distinct_from_all_other_tags() {
    let string = value_tags::STRING_TAG;
    assert_ne!(string, value_tags::INT_TAG);
    assert_ne!(string, value_tags::OBJECT_TAG);
    assert_ne!(string, value_tags::NONE_TAG);
    assert_ne!(string, value_tags::BOOL_TAG);
}

#[test]
fn test_string_tag_check_roundtrip() {
    let string_value = value_tags::QNAN_BITS | value_tags::STRING_TAG | 0x1234_5678;
    let extracted_tag = (string_value >> 48) as u16;
    assert_eq!(extracted_tag, string_tag_check());
}

#[test]
fn test_payload_extraction_preserves_pointer() {
    let ptr: u64 = 0x0000_DEAD_BEEF_CAFE;
    let boxed = value_tags::QNAN_BITS | value_tags::STRING_TAG | ptr;
    let payload = boxed & value_tags::PAYLOAD_MASK;
    assert_eq!(payload, ptr);
}

// =========================================================================
// StrConcatTemplate Tests
// =========================================================================

#[test]
fn test_str_concat_template_creation() {
    let t = StrConcatTemplate::new(0, 1, 2, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.lhs_reg, 1);
    assert_eq!(t.rhs_reg, 2);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_str_concat_template_estimated_size() {
    let t = StrConcatTemplate::new(0, 1, 2, 0);
    assert_eq!(t.estimated_size(), 120);
}

#[test]
fn test_str_concat_template_emits_code() {
    let code = emit_and_finalize(&StrConcatTemplate::new(0, 1, 2, 0));
    assert!(!code.is_empty(), "StrConcatTemplate should emit code");
}

#[test]
fn test_str_concat_template_different_registers() {
    let configs = [(0, 1, 2), (3, 4, 5), (0, 0, 1), (2, 1, 0)];
    for (dst, lhs, rhs) in configs {
        let code = emit_and_finalize(&StrConcatTemplate::new(dst, lhs, rhs, 0));
        assert!(
            !code.is_empty(),
            "Should emit code for ({}, {}, {})",
            dst,
            lhs,
            rhs
        );
    }
}

#[test]
fn test_str_concat_code_size_within_estimate() {
    let t = StrConcatTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size() * 3,
        "Code size {} exceeds 3x estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_str_concat_same_src_dst() {
    let code = emit_and_finalize(&StrConcatTemplate::new(1, 1, 2, 0));
    assert!(!code.is_empty());
}

// =========================================================================
// StrRepeatTemplate Tests
// =========================================================================

#[test]
fn test_str_repeat_template_creation_str_first() {
    let t = StrRepeatTemplate::new(0, 1, 2, true, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.str_reg, 1);
    assert_eq!(t.count_reg, 2);
    assert!(t.str_first);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_str_repeat_template_creation_int_first() {
    let t = StrRepeatTemplate::new(3, 4, 5, false, 1);
    assert_eq!(t.dst_reg, 3);
    assert_eq!(t.str_reg, 4);
    assert_eq!(t.count_reg, 5);
    assert!(!t.str_first);
    assert_eq!(t.deopt_idx, 1);
}

#[test]
fn test_str_repeat_template_estimated_size() {
    let t = StrRepeatTemplate::new(0, 1, 2, true, 0);
    assert_eq!(t.estimated_size(), 128);
}

#[test]
fn test_str_repeat_template_emits_code() {
    let code = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, true, 0));
    assert!(!code.is_empty(), "StrRepeatTemplate should emit code");
}

#[test]
fn test_str_repeat_max_inline_constant() {
    assert_eq!(StrRepeatTemplate::MAX_REPEAT_INLINE, 1_000_000);
    assert!(StrRepeatTemplate::MAX_REPEAT_INLINE > 0);
    assert!(StrRepeatTemplate::MAX_REPEAT_INLINE <= i32::MAX as i64);
}

#[test]
fn test_str_repeat_template_both_orderings() {
    let code1 = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, true, 0));
    let code2 = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, false, 0));
    assert!(!code1.is_empty());
    assert!(!code2.is_empty());
}

#[test]
fn test_str_repeat_code_size_within_estimate() {
    let t = StrRepeatTemplate::new(0, 1, 2, true, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size() * 3,
        "Code size {} exceeds 3x estimate {}",
        code.len(),
        t.estimated_size()
    );
}

// =========================================================================
// StrEqualTemplate Tests
// =========================================================================

#[test]
fn test_str_equal_template_creation() {
    let t = StrEqualTemplate::new(0, 1, 2, false, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.lhs_reg, 1);
    assert_eq!(t.rhs_reg, 2);
    assert!(!t.negate);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_str_not_equal_template_creation() {
    let t = StrEqualTemplate::new(3, 4, 5, true, 1);
    assert_eq!(t.dst_reg, 3);
    assert_eq!(t.lhs_reg, 4);
    assert_eq!(t.rhs_reg, 5);
    assert!(t.negate);
    assert_eq!(t.deopt_idx, 1);
}

#[test]
fn test_str_equal_template_estimated_size() {
    let t = StrEqualTemplate::new(0, 1, 2, false, 0);
    assert_eq!(t.estimated_size(), 96);
}

#[test]
fn test_str_equal_template_emits_code() {
    let code = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, false, 0));
    assert!(!code.is_empty(), "StrEqualTemplate should emit code");
}

#[test]
fn test_str_not_equal_template_emits_code() {
    let code = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, true, 0));
    assert!(!code.is_empty(), "StrNotEqualTemplate should emit code");
}

#[test]
fn test_str_equal_vs_not_equal_emit_different_code() {
    let code_eq = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, false, 0));
    let code_ne = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, true, 0));
    assert_ne!(code_eq, code_ne, "== and != should generate different code");
}

#[test]
fn test_str_equal_code_size_within_estimate() {
    let t = StrEqualTemplate::new(0, 1, 2, false, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size() * 3,
        "Code size {} exceeds 3x estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_str_equal_all_same_registers() {
    let code = emit_and_finalize(&StrEqualTemplate::new(0, 0, 0, false, 0));
    assert!(!code.is_empty());
}

// =========================================================================
// StrCompareTemplate Tests
// =========================================================================

#[test]
fn test_str_compare_op_as_str() {
    assert_eq!(StrCompareOp::Lt.as_str(), "<");
    assert_eq!(StrCompareOp::Le.as_str(), "<=");
    assert_eq!(StrCompareOp::Gt.as_str(), ">");
    assert_eq!(StrCompareOp::Ge.as_str(), ">=");
}

#[test]
fn test_str_compare_template_creation() {
    let t = StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.lhs_reg, 1);
    assert_eq!(t.rhs_reg, 2);
    assert_eq!(t.op, StrCompareOp::Lt);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_str_compare_template_estimated_size() {
    let t = StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0);
    assert_eq!(t.estimated_size(), 112);
}

#[test]
fn test_str_compare_template_emits_code_all_ops() {
    let ops = [
        StrCompareOp::Lt,
        StrCompareOp::Le,
        StrCompareOp::Gt,
        StrCompareOp::Ge,
    ];
    for op in ops {
        let code = emit_and_finalize(&StrCompareTemplate::new(0, 1, 2, op, 0));
        assert!(
            !code.is_empty(),
            "StrCompareTemplate({:?}) should emit code",
            op
        );
    }
}

#[test]
fn test_str_compare_code_size_within_estimate() {
    let t = StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size() * 3,
        "Code size {} exceeds 3x estimate {}",
        code.len(),
        t.estimated_size()
    );
}

// =========================================================================
// Cross-Template Consistency Tests
// =========================================================================

#[test]
fn test_all_templates_produce_nonzero_code() {
    let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
        Box::new(StrConcatTemplate::new(0, 1, 2, 0)),
        Box::new(StrRepeatTemplate::new(0, 1, 2, true, 0)),
        Box::new(StrEqualTemplate::new(0, 1, 2, false, 0)),
        Box::new(StrEqualTemplate::new(0, 1, 2, true, 0)),
        Box::new(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0)),
        Box::new(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Le, 0)),
        Box::new(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Gt, 0)),
        Box::new(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Ge, 0)),
    ];

    for (i, template) in templates.iter().enumerate() {
        let code = emit_and_finalize(template.as_ref());
        assert!(
            !code.is_empty(),
            "Template {} should produce non-empty code",
            i
        );
    }
}

#[test]
fn test_all_templates_have_positive_estimated_size() {
    assert!(StrConcatTemplate::new(0, 1, 2, 0).estimated_size() > 0);
    assert!(StrRepeatTemplate::new(0, 1, 2, true, 0).estimated_size() > 0);
    assert!(StrEqualTemplate::new(0, 1, 2, false, 0).estimated_size() > 0);
    assert!(StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0).estimated_size() > 0);
}

#[test]
fn test_estimated_sizes_are_reasonable() {
    let sizes = [
        StrConcatTemplate::new(0, 1, 2, 0).estimated_size(),
        StrRepeatTemplate::new(0, 1, 2, true, 0).estimated_size(),
        StrEqualTemplate::new(0, 1, 2, false, 0).estimated_size(),
        StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0).estimated_size(),
    ];
    for size in sizes {
        assert!(size >= 64, "Size {} too small", size);
        assert!(size <= 256, "Size {} too large", size);
    }
}

// =========================================================================
// Helper Function Tests
// =========================================================================

#[test]
fn test_emit_string_box() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let acc = ctx.regs.accumulator;
        let scratch = ctx.regs.scratch1;
        emit_string_box(&mut ctx, acc, scratch);
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

#[test]
fn test_emit_string_check_and_extract() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let deopt = ctx.create_deopt_label();
        let acc = ctx.regs.accumulator;
        let scratch = ctx.regs.scratch1;
        emit_string_check_and_extract(&mut ctx, acc, acc, scratch, deopt);
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

#[test]
fn test_emit_int_check_for_str() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        let deopt = ctx.create_deopt_label();
        let acc = ctx.regs.accumulator;
        let scratch = ctx.regs.scratch1;
        emit_int_check_and_extract_for_str(&mut ctx, acc, acc, scratch, deopt);
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
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

        StrConcatTemplate::new(0, 1, 2, deopt0).emit(&mut ctx);
        StrEqualTemplate::new(3, 4, 5, false, deopt1).emit(&mut ctx);
        StrRepeatTemplate::new(6, 7, 8, true, deopt2).emit(&mut ctx);
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
            StrEqualTemplate::new(i, i.wrapping_add(1), i.wrapping_add(2), i % 2 == 0, deopt)
                .emit(&mut ctx);
        }
        for label in &ctx.deopt_labels {
            ctx.asm.bind_label(*label);
        }
    }
    let code = asm.finalize().expect("finalize should succeed");
    assert!(!code.is_empty());
}

#[test]
fn test_high_register_indices() {
    let code = emit_and_finalize(&StrConcatTemplate::new(14, 13, 12, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_str_concat_adjacent_registers() {
    let code = emit_and_finalize(&StrConcatTemplate::new(0, 1, 2, 0));
    assert!(!code.is_empty());
}

#[test]
fn test_str_repeat_with_reversed_operands() {
    let code_str_first = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, true, 0));
    let code_int_first = emit_and_finalize(&StrRepeatTemplate::new(0, 1, 2, false, 0));
    assert!(!code_str_first.is_empty());
    assert!(!code_int_first.is_empty());
}

#[test]
fn test_str_compare_all_ops_same_size() {
    let code_lt = emit_and_finalize(&StrCompareTemplate::new(0, 1, 2, StrCompareOp::Lt, 0));
    let code_gt = emit_and_finalize(&StrCompareTemplate::new(0, 1, 2, StrCompareOp::Gt, 0));
    assert_eq!(
        code_lt.len(),
        code_gt.len(),
        "All compare ops should generate same-sized code"
    );
}

#[test]
fn test_str_equal_both_generate_code() {
    let code_eq = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, false, 0));
    let code_ne = emit_and_finalize(&StrEqualTemplate::new(0, 1, 2, true, 0));
    assert!(!code_eq.is_empty());
    assert!(!code_ne.is_empty());
}
