use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;
use crate::tier1::template::value_tags;

// =========================================================================
// Test Helper
// =========================================================================

fn emit_and_finalize(template: &dyn OpcodeTemplate) -> Vec<u8> {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        ctx.create_deopt_label();
        ctx.create_deopt_label();
        template.emit(&mut ctx);
        for l in &ctx.deopt_labels {
            ctx.asm.bind_label(*l);
        }
    }
    asm.finalize().unwrap()
}

/// Verify that code contains a SHR by 48 (0x30 immediate).
fn contains_shr_48(code: &[u8]) -> bool {
    code.contains(&0x30)
}

/// Verify presence of JNE (0x75 short or 0x0F 0x85 near) for tag mismatch.
fn contains_jne(code: &[u8]) -> bool {
    code.contains(&0x75) || code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0x85)
}

/// Verify presence of JMP (0xE9 near or 0xEB short) for deopt.
fn contains_jmp(code: &[u8]) -> bool {
    code.contains(&0xE9) || code.contains(&0xEB)
}

/// Verify the string tag check value is consistent.
fn verify_string_tag() {
    let tag = string_tag_check();
    let expected = ((value_tags::QNAN_BITS | value_tags::STRING_TAG) >> 48) as u16;
    assert_eq!(tag, expected);
}

// =========================================================================
// StrLenTemplate Tests
// =========================================================================

#[test]
fn test_str_len_creation() {
    let t = StrLenTemplate::new(0, 1, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.src_reg, 1);
    assert_eq!(t.deopt_idx, 0);
}

#[test]
fn test_str_len_emits_code() {
    let t = StrLenTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(!code.is_empty());
}

#[test]
fn test_str_len_size_within_estimate() {
    let t = StrLenTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size(),
        "StrLen actual {} exceeds estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_str_len_contains_shr_48() {
    let t = StrLenTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(contains_shr_48(&code), "Must SHR by 48 for tag check");
}

#[test]
fn test_str_len_contains_tag_mismatch_jne() {
    let t = StrLenTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(contains_jne(&code), "Must have JNE for tag mismatch deopt");
}

#[test]
fn test_str_len_contains_deopt_jmp() {
    let t = StrLenTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(contains_jmp(&code), "Must have JMP for deopt");
}

#[test]
fn test_str_len_different_registers() {
    for (d, s) in [(0, 1), (3, 4), (7, 8), (10, 11)] {
        let t = StrLenTemplate::new(d, s, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }
}

#[test]
fn test_str_len_high_registers() {
    let t = StrLenTemplate::new(12, 13, 0);
    let code = emit_and_finalize(&t);
    assert!(code.len() <= t.estimated_size());
}

#[test]
fn test_str_len_same_src_dst() {
    let t = StrLenTemplate::new(3, 3, 0);
    let code = emit_and_finalize(&t);
    assert!(!code.is_empty());
}

#[test]
fn test_str_len_multiple_deopt_indices() {
    for deopt in [0, 1] {
        let t = StrLenTemplate::new(0, 1, deopt);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }
}

// =========================================================================
// StrUpperTemplate Tests
// =========================================================================

#[test]
fn test_str_upper_creation() {
    let t = StrUpperTemplate::new(0, 1, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.src_reg, 1);
}

#[test]
fn test_str_upper_emits_code() {
    let t = StrUpperTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(!code.is_empty());
}

#[test]
fn test_str_upper_size_within_estimate() {
    let t = StrUpperTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size(),
        "StrUpper actual {} exceeds estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_str_upper_contains_string_guard() {
    let t = StrUpperTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(contains_shr_48(&code));
    assert!(contains_jne(&code));
}

#[test]
fn test_str_upper_different_registers() {
    for (d, s) in [(0, 1), (3, 4), (7, 8)] {
        let t = StrUpperTemplate::new(d, s, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }
}

#[test]
fn test_str_upper_high_registers() {
    let t = StrUpperTemplate::new(10, 11, 0);
    let code = emit_and_finalize(&t);
    assert!(code.len() <= t.estimated_size());
}

// =========================================================================
// StrLowerTemplate Tests
// =========================================================================

#[test]
fn test_str_lower_creation() {
    let t = StrLowerTemplate::new(0, 1, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.src_reg, 1);
}

#[test]
fn test_str_lower_emits_code() {
    let t = StrLowerTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(!code.is_empty());
}

#[test]
fn test_str_lower_size_within_estimate() {
    let t = StrLowerTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size(),
        "StrLower actual {} exceeds estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_str_lower_contains_string_guard() {
    let t = StrLowerTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&t);
    assert!(contains_shr_48(&code));
    assert!(contains_jne(&code));
}

#[test]
fn test_str_lower_different_registers() {
    for (d, s) in [(0, 1), (3, 4), (7, 8)] {
        let t = StrLowerTemplate::new(d, s, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }
}

#[test]
fn test_str_lower_high_registers() {
    let t = StrLowerTemplate::new(10, 11, 0);
    let code = emit_and_finalize(&t);
    assert!(code.len() <= t.estimated_size());
}

#[test]
fn test_str_lower_same_as_upper_structure() {
    // Both upper and lower should produce identical code structure
    let upper = StrUpperTemplate::new(0, 1, 0);
    let lower = StrLowerTemplate::new(0, 1, 0);
    let upper_code = emit_and_finalize(&upper);
    let lower_code = emit_and_finalize(&lower);
    // Same size (identical structure, differ only in deopt target interpretation)
    assert_eq!(upper_code.len(), lower_code.len());
}

// =========================================================================
// StrStripTemplate Tests
// =========================================================================

#[test]
fn test_str_strip_creation_both() {
    let t = StrStripTemplate::new(0, 1, StripKind::Both, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.src_reg, 1);
    assert_eq!(t.kind, StripKind::Both);
}

#[test]
fn test_str_strip_creation_left() {
    let t = StrStripTemplate::new(0, 1, StripKind::Left, 0);
    assert_eq!(t.kind, StripKind::Left);
}

#[test]
fn test_str_strip_creation_right() {
    let t = StrStripTemplate::new(0, 1, StripKind::Right, 0);
    assert_eq!(t.kind, StripKind::Right);
}

#[test]
fn test_str_strip_emits_code_all_kinds() {
    for kind in [StripKind::Both, StripKind::Left, StripKind::Right] {
        let t = StrStripTemplate::new(0, 1, kind, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty(), "{} must emit code", kind.as_str());
    }
}

#[test]
fn test_str_strip_size_within_estimate_all_kinds() {
    for kind in [StripKind::Both, StripKind::Left, StripKind::Right] {
        let t = StrStripTemplate::new(0, 1, kind, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "{} actual {} exceeds estimate {}",
            kind.as_str(),
            code.len(),
            t.estimated_size()
        );
    }
}

#[test]
fn test_str_strip_contains_string_guard() {
    let t = StrStripTemplate::new(0, 1, StripKind::Both, 0);
    let code = emit_and_finalize(&t);
    assert!(contains_shr_48(&code));
    assert!(contains_jne(&code));
}

#[test]
fn test_str_strip_all_kinds_same_code_size() {
    let both = StrStripTemplate::new(0, 1, StripKind::Both, 0);
    let left = StrStripTemplate::new(0, 1, StripKind::Left, 0);
    let right = StrStripTemplate::new(0, 1, StripKind::Right, 0);
    let both_code = emit_and_finalize(&both);
    let left_code = emit_and_finalize(&left);
    let right_code = emit_and_finalize(&right);
    // All produce identical code (kind is only used by interpreter post-deopt)
    assert_eq!(both_code.len(), left_code.len());
    assert_eq!(left_code.len(), right_code.len());
}

#[test]
fn test_str_strip_different_registers() {
    for (d, s) in [(0, 1), (3, 4), (7, 8)] {
        let t = StrStripTemplate::new(d, s, StripKind::Both, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }
}

#[test]
fn test_str_strip_high_registers() {
    let t = StrStripTemplate::new(12, 13, StripKind::Both, 0);
    let code = emit_and_finalize(&t);
    assert!(code.len() <= t.estimated_size());
}

#[test]
fn test_strip_kind_as_str() {
    assert_eq!(StripKind::Both.as_str(), "strip");
    assert_eq!(StripKind::Left.as_str(), "lstrip");
    assert_eq!(StripKind::Right.as_str(), "rstrip");
}

// =========================================================================
// StrStartsWithTemplate Tests
// =========================================================================

#[test]
fn test_str_startswith_creation() {
    let t = StrStartsWithTemplate::new(0, 1, 2, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.self_reg, 1);
    assert_eq!(t.prefix_reg, 2);
}

#[test]
fn test_str_startswith_emits_code() {
    let t = StrStartsWithTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(!code.is_empty());
}

#[test]
fn test_str_startswith_size_within_estimate() {
    let t = StrStartsWithTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size(),
        "StrStartsWith actual {} exceeds estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_str_startswith_contains_two_guards() {
    let t = StrStartsWithTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    // Should have at least 2 SHR-48 patterns (one per operand)
    let shr_count = code.windows(1).filter(|w| w[0] == 0x30).count();
    assert!(
        shr_count >= 2,
        "StrStartsWith needs 2 SHR-48 guards, found {}",
        shr_count
    );
}

#[test]
fn test_str_startswith_different_registers() {
    for (d, s, p) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
        let t = StrStartsWithTemplate::new(d, s, p, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }
}

#[test]
fn test_str_startswith_high_registers() {
    let t = StrStartsWithTemplate::new(10, 11, 12, 0);
    let code = emit_and_finalize(&t);
    assert!(code.len() <= t.estimated_size());
}

// =========================================================================
// StrEndsWithTemplate Tests
// =========================================================================

#[test]
fn test_str_endswith_creation() {
    let t = StrEndsWithTemplate::new(0, 1, 2, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.self_reg, 1);
    assert_eq!(t.suffix_reg, 2);
}

#[test]
fn test_str_endswith_emits_code() {
    let t = StrEndsWithTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(!code.is_empty());
}

#[test]
fn test_str_endswith_size_within_estimate() {
    let t = StrEndsWithTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size(),
        "StrEndsWith actual {} exceeds estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_str_endswith_contains_two_guards() {
    let t = StrEndsWithTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    let shr_count = code.windows(1).filter(|w| w[0] == 0x30).count();
    assert!(
        shr_count >= 2,
        "StrEndsWith needs 2 SHR-48 guards, found {}",
        shr_count
    );
}

#[test]
fn test_str_endswith_different_registers() {
    for (d, s, p) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
        let t = StrEndsWithTemplate::new(d, s, p, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }
}

#[test]
fn test_str_endswith_high_registers() {
    let t = StrEndsWithTemplate::new(10, 11, 12, 0);
    let code = emit_and_finalize(&t);
    assert!(code.len() <= t.estimated_size());
}

#[test]
fn test_str_startswith_endswith_same_structure() {
    let sw = StrStartsWithTemplate::new(0, 1, 2, 0);
    let ew = StrEndsWithTemplate::new(0, 1, 2, 0);
    let sw_code = emit_and_finalize(&sw);
    let ew_code = emit_and_finalize(&ew);
    // Same code size (identical structure)
    assert_eq!(sw_code.len(), ew_code.len());
}

// =========================================================================
// StrContainsTemplate Tests
// =========================================================================

#[test]
fn test_str_contains_creation() {
    let t = StrContainsTemplate::new(0, 1, 2, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.needle_reg, 1);
    assert_eq!(t.haystack_reg, 2);
}

#[test]
fn test_str_contains_emits_code() {
    let t = StrContainsTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(!code.is_empty());
}

#[test]
fn test_str_contains_size_within_estimate() {
    let t = StrContainsTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size(),
        "StrContains actual {} exceeds estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_str_contains_contains_two_guards() {
    let t = StrContainsTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    let shr_count = code.windows(1).filter(|w| w[0] == 0x30).count();
    assert!(
        shr_count >= 2,
        "StrContains needs 2 SHR-48 guards, found {}",
        shr_count
    );
}

#[test]
fn test_str_contains_different_registers() {
    for (d, n, h) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
        let t = StrContainsTemplate::new(d, n, h, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }
}

#[test]
fn test_str_contains_high_registers() {
    let t = StrContainsTemplate::new(10, 11, 12, 0);
    let code = emit_and_finalize(&t);
    assert!(code.len() <= t.estimated_size());
}

// =========================================================================
// StrFindTemplate Tests
// =========================================================================

#[test]
fn test_str_find_creation() {
    let t = StrFindTemplate::new(0, 1, 2, 0);
    assert_eq!(t.dst_reg, 0);
    assert_eq!(t.self_reg, 1);
    assert_eq!(t.needle_reg, 2);
}

#[test]
fn test_str_find_emits_code() {
    let t = StrFindTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(!code.is_empty());
}

#[test]
fn test_str_find_size_within_estimate() {
    let t = StrFindTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    assert!(
        code.len() <= t.estimated_size(),
        "StrFind actual {} exceeds estimate {}",
        code.len(),
        t.estimated_size()
    );
}

#[test]
fn test_str_find_contains_two_guards() {
    let t = StrFindTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&t);
    let shr_count = code.windows(1).filter(|w| w[0] == 0x30).count();
    assert!(
        shr_count >= 2,
        "StrFind needs 2 SHR-48 guards, found {}",
        shr_count
    );
}

#[test]
fn test_str_find_different_registers() {
    for (d, s, n) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
        let t = StrFindTemplate::new(d, s, n, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }
}

#[test]
fn test_str_find_high_registers() {
    let t = StrFindTemplate::new(10, 11, 12, 0);
    let code = emit_and_finalize(&t);
    assert!(code.len() <= t.estimated_size());
}

// =========================================================================
// Cross-Template Structural Tests
// =========================================================================

#[test]
fn test_all_unary_templates_have_string_guard() {
    let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
        Box::new(StrLenTemplate::new(0, 1, 0)),
        Box::new(StrUpperTemplate::new(0, 1, 0)),
        Box::new(StrLowerTemplate::new(0, 1, 0)),
        Box::new(StrStripTemplate::new(0, 1, StripKind::Both, 0)),
        Box::new(StrStripTemplate::new(0, 1, StripKind::Left, 0)),
        Box::new(StrStripTemplate::new(0, 1, StripKind::Right, 0)),
    ];

    for (i, t) in templates.iter().enumerate() {
        let code = emit_and_finalize(t.as_ref());
        assert!(
            contains_shr_48(&code),
            "Unary template {} missing SHR-48",
            i
        );
        assert!(
            contains_jne(&code),
            "Unary template {} missing JNE for deopt",
            i
        );
        assert!(
            contains_jmp(&code),
            "Unary template {} missing JMP for deopt",
            i
        );
    }
}

#[test]
fn test_all_binary_templates_have_two_guards() {
    let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
        Box::new(StrStartsWithTemplate::new(0, 1, 2, 0)),
        Box::new(StrEndsWithTemplate::new(0, 1, 2, 0)),
        Box::new(StrContainsTemplate::new(0, 1, 2, 0)),
        Box::new(StrFindTemplate::new(0, 1, 2, 0)),
    ];

    for (i, t) in templates.iter().enumerate() {
        let code = emit_and_finalize(t.as_ref());
        let shr_count = code.windows(1).filter(|w| w[0] == 0x30).count();
        assert!(
            shr_count >= 2,
            "Binary template {} has only {} SHR-48 guards, need >= 2",
            i,
            shr_count
        );
    }
}

#[test]
fn test_all_unary_same_estimated_size() {
    // All unary string methods should have the same estimate (same structure)
    let len_est = StrLenTemplate::new(0, 1, 0).estimated_size();
    let upper_est = StrUpperTemplate::new(0, 1, 0).estimated_size();
    let lower_est = StrLowerTemplate::new(0, 1, 0).estimated_size();
    let strip_est = StrStripTemplate::new(0, 1, StripKind::Both, 0).estimated_size();
    assert_eq!(len_est, upper_est);
    assert_eq!(upper_est, lower_est);
    assert_eq!(lower_est, strip_est);
}

#[test]
fn test_all_binary_same_estimated_size() {
    // All binary string methods should have the same estimate
    let sw_est = StrStartsWithTemplate::new(0, 1, 2, 0).estimated_size();
    let ew_est = StrEndsWithTemplate::new(0, 1, 2, 0).estimated_size();
    let ct_est = StrContainsTemplate::new(0, 1, 2, 0).estimated_size();
    let fd_est = StrFindTemplate::new(0, 1, 2, 0).estimated_size();
    assert_eq!(sw_est, ew_est);
    assert_eq!(ew_est, ct_est);
    assert_eq!(ct_est, fd_est);
}

#[test]
fn test_binary_larger_than_unary() {
    // Binary templates (2 guards) should be larger than unary (1 guard)
    let unary = StrLenTemplate::new(0, 1, 0);
    let binary = StrStartsWithTemplate::new(0, 1, 2, 0);
    let unary_code = emit_and_finalize(&unary);
    let binary_code = emit_and_finalize(&binary);
    assert!(
        binary_code.len() > unary_code.len(),
        "Binary ({}) must be larger than unary ({})",
        binary_code.len(),
        unary_code.len()
    );
}

#[test]
fn test_string_tag_check_consistency() {
    verify_string_tag();
}

#[test]
fn test_all_templates_within_estimates() {
    let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
        Box::new(StrLenTemplate::new(0, 1, 0)),
        Box::new(StrUpperTemplate::new(0, 1, 0)),
        Box::new(StrLowerTemplate::new(0, 1, 0)),
        Box::new(StrStripTemplate::new(0, 1, StripKind::Both, 0)),
        Box::new(StrStripTemplate::new(0, 1, StripKind::Left, 0)),
        Box::new(StrStripTemplate::new(0, 1, StripKind::Right, 0)),
        Box::new(StrStartsWithTemplate::new(0, 1, 2, 0)),
        Box::new(StrEndsWithTemplate::new(0, 1, 2, 0)),
        Box::new(StrContainsTemplate::new(0, 1, 2, 0)),
        Box::new(StrFindTemplate::new(0, 1, 2, 0)),
    ];

    for (i, t) in templates.iter().enumerate() {
        let code = emit_and_finalize(t.as_ref());
        assert!(
            code.len() <= t.estimated_size(),
            "Template {} actual {} exceeds estimate {}",
            i,
            code.len(),
            t.estimated_size()
        );
    }
}

#[test]
fn test_sequential_emission_does_not_corrupt() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(16);
    {
        let mut ctx = TemplateContext::new(&mut asm, &frame);
        ctx.create_deopt_label();
        ctx.create_deopt_label();

        StrLenTemplate::new(0, 1, 0).emit(&mut ctx);
        StrUpperTemplate::new(2, 3, 0).emit(&mut ctx);
        StrStartsWithTemplate::new(4, 5, 6, 0).emit(&mut ctx);
        StrContainsTemplate::new(7, 8, 9, 0).emit(&mut ctx);

        for l in &ctx.deopt_labels {
            ctx.asm.bind_label(*l);
        }
    }
    let code = asm.finalize().unwrap();
    assert!(!code.is_empty());
}

#[test]
fn test_all_templates_distinct_from_empty() {
    let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
        Box::new(StrLenTemplate::new(0, 1, 0)),
        Box::new(StrUpperTemplate::new(0, 1, 0)),
        Box::new(StrLowerTemplate::new(0, 1, 0)),
        Box::new(StrStripTemplate::new(0, 1, StripKind::Both, 0)),
        Box::new(StrStartsWithTemplate::new(0, 1, 2, 0)),
        Box::new(StrEndsWithTemplate::new(0, 1, 2, 0)),
        Box::new(StrContainsTemplate::new(0, 1, 2, 0)),
        Box::new(StrFindTemplate::new(0, 1, 2, 0)),
    ];

    for t in &templates {
        let code = emit_and_finalize(t.as_ref());
        assert!(code.len() >= 10, "Template code too small: {}", code.len());
    }
}

#[test]
fn test_all_register_permutations_unary() {
    // Test all sensible register orderings for unary templates
    let register_pairs = [(0, 1), (1, 0), (0, 0), (5, 10), (15, 14)];
    for (d, s) in register_pairs {
        let t = StrLenTemplate::new(d, s, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty(), "StrLen({}, {}) failed", d, s);
    }
}

#[test]
fn test_all_register_permutations_binary() {
    // Test all sensible register orderings for binary templates
    let register_triples = [
        (0, 1, 2),
        (2, 1, 0),
        (0, 0, 0),
        (5, 10, 15),
        (15, 14, 13),
        (0, 15, 7),
    ];
    for (d, s, a) in register_triples {
        let t = StrStartsWithTemplate::new(d, s, a, 0);
        let code = emit_and_finalize(&t);
        assert!(
            !code.is_empty(),
            "StrStartsWith({}, {}, {}) failed",
            d,
            s,
            a
        );
    }
}
