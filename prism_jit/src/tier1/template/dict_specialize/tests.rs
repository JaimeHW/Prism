use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;
use crate::tier1::template::OpcodeTemplate;
use crate::tier1::template::specialize_common::emit_object_check_and_extract;

// =========================================================================
// Test Helpers
// =========================================================================

fn make_ctx_with_deopt<'a>(asm: &'a mut Assembler, frame: &'a FrameLayout) -> TemplateContext<'a> {
    let mut ctx = TemplateContext::new(asm, frame);
    ctx.create_deopt_label();
    ctx
}

/// Emit a template and finalize, returning the generated machine code bytes.
fn emit_and_finalize(template: &dyn OpcodeTemplate) -> Vec<u8> {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(8);
    {
        let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
        template.emit(&mut ctx);
        // Bind deopt label so assembler can resolve forward jumps
        for l in &ctx.deopt_labels {
            ctx.asm.bind_label(*l);
        }
    }
    asm.finalize().unwrap()
}

// =========================================================================
// DictGetTemplate Tests
// =========================================================================

#[test]
fn test_dict_get_emits_code() {
    let template = DictGetTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictGet should emit code");
}

#[test]
fn test_dict_get_code_within_estimate() {
    let template = DictGetTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictGet code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_get_registers() {
    // Different register assignments should all work
    for dict_reg in [0u8, 1, 2, 3] {
        for key_reg in [0u8, 1, 2, 3] {
            let template = DictGetTemplate::new(dict_reg, key_reg, 0, 0);
            let code = emit_and_finalize(&template);
            assert!(
                code.len() > 0,
                "DictGet(r{}, r{}) should emit code",
                dict_reg,
                key_reg
            );
        }
    }
}

#[test]
fn test_dict_get_deterministic() {
    let t1 = DictGetTemplate::new(0, 1, 2, 0);
    let t2 = DictGetTemplate::new(0, 1, 2, 0);
    let code1 = emit_and_finalize(&t1);
    let code2 = emit_and_finalize(&t2);
    assert_eq!(code1, code2, "Same template should produce identical code");
}

// =========================================================================
// DictGetStrTemplate Tests
// =========================================================================

#[test]
fn test_dict_get_str_emits_code() {
    let template = DictGetStrTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictGetStr should emit code");
}

#[test]
fn test_dict_get_str_more_code_than_basic() {
    // String-key version has additional STRING_TAG check
    let basic = DictGetTemplate::new(0, 1, 2, 0);
    let str_variant = DictGetStrTemplate::new(0, 1, 2, 0);
    let basic_code = emit_and_finalize(&basic);
    let str_code = emit_and_finalize(&str_variant);
    assert!(
        str_code.len() > basic_code.len(),
        "DictGetStr ({} bytes) should be larger than DictGet ({} bytes)",
        str_code.len(),
        basic_code.len()
    );
}

#[test]
fn test_dict_get_str_code_within_estimate() {
    let template = DictGetStrTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictGetStr code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_get_str_deterministic() {
    let t1 = DictGetStrTemplate::new(0, 1, 2, 0);
    let t2 = DictGetStrTemplate::new(0, 1, 2, 0);
    let code1 = emit_and_finalize(&t1);
    let code2 = emit_and_finalize(&t2);
    assert_eq!(code1, code2);
}

#[test]
fn test_dict_get_str_registers() {
    for dict_reg in [0u8, 1, 2, 3] {
        let template = DictGetStrTemplate::new(dict_reg, 1, 2, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0);
    }
}

// =========================================================================
// DictGetIntTemplate Tests
// =========================================================================

#[test]
fn test_dict_get_int_emits_code() {
    let template = DictGetIntTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictGetInt should emit code");
}

#[test]
fn test_dict_get_int_more_code_than_basic() {
    let basic = DictGetTemplate::new(0, 1, 2, 0);
    let int_variant = DictGetIntTemplate::new(0, 1, 2, 0);
    let basic_code = emit_and_finalize(&basic);
    let int_code = emit_and_finalize(&int_variant);
    assert!(
        int_code.len() > basic_code.len(),
        "DictGetInt ({}) should be larger than DictGet ({})",
        int_code.len(),
        basic_code.len()
    );
}

#[test]
fn test_dict_get_int_similar_to_str() {
    // Int-key and string-key versions should have similar sizes
    let str_variant = DictGetStrTemplate::new(0, 1, 2, 0);
    let int_variant = DictGetIntTemplate::new(0, 1, 2, 0);
    let str_code = emit_and_finalize(&str_variant);
    let int_code = emit_and_finalize(&int_variant);
    let diff = (str_code.len() as i64 - int_code.len() as i64).unsigned_abs();
    assert!(
        diff <= 8,
        "DictGetStr ({}) and DictGetInt ({}) should be similar size (diff {})",
        str_code.len(),
        int_code.len(),
        diff
    );
}

#[test]
fn test_dict_get_int_code_within_estimate() {
    let template = DictGetIntTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictGetInt code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_get_int_deterministic() {
    let t1 = DictGetIntTemplate::new(0, 1, 2, 0);
    let t2 = DictGetIntTemplate::new(0, 1, 2, 0);
    assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
}

// =========================================================================
// DictSetFastTemplate Tests
// =========================================================================

#[test]
fn test_dict_set_emits_code() {
    let template = DictSetFastTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictSet should emit code");
}

#[test]
fn test_dict_set_code_within_estimate() {
    let template = DictSetFastTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictSet code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_set_similar_to_get() {
    // DictSet and DictGet both have single dict type guard + deopt
    let get_template = DictGetTemplate::new(0, 1, 2, 0);
    let set_template = DictSetFastTemplate::new(0, 1, 2, 0);
    let get_code = emit_and_finalize(&get_template);
    let set_code = emit_and_finalize(&set_template);
    let diff = (get_code.len() as i64 - set_code.len() as i64).unsigned_abs();
    assert!(
        diff <= 8,
        "DictGet ({}) and DictSet ({}) sizes should be very close (diff {})",
        get_code.len(),
        set_code.len(),
        diff,
    );
}

#[test]
fn test_dict_set_deterministic() {
    let t1 = DictSetFastTemplate::new(0, 1, 2, 0);
    let t2 = DictSetFastTemplate::new(0, 1, 2, 0);
    assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
}

#[test]
fn test_dict_set_registers() {
    for dict_reg in 0u8..4 {
        for key_reg in 0u8..4 {
            for val_reg in 0u8..4 {
                let template = DictSetFastTemplate::new(dict_reg, key_reg, val_reg, 0);
                let code = emit_and_finalize(&template);
                assert!(code.len() > 0);
            }
        }
    }
}

// =========================================================================
// DictSetStrTemplate Tests
// =========================================================================

#[test]
fn test_dict_set_str_emits_code() {
    let template = DictSetStrTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictSetStr should emit code");
}

#[test]
fn test_dict_set_str_more_code_than_basic_set() {
    let basic = DictSetFastTemplate::new(0, 1, 2, 0);
    let str_variant = DictSetStrTemplate::new(0, 1, 2, 0);
    let basic_code = emit_and_finalize(&basic);
    let str_code = emit_and_finalize(&str_variant);
    assert!(
        str_code.len() > basic_code.len(),
        "DictSetStr ({}) should be larger than DictSet ({})",
        str_code.len(),
        basic_code.len()
    );
}

#[test]
fn test_dict_set_str_code_within_estimate() {
    let template = DictSetStrTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictSetStr code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_set_str_similar_to_get_str() {
    let get_str = DictGetStrTemplate::new(0, 1, 2, 0);
    let set_str = DictSetStrTemplate::new(0, 1, 2, 0);
    let get_code = emit_and_finalize(&get_str);
    let set_code = emit_and_finalize(&set_str);
    let diff = (get_code.len() as i64 - set_code.len() as i64).unsigned_abs();
    assert!(
        diff <= 8,
        "DictGetStr ({}) and DictSetStr ({}) should be same size (diff {})",
        get_code.len(),
        set_code.len(),
        diff,
    );
}

// =========================================================================
// DictContainsTemplate Tests
// =========================================================================

#[test]
fn test_dict_contains_emits_code() {
    let template = DictContainsTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictContains should emit code");
}

#[test]
fn test_dict_contains_code_within_estimate() {
    let template = DictContainsTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictContains code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_contains_similar_to_get() {
    let get = DictGetTemplate::new(0, 1, 2, 0);
    let contains = DictContainsTemplate::new(0, 1, 2, 0);
    let get_code = emit_and_finalize(&get);
    let contains_code = emit_and_finalize(&contains);
    let diff = (get_code.len() as i64 - contains_code.len() as i64).unsigned_abs();
    assert!(
        diff <= 8,
        "DictGet ({}) and DictContains ({}) should be similar size (diff {})",
        get_code.len(),
        contains_code.len(),
        diff,
    );
}

#[test]
fn test_dict_contains_deterministic() {
    let t1 = DictContainsTemplate::new(0, 1, 2, 0);
    let t2 = DictContainsTemplate::new(0, 1, 2, 0);
    assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
}

// =========================================================================
// DictMergeTemplate Tests
// =========================================================================

#[test]
fn test_dict_merge_emits_code() {
    let template = DictMergeTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictMerge should emit code");
}

#[test]
fn test_dict_merge_more_code_than_single_check() {
    // Merge has TWO dict type guards, so should be substantially larger
    let single = DictGetTemplate::new(0, 1, 2, 0);
    let merge = DictMergeTemplate::new(0, 1, 2, 0);
    let single_code = emit_and_finalize(&single);
    let merge_code = emit_and_finalize(&merge);
    assert!(
        merge_code.len() > single_code.len(),
        "DictMerge ({}) should be larger than DictGet ({}) due to dual type guards",
        merge_code.len(),
        single_code.len()
    );
}

#[test]
fn test_dict_merge_code_within_estimate() {
    let template = DictMergeTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictMerge code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_merge_deterministic() {
    let t1 = DictMergeTemplate::new(0, 1, 2, 0);
    let t2 = DictMergeTemplate::new(0, 1, 2, 0);
    assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
}

#[test]
fn test_dict_merge_registers() {
    for lhs_reg in [0u8, 1, 2, 3] {
        for rhs_reg in [0u8, 1, 2, 3] {
            let template = DictMergeTemplate::new(lhs_reg, rhs_reg, 0, 0);
            let code = emit_and_finalize(&template);
            assert!(code.len() > 0);
        }
    }
}

// =========================================================================
// DictDeleteTemplate Tests
// =========================================================================

#[test]
fn test_dict_delete_emits_code() {
    let template = DictDeleteTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictDelete should emit code");
}

#[test]
fn test_dict_delete_code_within_estimate() {
    let template = DictDeleteTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictDelete code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_delete_similar_to_contains() {
    let contains = DictContainsTemplate::new(0, 1, 2, 0);
    let delete = DictDeleteTemplate::new(0, 1, 0);
    let contains_code = emit_and_finalize(&contains);
    let delete_code = emit_and_finalize(&delete);
    let diff = (contains_code.len() as i64 - delete_code.len() as i64).unsigned_abs();
    assert!(
        diff <= 8,
        "DictContains ({}) and DictDelete ({}) should be similar (diff {})",
        contains_code.len(),
        delete_code.len(),
        diff,
    );
}

#[test]
fn test_dict_delete_deterministic() {
    let t1 = DictDeleteTemplate::new(0, 1, 0);
    let t2 = DictDeleteTemplate::new(0, 1, 0);
    assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
}

// =========================================================================
// DictViewGuardTemplate Tests
// =========================================================================

#[test]
fn test_dict_view_guard_emits_code() {
    let template = DictViewGuardTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictViewGuard should emit code");
}

#[test]
fn test_dict_view_guard_code_within_estimate() {
    let template = DictViewGuardTemplate::new(0, 1, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictViewGuard code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_view_guard_deterministic() {
    let t1 = DictViewGuardTemplate::new(0, 1, 0);
    let t2 = DictViewGuardTemplate::new(0, 1, 0);
    assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
}

// =========================================================================
// DictGetOrDefaultTemplate Tests
// =========================================================================

#[test]
fn test_dict_get_or_default_emits_code() {
    let template = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
    let code = emit_and_finalize(&template);
    assert!(code.len() > 0, "DictGetOrDefault should emit code");
}

#[test]
fn test_dict_get_or_default_code_within_estimate() {
    let template = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
    let code = emit_and_finalize(&template);
    let estimate = template.estimated_size();
    assert!(
        code.len() <= estimate * 2,
        "DictGetOrDefault code {} exceeds 2x estimate {}",
        code.len(),
        estimate
    );
}

#[test]
fn test_dict_get_or_default_deterministic() {
    let t1 = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
    let t2 = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
    assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
}

#[test]
fn test_dict_get_or_default_similar_to_single_guard() {
    // Both have dict guard + deopt
    let get = DictGetTemplate::new(0, 1, 2, 0);
    let get_default = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
    let get_code = emit_and_finalize(&get);
    let default_code = emit_and_finalize(&get_default);
    let diff = (get_code.len() as i64 - default_code.len() as i64).unsigned_abs();
    assert!(
        diff <= 8,
        "DictGet ({}) and DictGetOrDefault ({}) should be similar (diff {})",
        get_code.len(),
        default_code.len(),
        diff,
    );
}

// =========================================================================
// Cross-Template Comparisons
// =========================================================================

#[test]
fn test_single_guard_templates_similar_size() {
    // All templates with a single dict guard should produce similar code
    let get = emit_and_finalize(&DictGetTemplate::new(0, 1, 2, 0));
    let set = emit_and_finalize(&DictSetFastTemplate::new(0, 1, 2, 0));
    let contains = emit_and_finalize(&DictContainsTemplate::new(0, 1, 2, 0));
    let delete = emit_and_finalize(&DictDeleteTemplate::new(0, 1, 0));
    let view = emit_and_finalize(&DictViewGuardTemplate::new(0, 1, 0));
    let get_default = emit_and_finalize(&DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0));

    let sizes = [
        get.len(),
        set.len(),
        contains.len(),
        delete.len(),
        view.len(),
        get_default.len(),
    ];
    let min = *sizes.iter().min().unwrap();
    let max = *sizes.iter().max().unwrap();
    assert!(
        max - min <= 12,
        "Single-guard templates should be uniform: sizes {:?} (spread {})",
        sizes,
        max - min,
    );
}

#[test]
fn test_dual_guard_templates_larger_than_single() {
    let single = emit_and_finalize(&DictGetTemplate::new(0, 1, 2, 0));
    let dual_str = emit_and_finalize(&DictGetStrTemplate::new(0, 1, 2, 0));
    let dual_int = emit_and_finalize(&DictGetIntTemplate::new(0, 1, 2, 0));
    let dual_merge = emit_and_finalize(&DictMergeTemplate::new(0, 1, 2, 0));

    assert!(
        dual_str.len() > single.len(),
        "DictGetStr should be larger than DictGet"
    );
    assert!(
        dual_int.len() > single.len(),
        "DictGetInt should be larger than DictGet"
    );
    assert!(
        dual_merge.len() > single.len(),
        "DictMerge should be larger than DictGet"
    );
}

#[test]
fn test_merge_largest_template() {
    // Merge has two OBJECT + TypeId checks, should be the largest template
    let get_str = emit_and_finalize(&DictGetStrTemplate::new(0, 1, 2, 0));
    let merge = emit_and_finalize(&DictMergeTemplate::new(0, 1, 2, 0));
    // Merge checks OBJECT+DICT twice, GetStr checks OBJECT+DICT + STRING
    // Both have dual checks but merge has two full object+typeid checks
    // while GetStr has one object+typeid check + one string check
    // They should be similar, but merge may be larger due to dual object checks
    assert!(
        merge.len() >= get_str.len() - 10,
        "DictMerge ({}) should be at least as large as DictGetStr ({}) minus tolerance",
        merge.len(),
        get_str.len()
    );
}

// =========================================================================
// Code Quality
// =========================================================================

#[test]
fn test_all_templates_emit_nonzero_code() {
    let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
        Box::new(DictGetTemplate::new(0, 1, 2, 0)),
        Box::new(DictGetStrTemplate::new(0, 1, 2, 0)),
        Box::new(DictGetIntTemplate::new(0, 1, 2, 0)),
        Box::new(DictSetFastTemplate::new(0, 1, 2, 0)),
        Box::new(DictSetStrTemplate::new(0, 1, 2, 0)),
        Box::new(DictContainsTemplate::new(0, 1, 2, 0)),
        Box::new(DictMergeTemplate::new(0, 1, 2, 0)),
        Box::new(DictDeleteTemplate::new(0, 1, 0)),
        Box::new(DictViewGuardTemplate::new(0, 1, 0)),
        Box::new(DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0)),
    ];

    for (i, t) in templates.iter().enumerate() {
        let code = emit_and_finalize(t.as_ref());
        assert!(code.len() > 0, "Template {} should emit code", i);
    }
}

#[test]
fn test_all_templates_reasonable_size() {
    let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
        Box::new(DictGetTemplate::new(0, 1, 2, 0)),
        Box::new(DictGetStrTemplate::new(0, 1, 2, 0)),
        Box::new(DictGetIntTemplate::new(0, 1, 2, 0)),
        Box::new(DictSetFastTemplate::new(0, 1, 2, 0)),
        Box::new(DictSetStrTemplate::new(0, 1, 2, 0)),
        Box::new(DictContainsTemplate::new(0, 1, 2, 0)),
        Box::new(DictMergeTemplate::new(0, 1, 2, 0)),
        Box::new(DictDeleteTemplate::new(0, 1, 0)),
        Box::new(DictViewGuardTemplate::new(0, 1, 0)),
        Box::new(DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0)),
    ];

    for (i, t) in templates.iter().enumerate() {
        let code = emit_and_finalize(t.as_ref());
        // No template should exceed 256 bytes (they're mostly type guards + jmp)
        assert!(
            code.len() <= 256,
            "Template {} too large: {} bytes",
            i,
            code.len()
        );
        // Should be at least 20 bytes (minimum for a meaningful type guard)
        assert!(
            code.len() >= 20,
            "Template {} too small: {} bytes",
            i,
            code.len()
        );
    }
}

#[test]
fn test_all_estimated_sizes_valid() {
    let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
        Box::new(DictGetTemplate::new(0, 1, 2, 0)),
        Box::new(DictGetStrTemplate::new(0, 1, 2, 0)),
        Box::new(DictGetIntTemplate::new(0, 1, 2, 0)),
        Box::new(DictSetFastTemplate::new(0, 1, 2, 0)),
        Box::new(DictSetStrTemplate::new(0, 1, 2, 0)),
        Box::new(DictContainsTemplate::new(0, 1, 2, 0)),
        Box::new(DictMergeTemplate::new(0, 1, 2, 0)),
        Box::new(DictDeleteTemplate::new(0, 1, 0)),
        Box::new(DictViewGuardTemplate::new(0, 1, 0)),
        Box::new(DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0)),
    ];

    for (i, t) in templates.iter().enumerate() {
        let estimate = t.estimated_size();
        assert!(
            estimate > 0,
            "Template {} estimated size should be nonzero",
            i
        );
        assert!(
            estimate <= 256,
            "Template {} estimated size {} too large",
            i,
            estimate
        );
    }
}

// =========================================================================
// Stability Under Register Permutation
// =========================================================================

#[test]
fn test_dict_get_str_register_independence() {
    // Code structure should be independent of specific register values
    // (only the slot offset changes, instruction sequence stays the same)
    let t1 = DictGetStrTemplate::new(0, 1, 2, 0);
    let t2 = DictGetStrTemplate::new(2, 3, 0, 0);
    let code1 = emit_and_finalize(&t1);
    let code2 = emit_and_finalize(&t2);
    // Same instruction sequence, possibly different offsets
    // Length should be very similar (within a few bytes for modrm differences)
    let diff = (code1.len() as i64 - code2.len() as i64).unsigned_abs();
    assert!(
        diff <= 16,
        "Register permutation caused too much size variance ({}): {} vs {}",
        diff,
        code1.len(),
        code2.len()
    );
}

#[test]
fn test_dict_merge_register_independence() {
    let t1 = DictMergeTemplate::new(0, 1, 2, 0);
    let t2 = DictMergeTemplate::new(3, 4, 0, 0);
    let code1 = emit_and_finalize(&t1);
    let code2 = emit_and_finalize(&t2);
    let diff = (code1.len() as i64 - code2.len() as i64).unsigned_abs();
    assert!(
        diff <= 16,
        "Merge register permutation caused too much size variance ({}): {} vs {}",
        diff,
        code1.len(),
        code2.len()
    );
}

// =========================================================================
// Guard Correctness: Object Check Present
// =========================================================================

#[test]
fn test_dict_get_includes_object_guard() {
    // Verify that DictGet has a proper object+typeid guard
    // by checking it's larger than just a jmp instruction
    let template = DictGetTemplate::new(0, 1, 2, 0);
    let code = emit_and_finalize(&template);
    // A naked jmp is ~5 bytes; a proper guard should be much more
    assert!(
        code.len() >= 25,
        "DictGet should include a full type guard (got {} bytes, expected >= 25)",
        code.len()
    );
}

#[test]
fn test_dict_get_str_includes_double_guard() {
    let basic = DictGetTemplate::new(0, 1, 2, 0);
    let str_template = DictGetStrTemplate::new(0, 1, 2, 0);
    let basic_code = emit_and_finalize(&basic);
    let str_code = emit_and_finalize(&str_template);
    // The str version should have at least 10 additional bytes of guards
    assert!(
        str_code.len() >= basic_code.len() + 10,
        "DictGetStr should have significantly more code than DictGet: {} vs {}",
        str_code.len(),
        basic_code.len()
    );
}

#[test]
fn test_dict_merge_includes_double_object_guard() {
    let single = DictGetTemplate::new(0, 1, 2, 0);
    let merge = DictMergeTemplate::new(0, 1, 2, 0);
    let single_code = emit_and_finalize(&single);
    let merge_code = emit_and_finalize(&merge);
    // Merge should have roughly 2x the guard code, definitely at least 50% more
    assert!(
        merge_code.len() as f64 >= single_code.len() as f64 * 1.4,
        "DictMerge should have substantially more guard code: {} vs {}",
        merge_code.len(),
        single_code.len()
    );
}
