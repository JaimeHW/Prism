use super::*;
use crate::backend::x64::Assembler;
use crate::tier1::frame::FrameLayout;

#[test]
fn test_template_context_creation() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let ctx = TemplateContext::new(&mut asm, &frame);

    assert_eq!(ctx.bc_offset, 0);
    assert!(ctx.deopt_labels.is_empty());
}

#[test]
fn test_deopt_label_creation() {
    let mut asm = Assembler::new();
    let frame = FrameLayout::minimal(4);
    let mut ctx = TemplateContext::new(&mut asm, &frame);

    let idx1 = ctx.create_deopt_label();
    let idx2 = ctx.create_deopt_label();

    assert_eq!(idx1, 0);
    assert_eq!(idx2, 1);
    assert_eq!(ctx.deopt_labels.len(), 2);
}

// =========================================================================
// Value Tag Bit Pattern Tests
// =========================================================================

#[test]
fn test_value_tags_none() {
    // None = QNAN | (0 << 48) = 0x7FF8_0000_0000_0000
    assert_eq!(value_tags::none_value(), 0x7FF8_0000_0000_0000);
    assert_eq!(value_tags::none_tag_check(), 0x7FF8);
}

#[test]
fn test_value_tags_bool() {
    // True = QNAN | (1 << 48) | 1 = 0x7FF9_0000_0000_0001
    assert_eq!(value_tags::true_value(), 0x7FF9_0000_0000_0001);
    // False = QNAN | (1 << 48) | 0 = 0x7FF9_0000_0000_0000
    assert_eq!(value_tags::false_value(), 0x7FF9_0000_0000_0000);
    assert_eq!(value_tags::bool_tag_check(), 0x7FF9);
}

#[test]
fn test_value_tags_int() {
    // Int tag check = upper 16 bits of (QNAN | INT_TAG) = 0x7FFA
    assert_eq!(value_tags::int_tag_check(), 0x7FFA);
    assert_eq!(value_tags::INT_PATTERN, 0x7FFA_0000_0000_0000);
}

#[test]
fn test_value_tags_object() {
    assert_eq!(value_tags::object_tag_check(), 0x7FFB);
    assert_eq!(value_tags::OBJECT_PATTERN, 0x7FFB_0000_0000_0000);
}

#[test]
fn test_value_tags_string() {
    assert_eq!(value_tags::string_tag_check(), 0x7FFC);
    assert_eq!(value_tags::STRING_PATTERN, 0x7FFC_0000_0000_0000);
}

#[test]
fn test_value_tags_no_overlap() {
    // Verify all tag patterns are distinct
    let patterns = [
        value_tags::NONE_PATTERN,
        value_tags::BOOL_PATTERN,
        value_tags::INT_PATTERN,
        value_tags::OBJECT_PATTERN,
        value_tags::STRING_PATTERN,
    ];
    for i in 0..patterns.len() {
        for j in (i + 1)..patterns.len() {
            assert_ne!(
                patterns[i], patterns[j],
                "Tag patterns {} and {} must not overlap",
                i, j
            );
        }
    }
}

#[test]
fn test_value_tags_tag_mask_extracts_correctly() {
    // TAG_MASK should extract QNAN + 3-bit tag, stripping payload
    let int_42 = value_tags::box_int(42);
    assert_eq!(int_42 & value_tags::TAG_MASK, value_tags::INT_PATTERN);

    let none = value_tags::none_value();
    assert_eq!(none & value_tags::TAG_MASK, value_tags::NONE_PATTERN);

    let true_val = value_tags::true_value();
    assert_eq!(true_val & value_tags::TAG_MASK, value_tags::BOOL_PATTERN);
}

// =========================================================================
// Boxing Tests
// =========================================================================

#[test]
fn test_box_int() {
    let boxed = value_tags::box_int(42);
    assert_eq!(boxed & value_tags::PAYLOAD_MASK, 42);
    assert_eq!((boxed >> 48) as u16, value_tags::int_tag_check());
}

#[test]
fn test_box_int_negative() {
    let boxed = value_tags::box_int(-1);
    // Payload should be the lower 48 bits of -1 (sign-extended)
    let payload = boxed & value_tags::PAYLOAD_MASK;
    assert_eq!(payload, 0x0000_FFFF_FFFF_FFFF);
    assert_eq!((boxed >> 48) as u16, value_tags::int_tag_check());
}

#[test]
fn test_box_int_zero() {
    let boxed = value_tags::box_int(0);
    assert_eq!(boxed & value_tags::PAYLOAD_MASK, 0);
    assert_eq!((boxed >> 48) as u16, value_tags::int_tag_check());
}

#[test]
fn test_box_bool() {
    assert_eq!(value_tags::box_bool(true), value_tags::true_value());
    assert_eq!(value_tags::box_bool(false), value_tags::false_value());
}

#[test]
fn test_box_object() {
    let ptr = 0x0000_1234_5678_9ABCu64;
    let boxed = value_tags::box_object(ptr);
    assert_eq!(boxed & value_tags::PAYLOAD_MASK, ptr);
    assert_eq!((boxed >> 48) as u16, value_tags::object_tag_check());
}

#[test]
fn test_box_string() {
    let ptr = 0x0000_DEAD_BEEF_CAFEu64;
    let boxed = value_tags::box_string(ptr);
    assert_eq!(boxed & value_tags::PAYLOAD_MASK, ptr);
    assert_eq!((boxed >> 48) as u16, value_tags::string_tag_check());
}

// =========================================================================
// Cross-Validation with prism_core::Value
// =========================================================================
//
// These tests are the ultimate proof of correctness: they verify that
// JIT-produced bit patterns are byte-identical to interpreter-produced ones.
// Value is #[repr(transparent)] wrapping u64, so transmute is sound.

/// Extract raw u64 bits from a prism_core::Value.
///
/// # Safety
/// Value is `#[repr(transparent)]` around `u64`, so this is always sound.
fn value_bits(v: prism_core::Value) -> u64 {
    // SAFETY: Value is #[repr(transparent)] wrapping u64
    unsafe { std::mem::transmute(v) }
}

#[test]
fn test_cross_validate_none_with_core() {
    let core_none = prism_core::Value::none();
    assert_eq!(value_bits(core_none), value_tags::none_value());
}

#[test]
fn test_cross_validate_true_with_core() {
    let core_true = prism_core::Value::bool(true);
    assert_eq!(value_bits(core_true), value_tags::true_value());
}

#[test]
fn test_cross_validate_false_with_core() {
    let core_false = prism_core::Value::bool(false);
    assert_eq!(value_bits(core_false), value_tags::false_value());
}

#[test]
fn test_cross_validate_int_with_core() {
    for i in [0i64, 1, -1, 42, -42, 255, 1000, -1000] {
        let core_int = prism_core::Value::int(i).unwrap();
        let jit_int = value_tags::box_int(i);
        let core_bits = value_bits(core_int);
        assert_eq!(
            core_bits, jit_int,
            "Mismatch for int {}: core={:#018x}, jit={:#018x}",
            i, core_bits, jit_int
        );
    }
}

#[test]
fn test_cross_validate_int_tag_check_with_core() {
    let core_int = prism_core::Value::int(0).unwrap();
    let core_upper = (value_bits(core_int) >> 48) as u16;
    assert_eq!(core_upper, value_tags::int_tag_check());
}

#[test]
fn test_cross_validate_string_tag_pattern_with_core() {
    assert_eq!(
        value_tags::STRING_PATTERN,
        prism_core::value::STRING_TAG_PATTERN
    );
}

#[test]
fn test_cross_validate_int_tag_pattern_with_core() {
    assert_eq!(value_tags::INT_PATTERN, prism_core::value::INT_TAG_PATTERN);
}

#[test]
fn test_cross_validate_payload_mask_with_core() {
    assert_eq!(
        value_tags::PAYLOAD_MASK,
        prism_core::value::VALUE_PAYLOAD_MASK
    );
}

#[test]
fn test_cross_validate_tag_mask_with_core() {
    assert_eq!(value_tags::TAG_MASK, prism_core::value::TYPE_TAG_MASK);
}

#[test]
fn test_template_registry() {
    let mut registry = TemplateRegistry::new();
    assert_eq!(registry.stats(), (0, 0));

    registry.record_emission(100);
    registry.record_emission(50);
    assert_eq!(registry.stats(), (2, 150));
}
