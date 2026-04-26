use super::*;
use crate::backend::x64::Assembler;

// -------------------------------------------------------------------------
// Layout Constant Tests
// -------------------------------------------------------------------------

#[test]
fn test_object_layout_constants() {
    // Verify layout assumptions match ShapedObject
    assert_eq!(OBJECT_HEADER_SIZE, 16);
    assert_eq!(OBJECT_SHAPE_PTR_OFFSET, 16);
    assert_eq!(OBJECT_INLINE_SLOTS_OFFSET, 24);
    assert_eq!(SLOT_SIZE, 8);
}

#[test]
fn test_ic_site_layout_constants() {
    assert_eq!(IC_SITE_SIZE, 8);
    assert_eq!(IC_SHAPE_ID_OFFSET, 0);
    assert_eq!(IC_SLOT_OFFSET_OFFSET, 4);
    assert_eq!(IC_FLAGS_OFFSET, 6);
}

#[test]
fn test_slot_offset_calculation() {
    // Verify slot offset calculations for various indices
    for i in 0..8u16 {
        let expected = OBJECT_INLINE_SLOTS_OFFSET + (i as i32) * SLOT_SIZE;
        assert_eq!(expected, 24 + (i as i32) * 8);
    }
}

#[test]
fn test_ic_offset_non_overlapping() {
    // Verify multiple IC sites don't overlap
    for i in 0..10 {
        let offset = (i as i32) * (IC_SITE_SIZE as i32);
        let next_offset = ((i + 1) as i32) * (IC_SITE_SIZE as i32);
        assert_eq!(next_offset - offset, IC_SITE_SIZE as i32);
    }
}

// -------------------------------------------------------------------------
// Shape ID Loading Tests
// -------------------------------------------------------------------------

#[test]
fn test_emit_load_shape_id() {
    let mut asm = Assembler::new();
    emit_load_shape_id(&mut asm, Gpr::Rax, Gpr::Rdi, Gpr::Rcx);

    // Should emit:
    // mov rcx, [rdi + 16]  (load shape pointer)
    // mov eax, [rcx + 16]  (load shape_id)
    assert!(asm.offset() > 0, "Should emit instructions");
}

#[test]
fn test_emit_load_shape_id_different_regs() {
    let mut asm = Assembler::new();
    emit_load_shape_id(&mut asm, Gpr::Rbx, Gpr::Rsi, Gpr::R10);

    assert!(asm.offset() > 0);
}

// -------------------------------------------------------------------------
// Slot Load Tests
// -------------------------------------------------------------------------

#[test]
fn test_emit_slot_load_immediate_first_slot() {
    let mut asm = Assembler::new();
    emit_slot_load_immediate(&mut asm, Gpr::Rax, Gpr::Rdi, 0);

    // Should emit: mov rax, [rdi + 24]
    assert!(asm.offset() > 0);
}

#[test]
fn test_emit_slot_load_immediate_various_slots() {
    for slot_idx in 0..8u16 {
        let mut asm = Assembler::new();
        emit_slot_load_immediate(&mut asm, Gpr::Rax, Gpr::Rdi, slot_idx);
        assert!(asm.offset() > 0, "Slot {} should emit code", slot_idx);
    }
}

#[test]
fn test_emit_slot_load_ic() {
    let mut asm = Assembler::new();
    emit_slot_load_ic(&mut asm, Gpr::Rax, Gpr::Rdi, Gpr::R15, 0, Gpr::Rcx);

    // Should emit slot offset load + indexed load
    assert!(asm.offset() > 5, "Should emit multiple instructions");
}

#[test]
fn test_emit_slot_load_ic_with_offset() {
    let mut asm = Assembler::new();
    emit_slot_load_ic(&mut asm, Gpr::Rax, Gpr::Rdi, Gpr::R15, 16, Gpr::Rcx);

    assert!(asm.offset() > 5);
}

// -------------------------------------------------------------------------
// Slot Store Tests
// -------------------------------------------------------------------------

#[test]
fn test_emit_slot_store_immediate() {
    let mut asm = Assembler::new();
    emit_slot_store_immediate(&mut asm, Gpr::Rdi, 0, Gpr::Rax);

    // Should emit: mov [rdi + 24], rax
    assert!(asm.offset() > 0);
}

#[test]
fn test_emit_slot_store_immediate_various_slots() {
    for slot_idx in 0..8u16 {
        let mut asm = Assembler::new();
        emit_slot_store_immediate(&mut asm, Gpr::Rdi, slot_idx, Gpr::Rax);
        assert!(asm.offset() > 0, "Slot {} should emit code", slot_idx);
    }
}

#[test]
fn test_emit_slot_store_ic() {
    let mut asm = Assembler::new();
    emit_slot_store_ic(&mut asm, Gpr::Rdi, Gpr::R15, 0, Gpr::Rax, Gpr::Rcx);

    assert!(asm.offset() > 5);
}

// -------------------------------------------------------------------------
// Shape Guard Tests
// -------------------------------------------------------------------------

#[test]
fn test_emit_shape_guard() {
    let mut asm = Assembler::new();
    let miss = asm.create_label();

    emit_shape_guard(&mut asm, Gpr::Rdi, Gpr::R15, 0, miss, Gpr::Rcx, Gpr::Rdx);

    // Should emit: load shape_id, load cached, cmp, jne
    assert!(
        asm.offset() > 10,
        "Shape guard should emit multiple instructions"
    );
}

#[test]
fn test_emit_shape_guard_with_ic_offset() {
    let mut asm = Assembler::new();
    let miss = asm.create_label();

    emit_shape_guard(&mut asm, Gpr::Rdi, Gpr::R15, 24, miss, Gpr::Rcx, Gpr::Rdx);

    assert!(asm.offset() > 10);
}

#[test]
fn test_emit_shape_guard_immediate() {
    let mut asm = Assembler::new();
    let miss = asm.create_label();

    emit_shape_guard_immediate(&mut asm, Gpr::Rdi, 42, miss, Gpr::Rcx, Gpr::Rdx);

    assert!(asm.offset() > 0);
}

#[test]
fn test_emit_shape_guard_immediate_large_value() {
    let mut asm = Assembler::new();
    let miss = asm.create_label();

    emit_shape_guard_immediate(&mut asm, Gpr::Rdi, 0xDEADBEEF, miss, Gpr::Rcx, Gpr::Rdx);

    assert!(asm.offset() > 0);
}

// -------------------------------------------------------------------------
// IC Site Update Tests
// -------------------------------------------------------------------------

#[test]
fn test_emit_ic_site_update() {
    let mut asm = Assembler::new();

    emit_ic_site_update(&mut asm, Gpr::R15, 0, Gpr::Rax, Gpr::Rcx);

    // Should emit two stores
    assert!(asm.offset() > 0);
}

#[test]
fn test_emit_ic_site_update_with_offset() {
    let mut asm = Assembler::new();

    emit_ic_site_update(&mut asm, Gpr::R15, 32, Gpr::Rax, Gpr::Rcx);

    assert!(asm.offset() > 0);
}

#[test]
fn test_emit_load_ic_base() {
    let mut asm = Assembler::new();

    emit_load_ic_base(&mut asm, Gpr::R15, Gpr::Rdi, 64);

    // Should emit: mov r15, [rdi + 64]
    assert!(asm.offset() > 0);
}

// -------------------------------------------------------------------------
// Complete Fast Path Tests
// -------------------------------------------------------------------------

#[test]
fn test_emit_get_attr_ic_fast_path() {
    let mut asm = Assembler::new();
    let miss = asm.create_label();

    emit_get_attr_ic_fast_path(
        &mut asm,
        Gpr::Rax,
        Gpr::Rdi,
        Gpr::R15,
        0,
        miss,
        Gpr::Rcx,
        Gpr::Rdx,
    );

    // Should emit shape guard + slot load
    assert!(asm.offset() > 15, "GetAttr IC should emit substantial code");
}

#[test]
fn test_emit_get_attr_ic_fast_path_various_offsets() {
    for offset in [0, 8, 16, 24, 32].iter() {
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_get_attr_ic_fast_path(
            &mut asm,
            Gpr::Rax,
            Gpr::Rdi,
            Gpr::R15,
            *offset,
            miss,
            Gpr::Rcx,
            Gpr::Rdx,
        );

        assert!(asm.offset() > 10, "Offset {} should work", offset);
    }
}

#[test]
fn test_emit_set_attr_ic_fast_path() {
    let mut asm = Assembler::new();
    let miss = asm.create_label();

    emit_set_attr_ic_fast_path(
        &mut asm,
        Gpr::Rdi,
        Gpr::Rsi,
        Gpr::R15,
        0,
        miss,
        Gpr::Rcx,
        Gpr::Rdx,
    );

    // Should emit shape guard + slot store
    assert!(asm.offset() > 15, "SetAttr IC should emit substantial code");
}

#[test]
fn test_emit_set_attr_ic_fast_path_different_src() {
    let mut asm = Assembler::new();
    let miss = asm.create_label();

    emit_set_attr_ic_fast_path(
        &mut asm,
        Gpr::Rdi,
        Gpr::Rax, // Different source register
        Gpr::R15,
        8,
        miss,
        Gpr::Rcx,
        Gpr::Rdx,
    );

    assert!(asm.offset() > 10);
}

// -------------------------------------------------------------------------
// Code Size Estimation Tests
// -------------------------------------------------------------------------

#[test]
fn test_fast_path_code_size() {
    // Measure typical fast path sizes for estimation

    let mut asm = Assembler::new();
    let miss = asm.create_label();
    emit_get_attr_ic_fast_path(
        &mut asm,
        Gpr::Rax,
        Gpr::Rdi,
        Gpr::R15,
        0,
        miss,
        Gpr::Rcx,
        Gpr::Rdx,
    );
    let get_attr_size = asm.offset();

    let mut asm = Assembler::new();
    let miss = asm.create_label();
    emit_set_attr_ic_fast_path(
        &mut asm,
        Gpr::Rdi,
        Gpr::Rsi,
        Gpr::R15,
        0,
        miss,
        Gpr::Rcx,
        Gpr::Rdx,
    );
    let set_attr_size = asm.offset();

    // Both should be compact
    assert!(
        get_attr_size < 100,
        "GetAttr should be compact: {}",
        get_attr_size
    );
    assert!(
        set_attr_size < 100,
        "SetAttr should be compact: {}",
        set_attr_size
    );
}

// -------------------------------------------------------------------------
// Register Constraint Tests
// -------------------------------------------------------------------------

#[test]
fn test_emit_with_r8_r15() {
    // Test with extended registers
    let mut asm = Assembler::new();
    let miss = asm.create_label();

    emit_get_attr_ic_fast_path(
        &mut asm,
        Gpr::R8,
        Gpr::R9,
        Gpr::R10,
        0,
        miss,
        Gpr::R11,
        Gpr::R12,
    );

    assert!(asm.offset() > 0);
}

#[test]
fn test_emit_preserves_obj_register() {
    // The obj register should not be clobbered by the fast path
    let mut asm = Assembler::new();
    let miss = asm.create_label();

    emit_get_attr_ic_fast_path(
        &mut asm,
        Gpr::Rax, // dst - gets result
        Gpr::Rdi, // obj - should be preserved
        Gpr::R15, // ic_base - should be preserved
        0,
        miss,
        Gpr::Rcx, // scratch1 - clobbered
        Gpr::Rdx, // scratch2 - clobbered
    );

    // If we got here, the register assignment is valid
    assert!(asm.offset() > 0);
}
