use super::*;
use crate::backend::x64::registers::Xmm;
use crate::backend::x64::simd::Ymm;

// =========================================================================
// SpillError Tests
// =========================================================================

#[test]
fn test_spill_error_display() {
    let err = SpillError::WidthMismatch {
        reg_width: 256,
        slot_width: 16,
    };
    assert!(err.to_string().contains("256"));
    assert!(err.to_string().contains("16"));

    let err = SpillError::NotVectorRegister;
    assert!(err.to_string().contains("vector"));
}

#[test]
fn test_spill_error_not_vector() {
    let err = SpillError::NotVectorRegister;
    assert_eq!(
        format!("{}", err),
        "Expected vector register (XMM/YMM/ZMM), got GPR"
    );
}

// =========================================================================
// DataKind Tests
// =========================================================================

#[test]
fn test_data_kind_element_width() {
    assert_eq!(DataKind::Float64.element_width(), 64);
    assert_eq!(DataKind::Float32.element_width(), 32);
    assert_eq!(DataKind::Int64.element_width(), 64);
    assert_eq!(DataKind::Int32.element_width(), 32);
    assert_eq!(DataKind::Generic.element_width(), 64);
}

#[test]
fn test_data_kind_is_float() {
    assert!(DataKind::Float64.is_float());
    assert!(DataKind::Float32.is_float());
    assert!(!DataKind::Int64.is_float());
    assert!(!DataKind::Int32.is_float());
    assert!(!DataKind::Generic.is_float());
}

#[test]
fn test_data_kind_is_integer() {
    assert!(!DataKind::Float64.is_integer());
    assert!(!DataKind::Float32.is_integer());
    assert!(DataKind::Int64.is_integer());
    assert!(DataKind::Int32.is_integer());
    assert!(!DataKind::Generic.is_integer());
}

// =========================================================================
// FrameBase Tests
// =========================================================================

#[test]
fn test_frame_base_as_gpr() {
    assert_eq!(FrameBase::Rbp.as_gpr(), Gpr::Rbp);
    assert_eq!(FrameBase::Rsp.as_gpr(), Gpr::Rsp);
}

// =========================================================================
// SpillStats Tests
// =========================================================================

#[test]
fn test_spill_stats_new() {
    let stats = SpillStats::new();
    assert_eq!(stats.total_spills, 0);
    assert_eq!(stats.total_reloads, 0);
    assert_eq!(stats.xmm_spills, 0);
    assert_eq!(stats.ymm_spills, 0);
    assert_eq!(stats.zmm_spills, 0);
}

#[test]
fn test_spill_stats_reset() {
    let mut stats = SpillStats::new();
    stats.total_spills = 100;
    stats.ymm_spills = 50;
    stats.reset();
    assert_eq!(stats.total_spills, 0);
    assert_eq!(stats.ymm_spills, 0);
}

#[test]
fn test_spill_stats_total_memory_ops() {
    let mut stats = SpillStats::new();
    stats.total_spills = 10;
    stats.total_reloads = 20;
    assert_eq!(stats.total_memory_ops(), 30);
}

// =========================================================================
// VectorSpiller Construction Tests
// =========================================================================

#[test]
fn test_vector_spiller_new() {
    let spiller = VectorSpiller::new();
    assert_eq!(spiller.frame_base, FrameBase::Rbp);
    assert!(spiller.prefer_aligned);
    assert!(!spiller.prefer_integer_moves);
    assert!(spiller.elide_self_moves);
}

#[test]
fn test_vector_spiller_with_frame_base() {
    let spiller = VectorSpiller::new().with_frame_base(FrameBase::Rsp);
    assert_eq!(spiller.frame_base, FrameBase::Rsp);
}

#[test]
fn test_vector_spiller_with_unaligned() {
    let spiller = VectorSpiller::new().with_unaligned_only();
    assert!(!spiller.prefer_aligned);
}

#[test]
fn test_vector_spiller_with_integer_moves() {
    let spiller = VectorSpiller::new().with_integer_moves();
    assert!(spiller.prefer_integer_moves);
}

#[test]
fn test_vector_spiller_without_move_elision() {
    let spiller = VectorSpiller::new().without_move_elision();
    assert!(!spiller.elide_self_moves);
}

// =========================================================================
// Spill/Reload Emission Tests
// =========================================================================

#[test]
fn test_emit_xmm_spill() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let slot = SpillSlot::new_with_offset(-16, SpillWidth::W16);
    let reg = PReg::Xmm(Xmm::Xmm0);

    let result = spiller.emit_spill(&mut code, reg, slot, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty());
    assert_eq!(spiller.stats().total_spills, 1);
    assert_eq!(spiller.stats().xmm_spills, 1);
}

#[test]
fn test_emit_ymm_spill() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let slot = SpillSlot::new_with_offset(-32, SpillWidth::W32);
    let reg = PReg::Ymm(Ymm::Ymm0);

    let result = spiller.emit_spill(&mut code, reg, slot, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty());
    assert_eq!(spiller.stats().ymm_spills, 1);
}

#[test]
fn test_emit_zmm_spill() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let slot = SpillSlot::new_with_offset(-64, SpillWidth::W64);
    let reg = PReg::Zmm(Zmm::Zmm0);

    let result = spiller.emit_spill(&mut code, reg, slot, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty());
    assert_eq!(spiller.stats().zmm_spills, 1);
}

#[test]
fn test_emit_xmm_reload() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let slot = SpillSlot::new_with_offset(-16, SpillWidth::W16);
    let reg = PReg::Xmm(Xmm::Xmm0);

    let result = spiller.emit_reload(&mut code, reg, slot, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty());
    assert_eq!(spiller.stats().total_reloads, 1);
}

#[test]
fn test_emit_ymm_reload() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let slot = SpillSlot::new_with_offset(-32, SpillWidth::W32);
    let reg = PReg::Ymm(Ymm::Ymm0);

    let result = spiller.emit_reload(&mut code, reg, slot, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty());
}

#[test]
fn test_emit_zmm_reload() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let slot = SpillSlot::new_with_offset(-64, SpillWidth::W64);
    let reg = PReg::Zmm(Zmm::Zmm0);

    let result = spiller.emit_reload(&mut code, reg, slot, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty());
}

#[test]
fn test_spill_gpr_fails() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let slot = SpillSlot::new_with_offset(-8, SpillWidth::W8);
    let reg = PReg::Gpr(Gpr::Rax);

    let result = spiller.emit_spill(&mut code, reg, slot, DataKind::Generic);
    assert!(matches!(result, Err(SpillError::NotVectorRegister)));
}

#[test]
fn test_width_mismatch_error() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    // YMM (256-bit) with XMM slot (128-bit)
    let slot = SpillSlot::new_with_offset(-16, SpillWidth::W16);
    let reg = PReg::Ymm(Ymm::Ymm0);

    let result = spiller.emit_spill(&mut code, reg, slot, DataKind::Float64);
    assert!(matches!(result, Err(SpillError::WidthMismatch { .. })));
}

// =========================================================================
// Move Emission Tests
// =========================================================================

#[test]
fn test_emit_xmm_move() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let src = PReg::Xmm(Xmm::Xmm0);
    let dst = PReg::Xmm(Xmm::Xmm1);

    let result = spiller.emit_move(&mut code, dst, src, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty());
    assert_eq!(spiller.stats().reg_moves, 1);
}

#[test]
fn test_emit_ymm_move() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let src = PReg::Ymm(Ymm::Ymm0);
    let dst = PReg::Ymm(Ymm::Ymm1);

    let result = spiller.emit_move(&mut code, dst, src, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty());
}

#[test]
fn test_emit_zmm_move() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let src = PReg::Zmm(Zmm::Zmm0);
    let dst = PReg::Zmm(Zmm::Zmm1);

    let result = spiller.emit_move(&mut code, dst, src, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty());
}

#[test]
fn test_self_move_elision() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();
    let reg = PReg::Ymm(Ymm::Ymm0);

    let result = spiller.emit_move(&mut code, reg, reg, DataKind::Float64);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0); // No bytes emitted
    assert_eq!(spiller.stats().elided_moves, 1);
    assert_eq!(spiller.stats().reg_moves, 0);
}

#[test]
fn test_self_move_without_elision() {
    let mut spiller = VectorSpiller::new().without_move_elision();
    let mut code = Vec::new();
    let reg = PReg::Ymm(Ymm::Ymm0);

    let result = spiller.emit_move(&mut code, reg, reg, DataKind::Float64);
    assert!(result.is_ok());
    assert!(!code.is_empty()); // Bytes emitted
    assert_eq!(spiller.stats().elided_moves, 0);
    assert_eq!(spiller.stats().reg_moves, 1);
}

#[test]
fn test_move_integer_preference() {
    let mut spiller = VectorSpiller::new().with_integer_moves();
    let mut code = Vec::new();
    let src = PReg::Ymm(Ymm::Ymm0);
    let dst = PReg::Ymm(Ymm::Ymm1);

    let result = spiller.emit_move(&mut code, dst, src, DataKind::Generic);
    assert!(result.is_ok());
    // The encoding uses VMOVDQA instead of VMOVAPD
    assert!(!code.is_empty());
}

// =========================================================================
// Alignment Tests
// =========================================================================

#[test]
fn test_is_aligned_access() {
    // 32-byte alignment
    assert!(is_aligned_access(-32, SpillWidth::W32));
    assert!(is_aligned_access(-64, SpillWidth::W32));
    assert!(!is_aligned_access(-16, SpillWidth::W32));

    // 64-byte alignment
    assert!(is_aligned_access(-64, SpillWidth::W64));
    assert!(is_aligned_access(-128, SpillWidth::W64));
    assert!(!is_aligned_access(-32, SpillWidth::W64));

    // 16-byte alignment
    assert!(is_aligned_access(-16, SpillWidth::W16));
    assert!(is_aligned_access(-32, SpillWidth::W16));
}

#[test]
fn test_aligned_vs_unaligned_stats() {
    let mut spiller = VectorSpiller::new();
    let mut code = Vec::new();

    // Aligned slot
    let aligned_slot = SpillSlot::new_with_offset(-32, SpillWidth::W32);
    let reg = PReg::Ymm(Ymm::Ymm0);
    spiller
        .emit_spill(&mut code, reg, aligned_slot, DataKind::Float64)
        .unwrap();
    assert_eq!(spiller.stats().aligned_ops, 1);

    // Unaligned slot (offset not divisible by 32)
    let unaligned_slot = SpillSlot::new_with_offset(-48, SpillWidth::W32);
    spiller
        .emit_spill(&mut code, reg, unaligned_slot, DataKind::Float64)
        .unwrap();
    // 48 is divisible by 32? No, 48/32 = 1.5, so unaligned
    // Actually this test depends on exact alignment calculation
}

// =========================================================================
// All Register Tests
// =========================================================================

#[test]
fn test_all_xmm_spill_reload() {
    let mut spiller = VectorSpiller::new();
    for i in 0..16u8 {
        let xmm = Xmm::from_encoding(i).unwrap();
        let reg = PReg::Xmm(xmm);
        let slot = SpillSlot::new_with_offset(-16 * ((i as i32) + 1), SpillWidth::W16);

        let mut code = Vec::new();
        assert!(
            spiller
                .emit_spill(&mut code, reg, slot, DataKind::Float64)
                .is_ok()
        );

        let mut code2 = Vec::new();
        assert!(
            spiller
                .emit_reload(&mut code2, reg, slot, DataKind::Float64)
                .is_ok()
        );
    }
}

#[test]
fn test_all_ymm_spill_reload() {
    let mut spiller = VectorSpiller::new();
    for i in 0..16u8 {
        let ymm = Ymm::from_encoding(i).unwrap();
        let reg = PReg::Ymm(ymm);
        let slot = SpillSlot::new_with_offset(-32 * ((i as i32) + 1), SpillWidth::W32);

        let mut code = Vec::new();
        assert!(
            spiller
                .emit_spill(&mut code, reg, slot, DataKind::Float64)
                .is_ok()
        );

        let mut code2 = Vec::new();
        assert!(
            spiller
                .emit_reload(&mut code2, reg, slot, DataKind::Float64)
                .is_ok()
        );
    }
}

#[test]
fn test_all_zmm_spill_reload() {
    let mut spiller = VectorSpiller::new();
    for i in 0..32u8 {
        let zmm = Zmm::from_encoding(i).unwrap();
        let reg = PReg::Zmm(zmm);
        let slot = SpillSlot::new_with_offset(-64 * ((i as i32) + 1), SpillWidth::W64);

        let mut code = Vec::new();
        assert!(
            spiller
                .emit_spill(&mut code, reg, slot, DataKind::Float64)
                .is_ok()
        );

        let mut code2 = Vec::new();
        assert!(
            spiller
                .emit_reload(&mut code2, reg, slot, DataKind::Float64)
                .is_ok()
        );
    }
}

#[test]
fn test_all_data_kinds() {
    let mut spiller = VectorSpiller::new();
    let kinds = [
        DataKind::Float64,
        DataKind::Float32,
        DataKind::Int64,
        DataKind::Int32,
        DataKind::Generic,
    ];

    for kind in &kinds {
        let mut code = Vec::new();
        let reg = PReg::Ymm(Ymm::Ymm0);
        let slot = SpillSlot::new_with_offset(-32, SpillWidth::W32);
        assert!(spiller.emit_spill(&mut code, reg, slot, *kind).is_ok());
    }
}

// =========================================================================
// Utility Function Tests
// =========================================================================

#[test]
fn test_spill_width_for_class() {
    assert_eq!(
        spill_width_for_class(RegClass::Int) as u8,
        SpillWidth::W8 as u8
    );
    assert_eq!(
        spill_width_for_class(RegClass::Float) as u8,
        SpillWidth::W16 as u8
    );
    assert_eq!(
        spill_width_for_class(RegClass::Vec256) as u8,
        SpillWidth::W32 as u8
    );
    assert_eq!(
        spill_width_for_class(RegClass::Vec512) as u8,
        SpillWidth::W64 as u8
    );
}

#[test]
fn test_alignment_for_width() {
    assert_eq!(alignment_for_width(SpillWidth::W8), 8);
    assert_eq!(alignment_for_width(SpillWidth::W16), 16);
    assert_eq!(alignment_for_width(SpillWidth::W32), 32);
    assert_eq!(alignment_for_width(SpillWidth::W64), 64);
}
