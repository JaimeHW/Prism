use super::*;
use crate::ir::builder::{
    ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
};

#[test]
fn test_liveness_simple() {
    let mut builder = GraphBuilder::new(4, 2);

    let p0 = builder.parameter(0).unwrap();
    let p1 = builder.parameter(1).unwrap();
    let sum = builder.int_add(p0, p1);
    let _ret = builder.return_value(sum);

    let graph = builder.finish();
    let analysis = LivenessAnalysis::analyze(&graph);

    // Should have vregs for parameters and sum
    assert!(analysis.vreg_count() >= 3);
}

#[test]
fn test_liveness_constants() {
    let mut builder = GraphBuilder::new(4, 0);

    let a = builder.const_int(10);
    let b = builder.const_int(20);
    let sum = builder.int_add(a, b);
    let _ret = builder.return_value(sum);

    let graph = builder.finish();
    let analysis = LivenessAnalysis::analyze(&graph);

    // Both constants and sum should have vregs
    assert!(analysis.vreg_count() >= 3);
}

#[test]
fn test_liveness_interval_ranges() {
    let mut builder = GraphBuilder::new(4, 1);

    let p0 = builder.parameter(0).unwrap();
    let one = builder.const_int(1);
    let add1 = builder.int_add(p0, one);
    let add2 = builder.int_add(add1, one);
    let _ret = builder.return_value(add2);

    let graph = builder.finish();
    let analysis = LivenessAnalysis::analyze(&graph);

    // `one` should have a live range spanning both additions
    if let Some(vreg) = analysis.vreg_for_node(one) {
        let interval = analysis.interval(vreg).unwrap();
        assert!(!interval.is_empty());
        // The constant is used in two places, so should have uses
        assert!(interval.uses().len() >= 2);
    }
}

// =========================================================================
// Vector Register Class Tests
// =========================================================================

#[test]
fn test_reg_class_for_int64() {
    // Int64 should map to Int (GPR)
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::Int64),
        RegClass::Int
    );
}

#[test]
fn test_reg_class_for_float64() {
    // Float64 should map to Float (XMM)
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::Float64),
        RegClass::Float
    );
}

#[test]
fn test_reg_class_for_128bit_vectors() {
    // All 128-bit vectors should map to Float (XMM)
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V2I64),
        RegClass::Float
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V4I32),
        RegClass::Float
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V8I16),
        RegClass::Float
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V16I8),
        RegClass::Float
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V2F64),
        RegClass::Float
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V4F32),
        RegClass::Float
    );
}

#[test]
fn test_reg_class_for_256bit_vectors() {
    // All 256-bit vectors should map to Vec256 (YMM)
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V4I64),
        RegClass::Vec256
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V8I32),
        RegClass::Vec256
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V16I16),
        RegClass::Vec256
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V32I8),
        RegClass::Vec256
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V4F64),
        RegClass::Vec256
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V8F32),
        RegClass::Vec256
    );
}

#[test]
fn test_reg_class_for_512bit_vectors() {
    // All 512-bit vectors should map to Vec512 (ZMM)
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V8I64),
        RegClass::Vec512
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V16I32),
        RegClass::Vec512
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V8F64),
        RegClass::Vec512
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::V16F32),
        RegClass::Vec512
    );
}

#[test]
fn test_reg_class_for_bool() {
    // Bool should map to Int (GPR)
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::Bool),
        RegClass::Int
    );
}

#[test]
fn test_reg_class_for_objects() {
    // Object types should map to Int (GPR) - they're pointers
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::Object),
        RegClass::Int
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::List),
        RegClass::Int
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::Tuple),
        RegClass::Int
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::String),
        RegClass::Int
    );
}

#[test]
fn test_reg_class_for_special_types() {
    // Top/Bottom and side-effect types should map to Int
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::Top),
        RegClass::Int
    );
    assert_eq!(
        LivenessAnalysis::reg_class_for_type(ValueType::Bottom),
        RegClass::Int
    );
}

#[test]
fn test_vector_reg_class_width_consistency() {
    // Verify that reg_class width matches ValueType bit_width for vectors
    let test_cases = [
        (ValueType::V2I64, 128),
        (ValueType::V4I64, 256),
        (ValueType::V8I64, 512),
        (ValueType::V2F64, 128),
        (ValueType::V4F64, 256),
        (ValueType::V8F64, 512),
    ];

    for (ty, expected_bits) in test_cases {
        let reg_class = LivenessAnalysis::reg_class_for_type(ty);
        assert_eq!(
            reg_class.width(),
            expected_bits,
            "Mismatch for {:?}: reg_class {:?} has width {} but type has {} bits",
            ty,
            reg_class,
            reg_class.width(),
            expected_bits
        );
    }
}

#[test]
fn test_vector_reg_class_spill_size_matches() {
    // Verify spill sizes are correct for vector types
    let test_cases = [
        (ValueType::Int64, 8),    // GPR: 8 bytes
        (ValueType::Float64, 16), // XMM: 16 bytes
        (ValueType::V2I64, 16),   // XMM: 16 bytes
        (ValueType::V4I64, 32),   // YMM: 32 bytes
        (ValueType::V8I64, 64),   // ZMM: 64 bytes
    ];

    for (ty, expected_spill_bytes) in test_cases {
        let reg_class = LivenessAnalysis::reg_class_for_type(ty);
        assert_eq!(
            reg_class.spill_size(),
            expected_spill_bytes,
            "{:?} should have spill size {} but got {}",
            ty,
            expected_spill_bytes,
            reg_class.spill_size()
        );
    }
}

#[test]
fn test_vector_type_is_vector_class() {
    // All 128+ bit vector types should have is_vector() true for their reg_class
    let vector_types = [
        ValueType::V2I64,
        ValueType::V4I64,
        ValueType::V8I64,
        ValueType::V2F64,
        ValueType::V4F64,
        ValueType::V8F64,
    ];

    for ty in vector_types {
        let reg_class = LivenessAnalysis::reg_class_for_type(ty);
        assert!(
            reg_class.is_vector(),
            "{:?} should map to a vector register class, but got {:?}",
            ty,
            reg_class
        );
    }
}

#[test]
fn test_wide_vector_class_256_512() {
    // 256+ bit vectors should have is_wide_vector() true
    let wide_types = [
        ValueType::V4I64,
        ValueType::V8I64,
        ValueType::V4F64,
        ValueType::V8F64,
    ];

    for ty in wide_types {
        let reg_class = LivenessAnalysis::reg_class_for_type(ty);
        assert!(
            reg_class.is_wide_vector(),
            "{:?} should map to a wide vector class, but got {:?}",
            ty,
            reg_class
        );
    }

    // 128-bit vectors are NOT wide
    let narrow_types = [ValueType::V2I64, ValueType::V2F64];
    for ty in narrow_types {
        let reg_class = LivenessAnalysis::reg_class_for_type(ty);
        assert!(
            !reg_class.is_wide_vector(),
            "{:?} should NOT map to a wide vector class, but got {:?}",
            ty,
            reg_class
        );
    }
}
