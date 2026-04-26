use super::*;

#[test]
fn test_type_identity() {
    assert_eq!(ValueType::Int64.meet(ValueType::Int64), ValueType::Int64);
    assert_eq!(
        ValueType::Float64.meet(ValueType::Float64),
        ValueType::Float64
    );
}

#[test]
fn test_type_meet_numeric() {
    assert_eq!(
        ValueType::Int64.meet(ValueType::Float64),
        ValueType::Numeric
    );
    assert_eq!(
        ValueType::Float64.meet(ValueType::Int64),
        ValueType::Numeric
    );
    assert_eq!(
        ValueType::Int64.meet(ValueType::Numeric),
        ValueType::Numeric
    );
}

#[test]
fn test_type_meet_top_bottom() {
    assert_eq!(ValueType::Top.meet(ValueType::Int64), ValueType::Int64);
    assert_eq!(ValueType::Int64.meet(ValueType::Top), ValueType::Int64);
    assert_eq!(ValueType::Bottom.meet(ValueType::Int64), ValueType::Bottom);
}

#[test]
fn test_type_subtype() {
    assert!(ValueType::Int64.is_subtype_of(ValueType::Numeric));
    assert!(ValueType::Float64.is_subtype_of(ValueType::Numeric));
    assert!(ValueType::List.is_subtype_of(ValueType::Sequence));
    assert!(ValueType::List.is_subtype_of(ValueType::Object));
    assert!(ValueType::Bottom.is_subtype_of(ValueType::Int64));
    assert!(ValueType::Int64.is_subtype_of(ValueType::Top));
}

#[test]
fn test_type_tuple() {
    let single = TypeTuple::single(ValueType::Int64);
    assert_eq!(single.len(), 1);
    assert_eq!(single.get(0), Some(ValueType::Int64));

    let pair = TypeTuple::pair(ValueType::Int64, ValueType::Int64);
    assert_eq!(pair.len(), 2);
}

// =========================================================================
// Vector Type Tests
// =========================================================================

#[test]
fn test_vector_type_is_vector_128bit() {
    // 128-bit vectors
    assert!(ValueType::V2I64.is_vector());
    assert!(ValueType::V4I32.is_vector());
    assert!(ValueType::V8I16.is_vector());
    assert!(ValueType::V16I8.is_vector());
    assert!(ValueType::V2F64.is_vector());
    assert!(ValueType::V4F32.is_vector());
}

#[test]
fn test_vector_type_is_vector_256bit() {
    // 256-bit vectors
    assert!(ValueType::V4I64.is_vector());
    assert!(ValueType::V8I32.is_vector());
    assert!(ValueType::V16I16.is_vector());
    assert!(ValueType::V32I8.is_vector());
    assert!(ValueType::V4F64.is_vector());
    assert!(ValueType::V8F32.is_vector());
}

#[test]
fn test_vector_type_is_vector_512bit() {
    // 512-bit vectors
    assert!(ValueType::V8I64.is_vector());
    assert!(ValueType::V16I32.is_vector());
    assert!(ValueType::V8F64.is_vector());
    assert!(ValueType::V16F32.is_vector());
}

#[test]
fn test_scalar_types_are_not_vectors() {
    assert!(!ValueType::Int64.is_vector());
    assert!(!ValueType::Float64.is_vector());
    assert!(!ValueType::Bool.is_vector());
    assert!(!ValueType::Object.is_vector());
    assert!(!ValueType::Top.is_vector());
    assert!(!ValueType::Bottom.is_vector());
}

#[test]
fn test_vector_lanes_128bit() {
    assert_eq!(ValueType::V2I64.lanes(), 2);
    assert_eq!(ValueType::V4I32.lanes(), 4);
    assert_eq!(ValueType::V8I16.lanes(), 8);
    assert_eq!(ValueType::V16I8.lanes(), 16);
    assert_eq!(ValueType::V2F64.lanes(), 2);
    assert_eq!(ValueType::V4F32.lanes(), 4);
}

#[test]
fn test_vector_lanes_256bit() {
    assert_eq!(ValueType::V4I64.lanes(), 4);
    assert_eq!(ValueType::V8I32.lanes(), 8);
    assert_eq!(ValueType::V16I16.lanes(), 16);
    assert_eq!(ValueType::V32I8.lanes(), 32);
    assert_eq!(ValueType::V4F64.lanes(), 4);
    assert_eq!(ValueType::V8F32.lanes(), 8);
}

#[test]
fn test_vector_lanes_512bit() {
    assert_eq!(ValueType::V8I64.lanes(), 8);
    assert_eq!(ValueType::V16I32.lanes(), 16);
    assert_eq!(ValueType::V8F64.lanes(), 8);
    assert_eq!(ValueType::V16F32.lanes(), 16);
}

#[test]
fn test_scalar_lanes() {
    assert_eq!(ValueType::Int64.lanes(), 1);
    assert_eq!(ValueType::Float64.lanes(), 1);
    assert_eq!(ValueType::Bool.lanes(), 1);
}

#[test]
fn test_vector_element_type() {
    // Integer vectors -> Int64
    assert_eq!(ValueType::V2I64.element_type(), ValueType::Int64);
    assert_eq!(ValueType::V4I64.element_type(), ValueType::Int64);
    assert_eq!(ValueType::V8I64.element_type(), ValueType::Int64);
    assert_eq!(ValueType::V4I32.element_type(), ValueType::Int64);
    assert_eq!(ValueType::V8I32.element_type(), ValueType::Int64);
    assert_eq!(ValueType::V16I32.element_type(), ValueType::Int64);

    // Float vectors -> Float64
    assert_eq!(ValueType::V2F64.element_type(), ValueType::Float64);
    assert_eq!(ValueType::V4F64.element_type(), ValueType::Float64);
    assert_eq!(ValueType::V8F64.element_type(), ValueType::Float64);
    assert_eq!(ValueType::V4F32.element_type(), ValueType::Float64);
}

#[test]
fn test_scalar_element_type_is_self() {
    assert_eq!(ValueType::Int64.element_type(), ValueType::Int64);
    assert_eq!(ValueType::Float64.element_type(), ValueType::Float64);
    assert_eq!(ValueType::Bool.element_type(), ValueType::Bool);
}

#[test]
fn test_vector_bit_width() {
    // 128-bit
    assert_eq!(ValueType::V2I64.bit_width(), 128);
    assert_eq!(ValueType::V4I32.bit_width(), 128);
    assert_eq!(ValueType::V2F64.bit_width(), 128);

    // 256-bit
    assert_eq!(ValueType::V4I64.bit_width(), 256);
    assert_eq!(ValueType::V8I32.bit_width(), 256);
    assert_eq!(ValueType::V4F64.bit_width(), 256);

    // 512-bit
    assert_eq!(ValueType::V8I64.bit_width(), 512);
    assert_eq!(ValueType::V16I32.bit_width(), 512);
    assert_eq!(ValueType::V8F64.bit_width(), 512);
}

#[test]
fn test_scalar_bit_width() {
    assert_eq!(ValueType::Int64.bit_width(), 64);
    assert_eq!(ValueType::Float64.bit_width(), 64);
    assert_eq!(ValueType::Bool.bit_width(), 8);
}

#[test]
fn test_vector_of_i64() {
    assert_eq!(
        ValueType::vector_of(ValueType::Int64, 2),
        Some(ValueType::V2I64)
    );
    assert_eq!(
        ValueType::vector_of(ValueType::Int64, 4),
        Some(ValueType::V4I64)
    );
    assert_eq!(
        ValueType::vector_of(ValueType::Int64, 8),
        Some(ValueType::V8I64)
    );
}

#[test]
fn test_vector_of_f64() {
    assert_eq!(
        ValueType::vector_of(ValueType::Float64, 2),
        Some(ValueType::V2F64)
    );
    assert_eq!(
        ValueType::vector_of(ValueType::Float64, 4),
        Some(ValueType::V4F64)
    );
    assert_eq!(
        ValueType::vector_of(ValueType::Float64, 8),
        Some(ValueType::V8F64)
    );
}

#[test]
fn test_vector_of_unsupported() {
    // Unsupported lane counts
    assert_eq!(ValueType::vector_of(ValueType::Int64, 1), None);
    assert_eq!(ValueType::vector_of(ValueType::Int64, 3), None);
    assert_eq!(ValueType::vector_of(ValueType::Int64, 16), None);

    // Unsupported element types
    assert_eq!(ValueType::vector_of(ValueType::Bool, 2), None);
    assert_eq!(ValueType::vector_of(ValueType::Object, 4), None);
}

#[test]
fn test_vector_roundtrip() {
    // Create vector, then verify element type and lanes
    let v = ValueType::V4F64;
    assert_eq!(v.element_type(), ValueType::Float64);
    assert_eq!(v.lanes(), 4);
    assert_eq!(v.bit_width(), 256);

    // Recreate from element type and lanes
    let recreated = ValueType::vector_of(v.element_type(), v.lanes());
    assert_eq!(recreated, Some(v));
}

#[test]
fn test_vector_debug_formatting() {
    // 128-bit
    assert_eq!(format!("{:?}", ValueType::V2I64), "v2i64");
    assert_eq!(format!("{:?}", ValueType::V4I32), "v4i32");
    assert_eq!(format!("{:?}", ValueType::V2F64), "v2f64");
    assert_eq!(format!("{:?}", ValueType::V4F32), "v4f32");

    // 256-bit
    assert_eq!(format!("{:?}", ValueType::V4I64), "v4i64");
    assert_eq!(format!("{:?}", ValueType::V8I32), "v8i32");
    assert_eq!(format!("{:?}", ValueType::V4F64), "v4f64");

    // 512-bit
    assert_eq!(format!("{:?}", ValueType::V8I64), "v8i64");
    assert_eq!(format!("{:?}", ValueType::V16I32), "v16i32");
    assert_eq!(format!("{:?}", ValueType::V8F64), "v8f64");
}

#[test]
fn test_all_vector_types_have_consistent_width() {
    // Verify lanes * element_width = bit_width

    // 128-bit: 2 * 64 = 128
    assert_eq!(
        ValueType::V2I64.lanes() as u16 * 64,
        ValueType::V2I64.bit_width()
    );
    assert_eq!(
        ValueType::V2F64.lanes() as u16 * 64,
        ValueType::V2F64.bit_width()
    );

    // 128-bit: 4 * 32 = 128
    assert_eq!(
        ValueType::V4I32.lanes() as u16 * 32,
        ValueType::V4I32.bit_width()
    );
    assert_eq!(
        ValueType::V4F32.lanes() as u16 * 32,
        ValueType::V4F32.bit_width()
    );

    // 256-bit: 4 * 64 = 256
    assert_eq!(
        ValueType::V4I64.lanes() as u16 * 64,
        ValueType::V4I64.bit_width()
    );
    assert_eq!(
        ValueType::V4F64.lanes() as u16 * 64,
        ValueType::V4F64.bit_width()
    );

    // 512-bit: 8 * 64 = 512
    assert_eq!(
        ValueType::V8I64.lanes() as u16 * 64,
        ValueType::V8I64.bit_width()
    );
    assert_eq!(
        ValueType::V8F64.lanes() as u16 * 64,
        ValueType::V8F64.bit_width()
    );
}
