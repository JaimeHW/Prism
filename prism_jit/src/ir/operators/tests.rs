use super::*;

#[test]
fn test_arith_op_commutative() {
    assert!(ArithOp::Add.is_commutative());
    assert!(ArithOp::Mul.is_commutative());
    assert!(!ArithOp::Sub.is_commutative());
    assert!(!ArithOp::TrueDiv.is_commutative());
}

#[test]
fn test_arith_op_identity() {
    assert_eq!(ArithOp::Add.identity(), Some(0));
    assert_eq!(ArithOp::Mul.identity(), Some(1));
    assert_eq!(ArithOp::Pow.identity(), Some(1));
}

#[test]
fn test_cmp_op_inverse() {
    assert_eq!(CmpOp::Lt.inverse(), CmpOp::Ge);
    assert_eq!(CmpOp::Eq.inverse(), CmpOp::Ne);
    assert_eq!(CmpOp::Ne.inverse(), CmpOp::Eq);
}

#[test]
fn test_cmp_op_swap() {
    assert_eq!(CmpOp::Lt.swap(), CmpOp::Gt);
    assert_eq!(CmpOp::Le.swap(), CmpOp::Ge);
    assert_eq!(CmpOp::Eq.swap(), CmpOp::Eq);
}

#[test]
fn test_operator_category() {
    assert_eq!(Operator::ConstInt(42).category(), OpCategory::Constant);
    assert_eq!(
        Operator::IntOp(ArithOp::Add).category(),
        OpCategory::Arithmetic
    );
    assert_eq!(
        Operator::IntCmp(CmpOp::Lt).category(),
        OpCategory::Comparison
    );
    assert_eq!(Operator::Phi.category(), OpCategory::Phi);
}

#[test]
fn test_operator_pure() {
    assert!(Operator::ConstInt(42).is_pure());
    assert!(Operator::IntOp(ArithOp::Add).is_pure());
    assert!(Operator::Phi.is_pure());
    assert!(!Operator::Call(CallKind::Direct).is_pure());
    assert!(!Operator::Memory(MemoryOp::StoreField).is_pure());
}

#[test]
fn test_operator_commutative() {
    assert!(Operator::IntOp(ArithOp::Add).is_commutative());
    assert!(!Operator::IntOp(ArithOp::Sub).is_commutative());
    assert!(Operator::IntCmp(CmpOp::Eq).is_commutative());
    assert!(!Operator::IntCmp(CmpOp::Lt).is_commutative());
}

// =========================================================================
// Vector Operator Tests
// =========================================================================

#[test]
fn test_vector_op_constants() {
    assert_eq!(VectorOp::V2I64.element, ValueType::Int64);
    assert_eq!(VectorOp::V2I64.lanes, 2);
    assert_eq!(VectorOp::V4I64.lanes, 4);
    assert_eq!(VectorOp::V8I64.lanes, 8);
    assert_eq!(VectorOp::V2F64.element, ValueType::Float64);
    assert_eq!(VectorOp::V4F64.lanes, 4);
    assert_eq!(VectorOp::V8F64.lanes, 8);
}

#[test]
fn test_vector_op_bit_width() {
    assert_eq!(VectorOp::V2I64.bit_width(), 128);
    assert_eq!(VectorOp::V4I64.bit_width(), 256);
    assert_eq!(VectorOp::V8I64.bit_width(), 512);
    assert_eq!(VectorOp::V2F64.bit_width(), 128);
    assert_eq!(VectorOp::V4F64.bit_width(), 256);
    assert_eq!(VectorOp::V8F64.bit_width(), 512);
}

#[test]
fn test_vector_op_value_type() {
    assert_eq!(VectorOp::V2I64.value_type(), ValueType::V2I64);
    assert_eq!(VectorOp::V4I64.value_type(), ValueType::V4I64);
    assert_eq!(VectorOp::V8I64.value_type(), ValueType::V8I64);
    assert_eq!(VectorOp::V2F64.value_type(), ValueType::V2F64);
    assert_eq!(VectorOp::V4F64.value_type(), ValueType::V4F64);
    assert_eq!(VectorOp::V8F64.value_type(), ValueType::V8F64);
}

#[test]
fn test_vector_op_is_float_integer() {
    assert!(!VectorOp::V2I64.is_float());
    assert!(VectorOp::V2I64.is_integer());
    assert!(VectorOp::V2F64.is_float());
    assert!(!VectorOp::V2F64.is_integer());
}

#[test]
fn test_vector_arith_kind_commutative() {
    assert!(VectorArithKind::Add.is_commutative());
    assert!(VectorArithKind::Mul.is_commutative());
    assert!(VectorArithKind::Min.is_commutative());
    assert!(VectorArithKind::Max.is_commutative());
    assert!(!VectorArithKind::Sub.is_commutative());
    assert!(!VectorArithKind::Div.is_commutative());
}

#[test]
fn test_vector_arith_kind_unary() {
    assert!(VectorArithKind::Abs.is_unary());
    assert!(VectorArithKind::Neg.is_unary());
    assert!(VectorArithKind::Sqrt.is_unary());
    assert!(!VectorArithKind::Add.is_unary());
    assert!(!VectorArithKind::Mul.is_unary());
}

#[test]
fn test_vector_shuffle_identity() {
    let shuffle = VectorShuffle::identity(4);
    assert_eq!(shuffle.lanes, 4);
    assert_eq!(shuffle.indices[0], 0);
    assert_eq!(shuffle.indices[1], 1);
    assert_eq!(shuffle.indices[2], 2);
    assert_eq!(shuffle.indices[3], 3);
}

#[test]
fn test_vector_shuffle_broadcast() {
    let shuffle = VectorShuffle::broadcast(4);
    assert_eq!(shuffle.lanes, 4);
    assert_eq!(shuffle.indices[0], 0);
    assert_eq!(shuffle.indices[1], 0);
    assert_eq!(shuffle.indices[2], 0);
    assert_eq!(shuffle.indices[3], 0);
}

#[test]
fn test_vector_shuffle_reverse() {
    let shuffle = VectorShuffle::reverse(4);
    assert_eq!(shuffle.lanes, 4);
    assert_eq!(shuffle.indices[0], 3);
    assert_eq!(shuffle.indices[1], 2);
    assert_eq!(shuffle.indices[2], 1);
    assert_eq!(shuffle.indices[3], 0);
}

#[test]
fn test_vector_operator_category() {
    let vadd = Operator::VectorArith(VectorOp::V4F64, VectorArithKind::Add);
    assert_eq!(vadd.category(), OpCategory::Vector);

    let vload = Operator::VectorMemory(VectorOp::V4F64, VectorMemoryKind::LoadAligned);
    assert_eq!(vload.category(), OpCategory::Vector);

    let vbcast = Operator::VectorBroadcast(VectorOp::V4F64);
    assert_eq!(vbcast.category(), OpCategory::Vector);
}

#[test]
fn test_vector_operator_pure() {
    // Pure vector operations
    assert!(Operator::VectorArith(VectorOp::V4F64, VectorArithKind::Add).is_pure());
    assert!(Operator::VectorFma(VectorOp::V4F64).is_pure());
    assert!(Operator::VectorBroadcast(VectorOp::V4F64).is_pure());
    assert!(Operator::VectorExtract(VectorOp::V4F64, 0).is_pure());
    assert!(Operator::VectorHadd(VectorOp::V4F64).is_pure());
    assert!(Operator::VectorSplat(VectorOp::V4F64, 0).is_pure());

    // Vector memory operations have side effects
    assert!(!Operator::VectorMemory(VectorOp::V4F64, VectorMemoryKind::LoadAligned).is_pure());
    assert!(!Operator::VectorMemory(VectorOp::V4F64, VectorMemoryKind::StoreAligned).is_pure());
}

#[test]
fn test_vector_operator_commutative() {
    // Commutative vector operations
    let vadd = Operator::VectorArith(VectorOp::V4F64, VectorArithKind::Add);
    assert!(vadd.is_commutative());

    let vmul = Operator::VectorArith(VectorOp::V4F64, VectorArithKind::Mul);
    assert!(vmul.is_commutative());

    // Non-commutative vector operations
    let vsub = Operator::VectorArith(VectorOp::V4F64, VectorArithKind::Sub);
    assert!(!vsub.is_commutative());

    let vdiv = Operator::VectorArith(VectorOp::V4F64, VectorArithKind::Div);
    assert!(!vdiv.is_commutative());
}

#[test]
fn test_vector_operator_result_type() {
    // Vector arithmetic returns vector type
    let vadd = Operator::VectorArith(VectorOp::V4F64, VectorArithKind::Add);
    assert_eq!(vadd.result_type(&[]), ValueType::V4F64);

    // Vector extract returns scalar
    let vextract = Operator::VectorExtract(VectorOp::V4F64, 1);
    assert_eq!(vextract.result_type(&[]), ValueType::Float64);

    // Horizontal add returns scalar
    let vhadd = Operator::VectorHadd(VectorOp::V4I64);
    assert_eq!(vhadd.result_type(&[]), ValueType::Int64);

    // Vector load returns vector
    let vload = Operator::VectorMemory(VectorOp::V4F64, VectorMemoryKind::LoadAligned);
    assert_eq!(vload.result_type(&[]), ValueType::V4F64);

    // Vector store returns effect
    let vstore = Operator::VectorMemory(VectorOp::V4F64, VectorMemoryKind::StoreAligned);
    assert_eq!(vstore.result_type(&[]), ValueType::Effect);
}

#[test]
fn test_vector_operator_128bit() {
    let vop = VectorOp::V2F64;
    let vadd = Operator::VectorArith(vop, VectorArithKind::Add);
    assert_eq!(vadd.result_type(&[]), ValueType::V2F64);
    assert_eq!(vop.bit_width(), 128);
}

#[test]
fn test_vector_operator_512bit() {
    let vop = VectorOp::V8F64;
    let vadd = Operator::VectorArith(vop, VectorArithKind::Add);
    assert_eq!(vadd.result_type(&[]), ValueType::V8F64);
    assert_eq!(vop.bit_width(), 512);
}

#[test]
fn test_vector_fma_operator() {
    let vfma = Operator::VectorFma(VectorOp::V4F64);
    assert!(vfma.is_pure());
    assert_eq!(vfma.category(), OpCategory::Vector);
    assert_eq!(vfma.result_type(&[]), ValueType::V4F64);
}

#[test]
fn test_vector_blend_operator() {
    let vblend = Operator::VectorBlend(VectorOp::V4F64);
    assert!(vblend.is_pure());
    assert_eq!(vblend.category(), OpCategory::Vector);
    assert_eq!(vblend.result_type(&[]), ValueType::V4F64);
}

#[test]
fn test_vector_cmp_operator() {
    let vcmp = Operator::VectorCmp(VectorOp::V4F64, CmpOp::Lt);
    assert!(vcmp.is_pure());
    assert_eq!(vcmp.category(), OpCategory::Vector);
    // Comparison returns a vector mask (same type as input vector)
    assert_eq!(vcmp.result_type(&[]), ValueType::V4F64);
}
