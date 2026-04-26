use super::*;

fn eval() -> ConstEvaluator {
    ConstEvaluator::new()
}

// =========================================================================
// Integer Arithmetic Tests
// =========================================================================

#[test]
fn test_int_add() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Add),
        &LatticeValue::int(10),
        &LatticeValue::int(32),
    );
    assert_eq!(result, LatticeValue::int(42));
}

#[test]
fn test_int_sub() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Sub),
        &LatticeValue::int(50),
        &LatticeValue::int(8),
    );
    assert_eq!(result, LatticeValue::int(42));
}

#[test]
fn test_int_mul() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Mul),
        &LatticeValue::int(6),
        &LatticeValue::int(7),
    );
    assert_eq!(result, LatticeValue::int(42));
}

#[test]
fn test_int_div() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::FloorDiv),
        &LatticeValue::int(84),
        &LatticeValue::int(2),
    );
    assert_eq!(result, LatticeValue::int(42));
}

#[test]
fn test_int_div_by_zero() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::FloorDiv),
        &LatticeValue::int(42),
        &LatticeValue::int(0),
    );
    assert!(result.is_overdefined());
}

#[test]
fn test_int_mod() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Mod),
        &LatticeValue::int(10),
        &LatticeValue::int(3),
    );
    assert_eq!(result, LatticeValue::int(1));
}

#[test]
fn test_int_overflow_add() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Add),
        &LatticeValue::int(i64::MAX),
        &LatticeValue::int(1),
    );
    assert!(result.is_overdefined());
}

#[test]
fn test_int_overflow_mul() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Mul),
        &LatticeValue::int(i64::MAX),
        &LatticeValue::int(2),
    );
    assert!(result.is_overdefined());
}

#[test]
fn test_int_pow() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Pow),
        &LatticeValue::int(2),
        &LatticeValue::int(10),
    );
    assert_eq!(result, LatticeValue::int(1024));
}

#[test]
fn test_int_pow_zero() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Pow),
        &LatticeValue::int(5),
        &LatticeValue::int(0),
    );
    assert_eq!(result, LatticeValue::int(1));
}

// =========================================================================
// Float Arithmetic Tests
// =========================================================================

#[test]
fn test_float_add() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::FloatOp(ArithOp::Add),
        &LatticeValue::float(1.5),
        &LatticeValue::float(2.5),
    );
    assert_eq!(result, LatticeValue::float(4.0));
}

#[test]
fn test_float_sub() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::FloatOp(ArithOp::Sub),
        &LatticeValue::float(5.0),
        &LatticeValue::float(3.0),
    );
    assert_eq!(result, LatticeValue::float(2.0));
}

#[test]
fn test_float_mul() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::FloatOp(ArithOp::Mul),
        &LatticeValue::float(3.0),
        &LatticeValue::float(4.0),
    );
    assert_eq!(result, LatticeValue::float(12.0));
}

#[test]
fn test_float_div() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::FloatOp(ArithOp::TrueDiv),
        &LatticeValue::float(10.0),
        &LatticeValue::float(4.0),
    );
    assert_eq!(result, LatticeValue::float(2.5));
}

#[test]
fn test_float_div_by_zero() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::FloatOp(ArithOp::TrueDiv),
        &LatticeValue::float(1.0),
        &LatticeValue::float(0.0),
    );
    assert!(result.is_overdefined());
}

#[test]
fn test_float_pow() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::FloatOp(ArithOp::Pow),
        &LatticeValue::float(2.0),
        &LatticeValue::float(3.0),
    );
    assert_eq!(result, LatticeValue::float(8.0));
}

// =========================================================================
// Comparison Tests
// =========================================================================

#[test]
fn test_int_cmp_eq() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntCmp(CmpOp::Eq),
        &LatticeValue::int(5),
        &LatticeValue::int(5),
    );
    assert_eq!(result, LatticeValue::bool(true));

    let result = e.eval_binary(
        &Operator::IntCmp(CmpOp::Eq),
        &LatticeValue::int(5),
        &LatticeValue::int(6),
    );
    assert_eq!(result, LatticeValue::bool(false));
}

#[test]
fn test_int_cmp_lt() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntCmp(CmpOp::Lt),
        &LatticeValue::int(3),
        &LatticeValue::int(5),
    );
    assert_eq!(result, LatticeValue::bool(true));

    let result = e.eval_binary(
        &Operator::IntCmp(CmpOp::Lt),
        &LatticeValue::int(5),
        &LatticeValue::int(3),
    );
    assert_eq!(result, LatticeValue::bool(false));
}

#[test]
fn test_int_cmp_le() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntCmp(CmpOp::Le),
        &LatticeValue::int(5),
        &LatticeValue::int(5),
    );
    assert_eq!(result, LatticeValue::bool(true));
}

#[test]
fn test_int_cmp_gt() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntCmp(CmpOp::Gt),
        &LatticeValue::int(10),
        &LatticeValue::int(5),
    );
    assert_eq!(result, LatticeValue::bool(true));
}

#[test]
fn test_int_cmp_ge() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntCmp(CmpOp::Ge),
        &LatticeValue::int(5),
        &LatticeValue::int(5),
    );
    assert_eq!(result, LatticeValue::bool(true));
}

#[test]
fn test_int_cmp_ne() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntCmp(CmpOp::Ne),
        &LatticeValue::int(1),
        &LatticeValue::int(2),
    );
    assert_eq!(result, LatticeValue::bool(true));
}

#[test]
fn test_int_cmp_is_overdefined() {
    let e = eval();
    // 'is' comparison requires runtime identity check
    let result = e.eval_binary(
        &Operator::IntCmp(CmpOp::Is),
        &LatticeValue::int(5),
        &LatticeValue::int(5),
    );
    assert!(result.is_overdefined());
}

// =========================================================================
// Bitwise Operation Tests
// =========================================================================

#[test]
fn test_bitwise_and() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::Bitwise(BitwiseOp::And),
        &LatticeValue::int(0b1100),
        &LatticeValue::int(0b1010),
    );
    assert_eq!(result, LatticeValue::int(0b1000));
}

#[test]
fn test_bitwise_or() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::Bitwise(BitwiseOp::Or),
        &LatticeValue::int(0b1100),
        &LatticeValue::int(0b1010),
    );
    assert_eq!(result, LatticeValue::int(0b1110));
}

#[test]
fn test_bitwise_xor() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::Bitwise(BitwiseOp::Xor),
        &LatticeValue::int(0b1100),
        &LatticeValue::int(0b1010),
    );
    assert_eq!(result, LatticeValue::int(0b0110));
}

#[test]
fn test_shift_left() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::Bitwise(BitwiseOp::Shl),
        &LatticeValue::int(1),
        &LatticeValue::int(4),
    );
    assert_eq!(result, LatticeValue::int(16));
}

#[test]
fn test_shift_right() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::Bitwise(BitwiseOp::Shr),
        &LatticeValue::int(16),
        &LatticeValue::int(2),
    );
    assert_eq!(result, LatticeValue::int(4));
}

#[test]
fn test_shift_invalid() {
    let e = eval();
    // Shift by negative amount
    let result = e.eval_binary(
        &Operator::Bitwise(BitwiseOp::Shl),
        &LatticeValue::int(1),
        &LatticeValue::int(-1),
    );
    assert!(result.is_overdefined());

    // Shift by too large amount
    let result = e.eval_binary(
        &Operator::Bitwise(BitwiseOp::Shl),
        &LatticeValue::int(1),
        &LatticeValue::int(64),
    );
    assert!(result.is_overdefined());
}

// =========================================================================
// Unary Operation Tests
// =========================================================================

#[test]
fn test_unary_neg() {
    let e = eval();
    let result = e.eval_unary(&Operator::IntOp(ArithOp::Neg), &LatticeValue::int(5));
    assert_eq!(result, LatticeValue::int(-5));
}

#[test]
fn test_unary_not() {
    let e = eval();
    let result = e.eval_unary(&Operator::LogicalNot, &LatticeValue::bool(true));
    assert_eq!(result, LatticeValue::bool(false));

    let result = e.eval_unary(&Operator::LogicalNot, &LatticeValue::int(0));
    assert_eq!(result, LatticeValue::bool(true));
}

#[test]
fn test_unary_bitnot() {
    let e = eval();
    let result = e.eval_unary(&Operator::Bitwise(BitwiseOp::Not), &LatticeValue::int(0));
    assert_eq!(result, LatticeValue::int(-1));
}

// =========================================================================
// Phi Evaluation Tests
// =========================================================================

#[test]
fn test_phi_all_same() {
    let e = eval();
    let result = e.eval_phi(
        [
            LatticeValue::int(42),
            LatticeValue::int(42),
            LatticeValue::int(42),
        ]
        .into_iter(),
    );
    assert_eq!(result, LatticeValue::int(42));
}

#[test]
fn test_phi_different() {
    let e = eval();
    let result = e.eval_phi([LatticeValue::int(1), LatticeValue::int(2)].into_iter());
    assert!(result.is_overdefined());
}

#[test]
fn test_phi_with_undef() {
    let e = eval();
    let result = e.eval_phi([LatticeValue::undef(), LatticeValue::int(42)].into_iter());
    assert_eq!(result, LatticeValue::int(42));
}

#[test]
fn test_phi_empty() {
    let e = eval();
    let result = e.eval_phi(std::iter::empty());
    assert!(result.is_undef());
}

// =========================================================================
// Mixed Type Tests
// =========================================================================

#[test]
fn test_mixed_types_overdefined() {
    let e = eval();
    // Int + Float should be overdefined (different types in int op context)
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Add),
        &LatticeValue::int(1),
        &LatticeValue::float(2.0),
    );
    assert!(result.is_overdefined());
}

#[test]
fn test_overdefined_input() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Add),
        &LatticeValue::int(1),
        &LatticeValue::overdefined(),
    );
    assert!(result.is_overdefined());
}

#[test]
fn test_undef_input() {
    let e = eval();
    let result = e.eval_binary(
        &Operator::IntOp(ArithOp::Add),
        &LatticeValue::int(1),
        &LatticeValue::undef(),
    );
    assert!(result.is_overdefined());
}
