//! Constant Expression Evaluation for SCCP.
//!
//! This module provides compile-time evaluation of operations when all
//! operands are known constants. This is the "constant folding" component
//! of SCCP.
//!
//! # Supported Operations
//!
//! - Arithmetic: +, -, *, /, //, %, **
//! - Bitwise: &, |, ^, ~, <<, >>
//! - Comparison: ==, !=, <, <=, >, >=
//! - Logical: not
//! - Unary: -, +

use super::lattice::{Constant, LatticeValue};
use crate::ir::operators::{ArithOp, BitwiseOp, CmpOp, Operator};

// =============================================================================
// Evaluator
// =============================================================================

/// Constant expression evaluator.
///
/// Evaluates operations at compile time when all inputs are constants.
#[derive(Debug, Default)]
pub struct ConstEvaluator;

impl ConstEvaluator {
    /// Create a new evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate a unary operation.
    pub fn eval_unary(&self, op: &Operator, operand: &LatticeValue) -> LatticeValue {
        let constant = match operand.as_constant() {
            Some(c) => c,
            None => return operand.clone(),
        };

        match op {
            Operator::IntOp(ArithOp::Neg) | Operator::FloatOp(ArithOp::Neg) => constant
                .negate()
                .map_or(LatticeValue::overdefined(), LatticeValue::constant),
            Operator::LogicalNot => LatticeValue::constant(constant.logical_not()),
            Operator::Bitwise(BitwiseOp::Not) => constant
                .bitwise_not()
                .map_or(LatticeValue::overdefined(), LatticeValue::constant),
            _ => LatticeValue::overdefined(),
        }
    }

    /// Evaluate a binary operation on two constants.
    pub fn eval_binary(
        &self,
        op: &Operator,
        lhs: &LatticeValue,
        rhs: &LatticeValue,
    ) -> LatticeValue {
        let (lhs_const, rhs_const) = match (lhs.as_constant(), rhs.as_constant()) {
            (Some(l), Some(r)) => (l, r),
            _ => return LatticeValue::overdefined(),
        };

        match op {
            // Integer arithmetic
            Operator::IntOp(arith_op) => self.eval_int_arith(*arith_op, lhs_const, rhs_const),

            // Float arithmetic
            Operator::FloatOp(arith_op) => self.eval_float_arith(*arith_op, lhs_const, rhs_const),

            // Comparison
            Operator::IntCmp(cmp_op) => self.eval_int_cmp(*cmp_op, lhs_const, rhs_const),
            Operator::FloatCmp(cmp_op) => self.eval_float_cmp(*cmp_op, lhs_const, rhs_const),

            // Bitwise operations
            Operator::Bitwise(BitwiseOp::And) => self.eval_bitwise_and(lhs_const, rhs_const),
            Operator::Bitwise(BitwiseOp::Or) => self.eval_bitwise_or(lhs_const, rhs_const),
            Operator::Bitwise(BitwiseOp::Xor) => self.eval_bitwise_xor(lhs_const, rhs_const),
            Operator::Bitwise(BitwiseOp::Shl) => self.eval_shift_left(lhs_const, rhs_const),
            Operator::Bitwise(BitwiseOp::Shr) => self.eval_shift_right(lhs_const, rhs_const),

            _ => LatticeValue::overdefined(),
        }
    }

    /// Evaluate integer arithmetic.
    fn eval_int_arith(&self, op: ArithOp, lhs: &Constant, rhs: &Constant) -> LatticeValue {
        let (a, b) = match (lhs.as_int(), rhs.as_int()) {
            (Some(a), Some(b)) => (a, b),
            _ => return LatticeValue::overdefined(),
        };

        match op {
            ArithOp::Add => a
                .checked_add(b)
                .map_or(LatticeValue::overdefined(), LatticeValue::int),
            ArithOp::Sub => a
                .checked_sub(b)
                .map_or(LatticeValue::overdefined(), LatticeValue::int),
            ArithOp::Mul => a
                .checked_mul(b)
                .map_or(LatticeValue::overdefined(), LatticeValue::int),
            ArithOp::TrueDiv | ArithOp::FloorDiv => {
                if b == 0 {
                    LatticeValue::overdefined()
                } else {
                    a.checked_div(b)
                        .map_or(LatticeValue::overdefined(), LatticeValue::int)
                }
            }
            ArithOp::Mod => {
                if b == 0 {
                    LatticeValue::overdefined()
                } else {
                    a.checked_rem(b)
                        .map_or(LatticeValue::overdefined(), LatticeValue::int)
                }
            }
            ArithOp::Pow => {
                // Only handle non-negative exponents
                if b < 0 {
                    LatticeValue::overdefined()
                } else if b == 0 {
                    LatticeValue::int(1)
                } else if b <= 63 {
                    // Safe exponentiation
                    a.checked_pow(b as u32)
                        .map_or(LatticeValue::overdefined(), LatticeValue::int)
                } else {
                    LatticeValue::overdefined()
                }
            }
            ArithOp::Neg => a
                .checked_neg()
                .map_or(LatticeValue::overdefined(), LatticeValue::int),
            // Pos is essentially identity for integers
            ArithOp::Pos => LatticeValue::int(a),
            ArithOp::Abs => LatticeValue::int(a.abs()),
            ArithOp::MatMul => LatticeValue::overdefined(), // Not applicable to scalars
        }
    }

    /// Evaluate float arithmetic.
    fn eval_float_arith(&self, op: ArithOp, lhs: &Constant, rhs: &Constant) -> LatticeValue {
        let (a, b) = match (lhs.as_float(), rhs.as_float()) {
            (Some(a), Some(b)) => (a, b),
            _ => return LatticeValue::overdefined(),
        };

        let result = match op {
            ArithOp::Add => a + b,
            ArithOp::Sub => a - b,
            ArithOp::Mul => a * b,
            ArithOp::TrueDiv => {
                if b == 0.0 {
                    return LatticeValue::overdefined();
                }
                a / b
            }
            ArithOp::FloorDiv => {
                if b == 0.0 {
                    return LatticeValue::overdefined();
                }
                (a / b).floor()
            }
            ArithOp::Mod => {
                if b == 0.0 {
                    return LatticeValue::overdefined();
                }
                a % b
            }
            ArithOp::Pow => a.powf(b),
            ArithOp::Neg => -a,
            ArithOp::Pos => a,
            ArithOp::Abs => a.abs(),
            ArithOp::MatMul => return LatticeValue::overdefined(),
        };

        // Check for NaN/Inf which we treat as overdefined
        if result.is_nan() || result.is_infinite() {
            LatticeValue::overdefined()
        } else {
            LatticeValue::float(result)
        }
    }

    /// Evaluate integer comparison.
    fn eval_int_cmp(&self, op: CmpOp, lhs: &Constant, rhs: &Constant) -> LatticeValue {
        let (a, b) = match (lhs.as_int(), rhs.as_int()) {
            (Some(a), Some(b)) => (a, b),
            _ => return LatticeValue::overdefined(),
        };

        let result = match op {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
            // Identity and membership ops require runtime
            CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => {
                return LatticeValue::overdefined();
            }
        };

        LatticeValue::bool(result)
    }

    /// Evaluate float comparison.
    fn eval_float_cmp(&self, op: CmpOp, lhs: &Constant, rhs: &Constant) -> LatticeValue {
        let (a, b) = match (lhs.as_float(), rhs.as_float()) {
            (Some(a), Some(b)) => (a, b),
            _ => return LatticeValue::overdefined(),
        };

        // NaN comparisons
        if a.is_nan() || b.is_nan() {
            return match op {
                CmpOp::Ne => LatticeValue::bool(true),
                _ => LatticeValue::bool(false),
            };
        }

        let result = match op {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
            // Identity and membership ops require runtime
            CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => {
                return LatticeValue::overdefined();
            }
        };

        LatticeValue::bool(result)
    }

    /// Evaluate bitwise AND.
    fn eval_bitwise_and(&self, lhs: &Constant, rhs: &Constant) -> LatticeValue {
        match (lhs.as_int(), rhs.as_int()) {
            (Some(a), Some(b)) => LatticeValue::int(a & b),
            _ => LatticeValue::overdefined(),
        }
    }

    /// Evaluate bitwise OR.
    fn eval_bitwise_or(&self, lhs: &Constant, rhs: &Constant) -> LatticeValue {
        match (lhs.as_int(), rhs.as_int()) {
            (Some(a), Some(b)) => LatticeValue::int(a | b),
            _ => LatticeValue::overdefined(),
        }
    }

    /// Evaluate bitwise XOR.
    fn eval_bitwise_xor(&self, lhs: &Constant, rhs: &Constant) -> LatticeValue {
        match (lhs.as_int(), rhs.as_int()) {
            (Some(a), Some(b)) => LatticeValue::int(a ^ b),
            _ => LatticeValue::overdefined(),
        }
    }

    /// Evaluate left shift.
    fn eval_shift_left(&self, lhs: &Constant, rhs: &Constant) -> LatticeValue {
        match (lhs.as_int(), rhs.as_int()) {
            (Some(a), Some(b)) => {
                if b < 0 || b >= 64 {
                    LatticeValue::overdefined()
                } else {
                    LatticeValue::int(a << b)
                }
            }
            _ => LatticeValue::overdefined(),
        }
    }

    /// Evaluate right shift (arithmetic).
    fn eval_shift_right(&self, lhs: &Constant, rhs: &Constant) -> LatticeValue {
        match (lhs.as_int(), rhs.as_int()) {
            (Some(a), Some(b)) => {
                if b < 0 || b >= 64 {
                    LatticeValue::overdefined()
                } else {
                    LatticeValue::int(a >> b)
                }
            }
            _ => LatticeValue::overdefined(),
        }
    }

    /// Evaluate a phi node by meeting all incoming values.
    pub fn eval_phi(&self, values: impl Iterator<Item = LatticeValue>) -> LatticeValue {
        let mut result = LatticeValue::undef();
        for v in values {
            result = result.meet(&v);
            // Short-circuit if we hit top
            if result.is_overdefined() {
                break;
            }
        }
        result
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
}
