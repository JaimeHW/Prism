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
