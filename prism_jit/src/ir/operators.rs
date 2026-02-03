//! IR operator definitions for the Sea-of-Nodes IR.
//!
//! Operators are organized by category:
//! - **Constants**: Fixed values (integers, floats, etc.)
//! - **Arithmetic**: Binary and unary math operations
//! - **Comparison**: Relational and equality tests
//! - **Memory**: Load, store, allocation
//! - **Control**: Branch, merge, loop, return
//! - **Guards**: Runtime type/bounds checks
//! - **Calls**: Function invocations
//!
//! Each operator carries semantic information for optimization:
//! - Commutativity (for GVN canonicalization)
//! - Associativity (for strength reduction)
//! - Pure vs. effectful (for DCE and scheduling)

use super::types::ValueType;

// =============================================================================
// Operator Categories
// =============================================================================

/// Operator category for fast dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OpCategory {
    /// Constant values.
    Constant = 0,
    /// Arithmetic operations.
    Arithmetic = 1,
    /// Comparison operations.
    Comparison = 2,
    /// Bitwise operations.
    Bitwise = 3,
    /// Memory operations.
    Memory = 4,
    /// Control flow.
    Control = 5,
    /// Guards (type checks, bounds checks).
    Guard = 6,
    /// Function calls.
    Call = 7,
    /// Projection (extracting tuple elements).
    Projection = 8,
    /// Phi nodes (SSA merge).
    Phi = 9,
}

// =============================================================================
// Arithmetic Operators
// =============================================================================

/// Arithmetic operator kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArithOp {
    // Binary operations
    /// Addition: a + b
    Add = 0,
    /// Subtraction: a - b
    Sub = 1,
    /// Multiplication: a * b
    Mul = 2,
    /// True division: a / b
    TrueDiv = 3,
    /// Floor division: a // b
    FloorDiv = 4,
    /// Modulo: a % b
    Mod = 5,
    /// Power: a ** b
    Pow = 6,
    /// Matrix multiply: a @ b
    MatMul = 7,

    // Unary operations
    /// Negation: -a
    Neg = 16,
    /// Positive: +a (usually no-op, but may box)
    Pos = 17,
    /// Absolute value: abs(a)
    Abs = 18,
}

impl ArithOp {
    /// Check if this operation is commutative.
    #[inline]
    pub const fn is_commutative(self) -> bool {
        matches!(self, ArithOp::Add | ArithOp::Mul)
    }

    /// Check if this operation is associative.
    #[inline]
    pub const fn is_associative(self) -> bool {
        matches!(self, ArithOp::Add | ArithOp::Mul)
    }

    /// Check if this is a binary operation.
    #[inline]
    pub const fn is_binary(self) -> bool {
        (self as u8) < 16
    }

    /// Check if this is a unary operation.
    #[inline]
    pub const fn is_unary(self) -> bool {
        (self as u8) >= 16
    }

    /// Get the identity element if this operation has one.
    pub const fn identity(self) -> Option<i64> {
        match self {
            ArithOp::Add | ArithOp::Sub => Some(0),
            ArithOp::Mul | ArithOp::TrueDiv | ArithOp::FloorDiv => Some(1),
            ArithOp::Pow => Some(1), // x ** 1 = x
            _ => None,
        }
    }

    /// Get the absorbing element if this operation has one.
    pub const fn absorbing(self) -> Option<i64> {
        match self {
            ArithOp::Mul => Some(0), // x * 0 = 0
            _ => None,
        }
    }

    /// Infer result type from operand types.
    pub const fn result_type(self, lhs: ValueType, rhs: ValueType) -> ValueType {
        use ValueType::*;

        // Unary ops
        if self.is_unary() {
            return lhs;
        }

        // Same type -> same type (for int/float)
        if lhs as u8 == rhs as u8 {
            if matches!(lhs, Int64 | Float64) {
                // Division always produces float
                if matches!(self, ArithOp::TrueDiv) {
                    return Float64;
                }
                return lhs;
            }
        }

        // Int op Float -> Float
        if (matches!(lhs, Int64) && matches!(rhs, Float64))
            || (matches!(lhs, Float64) && matches!(rhs, Int64))
        {
            return Float64;
        }

        // Division produces float
        if matches!(self, ArithOp::TrueDiv) {
            return Float64;
        }

        // Power may produce float
        if matches!(self, ArithOp::Pow) {
            return Numeric;
        }

        // Unknown
        Numeric
    }
}

// =============================================================================
// Comparison Operators
// =============================================================================

/// Comparison operator kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CmpOp {
    /// Less than: a < b
    Lt = 0,
    /// Less than or equal: a <= b
    Le = 1,
    /// Equal: a == b
    Eq = 2,
    /// Not equal: a != b
    Ne = 3,
    /// Greater than: a > b
    Gt = 4,
    /// Greater than or equal: a >= b
    Ge = 5,
    /// Identity: a is b
    Is = 6,
    /// Not identity: a is not b
    IsNot = 7,
    /// Membership: a in b
    In = 8,
    /// Not membership: a not in b
    NotIn = 9,
}

impl CmpOp {
    /// Get the inverse of this comparison.
    #[inline]
    pub const fn inverse(self) -> Self {
        match self {
            CmpOp::Lt => CmpOp::Ge,
            CmpOp::Le => CmpOp::Gt,
            CmpOp::Eq => CmpOp::Ne,
            CmpOp::Ne => CmpOp::Eq,
            CmpOp::Gt => CmpOp::Le,
            CmpOp::Ge => CmpOp::Lt,
            CmpOp::Is => CmpOp::IsNot,
            CmpOp::IsNot => CmpOp::Is,
            CmpOp::In => CmpOp::NotIn,
            CmpOp::NotIn => CmpOp::In,
        }
    }

    /// Get the swapped version (swap operands).
    #[inline]
    pub const fn swap(self) -> Self {
        match self {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::Le => CmpOp::Ge,
            CmpOp::Eq => CmpOp::Eq,
            CmpOp::Ne => CmpOp::Ne,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::Ge => CmpOp::Le,
            CmpOp::Is => CmpOp::Is,
            CmpOp::IsNot => CmpOp::IsNot,
            _ => self, // In/NotIn don't swap
        }
    }

    /// Check if this comparison is commutative.
    #[inline]
    pub const fn is_commutative(self) -> bool {
        matches!(self, CmpOp::Eq | CmpOp::Ne | CmpOp::Is | CmpOp::IsNot)
    }
}

// =============================================================================
// Bitwise Operators
// =============================================================================

/// Bitwise operator kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BitwiseOp {
    /// Bitwise AND: a & b
    And = 0,
    /// Bitwise OR: a | b
    Or = 1,
    /// Bitwise XOR: a ^ b
    Xor = 2,
    /// Left shift: a << b
    Shl = 3,
    /// Right shift: a >> b
    Shr = 4,
    /// Bitwise NOT: ~a
    Not = 16,
}

impl BitwiseOp {
    /// Check if this operation is commutative.
    #[inline]
    pub const fn is_commutative(self) -> bool {
        matches!(self, BitwiseOp::And | BitwiseOp::Or | BitwiseOp::Xor)
    }

    /// Check if this is a binary operation.
    #[inline]
    pub const fn is_binary(self) -> bool {
        (self as u8) < 16
    }
}

// =============================================================================
// Guard Operators
// =============================================================================

/// Guard kind for runtime checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GuardKind {
    /// Type guard: ensure value has expected type.
    Type = 0,
    /// Bounds guard: ensure index is within range.
    Bounds = 1,
    /// Null guard: ensure value is not None.
    NotNull = 2,
    /// Overflow guard: operation didn't overflow.
    Overflow = 3,
    /// Zero divisor guard: divisor is not zero.
    NonZeroDivisor = 4,
}

// =============================================================================
// Call Operators
// =============================================================================

/// Call kind for different invocation patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CallKind {
    /// Direct function call: f(args)
    Direct = 0,
    /// Method call: obj.method(args)
    Method = 1,
    /// Call with keyword arguments.
    Keyword = 2,
    /// Tail call (reuses frame).
    Tail = 3,
    /// Runtime helper call (internal).
    Runtime = 4,
}

// =============================================================================
// Memory Operators
// =============================================================================

/// Memory operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryOp {
    /// Load from object field.
    LoadField = 0,
    /// Store to object field.
    StoreField = 1,
    /// Load from array/list element.
    LoadElement = 2,
    /// Store to array/list element.
    StoreElement = 3,
    /// Allocate new object.
    Alloc = 4,
    /// Allocate array/list.
    AllocArray = 5,
}

// =============================================================================
// Control Flow Operators
// =============================================================================

/// Control flow operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ControlOp {
    /// Start node (entry point).
    Start = 0,
    /// End node (exit point).
    End = 1,
    /// Region (control merge).
    Region = 2,
    /// Loop header.
    Loop = 3,
    /// If branch.
    If = 4,
    /// Return from function.
    Return = 5,
    /// Throw exception.
    Throw = 6,
    /// Deoptimize to interpreter.
    Deopt = 7,
}

// =============================================================================
// Operator (Unified)
// =============================================================================

/// Unified operator representation.
///
/// This enum covers all possible operations in the IR.
/// Each variant carries the specific operator details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operator {
    // Constants
    /// Integer constant.
    ConstInt(i64),
    /// Float constant.
    ConstFloat(u64), // Stored as bits for Hash/Eq
    /// Boolean constant.
    ConstBool(bool),
    /// None constant.
    ConstNone,
    /// Parameter (function argument).
    Parameter(u16),

    // Arithmetic
    /// Typed integer arithmetic.
    IntOp(ArithOp),
    /// Typed float arithmetic.
    FloatOp(ArithOp),
    /// Generic (polymorphic) arithmetic.
    GenericOp(ArithOp),
    /// Unsigned multiply high (upper 64 bits of 128-bit product).
    MulHigh,
    /// Signed multiply high (upper 64 bits of 128-bit product).
    MulHighSigned,

    // Comparison
    /// Typed integer comparison.
    IntCmp(CmpOp),
    /// Typed float comparison.
    FloatCmp(CmpOp),
    /// Generic comparison.
    GenericCmp(CmpOp),

    // Bitwise
    /// Bitwise operation (integers only).
    Bitwise(BitwiseOp),

    // Logical
    /// Logical NOT: not x
    LogicalNot,

    // Memory
    /// Memory operation.
    Memory(MemoryOp),

    // Control flow
    /// Control operation.
    Control(ControlOp),

    // Guards
    /// Runtime guard.
    Guard(GuardKind),

    // Calls
    /// Function/method call.
    Call(CallKind),

    // SSA
    /// Phi node for value merging.
    Phi,
    /// LoopPhi for loop-carried values.
    LoopPhi,
    /// Projection (extract from tuple result).
    Projection(u8),

    // Container operations
    /// Build list from elements.
    BuildList(u16),
    /// Build tuple from elements.
    BuildTuple(u16),
    /// Build dict from key-value pairs.
    BuildDict(u16),
    /// Get iterator.
    GetIter,
    /// Get next from iterator.
    IterNext,
    /// Get item: obj[key]
    GetItem,
    /// Set item: obj[key] = value
    SetItem,
    /// Get attribute: obj.attr
    GetAttr,
    /// Set attribute: obj.attr = value
    SetAttr,
    /// Get length: len(obj)
    Len,

    // Type operations
    /// Type check: isinstance(obj, type)
    TypeCheck,
    /// Box primitive to object.
    Box,
    /// Unbox object to primitive.
    Unbox,
}

impl Operator {
    /// Get the category of this operator.
    pub const fn category(&self) -> OpCategory {
        match self {
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone
            | Operator::Parameter(_) => OpCategory::Constant,

            Operator::IntOp(_) | Operator::FloatOp(_) | Operator::GenericOp(_) => {
                OpCategory::Arithmetic
            }

            Operator::MulHigh | Operator::MulHighSigned => OpCategory::Arithmetic,

            Operator::IntCmp(_) | Operator::FloatCmp(_) | Operator::GenericCmp(_) => {
                OpCategory::Comparison
            }

            Operator::Bitwise(_) => OpCategory::Bitwise,

            Operator::Memory(_) => OpCategory::Memory,

            Operator::Control(_) => OpCategory::Control,

            Operator::Guard(_) => OpCategory::Guard,

            Operator::Call(_) => OpCategory::Call,

            Operator::Projection(_) => OpCategory::Projection,

            Operator::Phi | Operator::LoopPhi => OpCategory::Phi,

            _ => OpCategory::Memory, // Container/type ops are memory-like
        }
    }

    /// Check if this operator is pure (no side effects).
    pub const fn is_pure(&self) -> bool {
        match self {
            // Constants are pure
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone
            | Operator::Parameter(_) => true,

            // Arithmetic is pure
            Operator::IntOp(_)
            | Operator::FloatOp(_)
            | Operator::IntCmp(_)
            | Operator::FloatCmp(_)
            | Operator::Bitwise(_)
            | Operator::LogicalNot
            | Operator::MulHigh
            | Operator::MulHighSigned => true,

            // SSA nodes are pure
            Operator::Phi | Operator::LoopPhi | Operator::Projection(_) => true,

            // Type ops are pure
            Operator::TypeCheck | Operator::Box | Operator::Unbox | Operator::Len => true,

            // Memory, control, calls, guards have effects
            _ => false,
        }
    }

    /// Check if this operator is commutative.
    pub const fn is_commutative(&self) -> bool {
        match self {
            Operator::IntOp(op) | Operator::FloatOp(op) | Operator::GenericOp(op) => {
                op.is_commutative()
            }
            Operator::IntCmp(op) | Operator::FloatCmp(op) | Operator::GenericCmp(op) => {
                op.is_commutative()
            }
            Operator::Bitwise(op) => op.is_commutative(),
            _ => false,
        }
    }

    /// Get the result type for this operator given input types.
    pub fn result_type(&self, inputs: &[ValueType]) -> ValueType {
        match self {
            Operator::ConstInt(_) => ValueType::Int64,
            Operator::ConstFloat(_) => ValueType::Float64,
            Operator::ConstBool(_) => ValueType::Bool,
            Operator::ConstNone => ValueType::None,
            Operator::Parameter(_) => ValueType::Top,

            Operator::IntOp(op) => {
                if inputs.len() >= 2 {
                    op.result_type(inputs[0], inputs[1])
                } else if !inputs.is_empty() {
                    inputs[0]
                } else {
                    ValueType::Int64
                }
            }

            Operator::FloatOp(_) => ValueType::Float64,
            Operator::GenericOp(_) => ValueType::Numeric,

            // MulHigh operations return the upper 64 bits of 128-bit product
            Operator::MulHigh | Operator::MulHighSigned => ValueType::Int64,

            Operator::IntCmp(_) | Operator::FloatCmp(_) | Operator::GenericCmp(_) => {
                ValueType::Bool
            }

            Operator::Bitwise(_) => ValueType::Int64,
            Operator::LogicalNot => ValueType::Bool,

            Operator::Control(_) => ValueType::Control,
            Operator::Phi | Operator::LoopPhi => {
                // Meet all input types
                inputs.iter().fold(ValueType::Top, |acc, &t| acc.meet(t))
            }

            Operator::BuildList(_) => ValueType::List,
            Operator::BuildTuple(_) => ValueType::Tuple,
            Operator::BuildDict(_) => ValueType::Dict,
            Operator::GetIter => ValueType::Object,
            Operator::IterNext => ValueType::Top,
            Operator::GetItem => ValueType::Top,
            Operator::GetAttr => ValueType::Top,
            Operator::Len => ValueType::Int64,
            Operator::TypeCheck => ValueType::Bool,

            _ => ValueType::Top,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
}
