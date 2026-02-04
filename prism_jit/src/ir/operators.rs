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
    /// Vector/SIMD operations.
    Vector = 10,
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
// Vector/SIMD Operators
// =============================================================================

/// Descriptor for a vector operation.
///
/// This specifies the element type and number of lanes for SIMD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorOp {
    /// Element type (Int64 or Float64).
    pub element: ValueType,
    /// Number of lanes (2, 4, or 8).
    pub lanes: u8,
}

impl VectorOp {
    /// Create a new vector operation descriptor.
    #[inline]
    pub const fn new(element: ValueType, lanes: u8) -> Self {
        Self { element, lanes }
    }

    /// Vector of 2 × i64 (128-bit).
    pub const V2I64: Self = Self {
        element: ValueType::Int64,
        lanes: 2,
    };
    /// Vector of 4 × i64 (256-bit).
    pub const V4I64: Self = Self {
        element: ValueType::Int64,
        lanes: 4,
    };
    /// Vector of 8 × i64 (512-bit).
    pub const V8I64: Self = Self {
        element: ValueType::Int64,
        lanes: 8,
    };
    /// Vector of 2 × f64 (128-bit).
    pub const V2F64: Self = Self {
        element: ValueType::Float64,
        lanes: 2,
    };
    /// Vector of 4 × f64 (256-bit).
    pub const V4F64: Self = Self {
        element: ValueType::Float64,
        lanes: 4,
    };
    /// Vector of 8 × f64 (512-bit).
    pub const V8F64: Self = Self {
        element: ValueType::Float64,
        lanes: 8,
    };

    /// Get the bit width of this vector (128, 256, or 512).
    #[inline]
    pub const fn bit_width(&self) -> u16 {
        let elem_bits = match self.element {
            ValueType::Int64 | ValueType::Float64 => 64,
            _ => 64, // Default to 64-bit elements
        };
        (self.lanes as u16) * elem_bits
    }

    /// Get the corresponding ValueType for this vector.
    #[inline]
    pub const fn value_type(&self) -> ValueType {
        match (self.element, self.lanes) {
            (ValueType::Int64, 2) => ValueType::V2I64,
            (ValueType::Int64, 4) => ValueType::V4I64,
            (ValueType::Int64, 8) => ValueType::V8I64,
            (ValueType::Float64, 2) => ValueType::V2F64,
            (ValueType::Float64, 4) => ValueType::V4F64,
            (ValueType::Float64, 8) => ValueType::V8F64,
            _ => ValueType::Top,
        }
    }

    /// Check if this is a floating-point vector.
    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(self.element, ValueType::Float64)
    }

    /// Check if this is an integer vector.
    #[inline]
    pub const fn is_integer(&self) -> bool {
        matches!(self.element, ValueType::Int64)
    }
}

/// Vector arithmetic operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VectorArithKind {
    /// Vector add (element-wise): a + b
    Add = 0,
    /// Vector subtract (element-wise): a - b
    Sub = 1,
    /// Vector multiply (element-wise): a * b
    Mul = 2,
    /// Vector divide (element-wise): a / b
    Div = 3,
    /// Vector min (element-wise): min(a, b)
    Min = 4,
    /// Vector max (element-wise): max(a, b)
    Max = 5,
    /// Vector abs (element-wise): |a|
    Abs = 6,
    /// Vector negate (element-wise): -a
    Neg = 7,
    /// Vector square root (element-wise): sqrt(a)
    Sqrt = 8,
}

impl VectorArithKind {
    /// Check if this operation is commutative.
    #[inline]
    pub const fn is_commutative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Min | Self::Max)
    }

    /// Check if this is a unary operation.
    #[inline]
    pub const fn is_unary(self) -> bool {
        matches!(self, Self::Abs | Self::Neg | Self::Sqrt)
    }
}

/// Vector memory operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VectorMemoryKind {
    /// Aligned load from memory.
    LoadAligned = 0,
    /// Unaligned load from memory.
    LoadUnaligned = 1,
    /// Aligned store to memory.
    StoreAligned = 2,
    /// Unaligned store to memory.
    StoreUnaligned = 3,
    /// Gather (indexed load from multiple addresses).
    Gather = 4,
    /// Scatter (indexed store to multiple addresses).
    Scatter = 5,
}

/// Vector shuffle descriptor.
///
/// Encodes a permutation of lanes. Each element is the source lane index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorShuffle {
    /// Permutation indices (up to 8 lanes).
    /// Value 0xFF means undefined/zero.
    pub indices: [u8; 8],
    /// Number of lanes in use.
    pub lanes: u8,
}

impl VectorShuffle {
    /// Identity shuffle (no permutation).
    pub const fn identity(lanes: u8) -> Self {
        let mut indices = [0xFF; 8];
        let mut i = 0;
        while i < lanes as usize && i < 8 {
            indices[i] = i as u8;
            i += 1;
        }
        Self { indices, lanes }
    }

    /// Broadcast lane 0 to all lanes.
    pub const fn broadcast(lanes: u8) -> Self {
        let mut indices = [0; 8];
        let mut i = 0;
        while i < 8 {
            indices[i] = 0;
            i += 1;
        }
        Self { indices, lanes }
    }

    /// Reverse lanes.
    pub const fn reverse(lanes: u8) -> Self {
        let mut indices = [0xFF; 8];
        let mut i = 0;
        while i < lanes as usize && i < 8 {
            indices[i] = lanes - 1 - i as u8;
            i += 1;
        }
        Self { indices, lanes }
    }
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

    // =========================================================================
    // Vector/SIMD Operations
    // =========================================================================
    /// Vector arithmetic (add, sub, mul, div, etc.).
    VectorArith(VectorOp, VectorArithKind),

    /// Fused multiply-add: a * b + c (element-wise).
    VectorFma(VectorOp),

    /// Vector memory operation (load/store).
    VectorMemory(VectorOp, VectorMemoryKind),

    /// Broadcast scalar to all vector lanes.
    VectorBroadcast(VectorOp),

    /// Extract scalar from vector lane.
    VectorExtract(VectorOp, u8),

    /// Insert scalar into vector lane.
    VectorInsert(VectorOp, u8),

    /// Shuffle/permute vector lanes.
    VectorShuffle(VectorOp, VectorShuffle),

    /// Horizontal add (sum all lanes to scalar).
    VectorHadd(VectorOp),

    /// Vector comparison (element-wise, returns vector mask).
    VectorCmp(VectorOp, CmpOp),

    /// Vector blend (select lanes from two vectors based on mask).
    VectorBlend(VectorOp),

    /// Vector splat (create vector with all lanes = constant).
    VectorSplat(VectorOp, i64),
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

            // Vector operations
            Operator::VectorArith(..)
            | Operator::VectorFma(_)
            | Operator::VectorMemory(..)
            | Operator::VectorBroadcast(_)
            | Operator::VectorExtract(..)
            | Operator::VectorInsert(..)
            | Operator::VectorShuffle(..)
            | Operator::VectorHadd(_)
            | Operator::VectorCmp(..)
            | Operator::VectorBlend(_)
            | Operator::VectorSplat(..) => OpCategory::Vector,

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

            // Pure vector operations (no memory access)
            Operator::VectorArith(..)
            | Operator::VectorFma(_)
            | Operator::VectorBroadcast(_)
            | Operator::VectorExtract(..)
            | Operator::VectorInsert(..)
            | Operator::VectorShuffle(..)
            | Operator::VectorHadd(_)
            | Operator::VectorCmp(..)
            | Operator::VectorBlend(_)
            | Operator::VectorSplat(..) => true,

            // Vector memory operations have side effects
            Operator::VectorMemory(..) => false,

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
            // Vector arithmetic commutativity
            Operator::VectorArith(_, kind) => kind.is_commutative(),
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

            // Vector operations return vector types
            Operator::VectorArith(vop, _)
            | Operator::VectorFma(vop)
            | Operator::VectorBroadcast(vop)
            | Operator::VectorInsert(vop, _)
            | Operator::VectorShuffle(vop, _)
            | Operator::VectorBlend(vop)
            | Operator::VectorSplat(vop, _)
            | Operator::VectorCmp(vop, _) => vop.value_type(),

            // Extract returns scalar element type
            Operator::VectorExtract(vop, _) => vop.element,

            // Horizontal add returns scalar
            Operator::VectorHadd(vop) => vop.element,

            // Memory operations: loads return vector type
            Operator::VectorMemory(vop, kind) => {
                match kind {
                    VectorMemoryKind::LoadAligned
                    | VectorMemoryKind::LoadUnaligned
                    | VectorMemoryKind::Gather => vop.value_type(),
                    // Stores return nothing meaningful
                    _ => ValueType::Effect,
                }
            }

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
}
