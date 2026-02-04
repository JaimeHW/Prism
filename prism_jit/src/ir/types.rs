//! Value type system for the Sea-of-Nodes IR.
//!
//! The type system is designed for:
//! - **Fast type narrowing**: Lattice-based meet operation for type inference
//! - **Specialization**: Distinguishing int/float/object for code generation
//! - **Guard elimination**: Proving type guards redundant
//!
//! The type lattice is:
//! ```text
//!                    Top (unknown)
//!                   /    |    \
//!              Int64   Float64  Object
//!               |        |        /  \
//!              ...      ...    Tuple  List ...
//!                   \    |    /
//!                    Bottom (unreachable)
//! ```

use std::fmt;

// =============================================================================
// Core Value Types
// =============================================================================

/// Value type in the IR type system.
///
/// This is a compact representation using tagged union semantics.
/// The type forms a lattice for type inference and narrowing.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ValueType {
    /// Unknown type (top of lattice) - could be anything.
    Top = 0,

    /// 64-bit signed integer (specialized path).
    Int64 = 1,

    /// 64-bit floating-point (specialized path).
    Float64 = 2,

    /// Boolean value (True/False).
    Bool = 3,

    /// None singleton.
    None = 4,

    /// Tuple object.
    Tuple = 5,

    /// List object.
    List = 6,

    /// Dict object.
    Dict = 7,

    /// Set object.
    Set = 8,

    /// String object.
    String = 9,

    /// Bytes object.
    Bytes = 10,

    /// Function object.
    Function = 11,

    /// Any object (may need runtime type checks).
    Object = 12,

    /// Numeric value (int or float - not yet specialized).
    Numeric = 13,

    /// Sequence type (tuple, list, string - for len/getitem).
    Sequence = 14,

    /// Iterable type (anything supporting __iter__).
    Iterable = 15,

    // =========================================================================
    // Vector/SIMD Types (128-bit, SSE)
    // =========================================================================
    /// Vector of 2 × i64 (128-bit, SSE2).
    V2I64 = 64,

    /// Vector of 4 × i32 (128-bit, SSE2).
    V4I32 = 65,

    /// Vector of 8 × i16 (128-bit, SSE2).
    V8I16 = 66,

    /// Vector of 16 × i8 (128-bit, SSE2).
    V16I8 = 67,

    /// Vector of 2 × f64 (128-bit, SSE2).
    V2F64 = 68,

    /// Vector of 4 × f32 (128-bit, SSE).
    V4F32 = 69,

    // =========================================================================
    // Vector/SIMD Types (256-bit, AVX/AVX2)
    // =========================================================================
    /// Vector of 4 × i64 (256-bit, AVX2).
    V4I64 = 80,

    /// Vector of 8 × i32 (256-bit, AVX2).
    V8I32 = 81,

    /// Vector of 16 × i16 (256-bit, AVX2).
    V16I16 = 82,

    /// Vector of 32 × i8 (256-bit, AVX2).
    V32I8 = 83,

    /// Vector of 4 × f64 (256-bit, AVX).
    V4F64 = 84,

    /// Vector of 8 × f32 (256-bit, AVX).
    V8F32 = 85,

    // =========================================================================
    // Vector/SIMD Types (512-bit, AVX-512)
    // =========================================================================
    /// Vector of 8 × i64 (512-bit, AVX-512).
    V8I64 = 96,

    /// Vector of 16 × i32 (512-bit, AVX-512).
    V16I32 = 97,

    /// Vector of 8 × f64 (512-bit, AVX-512).
    V8F64 = 98,

    /// Vector of 16 × f32 (512-bit, AVX-512).
    V16F32 = 99,

    // =========================================================================
    // Special/Effect Types
    // =========================================================================
    /// Control token (for scheduling, not a value).
    Control = 250,

    /// Memory effect token (for aliasing analysis).
    Memory = 251,

    /// Effect token (for side-effect ordering).
    Effect = 252,

    /// Unreachable/dead code (bottom of lattice).
    Bottom = 255,
}

impl ValueType {
    /// Check if this type is numeric (int or float).
    #[inline]
    pub const fn is_numeric(self) -> bool {
        matches!(
            self,
            ValueType::Int64 | ValueType::Float64 | ValueType::Numeric
        )
    }

    /// Check if this type is an object (heap-allocated).
    #[inline]
    pub const fn is_object(self) -> bool {
        matches!(
            self,
            ValueType::Tuple
                | ValueType::List
                | ValueType::Dict
                | ValueType::Set
                | ValueType::String
                | ValueType::Bytes
                | ValueType::Function
                | ValueType::Object
        )
    }

    /// Check if this type is a sequence (list, tuple, string).
    #[inline]
    pub const fn is_sequence(self) -> bool {
        matches!(
            self,
            ValueType::Tuple | ValueType::List | ValueType::String | ValueType::Sequence
        )
    }

    /// Check if this type is a control/effect token.
    #[inline]
    pub const fn is_side_effect(self) -> bool {
        matches!(
            self,
            ValueType::Control | ValueType::Memory | ValueType::Effect
        )
    }

    /// Check if this type is a SIMD vector type.
    #[inline]
    pub const fn is_vector(self) -> bool {
        matches!(
            self,
            ValueType::V2I64
                | ValueType::V4I32
                | ValueType::V8I16
                | ValueType::V16I8
                | ValueType::V2F64
                | ValueType::V4F32
                | ValueType::V4I64
                | ValueType::V8I32
                | ValueType::V16I16
                | ValueType::V32I8
                | ValueType::V4F64
                | ValueType::V8F32
                | ValueType::V8I64
                | ValueType::V16I32
                | ValueType::V8F64
                | ValueType::V16F32
        )
    }

    /// Get the number of lanes for a vector type (returns 1 for scalars).
    #[inline]
    pub const fn lanes(self) -> u8 {
        match self {
            // 128-bit vectors
            ValueType::V2I64 | ValueType::V2F64 => 2,
            ValueType::V4I32 | ValueType::V4F32 => 4,
            ValueType::V8I16 => 8,
            ValueType::V16I8 => 16,
            // 256-bit vectors
            ValueType::V4I64 | ValueType::V4F64 => 4,
            ValueType::V8I32 | ValueType::V8F32 => 8,
            ValueType::V16I16 => 16,
            ValueType::V32I8 => 32,
            // 512-bit vectors
            ValueType::V8I64 | ValueType::V8F64 => 8,
            ValueType::V16I32 | ValueType::V16F32 => 16,
            // Scalars
            _ => 1,
        }
    }

    /// Get the scalar element type for a vector type.
    #[inline]
    pub const fn element_type(self) -> ValueType {
        match self {
            // Integer vectors
            ValueType::V2I64 | ValueType::V4I64 | ValueType::V8I64 => ValueType::Int64,
            ValueType::V4I32 | ValueType::V8I32 | ValueType::V16I32 => ValueType::Int64, // repr as i64
            ValueType::V8I16 | ValueType::V16I16 => ValueType::Int64,
            ValueType::V16I8 | ValueType::V32I8 => ValueType::Int64,
            // Float vectors
            ValueType::V2F64 | ValueType::V4F64 | ValueType::V8F64 => ValueType::Float64,
            ValueType::V4F32 | ValueType::V8F32 | ValueType::V16F32 => ValueType::Float64, // repr as f64
            // Non-vector types return self
            _ => self,
        }
    }

    /// Get the bit width for this type (for vectors, the total width).
    #[inline]
    pub const fn bit_width(self) -> u16 {
        match self {
            // Scalars
            ValueType::Int64 | ValueType::Float64 => 64,
            ValueType::Bool | ValueType::None => 8,
            // 128-bit vectors
            ValueType::V2I64
            | ValueType::V4I32
            | ValueType::V8I16
            | ValueType::V16I8
            | ValueType::V2F64
            | ValueType::V4F32 => 128,
            // 256-bit vectors
            ValueType::V4I64
            | ValueType::V8I32
            | ValueType::V16I16
            | ValueType::V32I8
            | ValueType::V4F64
            | ValueType::V8F32 => 256,
            // 512-bit vectors
            ValueType::V8I64 | ValueType::V16I32 | ValueType::V8F64 | ValueType::V16F32 => 512,
            // Other types
            _ => 64,
        }
    }

    /// Create a vector type from element type and lane count.
    /// Returns None if the combination is not supported.
    #[inline]
    pub const fn vector_of(element: ValueType, lanes: u8) -> Option<ValueType> {
        match (element, lanes) {
            // i64 vectors
            (ValueType::Int64, 2) => Some(ValueType::V2I64),
            (ValueType::Int64, 4) => Some(ValueType::V4I64),
            (ValueType::Int64, 8) => Some(ValueType::V8I64),
            // f64 vectors
            (ValueType::Float64, 2) => Some(ValueType::V2F64),
            (ValueType::Float64, 4) => Some(ValueType::V4F64),
            (ValueType::Float64, 8) => Some(ValueType::V8F64),
            _ => None,
        }
    }

    /// Check if this is a concrete (non-abstract) type.
    #[inline]
    pub const fn is_concrete(self) -> bool {
        !matches!(
            self,
            ValueType::Top
                | ValueType::Numeric
                | ValueType::Sequence
                | ValueType::Iterable
                | ValueType::Object
                | ValueType::Bottom
        )
    }

    /// Lattice meet: compute the greatest lower bound of two types.
    ///
    /// This is used during type inference to narrow types at control merge points.
    #[inline]
    pub const fn meet(self, other: ValueType) -> ValueType {
        use ValueType::*;

        // Identity
        if self as u8 == other as u8 {
            return self;
        }

        // Top dominates
        if matches!(self, Top) {
            return other;
        }
        if matches!(other, Top) {
            return self;
        }

        // Bottom is absorbed
        if matches!(self, Bottom) || matches!(other, Bottom) {
            return Bottom;
        }

        // Numeric subsumes int/float
        if self.is_numeric() && other.is_numeric() {
            if matches!(self, Int64) && matches!(other, Int64) {
                return Int64;
            }
            if matches!(self, Float64) && matches!(other, Float64) {
                return Float64;
            }
            return Numeric;
        }

        // Sequence subsumes list/tuple/string
        if self.is_sequence() && other.is_sequence() {
            if matches!(self, List) && matches!(other, List) {
                return List;
            }
            if matches!(self, Tuple) && matches!(other, Tuple) {
                return Tuple;
            }
            if matches!(self, String) && matches!(other, String) {
                return String;
            }
            return Sequence;
        }

        // Object subsumes all heap types
        if self.is_object() && other.is_object() {
            return Object;
        }

        // Fall back to Top (incompatible types)
        Top
    }

    /// Check if this type is a subtype of another.
    #[inline]
    pub const fn is_subtype_of(self, other: ValueType) -> bool {
        use ValueType::*;

        if self as u8 == other as u8 {
            return true;
        }

        // Top is supertype of everything
        if matches!(other, Top) {
            return true;
        }

        // Bottom is subtype of everything
        if matches!(self, Bottom) {
            return true;
        }

        // Numeric contains int/float
        if matches!(other, Numeric) && self.is_numeric() {
            return true;
        }

        // Sequence contains list/tuple/string
        if matches!(other, Sequence) && self.is_sequence() {
            return true;
        }

        // Object contains all heap types
        if matches!(other, Object) && self.is_object() {
            return true;
        }

        // Iterable contains sequences and more
        if matches!(other, Iterable) {
            return self.is_sequence() || matches!(self, Dict | Set | Iterable);
        }

        false
    }
}

impl fmt::Debug for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::Top => write!(f, "⊤"),
            ValueType::Int64 => write!(f, "i64"),
            ValueType::Float64 => write!(f, "f64"),
            ValueType::Bool => write!(f, "bool"),
            ValueType::None => write!(f, "None"),
            ValueType::Tuple => write!(f, "tuple"),
            ValueType::List => write!(f, "list"),
            ValueType::Dict => write!(f, "dict"),
            ValueType::Set => write!(f, "set"),
            ValueType::String => write!(f, "str"),
            ValueType::Bytes => write!(f, "bytes"),
            ValueType::Function => write!(f, "fn"),
            ValueType::Object => write!(f, "obj"),
            ValueType::Numeric => write!(f, "num"),
            ValueType::Sequence => write!(f, "seq"),
            ValueType::Iterable => write!(f, "iter"),
            // 128-bit vector types
            ValueType::V2I64 => write!(f, "v2i64"),
            ValueType::V4I32 => write!(f, "v4i32"),
            ValueType::V8I16 => write!(f, "v8i16"),
            ValueType::V16I8 => write!(f, "v16i8"),
            ValueType::V2F64 => write!(f, "v2f64"),
            ValueType::V4F32 => write!(f, "v4f32"),
            // 256-bit vector types
            ValueType::V4I64 => write!(f, "v4i64"),
            ValueType::V8I32 => write!(f, "v8i32"),
            ValueType::V16I16 => write!(f, "v16i16"),
            ValueType::V32I8 => write!(f, "v32i8"),
            ValueType::V4F64 => write!(f, "v4f64"),
            ValueType::V8F32 => write!(f, "v8f32"),
            // 512-bit vector types
            ValueType::V8I64 => write!(f, "v8i64"),
            ValueType::V16I32 => write!(f, "v16i32"),
            ValueType::V8F64 => write!(f, "v8f64"),
            ValueType::V16F32 => write!(f, "v16f32"),
            // Special types
            ValueType::Control => write!(f, "ctrl"),
            ValueType::Memory => write!(f, "mem"),
            ValueType::Effect => write!(f, "eff"),
            ValueType::Bottom => write!(f, "⊥"),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl Default for ValueType {
    fn default() -> Self {
        ValueType::Top
    }
}

// =============================================================================
// Type Tuple (for multi-result nodes)
// =============================================================================

/// A tuple of value types (for multi-result nodes like divmod).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeTuple {
    types: [ValueType; 4],
    len: u8,
}

impl TypeTuple {
    /// Empty tuple.
    pub const EMPTY: TypeTuple = TypeTuple {
        types: [ValueType::Bottom; 4],
        len: 0,
    };

    /// Single type.
    pub const fn single(ty: ValueType) -> Self {
        TypeTuple {
            types: [ty, ValueType::Bottom, ValueType::Bottom, ValueType::Bottom],
            len: 1,
        }
    }

    /// Pair of types.
    pub const fn pair(a: ValueType, b: ValueType) -> Self {
        TypeTuple {
            types: [a, b, ValueType::Bottom, ValueType::Bottom],
            len: 2,
        }
    }

    /// Get type at index.
    pub fn get(&self, index: usize) -> Option<ValueType> {
        if index < self.len as usize {
            Some(self.types[index])
        } else {
            None
        }
    }

    /// Get the length.
    pub const fn len(&self) -> usize {
        self.len as usize
    }

    /// Check if empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl fmt::Debug for TypeTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for i in 0..self.len as usize {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", self.types[i])?;
        }
        write!(f, ")")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
}
