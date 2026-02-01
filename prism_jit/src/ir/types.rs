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
}
