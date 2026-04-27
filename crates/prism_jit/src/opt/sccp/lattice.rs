//! Value Lattice for SCCP.
//!
//! This module implements the lattice structure used by Sparse Conditional
//! Constant Propagation. The lattice represents knowledge about a value:
//!
//! ```text
//!          Undef (⊥)
//!            |
//!     Constant(v1, v2, ...)
//!            |
//!      Overdefined (⊤)
//! ```
//!
//! Values flow upward through the lattice:
//! - `Undef`: Value is never defined (bottom element)
//! - `Constant(v)`: Value is known to be constant `v`
//! - `Overdefined`: Value varies at runtime (top element)
//!
//! The lattice is finite-height (3 levels), guaranteeing termination.

use std::sync::Arc;

// =============================================================================
// Constant Representation
// =============================================================================

/// Compile-time constant values.
///
/// Supports all Python constant types relevant to optimization.
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit float.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// Python None.
    None,
    /// Interned string constant.
    String(Arc<str>),
    /// Empty tuple (common pattern).
    EmptyTuple,
    /// Empty list.
    EmptyList,
    /// Empty dict.
    EmptyDict,
}

impl Constant {
    /// Create an integer constant.
    #[inline]
    pub fn int(v: i64) -> Self {
        Self::Int(v)
    }

    /// Create a float constant.
    #[inline]
    pub fn float(v: f64) -> Self {
        Self::Float(v)
    }

    /// Create a boolean constant.
    #[inline]
    pub fn bool(v: bool) -> Self {
        Self::Bool(v)
    }

    /// Create a string constant.
    #[inline]
    pub fn string(s: impl Into<Arc<str>>) -> Self {
        Self::String(s.into())
    }

    /// Check if this is an integer.
    #[inline]
    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    /// Check if this is a float.
    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float(_))
    }

    /// Check if this is a boolean.
    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool(_))
    }

    /// Get as integer if applicable.
    #[inline]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as float if applicable.
    #[inline]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            // Integer to float coercion
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get as boolean if applicable.
    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get truthiness of this constant.
    pub fn truthiness(&self) -> bool {
        match self {
            Self::Int(v) => *v != 0,
            Self::Float(v) => *v != 0.0,
            Self::Bool(v) => *v,
            Self::None => false,
            Self::String(s) => !s.is_empty(),
            Self::EmptyTuple | Self::EmptyList | Self::EmptyDict => false,
        }
    }

    /// Negate this constant if it's numeric.
    pub fn negate(&self) -> Option<Self> {
        match self {
            Self::Int(v) => v.checked_neg().map(Self::Int),
            Self::Float(v) => Some(Self::Float(-v)),
            _ => None,
        }
    }

    /// Logical not of this constant.
    pub fn logical_not(&self) -> Self {
        Self::Bool(!self.truthiness())
    }

    /// Bitwise not of this constant if it's an integer.
    pub fn bitwise_not(&self) -> Option<Self> {
        match self {
            Self::Int(v) => Some(Self::Int(!v)),
            _ => None,
        }
    }
}

// =============================================================================
// Lattice Value
// =============================================================================

/// SCCP lattice value representing knowledge about a node's value.
///
/// The lattice has three levels:
/// 1. `Undef` (bottom) - Value is undefined/never computed
/// 2. `Constant(v)` - Value is the constant `v`  
/// 3. `Overdefined` (top) - Value varies at runtime
#[derive(Debug, Clone, PartialEq)]
pub enum LatticeValue {
    /// Value is undefined (never computed).
    ///
    /// This is the bottom element of the lattice.
    /// All values start here until proven otherwise.
    Undef,

    /// Value is a known constant.
    ///
    /// This represents definite knowledge that the node
    /// always produces this specific value.
    Constant(Constant),

    /// Value varies at runtime.
    ///
    /// This is the top element of the lattice.
    /// Once a value becomes Overdefined, it stays that way.
    Overdefined,
}

impl LatticeValue {
    /// Create an undefined lattice value.
    #[inline]
    pub fn undef() -> Self {
        Self::Undef
    }

    /// Create an overdefined lattice value.
    #[inline]
    pub fn overdefined() -> Self {
        Self::Overdefined
    }

    /// Create a constant lattice value.
    #[inline]
    pub fn constant(c: Constant) -> Self {
        Self::Constant(c)
    }

    /// Create an integer constant.
    #[inline]
    pub fn int(v: i64) -> Self {
        Self::Constant(Constant::Int(v))
    }

    /// Create a float constant.
    #[inline]
    pub fn float(v: f64) -> Self {
        Self::Constant(Constant::Float(v))
    }

    /// Create a boolean constant.
    #[inline]
    pub fn bool(v: bool) -> Self {
        Self::Constant(Constant::Bool(v))
    }

    /// Check if this is undefined.
    #[inline]
    pub fn is_undef(&self) -> bool {
        matches!(self, Self::Undef)
    }

    /// Check if this is a constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Check if this is overdefined.
    #[inline]
    pub fn is_overdefined(&self) -> bool {
        matches!(self, Self::Overdefined)
    }

    /// Get the constant value if available.
    #[inline]
    pub fn as_constant(&self) -> Option<&Constant> {
        match self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Get the constant value, consuming self.
    #[inline]
    pub fn into_constant(self) -> Option<Constant> {
        match self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Meet operation (lattice join).
    ///
    /// Computes the least upper bound of two lattice values.
    /// This is used when control flow merges (phi nodes).
    ///
    /// ```text
    /// meet(Undef, x) = x
    /// meet(x, Undef) = x
    /// meet(Constant(a), Constant(a)) = Constant(a)
    /// meet(Constant(a), Constant(b)) = Overdefined  (if a != b)
    /// meet(Overdefined, x) = Overdefined
    /// meet(x, Overdefined) = Overdefined
    /// ```
    pub fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            // Bottom element is identity
            (Self::Undef, x) | (x, Self::Undef) => x.clone(),

            // Top element absorbs
            (Self::Overdefined, _) | (_, Self::Overdefined) => Self::Overdefined,

            // Same constant stays constant
            (Self::Constant(a), Self::Constant(b)) => {
                if a == b {
                    Self::Constant(a.clone())
                } else {
                    Self::Overdefined
                }
            }
        }
    }

    /// Check if this value is higher in the lattice than another.
    ///
    /// Returns true if self > other in the partial order.
    pub fn higher_than(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Overdefined, Self::Constant(_)) => true,
            (Self::Overdefined, Self::Undef) => true,
            (Self::Constant(_), Self::Undef) => true,
            _ => false,
        }
    }

    /// Merge this value with another (in-place meet).
    ///
    /// Returns true if `self` was changed.
    pub fn merge(&mut self, other: &Self) -> bool {
        let new_value = self.meet(other);
        if new_value != *self {
            *self = new_value;
            true
        } else {
            false
        }
    }
}

impl Default for LatticeValue {
    fn default() -> Self {
        Self::Undef
    }
}
