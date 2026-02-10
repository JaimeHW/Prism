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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Constant Tests
    // =========================================================================

    #[test]
    fn test_constant_int() {
        let c = Constant::int(42);
        assert!(c.is_int());
        assert!(!c.is_float());
        assert_eq!(c.as_int(), Some(42));
    }

    #[test]
    fn test_constant_float() {
        let c = Constant::float(3.14);
        assert!(c.is_float());
        assert!(!c.is_int());
        assert_eq!(c.as_float(), Some(3.14));
    }

    #[test]
    fn test_constant_bool() {
        let t = Constant::bool(true);
        let f = Constant::bool(false);
        assert!(t.is_bool());
        assert_eq!(t.as_bool(), Some(true));
        assert_eq!(f.as_bool(), Some(false));
    }

    #[test]
    fn test_constant_string() {
        let c = Constant::string("hello");
        assert!(matches!(c, Constant::String(_)));
        if let Constant::String(s) = c {
            assert_eq!(&*s, "hello");
        }
    }

    #[test]
    fn test_constant_truthiness() {
        assert!(Constant::int(1).truthiness());
        assert!(!Constant::int(0).truthiness());
        assert!(Constant::float(0.1).truthiness());
        assert!(!Constant::float(0.0).truthiness());
        assert!(Constant::bool(true).truthiness());
        assert!(!Constant::bool(false).truthiness());
        assert!(!Constant::None.truthiness());
        assert!(Constant::string("x").truthiness());
        assert!(!Constant::string("").truthiness());
        assert!(!Constant::EmptyTuple.truthiness());
        assert!(!Constant::EmptyList.truthiness());
        assert!(!Constant::EmptyDict.truthiness());
    }

    #[test]
    fn test_constant_negate() {
        assert_eq!(Constant::int(5).negate(), Some(Constant::int(-5)));
        assert_eq!(Constant::int(-3).negate(), Some(Constant::int(3)));
        assert_eq!(Constant::float(2.5).negate(), Some(Constant::float(-2.5)));
        assert_eq!(Constant::bool(true).negate(), None);
    }

    #[test]
    fn test_constant_logical_not() {
        assert_eq!(Constant::bool(true).logical_not(), Constant::bool(false));
        assert_eq!(Constant::bool(false).logical_not(), Constant::bool(true));
        assert_eq!(Constant::int(0).logical_not(), Constant::bool(true));
        assert_eq!(Constant::int(1).logical_not(), Constant::bool(false));
    }

    #[test]
    fn test_constant_bitwise_not() {
        assert_eq!(Constant::int(0).bitwise_not(), Some(Constant::int(-1)));
        assert_eq!(Constant::int(-1).bitwise_not(), Some(Constant::int(0)));
        assert_eq!(Constant::float(1.0).bitwise_not(), None);
    }

    #[test]
    fn test_constant_int_to_float_coercion() {
        let c = Constant::int(42);
        assert_eq!(c.as_float(), Some(42.0));
    }

    // =========================================================================
    // LatticeValue Construction Tests
    // =========================================================================

    #[test]
    fn test_lattice_undef() {
        let v = LatticeValue::undef();
        assert!(v.is_undef());
        assert!(!v.is_constant());
        assert!(!v.is_overdefined());
    }

    #[test]
    fn test_lattice_overdefined() {
        let v = LatticeValue::overdefined();
        assert!(v.is_overdefined());
        assert!(!v.is_constant());
        assert!(!v.is_undef());
    }

    #[test]
    fn test_lattice_constant() {
        let v = LatticeValue::int(42);
        assert!(v.is_constant());
        assert!(!v.is_undef());
        assert!(!v.is_overdefined());
        assert_eq!(v.as_constant(), Some(&Constant::Int(42)));
    }

    #[test]
    fn test_lattice_into_constant() {
        let v = LatticeValue::int(100);
        assert_eq!(v.into_constant(), Some(Constant::Int(100)));

        let v = LatticeValue::overdefined();
        assert_eq!(v.into_constant(), None);
    }

    #[test]
    fn test_lattice_default() {
        let v = LatticeValue::default();
        assert!(v.is_undef());
    }

    // =========================================================================
    // Meet Operation Tests
    // =========================================================================

    #[test]
    fn test_meet_undef_identity() {
        let undef = LatticeValue::undef();
        let const_42 = LatticeValue::int(42);
        let overdefined = LatticeValue::overdefined();

        // meet(Undef, x) = x
        assert_eq!(undef.meet(&const_42), const_42);
        assert_eq!(undef.meet(&overdefined), overdefined);
        assert_eq!(undef.meet(&undef), undef);

        // meet(x, Undef) = x
        assert_eq!(const_42.meet(&undef), const_42);
        assert_eq!(overdefined.meet(&undef), overdefined);
    }

    #[test]
    fn test_meet_overdefined_absorbs() {
        let overdefined = LatticeValue::overdefined();
        let const_42 = LatticeValue::int(42);
        let undef = LatticeValue::undef();

        // meet(Overdefined, x) = Overdefined
        assert_eq!(overdefined.meet(&const_42), overdefined);
        assert_eq!(overdefined.meet(&undef), overdefined);
        assert_eq!(overdefined.meet(&overdefined), overdefined);

        // meet(x, Overdefined) = Overdefined
        assert_eq!(const_42.meet(&overdefined), overdefined);
    }

    #[test]
    fn test_meet_same_constants() {
        let a = LatticeValue::int(42);
        let b = LatticeValue::int(42);
        assert_eq!(a.meet(&b), LatticeValue::int(42));

        let x = LatticeValue::float(3.14);
        let y = LatticeValue::float(3.14);
        assert_eq!(x.meet(&y), LatticeValue::float(3.14));
    }

    #[test]
    fn test_meet_different_constants() {
        let a = LatticeValue::int(1);
        let b = LatticeValue::int(2);
        assert_eq!(a.meet(&b), LatticeValue::overdefined());

        let x = LatticeValue::bool(true);
        let y = LatticeValue::bool(false);
        assert_eq!(x.meet(&y), LatticeValue::overdefined());
    }

    // =========================================================================
    // Higher Than Tests
    // =========================================================================

    #[test]
    fn test_higher_than() {
        let undef = LatticeValue::undef();
        let constant = LatticeValue::int(0);
        let overdefined = LatticeValue::overdefined();

        // Overdefined is higher than everything except itself
        assert!(overdefined.higher_than(&constant));
        assert!(overdefined.higher_than(&undef));
        assert!(!overdefined.higher_than(&overdefined));

        // Constant is higher than Undef only
        assert!(constant.higher_than(&undef));
        assert!(!constant.higher_than(&constant));
        assert!(!constant.higher_than(&overdefined));

        // Undef is lower than everything
        assert!(!undef.higher_than(&undef));
        assert!(!undef.higher_than(&constant));
        assert!(!undef.higher_than(&overdefined));
    }

    // =========================================================================
    // Merge Tests
    // =========================================================================

    #[test]
    fn test_merge_undef_to_constant() {
        let mut v = LatticeValue::undef();
        let changed = v.merge(&LatticeValue::int(42));
        assert!(changed);
        assert_eq!(v, LatticeValue::int(42));
    }

    #[test]
    fn test_merge_constant_to_overdefined() {
        let mut v = LatticeValue::int(1);
        let changed = v.merge(&LatticeValue::int(2));
        assert!(changed);
        assert_eq!(v, LatticeValue::overdefined());
    }

    #[test]
    fn test_merge_same_constant_no_change() {
        let mut v = LatticeValue::int(42);
        let changed = v.merge(&LatticeValue::int(42));
        assert!(!changed);
        assert_eq!(v, LatticeValue::int(42));
    }

    #[test]
    fn test_merge_overdefined_no_change() {
        let mut v = LatticeValue::overdefined();
        let changed = v.merge(&LatticeValue::int(42));
        assert!(!changed);
        assert_eq!(v, LatticeValue::overdefined());
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_constant_equality() {
        assert_eq!(Constant::int(0), Constant::int(0));
        assert_ne!(Constant::int(0), Constant::int(1));
        assert_ne!(Constant::int(0), Constant::float(0.0));
    }

    #[test]
    fn test_float_nan_handling() {
        // NaN is not equal to itself, but our constants use Rust's PartialEq
        let nan1 = Constant::float(f64::NAN);
        let nan2 = Constant::float(f64::NAN);
        // Due to NaN != NaN, these will not be equal
        assert_ne!(nan1, nan2);
    }

    #[test]
    fn test_constant_negate_overflow() {
        // i64::MIN cannot be negated without overflow
        let min = Constant::int(i64::MIN);
        assert_eq!(min.negate(), None);
    }
}
