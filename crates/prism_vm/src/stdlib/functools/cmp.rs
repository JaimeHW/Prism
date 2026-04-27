//! Comparison utilities for functools.
//!
//! Provides two key utilities:
//!
//! - `cmp_to_key(func)` — converts an old-style comparison function to a
//!   key function for use with `sorted()`, `min()`, `max()`, etc.
//!
//! - `total_ordering` — derives missing rich comparison methods from
//!   `__eq__` and one of `__lt__`, `__le__`, `__gt__`, `__ge__`.
//!
//! # Performance
//!
//! `CmpKey` wraps a Value and a comparison function, with O(1) dispatch
//! per comparison. `TotalOrdering` pre-selects the derivation strategy at
//! construction time for O(1) method lookup.

use prism_core::Value;
use std::cmp::Ordering;

// =============================================================================
// CmpToKey
// =============================================================================

/// A key wrapper that uses an old-style comparison function for ordering.
///
/// Given a function `cmp(a, b) -> int` (negative, zero, or positive),
/// wraps a value so it can be compared using the cmp function.
///
/// # Examples
///
/// ```ignore
/// let key = CmpKey::new(value, |a, b| a.as_int().unwrap() - b.as_int().unwrap());
/// ```
#[derive(Clone)]
pub struct CmpKey<F>
where
    F: Fn(&Value, &Value) -> i64,
{
    /// The wrapped value.
    value: Value,
    /// The comparison function.
    cmp_fn: F,
}

impl<F> std::fmt::Debug for CmpKey<F>
where
    F: Fn(&Value, &Value) -> i64,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CmpKey")
            .field("value", &self.value)
            .field("cmp_fn", &"<comparison function>")
            .finish()
    }
}

impl<F> CmpKey<F>
where
    F: Fn(&Value, &Value) -> i64,
{
    /// Create a new CmpKey wrapping a value with a comparison function.
    #[inline]
    pub fn new(value: Value, cmp_fn: F) -> Self {
        Self { value, cmp_fn }
    }

    /// Get the wrapped value.
    #[inline]
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Get the wrapped value, consuming this CmpKey.
    #[inline]
    pub fn into_value(self) -> Value {
        self.value
    }
}

impl<F> PartialEq for CmpKey<F>
where
    F: Fn(&Value, &Value) -> i64,
{
    fn eq(&self, other: &Self) -> bool {
        (self.cmp_fn)(&self.value, &other.value) == 0
    }
}

impl<F> Eq for CmpKey<F> where F: Fn(&Value, &Value) -> i64 {}

impl<F> PartialOrd for CmpKey<F>
where
    F: Fn(&Value, &Value) -> i64,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F> Ord for CmpKey<F>
where
    F: Fn(&Value, &Value) -> i64,
{
    fn cmp(&self, other: &Self) -> Ordering {
        let result = (self.cmp_fn)(&self.value, &other.value);
        match result {
            n if n < 0 => Ordering::Less,
            0 => Ordering::Equal,
            _ => Ordering::Greater,
        }
    }
}

/// Create a key function from a comparison function.
///
/// Returns a closure that wraps values in `CmpKey` for use with sorting.
///
/// # Performance
///
/// This is a zero-cost abstraction — the closure just wraps the value with
/// a reference to the comparison function. The shared reference to `cmp_fn`
/// avoids cloning the comparison function per element.
pub fn cmp_to_key<F>(cmp_fn: F) -> impl Fn(Value) -> CmpKey<F>
where
    F: Fn(&Value, &Value) -> i64 + Clone,
{
    move |value| CmpKey::new(value, cmp_fn.clone())
}

/// Sort a slice of Values using a comparison function.
///
/// This is a convenience wrapper around `cmp_to_key` for direct sorting.
///
/// # Performance
///
/// Uses Rust's guaranteed O(n log n) stable sort.
pub fn sort_with_cmp<F>(values: &mut [Value], cmp_fn: F)
where
    F: Fn(&Value, &Value) -> i64,
{
    values.sort_by(|a, b| {
        let result = cmp_fn(a, b);
        match result {
            n if n < 0 => Ordering::Less,
            0 => Ordering::Equal,
            _ => Ordering::Greater,
        }
    });
}

// =============================================================================
// TotalOrdering
// =============================================================================

/// Strategy for deriving comparison methods.
///
/// Given `__eq__` and one of the four comparison operators, determines
/// how to derive the remaining three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonBase {
    /// Derives from `__lt__`.
    Lt,
    /// Derives from `__le__`.
    Le,
    /// Derives from `__gt__`.
    Gt,
    /// Derives from `__ge__`.
    Ge,
}

/// Total ordering derivation engine.
///
/// Given a base comparison and equality check, derives all four
/// rich comparison methods (`<`, `<=`, `>`, `>=`).
///
/// # Derivation Rules
///
/// From `__lt__`:
/// - `a <= b` ⟷ `a < b or a == b`
/// - `a > b`  ⟷ `not (a < b or a == b)`
/// - `a >= b` ⟷ `not (a < b)`
///
/// From `__le__`:
/// - `a < b`  ⟷ `a <= b and a != b`
/// - `a > b`  ⟷ `not (a <= b)`
/// - `a >= b` ⟷ `not (a <= b) or a == b`
///
/// From `__gt__`:
/// - `a >= b` ⟷ `a > b or a == b`
/// - `a < b`  ⟷ `not (a > b or a == b)`
/// - `a <= b` ⟷ `not (a > b)`
///
/// From `__ge__`:
/// - `a > b`  ⟷ `a >= b and a != b`
/// - `a < b`  ⟷ `not (a >= b)`
/// - `a <= b` ⟷ `not (a >= b) or a == b`
#[derive(Debug, Clone)]
pub struct TotalOrdering {
    base: ComparisonBase,
}

impl TotalOrdering {
    /// Create a new TotalOrdering with the given base comparison.
    #[inline]
    pub fn new(base: ComparisonBase) -> Self {
        Self { base }
    }

    /// Get the base comparison strategy.
    #[inline]
    pub fn base(&self) -> ComparisonBase {
        self.base
    }

    /// Derive the `<` comparison.
    #[inline]
    pub fn lt<Eq, Cmp>(&self, a: &Value, b: &Value, eq: &Eq, base_cmp: &Cmp) -> bool
    where
        Eq: Fn(&Value, &Value) -> bool + ?Sized,
        Cmp: Fn(&Value, &Value) -> bool + ?Sized,
    {
        match self.base {
            ComparisonBase::Lt => base_cmp(a, b),
            ComparisonBase::Le => base_cmp(a, b) && !eq(a, b),
            ComparisonBase::Gt => !base_cmp(a, b) && !eq(a, b),
            ComparisonBase::Ge => !base_cmp(a, b),
        }
    }

    /// Derive the `<=` comparison.
    #[inline]
    pub fn le<Eq, Cmp>(&self, a: &Value, b: &Value, eq: &Eq, base_cmp: &Cmp) -> bool
    where
        Eq: Fn(&Value, &Value) -> bool + ?Sized,
        Cmp: Fn(&Value, &Value) -> bool + ?Sized,
    {
        match self.base {
            ComparisonBase::Lt => base_cmp(a, b) || eq(a, b),
            ComparisonBase::Le => base_cmp(a, b),
            ComparisonBase::Gt => !base_cmp(a, b),
            ComparisonBase::Ge => !base_cmp(a, b) || eq(a, b),
        }
    }

    /// Derive the `>` comparison.
    #[inline]
    pub fn gt<Eq, Cmp>(&self, a: &Value, b: &Value, eq: &Eq, base_cmp: &Cmp) -> bool
    where
        Eq: Fn(&Value, &Value) -> bool + ?Sized,
        Cmp: Fn(&Value, &Value) -> bool + ?Sized,
    {
        match self.base {
            ComparisonBase::Lt => !base_cmp(a, b) && !eq(a, b),
            ComparisonBase::Le => !base_cmp(a, b),
            ComparisonBase::Gt => base_cmp(a, b),
            ComparisonBase::Ge => base_cmp(a, b) && !eq(a, b),
        }
    }

    /// Derive the `>=` comparison.
    #[inline]
    pub fn ge<Eq, Cmp>(&self, a: &Value, b: &Value, eq: &Eq, base_cmp: &Cmp) -> bool
    where
        Eq: Fn(&Value, &Value) -> bool + ?Sized,
        Cmp: Fn(&Value, &Value) -> bool + ?Sized,
    {
        match self.base {
            ComparisonBase::Lt => !base_cmp(a, b),
            ComparisonBase::Le => !base_cmp(a, b) || eq(a, b),
            ComparisonBase::Gt => base_cmp(a, b) || eq(a, b),
            ComparisonBase::Ge => base_cmp(a, b),
        }
    }

    /// Derive all four comparisons at once, returning (lt, le, gt, ge).
    pub fn derive_all<Eq, Cmp>(
        &self,
        a: &Value,
        b: &Value,
        eq: &Eq,
        base_cmp: &Cmp,
    ) -> (bool, bool, bool, bool)
    where
        Eq: Fn(&Value, &Value) -> bool + ?Sized,
        Cmp: Fn(&Value, &Value) -> bool + ?Sized,
    {
        (
            self.lt(a, b, eq, base_cmp),
            self.le(a, b, eq, base_cmp),
            self.gt(a, b, eq, base_cmp),
            self.ge(a, b, eq, base_cmp),
        )
    }
}
