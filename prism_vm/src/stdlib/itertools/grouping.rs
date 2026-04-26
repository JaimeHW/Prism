//! Grouping and windowing iterator constructors.
//!
//! Provides `groupby`, `pairwise`, and `batched` — iterators that group
//! adjacent elements by key, window, or batch size.
//!
//! # Performance Characteristics
//!
//! | Iterator | Time per `next()` | Space |
//! |----------|-------------------|-------|
//! | `GroupBy` | O(group_size) | O(group_size) current group |
//! | `Pairwise` | O(1) | O(1) — one saved element |
//! | `Batched` | O(batch_size) | O(batch_size) per batch |

use prism_core::Value;

// =============================================================================
// GroupBy
// =============================================================================

/// Groups consecutive elements by a key function.
///
/// Equivalent to Python's `itertools.groupby(iterable, key=None)`.
///
/// Yields `(key, group)` pairs where `group` is a `Vec<Value>` of consecutive
/// elements with the same key.
///
/// # Performance
///
/// - O(n) total over all elements — each element is consumed exactly once
/// - O(k) space where k = size of the largest group
///
/// Note: Unlike Python's lazy groupby, this materializes each group into a Vec
/// for simplicity and safety. The Python version's lazy groups have subtle
/// invalidation semantics that don't map well to Rust's ownership model.
#[derive(Clone)]
pub struct GroupBy<I, K>
where
    I: Iterator<Item = Value>,
    K: FnMut(&Value) -> Value,
{
    iter: std::iter::Peekable<I>,
    key_fn: K,
}

impl<I, K> GroupBy<I, K>
where
    I: Iterator<Item = Value>,
    K: FnMut(&Value) -> Value,
{
    /// Create a new groupby iterator.
    #[inline]
    pub fn new(iter: I, key_fn: K) -> Self {
        Self {
            iter: iter.peekable(),
            key_fn,
        }
    }
}

impl<I, K> Iterator for GroupBy<I, K>
where
    I: Iterator<Item = Value>,
    K: FnMut(&Value) -> Value,
{
    type Item = (Value, Vec<Value>);

    fn next(&mut self) -> Option<(Value, Vec<Value>)> {
        // Get the first element and its key
        let first = self.iter.next()?;
        let key = (self.key_fn)(&first);
        let mut group = vec![first];

        // Collect all consecutive elements with the same key
        loop {
            match self.iter.peek() {
                Some(val) => {
                    let next_key = (self.key_fn)(val);
                    if values_equal(&key, &next_key) {
                        group.push(self.iter.next().unwrap());
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }

        Some((key, group))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        // At minimum 0 groups (if empty), at most one group per element
        (0, upper)
    }
}

/// Identity key function — groups by the element itself.
#[inline]
pub fn identity_key(v: &Value) -> Value {
    v.clone()
}

// =============================================================================
// Pairwise
// =============================================================================

/// Yields successive overlapping pairs from the iterable.
///
/// Equivalent to Python's `itertools.pairwise(iterable)`.
///
/// `pairwise([1, 2, 3, 4])` → `(1, 2), (2, 3), (3, 4)`
///
/// # Performance
///
/// - O(1) per `next()` — saves one element
/// - O(1) space — stores exactly one previous value
#[derive(Debug, Clone)]
pub struct Pairwise<I> {
    iter: I,
    prev: Option<Value>,
    started: bool,
}

impl<I> Pairwise<I>
where
    I: Iterator<Item = Value>,
{
    /// Create a new pairwise iterator.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            prev: None,
            started: false,
        }
    }
}

impl<I> Iterator for Pairwise<I>
where
    I: Iterator<Item = Value>,
{
    type Item = (Value, Value);

    #[inline]
    fn next(&mut self) -> Option<(Value, Value)> {
        if !self.started {
            self.started = true;
            self.prev = self.iter.next();
        }

        let prev = self.prev.take()?;
        let next = self.iter.next()?;
        self.prev = Some(next.clone());
        Some((prev, next))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        let lo = lo.saturating_sub(if self.started { 0 } else { 1 });
        let hi = hi.map(|h| h.saturating_sub(if self.started { 0 } else { 1 }));
        (lo, hi)
    }
}

impl<I> std::iter::FusedIterator for Pairwise<I> where I: Iterator<Item = Value> {}

// =============================================================================
// Batched
// =============================================================================

/// Batch elements into fixed-size chunks.
///
/// Equivalent to Python's `itertools.batched(iterable, n)` (Python 3.12+).
///
/// The last batch may be shorter than `n` if the iterable is exhausted.
///
/// # Performance
///
/// - O(n) per `next()` where n = batch size
/// - O(n) space for the current batch
///
/// # Panics
///
/// Panics if `batch_size` is 0.
#[derive(Debug, Clone)]
pub struct Batched<I> {
    iter: I,
    batch_size: usize,
    done: bool,
}

impl<I> Batched<I>
where
    I: Iterator<Item = Value>,
{
    /// Create a new batched iterator.
    ///
    /// # Panics
    ///
    /// Panics if `batch_size` is 0.
    #[inline]
    pub fn new(iter: I, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batched() batch_size must be >= 1");
        Self {
            iter,
            batch_size,
            done: false,
        }
    }
}

impl<I> Iterator for Batched<I>
where
    I: Iterator<Item = Value>,
{
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Vec<Value>> {
        if self.done {
            return None;
        }

        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            match self.iter.next() {
                Some(val) => batch.push(val),
                None => {
                    self.done = true;
                    break;
                }
            }
        }

        if batch.is_empty() {
            self.done = true;
            None
        } else {
            Some(batch)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }
        let (lo, hi) = self.iter.size_hint();
        let lo = (lo + self.batch_size - 1) / self.batch_size;
        let hi = hi.map(|h| (h + self.batch_size - 1) / self.batch_size);
        (lo, hi)
    }
}

impl<I> std::iter::FusedIterator for Batched<I> where I: Iterator<Item = Value> {}

// =============================================================================
// Triplewise
// =============================================================================

/// Yields successive overlapping triples from the iterable.
///
/// `triplewise([1, 2, 3, 4, 5])` → `(1, 2, 3), (2, 3, 4), (3, 4, 5)`
///
/// # Performance
///
/// - O(1) per `next()`
/// - O(1) space — stores exactly two previous values
#[derive(Debug, Clone)]
pub struct Triplewise<I> {
    iter: I,
    prev1: Option<Value>,
    prev2: Option<Value>,
    started: bool,
}

impl<I> Triplewise<I>
where
    I: Iterator<Item = Value>,
{
    /// Create a new triplewise iterator.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            prev1: None,
            prev2: None,
            started: false,
        }
    }
}

impl<I> Iterator for Triplewise<I>
where
    I: Iterator<Item = Value>,
{
    type Item = (Value, Value, Value);

    #[inline]
    fn next(&mut self) -> Option<(Value, Value, Value)> {
        if !self.started {
            self.started = true;
            self.prev1 = self.iter.next();
            self.prev2 = self.iter.next();
        }

        let a = self.prev1.take()?;
        let b = self.prev2.take()?;
        let c = self.iter.next()?;

        self.prev1 = Some(b.clone());
        self.prev2 = Some(c.clone());
        Some((a, b, c))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        (lo, hi)
    }
}

impl<I> std::iter::FusedIterator for Triplewise<I> where I: Iterator<Item = Value> {}

// =============================================================================
// Helpers
// =============================================================================

/// Compare two Values for equality (used by groupby).
///
/// Handles int, float, bool, none, and string comparison.
#[inline]
pub fn values_equal(a: &Value, b: &Value) -> bool {
    // Fast path: bitwise equality (works for interned strings, ints, bools, none)
    if a.to_bits() == b.to_bits() {
        return true;
    }
    // Float comparison (handles different bit patterns for same value)
    if let (Some(fa), Some(fb)) = (a.as_float(), b.as_float()) {
        return fa == fb;
    }
    // Int-float cross comparison
    if let (Some(ia), Some(fb)) = (a.as_int(), b.as_float()) {
        return (ia as f64) == fb;
    }
    if let (Some(fa), Some(ib)) = (a.as_float(), b.as_int()) {
        return fa == (ib as f64);
    }
    false
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
