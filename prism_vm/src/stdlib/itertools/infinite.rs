//! Infinite iterator constructors.
//!
//! Provides `count`, `cycle`, and `repeat` — the three infinite iterators
//! from Python's `itertools` module.
//!
//! # Performance Characteristics
//!
//! | Iterator | Time per `next()` | Space |
//! |----------|-------------------|-------|
//! | `Count`  | O(1) — single add | O(1) inline |
//! | `Cycle`  | O(1) amortized    | O(n) pool clone |
//! | `Repeat` | O(1) — clone      | O(1) inline |
//!
//! All iterators are fused and provide accurate `size_hint()` where applicable.

use prism_core::Value;

// =============================================================================
// Count
// =============================================================================

/// Infinite counter starting at `start` with a given `step`.
///
/// Equivalent to Python's `itertools.count(start=0, step=1)`.
///
/// # Performance
///
/// Each `next()` call is a single integer addition — O(1) time, O(1) space.
/// For integer-only counts, uses `i64` arithmetic (no float conversion).
/// When either `start` or `step` is a float, promotes to `f64` arithmetic.
#[derive(Debug, Clone)]
pub struct Count {
    current: CountState,
    step: CountStep,
}

/// Internal state: either pure integer or float arithmetic.
#[derive(Debug, Clone)]
enum CountState {
    Int(i64),
    Float(f64),
}

/// Step value for count iterator.
#[derive(Debug, Clone, Copy)]
enum CountStep {
    Int(i64),
    Float(f64),
}

impl Count {
    /// Create a new integer counter.
    ///
    /// # Examples
    /// ```ignore
    /// let c = Count::new(0, 1);
    /// // yields: 0, 1, 2, 3, ...
    /// ```
    #[inline]
    pub fn new(start: i64, step: i64) -> Self {
        Self {
            current: CountState::Int(start),
            step: CountStep::Int(step),
        }
    }

    /// Create a float counter.
    ///
    /// # Examples
    /// ```ignore
    /// let c = Count::new_float(0.5, 0.1);
    /// // yields: 0.5, 0.6, 0.7, ...
    /// ```
    #[inline]
    pub fn new_float(start: f64, step: f64) -> Self {
        Self {
            current: CountState::Float(start),
            step: CountStep::Float(step),
        }
    }

    /// Create a counter from Value arguments.
    ///
    /// Automatically promotes to float arithmetic if either argument is a float.
    #[inline]
    pub fn from_values(start: &Value, step: &Value) -> Option<Self> {
        // Try pure integer first
        if let (Some(s), Some(st)) = (start.as_int(), step.as_int()) {
            return Some(Self::new(s, st));
        }
        // Fall back to float coercion
        let s = start
            .as_float()
            .or_else(|| start.as_int().map(|i| i as f64))?;
        let st = step
            .as_float()
            .or_else(|| step.as_int().map(|i| i as f64))?;
        Some(Self::new_float(s, st))
    }
}

impl Iterator for Count {
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        let step = self.step;
        match &mut self.current {
            CountState::Int(curr) => match step {
                CountStep::Int(s) => {
                    let val = *curr;
                    *curr = curr.wrapping_add(s);
                    Some(Value::int_unchecked(val))
                }
                CountStep::Float(s) => {
                    let val = *curr as f64;
                    self.current = CountState::Float(val + s);
                    Some(Value::float(val))
                }
            },
            CountState::Float(curr) => {
                let val = *curr;
                match step {
                    CountStep::Float(s) => *curr += s,
                    CountStep::Int(s) => *curr += s as f64,
                }
                Some(Value::float(val))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None) // infinite
    }
}

impl std::iter::FusedIterator for Count {}

// =============================================================================
// Cycle
// =============================================================================

/// Infinitely repeats the elements of a saved pool.
///
/// Equivalent to Python's `itertools.cycle(iterable)`.
///
/// # Performance
///
/// - First pass: O(1) per element (consuming the source iterator + saving)
/// - Subsequent passes: O(1) per element (index into pool)
/// - Space: O(n) where n is the number of elements in the source
///
/// The pool is pre-allocated with the source iterator's `size_hint` for
/// zero-reallocation in the common case.
#[derive(Debug, Clone)]
pub struct Cycle {
    /// Saved elements for replay.
    pool: Vec<Value>,
    /// Current index into the pool during replay.
    index: usize,
    /// Whether we've finished consuming the source.
    exhausted: bool,
    /// Source iterator (consumed on first pass, then dropped conceptually).
    source: Vec<Value>,
    /// Position in source during first pass.
    source_pos: usize,
}

impl Cycle {
    /// Create from an iterable, consuming it into a pool.
    ///
    /// The pool is eagerly built for maximum replay performance.
    pub fn new(iterable: impl IntoIterator<Item = Value>) -> Self {
        let iter = iterable.into_iter();
        let (hint, _) = iter.size_hint();
        let mut pool = Vec::with_capacity(hint);
        let source: Vec<Value> = iter.collect();
        Self {
            pool,
            index: 0,
            exhausted: false,
            source,
            source_pos: 0,
        }
    }

    /// Create from a pre-built pool (avoids double collection).
    #[inline]
    pub fn from_pool(pool: Vec<Value>) -> Self {
        Self {
            pool,
            index: 0,
            exhausted: true,
            source: Vec::new(),
            source_pos: 0,
        }
    }

    /// Returns the number of elements in the cycle pool.
    #[inline]
    pub fn pool_len(&self) -> usize {
        if self.exhausted {
            self.pool.len()
        } else {
            self.source.len()
        }
    }
}

impl Iterator for Cycle {
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        if !self.exhausted {
            // First pass: consume from source
            if self.source_pos < self.source.len() {
                let val = self.source[self.source_pos].clone();
                self.pool.push(val.clone());
                self.source_pos += 1;
                return Some(val);
            }
            // Source exhausted
            self.exhausted = true;
            // Free the source
            self.source = Vec::new();
            self.index = 0;
        }

        // Replay from pool
        if self.pool.is_empty() {
            return None; // Empty source = no cycle
        }

        let val = self.pool[self.index].clone();
        self.index = (self.index + 1) % self.pool.len();
        Some(val)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if !self.exhausted && self.source.is_empty() && self.pool.is_empty() {
            (0, Some(0))
        } else {
            (usize::MAX, None)
        }
    }
}

impl std::iter::FusedIterator for Cycle {}

// =============================================================================
// Repeat
// =============================================================================

/// Repeats a single value, either infinitely or a fixed number of times.
///
/// Equivalent to Python's `itertools.repeat(object[, times])`.
///
/// # Performance
///
/// - O(1) per `next()` (clone of the stored value)
/// - O(1) space (stores one value + optional counter)
/// - Implements `ExactSizeIterator` when bounded
#[derive(Debug, Clone)]
pub struct Repeat {
    value: Value,
    remaining: Option<usize>,
}

impl Repeat {
    /// Create an infinite repeater.
    #[inline]
    pub fn forever(value: Value) -> Self {
        Self {
            value,
            remaining: None,
        }
    }

    /// Create a bounded repeater.
    #[inline]
    pub fn times(value: Value, n: usize) -> Self {
        Self {
            value,
            remaining: Some(n),
        }
    }

    /// Check if this repeater is bounded.
    #[inline]
    pub fn is_bounded(&self) -> bool {
        self.remaining.is_some()
    }

    /// Get remaining count (None if infinite).
    #[inline]
    pub fn remaining(&self) -> Option<usize> {
        self.remaining
    }
}

impl Iterator for Repeat {
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        match &mut self.remaining {
            Some(0) => None,
            Some(n) => {
                *n -= 1;
                Some(self.value.clone())
            }
            None => Some(self.value.clone()),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.remaining {
            Some(n) => (n, Some(n)),
            None => (usize::MAX, None),
        }
    }
}

impl ExactSizeIterator for Repeat {
    fn len(&self) -> usize {
        self.remaining.unwrap_or(usize::MAX)
    }
}

impl std::iter::FusedIterator for Repeat {}
