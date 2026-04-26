//! Itertools recipes — common patterns from Python's itertools documentation.
//!
//! These are utility iterators and functions built on top of the core
//! itertools primitives, matching the "recipes" section of Python's
//! itertools documentation.
//!
//! # Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | `flatten` | Flatten one level of nesting |
//! | `unique_everseen` | Unique elements preserving order |
//! | `unique_justseen` | Remove consecutive duplicates |
//! | `sliding_window` | Generic n-width sliding window |
//! | `roundrobin` | Interleave multiple iterators |
//! | `accumulate` | Running totals/reductions |
//! | `partition` | Split by predicate |
//! | `quantify` | Count items matching predicate |

use prism_core::Value;
use std::collections::HashSet;
use std::collections::VecDeque;

// =============================================================================
// Flatten
// =============================================================================

/// Flattens one level of nesting from an iterator of Vec<Value>.
///
/// Equivalent to Python's `itertools.chain.from_iterable(iterable)`.
///
/// # Performance
///
/// - O(1) per element (amortized)
/// - Lazy: processes one inner Vec at a time
#[derive(Debug, Clone)]
pub struct Flatten<I> {
    outer: I,
    inner: std::vec::IntoIter<Value>,
}

impl<I> Flatten<I>
where
    I: Iterator<Item = Vec<Value>>,
{
    /// Create a new flatten iterator.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            outer: iter,
            inner: Vec::new().into_iter(),
        }
    }
}

impl<I> Iterator for Flatten<I>
where
    I: Iterator<Item = Vec<Value>>,
{
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        loop {
            if let Some(val) = self.inner.next() {
                return Some(val);
            }
            let next_vec = self.outer.next()?;
            self.inner = next_vec.into_iter();
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, outer_hi) = self.outer.size_hint();
        let inner_remaining = self.inner.len();
        (
            inner_remaining,
            outer_hi.map(|h| h * 1024 + inner_remaining),
        ) // rough upper bound
    }
}

impl<I> std::iter::FusedIterator for Flatten<I> where I: Iterator<Item = Vec<Value>> {}

// =============================================================================
// UniqueEverseen
// =============================================================================

/// Yields unique elements, preserving first-seen order.
///
/// Equivalent to Python's `more_itertools.unique_everseen()`.
///
/// Uses a `HashSet<u64>` of value bit patterns for O(1) lookup.
///
/// # Performance
///
/// - O(1) amortized per `next()` (hash probe)
/// - O(k) space where k = number of unique elements
///
/// Note: Uses bit-pattern hashing, which works correctly for interned
/// strings, ints, bools, and None. For floats, different NaN bit patterns
/// may not deduplicate (matching Python's behavior for unhashable types).
#[derive(Debug, Clone)]
pub struct UniqueEverseen<I> {
    iter: I,
    seen: HashSet<u64>,
}

impl<I> UniqueEverseen<I>
where
    I: Iterator<Item = Value>,
{
    /// Create a new unique_everseen iterator.
    #[inline]
    pub fn new(iter: I) -> Self {
        let (hint, _) = iter.size_hint();
        Self {
            iter,
            seen: HashSet::with_capacity(hint.min(1024)),
        }
    }

    /// Create with a pre-sized capacity.
    #[inline]
    pub fn with_capacity(iter: I, cap: usize) -> Self {
        Self {
            iter,
            seen: HashSet::with_capacity(cap),
        }
    }
}

impl<I> Iterator for UniqueEverseen<I>
where
    I: Iterator<Item = Value>,
{
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        loop {
            let val = self.iter.next()?;
            let bits = val.to_bits();
            if self.seen.insert(bits) {
                return Some(val);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.iter.size_hint().1)
    }
}

impl<I> std::iter::FusedIterator for UniqueEverseen<I> where I: Iterator<Item = Value> {}

// =============================================================================
// UniqueJustseen
// =============================================================================

/// Removes consecutive duplicate elements.
///
/// Equivalent to the `unique_justseen` recipe from Python docs.
///
/// # Performance
///
/// - O(1) per `next()` — one comparison
/// - O(1) space — stores one previous value
#[derive(Debug, Clone)]
pub struct UniqueJustseen<I> {
    iter: I,
    prev_bits: Option<u64>,
}

impl<I> UniqueJustseen<I>
where
    I: Iterator<Item = Value>,
{
    /// Create a new unique_justseen iterator.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            prev_bits: None,
        }
    }
}

impl<I> Iterator for UniqueJustseen<I>
where
    I: Iterator<Item = Value>,
{
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        loop {
            let val = self.iter.next()?;
            let bits = val.to_bits();
            if self.prev_bits != Some(bits) {
                self.prev_bits = Some(bits);
                return Some(val);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        // At least 1 if source has elements, at most same as source
        (lo.min(1), hi)
    }
}

impl<I> std::iter::FusedIterator for UniqueJustseen<I> where I: Iterator<Item = Value> {}

// =============================================================================
// SlidingWindow
// =============================================================================

/// Generic sliding window of size `n`.
///
/// Equivalent to the `sliding_window` recipe from Python 3.12+ itertools.
///
/// `sliding_window([1,2,3,4,5], 3)` → `[1,2,3], [2,3,4], [3,4,5]`
///
/// # Performance
///
/// - O(1) per `next()` after initial fill (VecDeque push/pop)
/// - O(n) space for the window
#[derive(Debug, Clone)]
pub struct SlidingWindow<I> {
    iter: I,
    window: VecDeque<Value>,
    window_size: usize,
    filled: bool,
}

impl<I> SlidingWindow<I>
where
    I: Iterator<Item = Value>,
{
    /// Create a new sliding window of size `n`.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is 0.
    pub fn new(iter: I, window_size: usize) -> Self {
        assert!(window_size > 0, "sliding_window size must be >= 1");
        Self {
            iter,
            window: VecDeque::with_capacity(window_size),
            window_size,
            filled: false,
        }
    }
}

impl<I> Iterator for SlidingWindow<I>
where
    I: Iterator<Item = Value>,
{
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Vec<Value>> {
        if !self.filled {
            // Fill the initial window
            while self.window.len() < self.window_size {
                match self.iter.next() {
                    Some(val) => self.window.push_back(val),
                    None => return None, // Not enough elements
                }
            }
            self.filled = true;
            return Some(self.window.iter().cloned().collect());
        }

        // Slide: pop front, push back
        let next_val = self.iter.next()?;
        self.window.pop_front();
        self.window.push_back(next_val);
        Some(self.window.iter().cloned().collect())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        if !self.filled {
            (0, hi)
        } else {
            (lo, hi.map(|h| h + 1))
        }
    }
}

impl<I> std::iter::FusedIterator for SlidingWindow<I> where I: Iterator<Item = Value> {}

// =============================================================================
// RoundRobin
// =============================================================================

/// Interleaves elements from multiple iterators.
///
/// Equivalent to the `roundrobin` recipe from Python docs.
///
/// `roundrobin([1,2,3], [4,5], [6,7,8,9])` → `1, 4, 6, 2, 5, 7, 3, 8, 9`
///
/// When an iterator is exhausted, it is removed; the remaining iterators
/// continue to contribute in round-robin order.
///
/// # Performance
///
/// - O(1) per element (amortized)
/// - O(k) space where k = number of active iterators
#[derive(Debug)]
pub struct RoundRobin {
    iterators: VecDeque<std::vec::IntoIter<Value>>,
}

impl RoundRobin {
    /// Create from multiple Vec<Value> iterables.
    pub fn new(iterables: Vec<Vec<Value>>) -> Self {
        let mut iters = VecDeque::with_capacity(iterables.len());
        for iterable in iterables {
            if !iterable.is_empty() {
                iters.push_back(iterable.into_iter());
            }
        }
        Self { iterators: iters }
    }
}

impl Iterator for RoundRobin {
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        loop {
            let mut iter = self.iterators.pop_front()?;
            if let Some(val) = iter.next() {
                self.iterators.push_back(iter);
                return Some(val);
            }
            // Iterator exhausted, drop it and try next
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let total: usize = self.iterators.iter().map(|i| i.len()).sum();
        (total, Some(total))
    }
}

impl ExactSizeIterator for RoundRobin {
    fn len(&self) -> usize {
        self.iterators.iter().map(|i| i.len()).sum()
    }
}

impl std::iter::FusedIterator for RoundRobin {}

// =============================================================================
// Accumulate
// =============================================================================

/// Running accumulation / scan with a binary function.
///
/// Equivalent to Python's `itertools.accumulate(iterable, func, *, initial=None)`.
///
/// The default function is addition. If `initial` is provided, it is placed
/// before the first element.
///
/// # Performance
///
/// - O(1) per `next()` — single function call
/// - O(1) space — stores the running accumulator
#[derive(Clone)]
pub struct Accumulate<I, F> {
    iter: I,
    func: F,
    acc: Option<Value>,
    yielded_initial: bool,
    initial: Option<Value>,
}

impl<I, F> Accumulate<I, F>
where
    I: Iterator<Item = Value>,
    F: FnMut(&Value, &Value) -> Value,
{
    /// Create accumulate with a binary function.
    #[inline]
    pub fn new(iter: I, func: F) -> Self {
        Self {
            iter,
            func,
            acc: None,
            yielded_initial: false,
            initial: None,
        }
    }

    /// Create accumulate with an initial value.
    #[inline]
    pub fn with_initial(iter: I, func: F, initial: Value) -> Self {
        Self {
            iter,
            func,
            acc: None,
            yielded_initial: false,
            initial: Some(initial),
        }
    }
}

impl<I, F> Iterator for Accumulate<I, F>
where
    I: Iterator<Item = Value>,
    F: FnMut(&Value, &Value) -> Value,
{
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        // Yield initial value first
        if !self.yielded_initial {
            self.yielded_initial = true;
            if let Some(initial) = self.initial.take() {
                self.acc = Some(initial.clone());
                return Some(initial);
            }
        }

        let val = self.iter.next()?;
        let result = match self.acc.take() {
            Some(acc) => (self.func)(&acc, &val),
            None => val, // first element (no initial)
        };
        self.acc = Some(result.clone());
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        let extra = if !self.yielded_initial && self.initial.is_some() {
            1
        } else {
            0
        };
        (lo + extra, hi.map(|h| h + extra))
    }
}

// =============================================================================
// Standalone functions
// =============================================================================

/// Split elements by predicate into two Vecs.
///
/// `partition(pred, iterable)` → `(falses, trues)`
///
/// Returns `(Vec<Value>, Vec<Value>)` where the first contains elements
/// where the predicate is false, and the second where it's true.
///
/// # Performance
///
/// - O(n) time, O(n) space
pub fn partition<I, P>(iterable: I, mut pred: P) -> (Vec<Value>, Vec<Value>)
where
    I: Iterator<Item = Value>,
    P: FnMut(&Value) -> bool,
{
    let (hint, _) = iterable.size_hint();
    let half = hint / 2;
    let mut falses = Vec::with_capacity(half);
    let mut trues = Vec::with_capacity(half);

    for item in iterable {
        if pred(&item) {
            trues.push(item);
        } else {
            falses.push(item);
        }
    }

    (falses, trues)
}

/// Count how many items in the iterable match the predicate.
///
/// Equivalent to `sum(1 for x in iterable if pred(x))`.
///
/// # Performance
///
/// - O(n) time, O(1) space
#[inline]
pub fn quantify<I, P>(iterable: I, mut pred: P) -> usize
where
    I: Iterator<Item = Value>,
    P: FnMut(&Value) -> bool,
{
    iterable.filter(|v| pred(v)).count()
}

/// Consume an iterator and return the first `n` and last `n` elements.
///
/// Useful for preview/summary of large iterators.
///
/// # Performance
///
/// - O(total) time, O(n) space for the tail buffer
pub fn head_tail<I>(iterable: I, n: usize) -> (Vec<Value>, Vec<Value>)
where
    I: Iterator<Item = Value>,
{
    if n == 0 {
        // Consume the iterator but return nothing
        let _ = iterable.count();
        return (Vec::new(), Vec::new());
    }

    let mut head = Vec::with_capacity(n);
    let mut tail: VecDeque<Value> = VecDeque::with_capacity(n);
    let mut count = 0;

    for item in iterable {
        if count < n {
            head.push(item.clone());
        }
        if tail.len() >= n {
            tail.pop_front();
        }
        tail.push_back(item);
        count += 1;
    }

    (head, tail.into_iter().collect())
}
