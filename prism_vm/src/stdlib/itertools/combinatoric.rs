//! Combinatoric iterator constructors.
//!
//! Provides `product`, `permutations`, `combinations`, and
//! `combinations_with_replacement` — producing all possible arrangements
//! from input pools.
//!
//! # Performance Characteristics
//!
//! | Iterator | Output Size | Space |
//! |----------|-------------|-------|
//! | `Product` (n pools, each k) | k^n | O(n) indices + O(k*n) pools |
//! | `Permutations(n, r)` | n!/(n-r)! | O(n) indices + O(n) pool |
//! | `Combinations(n, r)` | C(n,r) | O(r) indices + O(n) pool |
//! | `CombWithRepl(n, r)` | C(n+r-1,r) | O(r) indices + O(n) pool |
//!
//! All use SmallVec for index arrays to avoid heap allocation for small `r`.

use prism_core::Value;
use smallvec::SmallVec;

/// Stack-allocated index threshold. For r <= 8, indices live on the stack.
const SMALL_INDEX: usize = 8;

type IndexVec = SmallVec<[usize; SMALL_INDEX]>;

// =============================================================================
// Product
// =============================================================================

/// Cartesian product of input iterables.
///
/// Equivalent to Python's `itertools.product(*iterables, repeat=1)`.
///
/// # Algorithm
///
/// Uses odometer-style index advancement: the rightmost index increments
/// first, cascading left on overflow. This produces lexicographic order
/// matching CPython exactly.
///
/// # Performance
///
/// - O(1) per `next()` (amortized — index cascade is O(n) worst case
///   but averaged over all outputs is O(1))
/// - O(n) space for the index vector where n = number of pools
#[derive(Debug, Clone)]
pub struct Product {
    /// The pools of values to combine.
    pools: Vec<Vec<Value>>,
    /// Current index into each pool.
    indices: IndexVec,
    /// Whether we've finished.
    done: bool,
    /// Whether this is the first call to next().
    first: bool,
}

impl Product {
    /// Create a Cartesian product from multiple pools.
    pub fn new(pools: Vec<Vec<Value>>) -> Self {
        // If any pool is empty, the product is empty
        let done = pools.iter().any(|p| p.is_empty());
        let n = pools.len();
        let mut indices = IndexVec::with_capacity(n);
        indices.resize(n, 0);

        Self {
            pools,
            indices,
            done,
            first: true,
        }
    }

    /// Create with repeat (Python's `repeat` parameter).
    ///
    /// `product(pool, repeat=3)` = `product(pool, pool, pool)`
    pub fn with_repeat(pool: Vec<Value>, repeat: usize) -> Self {
        let pools = vec![pool; repeat];
        Self::new(pools)
    }

    /// Get the total number of elements in the product.
    pub fn total_size(&self) -> usize {
        if self.pools.is_empty() {
            return 1; // empty product yields one empty tuple
        }
        self.pools.iter().map(|p| p.len()).product()
    }

    /// Build the current tuple from indices.
    #[inline]
    fn current_tuple(&self) -> Vec<Value> {
        self.indices
            .iter()
            .zip(self.pools.iter())
            .map(|(&idx, pool)| pool[idx].clone())
            .collect()
    }

    /// Advance the odometer indices (rightmost first).
    #[inline]
    fn advance(&mut self) -> bool {
        // Increment from the right, cascading left on overflow
        for i in (0..self.indices.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.pools[i].len() {
                return true;
            }
            self.indices[i] = 0;
        }
        false // all indices wrapped around → done
    }
}

impl Iterator for Product {
    type Item = Vec<Value>;

    #[inline]
    fn next(&mut self) -> Option<Vec<Value>> {
        if self.done {
            return None;
        }

        if self.first {
            self.first = false;
            if self.pools.is_empty() {
                self.done = true;
                return Some(Vec::new()); // single empty tuple
            }
            return Some(self.current_tuple());
        }

        if self.advance() {
            Some(self.current_tuple())
        } else {
            self.done = true;
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            (0, Some(0))
        } else {
            let total = self.total_size();
            (total, Some(total))
        }
    }
}

impl std::iter::FusedIterator for Product {}

// =============================================================================
// Permutations
// =============================================================================

/// Successive r-length permutations of elements from the pool.
///
/// Equivalent to Python's `itertools.permutations(iterable, r=None)`.
///
/// # Algorithm
///
/// Uses the lexicographic permutation generation algorithm. Indices cycle through
/// all possible orderings without repetition.
///
/// # Performance
///
/// - O(r) per `next()` for tuple construction
/// - O(n) space for pool + indices + cycles
#[derive(Debug, Clone)]
pub struct Permutations {
    pool: Vec<Value>,
    indices: Vec<usize>,
    cycles: Vec<usize>,
    r: usize,
    first: bool,
    done: bool,
}

impl Permutations {
    /// Create permutations of length `r` from the pool.
    pub fn new(pool: Vec<Value>, r: usize) -> Self {
        let n = pool.len();
        if r > n {
            return Self {
                pool,
                indices: Vec::new(),
                cycles: Vec::new(),
                r,
                first: true,
                done: true,
            };
        }

        let indices: Vec<usize> = (0..n).collect();
        let cycles: Vec<usize> = (n - r + 1..=n).rev().collect();

        Self {
            pool,
            indices,
            cycles,
            r,
            first: true,
            done: false,
        }
    }

    /// Create full-length permutations.
    pub fn full(pool: Vec<Value>) -> Self {
        let r = pool.len();
        Self::new(pool, r)
    }

    /// Get the current permutation tuple.
    #[inline]
    fn current_tuple(&self) -> Vec<Value> {
        self.indices[..self.r]
            .iter()
            .map(|&i| self.pool[i].clone())
            .collect()
    }
}

impl Iterator for Permutations {
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Vec<Value>> {
        if self.done {
            return None;
        }

        if self.first {
            self.first = false;
            return Some(self.current_tuple());
        }

        let n = self.pool.len();

        // Advance using the cycles algorithm (identical to CPython)
        for i in (0..self.r).rev() {
            self.cycles[i] -= 1;
            if self.cycles[i] == 0 {
                // Rotate indices[i..n] left by 1
                let saved = self.indices[i];
                for j in i..n - 1 {
                    self.indices[j] = self.indices[j + 1];
                }
                self.indices[n - 1] = saved;
                self.cycles[i] = n - i;
            } else {
                let j = n - self.cycles[i];
                self.indices.swap(i, j);
                return Some(self.current_tuple());
            }
        }

        self.done = true;
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }
        // n! / (n-r)! but we don't track position, so approximate
        (0, None)
    }
}

impl std::iter::FusedIterator for Permutations {}

// =============================================================================
// Combinations
// =============================================================================

/// Successive r-length combinations of elements from the pool.
///
/// Equivalent to Python's `itertools.combinations(iterable, r)`.
///
/// # Algorithm
///
/// Index-based lexicographic generation. Each combination is represented
/// by `r` indices `i_0 < i_1 < ... < i_{r-1}` into the pool.
///
/// # Performance
///
/// - O(r) per `next()` for advance + tuple construction
/// - O(r) space for indices (SmallVec-backed)
#[derive(Debug, Clone)]
pub struct Combinations {
    pool: Vec<Value>,
    indices: IndexVec,
    r: usize,
    first: bool,
    done: bool,
}

impl Combinations {
    /// Create r-length combinations from the pool.
    pub fn new(pool: Vec<Value>, r: usize) -> Self {
        let n = pool.len();
        if r > n {
            return Self {
                pool,
                indices: IndexVec::new(),
                r,
                first: true,
                done: true,
            };
        }

        let mut indices = IndexVec::with_capacity(r);
        for i in 0..r {
            indices.push(i);
        }

        Self {
            pool,
            indices,
            r,
            first: true,
            done: false,
        }
    }

    /// Get the current combination tuple.
    #[inline]
    fn current_tuple(&self) -> Vec<Value> {
        self.indices.iter().map(|&i| self.pool[i].clone()).collect()
    }
}

impl Iterator for Combinations {
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Vec<Value>> {
        if self.done {
            return None;
        }

        if self.first {
            self.first = false;
            if self.r == 0 {
                self.done = true;
                return Some(Vec::new()); // single empty combination
            }
            return Some(self.current_tuple());
        }

        let n = self.pool.len();

        // Find the rightmost index that can be incremented
        let mut i = self.r;
        loop {
            if i == 0 {
                self.done = true;
                return None;
            }
            i -= 1;
            if self.indices[i] != i + n - self.r {
                break;
            }
        }

        // Increment it and reset all indices to its right
        self.indices[i] += 1;
        for j in (i + 1)..self.r {
            self.indices[j] = self.indices[j - 1] + 1;
        }

        Some(self.current_tuple())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done { (0, Some(0)) } else { (0, None) }
    }
}

impl std::iter::FusedIterator for Combinations {}

// =============================================================================
// CombinationsWithReplacement
// =============================================================================

/// Successive r-length combinations with replacement (repetition allowed).
///
/// Equivalent to Python's `itertools.combinations_with_replacement(iterable, r)`.
///
/// # Algorithm
///
/// Uses rising index vector: indices satisfy `i_0 <= i_1 <= ... <= i_{r-1}`.
///
/// # Performance
///
/// - O(r) per `next()` for advance + tuple construction
/// - O(r) space for indices (SmallVec-backed)
#[derive(Debug, Clone)]
pub struct CombinationsWithReplacement {
    pool: Vec<Value>,
    indices: IndexVec,
    r: usize,
    first: bool,
    done: bool,
}

impl CombinationsWithReplacement {
    /// Create r-length combinations with replacement.
    pub fn new(pool: Vec<Value>, r: usize) -> Self {
        if pool.is_empty() && r > 0 {
            return Self {
                pool,
                indices: IndexVec::new(),
                r,
                first: true,
                done: true,
            };
        }

        let mut indices = IndexVec::with_capacity(r);
        indices.resize(r, 0); // all start at 0

        Self {
            pool,
            indices,
            r,
            first: true,
            done: false,
        }
    }

    /// Get the current tuple.
    #[inline]
    fn current_tuple(&self) -> Vec<Value> {
        self.indices.iter().map(|&i| self.pool[i].clone()).collect()
    }
}

impl Iterator for CombinationsWithReplacement {
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Vec<Value>> {
        if self.done {
            return None;
        }

        if self.first {
            self.first = false;
            if self.r == 0 {
                self.done = true;
                return Some(Vec::new());
            }
            return Some(self.current_tuple());
        }

        let n = self.pool.len();

        // Find the rightmost index that can be incremented
        let mut i = self.r;
        loop {
            if i == 0 {
                self.done = true;
                return None;
            }
            i -= 1;
            if self.indices[i] != n - 1 {
                break;
            }
        }

        // Increment it and set all following indices to the same value
        let new_val = self.indices[i] + 1;
        for j in i..self.r {
            self.indices[j] = new_val;
        }

        Some(self.current_tuple())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done { (0, Some(0)) } else { (0, None) }
    }
}

impl std::iter::FusedIterator for CombinationsWithReplacement {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
