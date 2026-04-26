//! High-performance `reduce` implementation.
//!
//! Applies a two-argument function cumulatively to the items of an iterable,
//! from left to right, reducing the iterable to a single value.
//!
//! # Performance Characteristics
//!
//! | Operation | Time | Space |
//! |-----------|------|-------|
//! | `reduce(f, iter)` | O(n) | O(1) |
//! | `reduce(f, iter, init)` | O(n) | O(1) |
//!
//! Single pass, zero intermediate allocations.

use prism_core::Value;

// =============================================================================
// Reduce Function
// =============================================================================

/// Error type for reduce operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceError {
    /// Iterable was empty and no initializer was provided.
    EmptySequence,
    /// The function returned an error.
    FunctionError(String),
}

impl std::fmt::Display for ReduceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReduceError::EmptySequence => {
                write!(f, "reduce() of empty iterable with no initial value")
            }
            ReduceError::FunctionError(msg) => write!(f, "reduce() function error: {}", msg),
        }
    }
}

impl std::error::Error for ReduceError {}

/// Apply `function` cumulatively to items of `iterable`, reducing to a single value.
///
/// If `initializer` is present, it is placed before the items of the iterable
/// in the reduction, and serves as a default when the iterable is empty.
///
/// # Errors
///
/// Returns `ReduceError::EmptySequence` if the iterable is empty and no
/// initializer is provided.
///
/// # Performance
///
/// - Single pass over the iterable (O(n))
/// - No intermediate allocations
/// - Short-circuits on function error
#[inline]
pub fn reduce<F>(
    function: F,
    iterable: impl IntoIterator<Item = Value>,
    initializer: Option<Value>,
) -> Result<Value, ReduceError>
where
    F: Fn(&Value, &Value) -> Value,
{
    let mut iter = iterable.into_iter();

    // Get initial accumulator
    let mut accumulator = match initializer {
        Some(init) => init,
        None => iter.next().ok_or(ReduceError::EmptySequence)?,
    };

    // Apply function cumulatively
    for item in iter {
        accumulator = function(&accumulator, &item);
    }

    Ok(accumulator)
}

/// Apply `function` cumulatively with a fallible function.
///
/// Like `reduce`, but the function may return an error.
#[inline]
pub fn reduce_fallible<F, E>(
    function: F,
    iterable: impl IntoIterator<Item = Value>,
    initializer: Option<Value>,
) -> Result<Value, ReduceError>
where
    F: Fn(&Value, &Value) -> Result<Value, E>,
    E: std::fmt::Display,
{
    let mut iter = iterable.into_iter();

    let mut accumulator = match initializer {
        Some(init) => init,
        None => iter.next().ok_or(ReduceError::EmptySequence)?,
    };

    for item in iter {
        accumulator =
            function(&accumulator, &item).map_err(|e| ReduceError::FunctionError(e.to_string()))?;
    }

    Ok(accumulator)
}

/// Scan (prefix reduction) — returns all intermediate accumulator values.
///
/// This is `itertools.accumulate` in Python, but included here as it shares
/// the reduction kernel.
///
/// # Performance
///
/// - Single pass O(n), allocates result vector of size n
pub fn accumulate<F>(
    function: F,
    iterable: impl IntoIterator<Item = Value>,
    initializer: Option<Value>,
) -> Vec<Value>
where
    F: Fn(&Value, &Value) -> Value,
{
    let iter = iterable.into_iter();
    let (size_hint, _) = iter.size_hint();
    let mut results = Vec::with_capacity(size_hint.max(1));

    let mut iter = iter.peekable();

    let mut accumulator = match initializer {
        Some(init) => {
            results.push(init.clone());
            init
        }
        None => match iter.next() {
            Some(first) => {
                results.push(first.clone());
                first
            }
            None => return results,
        },
    };

    for item in iter {
        accumulator = function(&accumulator, &item);
        results.push(accumulator.clone());
    }

    results
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod reduce_tests;
