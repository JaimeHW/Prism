//! Flyweight exception pool.
//!
//! This module provides pre-allocated singleton exceptions for common
//! control-flow exceptions like StopIteration and GeneratorExit.
//! This eliminates allocation for the most frequently thrown exceptions.
//!
//! # Performance Design
//!
//! - **Static singletons**: No allocation when raising StopIteration
//! - **Thread-safe access**: Uses static references
//! - **~10 cycles**: Throwing a flyweight exception is nearly free

use super::object::ExceptionObject;
use super::types::ExceptionTypeId;
use std::sync::{Arc, LazyLock};

// ============================================================================
// Static Flyweight Instances
// ============================================================================

/// Pre-allocated StopIteration exception.
///
/// Used when an iterator is exhausted. This is thrown millions of times
/// in typical Python code, so eliminating allocation is critical.
static STOP_ITERATION: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::StopIteration));

/// Pre-allocated StopAsyncIteration exception.
static STOP_ASYNC_ITERATION: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::StopAsyncIteration));

/// Pre-allocated GeneratorExit exception.
///
/// Raised when a generator's close() method is called.
static GENERATOR_EXIT: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::GeneratorExit));

/// Pre-allocated KeyboardInterrupt exception.
///
/// Raised when the user presses Ctrl+C.
static KEYBOARD_INTERRUPT: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::KeyboardInterrupt));

/// Pre-allocated MemoryError exception.
///
/// Used when memory allocation fails.
static MEMORY_ERROR: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::MemoryError));

/// Pre-allocated RecursionError exception.
///
/// Raised when maximum recursion depth is exceeded.
static RECURSION_ERROR: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::RecursionError));

// ============================================================================
// Flyweight Pool
// ============================================================================

/// Pool of pre-allocated flyweight exceptions.
///
/// Provides zero-allocation access to common control-flow exceptions.
pub struct FlyweightPool;

impl FlyweightPool {
    /// Returns a reference to the pre-allocated StopIteration exception.
    #[inline(always)]
    pub fn stop_iteration() -> &'static ExceptionObject {
        &STOP_ITERATION
    }

    /// Returns a reference to the pre-allocated StopAsyncIteration exception.
    #[inline(always)]
    pub fn stop_async_iteration() -> &'static ExceptionObject {
        &STOP_ASYNC_ITERATION
    }

    /// Returns a reference to the pre-allocated GeneratorExit exception.
    #[inline(always)]
    pub fn generator_exit() -> &'static ExceptionObject {
        &GENERATOR_EXIT
    }

    /// Returns a reference to the pre-allocated KeyboardInterrupt exception.
    #[inline(always)]
    pub fn keyboard_interrupt() -> &'static ExceptionObject {
        &KEYBOARD_INTERRUPT
    }

    /// Returns a reference to the pre-allocated MemoryError exception.
    #[inline(always)]
    pub fn memory_error() -> &'static ExceptionObject {
        &MEMORY_ERROR
    }

    /// Returns a reference to the pre-allocated RecursionError exception.
    #[inline(always)]
    pub fn recursion_error() -> &'static ExceptionObject {
        &RECURSION_ERROR
    }

    /// Returns a flyweight exception for the given type, if available.
    ///
    /// Returns None if no flyweight exists for this type.
    #[inline]
    pub fn get(type_id: ExceptionTypeId) -> Option<&'static ExceptionObject> {
        match type_id {
            ExceptionTypeId::StopIteration => Some(&STOP_ITERATION),
            ExceptionTypeId::StopAsyncIteration => Some(&STOP_ASYNC_ITERATION),
            ExceptionTypeId::GeneratorExit => Some(&GENERATOR_EXIT),
            ExceptionTypeId::KeyboardInterrupt => Some(&KEYBOARD_INTERRUPT),
            ExceptionTypeId::MemoryError => Some(&MEMORY_ERROR),
            ExceptionTypeId::RecursionError => Some(&RECURSION_ERROR),
            _ => None,
        }
    }

    /// Returns true if a flyweight exists for the given type.
    #[inline]
    pub const fn has_flyweight(type_id: ExceptionTypeId) -> bool {
        matches!(
            type_id,
            ExceptionTypeId::StopIteration
                | ExceptionTypeId::StopAsyncIteration
                | ExceptionTypeId::GeneratorExit
                | ExceptionTypeId::KeyboardInterrupt
                | ExceptionTypeId::MemoryError
                | ExceptionTypeId::RecursionError
        )
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Raises a StopIteration exception (zero-allocation).
#[inline(always)]
pub fn raise_stop_iteration() -> &'static ExceptionObject {
    FlyweightPool::stop_iteration()
}

/// Raises a GeneratorExit exception (zero-allocation).
#[inline(always)]
pub fn raise_generator_exit() -> &'static ExceptionObject {
    FlyweightPool::generator_exit()
}

/// Creates a StopIteration with a value.
///
/// Unlike the flyweight, this allocates because it needs args.
pub fn stop_iteration_with_value(value: prism_core::Value) -> Arc<ExceptionObject> {
    use super::object::ExceptionArgs;

    Arc::new(ExceptionObject::with_args(
        ExceptionTypeId::StopIteration,
        ExceptionArgs::single(value),
    ))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
