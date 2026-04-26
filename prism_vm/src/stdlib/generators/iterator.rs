//! Iterator protocol implementation for generators.
//!
//! This module provides the `__iter__` and `__next__` methods for generators,
//! enabling them to be used in for-loops and other iterator contexts.
//!
//! # Python Semantics
//!
//! ```python
//! def my_generator():
//!     yield 1
//!     yield 2
//!     yield 3
//!
//! gen = my_generator()
//! iter(gen) is gen  # True - generators are their own iterators
//!
//! next(gen)  # 1
//! next(gen)  # 2
//! next(gen)  # 3
//! next(gen)  # raises StopIteration
//! ```
//!
//! # Thread Safety
//!
//! Generators are NOT thread-safe. Concurrent access to a generator from
//! multiple threads leads to undefined behavior. The caller must ensure
//! exclusive access during iteration.

use prism_core::Value;

use super::object::GeneratorObject;
use super::state::GeneratorState;

// ============================================================================
// Iterator Result
// ============================================================================

/// Result of a generator iteration step.
#[derive(Debug, Clone, PartialEq)]
pub enum IterResult {
    /// Generator yielded a value.
    Yielded(Value),
    /// Generator returned a value (completed normally).
    Returned(Value),
    /// Generator raised an exception.
    Raised(GeneratorError),
}

impl IterResult {
    /// Returns true if the generator yielded a value.
    #[inline]
    pub fn is_yielded(&self) -> bool {
        matches!(self, Self::Yielded(_))
    }

    /// Returns true if the generator returned (completed).
    #[inline]
    pub fn is_returned(&self) -> bool {
        matches!(self, Self::Returned(_))
    }

    /// Returns true if the generator raised an exception.
    #[inline]
    pub fn is_raised(&self) -> bool {
        matches!(self, Self::Raised(_))
    }

    /// Extracts the yielded value, if any.
    #[inline]
    pub fn yielded_value(&self) -> Option<Value> {
        match self {
            Self::Yielded(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the returned value, if any.
    #[inline]
    pub fn returned_value(&self) -> Option<Value> {
        match self {
            Self::Returned(v) => Some(*v),
            _ => None,
        }
    }

    /// Converts to a Python-compatible result similar to StopIteration.value.
    #[inline]
    pub fn into_stop_iteration_value(self) -> Option<Value> {
        match self {
            Self::Returned(v) => Some(v),
            _ => None,
        }
    }
}

// ============================================================================
// Generator Error Types
// ============================================================================

/// Errors that can occur during generator execution.
#[derive(Debug, Clone, PartialEq)]
pub enum GeneratorError {
    /// Generator is already running (recursive call).
    AlreadyRunning,
    /// Generator is exhausted (already finished).
    Exhausted,
    /// Generator was never started but send() was called with non-None value.
    CantSendNonNone,
    /// An exception was thrown into the generator.
    ThrownException(GeneratorException),
    /// Generator raised StopIteration explicitly.
    StopIteration(Option<Value>),
    /// Runtime error during execution.
    RuntimeError(String),
}

impl GeneratorError {
    /// Creates a runtime error.
    pub fn runtime<S: Into<String>>(msg: S) -> Self {
        Self::RuntimeError(msg.into())
    }

    /// Returns true if the generator is simply exhausted.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        matches!(self, Self::Exhausted | Self::StopIteration(_))
    }

    /// Returns true if this is a fatal error.
    #[inline]
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::RuntimeError(_))
    }
}

impl std::fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRunning => write!(f, "generator already executing"),
            Self::Exhausted => write!(f, "StopIteration"),
            Self::CantSendNonNone => {
                write!(f, "can't send non-None value to a just-started generator")
            }
            Self::ThrownException(exc) => write!(f, "{}", exc),
            Self::StopIteration(None) => write!(f, "StopIteration"),
            Self::StopIteration(Some(_)) => write!(f, "StopIteration (with value)"),
            Self::RuntimeError(msg) => write!(f, "RuntimeError: {}", msg),
        }
    }
}

impl std::error::Error for GeneratorError {}

/// An exception thrown into a generator.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorException {
    /// Exception type name (e.g., "ValueError")
    pub type_name: String,
    /// Exception message.
    pub message: String,
    /// Optional exception value.
    pub value: Option<Value>,
}

impl GeneratorException {
    /// Creates a new exception.
    pub fn new<T: Into<String>, M: Into<String>>(type_name: T, message: M) -> Self {
        Self {
            type_name: type_name.into(),
            message: message.into(),
            value: None,
        }
    }

    /// Creates a new exception with a value.
    pub fn with_value<T: Into<String>, M: Into<String>>(
        type_name: T,
        message: M,
        value: Value,
    ) -> Self {
        Self {
            type_name: type_name.into(),
            message: message.into(),
            value: Some(value),
        }
    }
}

impl std::fmt::Display for GeneratorException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.type_name, self.message)
    }
}

// ============================================================================
// Generator Iterator
// ============================================================================

/// Iterator adapter for generators.
///
/// This wraps a `GeneratorObject` and provides a standard Rust iterator
/// interface for generators that only need simple iteration (no send/throw).
pub struct GeneratorIterator<'a> {
    /// The generator being iterated.
    generator: &'a mut GeneratorObject,
    /// Whether StopIteration has been raised.
    exhausted: bool,
}

impl<'a> GeneratorIterator<'a> {
    /// Creates a new iterator over a generator.
    #[inline]
    pub fn new(generator: &'a mut GeneratorObject) -> Self {
        Self {
            generator,
            exhausted: false,
        }
    }

    /// Returns a reference to the underlying generator.
    #[inline]
    pub fn generator(&self) -> &GeneratorObject {
        self.generator
    }

    /// Returns a mutable reference to the underlying generator.
    #[inline]
    pub fn generator_mut(&mut self) -> &mut GeneratorObject {
        self.generator
    }

    /// Checks if the generator is currently running.
    #[inline]
    pub fn is_running(&self) -> bool {
        self.generator.is_running()
    }

    /// Checks if the generator is exhausted.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.exhausted || self.generator.is_exhausted()
    }
}

// ============================================================================
// Iteration Helpers
// ============================================================================

/// Validates that a generator can be started or resumed.
///
/// Returns the previous state if the generator can be resumed,
/// or an appropriate error otherwise.
#[inline]
pub fn validate_resumable(generator: &GeneratorObject) -> Result<GeneratorState, GeneratorError> {
    match generator.state() {
        GeneratorState::Created => Ok(GeneratorState::Created),
        GeneratorState::Suspended => Ok(GeneratorState::Suspended),
        GeneratorState::Running => Err(GeneratorError::AlreadyRunning),
        GeneratorState::Exhausted => Err(GeneratorError::Exhausted),
    }
}

/// Validates that a send value is appropriate for the generator state.
///
/// You can only send a non-None value to a generator that has been started.
#[inline]
pub fn validate_send_value(
    generator: &GeneratorObject,
    value: Option<Value>,
) -> Result<(), GeneratorError> {
    match (generator.state(), value) {
        // Can always send None or to a suspended generator
        (GeneratorState::Suspended, _) => Ok(()),
        (GeneratorState::Created, None) => Ok(()),
        (GeneratorState::Created, Some(v)) if v.is_none() => Ok(()),
        // Cannot send non-None to a just-started generator
        (GeneratorState::Created, Some(_)) => Err(GeneratorError::CantSendNonNone),
        // Cannot send to running or exhausted generators
        (GeneratorState::Running, _) => Err(GeneratorError::AlreadyRunning),
        (GeneratorState::Exhausted, _) => Err(GeneratorError::Exhausted),
    }
}

/// Prepares a generator for the next iteration step.
///
/// This function:
/// 1. Validates that the generator can be resumed
/// 2. Optionally validates the send value
/// 3. Sets the send value if provided
/// 4. Starts the generator
///
/// Returns the previous state if successful.
pub fn prepare_iteration(
    generator: &mut GeneratorObject,
    send_value: Option<Value>,
) -> Result<GeneratorState, GeneratorError> {
    // Validate state
    let prev_state = validate_resumable(generator)?;

    // Validate and set send value
    if let Some(value) = send_value {
        validate_send_value(generator, Some(value))?;
        generator.set_send_value(value);
    }

    // Try to start the generator
    match generator.try_start() {
        Some(state) => Ok(state),
        None => {
            // This shouldn't happen if validation passed
            if generator.is_running() {
                Err(GeneratorError::AlreadyRunning)
            } else {
                Err(GeneratorError::Exhausted)
            }
        }
    }
}
