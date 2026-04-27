//! Generator method implementations (send, throw, close).
//!
//! This module provides the coroutine protocol methods:
//!
//! - `send(value)` - Send a value into the generator
//! - `throw(type, value, traceback)` - Throw an exception into the generator
//! - `close()` - Close the generator by throwing GeneratorExit
//!
//! # Python Semantics
//!
//! ```python
//! def coroutine():
//!     received = yield 1
//!     if received:
//!         yield received * 2
//!     else:
//!         yield 0
//!
//! gen = coroutine()
//! next(gen)       # Start: yields 1
//! gen.send(10)    # Sends 10, yields 20
//! gen.close()     # Throws GeneratorExit, generator stops
//! ```

use prism_core::Value;

use super::iterator::{GeneratorError, GeneratorException};
use super::object::GeneratorObject;
use super::state::GeneratorState;
use super::storage::LivenessMap;

// ============================================================================
// Send Protocol
// ============================================================================

/// Result of a send operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SendResult {
    /// Generator yielded a value.
    Yielded(Value),
    /// Generator returned (completed).
    Returned(Value),
    /// Operation failed.
    Error(GeneratorError),
}

impl SendResult {
    /// Returns the yielded value if present.
    #[inline]
    pub fn yielded(&self) -> Option<Value> {
        match self {
            Self::Yielded(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the returned value if present.
    #[inline]
    pub fn returned(&self) -> Option<Value> {
        match self {
            Self::Returned(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns true if the generator yielded.
    #[inline]
    pub fn is_yielded(&self) -> bool {
        matches!(self, Self::Yielded(_))
    }

    /// Returns true if the generator returned.
    #[inline]
    pub fn is_done(&self) -> bool {
        matches!(self, Self::Returned(_) | Self::Error(_))
    }
}

/// Prepares a generator for a send operation.
///
/// This validates the generator state and sets the send value.
/// Returns the previous state if successful.
pub fn prepare_send(
    generator: &mut GeneratorObject,
    value: Value,
) -> Result<GeneratorState, GeneratorError> {
    match generator.state() {
        GeneratorState::Created => {
            // Can only send None to a just-created generator
            if !value.is_none() {
                return Err(GeneratorError::CantSendNonNone);
            }
            generator.set_send_value(value);
            generator.try_start().ok_or(GeneratorError::Exhausted)
        }
        GeneratorState::Suspended => {
            // Can send any value to a suspended generator
            generator.set_send_value(value);
            generator.try_start().ok_or(GeneratorError::Exhausted)
        }
        GeneratorState::Running => Err(GeneratorError::AlreadyRunning),
        GeneratorState::Exhausted => Err(GeneratorError::Exhausted),
    }
}

/// Completes a send operation after the generator has yielded.
///
/// Call this after the VM executes the generator and it yields.
pub fn complete_send_yielded(generator: &mut GeneratorObject, yielded_value: Value) -> SendResult {
    // The generator should now be suspended by the VM
    // We return the yielded value
    SendResult::Yielded(yielded_value)
}

/// Completes a send operation after the generator has returned.
///
/// Call this after the VM executes the generator and it returns.
pub fn complete_send_returned(generator: &mut GeneratorObject, return_value: Value) -> SendResult {
    generator.exhaust();
    SendResult::Returned(return_value)
}

// ============================================================================
// Throw Protocol
// ============================================================================

/// Result of a throw operation.
#[derive(Debug, Clone, PartialEq)]
pub enum ThrowResult {
    /// Generator caught the exception and yielded.
    Yielded(Value),
    /// Generator caught the exception and returned.
    Returned(Value),
    /// Exception propagated out of the generator.
    Propagated(GeneratorException),
    /// Operation failed (generator not in valid state).
    Error(GeneratorError),
}

impl ThrowResult {
    /// Returns true if the exception was caught and the generator yielded.
    #[inline]
    pub fn is_yielded(&self) -> bool {
        matches!(self, Self::Yielded(_))
    }

    /// Returns true if the generator is now exhausted.
    #[inline]
    pub fn is_done(&self) -> bool {
        matches!(
            self,
            Self::Returned(_) | Self::Propagated(_) | Self::Error(_)
        )
    }
}

/// Prepares a generator for a throw operation.
///
/// This sets up the exception to be thrown when the generator resumes.
pub fn prepare_throw(
    generator: &mut GeneratorObject,
    exception: GeneratorException,
) -> Result<GeneratorState, GeneratorError> {
    match generator.state() {
        GeneratorState::Created => {
            // Throwing into a never-started generator exhausts it
            generator.try_start();
            generator.exhaust();
            Err(GeneratorError::ThrownException(exception))
        }
        GeneratorState::Suspended => {
            // Throw exception into suspended generator
            generator.try_start().ok_or(GeneratorError::Exhausted)
        }
        GeneratorState::Running => Err(GeneratorError::AlreadyRunning),
        GeneratorState::Exhausted => Err(GeneratorError::Exhausted),
    }
}

/// Completes a throw operation after the generator has handled it.
pub fn complete_throw_yielded(
    generator: &mut GeneratorObject,
    yielded_value: Value,
) -> ThrowResult {
    ThrowResult::Yielded(yielded_value)
}

/// Completes a throw operation after the generator has returned.
pub fn complete_throw_returned(
    generator: &mut GeneratorObject,
    return_value: Value,
) -> ThrowResult {
    generator.exhaust();
    ThrowResult::Returned(return_value)
}

/// Completes a throw operation when the exception propagated.
pub fn complete_throw_propagated(
    generator: &mut GeneratorObject,
    exception: GeneratorException,
) -> ThrowResult {
    generator.exhaust();
    ThrowResult::Propagated(exception)
}

// ============================================================================
// Close Protocol
// ============================================================================

/// Result of a close operation.
#[derive(Debug, Clone, PartialEq)]
pub enum CloseResult {
    /// Generator closed successfully.
    Closed,
    /// Generator raised an exception other than GeneratorExit.
    RuntimeError(GeneratorException),
    /// Generator yielded a value (which is an error).
    YieldedInFinally(Value),
}

impl CloseResult {
    /// Returns true if the generator closed successfully.
    #[inline]
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Closed)
    }

    /// Returns true if there was an error.
    #[inline]
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }
}

/// Prepares a generator for a close operation.
///
/// This throws GeneratorExit into the generator.
pub fn prepare_close(generator: &mut GeneratorObject) -> CloseResult {
    match generator.state() {
        GeneratorState::Created => {
            // Never-started generator can just be marked exhausted
            generator.try_start();
            generator.exhaust();
            CloseResult::Closed
        }
        GeneratorState::Exhausted => {
            // Already closed
            CloseResult::Closed
        }
        GeneratorState::Running => {
            // Can't close a running generator
            CloseResult::RuntimeError(GeneratorException::new(
                "ValueError",
                "generator already executing",
            ))
        }
        GeneratorState::Suspended => {
            // Need to throw GeneratorExit
            if generator.try_start().is_none() {
                CloseResult::Closed
            } else {
                // VM will handle throwing GeneratorExit
                CloseResult::Closed
            }
        }
    }
}

/// Completes a close operation after the generator has stopped.
pub fn complete_close_caught(generator: &mut GeneratorObject) -> CloseResult {
    generator.exhaust();
    CloseResult::Closed
}

/// Completes a close operation if the generator yielded (error case).
pub fn complete_close_yielded(generator: &mut GeneratorObject, value: Value) -> CloseResult {
    generator.exhaust();
    CloseResult::YieldedInFinally(value)
}

/// Completes a close operation if an exception propagated.
pub fn complete_close_exception(
    generator: &mut GeneratorObject,
    exception: GeneratorException,
) -> CloseResult {
    generator.exhaust();

    // GeneratorExit is expected and counts as success
    if exception.type_name == "GeneratorExit" {
        CloseResult::Closed
    } else {
        CloseResult::RuntimeError(exception)
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Creates a GeneratorExit exception for closing generators.
#[inline]
pub fn generator_exit() -> GeneratorException {
    GeneratorException::new("GeneratorExit", "generator exit")
}

/// Creates a StopIteration exception.
#[inline]
pub fn stop_iteration(value: Option<Value>) -> GeneratorException {
    match value {
        Some(v) => GeneratorException::with_value("StopIteration", "", v),
        None => GeneratorException::new("StopIteration", ""),
    }
}
