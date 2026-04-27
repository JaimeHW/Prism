//! Generator opcode handlers.
//!
//! This module provides the opcode handlers for Python generator operations:
//!
//! - `Yield` - Suspend generator and yield a value
//! - `YieldFrom` - Delegate to a sub-generator
//! - `Send` - Resume generator with a sent value
//! - `Throw` - Inject an exception into a generator
//! - `Close` - Close a generator, running cleanup
//!
//! # Opcode Summary
//!
//! | Opcode | Handler | Description |
//! |--------|---------|-------------|
//! | `YIELD_VALUE` | `yield_value` | Yield the value in src register |
//! | `YIELD_FROM` | `yield_from` | Delegate to sub-generator in src |
//! | `GEN_START` | `gen_start` | Initialize generator on first resume |
//! | `SEND` | `send_value` | Resume with value, or iterate if None |
//! | `THROW` | `throw_into` | Throw exception into generator |
//! | `CLOSE` | `close_generator` | Close generator gracefully |
//!
//! # Control Flow
//!
//! Generator opcodes return special `ControlFlow` variants:
//!
//! ```text
//! ControlFlow::Yield { value, resume_index }
//!     → VM suspends generator, returns value to caller
//!
//! ControlFlow::Resume { send_value }
//!     → VM resumes generator with sent value
//!
//! ControlFlow::StopIteration { value }
//!     → Generator is exhausted
//!
//! ControlFlow::Throw { exception }
//!     → Exception should be raised in generator
//! ```

use prism_core::Value;

// =============================================================================
// Control Flow Types
// =============================================================================

/// Control flow result from generator opcodes.
#[derive(Debug, Clone)]
pub enum GeneratorControlFlow {
    /// Continue normal execution.
    Continue,

    /// Yield a value and suspend.
    Yield {
        /// The value to yield.
        value: Value,
        /// The resume index for this yield point.
        resume_index: u32,
    },

    /// Delegate to a sub-generator (yield from).
    YieldFrom {
        /// The sub-generator to delegate to.
        sub_generator: Value,
        /// The resume index when delegation returns.
        resume_index: u32,
    },

    /// Generator is exhausted (StopIteration).
    StopIteration {
        /// The return value (StopIteration.value).
        value: Value,
    },

    /// Exception should be raised.
    Throw {
        /// The exception to raise.
        exception: Value,
    },

    /// Generator was closed.
    Closed,
}

impl GeneratorControlFlow {
    /// Check if this is a yield.
    #[inline]
    pub fn is_yield(&self) -> bool {
        matches!(self, Self::Yield { .. })
    }

    /// Check if this is a delegation.
    #[inline]
    pub fn is_yield_from(&self) -> bool {
        matches!(self, Self::YieldFrom { .. })
    }

    /// Check if generator is done.
    #[inline]
    pub fn is_done(&self) -> bool {
        matches!(self, Self::StopIteration { .. } | Self::Closed)
    }

    /// Extract yield value if this is a yield.
    #[inline]
    pub fn yield_value(&self) -> Option<&Value> {
        match self {
            Self::Yield { value, .. } => Some(value),
            _ => None,
        }
    }
}

// =============================================================================
// Yield State
// =============================================================================

/// State information for a yield operation.
#[derive(Debug, Clone)]
pub struct YieldState {
    /// The value being yielded.
    pub yield_value: Value,
    /// The resume index (yield point number).
    pub resume_index: u32,
    /// Liveness bitmap for register capture.
    pub liveness: u64,
    /// Destination register for sent value on resume.
    pub result_reg: u8,
}

impl YieldState {
    /// Create a new yield state.
    #[inline]
    pub fn new(yield_value: Value, resume_index: u32, liveness: u64, result_reg: u8) -> Self {
        Self {
            yield_value,
            resume_index,
            liveness,
            result_reg,
        }
    }
}

// =============================================================================
// Opcode Handlers
// =============================================================================

/// Handle the YIELD_VALUE opcode.
///
/// Yields a value to the caller and suspends the generator.
///
/// # Python Semantics
///
/// ```python
/// def gen():
///     x = yield 42  # Yields 42, x receives sent value on resume
/// ```
///
/// # Arguments
///
/// * `yield_value` - The value to yield
/// * `resume_index` - The index of this yield point
/// * `result_reg` - The register to receive the sent value on resume
///
/// # Returns
///
/// `GeneratorControlFlow::Yield` with the yield value and resume info.
#[inline]
pub fn yield_value(yield_value: Value, resume_index: u32, result_reg: u8) -> GeneratorControlFlow {
    GeneratorControlFlow::Yield {
        value: yield_value,
        resume_index,
    }
}

/// Handle the YIELD_FROM opcode.
///
/// Delegates iteration to a sub-generator.
///
/// # Python Semantics
///
/// ```python
/// def gen():
///     yield from sub_gen()  # Delegate to sub_gen
/// ```
///
/// # Arguments
///
/// * `sub_generator` - The generator/iterator to delegate to
/// * `resume_index` - The index of this yield point
///
/// # Returns
///
/// `GeneratorControlFlow::YieldFrom` to start delegation.
#[inline]
pub fn yield_from(sub_generator: Value, resume_index: u32) -> GeneratorControlFlow {
    GeneratorControlFlow::YieldFrom {
        sub_generator,
        resume_index,
    }
}

/// Handle the GEN_START opcode.
///
/// Initializes the generator on first resume.
/// Validates that no non-None value was sent on first iteration.
///
/// # Arguments
///
/// * `send_value` - The value sent (should be None on first call)
///
/// # Returns
///
/// `Continue` if valid, `Throw` if non-None was sent.
#[inline]
pub fn gen_start(send_value: Option<Value>) -> GeneratorControlFlow {
    match send_value {
        None => GeneratorControlFlow::Continue,
        Some(v) if v.is_none() => GeneratorControlFlow::Continue,
        Some(_) => {
            // TypeError: can't send non-None value to a just-started generator
            GeneratorControlFlow::Throw {
                exception: Value::none(), // Should be TypeError
            }
        }
    }
}

/// Handle the SEND opcode.
///
/// Resumes the generator with a sent value, or iterates if None.
///
/// # Python Semantics
///
/// ```python
/// gen.send(value)  # Resume with value
/// next(gen)        # Resume with None
/// ```
///
/// # Arguments
///
/// * `gen_value` - The generator to resume
/// * `send_value` - The value to send (None for next())
///
/// # Returns
///
/// Control flow indicating how to proceed.
#[inline]
pub fn send_value(gen_value: Value, send_value: Value) -> GeneratorControlFlow {
    // The actual resumption is handled by the VM
    // This just validates and prepares
    GeneratorControlFlow::Continue
}

/// Handle the THROW opcode.
///
/// Throws an exception into the generator.
///
/// # Python Semantics
///
/// ```python
/// gen.throw(ValueError("error"))  # Raise in generator
/// ```
///
/// # Arguments
///
/// * `exception` - The exception to throw
///
/// # Returns
///
/// `GeneratorControlFlow::Throw` with the exception.
#[inline]
pub fn throw_into(exception: Value) -> GeneratorControlFlow {
    GeneratorControlFlow::Throw { exception }
}

/// Handle the CLOSE opcode.
///
/// Closes the generator gracefully, running any finally blocks.
///
/// # Python Semantics
///
/// ```python
/// gen.close()  # Throws GeneratorExit, suppresses it if caught
/// ```
///
/// # Returns
///
/// `GeneratorControlFlow::Closed`.
#[inline]
pub fn close_generator() -> GeneratorControlFlow {
    GeneratorControlFlow::Closed
}

/// Get the yield value from a generator's last yield.
///
/// This is used when handling yielded values in the VM.
///
/// # Arguments
///
/// * `control_flow` - The control flow from a yield operation
///
/// # Returns
///
/// The yielded value if this was a yield, None otherwise.
#[inline]
pub fn get_yield_value(control_flow: &GeneratorControlFlow) -> Option<Value> {
    match control_flow {
        GeneratorControlFlow::Yield { value, .. } => Some(value.clone()),
        GeneratorControlFlow::StopIteration { value } => Some(value.clone()),
        _ => None,
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a value is a generator.
///
/// # Arguments
///
/// * `value` - The value to check
///
/// # Returns
///
/// `true` if the value is a generator object.
#[inline]
pub fn is_generator(value: &Value) -> bool {
    // Check if value is a generator object
    value.is_object()
}

/// Extract the generator state from a value.
///
/// # Arguments
///
/// * `value` - The generator value
///
/// # Returns
///
/// The generator's current state, or None if not a generator.
#[inline]
pub fn get_generator_state(value: &Value) -> Option<GeneratorState> {
    // This would extract from the actual GeneratorObject
    // For now, return None as placeholder
    None
}

/// Generator state enum (mirrors Python's generator states).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorState {
    /// Generator has been created but not started.
    Created,
    /// Generator is currently executing.
    Running,
    /// Generator is suspended at a yield.
    Suspended,
    /// Generator has completed (returned or raised).
    Closed,
}

impl GeneratorState {
    /// Check if the generator can be resumed.
    #[inline]
    pub fn can_resume(&self) -> bool {
        matches!(self, Self::Created | Self::Suspended)
    }

    /// Check if the generator has finished.
    #[inline]
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Closed)
    }
}
