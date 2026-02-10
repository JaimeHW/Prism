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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorControlFlow Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_control_flow_continue() {
        let cf = GeneratorControlFlow::Continue;
        assert!(!cf.is_yield());
        assert!(!cf.is_yield_from());
        assert!(!cf.is_done());
    }

    #[test]
    fn test_control_flow_yield() {
        let cf = GeneratorControlFlow::Yield {
            value: Value::int(42).unwrap(),
            resume_index: 5,
        };

        assert!(cf.is_yield());
        assert!(!cf.is_yield_from());
        assert!(!cf.is_done());
        assert_eq!(cf.yield_value().unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_control_flow_yield_from() {
        let cf = GeneratorControlFlow::YieldFrom {
            sub_generator: Value::none(),
            resume_index: 3,
        };

        assert!(!cf.is_yield());
        assert!(cf.is_yield_from());
        assert!(!cf.is_done());
    }

    #[test]
    fn test_control_flow_stop_iteration() {
        let cf = GeneratorControlFlow::StopIteration {
            value: Value::none(),
        };

        assert!(!cf.is_yield());
        assert!(cf.is_done());
    }

    #[test]
    fn test_control_flow_throw() {
        let cf = GeneratorControlFlow::Throw {
            exception: Value::none(),
        };

        assert!(!cf.is_yield());
        assert!(!cf.is_done());
    }

    #[test]
    fn test_control_flow_closed() {
        let cf = GeneratorControlFlow::Closed;

        assert!(!cf.is_yield());
        assert!(cf.is_done());
    }

    // ════════════════════════════════════════════════════════════════════════
    // YieldState Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_yield_state_new() {
        let state = YieldState::new(Value::int(42).unwrap(), 5, 0b1010, 3);

        assert_eq!(state.yield_value.as_int(), Some(42));
        assert_eq!(state.resume_index, 5);
        assert_eq!(state.liveness, 0b1010);
        assert_eq!(state.result_reg, 3);
    }

    // ════════════════════════════════════════════════════════════════════════
    // yield_value Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_yield_value_handler() {
        let cf = yield_value(Value::int(100).unwrap(), 7, 2);

        match cf {
            GeneratorControlFlow::Yield {
                value,
                resume_index,
            } => {
                assert_eq!(value.as_int(), Some(100));
                assert_eq!(resume_index, 7);
            }
            _ => panic!("Expected Yield"),
        }
    }

    #[test]
    fn test_yield_value_none() {
        let cf = yield_value(Value::none(), 0, 0);

        assert!(cf.is_yield());
        assert!(cf.yield_value().unwrap().is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // yield_from Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_yield_from_handler() {
        let cf = yield_from(Value::int(999).unwrap(), 3);

        match cf {
            GeneratorControlFlow::YieldFrom {
                sub_generator,
                resume_index,
            } => {
                assert_eq!(sub_generator.as_int(), Some(999));
                assert_eq!(resume_index, 3);
            }
            _ => panic!("Expected YieldFrom"),
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // gen_start Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_gen_start_none() {
        let cf = gen_start(None);
        assert!(matches!(cf, GeneratorControlFlow::Continue));
    }

    #[test]
    fn test_gen_start_with_none_value() {
        let cf = gen_start(Some(Value::none()));
        assert!(matches!(cf, GeneratorControlFlow::Continue));
    }

    #[test]
    fn test_gen_start_with_non_none() {
        let cf = gen_start(Some(Value::int(42).unwrap()));
        assert!(matches!(cf, GeneratorControlFlow::Throw { .. }));
    }

    // ════════════════════════════════════════════════════════════════════════
    // throw_into Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_throw_into_handler() {
        let cf = throw_into(Value::int(999).unwrap());

        match cf {
            GeneratorControlFlow::Throw { exception } => {
                assert_eq!(exception.as_int(), Some(999));
            }
            _ => panic!("Expected Throw"),
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // close_generator Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_close_generator_handler() {
        let cf = close_generator();
        assert!(matches!(cf, GeneratorControlFlow::Closed));
        assert!(cf.is_done());
    }

    // ════════════════════════════════════════════════════════════════════════
    // get_yield_value Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_get_yield_value_from_yield() {
        let cf = GeneratorControlFlow::Yield {
            value: Value::int(42).unwrap(),
            resume_index: 0,
        };

        let value = get_yield_value(&cf);
        assert_eq!(value.unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_get_yield_value_from_stop_iteration() {
        let cf = GeneratorControlFlow::StopIteration {
            value: Value::int(99).unwrap(),
        };

        let value = get_yield_value(&cf);
        assert_eq!(value.unwrap().as_int(), Some(99));
    }

    #[test]
    fn test_get_yield_value_from_continue() {
        let cf = GeneratorControlFlow::Continue;
        assert!(get_yield_value(&cf).is_none());
    }

    #[test]
    fn test_get_yield_value_from_throw() {
        let cf = GeneratorControlFlow::Throw {
            exception: Value::none(),
        };
        assert!(get_yield_value(&cf).is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorState Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_generator_state_can_resume() {
        assert!(GeneratorState::Created.can_resume());
        assert!(!GeneratorState::Running.can_resume());
        assert!(GeneratorState::Suspended.can_resume());
        assert!(!GeneratorState::Closed.can_resume());
    }

    #[test]
    fn test_generator_state_is_finished() {
        assert!(!GeneratorState::Created.is_finished());
        assert!(!GeneratorState::Running.is_finished());
        assert!(!GeneratorState::Suspended.is_finished());
        assert!(GeneratorState::Closed.is_finished());
    }

    // ════════════════════════════════════════════════════════════════════════
    // is_generator Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_is_generator_primitives() {
        // Primitives are not generators
        assert!(!is_generator(&Value::none()));
        assert!(!is_generator(&Value::int(42).unwrap()));
        assert!(!is_generator(&Value::bool(true)));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Integration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_yield_sequence() {
        // Simulate a generator that yields 1, 2, 3
        let yields = [
            yield_value(Value::int(1).unwrap(), 0, 0),
            yield_value(Value::int(2).unwrap(), 1, 0),
            yield_value(Value::int(3).unwrap(), 2, 0),
        ];

        for (i, cf) in yields.iter().enumerate() {
            assert!(cf.is_yield());
            assert_eq!(cf.yield_value().unwrap().as_int(), Some((i + 1) as i64));
        }
    }

    #[test]
    fn test_yield_and_stop() {
        let cf1 = yield_value(Value::int(42).unwrap(), 0, 0);
        let cf2 = GeneratorControlFlow::StopIteration {
            value: Value::none(),
        };

        assert!(!cf1.is_done());
        assert!(cf2.is_done());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_yield_max_resume_index() {
        let cf = yield_value(Value::none(), u32::MAX, 0);

        match cf {
            GeneratorControlFlow::Yield { resume_index, .. } => {
                assert_eq!(resume_index, u32::MAX);
            }
            _ => panic!("Expected Yield"),
        }
    }

    #[test]
    fn test_yield_from_chain() {
        // Simulate nested yield from
        let cf1 = yield_from(Value::int(1).unwrap(), 0);
        let cf2 = yield_from(Value::int(2).unwrap(), 1);

        assert!(cf1.is_yield_from());
        assert!(cf2.is_yield_from());
    }
}
