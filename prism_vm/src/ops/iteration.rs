//! Shared iteration helpers for VM-side iterator and generator driving.

use crate::VirtualMachine;
use crate::builtins::{iterator_to_value, value_to_iterator};
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::stdlib::generators::GeneratorObject;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::iter::IteratorObject;

/// Result of advancing an iterator-like object by one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum IterStep {
    /// The iterator produced a value.
    Yielded(Value),
    /// The iterator is exhausted.
    Exhausted,
}

/// Convert a value to an iterator object value following Python `iter()` rules.
pub(crate) fn ensure_iterator_value(value: Value) -> Result<Value, RuntimeError> {
    if let Some(ptr) = value.as_object_ptr() {
        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        if matches!(type_id, TypeId::ITERATOR | TypeId::GENERATOR) {
            return Ok(value);
        }
    }

    let iter =
        value_to_iterator(&value).map_err(|err| RuntimeError::type_error(err.to_string()))?;
    Ok(iterator_to_value(iter))
}

/// Advance a VM-visible iterator or generator by one step.
pub(crate) fn next_step(
    vm: &mut VirtualMachine,
    iterator: Value,
) -> Result<IterStep, RuntimeError> {
    let Some(ptr) = iterator.as_object_ptr() else {
        return Err(RuntimeError::type_error(format!(
            "'{}' object is not an iterator",
            iterator.type_name()
        )));
    };

    let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
    match type_id {
        TypeId::ITERATOR => {
            let iter = unsafe { &mut *(ptr as *mut IteratorObject) };
            Ok(match iter.next() {
                Some(value) => IterStep::Yielded(value),
                None => IterStep::Exhausted,
            })
        }
        TypeId::GENERATOR => {
            let generator = GeneratorObject::from_value_mut(iterator)
                .ok_or_else(|| RuntimeError::internal("invalid generator object"))?;
            match vm.resume_generator_for_send(generator, Value::none()) {
                Ok(crate::vm::GeneratorResumeOutcome::Yielded(value)) => {
                    Ok(IterStep::Yielded(value))
                }
                Ok(crate::vm::GeneratorResumeOutcome::Returned(_)) => Ok(IterStep::Exhausted),
                Err(err) if matches!(err.kind, RuntimeErrorKind::StopIteration) => {
                    Ok(IterStep::Exhausted)
                }
                Err(err) => Err(err),
            }
        }
        _ => Err(RuntimeError::type_error(format!(
            "'{}' object is not an iterator",
            type_id.name()
        ))),
    }
}
