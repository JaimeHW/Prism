//! Shared iteration helpers for VM-side iterator and generator driving.

use crate::VirtualMachine;
use crate::builtins::{iterator_to_value, value_to_iterator};
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::stdlib::generators::GeneratorObject;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::iter::IteratorObject;
use std::cell::RefCell;

/// Result of advancing an iterator-like object by one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum IterStep {
    /// The iterator produced a value.
    Yielded(Value),
    /// The iterator is exhausted.
    Exhausted,
}

/// Convert a value to an iterator object value following Python `iter()` rules.
pub(crate) fn ensure_iterator_value(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Value, RuntimeError> {
    if let Some(ptr) = value.as_object_ptr() {
        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        if matches!(type_id, TypeId::ITERATOR | TypeId::GENERATOR) {
            return Ok(value);
        }
    }

    if let Ok(iter) = value_to_iterator(&value) {
        return Ok(iterator_to_value(iter));
    }

    let bound = resolve_special_method(value, "__iter__").map_err(|_| {
        RuntimeError::type_error(format!("'{}' object is not iterable", value.type_name()))
    })?;
    let iterator = call_bound_method_target(vm, bound)?;

    if supports_next_protocol(iterator) {
        Ok(iterator)
    } else {
        Err(RuntimeError::type_error(
            "__iter__ returned non-iterator".to_string(),
        ))
    }
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
            let vm_cell = RefCell::new(vm);
            Ok(
                match iter.next_with(
                    &mut |callable, args| {
                        let vm = &mut *vm_cell.borrow_mut();
                        invoke_callable_value(vm, callable, args)
                    },
                    &mut |value| Ok(crate::truthiness::is_truthy(value)),
                    &mut |iterator| {
                        let vm = &mut *vm_cell.borrow_mut();
                        match next_step(vm, iterator)? {
                            IterStep::Yielded(value) => Ok(Some(value)),
                            IterStep::Exhausted => Ok(None),
                        }
                    },
                )? {
                    Some(value) => IterStep::Yielded(value),
                    None => IterStep::Exhausted,
                },
            )
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
        _ => {
            let bound = resolve_special_method(iterator, "__next__").map_err(|_| {
                RuntimeError::type_error(format!("'{}' object is not an iterator", type_id.name()))
            })?;
            match call_bound_method_target(vm, bound) {
                Ok(value) => Ok(IterStep::Yielded(value)),
                Err(err) if matches!(err.kind, RuntimeErrorKind::StopIteration) => {
                    Ok(IterStep::Exhausted)
                }
                Err(err) => Err(err),
            }
        }
    }
}

/// Collect all remaining values from an iterable using full VM iterator semantics.
pub(crate) fn collect_iterable_values(
    vm: &mut VirtualMachine,
    iterable: Value,
) -> Result<Vec<Value>, RuntimeError> {
    let iterator = ensure_iterator_value(vm, iterable)?;
    let mut values = Vec::new();

    loop {
        match next_step(vm, iterator)? {
            IterStep::Yielded(value) => values.push(value),
            IterStep::Exhausted => return Ok(values),
        }
    }
}

fn call_bound_method_target(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
) -> Result<Value, RuntimeError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
}

fn supports_next_protocol(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
    matches!(type_id, TypeId::ITERATOR | TypeId::GENERATOR)
        || resolve_special_method(value, "__next__").is_ok()
}
