//! Shared iteration helpers for VM-side iterator and generator driving.

use crate::VirtualMachine;
use crate::builtins::{iterator_to_value, try_length_hint, value_to_iterator};
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::ops::protocols::invoke_bound_method_with_operand;
use crate::stdlib::exceptions::ExceptionTypeId;
use crate::stdlib::generators::GeneratorObject;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::iter::{IteratorAdvanceError, IteratorObject};
use std::cell::RefCell;

impl From<IteratorAdvanceError> for RuntimeError {
    #[inline]
    fn from(err: IteratorAdvanceError) -> Self {
        RuntimeError::exception(ExceptionTypeId::RuntimeError.as_u8() as u16, err.message())
    }
}

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

    let iterator = match resolve_special_method(value, "__iter__") {
        Ok(bound) => call_bound_method_target(vm, bound)?,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return iterator_from_sequence_getitem(vm, value)?.ok_or_else(|| {
                RuntimeError::type_error(format!("'{}' object is not iterable", value.type_name()))
            });
        }
        Err(_) => {
            return Err(RuntimeError::type_error(format!(
                "'{}' object is not iterable",
                value.type_name()
            )));
        }
    };

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
                Err(err)
                    if runtime_error_matches_exception(
                        vm,
                        &err,
                        ExceptionTypeId::StopIteration,
                    ) =>
                {
                    Ok(IterStep::Exhausted)
                }
                Err(err) => Err(err),
            }
        }
        _ => {
            let bound = resolve_special_method(iterator, "__next__").map_err(|_| {
                RuntimeError::type_error(format!("'{}' object is not an iterator", type_id.name()))
            })?;
            let caller_exception_context = vm.capture_exception_context();
            match call_bound_method_target(vm, bound) {
                Ok(value) => Ok(IterStep::Yielded(value)),
                Err(err)
                    if runtime_error_matches_exception(
                        vm,
                        &err,
                        ExceptionTypeId::StopIteration,
                    ) =>
                {
                    vm.restore_exception_context(caller_exception_context);
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
    let capacity = try_length_hint(vm, iterable, 0)?;
    let iterator = ensure_iterator_value(vm, iterable)?;
    let mut values = Vec::new();
    values
        .try_reserve(capacity)
        .map_err(|_| RuntimeError::memory_error("length hint is too large"))?;

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

fn iterator_from_sequence_getitem(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Option<Value>, RuntimeError> {
    let bound = match resolve_special_method(value, "__getitem__") {
        Ok(bound) => bound,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => return Ok(None),
        Err(err) => return Err(err),
    };

    if bound.callable.is_none() {
        return Ok(None);
    }

    let mut values = Vec::new();
    let mut index = 0_i64;
    let caller_exception_context = vm.capture_exception_context();

    loop {
        let index_value = Value::int(index)
            .ok_or_else(|| RuntimeError::value_error("sequence index overflow"))?;
        match invoke_bound_method_with_operand(vm, bound, index_value) {
            Ok(item) => values.push(item),
            Err(err) if runtime_error_matches_exception(vm, &err, ExceptionTypeId::IndexError) => {
                vm.restore_exception_context(caller_exception_context);
                break;
            }
            Err(err)
                if runtime_error_matches_exception(vm, &err, ExceptionTypeId::StopIteration) =>
            {
                vm.restore_exception_context(caller_exception_context);
                break;
            }
            Err(err) => return Err(err),
        }
        index = index
            .checked_add(1)
            .ok_or_else(|| RuntimeError::value_error("sequence index overflow"))?;
    }

    Ok(Some(iterator_to_value(IteratorObject::from_values(values))))
}

fn supports_next_protocol(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
    matches!(type_id, TypeId::ITERATOR | TypeId::GENERATOR)
        || resolve_special_method(value, "__next__").is_ok()
}

#[inline]
fn runtime_error_matches_exception(
    vm: &VirtualMachine,
    err: &RuntimeError,
    expected: ExceptionTypeId,
) -> bool {
    match &err.kind {
        RuntimeErrorKind::IndexError { .. } => {
            expected == ExceptionTypeId::IndexError
                || ExceptionTypeId::IndexError.is_subclass_of(expected)
        }
        RuntimeErrorKind::StopIteration => {
            expected == ExceptionTypeId::StopIteration
                || ExceptionTypeId::StopIteration.is_subclass_of(expected)
        }
        RuntimeErrorKind::Exception { type_id, .. } => ExceptionTypeId::from_u8(*type_id as u8)
            .is_some_and(|actual| actual.is_subclass_of(expected)),
        RuntimeErrorKind::ControlTransferred => vm
            .get_active_exception_type_id()
            .and_then(|type_id| u8::try_from(type_id).ok())
            .and_then(ExceptionTypeId::from_u8)
            .is_some_and(|actual| actual.is_subclass_of(expected)),
        _ => false,
    }
}
