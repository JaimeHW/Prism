use crate::VirtualMachine;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::global_class_bitmap;
use prism_runtime::object::type_obj::TypeId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RichCompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl RichCompareOp {
    #[inline]
    fn left_method_name(self) -> &'static str {
        match self {
            Self::Eq => "__eq__",
            Self::Ne => "__ne__",
            Self::Lt => "__lt__",
            Self::Le => "__le__",
            Self::Gt => "__gt__",
            Self::Ge => "__ge__",
        }
    }

    #[inline]
    fn right_method_name(self) -> &'static str {
        match self {
            Self::Eq => "__eq__",
            Self::Ne => "__ne__",
            Self::Lt => "__gt__",
            Self::Le => "__ge__",
            Self::Gt => "__lt__",
            Self::Ge => "__le__",
        }
    }
}

#[inline]
pub(crate) fn invoke_bound_method_with_operand(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
    operand: Value,
) -> Result<Value, RuntimeError> {
    match target.implicit_self {
        Some(implicit_self) => {
            invoke_callable_value(vm, target.callable, &[implicit_self, operand])
        }
        None => invoke_callable_value(vm, target.callable, &[operand]),
    }
}

#[inline]
pub(crate) fn value_type_id(value: Value) -> TypeId {
    if let Some(ptr) = value.as_object_ptr() {
        return unsafe { (*(ptr as *const ObjectHeader)).type_id };
    }

    if value.is_none() {
        TypeId::NONE
    } else if value.is_bool() {
        TypeId::BOOL
    } else if value.is_int() {
        TypeId::INT
    } else if value.is_float() {
        TypeId::FLOAT
    } else if value.is_string() {
        TypeId::STR
    } else {
        TypeId::OBJECT
    }
}

#[inline]
fn is_runtime_subtype(actual: TypeId, target: TypeId) -> bool {
    if actual == target {
        return true;
    }
    if target == TypeId::OBJECT {
        return true;
    }
    if actual == TypeId::BOOL && target == TypeId::INT {
        return true;
    }
    if actual == TypeId::EXCEPTION_TYPE && target == TypeId::TYPE {
        return true;
    }

    global_class_bitmap(ClassId(actual.raw())).is_some_and(|bitmap| bitmap.is_subclass_of(target))
}

#[inline]
fn is_proper_runtime_subtype(actual: TypeId, target: TypeId) -> bool {
    actual != target && is_runtime_subtype(actual, target)
}

#[inline]
fn try_special_method_call(
    vm: &mut VirtualMachine,
    receiver: Value,
    method_name: &'static str,
    operand: Value,
) -> Result<Option<Value>, RuntimeError> {
    match resolve_special_method(receiver, method_name) {
        Ok(target) => {
            let result = invoke_bound_method_with_operand(vm, target, operand)?;
            if result == crate::builtins::builtin_not_implemented_value() {
                Ok(None)
            } else {
                Ok(Some(result))
            }
        }
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => Ok(None),
        Err(err) => Err(err),
    }
}

#[inline]
pub(crate) fn binary_special_method(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    left_method_name: &'static str,
    right_method_name: &'static str,
) -> Result<Option<Value>, RuntimeError> {
    let left_type = value_type_id(left);
    let right_type = value_type_id(right);

    if right_type != left_type && is_proper_runtime_subtype(right_type, left_type) {
        if let Some(result) = try_special_method_call(vm, right, right_method_name, left)? {
            return Ok(Some(result));
        }
        if let Some(result) = try_special_method_call(vm, left, left_method_name, right)? {
            return Ok(Some(result));
        }
        return Ok(None);
    }

    if let Some(result) = try_special_method_call(vm, left, left_method_name, right)? {
        return Ok(Some(result));
    }
    if right_type != left_type
        && let Some(result) = try_special_method_call(vm, right, right_method_name, left)?
    {
        return Ok(Some(result));
    }

    Ok(None)
}

#[inline]
pub(crate) fn inplace_special_method(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    method_name: &'static str,
) -> Result<Option<Value>, RuntimeError> {
    try_special_method_call(vm, left, method_name, right)
}

#[inline]
pub(crate) fn rich_compare_bool(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    op: RichCompareOp,
) -> Result<Option<bool>, RuntimeError> {
    let left_type = value_type_id(left);
    let right_type = value_type_id(right);
    let prefer_right = is_proper_runtime_subtype(right_type, left_type);

    let (
        first_receiver,
        first_method,
        first_operand,
        second_receiver,
        second_method,
        second_operand,
    ) = if prefer_right {
        (
            right,
            op.right_method_name(),
            left,
            left,
            op.left_method_name(),
            right,
        )
    } else {
        (
            left,
            op.left_method_name(),
            right,
            right,
            op.right_method_name(),
            left,
        )
    };

    if let Some(result) = try_special_method_call(vm, first_receiver, first_method, first_operand)?
    {
        return crate::truthiness::try_is_truthy(vm, result).map(Some);
    }
    if let Some(result) =
        try_special_method_call(vm, second_receiver, second_method, second_operand)?
    {
        return crate::truthiness::try_is_truthy(vm, result).map(Some);
    }

    Ok(None)
}
