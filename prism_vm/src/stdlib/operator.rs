//! Native `operator` module fast paths.
//!
//! CPython's `operator` module is mostly a thin callable facade over core VM
//! operations. Prism keeps those facades native so compatibility code does not
//! pay an import-time or dispatch tax for operations the interpreter already
//! implements on hot paths.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, builtin_not_implemented_value,
    runtime_error_to_builtin_error, try_len_value,
};
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::comparison::{compare_order_result, contains_value, eq_result, ne_result};
use crate::ops::iteration::{IterStep, ensure_iterator_value, next_step};
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::ops::objects::extract_type_id;
use crate::ops::protocols::RichCompareOp;
use crate::stdlib::exceptions::ExceptionTypeId;
use crate::truthiness::try_is_truthy;
use num_bigint::{BigInt, Sign};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::list::ListObject;
use std::sync::{Arc, LazyLock};

static TRUTH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.truth"), operator_truth));
static NOT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.not_"), operator_not));
static CONTAINS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("operator.contains"), operator_contains)
});
static COUNT_OF_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("operator.countOf"), operator_count_of)
});
static INDEX_OF_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("operator.indexOf"), operator_index_of)
});
static LENGTH_HINT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("operator.length_hint"), operator_length_hint)
});
static LT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.lt"), operator_lt));
static LE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.le"), operator_le));
static EQ_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.eq"), operator_eq));
static NE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.ne"), operator_ne));
static GT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.gt"), operator_gt));
static GE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.ge"), operator_ge));
static IS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("operator.is_"), operator_is));
static IS_NOT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("operator.is_not"), operator_is_not));

const EXPORTS: &[&str] = &[
    "contains",
    "countOf",
    "eq",
    "ge",
    "gt",
    "indexOf",
    "is_",
    "is_not",
    "le",
    "length_hint",
    "lt",
    "ne",
    "not_",
    "truth",
];

/// Native `operator` module descriptor.
#[derive(Debug, Clone)]
pub struct OperatorModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl OperatorModule {
    /// Create a native `operator` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS
                .iter()
                .copied()
                .chain(["__all__"])
                .map(Arc::from)
                .collect(),
            all: string_list_value(EXPORTS),
        }
    }
}

impl Default for OperatorModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for OperatorModule {
    fn name(&self) -> &str {
        "operator"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "truth" => Ok(builtin_value(&TRUTH_FUNCTION)),
            "not_" => Ok(builtin_value(&NOT_FUNCTION)),
            "contains" => Ok(builtin_value(&CONTAINS_FUNCTION)),
            "countOf" => Ok(builtin_value(&COUNT_OF_FUNCTION)),
            "indexOf" => Ok(builtin_value(&INDEX_OF_FUNCTION)),
            "length_hint" => Ok(builtin_value(&LENGTH_HINT_FUNCTION)),
            "eq" => Ok(builtin_value(&EQ_FUNCTION)),
            "ne" => Ok(builtin_value(&NE_FUNCTION)),
            "lt" => Ok(builtin_value(&LT_FUNCTION)),
            "le" => Ok(builtin_value(&LE_FUNCTION)),
            "gt" => Ok(builtin_value(&GT_FUNCTION)),
            "ge" => Ok(builtin_value(&GE_FUNCTION)),
            "is_" => Ok(builtin_value(&IS_FUNCTION)),
            "is_not" => Ok(builtin_value(&IS_NOT_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'operator' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}

#[inline]
fn expect_arg_count(function: &str, args: &[Value], expected: usize) -> Result<(), BuiltinError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{function}() takes exactly {expected} arguments ({} given)",
            args.len()
        )))
    }
}

fn operator_truth(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("truth", args, 1)?;
    try_is_truthy(vm, args[0])
        .map(Value::bool)
        .map_err(BuiltinError::Raised)
}

fn operator_not(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("not_", args, 1)?;
    try_is_truthy(vm, args[0])
        .map(|truthy| Value::bool(!truthy))
        .map_err(BuiltinError::Raised)
}

fn operator_contains(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("contains", args, 2)?;
    contains_value(vm, args[1], args[0])
        .map(Value::bool)
        .map_err(BuiltinError::Raised)
}

fn operator_count_of(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("countOf", args, 2)?;
    let iterator = ensure_iterator_value(vm, args[0]).map_err(runtime_error_to_builtin_error)?;
    let mut count = 0_i64;

    loop {
        match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
            IterStep::Yielded(item) => {
                if eq_result(vm, item, args[1]).map_err(runtime_error_to_builtin_error)? {
                    count = count.checked_add(1).ok_or_else(|| {
                        BuiltinError::OverflowError("countOf result is too large".to_string())
                    })?;
                }
            }
            IterStep::Exhausted => {
                return Ok(Value::int(count).expect("non-negative count should fit"));
            }
        }
    }
}

fn operator_index_of(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("indexOf", args, 2)?;
    let iterator = ensure_iterator_value(vm, args[0]).map_err(runtime_error_to_builtin_error)?;
    let mut index = 0_i64;

    loop {
        match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
            IterStep::Yielded(item) => {
                if eq_result(vm, item, args[1]).map_err(runtime_error_to_builtin_error)? {
                    return Ok(Value::int(index).expect("non-negative index should fit"));
                }
                index = index.checked_add(1).ok_or_else(|| {
                    BuiltinError::OverflowError("indexOf index is too large".to_string())
                })?;
            }
            IterStep::Exhausted => {
                return Err(BuiltinError::ValueError(
                    "sequence.index(x): x not in sequence".to_string(),
                ));
            }
        }
    }
}

fn operator_length_hint(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "length_hint expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let default = match args.get(1).copied() {
        Some(value) => parse_default_hint(value)?,
        None => BigInt::from(0),
    };

    match try_len_value(vm, args[0]) {
        Ok(len) => return ssize_len_to_value(len),
        Err(err) if is_type_error(&err) => {}
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    }

    let target = match resolve_special_method(args[0], "__length_hint__") {
        Ok(target) => target,
        Err(err) if err.is_attribute_error() => return Ok(bigint_to_value(default)),
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };

    let hint = match invoke_zero_arg_bound_method(vm, target) {
        Ok(value) => value,
        Err(err) if is_type_error(&err) => return Ok(bigint_to_value(default)),
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };

    if hint == builtin_not_implemented_value() {
        return Ok(bigint_to_value(default));
    }

    normalize_length_hint_result(hint)
}

#[inline]
fn invoke_zero_arg_bound_method(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
) -> Result<Value, RuntimeError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
}

#[inline]
fn parse_default_hint(value: Value) -> Result<BigInt, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(BigInt::from(usize::from(flag)));
    }

    let Some(default) = value_to_bigint(value) else {
        return Err(BuiltinError::TypeError(format!(
            "'{}' object cannot be interpreted as an integer",
            value_type_name(value)
        )));
    };

    if default < BigInt::from(isize::MIN) || default > BigInt::from(isize::MAX) {
        return Err(BuiltinError::OverflowError(
            "Python int too large to convert to C ssize_t".to_string(),
        ));
    }

    Ok(default)
}

#[inline]
fn normalize_length_hint_result(value: Value) -> Result<Value, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(bigint_to_value(BigInt::from(usize::from(flag))));
    }

    let Some(length) = value_to_bigint(value) else {
        return Err(BuiltinError::TypeError(format!(
            "__length_hint__ must be an integer, not {}",
            value_type_name(value)
        )));
    };

    if length.sign() == Sign::Minus {
        return Err(BuiltinError::ValueError(
            "__length_hint__() should return >= 0".to_string(),
        ));
    }
    if length > BigInt::from(isize::MAX) {
        return Err(BuiltinError::OverflowError(
            "cannot fit 'int' into an index-sized integer".to_string(),
        ));
    }

    Ok(bigint_to_value(length))
}

#[inline]
fn ssize_len_to_value(len: usize) -> Result<Value, BuiltinError> {
    if len > isize::MAX as usize {
        return Err(BuiltinError::OverflowError(
            "cannot fit 'int' into an index-sized integer".to_string(),
        ));
    }
    Ok(bigint_to_value(BigInt::from(len)))
}

#[inline]
fn is_type_error(err: &RuntimeError) -> bool {
    match err.kind() {
        RuntimeErrorKind::TypeError { .. }
        | RuntimeErrorKind::UnsupportedOperandTypes { .. }
        | RuntimeErrorKind::NotCallable { .. }
        | RuntimeErrorKind::NotIterable { .. }
        | RuntimeErrorKind::NotSubscriptable { .. } => true,
        RuntimeErrorKind::Exception { type_id, .. } => {
            *type_id == ExceptionTypeId::TypeError.as_u8() as u16
        }
        _ => false,
    }
}

#[inline]
fn value_type_name(value: Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.as_bool().is_some() {
        "bool"
    } else if value.as_int().is_some() {
        "int"
    } else if value.as_float().is_some() {
        "float"
    } else if value.is_string() {
        "str"
    } else if let Some(ptr) = value.as_object_ptr() {
        extract_type_id(ptr).name()
    } else {
        "object"
    }
}

fn operator_lt(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("lt", args, 2)?;
    compare_order_result(vm, args[0], args[1], RichCompareOp::Lt)
        .map(Value::bool)
        .map_err(BuiltinError::Raised)
}

fn operator_le(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("le", args, 2)?;
    compare_order_result(vm, args[0], args[1], RichCompareOp::Le)
        .map(Value::bool)
        .map_err(BuiltinError::Raised)
}

fn operator_eq(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("eq", args, 2)?;
    eq_result(vm, args[0], args[1])
        .map(Value::bool)
        .map_err(BuiltinError::Raised)
}

fn operator_ne(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("ne", args, 2)?;
    ne_result(vm, args[0], args[1])
        .map(Value::bool)
        .map_err(BuiltinError::Raised)
}

fn operator_gt(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("gt", args, 2)?;
    compare_order_result(vm, args[0], args[1], RichCompareOp::Gt)
        .map(Value::bool)
        .map_err(BuiltinError::Raised)
}

fn operator_ge(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("ge", args, 2)?;
    compare_order_result(vm, args[0], args[1], RichCompareOp::Ge)
        .map(Value::bool)
        .map_err(BuiltinError::Raised)
}

fn operator_is(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("is_", args, 2)?;
    Ok(Value::bool(args[0].raw_bits() == args[1].raw_bits()))
}

fn operator_is_not(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("is_not", args, 2)?;
    Ok(Value::bool(args[0].raw_bits() != args[1].raw_bits()))
}
