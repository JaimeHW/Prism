//! Native `operator` module fast paths.
//!
//! CPython's `operator` module is mostly a thin callable facade over core VM
//! operations. Prism keeps those facades native so compatibility code does not
//! pay an import-time or dispatch tax for operations the interpreter already
//! implements on hot paths.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::comparison::{compare_order_result, contains_value};
use crate::ops::protocols::RichCompareOp;
use crate::truthiness::try_is_truthy;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use std::sync::{Arc, LazyLock};

static TRUTH_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.truth"), operator_truth));
static NOT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.not_"), operator_not));
static CONTAINS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("operator.contains"), operator_contains)
});
static LT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("operator.lt"), operator_lt));
static IS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("operator.is_"), operator_is));
static IS_NOT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("operator.is_not"), operator_is_not));

const EXPORTS: &[&str] = &["contains", "is_", "is_not", "lt", "not_", "truth"];

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
            "lt" => Ok(builtin_value(&LT_FUNCTION)),
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

fn operator_lt(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_arg_count("lt", args, 2)?;
    compare_order_result(vm, args[0], args[1], RichCompareOp::Lt)
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
