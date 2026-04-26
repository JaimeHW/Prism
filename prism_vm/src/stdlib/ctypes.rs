//! Native `ctypes` compatibility facade.
//!
//! Prism does not embed CPython's libffi-backed `_ctypes` extension yet, but
//! CPython's regression suite uses a small `ctypes.pythonapi` surface to reach
//! C-API thread hooks. This module provides that ABI-shaped surface natively so
//! those calls participate in Prism's own thread and interpreter-lock runtime.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock};

static PY_OBJECT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("ctypes.py_object"), py_object));
static C_ULONG_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("ctypes.c_ulong"), c_ulong));
static C_LONG_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("ctypes.c_long"), c_long));
static C_INT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("ctypes.c_int"), c_int));
static C_VOID_P_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("ctypes.c_void_p"), c_ulong));

static PY_THREAD_STATE_SET_ASYNC_EXC_CALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("ctypes.pythonapi.PyThreadState_SetAsyncExc"),
        py_thread_state_set_async_exc,
    )
});
static PY_GIL_STATE_ENSURE_CALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("ctypes.pythonapi.PyGILState_Ensure"),
        py_gil_state_ensure,
    )
});
static PY_GIL_STATE_RELEASE_CALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("ctypes.pythonapi.PyGILState_Release"),
        py_gil_state_release,
    )
});

static PYTHONAPI: LazyLock<Value> = LazyLock::new(build_pythonapi);

/// Native `ctypes` module descriptor.
#[derive(Debug, Clone)]
pub struct CtypesModule {
    attrs: Vec<Arc<str>>,
}

impl CtypesModule {
    /// Create a new `ctypes` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("c_int"),
                Arc::from("c_long"),
                Arc::from("c_ulong"),
                Arc::from("c_void_p"),
                Arc::from("py_object"),
                Arc::from("pythonapi"),
            ],
        }
    }
}

impl Default for CtypesModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CtypesModule {
    fn name(&self) -> &str {
        "ctypes"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "c_int" => Ok(builtin_value(&C_INT_FUNCTION)),
            "c_long" => Ok(builtin_value(&C_LONG_FUNCTION)),
            "c_ulong" => Ok(builtin_value(&C_ULONG_FUNCTION)),
            "c_void_p" => Ok(builtin_value(&C_VOID_P_FUNCTION)),
            "py_object" => Ok(builtin_value(&PY_OBJECT_FUNCTION)),
            "pythonapi" => Ok(*PYTHONAPI),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'ctypes' has no attribute '{}'",
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

#[inline]
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn build_pythonapi() -> Value {
    let registry = shape_registry();
    let mut api = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
    for (name, callable) in pythonapi_symbols() {
        api.set_property(intern(name), api_function_value(name, callable), registry);
    }
    leak_object_value(api)
}

fn pythonapi_symbols() -> FxHashMap<&'static str, &'static BuiltinFunctionObject> {
    FxHashMap::from_iter([
        (
            "PyThreadState_SetAsyncExc",
            &*PY_THREAD_STATE_SET_ASYNC_EXC_CALL,
        ),
        ("PyGILState_Ensure", &*PY_GIL_STATE_ENSURE_CALL),
        ("PyGILState_Release", &*PY_GIL_STATE_RELEASE_CALL),
    ])
}

fn api_function_value(name: &str, callable: &'static BuiltinFunctionObject) -> Value {
    let registry = shape_registry();
    let mut function = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
    function.set_property(intern("__name__"), Value::string(intern(name)), registry);
    function.set_property(intern("__call__"), builtin_value(callable), registry);
    function.set_property(intern("argtypes"), Value::none(), registry);
    function.set_property(intern("restype"), Value::none(), registry);
    leak_object_value(function)
}

fn py_object(args: &[Value]) -> Result<Value, BuiltinError> {
    match args {
        [] => Ok(Value::none()),
        [value] => Ok(*value),
        _ => Err(BuiltinError::TypeError(format!(
            "py_object() takes at most 1 argument ({} given)",
            args.len()
        ))),
    }
}

fn c_ulong(args: &[Value]) -> Result<Value, BuiltinError> {
    let value = c_integer_arg(args, "c_ulong")?;
    if value < 0 {
        return Err(BuiltinError::OverflowError(
            "can't convert negative value to c_ulong".to_string(),
        ));
    }
    Ok(Value::int(value).expect("ctypes integer value should fit"))
}

fn c_long(args: &[Value]) -> Result<Value, BuiltinError> {
    let value = c_integer_arg(args, "c_long")?;
    Ok(Value::int(value).expect("ctypes integer value should fit"))
}

fn c_int(args: &[Value]) -> Result<Value, BuiltinError> {
    let value = c_integer_arg(args, "c_int")?;
    Ok(Value::int(value).expect("ctypes integer value should fit"))
}

fn c_integer_arg(args: &[Value], name: &str) -> Result<i64, BuiltinError> {
    match args {
        [] => Ok(0),
        [value] => value
            .as_int()
            .ok_or_else(|| BuiltinError::TypeError(format!("{name}() argument must be int"))),
        _ => Err(BuiltinError::TypeError(format!(
            "{name}() takes at most 1 argument ({} given)",
            args.len()
        ))),
    }
}

fn py_thread_state_set_async_exc(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "PyThreadState_SetAsyncExc() takes exactly 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let Some(ident) = args[1].as_int() else {
        return Err(BuiltinError::TypeError(
            "thread id must be an integer".to_string(),
        ));
    };
    if ident <= 0 {
        return Ok(Value::int(0).expect("zero fits in Value::int"));
    }

    let modified = super::_thread::set_pending_async_exception_for_ident(ident as u64, args[2]);
    Ok(Value::int(if modified { 1 } else { 0 }).expect("result fits in Value::int"))
}

fn py_gil_state_ensure(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "PyGILState_Ensure() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    Ok(Value::int(0).expect("CPython compatibility state token should fit"))
}

fn py_gil_state_release(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "PyGILState_Release() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    Ok(Value::none())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VirtualMachine;
    use crate::ops::calls::invoke_callable_value;
    use crate::ops::objects::{get_attribute_value, set_attribute_value};

    #[test]
    fn test_ctypes_exposes_pythonapi_thread_symbols() {
        let module = CtypesModule::new();
        let pythonapi = module
            .get_attr("pythonapi")
            .expect("pythonapi should be exposed");
        let mut vm = VirtualMachine::new();

        for name in [
            "PyThreadState_SetAsyncExc",
            "PyGILState_Ensure",
            "PyGILState_Release",
        ] {
            let symbol = get_attribute_value(&mut vm, pythonapi, &intern(name))
                .expect("pythonapi symbol should resolve");
            assert!(symbol.as_object_ptr().is_some());
        }
    }

    #[test]
    fn test_pythonapi_symbols_allow_ctypes_metadata_assignment() {
        let module = CtypesModule::new();
        let pythonapi = module.get_attr("pythonapi").unwrap();
        let mut vm = VirtualMachine::new();
        let symbol =
            get_attribute_value(&mut vm, pythonapi, &intern("PyThreadState_SetAsyncExc")).unwrap();

        set_attribute_value(&mut vm, symbol, &intern("argtypes"), Value::none())
            .expect("ctypes function metadata should be writable");
    }

    #[test]
    fn test_py_thread_state_set_async_exc_reports_unknown_thread() {
        let module = CtypesModule::new();
        let pythonapi = module.get_attr("pythonapi").unwrap();
        let mut vm = VirtualMachine::new();
        let symbol =
            get_attribute_value(&mut vm, pythonapi, &intern("PyThreadState_SetAsyncExc")).unwrap();

        let result = invoke_callable_value(
            &mut vm,
            symbol,
            &[
                Value::int(1_000_000_000).unwrap(),
                Value::string(intern("AsyncExc")),
            ],
        )
        .expect("unknown thread id should be a successful no-op");

        assert_eq!(result.as_int(), Some(0));
    }
}
