//! Minimal native `test.support` surface for CPython regression tests.
//!
//! CPython's `test.support` package imports a large portion of the stdlib at
//! module import time. Prism exposes the tiny support subset needed by early
//! compatibility targets natively, so individual regression modules can run
//! before the whole support stack is implemented.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::type_builtins::{SubclassBitmap, register_global_class};
use prism_runtime::object::type_obj::TypeId;
use std::sync::Arc;
use std::sync::LazyLock;

const TESTFN: &str = "@prism_test_tmp";

static ALWAYS_EQ_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_sentinel_class("_ALWAYS_EQ", &ALWAYS_EQ_METHODS));
static NEVER_EQ_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_sentinel_class("_NEVER_EQ", &NEVER_EQ_METHODS));
static ALWAYS_EQ_VALUE: LazyLock<Value> =
    LazyLock::new(|| sentinel_instance(ALWAYS_EQ_CLASS.as_ref()));
static NEVER_EQ_VALUE: LazyLock<Value> =
    LazyLock::new(|| sentinel_instance(NEVER_EQ_CLASS.as_ref()));

static ALWAYS_EQ_EQ_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support._ALWAYS_EQ.__eq__"), always_eq_eq)
});
static ALWAYS_EQ_NE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support._ALWAYS_EQ.__ne__"), always_eq_ne)
});
static NEVER_EQ_EQ_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support._NEVER_EQ.__eq__"), never_eq_eq)
});
static NEVER_EQ_NE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support._NEVER_EQ.__ne__"), never_eq_ne)
});
static NEVER_EQ_HASH_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support._NEVER_EQ.__hash__"), never_eq_hash)
});

const ALWAYS_EQ_METHODS: [(&str, &LazyLock<BuiltinFunctionObject>); 2] = [
    ("__eq__", &ALWAYS_EQ_EQ_FUNCTION),
    ("__ne__", &ALWAYS_EQ_NE_FUNCTION),
];
const NEVER_EQ_METHODS: [(&str, &LazyLock<BuiltinFunctionObject>); 3] = [
    ("__eq__", &NEVER_EQ_EQ_FUNCTION),
    ("__ne__", &NEVER_EQ_NE_FUNCTION),
    ("__hash__", &NEVER_EQ_HASH_FUNCTION),
];

/// Native `test.support` module descriptor.
#[derive(Debug, Clone)]
pub struct SupportModule {
    attrs: Vec<Arc<str>>,
}

impl SupportModule {
    /// Create a new `test.support` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("ALWAYS_EQ"),
                Arc::from("NEVER_EQ"),
                Arc::from("os_helper"),
            ],
        }
    }
}

impl Default for SupportModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SupportModule {
    fn name(&self) -> &str {
        "test.support"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "ALWAYS_EQ" => Ok(*ALWAYS_EQ_VALUE),
            "NEVER_EQ" => Ok(*NEVER_EQ_VALUE),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'test.support' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

fn build_sentinel_class(
    name: &'static str,
    methods: &'static [(&'static str, &'static LazyLock<BuiltinFunctionObject>)],
) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("test.support")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    for (method_name, method) in methods {
        class.set_attr(intern(method_name), builtin_value(&**method));
    }
    class.add_flags(
        ClassFlags::INITIALIZED
            | ClassFlags::NATIVE_HEAPTYPE
            | ClassFlags::HAS_EQ
            | ClassFlags::HASHABLE,
    );

    let class = Arc::new(class);
    let mut bitmap = SubclassBitmap::for_type(class.class_type_id());
    bitmap.set_bit(TypeId::OBJECT);
    register_global_class(Arc::clone(&class), bitmap);
    class
}

fn sentinel_instance(class: &PyClassObject) -> Value {
    crate::alloc_managed_value(allocate_heap_instance_for_class(class))
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn expect_binary_method(args: &[Value], name: &str) -> Result<(), BuiltinError> {
    if args.len() == 2 {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{name}() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )))
    }
}

fn always_eq_eq(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_binary_method(args, "__eq__")?;
    Ok(Value::bool(true))
}

fn always_eq_ne(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_binary_method(args, "__ne__")?;
    Ok(Value::bool(false))
}

fn never_eq_eq(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_binary_method(args, "__eq__")?;
    Ok(Value::bool(false))
}

fn never_eq_ne(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_binary_method(args, "__ne__")?;
    Ok(Value::bool(true))
}

fn never_eq_hash(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__hash__() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(Value::int(1).expect("constant hash fits in tagged int"))
}

/// Native `test.support.os_helper` module descriptor.
#[derive(Debug, Clone)]
pub struct OsHelperModule {
    attrs: Vec<Arc<str>>,
}

impl OsHelperModule {
    /// Create a new `test.support.os_helper` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("TESTFN")],
        }
    }
}

impl Default for OsHelperModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for OsHelperModule {
    fn name(&self) -> &str {
        "test.support.os_helper"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "TESTFN" => Ok(Value::string(intern(TESTFN))),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'test.support.os_helper' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}
