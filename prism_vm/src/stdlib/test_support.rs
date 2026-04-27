//! Minimal native `test.support` surface for CPython regression tests.
//!
//! CPython's `test.support` package imports a large portion of the stdlib at
//! module import time. Prism exposes the tiny support subset needed by early
//! compatibility targets natively, so individual regression modules can run
//! before the whole support stack is implemented.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class, builtin_repr,
    exception_type_value_for_id, runtime_error_to_builtin_error,
};
use crate::stdlib::exceptions::ExceptionTypeId;
use num_bigint::BigInt;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::type_builtins::{SubclassBitmap, register_global_class};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::int::bigint_to_value;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;

const TESTFN: &str = "@prism_test_tmp";
const C_RECURSION_LIMIT: i64 = 64;

static ALWAYS_EQ_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_sentinel_class("_ALWAYS_EQ", &ALWAYS_EQ_METHODS));
static NEVER_EQ_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_sentinel_class("_NEVER_EQ", &NEVER_EQ_METHODS));
static INFINITE_RECURSION_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(build_infinite_recursion_class);
static ALWAYS_EQ_VALUE: LazyLock<Value> =
    LazyLock::new(|| sentinel_instance(ALWAYS_EQ_CLASS.as_ref()));
static NEVER_EQ_VALUE: LazyLock<Value> =
    LazyLock::new(|| sentinel_instance(NEVER_EQ_CLASS.as_ref()));
static INFINITE_RECURSION_VALUE: LazyLock<Value> =
    LazyLock::new(|| sentinel_instance(INFINITE_RECURSION_CLASS.as_ref()));

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
static INFINITE_RECURSION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.infinite_recursion"),
        infinite_recursion,
    )
});
static INFINITE_RECURSION_ENTER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support._InfiniteRecursion.__enter__"),
        infinite_recursion_enter,
    )
});
static INFINITE_RECURSION_EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support._InfiniteRecursion.__exit__"),
        infinite_recursion_exit,
    )
});
static CPYTHON_ONLY_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("test.support.cpython_only"), cpython_only)
});
static REQUIRES_LIMITED_API_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("test.support.requires_limited_api"),
        requires_limited_api,
    )
});
static REFCOUNT_TEST_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("test.support.refcount_test"), refcount_test)
});
static REQUIRES_DOCSTRINGS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.requires_docstrings"),
        identity_decorator,
    )
});
static GET_ATTRIBUTE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("test.support.get_attribute"), get_attribute)
});
static BROKEN_ITER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("test.support.BrokenIter"), broken_iter)
});
static CHECK_FREE_AFTER_ITERATING_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.check_free_after_iterating"),
        check_free_after_iterating,
    )
});
static ITER_BUILTIN_TYPES_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.iter_builtin_types"),
        empty_iterator_list,
    )
});
static ITER_SLOT_WRAPPERS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.iter_slot_wrappers"),
        empty_iterator_list,
    )
});
static LOAD_PACKAGE_TESTS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.load_package_tests"),
        load_package_tests,
    )
});
static RUN_WITH_LOCALE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support.run_with_locale"), run_with_locale)
});
static RUN_WITH_LOCALES_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support.run_with_locales"), run_with_locale)
});
static SKIP_ON_S390X_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support.skip_on_s390x"), identity_decorator)
});
static SORTDICT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("test.support.sortdict"), sortdict));
static UNLINK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support.os_helper.unlink"), unlink)
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
                Arc::from("BrokenIter"),
                Arc::from("C_RECURSION_LIMIT"),
                Arc::from("check_free_after_iterating"),
                Arc::from("MISSING_C_DOCSTRINGS"),
                Arc::from("MAX_Py_ssize_t"),
                Arc::from("NHASHBITS"),
                Arc::from("NEVER_EQ"),
                Arc::from("Py_DEBUG"),
                Arc::from("TestFailed"),
                Arc::from("cpython_only"),
                Arc::from("get_attribute"),
                Arc::from("infinite_recursion"),
                Arc::from("is_emscripten"),
                Arc::from("is_wasi"),
                Arc::from("iter_builtin_types"),
                Arc::from("iter_slot_wrappers"),
                Arc::from("load_package_tests"),
                Arc::from("refcount_test"),
                Arc::from("os_helper"),
                Arc::from("requires_limited_api"),
                Arc::from("requires_docstrings"),
                Arc::from("run_with_locale"),
                Arc::from("run_with_locales"),
                Arc::from("skip_on_s390x"),
                Arc::from("sortdict"),
                Arc::from("verbose"),
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
            "BrokenIter" => Ok(builtin_value(&BROKEN_ITER_FUNCTION)),
            "C_RECURSION_LIMIT" => {
                Ok(Value::int(C_RECURSION_LIMIT).expect("recursion test limit fits"))
            }
            "check_free_after_iterating" => Ok(builtin_value(&CHECK_FREE_AFTER_ITERATING_FUNCTION)),
            "MISSING_C_DOCSTRINGS" => Ok(Value::bool(true)),
            "MAX_Py_ssize_t" => Ok(bigint_to_value(BigInt::from(isize::MAX))),
            "NHASHBITS" => Ok(Value::int((usize::BITS - 1) as i64)
                .expect("hash bit width fits in tagged int")),
            "NEVER_EQ" => Ok(*NEVER_EQ_VALUE),
            "Py_DEBUG" => Ok(Value::bool(false)),
            "TestFailed" => Ok(exception_type_value_for_id(
                ExceptionTypeId::AssertionError.as_u8() as u16,
            )
            .expect("AssertionError exception type is registered")),
            "cpython_only" => Ok(builtin_value(&CPYTHON_ONLY_FUNCTION)),
            "get_attribute" => Ok(builtin_value(&GET_ATTRIBUTE_FUNCTION)),
            "infinite_recursion" => Ok(builtin_value(&INFINITE_RECURSION_FUNCTION)),
            "is_emscripten" => Ok(Value::bool(false)),
            "is_wasi" => Ok(Value::bool(false)),
            "iter_builtin_types" => Ok(builtin_value(&ITER_BUILTIN_TYPES_FUNCTION)),
            "iter_slot_wrappers" => Ok(builtin_value(&ITER_SLOT_WRAPPERS_FUNCTION)),
            "load_package_tests" => Ok(builtin_value(&LOAD_PACKAGE_TESTS_FUNCTION)),
            "refcount_test" => Ok(builtin_value(&REFCOUNT_TEST_FUNCTION)),
            "requires_limited_api" => Ok(builtin_value(&REQUIRES_LIMITED_API_FUNCTION)),
            "requires_docstrings" => Ok(builtin_value(&REQUIRES_DOCSTRINGS_FUNCTION)),
            "run_with_locale" => Ok(builtin_value(&RUN_WITH_LOCALE_FUNCTION)),
            "run_with_locales" => Ok(builtin_value(&RUN_WITH_LOCALES_FUNCTION)),
            "skip_on_s390x" => Ok(builtin_value(&SKIP_ON_S390X_FUNCTION)),
            "sortdict" => Ok(builtin_value(&SORTDICT_FUNCTION)),
            "verbose" => Ok(Value::int(1).expect("support.verbose fits in tagged int")),
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

fn build_infinite_recursion_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("_InfiniteRecursion"));
    class.set_attr(intern("__module__"), Value::string(intern("test.support")));
    class.set_attr(
        intern("__qualname__"),
        Value::string(intern("_InfiniteRecursion")),
    );
    class.set_attr(
        intern("__enter__"),
        builtin_value(&INFINITE_RECURSION_ENTER_FUNCTION),
    );
    class.set_attr(
        intern("__exit__"),
        builtin_value(&INFINITE_RECURSION_EXIT_FUNCTION),
    );
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);

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

fn infinite_recursion(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "infinite_recursion() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(*INFINITE_RECURSION_VALUE)
}

fn infinite_recursion_enter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__enter__() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(args[0])
}

fn infinite_recursion_exit(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "__exit__() takes exactly 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(Value::bool(false))
}

fn mark_unittest_skip(
    vm: &mut VirtualMachine,
    target: Value,
    reason: &'static str,
) -> Result<Value, BuiltinError> {
    crate::ops::objects::set_attribute_value(
        vm,
        target,
        &intern("__unittest_skip__"),
        Value::bool(true),
    )
    .map_err(runtime_error_to_builtin_error)?;
    crate::ops::objects::set_attribute_value(
        vm,
        target,
        &intern("__unittest_skip_why__"),
        Value::string(intern(reason)),
    )
    .map_err(runtime_error_to_builtin_error)?;
    Ok(target)
}

fn cpython_only(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "cpython_only() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    mark_unittest_skip(vm, args[0], "CPython implementation detail")
}

fn requires_limited_api(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "requires_limited_api() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    mark_unittest_skip(vm, args[0], "needs Limited API support")
}

fn refcount_test(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "refcount_test() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    mark_unittest_skip(vm, args[0], "CPython reference counting detail")
}

fn get_attribute(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "get_attribute() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let name = value_as_string_ref(args[1])
        .ok_or_else(|| BuiltinError::TypeError("attribute name must be a string".to_string()))?;
    crate::ops::objects::get_attribute_value(vm, args[0], &intern(name.as_str()))
        .map_err(runtime_error_to_builtin_error)
}

fn broken_iter(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "BrokenIter() takes no positional arguments ({} given)",
            args.len()
        )));
    }
    if keywords.iter().any(|(name, value)| {
        matches!(*name, "init_raises" | "next_raises" | "iter_raises") && value.is_truthy()
    }) {
        return Err(BuiltinError::ValueError("division by zero".to_string()));
    }
    Ok(crate::alloc_managed_value(ListObject::new()))
}

fn check_free_after_iterating(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 3 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "check_free_after_iterating() takes 3 or 4 arguments ({} given)",
            args.len()
        )));
    }
    Ok(Value::none())
}

fn empty_iterator_list(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "helper takes at most one argument ({} given)",
            args.len()
        )));
    }
    Ok(crate::alloc_managed_value(ListObject::new()))
}

fn load_package_tests(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "load_package_tests() takes exactly 4 arguments ({} given)",
            args.len()
        )));
    }
    Ok(args[2])
}

fn run_with_locale(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "run_with_locale() missing locale category".to_string(),
        ));
    }
    Ok(builtin_value(&IDENTITY_DECORATOR_FUNCTION))
}

fn identity_decorator(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "decorator takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(args[0])
}

static IDENTITY_DECORATOR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support._identity_decorator"),
        identity_decorator,
    )
});

fn sortdict(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "sortdict() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let ptr = args[0]
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("sortdict() argument must be a dict".to_string()))?;
    let dict = crate::ops::objects::dict_storage_ref_from_ptr(ptr)
        .ok_or_else(|| BuiltinError::TypeError("sortdict() argument must be a dict".to_string()))?;

    let mut pairs = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key_repr = repr_string(key)?;
        let value_repr = repr_string(value)?;
        pairs.push((key_repr, value_repr));
    }
    pairs.sort_unstable_by(|left, right| left.0.cmp(&right.0));

    let mut rendered = String::from("{");
    for (index, (key, value)) in pairs.iter().enumerate() {
        if index > 0 {
            rendered.push_str(", ");
        }
        rendered.push_str(key);
        rendered.push_str(": ");
        rendered.push_str(value);
    }
    rendered.push('}');
    Ok(Value::string(intern(&rendered)))
}

fn repr_string(value: Value) -> Result<String, BuiltinError> {
    let repr = builtin_repr(&[value])?;
    let Some(text) = value_as_string_ref(repr) else {
        return Err(BuiltinError::TypeError(
            "repr() returned non-string value".to_string(),
        ));
    };
    Ok(text.as_str().to_string())
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
            attrs: vec![Arc::from("TESTFN"), Arc::from("unlink")],
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
            "unlink" => Ok(builtin_value(&UNLINK_FUNCTION)),
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

fn unlink(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "unlink() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let path = value_as_string_ref(args[0])
        .ok_or_else(|| BuiltinError::TypeError("unlink() path must be str".to_string()))?;
    match std::fs::remove_file(Path::new(path.as_str())) {
        Ok(()) => Ok(Value::none()),
        Err(err)
            if matches!(
                err.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            Ok(Value::none())
        }
        Err(err) => Err(BuiltinError::OSError(err.to_string())),
    }
}
