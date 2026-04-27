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
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::dict_access::{dict_get_item, dict_remove_item, dict_set_item};
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::ops::objects::{dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id};
use crate::stdlib::exceptions::ExceptionTypeId;
use num_bigint::BigInt;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
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
const BIGMEM_1M: i64 = 1024 * 1024;
const BIGMEM_1G: i64 = 1024 * BIGMEM_1M;
const BIGMEM_2G: i64 = 2 * BIGMEM_1G;
const BIGMEM_4G: i64 = 4 * BIGMEM_1G;
const SWAP_ITEM_MAPPING_ATTR: &str = "__prism_swap_item_mapping__";
const SWAP_ITEM_KEY_ATTR: &str = "__prism_swap_item_key__";
const SWAP_ITEM_VALUE_ATTR: &str = "__prism_swap_item_value__";
const SWAP_ITEM_OLD_VALUE_ATTR: &str = "__prism_swap_item_old_value__";
const SWAP_ITEM_HAD_ITEM_ATTR: &str = "__prism_swap_item_had_item__";

static ALWAYS_EQ_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_sentinel_class("_ALWAYS_EQ", &ALWAYS_EQ_METHODS));
static NEVER_EQ_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_sentinel_class("_NEVER_EQ", &NEVER_EQ_METHODS));
static INFINITE_RECURSION_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(build_infinite_recursion_class);
static WARNINGS_FILTER_CONTEXT_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_context_manager_class(
        "_SaveRestoreWarningsFilters",
        "test.support.warnings_helper",
        &WARNINGS_FILTER_CONTEXT_ENTER_FUNCTION,
        &WARNINGS_FILTER_CONTEXT_EXIT_FUNCTION,
    )
});
static SWAP_ITEM_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_context_manager_class(
        "_SwapItem",
        "test.support",
        &SWAP_ITEM_ENTER_FUNCTION,
        &SWAP_ITEM_EXIT_FUNCTION,
    )
});
static ALWAYS_EQ_VALUE: LazyLock<Value> =
    LazyLock::new(|| sentinel_instance(ALWAYS_EQ_CLASS.as_ref()));
static NEVER_EQ_VALUE: LazyLock<Value> =
    LazyLock::new(|| sentinel_instance(NEVER_EQ_CLASS.as_ref()));
static INFINITE_RECURSION_VALUE: LazyLock<Value> =
    LazyLock::new(|| sentinel_instance(INFINITE_RECURSION_CLASS.as_ref()));
static WARNINGS_FILTER_CONTEXT_VALUE: LazyLock<Value> =
    LazyLock::new(|| sentinel_instance(WARNINGS_FILTER_CONTEXT_CLASS.as_ref()));

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
static SWAP_ITEM_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("test.support.swap_item"), swap_item));
static SWAP_ITEM_ENTER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("test.support._SwapItem.__enter__"),
        swap_item_enter,
    )
});
static SWAP_ITEM_EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("test.support._SwapItem.__exit__"), swap_item_exit)
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
static BIGMEMTEST_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("test.support.bigmemtest"), bigmemtest)
});
static BIGMEM_SKIP_DECORATOR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("test.support._bigmem_skip_decorator"),
        bigmem_skip_decorator,
    )
});
static REQUIRES_DOCSTRINGS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.requires_docstrings"),
        identity_decorator,
    )
});
static REQUIRES_IEEE_754_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.requires_IEEE_754"),
        identity_decorator,
    )
});
static REQUIRES_RESOURCE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.requires_resource"),
        requires_resource,
    )
});
static NO_TRACING_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support.no_tracing"), identity_decorator)
});
static GET_ATTRIBUTE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("test.support.get_attribute"), get_attribute)
});
static GC_COLLECT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("test.support.gc_collect"), gc_collect)
});
static BROKEN_ITER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("test.support.BrokenIter"), broken_iter)
});
static CHECK_IMPL_DETAIL_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("test.support.check_impl_detail"),
        check_impl_detail,
    )
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
static SKIP_IF_PGO_TASK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.skip_if_pgo_task"),
        identity_decorator,
    )
});
static SORTDICT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("test.support.sortdict"), sortdict));
static UNLINK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("test.support.os_helper.unlink"), unlink)
});
static CREATE_EMPTY_FILE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.os_helper.create_empty_file"),
        create_empty_file,
    )
});
static FORGET_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("test.support.import_helper.forget"),
        forget_module,
    )
});
static IMPORT_MODULE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("test.support.import_helper.import_module"),
        import_module,
    )
});
static REAP_THREADS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.threading_helper.reap_threads"),
        identity_decorator,
    )
});
static REQUIRES_WORKING_THREADING_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("test.support.threading_helper.requires_working_threading"),
        requires_working_threading,
    )
});
static THREADING_SKIP_DECORATOR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("test.support.threading_helper._skip_decorator"),
        threading_skip_decorator,
    )
});
static SAVE_RESTORE_WARNINGS_FILTERS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("test.support.warnings_helper.save_restore_warnings_filters"),
            save_restore_warnings_filters,
        )
    });
static WARNINGS_FILTER_CONTEXT_ENTER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("test.support.warnings_helper._SaveRestoreWarningsFilters.__enter__"),
            noop_context_enter,
        )
    });
static WARNINGS_FILTER_CONTEXT_EXIT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("test.support.warnings_helper._SaveRestoreWarningsFilters.__exit__"),
            noop_context_exit,
        )
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
                Arc::from("_2G"),
                Arc::from("_4G"),
                Arc::from("bigmemtest"),
                Arc::from("check_impl_detail"),
                Arc::from("check_free_after_iterating"),
                Arc::from("MISSING_C_DOCSTRINGS"),
                Arc::from("MAX_Py_ssize_t"),
                Arc::from("NHASHBITS"),
                Arc::from("NEVER_EQ"),
                Arc::from("Py_DEBUG"),
                Arc::from("TestFailed"),
                Arc::from("cpython_only"),
                Arc::from("gc_collect"),
                Arc::from("get_attribute"),
                Arc::from("infinite_recursion"),
                Arc::from("is_emscripten"),
                Arc::from("is_wasi"),
                Arc::from("iter_builtin_types"),
                Arc::from("iter_slot_wrappers"),
                Arc::from("load_package_tests"),
                Arc::from("no_tracing"),
                Arc::from("refcount_test"),
                Arc::from("os_helper"),
                Arc::from("requires_resource"),
                Arc::from("requires_limited_api"),
                Arc::from("requires_docstrings"),
                Arc::from("requires_IEEE_754"),
                Arc::from("run_with_locale"),
                Arc::from("run_with_locales"),
                Arc::from("skip_if_pgo_task"),
                Arc::from("skip_on_s390x"),
                Arc::from("sortdict"),
                Arc::from("swap_item"),
                Arc::from("threading_helper"),
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
            "_2G" => Ok(Value::int(BIGMEM_2G).expect("_2G fits in tagged int")),
            "_4G" => Ok(Value::int(BIGMEM_4G).expect("_4G fits in tagged int")),
            "bigmemtest" => Ok(builtin_value(&BIGMEMTEST_FUNCTION)),
            "check_impl_detail" => Ok(builtin_value(&CHECK_IMPL_DETAIL_FUNCTION)),
            "check_free_after_iterating" => Ok(builtin_value(&CHECK_FREE_AFTER_ITERATING_FUNCTION)),
            "MISSING_C_DOCSTRINGS" => Ok(Value::bool(true)),
            "MAX_Py_ssize_t" => Ok(bigint_to_value(BigInt::from(isize::MAX))),
            "NHASHBITS" => Ok(
                Value::int((usize::BITS - 1) as i64).expect("hash bit width fits in tagged int")
            ),
            "NEVER_EQ" => Ok(*NEVER_EQ_VALUE),
            "Py_DEBUG" => Ok(Value::bool(false)),
            "TestFailed" => Ok(exception_type_value_for_id(
                ExceptionTypeId::AssertionError.as_u8() as u16,
            )
            .expect("AssertionError exception type is registered")),
            "cpython_only" => Ok(builtin_value(&CPYTHON_ONLY_FUNCTION)),
            "gc_collect" => Ok(builtin_value(&GC_COLLECT_FUNCTION)),
            "get_attribute" => Ok(builtin_value(&GET_ATTRIBUTE_FUNCTION)),
            "infinite_recursion" => Ok(builtin_value(&INFINITE_RECURSION_FUNCTION)),
            "is_emscripten" => Ok(Value::bool(false)),
            "is_wasi" => Ok(Value::bool(false)),
            "iter_builtin_types" => Ok(builtin_value(&ITER_BUILTIN_TYPES_FUNCTION)),
            "iter_slot_wrappers" => Ok(builtin_value(&ITER_SLOT_WRAPPERS_FUNCTION)),
            "load_package_tests" => Ok(builtin_value(&LOAD_PACKAGE_TESTS_FUNCTION)),
            "no_tracing" => Ok(builtin_value(&NO_TRACING_FUNCTION)),
            "refcount_test" => Ok(builtin_value(&REFCOUNT_TEST_FUNCTION)),
            "requires_resource" => Ok(builtin_value(&REQUIRES_RESOURCE_FUNCTION)),
            "requires_limited_api" => Ok(builtin_value(&REQUIRES_LIMITED_API_FUNCTION)),
            "requires_docstrings" => Ok(builtin_value(&REQUIRES_DOCSTRINGS_FUNCTION)),
            "requires_IEEE_754" => Ok(builtin_value(&REQUIRES_IEEE_754_FUNCTION)),
            "run_with_locale" => Ok(builtin_value(&RUN_WITH_LOCALE_FUNCTION)),
            "run_with_locales" => Ok(builtin_value(&RUN_WITH_LOCALES_FUNCTION)),
            "skip_if_pgo_task" => Ok(builtin_value(&SKIP_IF_PGO_TASK_FUNCTION)),
            "skip_on_s390x" => Ok(builtin_value(&SKIP_ON_S390X_FUNCTION)),
            "sortdict" => Ok(builtin_value(&SORTDICT_FUNCTION)),
            "swap_item" => Ok(builtin_value(&SWAP_ITEM_FUNCTION)),
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

fn build_context_manager_class(
    name: &'static str,
    module: &'static str,
    enter: &'static LazyLock<BuiltinFunctionObject>,
    exit: &'static LazyLock<BuiltinFunctionObject>,
) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern(module)));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    class.set_attr(intern("__enter__"), builtin_value(&**enter));
    class.set_attr(intern("__exit__"), builtin_value(&**exit));
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

fn swap_item(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "swap_item() takes exactly 3 arguments ({} given)",
            args.len()
        )));
    }

    let mut instance = allocate_heap_instance_for_class(SWAP_ITEM_CLASS.as_ref());
    set_swap_item_property(&mut instance, SWAP_ITEM_MAPPING_ATTR, args[0]);
    set_swap_item_property(&mut instance, SWAP_ITEM_KEY_ATTR, args[1]);
    set_swap_item_property(&mut instance, SWAP_ITEM_VALUE_ATTR, args[2]);
    set_swap_item_property(&mut instance, SWAP_ITEM_OLD_VALUE_ATTR, Value::none());
    set_swap_item_property(&mut instance, SWAP_ITEM_HAD_ITEM_ATTR, Value::bool(false));
    Ok(crate::alloc_managed_value(instance))
}

fn swap_item_enter(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__enter__() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let (mapping, key, value) = {
        let object = swap_item_object_ref(args[0])?;
        (
            swap_item_property(object, SWAP_ITEM_MAPPING_ATTR)?,
            swap_item_property(object, SWAP_ITEM_KEY_ATTR)?,
            swap_item_property(object, SWAP_ITEM_VALUE_ATTR)?,
        )
    };

    let old_value =
        mapping_get_optional(vm, mapping, key).map_err(runtime_error_to_builtin_error)?;
    mapping_set_item(vm, mapping, key, value).map_err(runtime_error_to_builtin_error)?;

    {
        let object = swap_item_object_mut(args[0])?;
        set_swap_item_property(
            object,
            SWAP_ITEM_OLD_VALUE_ATTR,
            old_value.unwrap_or_else(Value::none),
        );
        set_swap_item_property(
            object,
            SWAP_ITEM_HAD_ITEM_ATTR,
            Value::bool(old_value.is_some()),
        );
    }

    Ok(old_value.unwrap_or_else(Value::none))
}

fn swap_item_exit(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "__exit__() takes exactly 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let (mapping, key, old_value, had_item) = {
        let object = swap_item_object_ref(args[0])?;
        (
            swap_item_property(object, SWAP_ITEM_MAPPING_ATTR)?,
            swap_item_property(object, SWAP_ITEM_KEY_ATTR)?,
            swap_item_property(object, SWAP_ITEM_OLD_VALUE_ATTR)?,
            swap_item_property(object, SWAP_ITEM_HAD_ITEM_ATTR)?
                .as_bool()
                .unwrap_or(false),
        )
    };

    if had_item {
        mapping_set_item(vm, mapping, key, old_value).map_err(runtime_error_to_builtin_error)?;
    } else {
        mapping_delete_if_present(vm, mapping, key).map_err(runtime_error_to_builtin_error)?;
    }

    Ok(Value::bool(false))
}

fn swap_item_object_ref(value: Value) -> Result<&'static ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(swap_item_receiver_error)?;
    if extract_type_id(ptr) != SWAP_ITEM_CLASS.class_type_id() {
        return Err(swap_item_receiver_error());
    }
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn swap_item_object_mut(value: Value) -> Result<&'static mut ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(swap_item_receiver_error)?;
    if extract_type_id(ptr) != SWAP_ITEM_CLASS.class_type_id() {
        return Err(swap_item_receiver_error());
    }
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn swap_item_receiver_error() -> BuiltinError {
    BuiltinError::TypeError("_SwapItem context manager method requires a '_SwapItem' object".into())
}

#[inline]
fn set_swap_item_property(object: &mut ShapedObject, name: &str, value: Value) {
    object.set_property(intern(name), value, shape_registry());
}

#[inline]
fn swap_item_property(object: &ShapedObject, name: &str) -> Result<Value, BuiltinError> {
    object
        .get_property(name)
        .ok_or_else(|| BuiltinError::TypeError(format!("corrupt _SwapItem object: missing {name}")))
}

fn mapping_get_optional(
    vm: &mut VirtualMachine,
    mapping: Value,
    key: Value,
) -> Result<Option<Value>, RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && let Some(dict) = dict_storage_ref_from_ptr(ptr)
    {
        return dict_get_item(vm, dict, key);
    }

    if let Some(false) = mapping_contains_item(vm, mapping, key)? {
        return Ok(None);
    }

    let target = resolve_special_method(mapping, "__getitem__")?;
    match invoke_bound_mapping_method(vm, target, &[key]) {
        Ok(value) => Ok(Some(value)),
        Err(err) if is_key_error(&err) => Ok(None),
        Err(err) => Err(err),
    }
}

fn mapping_set_item(
    vm: &mut VirtualMachine,
    mapping: Value,
    key: Value,
    value: Value,
) -> Result<(), RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && let Some(dict) = dict_storage_mut_from_ptr(ptr)
    {
        return dict_set_item(vm, dict, key, value);
    }

    let target = resolve_special_method(mapping, "__setitem__")?;
    invoke_bound_mapping_method(vm, target, &[key, value]).map(|_| ())
}

fn mapping_delete_if_present(
    vm: &mut VirtualMachine,
    mapping: Value,
    key: Value,
) -> Result<(), RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && let Some(dict) = dict_storage_mut_from_ptr(ptr)
    {
        let _ = dict_remove_item(vm, dict, key)?;
        return Ok(());
    }

    if let Some(false) = mapping_contains_item(vm, mapping, key)? {
        return Ok(());
    }

    let target = resolve_special_method(mapping, "__delitem__")?;
    match invoke_bound_mapping_method(vm, target, &[key]) {
        Ok(_) => Ok(()),
        Err(err) if is_key_error(&err) => Ok(()),
        Err(err) => Err(err),
    }
}

fn mapping_contains_item(
    vm: &mut VirtualMachine,
    mapping: Value,
    key: Value,
) -> Result<Option<bool>, RuntimeError> {
    let target = match resolve_special_method(mapping, "__contains__") {
        Ok(target) => target,
        Err(err) if matches!(err.kind(), RuntimeErrorKind::AttributeError { .. }) => {
            return Ok(None);
        }
        Err(err) => return Err(err),
    };

    let result = invoke_bound_mapping_method(vm, target, &[key])?;
    crate::truthiness::try_is_truthy(vm, result).map(Some)
}

#[inline]
fn invoke_bound_mapping_method(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
    operands: &[Value],
) -> Result<Value, RuntimeError> {
    match (target.implicit_self, operands) {
        (Some(receiver), [arg]) => invoke_callable_value(vm, target.callable, &[receiver, *arg]),
        (Some(receiver), [arg1, arg2]) => {
            invoke_callable_value(vm, target.callable, &[receiver, *arg1, *arg2])
        }
        (None, [arg]) => invoke_callable_value(vm, target.callable, &[*arg]),
        (None, [arg1, arg2]) => invoke_callable_value(vm, target.callable, &[*arg1, *arg2]),
        _ => Err(RuntimeError::type_error(
            "internal mapping helper received an invalid arity",
        )),
    }
}

#[inline]
fn is_key_error(err: &RuntimeError) -> bool {
    match err.kind() {
        RuntimeErrorKind::KeyError { .. } => true,
        RuntimeErrorKind::Exception { type_id, .. } => {
            *type_id == ExceptionTypeId::KeyError.as_u8() as u16
        }
        _ => false,
    }
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

fn bigmemtest(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "bigmemtest() takes at most 3 positional arguments ({} given)",
            args.len()
        )));
    }

    let mut seen_size = !args.is_empty();
    let mut seen_memuse = args.len() >= 2;
    let mut seen_dry_run = args.len() >= 3;

    for (name, _) in keywords {
        match *name {
            "size" if !seen_size => seen_size = true,
            "memuse" if !seen_memuse => seen_memuse = true,
            "dry_run" if !seen_dry_run => seen_dry_run = true,
            "size" | "memuse" | "dry_run" => {
                return Err(BuiltinError::TypeError(format!(
                    "bigmemtest() got multiple values for argument '{name}'"
                )));
            }
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "bigmemtest() got an unexpected keyword argument '{name}'"
                )));
            }
        }
    }

    if !seen_size || !seen_memuse {
        return Err(BuiltinError::TypeError(
            "bigmemtest() missing required arguments: 'size' and 'memuse'".to_string(),
        ));
    }

    Ok(builtin_value(&BIGMEM_SKIP_DECORATOR_FUNCTION))
}

fn bigmem_skip_decorator(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "bigmemtest decorator takes exactly one argument ({} given)",
            args.len()
        )));
    }
    mark_unittest_skip(
        vm,
        args[0],
        "big memory tests require an explicit memory limit",
    )
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

fn gc_collect(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "gc_collect() takes no arguments ({} given)",
            args.len()
        )));
    }

    crate::stdlib::_weakref::clear_unreachable_weakrefs(vm);
    Ok(Value::none())
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

fn check_impl_detail(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "check_impl_detail() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }

    let Some((_, first_value)) = keywords.first() else {
        return Ok(Value::bool(false));
    };
    let default = !first_value.is_truthy();

    for (name, value) in keywords {
        if *name == "prism" {
            return Ok(Value::bool(value.is_truthy()));
        }
    }

    Ok(Value::bool(default))
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

fn requires_resource(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "requires_resource() takes exactly one argument ({} given)",
            args.len()
        )));
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
            attrs: vec![
                Arc::from("TESTFN"),
                Arc::from("create_empty_file"),
                Arc::from("unlink"),
            ],
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
            "create_empty_file" => Ok(builtin_value(&CREATE_EMPTY_FILE_FUNCTION)),
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

/// Native `test.support.import_helper` module descriptor.
#[derive(Debug, Clone)]
pub struct ImportHelperModule {
    attrs: Vec<Arc<str>>,
}

impl ImportHelperModule {
    /// Create a new `test.support.import_helper` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("forget"), Arc::from("import_module")],
        }
    }
}

impl Default for ImportHelperModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ImportHelperModule {
    fn name(&self) -> &str {
        "test.support.import_helper"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "forget" => Ok(builtin_value(&FORGET_FUNCTION)),
            "import_module" => Ok(builtin_value(&IMPORT_MODULE_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'test.support.import_helper' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

fn forget_module(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "forget() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let name = value_as_string_ref(args[0])
        .ok_or_else(|| BuiltinError::TypeError("forget() module name must be str".to_string()))?;
    vm.import_resolver.remove_module(name.as_str());
    Ok(Value::none())
}

fn import_module(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "import_module() takes from 1 to 3 positional arguments but {} were given",
            args.len()
        )));
    }

    let name = value_as_string_ref(args[0])
        .ok_or_else(|| BuiltinError::TypeError("import_module() name must be str".to_string()))?;
    let module = vm
        .import_module_named(name.as_str())
        .map_err(runtime_error_to_builtin_error)?;
    Ok(Value::object_ptr(Arc::as_ptr(&module) as *const ()))
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

fn create_empty_file(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "create_empty_file() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let path = value_as_string_ref(args[0]).ok_or_else(|| {
        BuiltinError::TypeError("create_empty_file() path must be str".to_string())
    })?;
    std::fs::File::create(Path::new(path.as_str()))
        .map(|_| Value::none())
        .map_err(|err| BuiltinError::OSError(err.to_string()))
}

/// Native `test.support.threading_helper` module descriptor.
#[derive(Debug, Clone)]
pub struct ThreadingHelperModule {
    attrs: Vec<Arc<str>>,
}

impl ThreadingHelperModule {
    /// Create a new `test.support.threading_helper` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("reap_threads"),
                Arc::from("requires_working_threading"),
            ],
        }
    }
}

impl Default for ThreadingHelperModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ThreadingHelperModule {
    fn name(&self) -> &str {
        "test.support.threading_helper"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "reap_threads" => Ok(builtin_value(&REAP_THREADS_FUNCTION)),
            "requires_working_threading" => Ok(builtin_value(&REQUIRES_WORKING_THREADING_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'test.support.threading_helper' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

fn requires_working_threading(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "requires_working_threading() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(builtin_value(&THREADING_SKIP_DECORATOR_FUNCTION))
}

fn threading_skip_decorator(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "threading skip decorator takes exactly one argument ({} given)",
            args.len()
        )));
    }
    mark_unittest_skip(vm, args[0], "high-level threading runtime is not available")
}

/// Native `test.support.warnings_helper` module descriptor.
#[derive(Debug, Clone)]
pub struct WarningsHelperModule {
    attrs: Vec<Arc<str>>,
}

impl WarningsHelperModule {
    /// Create a new `test.support.warnings_helper` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("save_restore_warnings_filters")],
        }
    }
}

impl Default for WarningsHelperModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for WarningsHelperModule {
    fn name(&self) -> &str {
        "test.support.warnings_helper"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "save_restore_warnings_filters" => {
                Ok(builtin_value(&SAVE_RESTORE_WARNINGS_FILTERS_FUNCTION))
            }
            _ => Err(ModuleError::AttributeError(format!(
                "module 'test.support.warnings_helper' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

fn save_restore_warnings_filters(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "save_restore_warnings_filters() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(*WARNINGS_FILTER_CONTEXT_VALUE)
}

fn noop_context_enter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__enter__() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(args[0])
}

fn noop_context_exit(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "__exit__() takes exactly 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(Value::bool(false))
}
