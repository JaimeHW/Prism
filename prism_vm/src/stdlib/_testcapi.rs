//! Native subset of CPython's `_testcapi` module used by regression tests.
//!
//! This module intentionally implements the pieces of `_testcapi` that exercise
//! public call semantics from Python. The helpers are native Rust functions so
//! Prism's CPython compatibility tests validate the runtime call paths directly.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class,
    runtime_error_to_builtin_error,
};
use crate::import::ModuleObject;
use crate::ops::calls::{invoke_callable_value, invoke_callable_value_with_keywords};
use crate::ops::objects::dict_storage_ref_from_ptr;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::descriptor::{
    BoundMethod, ClassMethodDescriptor, StaticMethodDescriptor,
};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, builtin_class_mro, class_id_to_type_id, global_class, global_class_bitmap,
    register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use smallvec::SmallVec;
use std::sync::{Arc, LazyLock};

const MODULE_NAME: &str = "_testcapi";

static METH_VARARGS: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("meth_varargs"), meth_varargs));
static METH_VARARGS_KEYWORDS: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("meth_varargs_keywords"), meth_varargs_keywords)
});
static METH_O: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("meth_o"), meth_o));
static METH_NOARGS: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("meth_noargs"), meth_noargs));
static METH_FASTCALL: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("meth_fastcall"), meth_fastcall));
static METH_FASTCALL_KEYWORDS: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("meth_fastcall_keywords"), meth_fastcall_keywords)
});

static METH_VARARGS_STATIC: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("meth_varargs"), meth_varargs_static));
static METH_VARARGS_KEYWORDS_STATIC: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("meth_varargs_keywords"),
        meth_varargs_keywords_static,
    )
});
static METH_O_STATIC: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("meth_o"), meth_o_static));
static METH_NOARGS_STATIC: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("meth_noargs"), meth_noargs_static));
static METH_FASTCALL_STATIC: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("meth_fastcall"), meth_fastcall_static)
});
static METH_FASTCALL_KEYWORDS_STATIC: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("meth_fastcall_keywords"),
        meth_fastcall_keywords_static,
    )
});

static PYOBJECT_FASTCALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_testcapi.pyobject_fastcall"), pyobject_fastcall)
});
static PYOBJECT_FASTCALLDICT: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("_testcapi.pyobject_fastcalldict"),
        pyobject_fastcalldict,
    )
});
static PYOBJECT_VECTORCALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("_testcapi.pyobject_vectorcall"),
        pyobject_vectorcall,
    )
});
static PYVECTORCALL_CALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("_testcapi.pyvectorcall_call"), pyvectorcall_call)
});
static FUNCTION_SETVECTORCALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_testcapi.function_setvectorcall"),
        function_setvectorcall,
    )
});
static MAKE_VECTORCALL_CLASS: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_testcapi.make_vectorcall_class"),
        make_vectorcall_class,
    )
});
static HAS_VECTORCALL_FLAG: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_testcapi.has_vectorcall_flag"),
        has_vectorcall_flag,
    )
});

static METHOD_DESCRIPTOR_CALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("_testcapi.MethodDescriptor.__call__"),
        method_descriptor_call,
    )
});
static METHOD_DESCRIPTOR2_CALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("_testcapi.MethodDescriptor2.__call__"),
        method_descriptor2_call,
    )
});
static METHOD_DESCRIPTOR_NOP_GET_CALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("_testcapi.MethodDescriptorNopGet.__call__"),
        method_descriptor_nop_get_call,
    )
});
static METHOD_DESCRIPTOR_GET: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("_testcapi.MethodDescriptor.__get__"),
        method_descriptor_get,
    )
});
static METHOD_DESCRIPTOR_NOP_GET: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("_testcapi.MethodDescriptorNopGet.__get__"),
        method_descriptor_nop_get,
    )
});
static VECTORCALL_CLASS_CALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("_testcapi.VectorCallClass.__call__"),
        vectorcall_class_call,
    )
});
static VECTORCALL_CLASS_SET_VECTORCALL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("_testcapi.VectorCallClass.set_vectorcall"),
        vectorcall_class_set_vectorcall,
    )
});

static METH_INSTANCE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_call_convention_class("MethInstance", MethodBinding::Instance));
static METH_CLASS_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_call_convention_class("MethClass", MethodBinding::Class));
static METH_STATIC_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_call_convention_class("MethStatic", MethodBinding::Static));
static METHOD_DESCRIPTOR_BASE_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_method_descriptor_class(
        "MethodDescriptorBase",
        None,
        builtin_value(&METHOD_DESCRIPTOR_CALL),
        builtin_value(&METHOD_DESCRIPTOR_GET),
        ClassFlags::HAS_VECTORCALL | ClassFlags::METHOD_DESCRIPTOR,
    )
});
static METHOD_DESCRIPTOR_DERIVED_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_method_descriptor_class(
        "MethodDescriptorDerived",
        Some(&METHOD_DESCRIPTOR_BASE_CLASS),
        Value::none(),
        Value::none(),
        ClassFlags::HAS_VECTORCALL | ClassFlags::METHOD_DESCRIPTOR,
    )
});
static METHOD_DESCRIPTOR_NOP_GET_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_method_descriptor_class(
        "MethodDescriptorNopGet",
        Some(&METHOD_DESCRIPTOR_BASE_CLASS),
        builtin_value(&METHOD_DESCRIPTOR_NOP_GET_CALL),
        builtin_value(&METHOD_DESCRIPTOR_NOP_GET),
        ClassFlags::empty(),
    )
});
static METHOD_DESCRIPTOR2_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_method_descriptor_class(
        "MethodDescriptor2",
        Some(&METHOD_DESCRIPTOR_BASE_CLASS),
        builtin_value(&METHOD_DESCRIPTOR2_CALL),
        Value::none(),
        ClassFlags::HAS_VECTORCALL,
    )
});

#[derive(Debug, Clone)]
pub struct TestCapiModule {
    attrs: Vec<Arc<str>>,
}

impl TestCapiModule {
    pub fn new() -> Self {
        Self {
            attrs: [
                "__doc__",
                "MethInstance",
                "MethClass",
                "MethStatic",
                "meth_varargs",
                "meth_varargs_keywords",
                "meth_o",
                "meth_noargs",
                "meth_fastcall",
                "meth_fastcall_keywords",
                "pyobject_fastcall",
                "pyobject_fastcalldict",
                "pyobject_vectorcall",
                "pyvectorcall_call",
                "function_setvectorcall",
                "make_vectorcall_class",
                "has_vectorcall_flag",
                "MethodDescriptorBase",
                "MethodDescriptorDerived",
                "MethodDescriptorNopGet",
                "MethodDescriptor2",
            ]
            .into_iter()
            .map(Arc::from)
            .collect(),
        }
    }
}

impl Module for TestCapiModule {
    fn name(&self) -> &str {
        MODULE_NAME
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__doc__" => Ok(Value::string(intern(
                "Native CPython C-API test helpers for Prism's regression suite.",
            ))),
            "MethInstance" => Ok(class_value(&METH_INSTANCE_CLASS)),
            "MethClass" => Ok(class_value(&METH_CLASS_CLASS)),
            "MethStatic" => Ok(class_value(&METH_STATIC_CLASS)),
            "meth_varargs" => Ok(builtin_value(&METH_VARARGS)),
            "meth_varargs_keywords" => Ok(builtin_value(&METH_VARARGS_KEYWORDS)),
            "meth_o" => Ok(builtin_value(&METH_O)),
            "meth_noargs" => Ok(builtin_value(&METH_NOARGS)),
            "meth_fastcall" => Ok(builtin_value(&METH_FASTCALL)),
            "meth_fastcall_keywords" => Ok(builtin_value(&METH_FASTCALL_KEYWORDS)),
            "pyobject_fastcall" => Ok(builtin_value(&PYOBJECT_FASTCALL)),
            "pyobject_fastcalldict" => Ok(builtin_value(&PYOBJECT_FASTCALLDICT)),
            "pyobject_vectorcall" => Ok(builtin_value(&PYOBJECT_VECTORCALL)),
            "pyvectorcall_call" => Ok(builtin_value(&PYVECTORCALL_CALL)),
            "function_setvectorcall" => Ok(builtin_value(&FUNCTION_SETVECTORCALL)),
            "make_vectorcall_class" => Ok(builtin_value(&MAKE_VECTORCALL_CLASS)),
            "has_vectorcall_flag" => Ok(builtin_value(&HAS_VECTORCALL_FLAG)),
            "MethodDescriptorBase" => Ok(class_value(&METHOD_DESCRIPTOR_BASE_CLASS)),
            "MethodDescriptorDerived" => Ok(class_value(&METHOD_DESCRIPTOR_DERIVED_CLASS)),
            "MethodDescriptorNopGet" => Ok(class_value(&METHOD_DESCRIPTOR_NOP_GET_CLASS)),
            "MethodDescriptor2" => Ok(class_value(&METHOD_DESCRIPTOR2_CLASS)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_testcapi' has no attribute '{name}'"
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }

    fn populate(&self, module: &ModuleObject) -> Result<(), ModuleError> {
        let module_value = Value::object_ptr(module as *const ModuleObject as *const ());

        module.set_attr("__doc__", self.get_attr("__doc__")?);
        module.set_attr("MethInstance", class_value(&METH_INSTANCE_CLASS));
        module.set_attr("MethClass", class_value(&METH_CLASS_CLASS));
        module.set_attr("MethStatic", class_value(&METH_STATIC_CLASS));

        module.set_attr(
            "meth_varargs",
            bound_builtin_value(&METH_VARARGS, module_value),
        );
        module.set_attr(
            "meth_varargs_keywords",
            bound_builtin_value(&METH_VARARGS_KEYWORDS, module_value),
        );
        module.set_attr("meth_o", bound_builtin_value(&METH_O, module_value));
        module.set_attr(
            "meth_noargs",
            bound_builtin_value(&METH_NOARGS, module_value),
        );
        module.set_attr(
            "meth_fastcall",
            bound_builtin_value(&METH_FASTCALL, module_value),
        );
        module.set_attr(
            "meth_fastcall_keywords",
            bound_builtin_value(&METH_FASTCALL_KEYWORDS, module_value),
        );

        module.set_attr("pyobject_fastcall", builtin_value(&PYOBJECT_FASTCALL));
        module.set_attr(
            "pyobject_fastcalldict",
            builtin_value(&PYOBJECT_FASTCALLDICT),
        );
        module.set_attr("pyobject_vectorcall", builtin_value(&PYOBJECT_VECTORCALL));
        module.set_attr("pyvectorcall_call", builtin_value(&PYVECTORCALL_CALL));
        module.set_attr(
            "function_setvectorcall",
            builtin_value(&FUNCTION_SETVECTORCALL),
        );
        module.set_attr(
            "make_vectorcall_class",
            builtin_value(&MAKE_VECTORCALL_CLASS),
        );
        module.set_attr("has_vectorcall_flag", builtin_value(&HAS_VECTORCALL_FLAG));
        module.set_attr(
            "MethodDescriptorBase",
            class_value(&METHOD_DESCRIPTOR_BASE_CLASS),
        );
        module.set_attr(
            "MethodDescriptorDerived",
            class_value(&METHOD_DESCRIPTOR_DERIVED_CLASS),
        );
        module.set_attr(
            "MethodDescriptorNopGet",
            class_value(&METHOD_DESCRIPTOR_NOP_GET_CLASS),
        );
        module.set_attr("MethodDescriptor2", class_value(&METHOD_DESCRIPTOR2_CLASS));
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum MethodBinding {
    Instance,
    Class,
    Static,
}

fn build_call_convention_class(name: &'static str, binding: MethodBinding) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern(MODULE_NAME)));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));

    match binding {
        MethodBinding::Instance => install_instance_methods(&mut class),
        MethodBinding::Class => install_class_methods(&mut class),
        MethodBinding::Static => install_static_methods(&mut class),
    }

    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);
    register_native_type(class)
}

fn install_instance_methods(class: &mut PyClassObject) {
    class.set_attr(intern("meth_varargs"), builtin_value(&METH_VARARGS));
    class.set_attr(
        intern("meth_varargs_keywords"),
        builtin_value(&METH_VARARGS_KEYWORDS),
    );
    class.set_attr(intern("meth_o"), builtin_value(&METH_O));
    class.set_attr(intern("meth_noargs"), builtin_value(&METH_NOARGS));
    class.set_attr(intern("meth_fastcall"), builtin_value(&METH_FASTCALL));
    class.set_attr(
        intern("meth_fastcall_keywords"),
        builtin_value(&METH_FASTCALL_KEYWORDS),
    );
}

fn install_class_methods(class: &mut PyClassObject) {
    class.set_attr(intern("meth_varargs"), classmethod_value(&METH_VARARGS));
    class.set_attr(
        intern("meth_varargs_keywords"),
        classmethod_value(&METH_VARARGS_KEYWORDS),
    );
    class.set_attr(intern("meth_o"), classmethod_value(&METH_O));
    class.set_attr(intern("meth_noargs"), classmethod_value(&METH_NOARGS));
    class.set_attr(intern("meth_fastcall"), classmethod_value(&METH_FASTCALL));
    class.set_attr(
        intern("meth_fastcall_keywords"),
        classmethod_value(&METH_FASTCALL_KEYWORDS),
    );
}

fn install_static_methods(class: &mut PyClassObject) {
    class.set_attr(
        intern("meth_varargs"),
        staticmethod_value(&METH_VARARGS_STATIC),
    );
    class.set_attr(
        intern("meth_varargs_keywords"),
        staticmethod_value(&METH_VARARGS_KEYWORDS_STATIC),
    );
    class.set_attr(intern("meth_o"), staticmethod_value(&METH_O_STATIC));
    class.set_attr(
        intern("meth_noargs"),
        staticmethod_value(&METH_NOARGS_STATIC),
    );
    class.set_attr(
        intern("meth_fastcall"),
        staticmethod_value(&METH_FASTCALL_STATIC),
    );
    class.set_attr(
        intern("meth_fastcall_keywords"),
        staticmethod_value(&METH_FASTCALL_KEYWORDS_STATIC),
    );
}

fn build_method_descriptor_class(
    name: &'static str,
    base: Option<&Arc<PyClassObject>>,
    call: Value,
    get: Value,
    extra_flags: ClassFlags,
) -> Arc<PyClassObject> {
    let mut class = match base {
        Some(base) => PyClassObject::new(intern(name), &[base.class_id()], |class_id| {
            if class_id == base.class_id() {
                Some(base.mro().iter().copied().collect())
            } else if class_id.0 < TypeId::FIRST_USER_TYPE {
                Some(builtin_class_mro(class_id_to_type_id(class_id)).into())
            } else {
                global_class(class_id).map(|class| class.mro().iter().copied().collect())
            }
        })
        .expect("_testcapi method descriptor class has a valid MRO"),
        None => PyClassObject::new_simple(intern(name)),
    };

    class.set_attr(intern("__module__"), Value::string(intern(MODULE_NAME)));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    if !call.is_none() {
        class.set_attr(intern("__call__"), call);
    }
    if !get.is_none() {
        class.set_attr(intern("__get__"), get);
    }

    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE | extra_flags);
    register_native_type(class)
}

fn register_native_type(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    let class = Arc::new(class);
    register_global_class(Arc::clone(&class), bitmap);
    class
}

#[inline]
fn class_value(class: &Arc<PyClassObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn bound_builtin_value(function: &'static BuiltinFunctionObject, receiver: Value) -> Value {
    crate::alloc_managed_value(function.bind(receiver))
}

#[inline]
fn classmethod_value(function: &'static BuiltinFunctionObject) -> Value {
    crate::alloc_managed_value(ClassMethodDescriptor::new(builtin_value(function)))
}

#[inline]
fn staticmethod_value(function: &'static BuiltinFunctionObject) -> Value {
    crate::alloc_managed_value(StaticMethodDescriptor::new(builtin_value(function)))
}

#[inline]
fn tuple_value(items: &[Value]) -> Value {
    crate::alloc_managed_value(TupleObject::from_slice(items))
}

#[inline]
fn tuple_value_from_vec(items: Vec<Value>) -> Value {
    crate::alloc_managed_value(TupleObject::from_vec(items))
}

#[inline]
fn dict_value(dict: DictObject) -> Value {
    crate::alloc_managed_value(dict)
}

fn reject_keywords(name: &str, keywords: &[(&str, Value)]) -> Result<(), BuiltinError> {
    if keywords.is_empty() {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{name}() takes no keyword arguments"
        )))
    }
}

fn split_bound_args<'a>(
    name: &str,
    args: &'a [Value],
) -> Result<(Value, &'a [Value]), BuiltinError> {
    let Some((&receiver, rest)) = args.split_first() else {
        return Err(BuiltinError::TypeError(format!(
            "{name}() missing required C receiver"
        )));
    };
    Ok((receiver, rest))
}

fn kwargs_dict_value(keywords: &[(&str, Value)]) -> Value {
    if keywords.is_empty() {
        return Value::none();
    }

    let mut dict = DictObject::with_capacity(keywords.len());
    for &(name, value) in keywords {
        dict.set(Value::string(intern(name)), value);
    }
    dict_value(dict)
}

fn meth_varargs(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    reject_keywords("meth_varargs", keywords)?;
    let (receiver, positional) = split_bound_args("meth_varargs", args)?;
    Ok(tuple_value(&[receiver, tuple_value(positional)]))
}

fn meth_varargs_keywords(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let (receiver, positional) = split_bound_args("meth_varargs_keywords", args)?;
    Ok(tuple_value(&[
        receiver,
        tuple_value(positional),
        kwargs_dict_value(keywords),
    ]))
}

fn meth_o(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    reject_keywords("meth_o", keywords)?;
    let (receiver, positional) = split_bound_args("meth_o", args)?;
    if positional.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "meth_o() takes exactly one argument ({} given)",
            positional.len()
        )));
    }
    Ok(tuple_value(&[receiver, positional[0]]))
}

fn meth_noargs(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    reject_keywords("meth_noargs", keywords)?;
    let (receiver, positional) = split_bound_args("meth_noargs", args)?;
    if !positional.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "meth_noargs() takes no arguments ({} given)",
            positional.len()
        )));
    }
    Ok(receiver)
}

fn meth_fastcall(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    reject_keywords("meth_fastcall", keywords)?;
    let (receiver, positional) = split_bound_args("meth_fastcall", args)?;
    Ok(tuple_value(&[receiver, tuple_value(positional)]))
}

fn meth_fastcall_keywords(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let (receiver, positional) = split_bound_args("meth_fastcall_keywords", args)?;
    Ok(tuple_value(&[
        receiver,
        tuple_value(positional),
        kwargs_dict_value(keywords),
    ]))
}

fn meth_varargs_static(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    reject_keywords("meth_varargs", keywords)?;
    Ok(tuple_value(&[Value::none(), tuple_value(args)]))
}

fn meth_varargs_keywords_static(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    Ok(tuple_value(&[
        Value::none(),
        tuple_value(args),
        kwargs_dict_value(keywords),
    ]))
}

fn meth_o_static(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    reject_keywords("meth_o", keywords)?;
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "meth_o() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(tuple_value(&[Value::none(), args[0]]))
}

fn meth_noargs_static(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    reject_keywords("meth_noargs", keywords)?;
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "meth_noargs() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(Value::none())
}

fn meth_fastcall_static(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    reject_keywords("meth_fastcall", keywords)?;
    Ok(tuple_value(&[Value::none(), tuple_value(args)]))
}

fn meth_fastcall_keywords_static(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    Ok(tuple_value(&[
        Value::none(),
        tuple_value(args),
        kwargs_dict_value(keywords),
    ]))
}

fn optional_tuple_items(
    value: Value,
    function: &str,
) -> Result<SmallVec<[Value; 8]>, BuiltinError> {
    if value.is_none() {
        return Ok(SmallVec::new());
    }

    let tuple = value_as_tuple_ref(value).ok_or_else(|| {
        BuiltinError::TypeError(format!("{function}() argument must be a tuple or None"))
    })?;
    Ok(tuple.as_slice().iter().copied().collect())
}

fn string_value(value: Value, function: &str) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|value| value.as_str().to_owned())
        .ok_or_else(|| BuiltinError::TypeError(format!("{function}() keywords must be strings")))
}

fn optional_keyword_names(
    value: Value,
    function: &str,
) -> Result<SmallVec<[String; 4]>, BuiltinError> {
    if value.is_none() {
        return Ok(SmallVec::new());
    }

    let tuple = value_as_tuple_ref(value).ok_or_else(|| {
        BuiltinError::TypeError(format!("{function}() kwnames must be a tuple or None"))
    })?;
    tuple
        .as_slice()
        .iter()
        .copied()
        .map(|value| string_value(value, function))
        .collect()
}

fn optional_kwargs_dict(
    value: Value,
    function: &str,
) -> Result<SmallVec<[(String, Value); 4]>, BuiltinError> {
    if value.is_none() {
        return Ok(SmallVec::new());
    }

    let dict = value
        .as_object_ptr()
        .and_then(dict_storage_ref_from_ptr)
        .ok_or_else(|| {
            BuiltinError::TypeError(format!("{function}() kwargs must be a dict or None"))
        })?;

    dict.iter()
        .map(|(key, value)| string_value(key, function).map(|name| (name, value)))
        .collect()
}

fn invoke_with_collected_keywords(
    vm: &mut VirtualMachine,
    callable: Value,
    positional: &[Value],
    keywords: &[(String, Value)],
) -> Result<Value, BuiltinError> {
    if keywords.is_empty() {
        return invoke_callable_value(vm, callable, positional)
            .map_err(runtime_error_to_builtin_error);
    }

    let keyword_refs: SmallVec<[(&str, Value); 4]> = keywords
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect();
    invoke_callable_value_with_keywords(vm, callable, positional, &keyword_refs)
        .map_err(runtime_error_to_builtin_error)
}

fn pyobject_fastcall(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "pyobject_fastcall() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let positional = optional_tuple_items(args[1], "pyobject_fastcall")?;
    invoke_callable_value(vm, args[0], &positional).map_err(runtime_error_to_builtin_error)
}

fn pyobject_fastcalldict(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "pyobject_fastcalldict() takes exactly 3 arguments ({} given)",
            args.len()
        )));
    }

    let positional = optional_tuple_items(args[1], "pyobject_fastcalldict")?;
    let keywords = optional_kwargs_dict(args[2], "pyobject_fastcalldict")?;
    invoke_with_collected_keywords(vm, args[0], &positional, &keywords)
}

fn pyobject_vectorcall(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "pyobject_vectorcall() takes exactly 3 arguments ({} given)",
            args.len()
        )));
    }

    let all_args = optional_tuple_items(args[1], "pyobject_vectorcall")?;
    let keyword_names = optional_keyword_names(args[2], "pyobject_vectorcall")?;
    if keyword_names.is_empty() {
        return invoke_callable_value(vm, args[0], &all_args)
            .map_err(runtime_error_to_builtin_error);
    }
    if keyword_names.len() > all_args.len() {
        return Err(BuiltinError::ValueError(
            "pyobject_vectorcall() has more keyword names than argument values".to_string(),
        ));
    }

    let positional_len = all_args.len() - keyword_names.len();
    let mut keywords: SmallVec<[(String, Value); 4]> = SmallVec::with_capacity(keyword_names.len());
    for (index, name) in keyword_names.into_iter().enumerate() {
        keywords.push((name, all_args[positional_len + index]));
    }
    invoke_with_collected_keywords(vm, args[0], &all_args[..positional_len], &keywords)
}

fn pyvectorcall_call(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if !keywords.is_empty() {
        return Err(BuiltinError::TypeError(
            "pyvectorcall_call() takes no keyword arguments".to_string(),
        ));
    }
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "pyvectorcall_call() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let positional = optional_tuple_items(args[1], "pyvectorcall_call")?;
    let keyword_values = if let Some(&kwargs) = args.get(2) {
        optional_kwargs_dict(kwargs, "pyvectorcall_call")?
    } else {
        SmallVec::new()
    };
    invoke_with_collected_keywords(vm, args[0], &positional, &keyword_values)
}

fn function_setvectorcall(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "function_setvectorcall() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let ptr = args[0]
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("'func' must be a function".to_string()))?;
    if object_type_id(ptr) != TypeId::FUNCTION {
        return Err(BuiltinError::TypeError(
            "'func' must be a function".to_string(),
        ));
    }

    let func = unsafe { &*(ptr as *const FunctionObject) };
    func.set_test_vectorcall_override();
    Ok(Value::none())
}

fn has_vectorcall_flag(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "has_vectorcall_flag() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let flag = crate::builtins::python_type_has_vectorcall_flag(args[0]).ok_or_else(|| {
        BuiltinError::TypeError("has_vectorcall_flag() argument must be a type".to_string())
    })?;
    Ok(Value::bool(flag))
}

fn make_vectorcall_class(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "make_vectorcall_class() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let base = match args.first().copied() {
        Some(value) => base_class_id_from_type_value(value)?,
        None => None,
    };

    let name = intern("VectorcallClass");
    let mut class = match base {
        Some(base_id) => PyClassObject::new(name.clone(), &[base_id], |class_id| {
            if class_id.0 < TypeId::FIRST_USER_TYPE {
                Some(builtin_class_mro(class_id_to_type_id(class_id)).into())
            } else {
                global_class(class_id).map(|class| class.mro().iter().copied().collect())
            }
        })
        .map_err(|err| BuiltinError::TypeError(err.to_string()))?,
        None => PyClassObject::new_simple(name.clone()),
    };

    class.set_attr(intern("__module__"), Value::string(intern(MODULE_NAME)));
    class.set_attr(intern("__qualname__"), Value::string(name));
    class.set_attr(intern("__call__"), builtin_value(&VECTORCALL_CLASS_CALL));
    class.set_attr(
        intern("set_vectorcall"),
        builtin_value(&VECTORCALL_CLASS_SET_VECTORCALL),
    );
    class.add_flags(
        ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE | ClassFlags::HAS_VECTORCALL,
    );

    Ok(class_value(&register_native_type(class)))
}

fn method_descriptor_call(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    reject_keywords("MethodDescriptor.__call__", keywords)?;
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "descriptor call missing receiver".to_string(),
        ));
    }
    Ok(Value::bool(true))
}

fn method_descriptor2_call(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    reject_keywords("MethodDescriptor2.__call__", keywords)?;
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "descriptor call missing receiver".to_string(),
        ));
    }
    Ok(Value::bool(false))
}

fn method_descriptor_nop_get_call(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    reject_keywords("MethodDescriptorNopGet.__call__", keywords)?;
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "descriptor call missing receiver".to_string(),
        ));
    }
    Ok(tuple_value(&args[1..]))
}

fn method_descriptor_get(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    reject_keywords("MethodDescriptor.__get__", keywords)?;
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "MethodDescriptor.__get__() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let descriptor = args[0];
    let instance = args[1];
    if instance.is_none() {
        return Ok(descriptor);
    }

    Ok(crate::alloc_managed_value(BoundMethod::new(
        descriptor, instance,
    )))
}

fn method_descriptor_nop_get(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    reject_keywords("MethodDescriptorNopGet.__get__", keywords)?;
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "MethodDescriptorNopGet.__get__() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }
    Ok(args[0])
}

fn vectorcall_class_call(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    reject_keywords("VectorCallClass.__call__", keywords)?;
    let Some(&receiver) = args.first() else {
        return Err(BuiltinError::TypeError(
            "VectorCallClass.__call__() missing receiver".to_string(),
        ));
    };

    let vectorcall_enabled = receiver
        .as_object_ptr()
        .and_then(|ptr| {
            let type_id = object_type_id(ptr);
            (type_id.raw() >= TypeId::FIRST_USER_TYPE)
                .then(|| unsafe { &*(ptr as *const ShapedObject) })
        })
        .and_then(|object| object.get_property("__prism_testcapi_vectorcall"))
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    let class_has_vectorcall = instance_type_value(receiver)
        .and_then(crate::builtins::python_type_has_vectorcall_flag)
        .unwrap_or(false);

    Ok(Value::string(intern(
        if vectorcall_enabled && class_has_vectorcall {
            "vectorcall"
        } else {
            "tp_call"
        },
    )))
}

fn vectorcall_class_set_vectorcall(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    reject_keywords("VectorCallClass.set_vectorcall", keywords)?;
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "set_vectorcall() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    if !value_is_instance_of_type_value(args[0], args[1])? {
        return Err(BuiltinError::TypeError(
            "set_vectorcall() receiver is not an instance of the provided type".to_string(),
        ));
    }

    let ptr = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("set_vectorcall() receiver must be an object".to_string())
    })?;
    if object_type_id(ptr).raw() < TypeId::FIRST_USER_TYPE {
        return Err(BuiltinError::TypeError(
            "set_vectorcall() receiver must be a heap object".to_string(),
        ));
    }

    let object = unsafe { &mut *(ptr as *mut ShapedObject) };
    object.set_property(
        intern("__prism_testcapi_vectorcall"),
        Value::bool(true),
        shape_registry(),
    );
    Ok(Value::none())
}

#[inline]
fn object_type_id(ptr: *const ()) -> TypeId {
    unsafe { (*(ptr as *const ObjectHeader)).type_id }
}

fn base_class_id_from_type_value(value: Value) -> Result<Option<ClassId>, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("argument must be a type".to_string()))?;
    if let Some(type_id) = crate::builtins::builtin_type_object_type_id(ptr) {
        return Ok((type_id != TypeId::OBJECT).then_some(ClassId(type_id.raw())));
    }
    if object_type_id(ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "argument must be a type".to_string(),
        ));
    }
    let class = unsafe { &*(ptr as *const PyClassObject) };
    Ok(Some(class.class_id()))
}

fn instance_type_value(value: Value) -> Option<Value> {
    let ptr = value.as_object_ptr()?;
    let type_id = object_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Some(crate::builtins::builtin_type_object_for_type_id(type_id));
    }
    let class = global_class(ClassId(type_id.raw()))?;
    Some(Value::object_ptr(Arc::as_ptr(&class) as *const ()))
}

fn value_is_instance_of_type_value(value: Value, type_value: Value) -> Result<bool, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(false);
    };
    let value_type = object_type_id(ptr);

    let target_ptr = type_value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("argument must be a type".to_string()))?;
    if let Some(target) = crate::builtins::builtin_type_object_type_id(target_ptr) {
        if target == TypeId::OBJECT {
            return Ok(true);
        }
        if value_type == target {
            return Ok(true);
        }
        if value_type.raw() >= TypeId::FIRST_USER_TYPE {
            return Ok(global_class_bitmap(ClassId(value_type.raw()))
                .is_some_and(|bitmap| bitmap.is_subclass_of(target)));
        }
        return Ok(false);
    }

    if object_type_id(target_ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "argument must be a type".to_string(),
        ));
    }
    let target_class = unsafe { &*(target_ptr as *const PyClassObject) };
    if value_type == target_class.class_type_id() {
        return Ok(true);
    }
    Ok(global_class_bitmap(ClassId(value_type.raw()))
        .is_some_and(|bitmap| bitmap.is_subclass_of(target_class.class_type_id())))
}

pub(crate) fn is_method_descriptor_nop_get_instance(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    object_type_id(ptr) == METHOD_DESCRIPTOR_NOP_GET_CLASS.class_type_id()
}

#[allow(dead_code)]
fn default_instance(class: &PyClassObject) -> Value {
    crate::alloc_managed_value(allocate_heap_instance_for_class(class))
}
