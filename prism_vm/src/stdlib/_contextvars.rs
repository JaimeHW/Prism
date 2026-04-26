//! Native `_contextvars` bootstrap module.
//!
//! CPython's pure-Python `contextvars.py` is a thin re-export layer over the
//! `_contextvars` accelerator. Prism only needs a compact subset of that
//! surface to support stdlib import chains such as `decimal`, so this module
//! focuses on the `ContextVar` operations exercised during bootstrap.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, create_exception};
use crate::error::RuntimeError;
use crate::ops::calls::invoke_callable_value;
use crate::stdlib::exceptions::types::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, global_class_bitmap, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::string::StringObject;
use std::sync::{Arc, LazyLock};

static CONTEXTVAR_NEW_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_contextvars.ContextVar.__new__"), contextvar_new)
});
static CONTEXTVAR_INIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_contextvars.ContextVar.__init__"),
        contextvar_init,
    )
});
static CONTEXTVAR_GET_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_contextvars.ContextVar.get"), contextvar_get)
});
static CONTEXTVAR_SET_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_contextvars.ContextVar.set"), contextvar_set)
});
static CONTEXTVAR_RESET_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_contextvars.ContextVar.reset"), contextvar_reset)
});
static CONTEXT_NEW_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_contextvars.Context.__new__"), context_new)
});
static CONTEXT_INIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_contextvars.Context.__init__"), context_init)
});
static CONTEXT_COPY_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_contextvars.Context.copy"), context_copy)
});
static CONTEXT_RUN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_contextvars.Context.run"), context_run)
});
static COPY_CONTEXT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_contextvars.copy_context"), copy_context)
});

static CONTEXTVAR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_contextvar_type("ContextVar"));
static TOKEN_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| build_token_type("Token"));
static CONTEXT_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_context_type("Context"));

/// Native `_contextvars` module descriptor.
pub struct ContextVarsModule {
    attrs: Vec<Arc<str>>,
}

impl ContextVarsModule {
    /// Create a new `_contextvars` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("Context"),
                Arc::from("ContextVar"),
                Arc::from("Token"),
                Arc::from("copy_context"),
            ],
        }
    }
}

impl Default for ContextVarsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ContextVarsModule {
    fn name(&self) -> &str {
        "_contextvars"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "Context" => Ok(context_type_value()),
            "ContextVar" => Ok(contextvar_type_value()),
            "Token" => Ok(token_type_value()),
            "copy_context" => Ok(builtin_value(&COPY_CONTEXT_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_contextvars' has no attribute '{}'",
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
fn class_value(class: &'static Arc<PyClassObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

#[inline]
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

#[inline]
fn contextvar_type_value() -> Value {
    class_value(&CONTEXTVAR_CLASS)
}

#[inline]
fn token_type_value() -> Value {
    class_value(&TOKEN_CLASS)
}

#[inline]
fn context_type_value() -> Value {
    class_value(&CONTEXT_CLASS)
}

fn build_contextvar_type(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("_contextvars")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    class.set_attr(intern("__new__"), builtin_value(&CONTEXTVAR_NEW_FUNCTION));
    class.set_attr(intern("__init__"), builtin_value(&CONTEXTVAR_INIT_FUNCTION));
    class.set_attr(intern("get"), builtin_value(&CONTEXTVAR_GET_FUNCTION));
    class.set_attr(intern("set"), builtin_value(&CONTEXTVAR_SET_FUNCTION));
    class.set_attr(intern("reset"), builtin_value(&CONTEXTVAR_RESET_FUNCTION));
    class.add_flags(
        ClassFlags::INITIALIZED
            | ClassFlags::HAS_NEW
            | ClassFlags::HAS_INIT
            | ClassFlags::NATIVE_HEAPTYPE,
    );
    register_native_type(class)
}

fn build_token_type(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("_contextvars")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);
    register_native_type(class)
}

fn build_context_type(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("_contextvars")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    class.set_attr(intern("__new__"), builtin_value(&CONTEXT_NEW_FUNCTION));
    class.set_attr(intern("__init__"), builtin_value(&CONTEXT_INIT_FUNCTION));
    class.set_attr(intern("copy"), builtin_value(&CONTEXT_COPY_FUNCTION));
    class.set_attr(intern("run"), builtin_value(&CONTEXT_RUN_FUNCTION));
    class.add_flags(
        ClassFlags::INITIALIZED
            | ClassFlags::HAS_NEW
            | ClassFlags::HAS_INIT
            | ClassFlags::NATIVE_HEAPTYPE,
    );
    register_native_type(class)
}

fn register_native_type(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

fn contextvar_name_property() -> prism_core::intern::InternedString {
    intern("__contextvar_name__")
}

fn contextvar_default_property() -> prism_core::intern::InternedString {
    intern("__contextvar_default__")
}

fn contextvar_has_default_property() -> prism_core::intern::InternedString {
    intern("__contextvar_has_default__")
}

fn contextvar_value_property() -> prism_core::intern::InternedString {
    intern("__contextvar_value__")
}

fn contextvar_has_value_property() -> prism_core::intern::InternedString {
    intern("__contextvar_has_value__")
}

fn token_var_property() -> prism_core::intern::InternedString {
    intern("__contextvar_token_var__")
}

fn token_old_value_property() -> prism_core::intern::InternedString {
    intern("__contextvar_token_old_value__")
}

fn token_had_old_value_property() -> prism_core::intern::InternedString {
    intern("__contextvar_token_had_old_value__")
}

fn token_used_property() -> prism_core::intern::InternedString {
    intern("__contextvar_token_used__")
}

fn contextvar_class_from_value(value: Value) -> Result<&'static PyClassObject, BuiltinError> {
    let Some(class_ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "__new__() argument 1 must be a class".to_string(),
        ));
    };
    if crate::ops::objects::extract_type_id(class_ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "__new__() argument 1 must be a class".to_string(),
        ));
    }

    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    let required_type_id = CONTEXTVAR_CLASS.class_type_id();
    if class.class_type_id() != required_type_id
        && !global_class_bitmap(class.class_id())
            .is_some_and(|bitmap| bitmap.is_subclass_of(required_type_id))
    {
        return Err(BuiltinError::TypeError(
            "__new__() argument 1 must be a subtype of ContextVar".to_string(),
        ));
    }
    Ok(class)
}

fn context_class_from_value(value: Value) -> Result<&'static PyClassObject, BuiltinError> {
    let Some(class_ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "__new__() argument 1 must be a class".to_string(),
        ));
    };
    if crate::ops::objects::extract_type_id(class_ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "__new__() argument 1 must be a class".to_string(),
        ));
    }

    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    let required_type_id = CONTEXT_CLASS.class_type_id();
    if class.class_type_id() != required_type_id
        && !global_class_bitmap(class.class_id())
            .is_some_and(|bitmap| bitmap.is_subclass_of(required_type_id))
    {
        return Err(BuiltinError::TypeError(
            "__new__() argument 1 must be a subtype of Context".to_string(),
        ));
    }
    Ok(class)
}

fn expect_shaped_self(
    args: &[Value],
    fn_name: &'static str,
    type_name: &'static str,
) -> Result<&'static mut ShapedObject, BuiltinError> {
    let Some(self_ptr) = args.first().and_then(|value| value.as_object_ptr()) else {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() requires a {type_name} instance"
        )));
    };
    Ok(unsafe { &mut *(self_ptr as *mut ShapedObject) })
}

fn value_to_string(value: Value, context: &'static str) -> Result<Value, BuiltinError> {
    if value.is_string() {
        return Ok(value);
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!("{context} must be a str")));
    };
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return Err(BuiltinError::TypeError(format!("{context} must be a str")));
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(Value::string(intern(string.as_str())))
}

fn raised_lookup_error(message: &'static str) -> BuiltinError {
    let text = Arc::<str>::from(message);
    let value = create_exception(ExceptionTypeId::LookupError, Some(Arc::clone(&text)));
    BuiltinError::Raised(RuntimeError::raised_exception(
        ExceptionTypeId::LookupError.as_u8() as u16,
        value,
        text,
    ))
}

fn new_context_instance(class: &PyClassObject) -> Value {
    leak_object_value(ShapedObject::new(
        class.class_type_id(),
        Arc::clone(class.instance_shape()),
    ))
}

fn new_token_value(var: Value, had_old_value: bool, old_value: Value) -> Value {
    let mut object = ShapedObject::new(
        TOKEN_CLASS.class_type_id(),
        Arc::clone(TOKEN_CLASS.instance_shape()),
    );
    let registry = shape_registry();
    object.set_property(token_var_property(), var, registry);
    object.set_property(token_old_value_property(), old_value, registry);
    object.set_property(
        token_had_old_value_property(),
        Value::bool(had_old_value),
        registry,
    );
    object.set_property(token_used_property(), Value::bool(false), registry);
    object.set_property(intern("var"), var, registry);
    object.set_property(intern("old_value"), old_value, registry);
    leak_object_value(object)
}

fn contextvar_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "__new__() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let class = contextvar_class_from_value(args[0])?;
    let name = value_to_string(args[1], "ContextVar() argument 'name'")?;
    let default = args.get(2).copied().unwrap_or_else(Value::none);
    let has_default = args.len() == 3;

    let mut object = ShapedObject::new(class.class_type_id(), Arc::clone(class.instance_shape()));
    let registry = shape_registry();
    object.set_property(contextvar_name_property(), name, registry);
    object.set_property(intern("name"), name, registry);
    object.set_property(contextvar_default_property(), default, registry);
    object.set_property(
        contextvar_has_default_property(),
        Value::bool(has_default),
        registry,
    );
    object.set_property(contextvar_value_property(), Value::none(), registry);
    object.set_property(
        contextvar_has_value_property(),
        Value::bool(false),
        registry,
    );
    Ok(leak_object_value(object))
}

fn contextvar_init(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "__init__() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let object = expect_shaped_self(args, "__init__", "ContextVar")?;
    let name = value_to_string(args[1], "ContextVar() argument 'name'")?;
    let default = args.get(2).copied().unwrap_or_else(Value::none);
    let has_default = args.len() == 3;
    let registry = shape_registry();
    object.set_property(contextvar_name_property(), name, registry);
    object.set_property(intern("name"), name, registry);
    object.set_property(contextvar_default_property(), default, registry);
    object.set_property(
        contextvar_has_default_property(),
        Value::bool(has_default),
        registry,
    );
    Ok(Value::none())
}

fn contextvar_get(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "get() takes from 0 to 1 positional arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let object = expect_shaped_self(args, "get", "ContextVar")?;
    if object
        .get_property(contextvar_has_value_property().as_ref())
        .and_then(|value| value.as_bool())
        == Some(true)
    {
        return Ok(object
            .get_property(contextvar_value_property().as_ref())
            .unwrap_or_else(Value::none));
    }

    if let Some(default) = args.get(1).copied() {
        return Ok(default);
    }

    if object
        .get_property(contextvar_has_default_property().as_ref())
        .and_then(|value| value.as_bool())
        == Some(true)
    {
        return Ok(object
            .get_property(contextvar_default_property().as_ref())
            .unwrap_or_else(Value::none));
    }

    Err(raised_lookup_error("ContextVar has no value"))
}

fn contextvar_set(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "set() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let object = expect_shaped_self(args, "set", "ContextVar")?;
    let had_old_value = object
        .get_property(contextvar_has_value_property().as_ref())
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let old_value = object
        .get_property(contextvar_value_property().as_ref())
        .unwrap_or_else(Value::none);
    let token = new_token_value(args[0], had_old_value, old_value);
    let registry = shape_registry();
    object.set_property(contextvar_value_property(), args[1], registry);
    object.set_property(contextvar_has_value_property(), Value::bool(true), registry);
    Ok(token)
}

fn contextvar_reset(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "reset() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let object = expect_shaped_self(args, "reset", "ContextVar")?;
    let Some(token_ptr) = args[1].as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "reset() argument must be a Token".to_string(),
        ));
    };
    let token = unsafe { &mut *(token_ptr as *mut ShapedObject) };
    if token.get_property(token_var_property().as_ref()).is_none() {
        return Err(BuiltinError::TypeError(
            "reset() argument must be a Token".to_string(),
        ));
    }
    if token.get_property(token_var_property().as_ref()) != Some(args[0]) {
        return Err(BuiltinError::ValueError(
            "Token was created by a different ContextVar".to_string(),
        ));
    }
    if token
        .get_property(token_used_property().as_ref())
        .and_then(|value| value.as_bool())
        == Some(true)
    {
        return Err(BuiltinError::ValueError(
            "Token has already been used once".to_string(),
        ));
    }

    let had_old_value = token
        .get_property(token_had_old_value_property().as_ref())
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let old_value = token
        .get_property(token_old_value_property().as_ref())
        .unwrap_or_else(Value::none);

    let registry = shape_registry();
    if had_old_value {
        object.set_property(contextvar_value_property(), old_value, registry);
        object.set_property(contextvar_has_value_property(), Value::bool(true), registry);
    } else {
        object.set_property(contextvar_value_property(), Value::none(), registry);
        object.set_property(
            contextvar_has_value_property(),
            Value::bool(false),
            registry,
        );
    }
    token.set_property(token_used_property(), Value::bool(true), registry);
    Ok(Value::none())
}

fn context_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__new__() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let class = context_class_from_value(args[0])?;
    Ok(new_context_instance(class))
}

fn context_init(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__init__() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }
    let _ = expect_shaped_self(args, "__init__", "Context")?;
    Ok(Value::none())
}

fn context_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "copy() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let _ = expect_shaped_self(args, "copy", "Context")?;
    Ok(new_context_instance(&CONTEXT_CLASS))
}

fn context_run(vm: &mut crate::VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 {
        return Err(BuiltinError::TypeError(format!(
            "run() expected at least 1 argument, got {}",
            args.len().saturating_sub(1)
        )));
    }

    let _ = expect_shaped_self(args, "run", "Context")?;
    invoke_callable_value(vm, args[1], &args[2..]).map_err(BuiltinError::Raised)
}

fn copy_context(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "copy_context() takes 0 positional arguments but {} were given",
            args.len()
        )));
    }
    Ok(new_context_instance(&CONTEXT_CLASS))
}
