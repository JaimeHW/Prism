//! Native `_weakref` bootstrap module.
//!
//! CPython's pure-Python stdlib imports `_weakref` very early through modules
//! such as `collections`. Prism provides a native compatibility surface here so
//! early bootstrap does not depend on the much heavier `weakref.py` stack.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::class::{ClassDict, ClassFlags, PyClassObject};
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, global_class_bitmap, global_class_registry, register_global_class, type_new,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

static GETWEAKREFCOUNT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_weakref.getweakrefcount"),
        builtin_getweakrefcount,
    )
});
static GETWEAKREFS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_weakref.getweakrefs"), builtin_getweakrefs)
});
static PROXY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_weakref.proxy"), builtin_proxy));
static REMOVE_DEAD_WEAKREF_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_weakref._remove_dead_weakref"),
        builtin_remove_dead_weakref,
    )
});
static REF_NEW_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_weakref.ReferenceType.__new__"), reference_new)
});
static REF_INIT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_weakref.ReferenceType.__init__"), reference_init)
});
static REF_CALL_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_weakref.ReferenceType.__call__"), reference_call)
});
static REFERENCE_TYPE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_reference_type("ReferenceType"));
static PROXY_TYPE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_placeholder_type("ProxyType"));
static CALLABLE_PROXY_TYPE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_placeholder_type("CallableProxyType"));

/// Native `_weakref` module descriptor.
pub struct WeakRefModule {
    attrs: Vec<Arc<str>>,
    all_value: Value,
}

impl WeakRefModule {
    /// Create a new `_weakref` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__all__"),
                Arc::from("getweakrefcount"),
                Arc::from("getweakrefs"),
                Arc::from("ref"),
                Arc::from("proxy"),
                Arc::from("ReferenceType"),
                Arc::from("ProxyType"),
                Arc::from("CallableProxyType"),
                Arc::from("_remove_dead_weakref"),
            ],
            all_value: export_names_value(),
        }
    }
}

impl Default for WeakRefModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for WeakRefModule {
    fn name(&self) -> &str {
        "_weakref"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all_value),
            "getweakrefcount" => Ok(builtin_value(&GETWEAKREFCOUNT_FUNCTION)),
            "getweakrefs" => Ok(builtin_value(&GETWEAKREFS_FUNCTION)),
            "ref" => Ok(reference_type_value()),
            "proxy" => Ok(builtin_value(&PROXY_FUNCTION)),
            "ReferenceType" => Ok(reference_type_value()),
            "ProxyType" => Ok(proxy_type_value()),
            "CallableProxyType" => Ok(callable_proxy_type_value()),
            "_remove_dead_weakref" => Ok(builtin_value(&REMOVE_DEAD_WEAKREF_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_weakref' has no attribute '{}'",
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
fn leak_object_value<T>(object: T) -> Value {
    let ptr = Box::into_raw(Box::new(object)) as *const ();
    Value::object_ptr(ptr)
}

fn export_names_value() -> Value {
    leak_object_value(TupleObject::from_vec(
        [
            "getweakrefcount",
            "getweakrefs",
            "ref",
            "proxy",
            "ReferenceType",
            "ProxyType",
            "CallableProxyType",
            "_remove_dead_weakref",
        ]
        .into_iter()
        .map(|name| Value::string(intern(name)))
        .collect(),
    ))
}

fn build_placeholder_type(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("_weakref")));
    class.add_flags(ClassFlags::INITIALIZED);
    register_native_type(class)
}

fn build_reference_type(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("_weakref")));
    class.set_attr(intern("__new__"), builtin_value(&REF_NEW_FUNCTION));
    class.set_attr(intern("__init__"), builtin_value(&REF_INIT_FUNCTION));
    class.set_attr(intern("__call__"), builtin_value(&REF_CALL_FUNCTION));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_NEW | ClassFlags::HAS_INIT);
    register_native_type(class)
}

fn register_native_type(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(class.class_type_id());
    bitmap.set_bit(TypeId::OBJECT);

    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

#[inline]
pub(crate) fn reference_type_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&REFERENCE_TYPE_CLASS) as *const ())
}

#[inline]
pub(crate) fn proxy_type_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&PROXY_TYPE_CLASS) as *const ())
}

#[inline]
pub(crate) fn callable_proxy_type_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&CALLABLE_PROXY_TYPE_CLASS) as *const ())
}

fn expect_object_arg(args: &[Value], fn_name: &str) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }

    if args[0].as_object_ptr().is_none() && !args[0].is_string() {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() argument 1 must be an object"
        )));
    }

    Ok(args[0])
}

pub(crate) fn builtin_getweakrefcount(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "getweakrefcount() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    Ok(Value::int(0).expect("zero fits in Value::int"))
}

pub(crate) fn builtin_getweakrefs(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "getweakrefs() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    Ok(leak_object_value(ListObject::new()))
}

fn reference_self(args: &[Value], fn_name: &str) -> Result<*mut ShapedObject, BuiltinError> {
    let Some(self_ptr) = args.first().and_then(|value| value.as_object_ptr()) else {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() requires a weak reference instance"
        )));
    };

    Ok(self_ptr as *mut ShapedObject)
}

fn reference_target_property() -> prism_core::intern::InternedString {
    intern("__weakref_target__")
}

fn reference_callback_property() -> prism_core::intern::InternedString {
    intern("__weakref_callback__")
}

fn reference_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "__new__() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let Some(class_ptr) = args[0].as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "__new__() argument 1 must be a class".to_string(),
        ));
    };
    if crate::ops::objects::extract_type_id(class_ptr) != TypeId::TYPE
        || crate::builtins::builtin_type_object_type_id(class_ptr).is_some()
    {
        return Err(BuiltinError::TypeError(
            "__new__() argument 1 must be a class".to_string(),
        ));
    }

    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    let reference_type_id = REFERENCE_TYPE_CLASS.class_type_id();
    if class.class_type_id() != reference_type_id
        && !global_class_bitmap(class.class_id())
            .is_some_and(|bitmap| bitmap.is_subclass_of(reference_type_id))
    {
        return Err(BuiltinError::TypeError(
            "__new__() argument 1 must be a subtype of ReferenceType".to_string(),
        ));
    }

    let registry = shape_registry();
    let mut object = ShapedObject::new(class.class_type_id(), Arc::clone(class.instance_shape()));
    object.set_property(reference_target_property(), args[1], registry);
    object.set_property(
        reference_callback_property(),
        args.get(2).copied().unwrap_or_else(Value::none),
        registry,
    );

    Ok(leak_object_value(object))
}

fn reference_init(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "__init__() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let self_ptr = reference_self(args, "__init__")?;
    let object = unsafe { &mut *self_ptr };
    let registry = shape_registry();
    object.set_property(reference_target_property(), args[1], registry);
    object.set_property(
        reference_callback_property(),
        args.get(2).copied().unwrap_or_else(Value::none),
        registry,
    );
    Ok(Value::none())
}

fn reference_call(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__call__() takes 1 positional argument but {} were given",
            args.len()
        )));
    }

    let self_ptr = reference_self(args, "__call__")?;
    let object = unsafe { &*self_ptr };
    Ok(object
        .get_property(reference_target_property().as_ref())
        .unwrap_or_else(Value::none))
}

pub(crate) fn builtin_proxy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_object_arg(args, "proxy")
}

pub(crate) fn builtin_remove_dead_weakref(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "_remove_dead_weakref() takes exactly two arguments ({} given)",
            args.len()
        )));
    }

    let Some(dict_ptr) = args[0].as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "_remove_dead_weakref() argument 1 must be dict".to_string(),
        ));
    };

    if crate::ops::objects::extract_type_id(dict_ptr) != TypeId::DICT {
        return Err(BuiltinError::TypeError(
            "_remove_dead_weakref() argument 1 must be dict".to_string(),
        ));
    }

    let dict = unsafe { &mut *(dict_ptr as *mut DictObject) };
    dict.remove(args[1]);
    Ok(Value::none())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
        let ptr = value
            .as_object_ptr()
            .expect("expected builtin function object");
        unsafe { &*(ptr as *const BuiltinFunctionObject) }
    }

    #[test]
    fn test_module_exposes_bootstrap_attributes() {
        let module = WeakRefModule::new();

        assert!(module.get_attr("proxy").unwrap().as_object_ptr().is_some());
        assert!(module.get_attr("ref").unwrap().as_object_ptr().is_some());
        assert!(
            module
                .get_attr("ReferenceType")
                .unwrap()
                .as_object_ptr()
                .is_some()
        );
    }

    #[test]
    fn test_proxy_returns_original_object() {
        let module = WeakRefModule::new();
        let proxy = builtin_from_value(module.get_attr("proxy").expect("proxy should exist"));
        let original = Value::string(intern("ordered-dict-root"));

        assert_eq!(proxy.call(&[original]).unwrap(), original);
    }

    #[test]
    fn test_reference_type_creates_callable_reference_instances() {
        let class_ptr = reference_type_value()
            .as_object_ptr()
            .expect("ReferenceType should be a class object");
        let class = unsafe { &*(class_ptr as *const PyClassObject) };
        let new_builtin = builtin_from_value(
            class
                .get_attr(&intern("__new__"))
                .expect("ReferenceType.__new__ should exist"),
        );
        let call_builtin = builtin_from_value(
            class
                .get_attr(&intern("__call__"))
                .expect("ReferenceType.__call__ should exist"),
        );
        let target = Value::string(intern("cached-module"));

        let instance = new_builtin
            .call(&[reference_type_value(), target])
            .expect("ReferenceType.__new__ should succeed");
        let recalled = call_builtin
            .call(&[instance])
            .expect("ReferenceType instances should be callable");

        assert_eq!(recalled, target);
    }

    #[test]
    fn test_reference_type_is_registered_for_subclass_creation() {
        let base_ptr = reference_type_value()
            .as_object_ptr()
            .expect("ReferenceType should be a class object");
        let base_class = unsafe { &*(base_ptr as *const PyClassObject) };
        let namespace = ClassDict::new();
        let result = type_new(
            intern("KeyedRef"),
            &[base_class.class_id()],
            &namespace,
            global_class_registry(),
        )
        .expect("ReferenceType should support subclass creation");
        register_global_class(result.class.clone(), result.bitmap);

        let subclass_value = Value::object_ptr(Arc::as_ptr(&result.class) as *const ());
        let target = Value::string(intern("bootstrap-entry"));
        let instance = reference_new(&[subclass_value, target])
            .expect("ReferenceType.__new__ should support registered subclasses");
        let instance_ptr = instance
            .as_object_ptr()
            .expect("subclass instance should be heap allocated");

        assert_eq!(
            crate::ops::objects::extract_type_id(instance_ptr),
            result.class.class_type_id()
        );
    }

    #[test]
    fn test_getweakrefs_returns_empty_list() {
        let module = WeakRefModule::new();
        let getweakrefs = builtin_from_value(
            module
                .get_attr("getweakrefs")
                .expect("getweakrefs should exist"),
        );
        let value = getweakrefs.call(&[Value::string(intern("probe"))]).unwrap();
        let ptr = value
            .as_object_ptr()
            .expect("list should be heap allocated");
        let list = unsafe { &*(ptr as *const ListObject) };
        assert!(list.is_empty());
    }

    #[test]
    fn test_remove_dead_weakref_removes_mapping_entry() {
        let module = WeakRefModule::new();
        let remove = builtin_from_value(
            module
                .get_attr("_remove_dead_weakref")
                .expect("remove helper should exist"),
        );

        let mut dict = DictObject::new();
        let key = Value::string(intern("dead"));
        dict.set(key, Value::int(1).unwrap());
        let dict_value = leak_object_value(dict);

        remove.call(&[dict_value, key]).unwrap();

        let dict_ptr = dict_value
            .as_object_ptr()
            .expect("dict should be heap object");
        let dict = unsafe { &*(dict_ptr as *const DictObject) };
        assert!(!dict.contains_key(key));
    }

    #[test]
    fn test_placeholder_types_report_module_name() {
        let value = reference_type_value();
        let ptr = value.as_object_ptr().expect("class should be heap object");
        let class = unsafe { &*(ptr as *const PyClassObject) };
        let module = class
            .get_attr(&intern("__module__"))
            .expect("__module__ should exist");

        let module_name = if module.is_string() {
            let ptr = module
                .as_string_object_ptr()
                .expect("interned string should expose pointer");
            interned_by_ptr(ptr as *const u8)
                .expect("module name should be interned")
                .as_str()
                .to_string()
        } else {
            let ptr = module
                .as_object_ptr()
                .expect("module name should be string object");
            let string = unsafe { &*(ptr as *const StringObject) };
            string.as_str().to_string()
        };

        assert_eq!(module_name, "_weakref");
    }
}
