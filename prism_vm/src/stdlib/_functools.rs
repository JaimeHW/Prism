//! Native `_functools` accelerator module.
//!
//! CPython's `functools.py` expects `_functools.partial` to be a native type.
//! In particular, the pure-Python fallback flattens any callable that merely
//! exposes `func` and `args` attributes, which is observably wrong for dynamic
//! callables such as `unittest.mock.Mock`. Prism provides a compact native
//! `partial` implementation so CPython's stdlib uses the accelerator path.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject, runtime_error_to_builtin_error};
use crate::ops::calls::{invoke_callable_value_with_keywords, value_supports_call_protocol};
use crate::ops::objects::{dict_storage_ref_from_ptr, extract_type_id};
use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, global_class_bitmap, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use std::sync::{Arc, LazyLock};

static PARTIAL_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_partial_class);
static PARTIAL_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("_functools.partial.__init__"), partial_init)
});
static PARTIAL_CALL_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("_functools.partial.__call__"), partial_call)
});

/// Native `_functools` module descriptor.
#[derive(Debug, Clone)]
pub struct FunctoolsNativeModule {
    attrs: Vec<Arc<str>>,
}

impl FunctoolsNativeModule {
    /// Create a new `_functools` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("partial")],
        }
    }
}

impl Default for FunctoolsNativeModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for FunctoolsNativeModule {
    fn name(&self) -> &str {
        "_functools"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "partial" => Ok(partial_class_value()),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_functools' has no attribute '{}'",
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
fn partial_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&PARTIAL_CLASS) as *const ())
}

fn build_partial_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("partial"));
    class.set_attr(intern("__module__"), Value::string(intern("functools")));
    class.set_attr(intern("__qualname__"), Value::string(intern("partial")));
    class.set_attr(
        intern("__doc__"),
        Value::string(intern(
            "partial(func, /, *args, **keywords) - new function with partial application",
        )),
    );
    class.set_attr(intern("__init__"), builtin_value(&PARTIAL_INIT_METHOD));
    class.set_attr(intern("__call__"), builtin_value(&PARTIAL_CALL_METHOD));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_INIT | ClassFlags::NATIVE_HEAPTYPE);

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    let class = Arc::new(class);
    register_global_class(Arc::clone(&class), bitmap);
    class
}

#[inline]
fn partial_func_attr() -> InternedString {
    intern("func")
}

#[inline]
fn partial_args_attr() -> InternedString {
    intern("args")
}

#[inline]
fn partial_keywords_attr() -> InternedString {
    intern("keywords")
}

#[inline]
fn partial_type_id() -> TypeId {
    PARTIAL_CLASS.class_type_id()
}

#[inline]
fn is_partial_type(type_id: TypeId) -> bool {
    type_id == partial_type_id()
        || (type_id.raw() >= TypeId::FIRST_USER_TYPE
            && global_class_bitmap(ClassId(type_id.raw()))
                .is_some_and(|bitmap| bitmap.is_subclass_of(partial_type_id())))
}

#[inline]
fn is_partial_value(value: Value) -> bool {
    value
        .as_object_ptr()
        .is_some_and(|ptr| is_partial_type(extract_type_id(ptr)))
}

fn partial_object(
    value: Value,
    context: &'static str,
) -> Result<&'static ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} requires a functools.partial object"
        )));
    };
    if !is_partial_type(extract_type_id(ptr)) {
        return Err(BuiltinError::TypeError(format!(
            "{context} requires a functools.partial object"
        )));
    }
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn partial_object_mut(
    value: Value,
    context: &'static str,
) -> Result<&'static mut ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} requires a functools.partial object"
        )));
    };
    if !is_partial_type(extract_type_id(ptr)) {
        return Err(BuiltinError::TypeError(format!(
            "{context} requires a functools.partial object"
        )));
    }
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn partial_attr(value: Value, name: InternedString) -> Result<Value, BuiltinError> {
    partial_object(value, "partial state")?
        .get_property_interned(&name)
        .ok_or_else(|| {
            BuiltinError::TypeError("invalid functools.partial object state".to_string())
        })
}

fn dict_ref(value: Value, context: &'static str) -> Result<&'static DictObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!("{context} must be a dict")));
    };
    dict_storage_ref_from_ptr(ptr)
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a dict")))
}

fn tuple_ref(value: Value, context: &'static str) -> Result<&'static TupleObject, BuiltinError> {
    value_as_tuple_ref(value)
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a tuple")))
}

fn alloc_value<T: prism_runtime::Trace>(
    vm: &mut VirtualMachine,
    object: T,
    context: &'static str,
) -> Result<Value, BuiltinError> {
    vm.allocator()
        .alloc(object)
        .map(|ptr| Value::object_ptr(ptr as *const ()))
        .ok_or_else(|| {
            BuiltinError::Raised(crate::error::RuntimeError::internal(format!(
                "out of memory: failed to allocate {context}"
            )))
        })
}

fn copy_keyword_dict(dict: &DictObject, capacity_hint: usize) -> DictObject {
    let mut copied = DictObject::with_capacity(dict.len() + capacity_hint);
    copied.update(dict);
    copied
}

fn overlay_keyword_args(dict: &mut DictObject, keywords: &[(&str, Value)]) {
    for &(name, value) in keywords {
        dict.set(Value::string(intern(name)), value);
    }
}

fn keyword_entries_from_dict(dict: &DictObject) -> Result<Vec<(String, Value)>, BuiltinError> {
    let mut entries = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let Some(key_string) = value_as_string_ref(key) else {
            return Err(BuiltinError::TypeError(
                "functools.partial keywords must be strings".to_string(),
            ));
        };
        entries.push((key_string.as_str().to_string(), value));
    }
    Ok(entries)
}

fn partial_init(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.len() < 2 {
        return Err(BuiltinError::TypeError(format!(
            "type 'functools.partial' takes at least one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let object = partial_object_mut(args[0], "partial.__init__")?;
    let mut function = args[1];
    if !value_supports_call_protocol(function) {
        return Err(BuiltinError::TypeError(
            "the first argument must be callable".to_string(),
        ));
    }

    let mut frozen_args = Vec::new();
    let mut frozen_keywords = if is_partial_value(function) {
        let inner_args = tuple_ref(partial_attr(function, partial_args_attr())?, "partial.args")?;
        frozen_args.reserve(inner_args.len() + args.len().saturating_sub(2));
        frozen_args.extend_from_slice(inner_args.as_slice());

        let inner_keywords = dict_ref(
            partial_attr(function, partial_keywords_attr())?,
            "partial.keywords",
        )?;
        let copied = copy_keyword_dict(inner_keywords, keywords.len());
        function = partial_attr(function, partial_func_attr())?;
        copied
    } else {
        frozen_args.reserve(args.len().saturating_sub(2));
        DictObject::with_capacity(keywords.len())
    };

    frozen_args.extend_from_slice(&args[2..]);
    overlay_keyword_args(&mut frozen_keywords, keywords);

    let args_value = alloc_value(
        vm,
        TupleObject::from_vec(frozen_args),
        "functools.partial args tuple",
    )?;
    let keywords_value = alloc_value(vm, frozen_keywords, "functools.partial keyword dict")?;

    let registry = shape_registry();
    object.set_property(partial_func_attr(), function, registry);
    object.set_property(partial_args_attr(), args_value, registry);
    object.set_property(partial_keywords_attr(), keywords_value, registry);
    Ok(Value::none())
}

fn partial_call(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "descriptor '__call__' requires a functools.partial object".to_string(),
        ));
    }

    let self_value = args[0];
    let function = partial_attr(self_value, partial_func_attr())?;
    let stored_args = tuple_ref(
        partial_attr(self_value, partial_args_attr())?,
        "partial.args",
    )?;
    let stored_keywords = dict_ref(
        partial_attr(self_value, partial_keywords_attr())?,
        "partial.keywords",
    )?;

    let mut call_args = Vec::with_capacity(stored_args.len() + args.len().saturating_sub(1));
    call_args.extend_from_slice(stored_args.as_slice());
    call_args.extend_from_slice(&args[1..]);

    if stored_keywords.is_empty() && keywords.is_empty() {
        return invoke_callable_value_with_keywords(vm, function, &call_args, &[])
            .map_err(runtime_error_to_builtin_error);
    }

    let mut keyword_entries = keyword_entries_from_dict(stored_keywords)?;
    for &(name, value) in keywords {
        if let Some((_, existing)) = keyword_entries
            .iter_mut()
            .find(|(existing_name, _)| existing_name == name)
        {
            *existing = value;
        } else {
            keyword_entries.push((name.to_string(), value));
        }
    }
    let keyword_refs = keyword_entries
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect::<Vec<_>>();

    invoke_callable_value_with_keywords(vm, function, &call_args, &keyword_refs)
        .map_err(runtime_error_to_builtin_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exposes_partial_type() {
        let module = FunctoolsNativeModule::new();
        assert_eq!(module.name(), "_functools");
        assert!(module.get_attr("partial").is_ok());
        assert_eq!(module.dir(), vec![Arc::from("partial")]);
    }

    #[test]
    fn test_partial_type_is_registered_native_heap_type() {
        let ptr = partial_class_value()
            .as_object_ptr()
            .expect("partial type should be an object");
        assert_eq!(extract_type_id(ptr), TypeId::TYPE);
        assert!(PARTIAL_CLASS.is_native_heaptype());
        assert!(global_class_bitmap(PARTIAL_CLASS.class_id()).is_some());
    }
}
