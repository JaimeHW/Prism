//! Native `inspect` bootstrap module.
//!
//! Prism relies on CPython's pure-Python `dataclasses` and `pprint` modules
//! during regression testing. Those modules import `inspect`, but Prism does
//! not yet support every parser/runtime feature required by CPython's full
//! `inspect.py`. This native module exposes the small compatibility surface
//! needed by those stdlib imports while keeping the API shape CPython expects.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::StringObject;
use std::sync::{Arc, LazyLock};

static GET_ANNOTATIONS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("inspect.get_annotations"),
        inspect_get_annotations,
    )
});
static SIGNATURE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("inspect.signature"), inspect_signature));

/// Minimal native `inspect` module descriptor.
#[derive(Debug, Clone)]
pub struct InspectModule {
    attrs: Vec<Arc<str>>,
}

impl InspectModule {
    /// Create a new `inspect` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("get_annotations"), Arc::from("signature")],
        }
    }
}

impl Default for InspectModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for InspectModule {
    fn name(&self) -> &str {
        "inspect"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "get_annotations" => Ok(builtin_value(&GET_ANNOTATIONS_FUNCTION)),
            "signature" => Ok(builtin_value(&SIGNATURE_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'inspect' has no attribute '{}'",
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
    Value::object_ptr(Box::into_raw(Box::new(object)) as *const ())
}

fn value_to_string(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        let interned = interned_by_ptr(ptr as *const u8)?;
        return Some(interned.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Some(string.as_str().to_string())
}

fn extract_annotations_mapping(value: Value) -> Option<Value> {
    let ptr = value.as_object_ptr()?;
    let type_id = crate::ops::objects::extract_type_id(ptr);

    match type_id {
        TypeId::TYPE => {
            let class = unsafe { &*(ptr as *const prism_runtime::object::class::PyClassObject) };
            class.get_attr(&intern("__annotations__"))
        }
        TypeId::OBJECT => {
            let shaped = unsafe { &*(ptr as *const ShapedObject) };
            shaped.get_property("__annotations__")
        }
        _ => None,
    }
}

fn copy_dict(value: Value) -> Result<Value, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("__annotations__ must be a dict".to_string()))?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::DICT {
        return Err(BuiltinError::TypeError(
            "__annotations__ must be a dict".to_string(),
        ));
    }

    let dict = unsafe { &*(ptr as *const DictObject) };
    let mut copy = DictObject::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        copy.set(key, value);
    }
    Ok(leak_object_value(copy))
}

fn inspect_get_annotations(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "get_annotations() takes from 1 to 4 positional arguments but {} were given",
            args.len()
        )));
    }

    let eval_str = args
        .get(3)
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    if eval_str {
        return Err(BuiltinError::NotImplemented(
            "inspect.get_annotations(eval_str=True) is not yet supported".to_string(),
        ));
    }

    if let Some(mapping) = extract_annotations_mapping(args[0]) {
        return copy_dict(mapping);
    }

    Ok(leak_object_value(DictObject::new()))
}

fn signature_text(value: Value) -> String {
    if let Some(text) = extract_annotations_mapping(value)
        && let Some(ptr) = text.as_object_ptr()
        && crate::ops::objects::extract_type_id(ptr) == TypeId::DICT
    {
        let dict = unsafe { &*(ptr as *const DictObject) };
        if dict.is_empty() {
            return "()".to_string();
        }
    }

    if let Some(ptr) = value.as_object_ptr()
        && crate::ops::objects::extract_type_id(ptr) == TypeId::TYPE
    {
        let class = unsafe { &*(ptr as *const prism_runtime::object::class::PyClassObject) };
        if let Some(text_signature) = class.get_attr(&intern("__text_signature__"))
            && let Some(text) = value_to_string(text_signature)
            && !text.is_empty()
        {
            return text;
        }
    }

    "()".to_string()
}

fn inspect_signature(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "signature() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    Ok(Value::string(intern(&signature_text(args[0]))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::object::class::PyClassObject;

    #[test]
    fn test_inspect_module_exposes_bootstrap_functions() {
        let module = InspectModule::new();
        assert!(module.get_attr("get_annotations").is_ok());
        assert!(module.get_attr("signature").is_ok());
    }

    #[test]
    fn test_get_annotations_returns_empty_dict_when_missing() {
        let value = inspect_get_annotations(&[Value::bool(true)]).expect("call should succeed");
        let ptr = value
            .as_object_ptr()
            .expect("result should be a dict object");
        let dict = unsafe { &*(ptr as *const DictObject) };
        assert!(dict.is_empty());
    }

    #[test]
    fn test_get_annotations_copies_shape_mapping() {
        let registry = shape_registry();
        let mut annotations = DictObject::new();
        annotations.set(Value::string(intern("value")), Value::string(intern("int")));
        let annotations_value = leak_object_value(annotations);

        let mut shaped = ShapedObject::with_empty_shape(registry.empty_shape());
        shaped.set_property(intern("__annotations__"), annotations_value, registry);
        let object_value = leak_object_value(shaped);

        let copied = inspect_get_annotations(&[object_value]).expect("call should succeed");
        let ptr = copied
            .as_object_ptr()
            .expect("copied annotations should be a dict");
        let dict = unsafe { &*(ptr as *const DictObject) };
        assert_eq!(
            dict.get(Value::string(intern("value"))),
            Some(Value::string(intern("int")))
        );
    }

    #[test]
    fn test_signature_returns_text_signature_when_present() {
        let class = PyClassObject::new_simple(intern("Callable"));
        class.set_attr(
            intern("__text_signature__"),
            Value::string(intern("(x, y=None)")),
        );
        let result = inspect_signature(&[Value::object_ptr(
            Arc::into_raw(Arc::new(class)) as *const ()
        )])
        .expect("signature should succeed");
        assert_eq!(
            interned_by_ptr(result.as_string_object_ptr().unwrap() as *const u8)
                .unwrap()
                .as_str(),
            "(x, y=None)"
        );
    }

    #[test]
    fn test_signature_defaults_to_empty_call_signature() {
        let result =
            inspect_signature(&[Value::int(1).unwrap()]).expect("signature should succeed");
        assert_eq!(
            interned_by_ptr(result.as_string_object_ptr().unwrap() as *const u8)
                .unwrap()
                .as_str(),
            "()"
        );
    }
}
