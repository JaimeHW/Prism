//! Native `inspect` bootstrap module.
//!
//! Prism relies on CPython's pure-Python `dataclasses` and `pprint` modules
//! during regression testing. Those modules import `inspect`, but Prism does
//! not yet support every parser/runtime feature required by CPython's full
//! `inspect.py`. This native module exposes the small compatibility surface
//! needed by those stdlib imports while keeping the API shape CPython expects.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, builtin_getattr_vm};
use crate::import::ModuleObject;
use prism_code::CodeFlags;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::string::StringObject;
use rustc_hash::FxHashSet;
use std::sync::{Arc, LazyLock};

use crate::stdlib::generators::GeneratorObject;

static GET_ANNOTATIONS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("inspect.get_annotations"),
        inspect_get_annotations,
    )
});
static SIGNATURE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("inspect.signature"), inspect_signature));
static ISMODULE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("inspect.ismodule"), inspect_ismodule));
static ISCLASS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("inspect.isclass"), inspect_isclass));
static ISFUNCTION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("inspect.isfunction"), inspect_isfunction)
});
static ISCOROUTINEFUNCTION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("inspect.iscoroutinefunction"),
        inspect_iscoroutinefunction,
    )
});
static ISAWAITABLE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("inspect.isawaitable"), inspect_isawaitable)
});
static ISMETHOD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("inspect.ismethod"), inspect_ismethod));
static ISROUTINE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("inspect.isroutine"), inspect_isroutine));
static ISMETHODDESCRIPTOR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("inspect.ismethoddescriptor"),
        inspect_ismethoddescriptor,
    )
});
static ISMETHODWRAPPER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("inspect.ismethodwrapper"),
        inspect_ismethodwrapper,
    )
});
static ISCODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("inspect.iscode"), inspect_iscode));
static ISFRAME_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("inspect.isframe"), inspect_isframe));
static ISTRACEBACK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("inspect.istraceback"), inspect_istraceback)
});
static GETMODULE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("inspect.getmodule"), inspect_getmodule)
});
static GETFILE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("inspect.getfile"), inspect_getfile));
static GETSOURCEFILE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("inspect.getsourcefile"), inspect_getsourcefile)
});
static UNWRAP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("inspect.unwrap"), inspect_unwrap));

const INSPECT_CODE_FLAGS: [(&str, i64); 8] = [
    ("CO_VARARGS", 0x0004),
    ("CO_VARKEYWORDS", 0x0008),
    ("CO_NESTED", 0x0010),
    ("CO_GENERATOR", 0x0020),
    ("CO_NOFREE", 0x0040),
    ("CO_COROUTINE", 0x0080),
    ("CO_ITERABLE_COROUTINE", 0x0100),
    ("CO_ASYNC_GENERATOR", 0x0200),
];

/// Minimal native `inspect` module descriptor.
#[derive(Debug, Clone)]
pub struct InspectModule {
    attrs: Vec<Arc<str>>,
}

impl InspectModule {
    /// Create a new `inspect` module descriptor.
    pub fn new() -> Self {
        let mut attrs = vec![
            Arc::from("get_annotations"),
            Arc::from("signature"),
            Arc::from("ismodule"),
            Arc::from("isclass"),
            Arc::from("isfunction"),
            Arc::from("iscoroutinefunction"),
            Arc::from("isawaitable"),
            Arc::from("ismethod"),
            Arc::from("isroutine"),
            Arc::from("ismethoddescriptor"),
            Arc::from("ismethodwrapper"),
            Arc::from("iscode"),
            Arc::from("isframe"),
            Arc::from("istraceback"),
            Arc::from("getmodule"),
            Arc::from("getfile"),
            Arc::from("getsourcefile"),
            Arc::from("unwrap"),
        ];
        attrs.extend(
            INSPECT_CODE_FLAGS
                .iter()
                .map(|(name, _)| Arc::<str>::from(*name)),
        );
        Self { attrs }
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
            "ismodule" => Ok(builtin_value(&ISMODULE_FUNCTION)),
            "isclass" => Ok(builtin_value(&ISCLASS_FUNCTION)),
            "isfunction" => Ok(builtin_value(&ISFUNCTION_FUNCTION)),
            "iscoroutinefunction" => Ok(builtin_value(&ISCOROUTINEFUNCTION_FUNCTION)),
            "isawaitable" => Ok(builtin_value(&ISAWAITABLE_FUNCTION)),
            "ismethod" => Ok(builtin_value(&ISMETHOD_FUNCTION)),
            "isroutine" => Ok(builtin_value(&ISROUTINE_FUNCTION)),
            "ismethoddescriptor" => Ok(builtin_value(&ISMETHODDESCRIPTOR_FUNCTION)),
            "ismethodwrapper" => Ok(builtin_value(&ISMETHODWRAPPER_FUNCTION)),
            "iscode" => Ok(builtin_value(&ISCODE_FUNCTION)),
            "isframe" => Ok(builtin_value(&ISFRAME_FUNCTION)),
            "istraceback" => Ok(builtin_value(&ISTRACEBACK_FUNCTION)),
            "getmodule" => Ok(builtin_value(&GETMODULE_FUNCTION)),
            "getfile" => Ok(builtin_value(&GETFILE_FUNCTION)),
            "getsourcefile" => Ok(builtin_value(&GETSOURCEFILE_FUNCTION)),
            "unwrap" => Ok(builtin_value(&UNWRAP_FUNCTION)),
            name => inspect_constant(name).ok_or_else(|| {
                ModuleError::AttributeError(format!("module 'inspect' has no attribute '{}'", name))
            }),
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

fn inspect_constant(name: &str) -> Option<Value> {
    INSPECT_CODE_FLAGS
        .iter()
        .find_map(|(constant_name, value)| {
            (*constant_name == name).then(|| Value::int(*value).unwrap())
        })
}

fn exact_arity(name: &str, args: &[Value], expected: usize) -> Result<Value, BuiltinError> {
    if args.len() != expected {
        return Err(BuiltinError::TypeError(format!(
            "{name}() takes exactly {expected} argument{} ({} given)",
            if expected == 1 { "" } else { "s" },
            args.len()
        )));
    }
    Ok(args[0])
}

#[inline]
fn object_type_id(value: Value) -> Option<TypeId> {
    value
        .as_object_ptr()
        .map(crate::ops::objects::extract_type_id)
}

#[inline]
fn is_module_type(type_id: TypeId) -> bool {
    type_id == TypeId::MODULE || type_id == TypeId::MODULE_OBJECT
}

#[inline]
fn predicate_result(
    name: &str,
    args: &[Value],
    predicate: impl FnOnce(Value) -> bool,
) -> Result<Value, BuiltinError> {
    Ok(Value::bool(predicate(exact_arity(name, args, 1)?)))
}

#[inline]
fn attribute_value(
    vm: &mut crate::VirtualMachine,
    target: Value,
    name: &str,
) -> Result<Option<Value>, BuiltinError> {
    let value = builtin_getattr_vm(vm, &[target, Value::string(intern(name)), Value::none()])?;
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

fn resolve_module_for_value(
    vm: &mut crate::VirtualMachine,
    value: Value,
) -> Result<Option<Arc<ModuleObject>>, BuiltinError> {
    if let Some(ptr) = value.as_object_ptr() {
        if is_module_type(crate::ops::objects::extract_type_id(ptr)) {
            return Ok(vm.module_from_globals_ptr(ptr));
        }
    }

    if let Some(ptr) = value.as_object_ptr()
        && crate::ops::objects::extract_type_id(ptr) == TypeId::METHOD
        && let Some(function) = attribute_value(vm, value, "__func__")?
    {
        return resolve_module_for_value(vm, function);
    }

    let Some(module_name_value) = attribute_value(vm, value, "__module__")? else {
        return Ok(None);
    };
    let Some(module_name) = value_to_string(module_name_value) else {
        return Ok(None);
    };

    if let Some(current_module) = vm.current_module_cloned()
        && current_module.name() == module_name
    {
        return Ok(Some(current_module));
    }

    Ok(vm.import_module_named(&module_name).ok())
}

#[inline]
fn module_value(module: &Arc<ModuleObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(module) as *const ())
}

fn source_file_value(
    vm: &mut crate::VirtualMachine,
    value: Value,
) -> Result<Option<Value>, BuiltinError> {
    if let Some(ptr) = value.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            type_id if is_module_type(type_id) => {
                if let Some(path) = attribute_value(vm, value, "__file__")? {
                    return Ok(Some(path));
                }
            }
            TypeId::FUNCTION => {
                let function = unsafe { &*(ptr as *const FunctionObject) };
                return Ok(Some(Value::string(intern(function.code.filename.as_ref()))));
            }
            TypeId::METHOD => {
                let method = unsafe { &*(ptr as *const BoundMethod) };
                return source_file_value(vm, method.function());
            }
            _ => {}
        }
    }

    let Some(module) = resolve_module_for_value(vm, value)? else {
        return Ok(None);
    };
    Ok(module.get_attr("__file__").filter(|path| !path.is_none()))
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

fn inspect_ismodule(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("ismodule", args, |value| {
        object_type_id(value).is_some_and(is_module_type)
    })
}

fn inspect_isclass(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("isclass", args, |value| {
        object_type_id(value).is_some_and(|type_id| type_id == TypeId::TYPE)
    })
}

fn inspect_isfunction(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("isfunction", args, |value| {
        object_type_id(value).is_some_and(|type_id| type_id == TypeId::FUNCTION)
    })
}

fn inspect_iscoroutinefunction(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("iscoroutinefunction", args, |value| {
        let Some(ptr) = value.as_object_ptr() else {
            return false;
        };

        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::FUNCTION => unsafe { &*(ptr as *const FunctionObject) }
                .code
                .flags
                .contains(CodeFlags::COROUTINE),
            TypeId::METHOD => {
                let method = unsafe { &*(ptr as *const BoundMethod) };
                let Some(function_ptr) = method.function().as_object_ptr() else {
                    return false;
                };
                if crate::ops::objects::extract_type_id(function_ptr) != TypeId::FUNCTION {
                    return false;
                }
                unsafe { &*(function_ptr as *const FunctionObject) }
                    .code
                    .flags
                    .contains(CodeFlags::COROUTINE)
            }
            _ => false,
        }
    })
}

fn inspect_isawaitable(
    vm: &mut crate::VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let value = exact_arity("isawaitable", args, 1)?;

    if let Some(generator) = GeneratorObject::from_value(value)
        && (generator.is_coroutine() || generator.is_async())
    {
        return Ok(Value::bool(true));
    }

    Ok(Value::bool(
        attribute_value(vm, value, "__await__")?.is_some(),
    ))
}

fn inspect_ismethod(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("ismethod", args, |value| {
        object_type_id(value).is_some_and(|type_id| type_id == TypeId::METHOD)
    })
}

fn inspect_isroutine(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("isroutine", args, |value| {
        object_type_id(value).is_some_and(|type_id| {
            matches!(
                type_id,
                TypeId::FUNCTION
                    | TypeId::METHOD
                    | TypeId::BUILTIN_FUNCTION
                    | TypeId::WRAPPER_DESCRIPTOR
                    | TypeId::METHOD_DESCRIPTOR
                    | TypeId::CLASSMETHOD_DESCRIPTOR
                    | TypeId::METHOD_WRAPPER
            )
        })
    })
}

fn inspect_ismethoddescriptor(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("ismethoddescriptor", args, |value| {
        object_type_id(value).is_some_and(|type_id| {
            matches!(
                type_id,
                TypeId::WRAPPER_DESCRIPTOR
                    | TypeId::METHOD_DESCRIPTOR
                    | TypeId::CLASSMETHOD_DESCRIPTOR
            )
        })
    })
}

fn inspect_ismethodwrapper(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("ismethodwrapper", args, |value| {
        object_type_id(value).is_some_and(|type_id| type_id == TypeId::METHOD_WRAPPER)
    })
}

fn inspect_iscode(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("iscode", args, |value| {
        object_type_id(value).is_some_and(|type_id| type_id == TypeId::CODE)
    })
}

fn inspect_isframe(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("isframe", args, |value| {
        object_type_id(value).is_some_and(|type_id| type_id == TypeId::FRAME)
    })
}

fn inspect_istraceback(args: &[Value]) -> Result<Value, BuiltinError> {
    predicate_result("istraceback", args, |value| {
        object_type_id(value).is_some_and(|type_id| type_id == TypeId::TRACEBACK)
    })
}

fn inspect_getmodule(
    vm: &mut crate::VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let value = exact_arity("getmodule", args, 1)?;
    if object_type_id(value).is_some_and(is_module_type) {
        return Ok(value);
    }

    Ok(resolve_module_for_value(vm, value)?
        .map(|module| module_value(&module))
        .unwrap_or_else(Value::none))
}

fn inspect_getfile(vm: &mut crate::VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(source_file_value(vm, exact_arity("getfile", args, 1)?)?.unwrap_or_else(Value::none))
}

fn inspect_getsourcefile(
    vm: &mut crate::VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    Ok(source_file_value(vm, exact_arity("getsourcefile", args, 1)?)?.unwrap_or_else(Value::none))
}

fn inspect_unwrap(vm: &mut crate::VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let mut current = exact_arity("unwrap", args, 1)?;
    let mut seen = FxHashSet::default();

    loop {
        if !seen.insert(current.to_bits()) {
            return Err(BuiltinError::ValueError(
                "wrapper loop when unwrapping object".to_string(),
            ));
        }

        let Some(next) = attribute_value(vm, current, "__wrapped__")? else {
            return Ok(current);
        };
        current = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_code::CodeObject;
    use prism_runtime::object::class::PyClassObject;
    use prism_runtime::object::descriptor::BoundMethod;
    use prism_runtime::object::shape::shape_registry;

    #[test]
    fn test_inspect_module_exposes_bootstrap_functions() {
        let module = InspectModule::new();
        assert!(module.get_attr("get_annotations").is_ok());
        assert!(module.get_attr("signature").is_ok());
        assert!(module.get_attr("ismodule").is_ok());
        assert!(module.get_attr("isclass").is_ok());
        assert!(module.get_attr("isfunction").is_ok());
        assert!(module.get_attr("iscoroutinefunction").is_ok());
        assert!(module.get_attr("isawaitable").is_ok());
        assert!(module.get_attr("ismethod").is_ok());
        assert!(module.get_attr("isroutine").is_ok());
        assert!(module.get_attr("ismethoddescriptor").is_ok());
        assert!(module.get_attr("ismethodwrapper").is_ok());
        assert!(module.get_attr("getmodule").is_ok());
        assert!(module.get_attr("getfile").is_ok());
        assert!(module.get_attr("getsourcefile").is_ok());
        assert!(module.get_attr("unwrap").is_ok());
        assert_eq!(
            module
                .get_attr("CO_GENERATOR")
                .expect("CO_GENERATOR should exist")
                .as_int(),
            Some(0x20)
        );
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

    #[test]
    fn test_inspect_predicates_classify_module_function_and_method_values() {
        let module_value =
            Value::object_ptr(Box::into_raw(Box::new(ModuleObject::new("mod"))) as *const ());
        let class_value = Value::object_ptr(Arc::into_raw(Arc::new(PyClassObject::new_simple(
            intern("Demo"),
        ))) as *const ());
        let function = Arc::new(FunctionObject::new(
            Arc::new(CodeObject::new("demo", "demo.py")),
            Arc::from("demo"),
            None,
            None,
        ));
        let function_value = Value::object_ptr(Arc::into_raw(Arc::clone(&function)) as *const ());
        let method_value = Value::object_ptr(Box::into_raw(Box::new(BoundMethod::new(
            function_value,
            Value::none(),
        ))) as *const ());

        assert_eq!(
            inspect_ismodule(&[module_value]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(inspect_isclass(&[class_value]).unwrap(), Value::bool(true));
        assert_eq!(
            inspect_isfunction(&[function_value]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            inspect_ismethod(&[method_value]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            inspect_isroutine(&[function_value]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            inspect_isroutine(&[method_value]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            inspect_isroutine(&[module_value]).unwrap(),
            Value::bool(false)
        );
    }

    #[test]
    fn test_inspect_isawaitable_recognizes_coroutine_generators_and_await_protocol() {
        let mut vm = crate::VirtualMachine::new();
        let mut code = CodeObject::new("coro", "demo.py");
        code.flags = CodeFlags::COROUTINE;
        let coroutine_value = leak_object_value(GeneratorObject::from_code(Arc::new(code)));

        assert_eq!(
            inspect_isawaitable(&mut vm, &[coroutine_value]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            inspect_isawaitable(&mut vm, &[Value::int(1).unwrap()]).unwrap(),
            Value::bool(false)
        );

        let registry = shape_registry();
        let mut awaitable = ShapedObject::with_empty_shape(registry.empty_shape());
        awaitable.set_property(intern("__await__"), Value::bool(true), registry);
        let awaitable_value = leak_object_value(awaitable);

        assert_eq!(
            inspect_isawaitable(&mut vm, &[awaitable_value]).unwrap(),
            Value::bool(true)
        );
    }

    #[test]
    fn test_inspect_getfile_uses_function_code_filename() {
        let mut vm = crate::VirtualMachine::new();
        let function = Arc::new(FunctionObject::new(
            Arc::new(CodeObject::new("demo", "demo_source.py")),
            Arc::from("demo"),
            None,
            None,
        ));
        let function_value = Value::object_ptr(Arc::into_raw(function) as *const ());

        let result = inspect_getfile(&mut vm, &[function_value]).expect("getfile should succeed");
        assert_eq!(
            interned_by_ptr(result.as_string_object_ptr().unwrap() as *const u8)
                .unwrap()
                .as_str(),
            "demo_source.py"
        );
    }
}
