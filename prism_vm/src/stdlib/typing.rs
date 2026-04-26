//! Minimal native `typing` module for CPython stdlib bootstrap.
//!
//! Prism does not yet implement CPython's `_typing` accelerator module, but a
//! number of stdlib modules import `typing` only for annotations and no-op
//! decorators. This module provides a compact runtime surface that keeps those
//! imports working without pretending to implement the full typing semantics.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::calls::value_supports_call_protocol;
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::allocation_context::alloc_static_value;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock, Mutex};

const MARKER_EXPORTS: &[&str] = &[
    "Annotated",
    "Any",
    "AnyStr",
    "BinaryIO",
    "Callable",
    "ClassVar",
    "Concatenate",
    "Dict",
    "Final",
    "FrozenSet",
    "IO",
    "Iterable",
    "Iterator",
    "List",
    "Literal",
    "LiteralString",
    "Match",
    "Never",
    "NoReturn",
    "NotRequired",
    "Optional",
    "ParamSpecArgs",
    "ParamSpecKwargs",
    "Pattern",
    "Required",
    "Self",
    "Sequence",
    "Set",
    "TextIO",
    "Tuple",
    "Type",
    "TypeAlias",
    "TypeAliasType",
    "TypeGuard",
    "Union",
    "Unpack",
];

const DECORATOR_EXPORTS: &[&str] = &[
    "final",
    "no_type_check",
    "no_type_check_decorator",
    "overload",
    "override",
    "runtime_checkable",
];

const FUNCTION_EXPORTS: &[&str] = &[
    "assert_type",
    "cast",
    "dataclass_transform",
    "get_args",
    "get_origin",
    "reveal_type",
];

const FACTORY_EXPORTS: &[&str] = &["NewType", "ParamSpec", "TypeVar", "TypeVarTuple"];
const CLASS_EXPORTS: &[&str] = &["Protocol"];

static TYPING_FORM_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_typing_form_class);
static PROTOCOL_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_typing_base_class("Protocol"));
static MARKER_CACHE: LazyLock<Mutex<FxHashMap<&'static str, Value>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

static TYPING_FORM_GETITEM_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("typing._TypingForm.__getitem__"),
        typing_form_getitem,
    )
});
static IDENTITY_DECORATOR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("typing._identity_decorator"),
        typing_identity_decorator,
    )
});
static CAST_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("typing.cast"), typing_cast));
static ASSERT_TYPE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("typing.assert_type"), typing_assert_type)
});
static REVEAL_TYPE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("typing.reveal_type"), typing_reveal_type)
});
static DATACLASS_TRANSFORM_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("typing.dataclass_transform"),
        typing_dataclass_transform,
    )
});
static GET_ORIGIN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("typing.get_origin"), typing_get_origin));
static GET_ARGS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("typing.get_args"), typing_get_args));
static TYPEVAR_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("typing.TypeVar"), typing_typevar));
static PARAMSPEC_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("typing.ParamSpec"), typing_paramspec)
});
static TYPEVARTUPLE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("typing.TypeVarTuple"), typing_typevartuple)
});
static NEWTYPE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("typing.NewType"), typing_newtype));

/// Minimal native `typing` module descriptor.
#[derive(Debug, Clone)]
pub struct TypingModule {
    attrs: Vec<Arc<str>>,
    all_value: Value,
}

impl TypingModule {
    /// Create a new `typing` module.
    pub fn new() -> Self {
        let mut names = vec![Arc::from("__all__"), Arc::from("TYPE_CHECKING")];
        names.extend(MARKER_EXPORTS.iter().copied().map(Arc::from));
        names.extend(DECORATOR_EXPORTS.iter().copied().map(Arc::from));
        names.extend(FUNCTION_EXPORTS.iter().copied().map(Arc::from));
        names.extend(FACTORY_EXPORTS.iter().copied().map(Arc::from));
        names.extend(CLASS_EXPORTS.iter().copied().map(Arc::from));

        Self {
            attrs: names,
            all_value: export_names_value(),
        }
    }
}

impl Default for TypingModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TypingModule {
    fn name(&self) -> &str {
        "typing"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all_value),
            "TYPE_CHECKING" => Ok(Value::bool(false)),

            "Protocol" => Ok(class_value(&PROTOCOL_CLASS)),

            "final"
            | "no_type_check"
            | "no_type_check_decorator"
            | "overload"
            | "override"
            | "runtime_checkable" => Ok(builtin_value(&IDENTITY_DECORATOR_FUNCTION)),
            "cast" => Ok(builtin_value(&CAST_FUNCTION)),
            "assert_type" => Ok(builtin_value(&ASSERT_TYPE_FUNCTION)),
            "reveal_type" => Ok(builtin_value(&REVEAL_TYPE_FUNCTION)),
            "dataclass_transform" => Ok(builtin_value(&DATACLASS_TRANSFORM_FUNCTION)),
            "get_origin" => Ok(builtin_value(&GET_ORIGIN_FUNCTION)),
            "get_args" => Ok(builtin_value(&GET_ARGS_FUNCTION)),
            "TypeVar" => Ok(builtin_value(&TYPEVAR_FUNCTION)),
            "ParamSpec" => Ok(builtin_value(&PARAMSPEC_FUNCTION)),
            "TypeVarTuple" => Ok(builtin_value(&TYPEVARTUPLE_FUNCTION)),
            "NewType" => Ok(builtin_value(&NEWTYPE_FUNCTION)),

            "Annotated" => Ok(cached_marker_value("Annotated")),
            "Any" => Ok(cached_marker_value("Any")),
            "AnyStr" => Ok(cached_marker_value("AnyStr")),
            "BinaryIO" => Ok(cached_marker_value("BinaryIO")),
            "Callable" => Ok(cached_marker_value("Callable")),
            "ClassVar" => Ok(cached_marker_value("ClassVar")),
            "Concatenate" => Ok(cached_marker_value("Concatenate")),
            "Dict" => Ok(cached_marker_value("Dict")),
            "Final" => Ok(cached_marker_value("Final")),
            "FrozenSet" => Ok(cached_marker_value("FrozenSet")),
            "IO" => Ok(cached_marker_value("IO")),
            "Iterable" => Ok(cached_marker_value("Iterable")),
            "Iterator" => Ok(cached_marker_value("Iterator")),
            "List" => Ok(cached_marker_value("List")),
            "Literal" => Ok(cached_marker_value("Literal")),
            "LiteralString" => Ok(cached_marker_value("LiteralString")),
            "Match" => Ok(cached_marker_value("Match")),
            "Never" => Ok(cached_marker_value("Never")),
            "NoReturn" => Ok(cached_marker_value("NoReturn")),
            "NotRequired" => Ok(cached_marker_value("NotRequired")),
            "Optional" => Ok(cached_marker_value("Optional")),
            "ParamSpecArgs" => Ok(cached_marker_value("ParamSpecArgs")),
            "ParamSpecKwargs" => Ok(cached_marker_value("ParamSpecKwargs")),
            "Pattern" => Ok(cached_marker_value("Pattern")),
            "Required" => Ok(cached_marker_value("Required")),
            "Self" => Ok(cached_marker_value("Self")),
            "Sequence" => Ok(cached_marker_value("Sequence")),
            "Set" => Ok(cached_marker_value("Set")),
            "TextIO" => Ok(cached_marker_value("TextIO")),
            "Tuple" => Ok(cached_marker_value("Tuple")),
            "Type" => Ok(cached_marker_value("Type")),
            "TypeAlias" => Ok(cached_marker_value("TypeAlias")),
            "TypeAliasType" => Ok(cached_marker_value("TypeAliasType")),
            "TypeGuard" => Ok(cached_marker_value("TypeGuard")),
            "Union" => Ok(cached_marker_value("Union")),
            "Unpack" => Ok(cached_marker_value("Unpack")),

            _ => Err(ModuleError::AttributeError(format!(
                "module 'typing' has no attribute '{}'",
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

fn export_names_value() -> Value {
    let mut names = Vec::with_capacity(
        1 + MARKER_EXPORTS.len()
            + DECORATOR_EXPORTS.len()
            + FUNCTION_EXPORTS.len()
            + FACTORY_EXPORTS.len()
            + CLASS_EXPORTS.len(),
    );
    names.extend(MARKER_EXPORTS.iter().copied());
    names.extend(DECORATOR_EXPORTS.iter().copied());
    names.extend(FUNCTION_EXPORTS.iter().copied());
    names.extend(FACTORY_EXPORTS.iter().copied());
    names.extend(CLASS_EXPORTS.iter().copied());

    crate::alloc_managed_value(TupleObject::from_vec(
        names
            .into_iter()
            .map(|name| Value::string(intern(name)))
            .collect(),
    ))
}

fn build_typing_form_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("_TypingForm"));
    class.set_attr(
        intern("__getitem__"),
        builtin_value(&TYPING_FORM_GETITEM_FUNCTION),
    );
    class.set_attr(intern("__module__"), Value::string(intern("typing")));
    class.set_attr(intern("__qualname__"), Value::string(intern("_TypingForm")));
    register_typing_class(class)
}

fn build_typing_base_class(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("typing")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    register_typing_class(class)
}

fn register_typing_class(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

fn cached_marker_value(name: &'static str) -> Value {
    let mut cache = MARKER_CACHE
        .lock()
        .expect("typing marker cache lock poisoned");
    if let Some(value) = cache.get(name).copied() {
        return value;
    }

    let value = new_marker_value(name);
    cache.insert(name, value);
    value
}

fn new_marker_value(name: &str) -> Value {
    let class = &*TYPING_FORM_CLASS;
    let mut object = ShapedObject::new(class.class_type_id(), Arc::clone(class.instance_shape()));
    let registry = shape_registry();
    let name_value = Value::string(intern(name));
    object.set_property(intern("__typing_name__"), name_value, registry);
    object.set_property(intern("__name__"), name_value, registry);
    object.set_property(
        intern("__module__"),
        Value::string(intern("typing")),
        registry,
    );
    alloc_static_value(object)
}

fn marker_name(value: Value) -> Result<InternedString, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("typing marker receiver must be an object".to_string())
    })?;
    let object = unsafe { &*(ptr as *const ShapedObject) };
    let name_value = object.get_property("__typing_name__").ok_or_else(|| {
        BuiltinError::TypeError("typing marker is missing __typing_name__ metadata".to_string())
    })?;
    marker_text(name_value)
}

fn marker_text(value: Value) -> Result<InternedString, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("typing name must be str".to_string()))?;
        return interned_by_ptr(ptr as *const u8).ok_or_else(|| {
            BuiltinError::TypeError("typing name must be interned str".to_string())
        });
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("typing name must be str".to_string()))?;
    if crate::ops::objects::extract_type_id(ptr) != prism_runtime::object::type_obj::TypeId::STR {
        return Err(BuiltinError::TypeError(
            "typing name must be str".to_string(),
        ));
    }
    let string = unsafe { &*(ptr as *const prism_runtime::types::string::StringObject) };
    Ok(intern(string.as_str()))
}

fn typing_form_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "__getitem__() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let name = marker_name(args[0])?;
    Ok(crate::alloc_managed_value(TupleObject::from_vec(vec![
        Value::string(name),
        args[1],
    ])))
}

fn typing_identity_decorator(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "typing decorator expected exactly 1 argument ({} given)",
            args.len()
        )));
    }
    Ok(args[0])
}

fn typing_cast(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "cast() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    Ok(args[1])
}

fn typing_assert_type(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "assert_type() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    Ok(args[0])
}

fn typing_reveal_type(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "reveal_type() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }
    Ok(args[0])
}

fn typing_dataclass_transform(
    args: &[Value],
    _keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.len() == 1 && value_supports_call_protocol(args[0]) {
        return Ok(args[0]);
    }
    Ok(builtin_value(&IDENTITY_DECORATOR_FUNCTION))
}

fn typing_get_origin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "get_origin() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    Ok(typing_alias_origin(args[0]).unwrap_or_else(Value::none))
}

fn typing_get_args(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "get_args() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    Ok(typing_alias_args(args[0]).unwrap_or_else(empty_tuple_value))
}

fn typing_alias_origin(value: Value) -> Option<Value> {
    let (name, _) = typing_alias_parts(value)?;
    Some(cached_marker_value(interned_marker_name(&name)?))
}

fn typing_alias_args(value: Value) -> Option<Value> {
    let (_, params) = typing_alias_parts(value)?;
    if value_is_tuple(params) {
        Some(params)
    } else {
        Some(tuple_value(vec![params]))
    }
}

fn typing_alias_parts(value: Value) -> Option<(InternedString, Value)> {
    let ptr = value.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != prism_runtime::object::type_obj::TypeId::TUPLE {
        return None;
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    if tuple.len() != 2 {
        return None;
    }

    let name = marker_text(tuple.as_slice()[0]).ok()?;
    let _ = interned_marker_name(&name)?;
    Some((name, tuple.as_slice()[1]))
}

fn value_is_tuple(value: Value) -> bool {
    value.as_object_ptr().is_some_and(|ptr| {
        crate::ops::objects::extract_type_id(ptr) == prism_runtime::object::type_obj::TypeId::TUPLE
    })
}

fn empty_tuple_value() -> Value {
    tuple_value(Vec::new())
}

fn tuple_value(items: Vec<Value>) -> Value {
    crate::alloc_managed_value(TupleObject::from_vec(items))
}

fn interned_marker_name(name: &InternedString) -> Option<&'static str> {
    MARKER_EXPORTS
        .iter()
        .copied()
        .find(|candidate| *candidate == name.as_str())
}

fn typing_typevar(args: &[Value], _keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    typing_named_factory("TypeVar", args)
}

fn typing_paramspec(args: &[Value], _keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    typing_named_factory("ParamSpec", args)
}

fn typing_typevartuple(args: &[Value], _keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    typing_named_factory("TypeVarTuple", args)
}

fn typing_newtype(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "NewType() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    let name = marker_text(args[0])?;
    Ok(new_marker_value(name.as_str()))
}

fn typing_named_factory(factory_name: &str, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "{factory_name}() missing required argument 'name'"
        )));
    }
    let name = marker_text(args[0])?;
    Ok(new_marker_value(&format!("~{}", name.as_str())))
}
