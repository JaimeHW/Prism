//! Native `types` module bootstrap surface.
//!
//! CPython's `types.py` mostly exposes names for runtime-owned object kinds.
//! Prism keeps the object constructors native so compatibility imports do not
//! require a large Python source module on startup.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject, runtime_error_to_builtin_error};
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::{
    dict_storage_ref_from_ptr, extract_type_id, get_attribute_value, set_attribute_value,
};
use crate::truthiness::try_is_truthy;
use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, global_class_bitmap, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

static NEW_CLASS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm_kw(Arc::from("types.new_class"), new_class));
static DYNAMIC_CLASS_ATTRIBUTE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(build_dynamic_class_attribute_class);
static DYNAMIC_CLASS_ATTRIBUTE_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("types.DynamicClassAttribute.__init__"),
        dynamic_class_attribute_init,
    )
});
static DYNAMIC_CLASS_ATTRIBUTE_GET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("types.DynamicClassAttribute.__get__"),
        dynamic_class_attribute_get,
    )
});
static DYNAMIC_CLASS_ATTRIBUTE_SET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("types.DynamicClassAttribute.__set__"),
        dynamic_class_attribute_set,
    )
});
static DYNAMIC_CLASS_ATTRIBUTE_DELETE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm(
            Arc::from("types.DynamicClassAttribute.__delete__"),
            dynamic_class_attribute_delete,
        )
    });
static DYNAMIC_CLASS_ATTRIBUTE_GETTER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm(
            Arc::from("types.DynamicClassAttribute.getter"),
            dynamic_class_attribute_getter,
        )
    });
static DYNAMIC_CLASS_ATTRIBUTE_SETTER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm(
            Arc::from("types.DynamicClassAttribute.setter"),
            dynamic_class_attribute_setter,
        )
    });
static DYNAMIC_CLASS_ATTRIBUTE_DELETER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm(
            Arc::from("types.DynamicClassAttribute.deleter"),
            dynamic_class_attribute_deleter,
        )
    });

const TYPE_ALIASES: &[(&str, TypeId)] = &[
    ("GenericAlias", TypeId::GENERIC_ALIAS),
    ("MappingProxyType", TypeId::MAPPING_PROXY),
    ("MethodType", TypeId::METHOD),
    ("ModuleType", TypeId::MODULE),
];

/// Native `types` module descriptor.
#[derive(Debug, Clone)]
pub struct TypesModule {
    attrs: Vec<Arc<str>>,
}

impl TypesModule {
    /// Create a new `types` module descriptor.
    pub fn new() -> Self {
        let mut attrs = TYPE_ALIASES
            .iter()
            .map(|(name, _)| Arc::from(*name))
            .collect::<Vec<_>>();
        attrs.push(Arc::from("DynamicClassAttribute"));
        attrs.push(Arc::from("new_class"));

        Self { attrs }
    }
}

impl Default for TypesModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TypesModule {
    fn name(&self) -> &str {
        "types"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        if let Some((_, type_id)) = TYPE_ALIASES.iter().find(|(alias, _)| *alias == name) {
            return Ok(type_value(*type_id));
        }

        match name {
            "DynamicClassAttribute" => Ok(dynamic_class_attribute_class_value()),
            "new_class" => Ok(builtin_value(&NEW_CLASS_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'types' has no attribute '{}'",
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
fn type_value(type_id: TypeId) -> Value {
    crate::builtins::builtin_type_object_for_type_id(type_id)
}

#[inline]
fn dynamic_class_attribute_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&DYNAMIC_CLASS_ATTRIBUTE_CLASS) as *const ())
}

fn build_dynamic_class_attribute_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("DynamicClassAttribute"));
    class.set_attr(intern("__module__"), Value::string(intern("types")));
    class.set_attr(
        intern("__qualname__"),
        Value::string(intern("DynamicClassAttribute")),
    );
    class.set_attr(
        intern("__doc__"),
        Value::string(intern("Route attribute access on a class to __getattr__.")),
    );
    class.set_attr(
        intern("__init__"),
        builtin_value(&DYNAMIC_CLASS_ATTRIBUTE_INIT_METHOD),
    );
    class.set_attr(
        intern("__get__"),
        builtin_value(&DYNAMIC_CLASS_ATTRIBUTE_GET_METHOD),
    );
    class.set_attr(
        intern("__set__"),
        builtin_value(&DYNAMIC_CLASS_ATTRIBUTE_SET_METHOD),
    );
    class.set_attr(
        intern("__delete__"),
        builtin_value(&DYNAMIC_CLASS_ATTRIBUTE_DELETE_METHOD),
    );
    class.set_attr(
        intern("getter"),
        builtin_value(&DYNAMIC_CLASS_ATTRIBUTE_GETTER_METHOD),
    );
    class.set_attr(
        intern("setter"),
        builtin_value(&DYNAMIC_CLASS_ATTRIBUTE_SETTER_METHOD),
    );
    class.set_attr(
        intern("deleter"),
        builtin_value(&DYNAMIC_CLASS_ATTRIBUTE_DELETER_METHOD),
    );
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_INIT | ClassFlags::NATIVE_HEAPTYPE);

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    let class = Arc::new(class);
    register_global_class(Arc::clone(&class), bitmap);
    class
}

#[derive(Clone, Copy)]
struct DynamicClassAttributeInitArgs {
    self_value: Value,
    fget: Value,
    fset: Value,
    fdel: Value,
    doc: Value,
}

fn dynamic_class_attribute_init(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_dynamic_class_attribute_init_args(args, keywords)?;
    dynamic_class_attribute_object(bound.self_value, "__init__")?;

    let doc = dynamic_class_attribute_doc_value(vm, bound.fget, bound.doc)?;
    let is_abstract = dynamic_class_attribute_accessor_is_abstract(vm, bound.fget)?;

    set_dynamic_class_attribute_attr(vm, bound.self_value, dca_fget_attr(), bound.fget)?;
    set_dynamic_class_attribute_attr(vm, bound.self_value, dca_fset_attr(), bound.fset)?;
    set_dynamic_class_attribute_attr(vm, bound.self_value, dca_fdel_attr(), bound.fdel)?;
    set_dynamic_class_attribute_attr(vm, bound.self_value, dca_doc_attr(), doc)?;
    set_dynamic_class_attribute_attr(
        vm,
        bound.self_value,
        dca_overwrite_doc_attr(),
        Value::bool(bound.doc.is_none()),
    )?;
    set_dynamic_class_attribute_attr(
        vm,
        bound.self_value,
        dca_is_abstract_method_attr(),
        Value::bool(is_abstract),
    )?;
    Ok(Value::none())
}

fn bind_dynamic_class_attribute_init_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<DynamicClassAttributeInitArgs, BuiltinError> {
    let Some(&self_value) = args.first() else {
        return Err(BuiltinError::TypeError(
            "DynamicClassAttribute.__init__() missing required self argument".to_string(),
        ));
    };
    if args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "DynamicClassAttribute.__init__() takes from 1 to 5 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut slots: [Option<Value>; 4] = [None, None, None, None];
    for (index, value) in args.iter().copied().skip(1).enumerate() {
        slots[index] = Some(value);
    }

    for &(name, value) in keywords {
        let index = match name {
            "fget" => 0,
            "fset" => 1,
            "fdel" => 2,
            "doc" => 3,
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "DynamicClassAttribute.__init__() got an unexpected keyword argument '{}'",
                    name
                )));
            }
        };
        if slots[index].is_some() {
            return Err(BuiltinError::TypeError(format!(
                "DynamicClassAttribute.__init__() got multiple values for argument '{}'",
                name
            )));
        }
        slots[index] = Some(value);
    }

    Ok(DynamicClassAttributeInitArgs {
        self_value,
        fget: slots[0].unwrap_or_else(Value::none),
        fset: slots[1].unwrap_or_else(Value::none),
        fdel: slots[2].unwrap_or_else(Value::none),
        doc: slots[3].unwrap_or_else(Value::none),
    })
}

fn dynamic_class_attribute_get(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_dynamic_class_attribute_arg_range("__get__", args, 2, 3)?;
    dynamic_class_attribute_object(args[0], "__get__")?;

    let instance = args[1];
    if instance.is_none() {
        if dynamic_class_attribute_state_truthy(vm, args[0], dca_is_abstract_method_attr())? {
            return Ok(args[0]);
        }
        return Err(BuiltinError::AttributeError(String::new()));
    }

    let fget = dynamic_class_attribute_state_attr(vm, args[0], dca_fget_attr())?;
    if fget.is_none() {
        return Err(BuiltinError::AttributeError(
            "unreadable attribute".to_string(),
        ));
    }

    invoke_callable_value(vm, fget, &[instance]).map_err(runtime_error_to_builtin_error)
}

fn dynamic_class_attribute_set(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_dynamic_class_attribute_arg_count("__set__", args, 3)?;
    dynamic_class_attribute_object(args[0], "__set__")?;

    let fset = dynamic_class_attribute_state_attr(vm, args[0], dca_fset_attr())?;
    if fset.is_none() {
        return Err(BuiltinError::AttributeError(
            "can't set attribute".to_string(),
        ));
    }

    invoke_callable_value(vm, fset, &[args[1], args[2]]).map_err(runtime_error_to_builtin_error)?;
    Ok(Value::none())
}

fn dynamic_class_attribute_delete(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_dynamic_class_attribute_arg_count("__delete__", args, 2)?;
    dynamic_class_attribute_object(args[0], "__delete__")?;

    let fdel = dynamic_class_attribute_state_attr(vm, args[0], dca_fdel_attr())?;
    if fdel.is_none() {
        return Err(BuiltinError::AttributeError(
            "can't delete attribute".to_string(),
        ));
    }

    invoke_callable_value(vm, fdel, &[args[1]]).map_err(runtime_error_to_builtin_error)?;
    Ok(Value::none())
}

fn dynamic_class_attribute_getter(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_dynamic_class_attribute_arg_count("getter", args, 2)?;
    dynamic_class_attribute_object(args[0], "getter")?;

    let overwrite_doc =
        dynamic_class_attribute_state_truthy(vm, args[0], dca_overwrite_doc_attr())?;
    let fdoc = if overwrite_doc {
        get_attribute_value(vm, args[1], &dca_doc_attr()).map_err(runtime_error_to_builtin_error)?
    } else {
        Value::none()
    };
    let doc = if try_is_truthy(vm, fdoc).map_err(runtime_error_to_builtin_error)? {
        fdoc
    } else {
        dynamic_class_attribute_state_attr(vm, args[0], dca_doc_attr())?
    };
    let fset = dynamic_class_attribute_state_attr(vm, args[0], dca_fset_attr())?;
    let fdel = dynamic_class_attribute_state_attr(vm, args[0], dca_fdel_attr())?;

    let result = dynamic_class_attribute_copy(vm, args[0], args[1], fset, fdel, doc)?;
    set_dynamic_class_attribute_attr(
        vm,
        result,
        dca_overwrite_doc_attr(),
        Value::bool(overwrite_doc),
    )?;
    Ok(result)
}

fn dynamic_class_attribute_setter(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_dynamic_class_attribute_arg_count("setter", args, 2)?;
    dynamic_class_attribute_object(args[0], "setter")?;

    let overwrite_doc =
        dynamic_class_attribute_state_truthy(vm, args[0], dca_overwrite_doc_attr())?;
    let fget = dynamic_class_attribute_state_attr(vm, args[0], dca_fget_attr())?;
    let fdel = dynamic_class_attribute_state_attr(vm, args[0], dca_fdel_attr())?;
    let doc = dynamic_class_attribute_state_attr(vm, args[0], dca_doc_attr())?;
    let result = dynamic_class_attribute_copy(vm, args[0], fget, args[1], fdel, doc)?;
    set_dynamic_class_attribute_attr(
        vm,
        result,
        dca_overwrite_doc_attr(),
        Value::bool(overwrite_doc),
    )?;
    Ok(result)
}

fn dynamic_class_attribute_deleter(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_dynamic_class_attribute_arg_count("deleter", args, 2)?;
    dynamic_class_attribute_object(args[0], "deleter")?;

    let overwrite_doc =
        dynamic_class_attribute_state_truthy(vm, args[0], dca_overwrite_doc_attr())?;
    let fget = dynamic_class_attribute_state_attr(vm, args[0], dca_fget_attr())?;
    let fset = dynamic_class_attribute_state_attr(vm, args[0], dca_fset_attr())?;
    let doc = dynamic_class_attribute_state_attr(vm, args[0], dca_doc_attr())?;
    let result = dynamic_class_attribute_copy(vm, args[0], fget, fset, args[1], doc)?;
    set_dynamic_class_attribute_attr(
        vm,
        result,
        dca_overwrite_doc_attr(),
        Value::bool(overwrite_doc),
    )?;
    Ok(result)
}

fn dynamic_class_attribute_copy(
    vm: &mut VirtualMachine,
    self_value: Value,
    fget: Value,
    fset: Value,
    fdel: Value,
    doc: Value,
) -> Result<Value, BuiltinError> {
    let class_value = get_attribute_value(vm, self_value, &intern("__class__"))
        .map_err(runtime_error_to_builtin_error)?;
    let result = invoke_callable_value(vm, class_value, &[fget, fset, fdel, doc])
        .map_err(runtime_error_to_builtin_error)?;
    dynamic_class_attribute_object(result, "copy")?;
    Ok(result)
}

fn dynamic_class_attribute_doc_value(
    vm: &mut VirtualMachine,
    fget: Value,
    doc: Value,
) -> Result<Value, BuiltinError> {
    if try_is_truthy(vm, doc).map_err(runtime_error_to_builtin_error)? {
        return Ok(doc);
    }

    get_attribute_value(vm, fget, &dca_doc_attr()).map_err(runtime_error_to_builtin_error)
}

fn dynamic_class_attribute_accessor_is_abstract(
    vm: &mut VirtualMachine,
    accessor: Value,
) -> Result<bool, BuiltinError> {
    match get_attribute_value(vm, accessor, &dca_is_abstract_method_attr()) {
        Ok(value) => try_is_truthy(vm, value).map_err(runtime_error_to_builtin_error),
        Err(err) if err.is_attribute_error() => Ok(false),
        Err(err) => Err(runtime_error_to_builtin_error(err)),
    }
}

fn dynamic_class_attribute_state_attr(
    vm: &mut VirtualMachine,
    value: Value,
    name: InternedString,
) -> Result<Value, BuiltinError> {
    get_attribute_value(vm, value, &name).map_err(runtime_error_to_builtin_error)
}

fn dynamic_class_attribute_state_truthy(
    vm: &mut VirtualMachine,
    value: Value,
    name: InternedString,
) -> Result<bool, BuiltinError> {
    let value = dynamic_class_attribute_state_attr(vm, value, name)?;
    try_is_truthy(vm, value).map_err(runtime_error_to_builtin_error)
}

fn set_dynamic_class_attribute_attr(
    vm: &mut VirtualMachine,
    value: Value,
    name: InternedString,
    attr_value: Value,
) -> Result<(), BuiltinError> {
    set_attribute_value(vm, value, &name, attr_value).map_err(runtime_error_to_builtin_error)
}

fn dynamic_class_attribute_object(
    value: Value,
    context: &'static str,
) -> Result<&'static ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "DynamicClassAttribute.{context} requires a DynamicClassAttribute object"
        )));
    };
    if !is_dynamic_class_attribute_type(extract_type_id(ptr)) {
        return Err(BuiltinError::TypeError(format!(
            "DynamicClassAttribute.{context} requires a DynamicClassAttribute object"
        )));
    }
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

#[inline]
fn is_dynamic_class_attribute_type(type_id: TypeId) -> bool {
    type_id == dynamic_class_attribute_type_id()
        || (type_id.raw() >= TypeId::FIRST_USER_TYPE
            && global_class_bitmap(ClassId(type_id.raw()))
                .is_some_and(|bitmap| bitmap.is_subclass_of(dynamic_class_attribute_type_id())))
}

#[inline]
fn dynamic_class_attribute_type_id() -> TypeId {
    DYNAMIC_CLASS_ATTRIBUTE_CLASS.class_type_id()
}

fn expect_dynamic_class_attribute_arg_count(
    method: &'static str,
    args: &[Value],
    expected: usize,
) -> Result<(), BuiltinError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "DynamicClassAttribute.{method}() takes exactly {} arguments ({} given)",
            expected.saturating_sub(1),
            args.len().saturating_sub(1)
        )))
    }
}

fn expect_dynamic_class_attribute_arg_range(
    method: &'static str,
    args: &[Value],
    min: usize,
    max: usize,
) -> Result<(), BuiltinError> {
    if (min..=max).contains(&args.len()) {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "DynamicClassAttribute.{method}() takes from {} to {} arguments ({} given)",
            min.saturating_sub(1),
            max.saturating_sub(1),
            args.len().saturating_sub(1)
        )))
    }
}

#[inline]
fn dca_fget_attr() -> InternedString {
    intern("fget")
}

#[inline]
fn dca_fset_attr() -> InternedString {
    intern("fset")
}

#[inline]
fn dca_fdel_attr() -> InternedString {
    intern("fdel")
}

#[inline]
fn dca_doc_attr() -> InternedString {
    intern("__doc__")
}

#[inline]
fn dca_overwrite_doc_attr() -> InternedString {
    intern("overwrite_doc")
}

#[inline]
fn dca_is_abstract_method_attr() -> InternedString {
    intern("__isabstractmethod__")
}

#[derive(Clone, Copy)]
struct NewClassArgs {
    name: Value,
    bases: Value,
    kwds: Value,
    exec_body: Value,
}

fn new_class(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_new_class_args(args, keywords)?;
    let namespace = crate::alloc_managed_value(DictObject::new());

    if !bound.exec_body.is_none() {
        invoke_callable_value(vm, bound.exec_body, &[namespace])
            .map_err(crate::builtins::runtime_error_to_builtin_error)?;
    }

    let type_args = [bound.name, bound.bases, namespace];
    let kwd_entries = new_class_keyword_entries(bound.kwds)?;
    if kwd_entries.is_empty() {
        return crate::builtins::builtin_type_with_vm(vm, &type_args);
    }

    let kwd_refs = kwd_entries
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect::<Vec<_>>();
    crate::builtins::call_builtin_type_kw_with_vm(
        vm,
        prism_runtime::object::type_obj::TypeId::TYPE,
        &type_args,
        &kwd_refs,
    )
}

fn bind_new_class_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<NewClassArgs, BuiltinError> {
    if args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "new_class() takes from 1 to 4 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut name = args.first().copied();
    let mut bases = args.get(1).copied();
    let mut kwds = args.get(2).copied();
    let mut exec_body = args.get(3).copied();

    for &(keyword, value) in keywords {
        match keyword {
            "name" => assign_new_class_keyword(&mut name, value, "name")?,
            "bases" => assign_new_class_keyword(&mut bases, value, "bases")?,
            "kwds" => assign_new_class_keyword(&mut kwds, value, "kwds")?,
            "exec_body" => assign_new_class_keyword(&mut exec_body, value, "exec_body")?,
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "new_class() got an unexpected keyword argument '{}'",
                    keyword
                )));
            }
        }
    }

    let name = name.ok_or_else(|| {
        BuiltinError::TypeError("new_class() missing required argument 'name'".to_string())
    })?;

    Ok(NewClassArgs {
        name,
        bases: bases.unwrap_or_else(empty_tuple_value),
        kwds: kwds.unwrap_or_else(Value::none),
        exec_body: exec_body.unwrap_or_else(Value::none),
    })
}

#[inline]
fn assign_new_class_keyword(
    slot: &mut Option<Value>,
    value: Value,
    name: &str,
) -> Result<(), BuiltinError> {
    if slot.is_some() {
        return Err(BuiltinError::TypeError(format!(
            "new_class() got multiple values for argument '{}'",
            name
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn new_class_keyword_entries(value: Value) -> Result<Vec<(String, Value)>, BuiltinError> {
    if value.is_none() {
        return Ok(Vec::new());
    }

    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("new_class() argument 'kwds' must be a dict or None".to_string())
    })?;
    let dict = dict_storage_ref_from_ptr(ptr).ok_or_else(|| {
        BuiltinError::TypeError("new_class() argument 'kwds' must be a dict or None".to_string())
    })?;

    let mut entries = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let name = value_as_string_ref(key)
            .ok_or_else(|| BuiltinError::TypeError("keywords must be strings".to_string()))?;
        entries.push((name.as_str().to_string(), value));
    }
    Ok(entries)
}

#[inline]
fn empty_tuple_value() -> Value {
    crate::alloc_managed_value(TupleObject::empty())
}
