use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::error::{RuntimeError, RuntimeErrorKind};
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::{builtin_class_mro, class_id_to_type_id, global_class};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{DescriptorViewObject, MappingProxyObject, MethodWrapperObject};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::memoryview::value_as_memoryview_ref;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

static DICT_FROMKEYS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("dict.fromkeys"),
        super::types::builtin_dict_fromkeys,
    )
});
static STR_MAKETRANS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("str.maketrans"),
        super::types::builtin_str_maketrans,
    )
});
static OBJECT_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("object.__new__"),
        super::types::builtin_object_new,
    )
});
static OBJECT_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("object.__init__"),
        super::types::builtin_object_init,
    )
});
static OBJECT_INIT_SUBCLASS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("object.__init_subclass__"),
        super::types::builtin_object_init_subclass,
    )
});
static TYPE_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("type.__new__"),
        super::types::builtin_type_new_with_vm,
    )
});
static TYPE_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("type.__init__"), super::types::builtin_type_init)
});
static TYPE_PREPARE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("type.__prepare__"), type_prepare));
static TUPLE_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("tuple.__new__"), super::types::builtin_tuple_new)
});
static INT_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("int.__new__"), super::types::builtin_int_new_vm)
});
static INT_FROM_BYTES_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("int.from_bytes"),
        super::types::builtin_int_from_bytes_vm,
    )
});
static FLOAT_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("float.__new__"), super::types::builtin_float_new)
});
static FLOAT_GETFORMAT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("float.__getformat__"),
        super::types::builtin_float_getformat,
    )
});
static STR_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("str.__new__"), super::types::builtin_str_new)
});
static BOOL_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("bool.__new__"), super::types::builtin_bool_new)
});
static BOOL_FROM_BYTES_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("bool.from_bytes"),
        super::types::builtin_int_from_bytes_vm,
    )
});
static BYTES_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("bytes.__new__"), super::types::builtin_bytes_new)
});
static BYTES_MAKETRANS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("bytes.maketrans"),
        super::types::builtin_bytes_maketrans,
    )
});
static BYTEARRAY_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("bytearray.__new__"),
        super::types::builtin_bytearray_new,
    )
});
static BYTEARRAY_MAKETRANS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("bytearray.maketrans"),
        super::types::builtin_bytes_maketrans,
    )
});
static MEMORYVIEW_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("memoryview.__new__"),
        super::types::builtin_memoryview_new,
    )
});
static LIST_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("list.__new__"), super::types::builtin_list_new)
});
static DICT_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("dict.__new__"), super::types::builtin_dict_new)
});
static SET_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("set.__new__"), super::types::builtin_set_new)
});
static FROZENSET_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("frozenset.__new__"),
        super::types::builtin_frozenset_new,
    )
});
static MODULE_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("module.__new__"),
        super::types::builtin_module_new,
    )
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReflectedValueKind {
    DictProxy,
    MroTuple,
    BasesTuple,
    BaseValue,
    DocValue,
    NameString,
    QualNameString,
    ModuleString,
    WrapperDescriptor,
    MethodDescriptor,
    ClassMethodDescriptor,
    GetSetDescriptor,
    MemberDescriptor,
}

#[derive(Clone, Copy)]
struct AttrSpec {
    name: &'static str,
    kind: ReflectedValueKind,
}

const NEW_WRAPPER_ATTR: AttrSpec = AttrSpec {
    name: "__new__",
    kind: ReflectedValueKind::WrapperDescriptor,
};

const TYPE_ATTRS: &[AttrSpec] = &[
    AttrSpec {
        name: "__dict__",
        kind: ReflectedValueKind::DictProxy,
    },
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "__init__",
        kind: ReflectedValueKind::WrapperDescriptor,
    },
    AttrSpec {
        name: "__prepare__",
        kind: ReflectedValueKind::ClassMethodDescriptor,
    },
];

const OBJECT_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "__dict__",
        kind: ReflectedValueKind::DictProxy,
    },
    AttrSpec {
        name: "__init__",
        kind: ReflectedValueKind::WrapperDescriptor,
    },
    AttrSpec {
        name: "__init_subclass__",
        kind: ReflectedValueKind::ClassMethodDescriptor,
    },
];

const INT_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "from_bytes",
        kind: ReflectedValueKind::ClassMethodDescriptor,
    },
];

const FLOAT_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "__getformat__",
        kind: ReflectedValueKind::ClassMethodDescriptor,
    },
];

const STR_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "__dict__",
        kind: ReflectedValueKind::DictProxy,
    },
    AttrSpec {
        name: "maketrans",
        kind: ReflectedValueKind::MethodDescriptor,
    },
    AttrSpec {
        name: "join",
        kind: ReflectedValueKind::MethodDescriptor,
    },
];

const BOOL_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "from_bytes",
        kind: ReflectedValueKind::ClassMethodDescriptor,
    },
];

const BYTES_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "maketrans",
        kind: ReflectedValueKind::MethodDescriptor,
    },
];

const BYTEARRAY_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "maketrans",
        kind: ReflectedValueKind::MethodDescriptor,
    },
];

const MEMORYVIEW_TYPE_ATTRS: &[AttrSpec] = &[NEW_WRAPPER_ATTR];

const LIST_TYPE_ATTRS: &[AttrSpec] = &[NEW_WRAPPER_ATTR];

const TUPLE_TYPE_ATTRS: &[AttrSpec] = &[NEW_WRAPPER_ATTR];

const DICT_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "__dict__",
        kind: ReflectedValueKind::DictProxy,
    },
    AttrSpec {
        name: "fromkeys",
        kind: ReflectedValueKind::ClassMethodDescriptor,
    },
];

const SET_TYPE_ATTRS: &[AttrSpec] = &[NEW_WRAPPER_ATTR];

const FROZENSET_TYPE_ATTRS: &[AttrSpec] = &[NEW_WRAPPER_ATTR];

const MODULE_TYPE_ATTRS: &[AttrSpec] = &[NEW_WRAPPER_ATTR];

const FUNCTION_TYPE_ATTRS: &[AttrSpec] = &[
    AttrSpec {
        name: "__dict__",
        kind: ReflectedValueKind::DictProxy,
    },
    AttrSpec {
        name: "__code__",
        kind: ReflectedValueKind::GetSetDescriptor,
    },
    AttrSpec {
        name: "__globals__",
        kind: ReflectedValueKind::MemberDescriptor,
    },
    AttrSpec {
        name: "__get__",
        kind: ReflectedValueKind::MethodDescriptor,
    },
];

const DEQUE_METHOD_NAMES: &[&str] = &[
    "__getitem__",
    "append",
    "appendleft",
    "pop",
    "popleft",
    "remove",
];
const LIST_METHOD_NAMES: &[&str] = &[
    "__iter__",
    "__len__",
    "__getitem__",
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "copy",
    "clear",
    "count",
    "index",
    "reverse",
    "sort",
];
const DICT_METHOD_NAMES: &[&str] = &[
    "__len__",
    "__contains__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "keys",
    "get",
    "values",
    "items",
    "pop",
    "popitem",
    "setdefault",
    "clear",
    "update",
    "copy",
];
const OBJECT_METHOD_NAMES: &[&str] = &["__eq__", "__ne__", "__setattr__", "__delattr__"];
const INT_METHOD_NAMES: &[&str] = &[
    "__add__",
    "__index__",
    "bit_length",
    "bit_count",
    "to_bytes",
];
const STR_METHOD_NAMES: &[&str] = &[
    "upper",
    "replace",
    "join",
    "isidentifier",
    "isascii",
    "startswith",
    "endswith",
    "rpartition",
];
const SET_METHOD_NAMES: &[&str] = &[
    "add",
    "remove",
    "discard",
    "pop",
    "clear",
    "update",
    "difference_update",
    "intersection_update",
    "symmetric_difference_update",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "isdisjoint",
    "issubset",
    "issuperset",
    "copy",
    "__contains__",
];
const FROZENSET_METHOD_NAMES: &[&str] = &[
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "isdisjoint",
    "issubset",
    "issuperset",
    "copy",
    "__contains__",
];
const BYTEARRAY_METHOD_NAMES: &[&str] = &["copy", "extend"];
const MEMORYVIEW_METHOD_NAMES: &[&str] = &[
    "tobytes",
    "tolist",
    "cast",
    "release",
    "__enter__",
    "__exit__",
];
const TUPLE_METHOD_NAMES: &[&str] = &["__iter__", "__len__", "__getitem__", "count", "index"];
const ITERATOR_METHOD_NAMES: &[&str] = &["__iter__", "__next__", "__length_hint__"];
const GENERATOR_METHOD_NAMES: &[&str] = &["close"];
const PROPERTY_METHOD_NAMES: &[&str] = &["getter", "setter", "deleter"];
const REGEX_PATTERN_METHOD_NAMES: &[&str] = &[
    "match",
    "search",
    "fullmatch",
    "findall",
    "finditer",
    "sub",
    "subn",
    "split",
];
const REGEX_MATCH_METHOD_NAMES: &[&str] = &[
    "__getitem__",
    "group",
    "groups",
    "groupdict",
    "start",
    "end",
    "span",
];

#[inline]
fn builtin_type_attr_specs(type_id: TypeId) -> &'static [AttrSpec] {
    match type_id {
        TypeId::TYPE => TYPE_ATTRS,
        TypeId::OBJECT => OBJECT_TYPE_ATTRS,
        TypeId::INT => INT_TYPE_ATTRS,
        TypeId::FLOAT => FLOAT_TYPE_ATTRS,
        TypeId::STR => STR_TYPE_ATTRS,
        TypeId::BOOL => BOOL_TYPE_ATTRS,
        TypeId::BYTES => BYTES_TYPE_ATTRS,
        TypeId::BYTEARRAY => BYTEARRAY_TYPE_ATTRS,
        TypeId::MEMORYVIEW => MEMORYVIEW_TYPE_ATTRS,
        TypeId::LIST => LIST_TYPE_ATTRS,
        TypeId::TUPLE => TUPLE_TYPE_ATTRS,
        TypeId::DICT => DICT_TYPE_ATTRS,
        TypeId::SET => SET_TYPE_ATTRS,
        TypeId::FROZENSET => FROZENSET_TYPE_ATTRS,
        TypeId::MODULE => MODULE_TYPE_ATTRS,
        TypeId::FUNCTION => FUNCTION_TYPE_ATTRS,
        _ => &[],
    }
}

#[inline]
fn find_attr_spec(type_id: TypeId, name: &InternedString) -> Option<ReflectedValueKind> {
    builtin_type_attr_specs(type_id)
        .iter()
        .find(|spec| spec.name == name.as_str())
        .map(|spec| spec.kind)
}

#[inline]
fn builtin_reflected_method_names(type_id: TypeId) -> &'static [&'static str] {
    match type_id {
        TypeId::DEQUE => DEQUE_METHOD_NAMES,
        TypeId::DICT => DICT_METHOD_NAMES,
        TypeId::OBJECT => OBJECT_METHOD_NAMES,
        TypeId::INT => INT_METHOD_NAMES,
        TypeId::STR => STR_METHOD_NAMES,
        TypeId::LIST => LIST_METHOD_NAMES,
        TypeId::SET => SET_METHOD_NAMES,
        TypeId::FROZENSET => FROZENSET_METHOD_NAMES,
        TypeId::BYTEARRAY => BYTEARRAY_METHOD_NAMES,
        TypeId::MEMORYVIEW => MEMORYVIEW_METHOD_NAMES,
        TypeId::TUPLE => TUPLE_METHOD_NAMES,
        TypeId::ITERATOR => ITERATOR_METHOD_NAMES,
        TypeId::GENERATOR => GENERATOR_METHOD_NAMES,
        TypeId::PROPERTY => PROPERTY_METHOD_NAMES,
        TypeId::REGEX_PATTERN => REGEX_PATTERN_METHOD_NAMES,
        TypeId::REGEX_MATCH => REGEX_MATCH_METHOD_NAMES,
        _ => &[],
    }
}

#[inline]
fn builtin_mapping_proxy_names(owner: TypeId) -> Vec<InternedString> {
    let mut names = Vec::with_capacity(
        builtin_type_attr_specs(owner).len() + builtin_reflected_method_names(owner).len(),
    );

    for spec in builtin_type_attr_specs(owner) {
        names.push(intern(spec.name));
    }

    for &name in builtin_reflected_method_names(owner) {
        let name = intern(name);
        if !names.iter().any(|existing| existing == &name) {
            names.push(name);
        }
    }

    names
}

#[inline]
fn synthesized_builtin_method_kind(
    type_id: TypeId,
    name: &InternedString,
) -> Option<ReflectedValueKind> {
    crate::ops::method_dispatch::resolve_builtin_instance_method(type_id, name.as_str()).map(|_| {
        if name.as_str().starts_with("__") {
            ReflectedValueKind::WrapperDescriptor
        } else {
            ReflectedValueKind::MethodDescriptor
        }
    })
}

#[inline]
fn reflected_builtin_attr_kind(
    type_id: TypeId,
    name: &InternedString,
) -> Option<ReflectedValueKind> {
    find_attr_spec(type_id, name).or_else(|| synthesized_builtin_method_kind(type_id, name))
}

#[inline]
fn reflected_type_attr_kind(name: &InternedString) -> Option<ReflectedValueKind> {
    match name.as_str() {
        "__dict__" => Some(ReflectedValueKind::DictProxy),
        "__mro__" => Some(ReflectedValueKind::MroTuple),
        "__bases__" => Some(ReflectedValueKind::BasesTuple),
        "__base__" => Some(ReflectedValueKind::BaseValue),
        "__doc__" => Some(ReflectedValueKind::DocValue),
        "__name__" => Some(ReflectedValueKind::NameString),
        "__qualname__" => Some(ReflectedValueKind::QualNameString),
        "__module__" => Some(ReflectedValueKind::ModuleString),
        _ => None,
    }
}

#[inline]
fn alloc_view<T>(
    vm: &mut VirtualMachine,
    object: T,
    context: &'static str,
) -> Result<Value, RuntimeError>
where
    T: prism_runtime::Trace,
{
    vm.allocator()
        .alloc(object)
        .map(|ptr| Value::object_ptr(ptr as *const ()))
        .ok_or_else(|| {
            RuntimeError::internal(format!("out of memory: failed to allocate {context}"))
        })
}

#[inline]
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

#[inline]
fn descriptor_type_id(kind: ReflectedValueKind) -> Option<TypeId> {
    match kind {
        ReflectedValueKind::WrapperDescriptor => Some(TypeId::WRAPPER_DESCRIPTOR),
        ReflectedValueKind::MethodDescriptor => Some(TypeId::METHOD_DESCRIPTOR),
        ReflectedValueKind::ClassMethodDescriptor => Some(TypeId::CLASSMETHOD_DESCRIPTOR),
        ReflectedValueKind::GetSetDescriptor => Some(TypeId::GETSET_DESCRIPTOR),
        ReflectedValueKind::MemberDescriptor => Some(TypeId::MEMBER_DESCRIPTOR),
        ReflectedValueKind::DictProxy
        | ReflectedValueKind::MroTuple
        | ReflectedValueKind::BasesTuple
        | ReflectedValueKind::BaseValue
        | ReflectedValueKind::DocValue
        | ReflectedValueKind::NameString
        | ReflectedValueKind::QualNameString
        | ReflectedValueKind::ModuleString => None,
    }
}

#[inline]
fn builtin_type_doc_value(owner: TypeId) -> Value {
    match builtin_type_doc(owner) {
        Some(doc) => Value::string(intern(doc)),
        None => Value::none(),
    }
}

const NONE_TYPE_DOC: &str = "The type of the None singleton.";

#[inline]
fn builtin_type_doc(owner: TypeId) -> Option<&'static str> {
    match owner {
        TypeId::NONE => Some(NONE_TYPE_DOC),
        TypeId::OBJECT => Some("The base class for all Python objects."),
        TypeId::TYPE => Some("Create a new type, or return the type of an object."),
        TypeId::BOOL => Some("Boolean type, representing True or False."),
        TypeId::INT => Some("Integer type with arbitrary precision semantics."),
        TypeId::FLOAT => Some("Floating point number type."),
        TypeId::STR => Some("Text string type."),
        TypeId::BYTES => Some("Immutable sequence of bytes."),
        TypeId::BYTEARRAY => Some("Mutable sequence of bytes."),
        TypeId::MEMORYVIEW => Some("Memory view over an object supporting the buffer protocol."),
        TypeId::LIST => Some("Mutable sequence type."),
        TypeId::TUPLE => Some("Immutable sequence type."),
        TypeId::DICT => Some("Dictionary mapping type."),
        TypeId::SET => Some("Mutable set type."),
        TypeId::FROZENSET => Some("Immutable set type."),
        TypeId::MODULE => Some("Module type."),
        TypeId::FUNCTION => Some("Function type."),
        TypeId::METHOD => Some("Method type."),
        TypeId::PROPERTY => Some("Property attribute descriptor."),
        TypeId::SLICE => Some("Slice object type."),
        TypeId::RANGE => Some("Immutable arithmetic progression type."),
        TypeId::COMPLEX => Some("Complex number type."),
        _ => None,
    }
}

fn type_prepare(args: &[Value], _keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "type.__prepare__() takes 2 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    Ok(leak_object_value(DictObject::new()))
}

#[inline]
fn user_type_doc_value(class: &PyClassObject) -> Value {
    class
        .get_attr(&intern("__doc__"))
        .unwrap_or_else(Value::none)
}

#[inline(always)]
fn builtin_method_value(method: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(method as *const BuiltinFunctionObject as *const ())
}

#[inline]
pub(crate) fn builtin_type_method_value(owner: TypeId, name: &str) -> Option<Value> {
    crate::ops::method_dispatch::resolve_builtin_instance_method(owner, name)
        .map(|cached| cached.method)
        .or_else(|| builtin_type_static_method_value(owner, name))
        .or_else(|| builtin_type_class_method_value(owner, name))
}

#[inline]
fn builtin_type_static_method_value(owner: TypeId, name: &str) -> Option<Value> {
    match (owner, name) {
        (TypeId::TYPE, "__new__") => Some(builtin_method_value(&TYPE_NEW_METHOD)),
        (TypeId::OBJECT, "__new__") => Some(builtin_method_value(&OBJECT_NEW_METHOD)),
        (TypeId::INT, "__new__") => Some(builtin_method_value(&INT_NEW_METHOD)),
        (TypeId::FLOAT, "__new__") => Some(builtin_method_value(&FLOAT_NEW_METHOD)),
        (TypeId::STR, "__new__") => Some(builtin_method_value(&STR_NEW_METHOD)),
        (TypeId::STR, "maketrans") => Some(builtin_method_value(&STR_MAKETRANS_METHOD)),
        (TypeId::BOOL, "__new__") => Some(builtin_method_value(&BOOL_NEW_METHOD)),
        (TypeId::BYTES, "__new__") => Some(builtin_method_value(&BYTES_NEW_METHOD)),
        (TypeId::BYTES, "maketrans") => Some(builtin_method_value(&BYTES_MAKETRANS_METHOD)),
        (TypeId::BYTEARRAY, "__new__") => Some(builtin_method_value(&BYTEARRAY_NEW_METHOD)),
        (TypeId::BYTEARRAY, "maketrans") => Some(builtin_method_value(&BYTEARRAY_MAKETRANS_METHOD)),
        (TypeId::MEMORYVIEW, "__new__") => Some(builtin_method_value(&MEMORYVIEW_NEW_METHOD)),
        (TypeId::LIST, "__new__") => Some(builtin_method_value(&LIST_NEW_METHOD)),
        (TypeId::TUPLE, "__new__") => Some(builtin_method_value(&TUPLE_NEW_METHOD)),
        (TypeId::DICT, "__new__") => Some(builtin_method_value(&DICT_NEW_METHOD)),
        (TypeId::SET, "__new__") => Some(builtin_method_value(&SET_NEW_METHOD)),
        (TypeId::FROZENSET, "__new__") => Some(builtin_method_value(&FROZENSET_NEW_METHOD)),
        (TypeId::MODULE, "__new__") => Some(builtin_method_value(&MODULE_NEW_METHOD)),
        _ => None,
    }
}

#[inline]
fn builtin_type_bound_method_value(owner: TypeId, name: &str) -> Option<Value> {
    match (owner, name) {
        (TypeId::TYPE, "__prepare__") => Some(builtin_method_value(&TYPE_PREPARE_METHOD)),
        (TypeId::DICT, "fromkeys") => Some(builtin_method_value(&DICT_FROMKEYS_METHOD)),
        (TypeId::INT, "from_bytes") => Some(builtin_method_value(&INT_FROM_BYTES_METHOD)),
        (TypeId::FLOAT, "__getformat__") => Some(builtin_method_value(&FLOAT_GETFORMAT_METHOD)),
        (TypeId::BOOL, "from_bytes") => Some(builtin_method_value(&BOOL_FROM_BYTES_METHOD)),
        (TypeId::OBJECT, "__init_subclass__") => {
            Some(builtin_method_value(&OBJECT_INIT_SUBCLASS_METHOD))
        }
        _ => None,
    }
}

#[inline]
fn builtin_type_class_method_value(owner: TypeId, name: &str) -> Option<Value> {
    match (owner, name) {
        (TypeId::TYPE, "__prepare__") => Some(builtin_method_value(&TYPE_PREPARE_METHOD)),
        (TypeId::DICT, "fromkeys") => Some(builtin_method_value(&DICT_FROMKEYS_METHOD)),
        (TypeId::INT, "from_bytes") => Some(builtin_method_value(&INT_FROM_BYTES_METHOD)),
        (TypeId::FLOAT, "__getformat__") => Some(builtin_method_value(&FLOAT_GETFORMAT_METHOD)),
        (TypeId::BOOL, "from_bytes") => Some(builtin_method_value(&BOOL_FROM_BYTES_METHOD)),
        (TypeId::OBJECT, "__init_subclass__") => {
            Some(builtin_method_value(&OBJECT_INIT_SUBCLASS_METHOD))
        }
        _ => None,
    }
}

pub(crate) fn reflected_descriptor_callable_value(
    descriptor_type: TypeId,
    owner: TypeId,
    name: &InternedString,
) -> Option<Value> {
    match descriptor_type {
        TypeId::WRAPPER_DESCRIPTOR => match (owner, name.as_str()) {
            (TypeId::OBJECT, "__init__") => Some(builtin_method_value(&OBJECT_INIT_METHOD)),
            (TypeId::TYPE, "__init__") => Some(builtin_method_value(&TYPE_INIT_METHOD)),
            _ => builtin_type_method_value(owner, name.as_str()),
        },
        TypeId::METHOD_DESCRIPTOR => builtin_type_method_value(owner, name.as_str()),
        TypeId::CLASSMETHOD_DESCRIPTOR => builtin_type_class_method_value(owner, name.as_str()),
        _ => None,
    }
}

fn materialize_attr_value(
    vm: &mut VirtualMachine,
    owner: TypeId,
    name: &InternedString,
    kind: ReflectedValueKind,
) -> Result<Value, RuntimeError> {
    match kind {
        ReflectedValueKind::DictProxy => alloc_view(
            vm,
            MappingProxyObject::for_builtin_type(owner),
            "mapping proxy view",
        ),
        ReflectedValueKind::MroTuple => alloc_view(
            vm,
            TupleObject::from_vec(
                builtin_class_mro(owner)
                    .into_iter()
                    .map(class_id_to_type_value)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            "type mro tuple",
        ),
        ReflectedValueKind::BasesTuple => alloc_view(
            vm,
            TupleObject::from_vec(
                builtin_direct_bases(owner)
                    .into_iter()
                    .map(class_id_to_type_value)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            "type bases tuple",
        ),
        ReflectedValueKind::BaseValue => builtin_base_value(owner),
        ReflectedValueKind::DocValue => Ok(builtin_type_doc_value(owner)),
        ReflectedValueKind::NameString => Ok(Value::string(intern(owner.name()))),
        ReflectedValueKind::QualNameString => Ok(Value::string(intern(owner.name()))),
        ReflectedValueKind::ModuleString => Ok(Value::string(intern(match owner {
            TypeId::DEQUE => "collections",
            _ => "builtins",
        }))),
        other => {
            let type_id = descriptor_type_id(other)
                .expect("descriptor type id required for non-mapping-proxy reflection");
            alloc_view(
                vm,
                DescriptorViewObject::new(type_id, owner, name.clone()),
                "descriptor view",
            )
        }
    }
}

fn materialize_attr_value_static(
    owner: TypeId,
    name: &InternedString,
    kind: ReflectedValueKind,
) -> Result<Value, RuntimeError> {
    match kind {
        ReflectedValueKind::DictProxy => Ok(leak_object_value(
            MappingProxyObject::for_builtin_type(owner),
        )),
        ReflectedValueKind::MroTuple => Ok(leak_object_value(TupleObject::from_vec(
            builtin_class_mro(owner)
                .into_iter()
                .map(class_id_to_type_value)
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        ReflectedValueKind::BasesTuple => Ok(leak_object_value(TupleObject::from_vec(
            builtin_direct_bases(owner)
                .into_iter()
                .map(class_id_to_type_value)
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        ReflectedValueKind::BaseValue => builtin_base_value(owner),
        ReflectedValueKind::DocValue => Ok(builtin_type_doc_value(owner)),
        ReflectedValueKind::NameString => Ok(Value::string(intern(owner.name()))),
        ReflectedValueKind::QualNameString => Ok(Value::string(intern(owner.name()))),
        ReflectedValueKind::ModuleString => Ok(Value::string(intern(match owner {
            TypeId::DEQUE => "collections",
            _ => "builtins",
        }))),
        other => {
            let type_id = descriptor_type_id(other)
                .expect("descriptor type id required for non-mapping-proxy reflection");
            Ok(leak_object_value(DescriptorViewObject::new(
                type_id,
                owner,
                name.clone(),
            )))
        }
    }
}

#[inline]
fn builtin_direct_bases(owner: TypeId) -> Vec<ClassId> {
    let mro = builtin_class_mro(owner);
    if mro.len() <= 1 {
        Vec::new()
    } else {
        vec![mro[1]]
    }
}

#[inline]
fn builtin_base_value(owner: TypeId) -> Result<Value, RuntimeError> {
    builtin_direct_bases(owner)
        .into_iter()
        .next()
        .map(class_id_to_type_value)
        .transpose()
        .map(|value| value.unwrap_or_else(Value::none))
}

fn class_id_to_type_value(class_id: ClassId) -> Result<Value, RuntimeError> {
    if class_id == ClassId::OBJECT {
        return Ok(crate::builtins::builtin_type_object_for_type_id(
            TypeId::OBJECT,
        ));
    }

    if class_id.0 < TypeId::FIRST_USER_TYPE {
        return Ok(crate::builtins::builtin_type_object_for_type_id(
            class_id_to_type_id(class_id),
        ));
    }

    if let Some(value) = super::exception_type::exception_type_value_for_proxy_class_id(class_id) {
        return Ok(value);
    }

    let class = global_class(class_id).ok_or_else(|| {
        RuntimeError::internal(format!(
            "missing heap type for reflected class id {}",
            class_id.0
        ))
    })?;
    Ok(Value::object_ptr(Arc::as_ptr(&class) as *const ()))
}

#[inline]
fn class_id_to_type_value_for_owner(
    class_id: ClassId,
    owner: &PyClassObject,
    owner_value: Value,
) -> Result<Value, RuntimeError> {
    if class_id == owner.class_id() {
        return Ok(owner_value);
    }

    class_id_to_type_value(class_id)
}

#[inline(always)]
fn heap_type_owner_value(class_ptr: *const PyClassObject) -> Value {
    Value::object_ptr(class_ptr as *const ())
}

#[inline]
fn user_type_direct_bases(class: &PyClassObject) -> Vec<ClassId> {
    if class.bases().is_empty() {
        vec![ClassId::OBJECT]
    } else {
        class.bases().to_vec()
    }
}

#[inline]
fn resolve_heap_type_mro_value(
    vm: &mut VirtualMachine,
    class: &PyClassObject,
    name: &InternedString,
    owner_value: Value,
    mut resolve_builtin: impl FnMut(&mut VirtualMachine, TypeId) -> Result<Option<Value>, RuntimeError>,
) -> Result<Option<Value>, RuntimeError> {
    for &class_id in class.mro() {
        if class_id == class.class_id() {
            if let Some(value) = class.get_attr(name) {
                return crate::ops::objects::resolve_class_attribute_in_vm(vm, value, owner_value)
                    .map(Some);
            }
            continue;
        }

        if class_id.0 < TypeId::FIRST_USER_TYPE {
            if let Some(value) = resolve_builtin(vm, class_id_to_type_id(class_id))? {
                return Ok(Some(value));
            }
            continue;
        }

        if let Some(parent) = global_class(class_id)
            && let Some(value) = parent.get_attr(name)
        {
            return crate::ops::objects::resolve_class_attribute_in_vm(vm, value, owner_value)
                .map(Some);
        }
    }

    Ok(None)
}

#[inline]
fn resolve_heap_type_mro_value_static(
    class: &PyClassObject,
    name: &InternedString,
    owner_value: Value,
    mut resolve_builtin: impl FnMut(TypeId) -> Result<Option<Value>, RuntimeError>,
) -> Result<Option<Value>, RuntimeError> {
    for &class_id in class.mro() {
        if class_id == class.class_id() {
            if let Some(value) = class.get_attr(name) {
                return Ok(Some(crate::ops::objects::resolve_class_attribute(
                    value,
                    owner_value,
                )));
            }
            continue;
        }

        if class_id.0 < TypeId::FIRST_USER_TYPE {
            if let Some(value) = resolve_builtin(class_id_to_type_id(class_id))? {
                return Ok(Some(value));
            }
            continue;
        }

        if let Some(parent) = global_class(class_id) {
            if let Some(value) = parent.get_attr(name) {
                return Ok(Some(crate::ops::objects::resolve_class_attribute(
                    value,
                    owner_value,
                )));
            }
        }
    }

    Ok(None)
}

fn user_type_attribute_value(
    vm: &mut VirtualMachine,
    class: &PyClassObject,
    class_ptr: *const PyClassObject,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let owner_value = heap_type_owner_value(class_ptr);
    match reflected_type_attr_kind(name) {
        Some(ReflectedValueKind::DictProxy) => alloc_view(
            vm,
            MappingProxyObject::for_user_class(class_ptr),
            "heap type mapping proxy view",
        )
        .map(Some),
        Some(ReflectedValueKind::MroTuple) => alloc_view(
            vm,
            TupleObject::from_vec(
                class
                    .mro()
                    .iter()
                    .copied()
                    .map(|class_id| class_id_to_type_value_for_owner(class_id, class, owner_value))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            "heap type mro tuple",
        )
        .map(Some),
        Some(ReflectedValueKind::BasesTuple) => alloc_view(
            vm,
            TupleObject::from_vec(
                user_type_direct_bases(class)
                    .into_iter()
                    .map(class_id_to_type_value)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            "heap type bases tuple",
        )
        .map(Some),
        Some(ReflectedValueKind::BaseValue) => user_type_direct_bases(class)
            .into_iter()
            .next()
            .map(class_id_to_type_value)
            .transpose()
            .map(|value| Some(value.unwrap_or_else(Value::none))),
        Some(ReflectedValueKind::DocValue) => Ok(Some(user_type_doc_value(class))),
        Some(ReflectedValueKind::NameString) => Ok(Some(Value::string(class.name().clone()))),
        Some(ReflectedValueKind::QualNameString) => Ok(Some(
            class
                .get_attr(&intern("__qualname__"))
                .unwrap_or_else(|| Value::string(class.name().clone())),
        )),
        Some(ReflectedValueKind::ModuleString) => Ok(Some(
            class
                .get_attr(&intern("__module__"))
                .unwrap_or_else(|| Value::string(intern("__main__"))),
        )),
        _ => resolve_heap_type_mro_value(vm, class, name, owner_value, |vm, owner| {
            builtin_bound_type_attribute_value(vm, owner, owner_value, name)
        }),
    }
}

fn user_type_attribute_value_static(
    class: &PyClassObject,
    class_ptr: *const PyClassObject,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let owner_value = heap_type_owner_value(class_ptr);
    match reflected_type_attr_kind(name) {
        Some(ReflectedValueKind::DictProxy) => Ok(Some(leak_object_value(
            MappingProxyObject::for_user_class(class_ptr),
        ))),
        Some(ReflectedValueKind::MroTuple) => Ok(Some(leak_object_value(TupleObject::from_vec(
            class
                .mro()
                .iter()
                .copied()
                .map(|class_id| class_id_to_type_value_for_owner(class_id, class, owner_value))
                .collect::<Result<Vec<_>, _>>()?,
        )))),
        Some(ReflectedValueKind::BasesTuple) => Ok(Some(leak_object_value(TupleObject::from_vec(
            user_type_direct_bases(class)
                .into_iter()
                .map(class_id_to_type_value)
                .collect::<Result<Vec<_>, _>>()?,
        )))),
        Some(ReflectedValueKind::BaseValue) => user_type_direct_bases(class)
            .into_iter()
            .next()
            .map(class_id_to_type_value)
            .transpose()
            .map(|value| Some(value.unwrap_or_else(Value::none))),
        Some(ReflectedValueKind::DocValue) => Ok(Some(user_type_doc_value(class))),
        Some(ReflectedValueKind::NameString) => Ok(Some(Value::string(class.name().clone()))),
        Some(ReflectedValueKind::QualNameString) => Ok(Some(
            class
                .get_attr(&intern("__qualname__"))
                .unwrap_or_else(|| Value::string(class.name().clone())),
        )),
        Some(ReflectedValueKind::ModuleString) => Ok(Some(
            class
                .get_attr(&intern("__module__"))
                .unwrap_or_else(|| Value::string(intern("__main__"))),
        )),
        _ => resolve_heap_type_mro_value_static(class, name, owner_value, |owner| {
            builtin_bound_type_attribute_value_static(owner, owner_value, name)
        }),
    }
}

#[inline]
fn user_type_has_attribute(class: &PyClassObject, name: &InternedString) -> bool {
    if reflected_type_attr_kind(name).is_some() {
        return true;
    }

    for &class_id in class.mro() {
        if class_id == class.class_id() {
            if class.has_attr(name) {
                return true;
            }
            continue;
        }

        if class_id.0 < TypeId::FIRST_USER_TYPE {
            if builtin_type_has_attribute(class_id_to_type_id(class_id), name) {
                return true;
            }
            continue;
        }

        if let Some(parent) = global_class(class_id) {
            if parent.has_attr(name) {
                return true;
            }
        }
    }

    false
}

#[inline]
fn builtin_mapping_proxy_attribute_value(
    vm: &mut VirtualMachine,
    owner: TypeId,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let Some(kind) = reflected_builtin_attr_kind(owner, name) else {
        return Ok(None);
    };
    materialize_attr_value(vm, owner, name, kind).map(Some)
}

fn builtin_mapping_proxy_attribute_value_static(
    owner: TypeId,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let Some(kind) = reflected_builtin_attr_kind(owner, name) else {
        return Ok(None);
    };
    materialize_attr_value_static(owner, name, kind).map(Some)
}

fn key_to_name(key: Value) -> Result<InternedString, RuntimeError> {
    if key.is_string() {
        let ptr = key
            .as_string_object_ptr()
            .ok_or_else(|| RuntimeError::type_error("mappingproxy keys must be strings"))?;
        return interned_by_ptr(ptr as *const u8)
            .or_else(|| {
                let string = unsafe { &*(ptr as *const StringObject) };
                Some(intern(string.as_str()))
            })
            .ok_or_else(|| RuntimeError::type_error("mappingproxy keys must be strings"));
    }

    let Some(ptr) = key.as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "mappingproxy keys must be strings",
        ));
    };
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return Err(RuntimeError::type_error(
            "mappingproxy keys must be strings",
        ));
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(intern(string.as_str()))
}

pub(crate) fn builtin_type_attribute_value(
    vm: &mut VirtualMachine,
    owner: TypeId,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if let Some(kind) = reflected_type_attr_kind(name) {
        return materialize_attr_value(vm, owner, name, kind).map(Some);
    }

    builtin_mapping_proxy_attribute_value(vm, owner, name)
}

pub(crate) fn builtin_type_attribute_value_static(
    owner: TypeId,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if let Some(kind) = reflected_type_attr_kind(name) {
        return materialize_attr_value_static(owner, name, kind).map(Some);
    }

    builtin_mapping_proxy_attribute_value_static(owner, name)
}

pub(crate) fn builtin_bound_type_attribute_value(
    vm: &mut VirtualMachine,
    owner: TypeId,
    owner_value: Value,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if owner == TypeId::OBJECT && name.as_str() == "__new__" {
        return Ok(builtin_type_method_value(owner, name.as_str()));
    }

    if let Some(method) = builtin_type_bound_method_value(owner, name.as_str()) {
        let ptr = method
            .as_object_ptr()
            .expect("builtin type method values must be heap-allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        let bound = Box::leak(Box::new(builtin.bind(owner_value)));
        return Ok(Some(Value::object_ptr(
            bound as *mut BuiltinFunctionObject as *const (),
        )));
    }

    if let Some(method) = builtin_type_method_value(owner, name.as_str()) {
        return Ok(Some(bind_builtin_type_method_if_needed(
            method,
            owner_value,
        )));
    }

    let Some(value) = builtin_type_attribute_value(vm, owner, name)? else {
        return Ok(None);
    };

    Ok(Some(bind_reflected_descriptor_if_needed(
        value,
        owner,
        owner_value,
        name,
    )))
}

pub(crate) fn builtin_type_object_attribute_value(
    vm: &mut VirtualMachine,
    owner: TypeId,
    owner_value: Value,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if let Some(value) = builtin_type_class_or_static_attribute_value(owner, owner_value, name) {
        return Ok(Some(value));
    }

    builtin_type_attribute_value(vm, owner, name)
}

pub(crate) fn builtin_bound_type_attribute_value_static(
    owner: TypeId,
    owner_value: Value,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if owner == TypeId::OBJECT && name.as_str() == "__new__" {
        return Ok(builtin_type_method_value(owner, name.as_str()));
    }

    if let Some(method) = builtin_type_bound_method_value(owner, name.as_str()) {
        let ptr = method
            .as_object_ptr()
            .expect("builtin type method values must be heap-allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        let bound = Box::leak(Box::new(builtin.bind(owner_value)));
        return Ok(Some(Value::object_ptr(
            bound as *mut BuiltinFunctionObject as *const (),
        )));
    }

    if let Some(method) = builtin_type_method_value(owner, name.as_str()) {
        return Ok(Some(bind_builtin_type_method_if_needed(
            method,
            owner_value,
        )));
    }

    let Some(value) = builtin_type_attribute_value_static(owner, name)? else {
        return Ok(None);
    };

    Ok(Some(bind_reflected_descriptor_if_needed(
        value,
        owner,
        owner_value,
        name,
    )))
}

pub(crate) fn builtin_type_class_or_static_attribute_value_static(
    owner: TypeId,
    owner_value: Value,
    name: &InternedString,
) -> Option<Value> {
    builtin_type_class_or_static_attribute_value(owner, owner_value, name)
}

fn builtin_type_class_or_static_attribute_value(
    owner: TypeId,
    owner_value: Value,
    name: &InternedString,
) -> Option<Value> {
    if let Some(method) = builtin_type_bound_method_value(owner, name.as_str()) {
        let ptr = method
            .as_object_ptr()
            .expect("builtin type method values must be heap-allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        let bound = Box::leak(Box::new(builtin.bind(owner_value)));
        return Some(Value::object_ptr(
            bound as *mut BuiltinFunctionObject as *const (),
        ));
    }

    builtin_type_static_method_value(owner, name.as_str())
}

#[inline]
fn bind_builtin_type_method_if_needed(method: Value, owner_value: Value) -> Value {
    if should_bind_builtin_type_callable(owner_value) {
        crate::ops::objects::bind_cached_builtin_method(method, owner_value)
    } else {
        method
    }
}

#[inline]
fn bind_reflected_descriptor_if_needed(
    value: Value,
    owner: TypeId,
    owner_value: Value,
    name: &InternedString,
) -> Value {
    if !should_bind_builtin_type_callable(owner_value) {
        return value;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };
    let descriptor_type = crate::ops::objects::extract_type_id(ptr);
    let Some(target) = reflected_descriptor_callable_value(descriptor_type, owner, name) else {
        return value;
    };

    crate::ops::objects::bind_cached_builtin_method(target, owner_value)
}

#[inline]
fn should_bind_builtin_type_callable(owner_value: Value) -> bool {
    !owner_value.as_object_ptr().is_some_and(|ptr| {
        matches!(
            crate::ops::objects::extract_type_id(ptr),
            TypeId::TYPE | TypeId::EXCEPTION_TYPE
        )
    })
}

#[inline]
pub(crate) fn builtin_type_has_attribute(owner: TypeId, name: &InternedString) -> bool {
    reflected_type_attr_kind(name).is_some() || reflected_builtin_attr_kind(owner, name).is_some()
}

pub(crate) fn heap_type_attribute_value(
    vm: &mut VirtualMachine,
    class_ptr: *const PyClassObject,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let class = unsafe { &*class_ptr };
    user_type_attribute_value(vm, class, class_ptr, name)
}

pub(crate) fn heap_type_attribute_value_static(
    class_ptr: *const PyClassObject,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let class = unsafe { &*class_ptr };
    user_type_attribute_value_static(class, class_ptr, name)
}

#[inline]
pub(crate) fn heap_type_has_attribute(
    class_ptr: *const PyClassObject,
    name: &InternedString,
) -> bool {
    user_type_has_attribute(unsafe { &*class_ptr }, name)
}

pub(crate) fn builtin_mapping_proxy_get_item(
    vm: &mut VirtualMachine,
    proxy: &MappingProxyObject,
    key: Value,
) -> Result<Option<Value>, RuntimeError> {
    match proxy.source() {
        prism_runtime::object::views::MappingProxySource::Dict(dict_value) => {
            Ok(mapping_proxy_backing_dict(dict_value)?.get(key))
        }
        prism_runtime::object::views::MappingProxySource::BuiltinType(owner) => {
            let name = key_to_name(key)?;
            builtin_mapping_proxy_attribute_value(vm, owner, &name)
        }
        prism_runtime::object::views::MappingProxySource::UserClass(class_ptr) => {
            let name = key_to_name(key)?;
            Ok(unsafe { &*(class_ptr as *const PyClassObject) }.get_attr(&name))
        }
    }
}

pub(crate) fn builtin_mapping_proxy_get_item_static(
    proxy: &MappingProxyObject,
    key: Value,
) -> Result<Option<Value>, RuntimeError> {
    match proxy.source() {
        prism_runtime::object::views::MappingProxySource::Dict(dict_value) => {
            Ok(mapping_proxy_backing_dict(dict_value)?.get(key))
        }
        prism_runtime::object::views::MappingProxySource::BuiltinType(owner) => {
            let name = key_to_name(key)?;
            builtin_mapping_proxy_attribute_value_static(owner, &name)
        }
        prism_runtime::object::views::MappingProxySource::UserClass(class_ptr) => {
            let name = key_to_name(key)?;
            Ok(unsafe { &*(class_ptr as *const PyClassObject) }.get_attr(&name))
        }
    }
}

pub(crate) fn builtin_mapping_proxy_contains_key(
    proxy: &MappingProxyObject,
    key: Value,
) -> Result<bool, RuntimeError> {
    match proxy.source() {
        prism_runtime::object::views::MappingProxySource::Dict(dict_value) => {
            Ok(mapping_proxy_backing_dict(dict_value)?.contains_key(key))
        }
        prism_runtime::object::views::MappingProxySource::BuiltinType(owner) => {
            let name = key_to_name(key)?;
            Ok(reflected_builtin_attr_kind(owner, &name).is_some())
        }
        prism_runtime::object::views::MappingProxySource::UserClass(class_ptr) => {
            let name = key_to_name(key)?;
            Ok(unsafe { &*(class_ptr as *const PyClassObject) }.has_attr(&name))
        }
    }
}

pub(crate) fn builtin_mapping_proxy_entries_static(
    proxy: &MappingProxyObject,
) -> Result<Vec<(Value, Value)>, RuntimeError> {
    match proxy.source() {
        prism_runtime::object::views::MappingProxySource::Dict(dict_value) => {
            Ok(mapping_proxy_backing_dict(dict_value)?.iter().collect())
        }
        prism_runtime::object::views::MappingProxySource::BuiltinType(owner) => {
            let names = builtin_mapping_proxy_names(owner);
            let mut entries = Vec::with_capacity(names.len());
            for name in names {
                if let Some(value) = builtin_mapping_proxy_attribute_value_static(owner, &name)? {
                    entries.push((Value::string(name), value));
                }
            }
            Ok(entries)
        }
        prism_runtime::object::views::MappingProxySource::UserClass(class_ptr) => {
            let class = unsafe { &*(class_ptr as *const PyClassObject) };
            let mut entries = Vec::new();
            class.for_each_attr(|name, value| entries.push((Value::string(name.clone()), value)));
            Ok(entries)
        }
    }
}

#[inline]
fn mapping_proxy_backing_dict(dict_value: Value) -> Result<&'static DictObject, RuntimeError> {
    let ptr = dict_value
        .as_object_ptr()
        .ok_or_else(|| RuntimeError::internal("mappingproxy dict source must be a heap object"))?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::DICT {
        return Err(RuntimeError::internal(
            "mappingproxy dict source must be a dict object",
        ));
    }
    Ok(unsafe { &*(ptr as *const DictObject) })
}

#[inline]
pub(crate) fn builtin_mapping_proxy_keys(
    proxy: &MappingProxyObject,
) -> Result<Vec<Value>, RuntimeError> {
    builtin_mapping_proxy_entries_static(proxy)
        .map(|entries| entries.into_iter().map(|(key, _)| key).collect::<Vec<_>>())
}

#[inline]
pub(crate) fn builtin_mapping_proxy_len(proxy: &MappingProxyObject) -> Result<usize, RuntimeError> {
    builtin_mapping_proxy_entries_static(proxy).map(|entries| entries.len())
}

pub(crate) fn builtin_instance_attribute_value(
    vm: &mut VirtualMachine,
    type_id: TypeId,
    receiver: Value,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    match (type_id, name.as_str()) {
        (owner, "__class__") if owner.raw() < TypeId::FIRST_USER_TYPE => Ok(Some(
            crate::builtins::builtin_type_object_for_type_id(owner),
        )),
        (owner, "__doc__") if owner.raw() < TypeId::FIRST_USER_TYPE => {
            Ok(Some(builtin_type_doc_value(owner)))
        }
        (TypeId::INT, "real") | (TypeId::INT, "numerator") => Ok(Some(receiver)),
        (TypeId::INT, "imag") => Ok(Some(Value::int(0).expect("zero should fit"))),
        (TypeId::INT, "denominator") => Ok(Some(Value::int(1).expect("one should fit"))),
        (TypeId::BOOL, "real") | (TypeId::BOOL, "numerator") => Ok(Some(
            Value::int(i64::from(
                receiver
                    .as_bool()
                    .expect("bool receiver should expose bool payload"),
            ))
            .expect("bool integer projection should fit"),
        )),
        (TypeId::BOOL, "imag") => Ok(Some(Value::int(0).expect("zero should fit"))),
        (TypeId::BOOL, "denominator") => Ok(Some(Value::int(1).expect("one should fit"))),
        (TypeId::MEMORYVIEW, attr) => memoryview_instance_attr_value(vm, receiver, attr),
        (TypeId::OBJECT, "__str__") => alloc_view(
            vm,
            MethodWrapperObject::new(TypeId::OBJECT, name.clone(), receiver),
            "method wrapper view",
        )
        .map(Some),
        _ => Ok(None),
    }
}

#[inline]
pub(crate) fn builtin_instance_has_attribute(type_id: TypeId, name: &InternedString) -> bool {
    if name.as_str() == "__class__" && type_id.raw() < TypeId::FIRST_USER_TYPE {
        return true;
    }

    if name.as_str() == "__doc__" && type_id.raw() < TypeId::FIRST_USER_TYPE {
        return true;
    }

    matches!(
        (type_id, name.as_str()),
        (TypeId::OBJECT, "__str__")
            | (TypeId::INT, "real" | "imag" | "numerator" | "denominator")
            | (TypeId::BOOL, "real" | "imag" | "numerator" | "denominator")
            | (
                TypeId::MEMORYVIEW,
                "format"
                    | "itemsize"
                    | "ndim"
                    | "shape"
                    | "strides"
                    | "nbytes"
                    | "readonly"
                    | "obj",
            )
    )
}

fn memoryview_instance_attr_value(
    vm: &mut VirtualMachine,
    receiver: Value,
    attr: &str,
) -> Result<Option<Value>, RuntimeError> {
    let Some(view) = value_as_memoryview_ref(receiver) else {
        return Ok(None);
    };
    if view.released() {
        return Err(RuntimeError::value_error(
            "operation forbidden on released memoryview object",
        ));
    }

    match attr {
        "format" => Ok(Some(Value::string(intern(view.format_str())))),
        "itemsize" => Ok(Some(usize_attr_value(view.item_size())?)),
        "ndim" => Ok(Some(usize_attr_value(view.ndim())?)),
        "shape" => Ok(Some(memoryview_tuple_attr(
            vm,
            view.shape(),
            "memoryview shape",
        )?)),
        "strides" => Ok(Some(memoryview_tuple_attr(
            vm,
            &view.strides(),
            "memoryview strides",
        )?)),
        "nbytes" => Ok(Some(usize_attr_value(view.nbytes())?)),
        "readonly" => Ok(Some(Value::bool(view.readonly()))),
        "obj" => Ok(Some(view.source())),
        _ => Ok(None),
    }
}

fn memoryview_tuple_attr(
    vm: &mut VirtualMachine,
    values: &[usize],
    context: &'static str,
) -> Result<Value, RuntimeError> {
    let mut items = Vec::with_capacity(values.len());
    for &value in values {
        items.push(usize_attr_value(value)?);
    }
    alloc_view(vm, TupleObject::from_vec(items), context)
}

fn usize_attr_value(value: usize) -> Result<Value, RuntimeError> {
    let value = i64::try_from(value).map_err(|_| {
        RuntimeError::new(RuntimeErrorKind::OverflowError {
            message: "attribute value does not fit in int".into(),
        })
    })?;
    Value::int(value).ok_or_else(|| {
        RuntimeError::new(RuntimeErrorKind::OverflowError {
            message: "attribute value does not fit in int".into(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::builtin_type_object_for_type_id;
    use prism_core::intern::intern;
    use prism_runtime::object::PyObject;
    use prism_runtime::object::class::PyClassObject;
    use prism_runtime::object::type_builtins::{
        SubclassBitmap, register_global_class, unregister_global_class,
    };
    use prism_runtime::object::views::{
        DescriptorViewObject, MappingProxyObject, MethodWrapperObject,
    };
    use prism_runtime::types::dict::DictObject;
    use prism_runtime::types::list::ListObject;
    use prism_runtime::types::tuple::TupleObject;
    use std::sync::Arc;

    #[test]
    fn test_builtin_type_attr_registry_covers_types_module_surface() {
        assert!(builtin_type_has_attribute(
            TypeId::TYPE,
            &intern("__dict__")
        ));
        assert!(builtin_type_has_attribute(TypeId::TYPE, &intern("__doc__")));
        assert!(builtin_type_has_attribute(
            TypeId::DICT,
            &intern("__name__")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::DICT,
            &intern("__bases__")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::GENERATOR,
            &intern("__mro__")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::OBJECT,
            &intern("__init__")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::OBJECT,
            &intern("__setattr__")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::OBJECT,
            &intern("__delattr__")
        ));
        assert!(builtin_type_has_attribute(TypeId::TYPE, &intern("__new__")));
        assert!(builtin_type_has_attribute(
            TypeId::OBJECT,
            &intern("__new__")
        ));
        assert!(builtin_type_has_attribute(TypeId::INT, &intern("__new__")));
        assert!(builtin_type_has_attribute(
            TypeId::FLOAT,
            &intern("__new__")
        ));
        assert!(builtin_type_has_attribute(TypeId::STR, &intern("__new__")));
        assert!(builtin_type_has_attribute(TypeId::BOOL, &intern("__new__")));
        assert!(builtin_type_has_attribute(
            TypeId::BYTES,
            &intern("__new__")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::BYTEARRAY,
            &intern("__new__")
        ));
        assert!(builtin_type_has_attribute(TypeId::LIST, &intern("__new__")));
        assert!(builtin_type_has_attribute(
            TypeId::TUPLE,
            &intern("__new__")
        ));
        assert!(builtin_type_has_attribute(TypeId::DICT, &intern("__new__")));
        assert!(builtin_type_has_attribute(TypeId::SET, &intern("__new__")));
        assert!(builtin_type_has_attribute(
            TypeId::FROZENSET,
            &intern("__new__")
        ));
        assert!(builtin_type_has_attribute(TypeId::STR, &intern("join")));
        assert!(builtin_type_has_attribute(TypeId::STR, &intern("replace")));
        assert!(builtin_type_has_attribute(
            TypeId::STR,
            &intern("maketrans")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::DICT,
            &intern("fromkeys")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::DICT,
            &intern("__setitem__")
        ));
        assert!(builtin_type_has_attribute(TypeId::DICT, &intern("pop")));
        assert!(builtin_type_has_attribute(
            TypeId::INT,
            &intern("bit_length")
        ));
        assert!(builtin_type_has_attribute(TypeId::INT, &intern("__add__")));
        assert!(builtin_type_has_attribute(
            TypeId::INT,
            &intern("__index__")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::INT,
            &intern("bit_count")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::FUNCTION,
            &intern("__code__")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::FUNCTION,
            &intern("__globals__")
        ));
    }

    #[test]
    fn test_builtin_instance_attr_registry_covers_method_wrapper_surface() {
        assert!(builtin_instance_has_attribute(
            TypeId::NONE,
            &intern("__doc__")
        ));
        assert!(builtin_instance_has_attribute(
            TypeId::LIST,
            &intern("__class__")
        ));
        assert!(builtin_instance_has_attribute(
            TypeId::OBJECT,
            &intern("__str__")
        ));
        assert!(!builtin_instance_has_attribute(
            TypeId::OBJECT,
            &intern("__repr__")
        ));
    }

    #[test]
    fn test_builtin_instance_attribute_value_exposes_builtin_class() {
        let mut vm = VirtualMachine::new();
        let list_ptr = Box::into_raw(Box::new(ListObject::new()));
        let list_value = Value::object_ptr(list_ptr as *const ());

        let class = builtin_instance_attribute_value(
            &mut vm,
            TypeId::LIST,
            list_value,
            &intern("__class__"),
        )
        .expect("list.__class__ lookup should succeed")
        .expect("list instances should expose __class__");

        assert_eq!(class, builtin_type_object_for_type_id(TypeId::LIST));

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_builtin_instance_attribute_value_exposes_none_doc() {
        let mut vm = VirtualMachine::new();
        let doc = builtin_instance_attribute_value(
            &mut vm,
            TypeId::NONE,
            Value::none(),
            &intern("__doc__"),
        )
        .expect("None.__doc__ lookup should succeed")
        .expect("None.__doc__ should exist");

        assert_eq!(doc, Value::string(intern(NONE_TYPE_DOC)));
    }

    #[test]
    fn test_builtin_instance_attribute_value_exposes_primitive_type_doc() {
        let mut vm = VirtualMachine::new();
        let doc = builtin_instance_attribute_value(
            &mut vm,
            TypeId::STR,
            Value::string(intern("seed")),
            &intern("__doc__"),
        )
        .expect("str.__doc__ lookup should succeed")
        .expect("str instances should inherit type documentation");

        assert_eq!(doc, Value::string(intern("Text string type.")));
    }

    #[test]
    fn test_builtin_mapping_proxy_names_include_iterator_protocol_methods() {
        let names = builtin_mapping_proxy_names(TypeId::ITERATOR);
        assert!(names.contains(&intern("__iter__")));
        assert!(names.contains(&intern("__next__")));
    }

    #[test]
    fn test_builtin_mapping_proxy_names_include_tuple_sequence_methods() {
        let names = builtin_mapping_proxy_names(TypeId::TUPLE);
        assert!(names.contains(&intern("__iter__")));
        assert!(names.contains(&intern("__len__")));
        assert!(names.contains(&intern("__getitem__")));
        assert!(names.contains(&intern("count")));
        assert!(names.contains(&intern("index")));
    }

    #[test]
    fn test_builtin_mapping_proxy_names_include_regex_match_subscription() {
        let names = builtin_mapping_proxy_names(TypeId::REGEX_MATCH);
        assert!(names.contains(&intern("__getitem__")));
    }

    #[test]
    fn test_builtin_mapping_proxy_names_include_int_bit_operations() {
        let names = builtin_mapping_proxy_names(TypeId::INT);
        assert!(names.contains(&intern("__add__")));
        assert!(names.contains(&intern("__index__")));
        assert!(names.contains(&intern("bit_length")));
        assert!(names.contains(&intern("bit_count")));
    }

    #[test]
    fn test_mapping_proxy_source_round_trip() {
        let proxy = MappingProxyObject::for_builtin_type(TypeId::DICT);
        assert_eq!(
            proxy.source(),
            prism_runtime::object::views::MappingProxySource::BuiltinType(TypeId::DICT)
        );
    }

    #[test]
    fn test_mapping_proxy_supports_heap_class_source() {
        let class = Arc::new(PyClassObject::new_simple(intern("ProxyClass")));
        let proxy = MappingProxyObject::for_user_class(Arc::as_ptr(&class));
        assert_eq!(
            proxy.source(),
            prism_runtime::object::views::MappingProxySource::UserClass(
                Arc::as_ptr(&class) as usize
            )
        );
    }

    #[test]
    fn test_mapping_proxy_entry_helpers_cover_heap_class_contents() {
        let class = Arc::new(PyClassObject::new_simple(intern("ProxyEntries")));
        class.set_attr(intern("token"), Value::int(7).unwrap());
        class.set_attr(intern("label"), Value::string(intern("ready")));
        let proxy = MappingProxyObject::for_user_class(Arc::as_ptr(&class));

        let mut keys = builtin_mapping_proxy_keys(&proxy)
            .expect("keys should materialize")
            .into_iter()
            .map(|value| {
                let ptr = value
                    .as_string_object_ptr()
                    .expect("mappingproxy keys should be interned strings");
                interned_by_ptr(ptr as *const u8)
                    .expect("interned string pointer should resolve")
                    .as_str()
                    .to_string()
            })
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(keys, vec!["label".to_string(), "token".to_string()]);

        let entries = builtin_mapping_proxy_entries_static(&proxy).expect("entries should exist");
        assert_eq!(entries.len(), 2);
        assert_eq!(
            builtin_mapping_proxy_len(&proxy).expect("len should succeed"),
            2
        );
    }

    #[test]
    fn test_builtin_mapping_proxy_exposes_core_new_descriptors() {
        for owner in [
            TypeId::TYPE,
            TypeId::OBJECT,
            TypeId::INT,
            TypeId::FLOAT,
            TypeId::STR,
            TypeId::BOOL,
            TypeId::LIST,
            TypeId::TUPLE,
            TypeId::DICT,
            TypeId::SET,
            TypeId::FROZENSET,
        ] {
            let proxy = MappingProxyObject::for_builtin_type(owner);
            assert!(
                builtin_mapping_proxy_contains_key(&proxy, Value::string(intern("__new__")))
                    .expect("membership should succeed"),
                "{owner:?}.__dict__ should expose __new__"
            );

            let value =
                builtin_mapping_proxy_get_item_static(&proxy, Value::string(intern("__new__")))
                    .expect("subscript should succeed")
                    .expect("__new__ should exist");
            let ptr = value
                .as_object_ptr()
                .expect("__new__ should materialize as a descriptor");
            let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
            assert_eq!(header.type_id, TypeId::WRAPPER_DESCRIPTOR);
        }
    }

    #[test]
    fn test_descriptor_view_uses_expected_runtime_type_ids() {
        let method =
            DescriptorViewObject::new(TypeId::METHOD_DESCRIPTOR, TypeId::STR, intern("join"));
        let classmethod = DescriptorViewObject::new(
            TypeId::CLASSMETHOD_DESCRIPTOR,
            TypeId::DICT,
            intern("fromkeys"),
        );
        assert_eq!(method.header().type_id, TypeId::METHOD_DESCRIPTOR);
        assert_eq!(classmethod.header().type_id, TypeId::CLASSMETHOD_DESCRIPTOR);
    }

    #[test]
    fn test_mapping_proxy_entries_support_dict_backing() {
        let mut dict = DictObject::new();
        dict.set(Value::string(intern("alpha")), Value::int(1).unwrap());
        dict.set(Value::string(intern("beta")), Value::int(2).unwrap());
        let dict_value = leak_object_value(dict);
        let proxy = MappingProxyObject::for_mapping(dict_value);

        let entries = builtin_mapping_proxy_entries_static(&proxy)
            .expect("dict-backed mappingproxy should expose entries");
        assert_eq!(
            entries,
            vec![
                (Value::string(intern("alpha")), Value::int(1).unwrap()),
                (Value::string(intern("beta")), Value::int(2).unwrap()),
            ]
        );
        assert!(
            builtin_mapping_proxy_contains_key(&proxy, Value::string(intern("alpha")))
                .expect("contains should succeed")
        );
        assert_eq!(
            builtin_mapping_proxy_get_item_static(&proxy, Value::string(intern("beta")))
                .expect("lookup should succeed"),
            Some(Value::int(2).unwrap())
        );
    }

    #[test]
    fn test_method_wrapper_uses_expected_runtime_type_id() {
        let wrapper = MethodWrapperObject::new(
            TypeId::OBJECT,
            intern("__str__"),
            builtin_type_object_for_type_id(TypeId::OBJECT),
        );
        assert_eq!(wrapper.header().type_id, TypeId::METHOD_WRAPPER);
    }

    #[test]
    fn test_builtin_type_method_value_exposes_dict_fromkeys_callable() {
        let value = builtin_type_method_value(TypeId::DICT, "fromkeys")
            .expect("dict.fromkeys should resolve");
        let ptr = value
            .as_object_ptr()
            .expect("method should be heap allocated");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
    }

    #[test]
    fn test_builtin_type_method_value_exposes_str_maketrans_callable() {
        let value = builtin_type_method_value(TypeId::STR, "maketrans")
            .expect("str.maketrans should resolve");
        let ptr = value
            .as_object_ptr()
            .expect("method should be heap allocated");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
    }

    #[test]
    fn test_builtin_type_method_value_exposes_bytes_maketrans_callable() {
        for owner in [TypeId::BYTES, TypeId::BYTEARRAY] {
            let value = builtin_type_method_value(owner, "maketrans")
                .expect("bytes-like maketrans should resolve");
            let ptr = value
                .as_object_ptr()
                .expect("method should be heap allocated");
            let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
            assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
        }
    }

    #[test]
    fn test_builtin_type_method_value_exposes_float_getformat_callable() {
        let value = builtin_type_method_value(TypeId::FLOAT, "__getformat__")
            .expect("float.__getformat__ should resolve");
        let ptr = value
            .as_object_ptr()
            .expect("method should be heap allocated");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
    }

    #[test]
    fn test_builtin_type_method_value_exposes_unbound_dict_setitem_callable() {
        let value = builtin_type_method_value(TypeId::DICT, "__setitem__")
            .expect("dict.__setitem__ should resolve");
        let ptr = value
            .as_object_ptr()
            .expect("method should be heap allocated");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
    }

    #[test]
    fn test_builtin_type_method_value_exposes_core_new_callables() {
        for owner in [
            TypeId::TYPE,
            TypeId::OBJECT,
            TypeId::INT,
            TypeId::FLOAT,
            TypeId::STR,
            TypeId::BOOL,
            TypeId::LIST,
            TypeId::TUPLE,
            TypeId::DICT,
            TypeId::SET,
            TypeId::FROZENSET,
        ] {
            let value = builtin_type_method_value(owner, "__new__")
                .unwrap_or_else(|| panic!("{owner:?}.__new__ should resolve"));
            let ptr = value
                .as_object_ptr()
                .expect("method should be heap allocated");
            let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
            assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
        }
    }

    #[test]
    fn test_reflected_descriptor_callable_value_exposes_object_init_builtin() {
        let value = reflected_descriptor_callable_value(
            TypeId::WRAPPER_DESCRIPTOR,
            TypeId::OBJECT,
            &intern("__init__"),
        )
        .expect("object.__init__ descriptor should resolve to a callable");
        let ptr = value
            .as_object_ptr()
            .expect("callable should be heap allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "object.__init__");
    }

    #[test]
    fn test_reflected_descriptor_callable_value_exposes_type_init_builtin() {
        let value = reflected_descriptor_callable_value(
            TypeId::WRAPPER_DESCRIPTOR,
            TypeId::TYPE,
            &intern("__init__"),
        )
        .expect("type.__init__ descriptor should resolve to a callable");
        let ptr = value
            .as_object_ptr()
            .expect("callable should be heap allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "type.__init__");
        assert!(
            builtin
                .call(&[builtin_type_object_for_type_id(TypeId::TYPE)])
                .expect("type.__init__ should accept an already-initialized type")
                .is_none()
        );
    }

    #[test]
    fn test_reflected_descriptor_callable_value_exposes_int_add_builtin() {
        let value = reflected_descriptor_callable_value(
            TypeId::WRAPPER_DESCRIPTOR,
            TypeId::INT,
            &intern("__add__"),
        )
        .expect("int.__add__ descriptor should resolve to a callable");
        let ptr = value
            .as_object_ptr()
            .expect("callable should be heap allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "int.__add__");
    }

    #[test]
    fn test_reflected_descriptor_callable_value_exposes_object_init_subclass_builtin() {
        let value = reflected_descriptor_callable_value(
            TypeId::CLASSMETHOD_DESCRIPTOR,
            TypeId::OBJECT,
            &intern("__init_subclass__"),
        )
        .expect("object.__init_subclass__ descriptor should resolve to a callable");
        let ptr = value
            .as_object_ptr()
            .expect("callable should be heap allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "object.__init_subclass__");
    }

    #[test]
    fn test_reflected_descriptor_callable_value_exposes_type_prepare_builtin() {
        let value = reflected_descriptor_callable_value(
            TypeId::CLASSMETHOD_DESCRIPTOR,
            TypeId::TYPE,
            &intern("__prepare__"),
        )
        .expect("type.__prepare__ descriptor should resolve to a callable");
        let ptr = value
            .as_object_ptr()
            .expect("callable should be heap allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "type.__prepare__");
    }

    #[test]
    fn test_builtin_bound_type_attribute_value_binds_type_prepare_receiver() {
        let mut vm = VirtualMachine::new();
        let type_type = builtin_type_object_for_type_id(TypeId::TYPE);
        let method = builtin_bound_type_attribute_value(
            &mut vm,
            TypeId::TYPE,
            type_type,
            &intern("__prepare__"),
        )
        .expect("binding should succeed")
        .expect("bound method should exist");

        let method_ptr = method
            .as_object_ptr()
            .expect("bound method should be allocated");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        let bases_ptr = Box::into_raw(Box::new(TupleObject::empty()));
        let result = builtin
            .call_with_keywords(
                &[
                    Value::string(intern("Prepared")),
                    Value::object_ptr(bases_ptr as *const ()),
                ],
                &[("flag", Value::bool(true))],
            )
            .expect("bound type.__prepare__ should be callable");
        let result_ptr = result.as_object_ptr().expect("result should be a dict");
        assert_eq!(
            unsafe { &*(result_ptr as *const prism_runtime::object::ObjectHeader) }.type_id,
            TypeId::DICT
        );

        unsafe {
            drop(Box::from_raw(result_ptr as *mut DictObject));
            drop(Box::from_raw(bases_ptr));
        }
    }

    #[test]
    fn test_builtin_bound_type_attribute_value_binds_dict_fromkeys_receiver() {
        let mut vm = VirtualMachine::new();
        let dict_type = builtin_type_object_for_type_id(TypeId::DICT);
        let method = builtin_bound_type_attribute_value(
            &mut vm,
            TypeId::DICT,
            dict_type,
            &intern("fromkeys"),
        )
        .expect("binding should succeed")
        .expect("bound method should exist");

        let method_ptr = method
            .as_object_ptr()
            .expect("bound method should be allocated");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        let keys_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ])));
        let result = builtin
            .call(&[Value::object_ptr(keys_ptr as *const ())])
            .expect("bound dict.fromkeys should be callable");
        let result_ptr = result.as_object_ptr().expect("result should be a dict");
        let dict = unsafe { &*(result_ptr as *const DictObject) };
        assert!(dict.get(Value::int(1).unwrap()).unwrap().is_none());
        assert!(dict.get(Value::int(2).unwrap()).unwrap().is_none());

        unsafe {
            drop(Box::from_raw(result_ptr as *mut DictObject));
            drop(Box::from_raw(keys_ptr));
        }
    }

    #[test]
    fn test_builtin_bound_type_attribute_value_binds_float_getformat_receiver() {
        let mut vm = VirtualMachine::new();
        let float_type = builtin_type_object_for_type_id(TypeId::FLOAT);
        let method = builtin_bound_type_attribute_value(
            &mut vm,
            TypeId::FLOAT,
            float_type,
            &intern("__getformat__"),
        )
        .expect("binding should succeed")
        .expect("bound method should exist");

        let method_ptr = method
            .as_object_ptr()
            .expect("bound method should be allocated");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        let result = builtin
            .call(&[Value::string(intern("double"))])
            .expect("bound float.__getformat__ should be callable");
        assert!(result.is_string());
    }

    #[test]
    fn test_builtin_bound_type_attribute_value_returns_unbound_dict_setitem() {
        let mut vm = VirtualMachine::new();
        let dict_type = builtin_type_object_for_type_id(TypeId::DICT);
        let method = builtin_bound_type_attribute_value(
            &mut vm,
            TypeId::DICT,
            dict_type,
            &intern("__setitem__"),
        )
        .expect("lookup should succeed")
        .expect("method should exist");

        let method_ptr = method
            .as_object_ptr()
            .expect("method should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };

        let dict_ptr = Box::into_raw(Box::new(DictObject::new()));
        let dict_value = Value::object_ptr(dict_ptr as *const ());
        let key = Value::string(intern("ready"));
        builtin
            .call(&[dict_value, key, Value::int(1).unwrap()])
            .expect("dict.__setitem__ should accept an explicit receiver");

        let dict = unsafe { &*(dict_ptr as *const DictObject) };
        assert_eq!(dict.get(key).unwrap().as_int(), Some(1));

        unsafe {
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_builtin_bound_type_attribute_value_binds_reflected_object_init_for_instances() {
        let mut vm = VirtualMachine::new();
        let instance = crate::builtins::builtin_object(&[]).expect("object() should succeed");
        let method = builtin_bound_type_attribute_value(
            &mut vm,
            TypeId::OBJECT,
            instance,
            &intern("__init__"),
        )
        .expect("lookup should succeed")
        .expect("object.__init__ should exist");

        let method_ptr = method
            .as_object_ptr()
            .expect("bound method should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "object.__init__");
        assert_eq!(builtin.bound_self(), Some(instance));
        assert!(
            builtin
                .call(&[])
                .expect("bound object.__init__ should execute")
                .is_none()
        );
    }

    #[test]
    fn test_builtin_bound_type_attribute_value_binds_object_init_subclass_for_types() {
        let mut vm = VirtualMachine::new();
        let object_type = builtin_type_object_for_type_id(TypeId::OBJECT);
        let method = builtin_bound_type_attribute_value(
            &mut vm,
            TypeId::OBJECT,
            object_type,
            &intern("__init_subclass__"),
        )
        .expect("lookup should succeed")
        .expect("object.__init_subclass__ should exist");

        let method_ptr = method
            .as_object_ptr()
            .expect("bound method should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "object.__init_subclass__");
        assert_eq!(builtin.bound_self(), Some(object_type));
        assert!(
            builtin
                .call_with_keywords(&[], &[])
                .expect("bound object.__init_subclass__ should execute")
                .is_none()
        );
    }

    #[test]
    fn test_builtin_bound_type_attribute_value_static_binds_reflected_object_init_for_instances() {
        let instance =
            crate::builtins::builtin_object(&[]).expect("object() should produce an instance");
        let method = builtin_bound_type_attribute_value_static(
            TypeId::OBJECT,
            instance,
            &intern("__init__"),
        )
        .expect("lookup should succeed")
        .expect("object.__init__ should exist");

        let method_ptr = method
            .as_object_ptr()
            .expect("bound method should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "object.__init__");
        assert_eq!(builtin.bound_self(), Some(instance));
        assert!(
            builtin
                .call(&[])
                .expect("bound object.__init__ should execute")
                .is_none()
        );
    }

    #[test]
    fn test_builtin_bound_type_attribute_value_static_binds_reflected_object_eq_for_primitives() {
        let receiver = Value::int(7).unwrap();
        let method =
            builtin_bound_type_attribute_value_static(TypeId::OBJECT, receiver, &intern("__eq__"))
                .expect("lookup should succeed")
                .expect("object.__eq__ should exist");

        let method_ptr = method
            .as_object_ptr()
            .expect("bound method should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "object.__eq__");
        assert_eq!(builtin.bound_self(), Some(receiver));
        assert_eq!(
            builtin
                .call(&[Value::int(7).unwrap()])
                .expect("bound object.__eq__ should execute"),
            Value::bool(true)
        );
    }

    #[test]
    fn test_builtin_bound_type_attribute_value_materializes_doc_slot() {
        let mut vm = VirtualMachine::new();
        let type_object = builtin_type_object_for_type_id(TypeId::TYPE);
        let doc = builtin_bound_type_attribute_value(
            &mut vm,
            TypeId::TYPE,
            type_object,
            &intern("__doc__"),
        )
        .expect("lookup should succeed");

        assert_eq!(doc, Some(Value::none()));
    }

    #[test]
    fn test_builtin_type_attribute_value_materializes_none_doc_slot() {
        let mut vm = VirtualMachine::new();
        let doc = builtin_type_attribute_value(&mut vm, TypeId::NONE, &intern("__doc__"))
            .expect("type(None).__doc__ lookup should succeed")
            .expect("type(None).__doc__ should exist");

        assert_eq!(doc, Value::string(intern(NONE_TYPE_DOC)));
    }

    #[test]
    fn test_builtin_type_attribute_value_materializes_type_init_descriptor() {
        let mut vm = VirtualMachine::new();
        let descriptor = builtin_type_attribute_value(&mut vm, TypeId::TYPE, &intern("__init__"))
            .expect("type.__init__ lookup should succeed")
            .expect("type.__init__ should exist");
        let descriptor_ptr = descriptor
            .as_object_ptr()
            .expect("type.__init__ should be heap allocated");
        let header = unsafe { &*(descriptor_ptr as *const prism_runtime::object::ObjectHeader) };
        assert_eq!(header.type_id, TypeId::WRAPPER_DESCRIPTOR);

        let view = unsafe { &*(descriptor_ptr as *const DescriptorViewObject) };
        assert_eq!(view.owner(), TypeId::TYPE);
        assert_eq!(view.name().as_str(), "__init__");
    }

    #[test]
    fn test_builtin_type_attribute_value_materializes_mro_tuple() {
        let mut vm = VirtualMachine::new();
        let value = builtin_type_attribute_value(&mut vm, TypeId::BOOL, &intern("__mro__"))
            .expect("lookup should succeed")
            .expect("__mro__ should exist");
        let tuple_ptr = value.as_object_ptr().expect("mro should be a tuple object");
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        assert_eq!(tuple.len(), 3);
        assert_eq!(
            tuple.as_slice()[0],
            builtin_type_object_for_type_id(TypeId::BOOL)
        );
        assert_eq!(
            tuple.as_slice()[1],
            builtin_type_object_for_type_id(TypeId::INT)
        );
        assert_eq!(
            tuple.as_slice()[2],
            builtin_type_object_for_type_id(TypeId::OBJECT)
        );
    }

    #[test]
    fn test_builtin_type_attribute_value_materializes_name_and_bases() {
        let mut vm = VirtualMachine::new();

        let name_value = builtin_type_attribute_value(&mut vm, TypeId::DICT, &intern("__name__"))
            .expect("lookup should succeed")
            .expect("__name__ should exist");
        let name_ptr = name_value
            .as_string_object_ptr()
            .expect("__name__ should be an interned string");
        assert_eq!(
            interned_by_ptr(name_ptr as *const u8).unwrap().as_str(),
            "dict"
        );

        let bases_value = builtin_type_attribute_value(&mut vm, TypeId::BOOL, &intern("__bases__"))
            .expect("lookup should succeed")
            .expect("__bases__ should exist");
        let bases_ptr = bases_value
            .as_object_ptr()
            .expect("__bases__ should be a tuple object");
        let bases = unsafe { &*(bases_ptr as *const TupleObject) };
        assert_eq!(bases.len(), 1);
        assert_eq!(
            bases.as_slice()[0],
            builtin_type_object_for_type_id(TypeId::INT)
        );
    }

    #[test]
    fn test_builtin_type_has_attribute_reports_type_prepare() {
        assert!(builtin_type_has_attribute(
            TypeId::TYPE,
            &intern("__init__")
        ));
        assert!(builtin_type_has_attribute(
            TypeId::TYPE,
            &intern("__prepare__")
        ));
    }

    #[test]
    fn test_heap_type_attribute_value_materializes_mro_tuple_and_dict_proxy() {
        let class = Arc::new(PyClassObject::new_simple(intern("HeapReflect")));
        class.set_attr(intern("token"), Value::int(7).unwrap());

        let mut bitmap = SubclassBitmap::new();
        bitmap.set_bit(TypeId::OBJECT);
        bitmap.set_bit(class.class_type_id());
        register_global_class(Arc::clone(&class), bitmap);

        let class_ptr = Arc::as_ptr(&class);
        let mut vm = VirtualMachine::new();

        let mro_value = heap_type_attribute_value(&mut vm, class_ptr, &intern("__mro__"))
            .expect("lookup should succeed")
            .expect("__mro__ should exist");
        let mro_ptr = mro_value.as_object_ptr().expect("mro should be a tuple");
        let mro = unsafe { &*(mro_ptr as *const TupleObject) };
        assert_eq!(mro.len(), 2);
        assert_eq!(mro.as_slice()[0], Value::object_ptr(class_ptr as *const ()));
        assert_eq!(
            mro.as_slice()[1],
            builtin_type_object_for_type_id(TypeId::OBJECT)
        );

        let dict_value = heap_type_attribute_value(&mut vm, class_ptr, &intern("__dict__"))
            .expect("lookup should succeed")
            .expect("__dict__ should exist");
        let dict_ptr = dict_value
            .as_object_ptr()
            .expect("__dict__ should be a proxy");
        let proxy = unsafe { &*(dict_ptr as *const MappingProxyObject) };
        let token = builtin_mapping_proxy_get_item(&mut vm, proxy, Value::string(intern("token")))
            .expect("subscript should succeed")
            .expect("token should exist");
        assert_eq!(token.as_int(), Some(7));
        let token_static =
            builtin_mapping_proxy_get_item_static(proxy, Value::string(intern("token")))
                .expect("static subscript should succeed")
                .expect("token should exist");
        assert_eq!(token_static.as_int(), Some(7));
        assert!(
            builtin_mapping_proxy_contains_key(proxy, Value::string(intern("token")))
                .expect("membership should succeed")
        );
    }

    #[test]
    fn test_heap_type_attribute_value_materializes_mro_tuple_when_owner_registry_entry_is_absent() {
        let class = Arc::new(PyClassObject::new_simple(intern("HeapReflectDetached")));

        let mut bitmap = SubclassBitmap::new();
        bitmap.set_bit(TypeId::OBJECT);
        bitmap.set_bit(class.class_type_id());
        register_global_class(Arc::clone(&class), bitmap);
        unregister_global_class(class.class_id());

        let class_ptr = Arc::as_ptr(&class);
        let mut vm = VirtualMachine::new();

        let mro_value = heap_type_attribute_value(&mut vm, class_ptr, &intern("__mro__"))
            .expect("lookup should succeed")
            .expect("__mro__ should exist");
        let mro_ptr = mro_value.as_object_ptr().expect("mro should be a tuple");
        let mro = unsafe { &*(mro_ptr as *const TupleObject) };
        assert_eq!(mro.len(), 2);
        assert_eq!(mro.as_slice()[0], Value::object_ptr(class_ptr as *const ()));
        assert_eq!(
            mro.as_slice()[1],
            builtin_type_object_for_type_id(TypeId::OBJECT)
        );
    }

    #[test]
    fn test_heap_type_attribute_value_inherits_builtin_object_ne() {
        let class = Arc::new(PyClassObject::new_simple(intern("HeapComparable")));

        let mut bitmap = SubclassBitmap::new();
        bitmap.set_bit(TypeId::OBJECT);
        bitmap.set_bit(class.class_type_id());
        register_global_class(Arc::clone(&class), bitmap);

        let class_ptr = Arc::as_ptr(&class);
        let mut vm = VirtualMachine::new();
        let value = heap_type_attribute_value(&mut vm, class_ptr, &intern("__ne__"))
            .expect("lookup should succeed")
            .expect("__ne__ should resolve from object");

        let ptr = value
            .as_object_ptr()
            .expect("object.__ne__ should be materialized as a builtin");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "object.__ne__");
    }

    #[test]
    fn test_heap_type_attribute_value_materializes_type_metadata() {
        let class = Arc::new(PyClassObject::new_simple(intern("HeapMetadata")));
        class.set_attr(intern("__doc__"), Value::string(intern("heap docs")));
        class.set_attr(intern("__module__"), Value::string(intern("pkg.runtime")));
        class.set_attr(
            intern("__qualname__"),
            Value::string(intern("pkg.runtime.HeapMetadata")),
        );

        let mut bitmap = SubclassBitmap::new();
        bitmap.set_bit(TypeId::OBJECT);
        bitmap.set_bit(class.class_type_id());
        register_global_class(Arc::clone(&class), bitmap);

        let class_ptr = Arc::as_ptr(&class);
        let mut vm = VirtualMachine::new();

        let name = heap_type_attribute_value(&mut vm, class_ptr, &intern("__name__"))
            .expect("lookup should succeed")
            .expect("__name__ should exist");
        let name_ptr = name
            .as_string_object_ptr()
            .expect("__name__ should be an interned string");
        assert_eq!(
            interned_by_ptr(name_ptr as *const u8).unwrap().as_str(),
            "HeapMetadata"
        );

        let module = heap_type_attribute_value(&mut vm, class_ptr, &intern("__module__"))
            .expect("lookup should succeed")
            .expect("__module__ should exist");
        let module_ptr = module
            .as_string_object_ptr()
            .expect("__module__ should be an interned string");
        assert_eq!(
            interned_by_ptr(module_ptr as *const u8).unwrap().as_str(),
            "pkg.runtime"
        );

        let doc = heap_type_attribute_value(&mut vm, class_ptr, &intern("__doc__"))
            .expect("lookup should succeed")
            .expect("__doc__ should exist");
        let doc_ptr = doc
            .as_string_object_ptr()
            .expect("__doc__ should be an interned string");
        assert_eq!(
            interned_by_ptr(doc_ptr as *const u8).unwrap().as_str(),
            "heap docs"
        );

        let bases = heap_type_attribute_value(&mut vm, class_ptr, &intern("__bases__"))
            .expect("lookup should succeed")
            .expect("__bases__ should exist");
        let bases_ptr = bases
            .as_object_ptr()
            .expect("__bases__ should be a tuple object");
        let bases = unsafe { &*(bases_ptr as *const TupleObject) };
        assert_eq!(bases.len(), 1);
        assert_eq!(
            bases.as_slice()[0],
            builtin_type_object_for_type_id(TypeId::OBJECT)
        );
    }

    #[test]
    fn test_heap_type_attribute_value_defaults_doc_to_none() {
        let class = Arc::new(PyClassObject::new_simple(intern("HeapMetadataNoDoc")));

        let mut bitmap = SubclassBitmap::new();
        bitmap.set_bit(TypeId::OBJECT);
        bitmap.set_bit(class.class_type_id());
        register_global_class(Arc::clone(&class), bitmap);

        let class_ptr = Arc::as_ptr(&class);
        let mut vm = VirtualMachine::new();

        let doc = heap_type_attribute_value(&mut vm, class_ptr, &intern("__doc__"))
            .expect("lookup should succeed")
            .expect("__doc__ should exist");
        assert!(doc.is_none());
    }
}
