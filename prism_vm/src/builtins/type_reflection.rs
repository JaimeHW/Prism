use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::error::{RuntimeError, RuntimeErrorKind};
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::{
    builtin_class_mro, class_id_to_type_id, global_class, global_direct_subclasses,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{DescriptorViewObject, MappingProxyObject, MethodWrapperObject};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::bigint_to_value;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::memoryview::value_as_memoryview_ref;
use prism_runtime::types::range::RangeObject;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

static DICT_FROMKEYS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("dict.fromkeys"),
        super::types::builtin_dict_fromkeys_vm,
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
static TYPE_MRO_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("type.mro"), type_mro));
static TYPE_SUBCLASSES_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("type.__subclasses__"), type_subclasses));
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
static DICT_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("dict.__init__"),
        super::types::builtin_dict_init_vm_kw,
    )
});
static SET_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("set.__new__"), super::types::builtin_set_new)
});
static SET_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.__init__"), super::types::builtin_set_init_vm)
});
static FROZENSET_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("frozenset.__new__"),
        super::types::builtin_frozenset_new,
    )
});
static FROZENSET_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("frozenset.__init__"),
        super::types::builtin_frozenset_init,
    )
});
static MODULE_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("module.__new__"),
        super::types::builtin_module_new,
    )
});
static ENUMERATE_NEW_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("enumerate.__new__"),
        super::itertools::builtin_enumerate_new_vm_kw,
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
    FlagsValue,
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
    AttrSpec {
        name: "mro",
        kind: ReflectedValueKind::MethodDescriptor,
    },
    AttrSpec {
        name: "__subclasses__",
        kind: ReflectedValueKind::MethodDescriptor,
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
        name: "__init__",
        kind: ReflectedValueKind::WrapperDescriptor,
    },
    AttrSpec {
        name: "__dict__",
        kind: ReflectedValueKind::DictProxy,
    },
    AttrSpec {
        name: "fromkeys",
        kind: ReflectedValueKind::ClassMethodDescriptor,
    },
];

const SET_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "__init__",
        kind: ReflectedValueKind::WrapperDescriptor,
    },
];

const FROZENSET_TYPE_ATTRS: &[AttrSpec] = &[
    NEW_WRAPPER_ATTR,
    AttrSpec {
        name: "__init__",
        kind: ReflectedValueKind::WrapperDescriptor,
    },
];

const MODULE_TYPE_ATTRS: &[AttrSpec] = &[NEW_WRAPPER_ATTR];

const ENUMERATE_TYPE_ATTRS: &[AttrSpec] = &[NEW_WRAPPER_ATTR];

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
    "__contains__",
    "__add__",
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
    "__init__",
    "__len__",
    "__contains__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__or__",
    "__ror__",
    "__ior__",
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
    "fromkeys",
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
const TUPLE_METHOD_NAMES: &[&str] = &[
    "__iter__",
    "__len__",
    "__getitem__",
    "__contains__",
    "count",
    "index",
];
const SLICE_METHOD_NAMES: &[&str] = &["__hash__", "indices"];
const ITERATOR_METHOD_NAMES: &[&str] = &["__iter__", "__next__", "__length_hint__"];
const GENERATOR_METHOD_NAMES: &[&str] = &["close"];
const PROPERTY_METHOD_NAMES: &[&str] = &[
    "__get__",
    "__set__",
    "__delete__",
    "__set_name__",
    "getter",
    "setter",
    "deleter",
];
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
        TypeId::ENUMERATE => ENUMERATE_TYPE_ATTRS,
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
        TypeId::SLICE => SLICE_METHOD_NAMES,
        TypeId::ITERATOR | TypeId::ENUMERATE => ITERATOR_METHOD_NAMES,
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
        "__flags__" => Some(ReflectedValueKind::FlagsValue),
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
        | ReflectedValueKind::ModuleString
        | ReflectedValueKind::FlagsValue => None,
    }
}

#[inline]
fn builtin_type_doc_value(owner: TypeId) -> Value {
    match builtin_type_doc(owner) {
        Some(doc) => Value::string(intern(doc)),
        None => Value::none(),
    }
}

pub(crate) const PY_TPFLAGS_HAVE_VECTORCALL: i64 = 1 << 11;
pub(crate) const PY_TPFLAGS_METHOD_DESCRIPTOR: i64 = 1 << 17;

#[inline]
fn python_flags_value(bits: i64) -> Value {
    Value::int(bits).expect("CPython-compatible type flags fit in Prism immediate ints")
}

#[inline]
fn python_type_flags_value_for_builtin(owner: TypeId) -> Value {
    python_flags_value(python_type_flags_for_builtin(owner))
}

#[inline]
fn python_type_flags_value_for_heap_type(class: &PyClassObject) -> Value {
    python_flags_value(python_type_flags_for_heap_type(class))
}

pub(crate) fn python_type_flags_for_type_value(value: Value) -> Option<i64> {
    let ptr = value.as_object_ptr()?;
    if crate::builtins::builtin_type_object_type_id(ptr).is_some() {
        return crate::builtins::builtin_type_object_type_id(ptr)
            .map(python_type_flags_for_builtin);
    }

    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    if header.type_id != TypeId::TYPE {
        return None;
    }

    let class = unsafe { &*(ptr as *const PyClassObject) };
    Some(python_type_flags_for_heap_type(class))
}

pub(crate) fn python_type_has_vectorcall_flag(value: Value) -> Option<bool> {
    python_type_flags_for_type_value(value).map(|flags| flags & PY_TPFLAGS_HAVE_VECTORCALL != 0)
}

fn python_type_flags_for_builtin(owner: TypeId) -> i64 {
    let mut flags = 0;
    if matches!(
        owner,
        TypeId::FUNCTION
            | TypeId::CLOSURE
            | TypeId::WRAPPER_DESCRIPTOR
            | TypeId::METHOD_DESCRIPTOR
            | TypeId::CLASSMETHOD_DESCRIPTOR
    ) {
        flags |= PY_TPFLAGS_METHOD_DESCRIPTOR;
    }
    flags
}

fn python_type_flags_for_heap_type(class: &PyClassObject) -> i64 {
    let mut flags = 0;
    if heap_type_has_effective_vectorcall(class) {
        flags |= PY_TPFLAGS_HAVE_VECTORCALL;
    }
    if class.flags().contains(ClassFlags::METHOD_DESCRIPTOR) {
        flags |= PY_TPFLAGS_METHOD_DESCRIPTOR;
    }
    flags
}

fn heap_type_has_effective_vectorcall(class: &PyClassObject) -> bool {
    if class_direct_call_overrides_inherited_vectorcall(class) {
        return false;
    }
    if class.flags().contains(ClassFlags::HAS_VECTORCALL) {
        return true;
    }

    class.mro().iter().copied().skip(1).any(|class_id| {
        if class_id.0 < TypeId::FIRST_USER_TYPE {
            return false;
        }
        global_class(class_id).is_some_and(|base| heap_type_has_effective_vectorcall(base.as_ref()))
    })
}

fn class_direct_call_overrides_inherited_vectorcall(class: &PyClassObject) -> bool {
    let Some(call) = class.get_attr(&intern("__call__")) else {
        return false;
    };

    if !class.flags().contains(ClassFlags::HAS_VECTORCALL) {
        return true;
    }

    !is_native_vectorcall_call_value(call)
}

fn is_native_vectorcall_call_value(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
    if header.type_id != TypeId::BUILTIN_FUNCTION {
        return false;
    }

    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    matches!(
        builtin.name(),
        "_testcapi.MethodDescriptor.__call__"
            | "_testcapi.MethodDescriptor2.__call__"
            | "_testcapi.VectorCallClass.__call__"
    )
}

#[inline]
fn builtin_instance_inherits_type_doc(type_id: TypeId) -> bool {
    type_id.raw() < TypeId::FIRST_USER_TYPE
        && !matches!(
            type_id,
            TypeId::FUNCTION
                | TypeId::CLOSURE
                | TypeId::METHOD
                | TypeId::BUILTIN_FUNCTION
                | TypeId::CLASSMETHOD
                | TypeId::STATICMETHOD
                | TypeId::PROPERTY
        )
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

fn type_mro(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "mro() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let class_value = args[0];
    let class_ptr = class_value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("descriptor 'mro' requires a type".to_string()))?;

    let mro_values = match crate::ops::objects::extract_type_id(class_ptr) {
        TypeId::TYPE => {
            if let Some(owner) = crate::builtins::builtin_type_object_type_id(class_ptr) {
                builtin_class_mro(owner)
                    .into_iter()
                    .map(class_id_to_type_value)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(super::runtime_error_to_builtin_error)?
            } else {
                let class = unsafe { &*(class_ptr as *const PyClassObject) };
                class
                    .mro()
                    .iter()
                    .copied()
                    .map(|class_id| class_id_to_type_value_for_owner(class_id, class, class_value))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(super::runtime_error_to_builtin_error)?
            }
        }
        TypeId::EXCEPTION_TYPE => {
            let class_id = crate::builtins::exception_proxy_class_id_from_ptr(class_ptr)
                .ok_or_else(|| {
                    BuiltinError::TypeError("descriptor 'mro' requires a type".to_string())
                })?;
            let class = global_class(class_id).ok_or_else(|| {
                BuiltinError::TypeError("descriptor 'mro' requires a type".to_string())
            })?;
            class
                .mro()
                .iter()
                .copied()
                .map(class_id_to_type_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(super::runtime_error_to_builtin_error)?
        }
        _ => {
            return Err(BuiltinError::TypeError(
                "descriptor 'mro' requires a type".to_string(),
            ));
        }
    };

    Ok(leak_object_value(ListObject::from_iter(mro_values)))
}

fn type_subclasses(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "type.__subclasses__() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let root = subclasses_root_type_id(args[0])?;
    let subclasses: Vec<Value> = global_direct_subclasses(root)
        .into_iter()
        .map(|class| Value::object_ptr(Arc::as_ptr(&class) as *const ()))
        .collect();

    Ok(leak_object_value(ListObject::from_iter(subclasses)))
}

fn subclasses_root_type_id(class_value: Value) -> Result<TypeId, BuiltinError> {
    let ptr = class_value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("descriptor '__subclasses__' requires a type".to_string())
    })?;

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::TYPE => Ok(
            crate::builtins::builtin_type_object_type_id(ptr).unwrap_or_else(|| {
                let class = unsafe { &*(ptr as *const PyClassObject) };
                class.class_type_id()
            }),
        ),
        TypeId::EXCEPTION_TYPE => crate::builtins::exception_proxy_class_id_from_ptr(ptr)
            .map(class_id_to_type_id)
            .ok_or_else(|| {
                BuiltinError::TypeError("descriptor '__subclasses__' requires a type".to_string())
            }),
        _ => Err(BuiltinError::TypeError(
            "descriptor '__subclasses__' requires a type".to_string(),
        )),
    }
}

#[inline]
fn user_type_doc_value(
    vm: &mut VirtualMachine,
    class: &PyClassObject,
    owner_value: Value,
) -> Result<Value, RuntimeError> {
    match class.get_attr(&intern("__doc__")) {
        Some(value) => crate::ops::objects::resolve_class_attribute_in_vm(vm, value, owner_value),
        None => Ok(Value::none()),
    }
}

#[inline]
fn user_type_doc_value_static(class: &PyClassObject) -> Value {
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
        (TypeId::ENUMERATE, "__new__") => Some(builtin_method_value(&ENUMERATE_NEW_METHOD)),
        _ => None,
    }
}

#[inline]
fn builtin_type_bound_method_value(owner: TypeId, name: &str) -> Option<Value> {
    match (owner, name) {
        (TypeId::TYPE, "mro") => Some(builtin_method_value(&TYPE_MRO_METHOD)),
        (TypeId::TYPE, "__prepare__") => Some(builtin_method_value(&TYPE_PREPARE_METHOD)),
        (TypeId::TYPE, "__subclasses__") => Some(builtin_method_value(&TYPE_SUBCLASSES_METHOD)),
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
        (TypeId::TYPE, "__subclasses__") => Some(builtin_method_value(&TYPE_SUBCLASSES_METHOD)),
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
            (TypeId::DICT, "__init__") => Some(builtin_method_value(&DICT_INIT_METHOD)),
            (TypeId::SET, "__init__") => Some(builtin_method_value(&SET_INIT_METHOD)),
            (TypeId::FROZENSET, "__init__") => Some(builtin_method_value(&FROZENSET_INIT_METHOD)),
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
        ReflectedValueKind::FlagsValue => Ok(python_type_flags_value_for_builtin(owner)),
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
        ReflectedValueKind::FlagsValue => Ok(python_type_flags_value_for_builtin(owner)),
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
        Some(ReflectedValueKind::DocValue) => user_type_doc_value(vm, class, owner_value).map(Some),
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
        Some(ReflectedValueKind::FlagsValue) => {
            Ok(Some(python_type_flags_value_for_heap_type(class)))
        }
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
        Some(ReflectedValueKind::DocValue) => Ok(Some(user_type_doc_value_static(class))),
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
        Some(ReflectedValueKind::FlagsValue) => {
            Ok(Some(python_type_flags_value_for_heap_type(class)))
        }
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
            owner,
            name.as_str(),
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
            owner,
            name.as_str(),
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
fn bind_builtin_type_method_if_needed(
    owner: TypeId,
    name: &str,
    method: Value,
    owner_value: Value,
) -> Value {
    if builtin_type_static_method_value(owner, name).is_some() {
        method
    } else if should_bind_builtin_type_callable(owner_value) {
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
        (owner, "__doc__") if builtin_instance_inherits_type_doc(owner) => {
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
        (TypeId::RANGE, attr) => range_instance_attr_value(receiver, attr),
        (TypeId::SLICE, attr) => slice_instance_attr_value(receiver, attr),
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

    if name.as_str() == "__doc__" && builtin_instance_inherits_type_doc(type_id) {
        return true;
    }

    matches!(
        (type_id, name.as_str()),
        (TypeId::OBJECT, "__str__")
            | (TypeId::INT, "real" | "imag" | "numerator" | "denominator")
            | (TypeId::BOOL, "real" | "imag" | "numerator" | "denominator")
            | (TypeId::RANGE, "start" | "stop" | "step")
            | (TypeId::SLICE, "start" | "stop" | "step")
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

fn range_instance_attr_value(receiver: Value, attr: &str) -> Result<Option<Value>, RuntimeError> {
    let Some(ptr) = receiver.as_object_ptr() else {
        return Ok(None);
    };
    let range = unsafe { &*(ptr as *const RangeObject) };
    Ok(match attr {
        "start" => Some(bigint_to_value(range.start_bigint())),
        "stop" => Some(bigint_to_value(range.stop_bigint())),
        "step" => Some(bigint_to_value(range.step_bigint())),
        _ => None,
    })
}

fn slice_instance_attr_value(receiver: Value, attr: &str) -> Result<Option<Value>, RuntimeError> {
    let Some(ptr) = receiver.as_object_ptr() else {
        return Ok(None);
    };
    let slice = unsafe { &*(ptr as *const SliceObject) };
    Ok(match attr {
        "start" => Some(slice.start_value()),
        "stop" => Some(slice.stop_value()),
        "step" => Some(slice.step_value()),
        _ => None,
    })
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
