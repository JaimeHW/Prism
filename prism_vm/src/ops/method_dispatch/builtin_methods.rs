//! Static builtin method objects for optimized method dispatch.
//!
//! These builtins are resolved directly by `LoadMethod` and invoked through the
//! existing `CallMethod` fast path, which keeps common container method calls on
//! the same optimized dispatch path as other builtins.

use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, builtin_mapping_proxy_contains_key,
    builtin_mapping_proxy_entries_static, builtin_mapping_proxy_get_item_static,
    builtin_mapping_proxy_len, builtin_not_implemented_value, get_iterator_mut, value_to_iterator,
};
use crate::error::RuntimeErrorKind;
use crate::ops::iteration::{IterStep, ensure_iterator_value, next_step};
use crate::ops::objects::{dict_storage_mut_from_ptr, dict_storage_ref_from_ptr};
use crate::stdlib::collections::deque::DequeObject;
use crate::stdlib::generators::{CloseResult, GeneratorObject, prepare_close};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::PropertyDescriptor;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{DictViewKind, DictViewObject, MappingProxyObject};
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock, Mutex};
use unicode_xid::UnicodeXID;

use super::method_cache::CachedMethod;

static LIST_APPEND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.append"), list_append));
static LIST_EXTEND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("list.extend"), list_extend_with_vm));
static LIST_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.copy"), list_copy));
static DEQUE_APPEND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.append"), deque_append));
static DEQUE_APPENDLEFT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.appendleft"), deque_appendleft));
static DEQUE_POP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.pop"), deque_pop));
static DEQUE_POPLEFT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.popleft"), deque_popleft));
static REGEX_PATTERN_MATCH_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Pattern.match"),
        crate::stdlib::re::builtin_pattern_match,
    )
});
static REGEX_PATTERN_SEARCH_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Pattern.search"),
        crate::stdlib::re::builtin_pattern_search,
    )
});
static REGEX_PATTERN_FULLMATCH_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Pattern.fullmatch"),
        crate::stdlib::re::builtin_pattern_fullmatch,
    )
});
static REGEX_MATCH_GROUP_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Match.group"),
        crate::stdlib::re::builtin_match_group,
    )
});
static REGEX_MATCH_START_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("Match.start"),
        crate::stdlib::re::builtin_match_start,
    )
});
static REGEX_MATCH_END_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("Match.end"), crate::stdlib::re::builtin_match_end)
});
static REGEX_MATCH_SPAN_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Match.span"),
        crate::stdlib::re::builtin_match_span,
    )
});
static DICT_KEYS_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.keys"), dict_keys));
static DICT_VALUES_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.values"), dict_values));
static DICT_ITEMS_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.items"), dict_items));
static DICT_GET_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.get"), dict_get));
static DICT_LEN_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.__len__"), dict_len));
static DICT_CONTAINS_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.__contains__"), dict_contains));
static DICT_GETITEM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.__getitem__"), dict_getitem));
static DICT_SETITEM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.__setitem__"), dict_setitem));
static DICT_DELITEM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.__delitem__"), dict_delitem));
static DICT_POP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.pop"), dict_pop));
static DICT_SETDEFAULT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.setdefault"), dict_setdefault));
static DICT_CLEAR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.clear"), dict_clear));
static DICT_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("dict.update"), dict_update_with_vm));
static DICT_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.copy"), dict_copy));
static MAPPING_PROXY_KEYS_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("mappingproxy.keys"), mappingproxy_keys));
static MAPPING_PROXY_VALUES_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("mappingproxy.values"), mappingproxy_values)
});
static MAPPING_PROXY_ITEMS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("mappingproxy.items"), mappingproxy_items)
});
static MAPPING_PROXY_GET_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("mappingproxy.get"), mappingproxy_get));
static MAPPING_PROXY_LEN_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("mappingproxy.__len__"), mappingproxy_len)
});
static MAPPING_PROXY_CONTAINS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("mappingproxy.__contains__"),
        mappingproxy_contains,
    )
});
static MAPPING_PROXY_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("mappingproxy.copy"), mappingproxy_copy));
static OBJECT_EQ_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("object.__eq__"), object_eq));
static OBJECT_NE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("object.__ne__"), object_ne));
static OBJECT_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("object.__repr__"), value_repr));
static OBJECT_STR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("object.__str__"), value_str));
static OBJECT_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("object.__format__"), value_format));
static TYPE_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("type.__repr__"), type_repr));
static INT_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.__repr__"), value_repr));
static INT_STR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.__str__"), value_str));
static INT_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.__format__"), value_format));
static BOOL_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bool.__repr__"), value_repr));
static BOOL_STR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bool.__str__"), value_str));
static BOOL_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bool.__format__"), value_format));
static FLOAT_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("float.__repr__"), value_repr));
static FLOAT_STR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("float.__str__"), value_str));
static FLOAT_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("float.__format__"), value_format));
static STR_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.__repr__"), value_repr));
static STR_STR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.__str__"), value_str));
static STR_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.__format__"), value_format));
static STR_UPPER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.upper"), str_upper));
static STR_LOWER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.lower"), str_lower));
static STR_REPLACE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.replace"), str_replace));
static STR_SPLIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.split"), str_split));
static STR_STRIP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.strip"), str_strip));
static STR_LSTRIP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.lstrip"), str_lstrip));
static STR_RSTRIP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.rstrip"), str_rstrip));
static STR_JOIN_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("str.join"), str_join_with_vm));
static STR_ISIDENTIFIER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.isidentifier"), str_isidentifier));
static STR_STARTSWITH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.startswith"), str_startswith));
static STR_ENDSWITH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.endswith"), str_endswith));
static SET_ADD_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.add"), set_add));
static SET_POP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.pop"), set_pop));
static SET_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.copy"), set_copy));
static SET_CONTAINS_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.__contains__"), set_contains));
static FROZENSET_CONTAINS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("frozenset.__contains__"), frozenset_contains)
});
static FROZENSET_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("frozenset.copy"), frozenset_copy));
static BYTEARRAY_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.copy"), bytearray_copy));
static GENERATOR_CLOSE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("generator.close"), generator_close));
static FUNCTION_GET_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("function.__get__"), function_get));
static PROPERTY_GETTER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("property.getter"), property_getter));
static PROPERTY_SETTER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("property.setter"), property_setter));
static PROPERTY_DELETER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("property.deleter"), property_deleter));
static GENERIC_DUNDER_METHODS: LazyLock<Mutex<FxHashMap<(u32, &'static str), usize>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

/// Resolve builtin list methods backed by static builtin function objects.
pub fn resolve_list_method(name: &str) -> Option<CachedMethod> {
    match name {
        "append" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_APPEND_METHOD,
        ))),
        "extend" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_EXTEND_METHOD,
        ))),
        "copy" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_COPY_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin deque methods backed by static builtin function objects.
pub fn resolve_deque_method(name: &str) -> Option<CachedMethod> {
    match name {
        "append" => Some(CachedMethod::simple(builtin_method_value(
            &DEQUE_APPEND_METHOD,
        ))),
        "appendleft" => Some(CachedMethod::simple(builtin_method_value(
            &DEQUE_APPENDLEFT_METHOD,
        ))),
        "pop" => Some(CachedMethod::simple(builtin_method_value(
            &DEQUE_POP_METHOD,
        ))),
        "popleft" => Some(CachedMethod::simple(builtin_method_value(
            &DEQUE_POPLEFT_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin regex pattern methods backed by static builtin function objects.
pub fn resolve_regex_pattern_method(name: &str) -> Option<CachedMethod> {
    match name {
        "match" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_PATTERN_MATCH_METHOD,
        ))),
        "search" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_PATTERN_SEARCH_METHOD,
        ))),
        "fullmatch" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_PATTERN_FULLMATCH_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin regex match methods backed by static builtin function objects.
pub fn resolve_regex_match_method(name: &str) -> Option<CachedMethod> {
    match name {
        "group" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_MATCH_GROUP_METHOD,
        ))),
        "start" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_MATCH_START_METHOD,
        ))),
        "end" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_MATCH_END_METHOD,
        ))),
        "span" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_MATCH_SPAN_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin dict methods backed by static builtin function objects.
pub fn resolve_dict_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__len__" => Some(CachedMethod::simple(builtin_method_value(&DICT_LEN_METHOD))),
        "__contains__" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_CONTAINS_METHOD,
        ))),
        "__getitem__" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_GETITEM_METHOD,
        ))),
        "__setitem__" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_SETITEM_METHOD,
        ))),
        "__delitem__" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_DELITEM_METHOD,
        ))),
        "keys" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_KEYS_METHOD,
        ))),
        "get" => Some(CachedMethod::simple(builtin_method_value(&DICT_GET_METHOD))),
        "values" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_VALUES_METHOD,
        ))),
        "items" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_ITEMS_METHOD,
        ))),
        "pop" => Some(CachedMethod::simple(builtin_method_value(&DICT_POP_METHOD))),
        "setdefault" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_SETDEFAULT_METHOD,
        ))),
        "clear" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_CLEAR_METHOD,
        ))),
        "update" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_UPDATE_METHOD,
        ))),
        "copy" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_COPY_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin mappingproxy methods backed by static builtin function objects.
pub fn resolve_mapping_proxy_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__len__" => Some(CachedMethod::simple(builtin_method_value(
            &MAPPING_PROXY_LEN_METHOD,
        ))),
        "__contains__" => Some(CachedMethod::simple(builtin_method_value(
            &MAPPING_PROXY_CONTAINS_METHOD,
        ))),
        "keys" => Some(CachedMethod::simple(builtin_method_value(
            &MAPPING_PROXY_KEYS_METHOD,
        ))),
        "values" => Some(CachedMethod::simple(builtin_method_value(
            &MAPPING_PROXY_VALUES_METHOD,
        ))),
        "items" => Some(CachedMethod::simple(builtin_method_value(
            &MAPPING_PROXY_ITEMS_METHOD,
        ))),
        "get" => Some(CachedMethod::simple(builtin_method_value(
            &MAPPING_PROXY_GET_METHOD,
        ))),
        "copy" => Some(CachedMethod::simple(builtin_method_value(
            &MAPPING_PROXY_COPY_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin object methods backed by static builtin function objects.
pub fn resolve_object_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__eq__" => Some(CachedMethod::simple(builtin_method_value(
            &OBJECT_EQ_METHOD,
        ))),
        "__ne__" => Some(CachedMethod::simple(builtin_method_value(
            &OBJECT_NE_METHOD,
        ))),
        "__repr__" => Some(CachedMethod::simple(builtin_method_value(
            &OBJECT_REPR_METHOD,
        ))),
        "__str__" => Some(CachedMethod::simple(builtin_method_value(
            &OBJECT_STR_METHOD,
        ))),
        "__format__" => Some(CachedMethod::simple(builtin_method_value(
            &OBJECT_FORMAT_METHOD,
        ))),
        _ => None,
    }
}

pub fn resolve_type_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__repr__" => Some(CachedMethod::simple(builtin_method_value(
            &TYPE_REPR_METHOD,
        ))),
        _ => None,
    }
}

pub fn resolve_int_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__repr__" => Some(CachedMethod::simple(builtin_method_value(&INT_REPR_METHOD))),
        "__str__" => Some(CachedMethod::simple(builtin_method_value(&INT_STR_METHOD))),
        "__format__" => Some(CachedMethod::simple(builtin_method_value(
            &INT_FORMAT_METHOD,
        ))),
        _ => None,
    }
}

pub fn resolve_bool_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__repr__" => Some(CachedMethod::simple(builtin_method_value(
            &BOOL_REPR_METHOD,
        ))),
        "__str__" => Some(CachedMethod::simple(builtin_method_value(&BOOL_STR_METHOD))),
        "__format__" => Some(CachedMethod::simple(builtin_method_value(
            &BOOL_FORMAT_METHOD,
        ))),
        _ => None,
    }
}

pub fn resolve_float_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__repr__" => Some(CachedMethod::simple(builtin_method_value(
            &FLOAT_REPR_METHOD,
        ))),
        "__str__" => Some(CachedMethod::simple(builtin_method_value(
            &FLOAT_STR_METHOD,
        ))),
        "__format__" => Some(CachedMethod::simple(builtin_method_value(
            &FLOAT_FORMAT_METHOD,
        ))),
        _ => None,
    }
}

pub fn resolve_generic_dunder_method(type_id: TypeId, name: &str) -> Option<CachedMethod> {
    let (name, function): (&'static str, fn(&[Value]) -> Result<Value, BuiltinError>) = match name {
        "__repr__" => ("__repr__", value_repr),
        "__str__" => ("__str__", value_str),
        "__format__" => ("__format__", value_format),
        "__reduce_ex__" => ("__reduce_ex__", value_reduce_ex),
        _ => return None,
    };

    if type_id == TypeId::TYPE && name == "__repr__" {
        return None;
    }

    Some(CachedMethod::simple(generic_dunder_method_value(
        type_id, name, function,
    )))
}

/// Resolve builtin set and frozenset methods backed by static builtin function objects.
pub fn resolve_str_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__repr__" => Some(CachedMethod::simple(builtin_method_value(&STR_REPR_METHOD))),
        "__str__" => Some(CachedMethod::simple(builtin_method_value(&STR_STR_METHOD))),
        "__format__" => Some(CachedMethod::simple(builtin_method_value(
            &STR_FORMAT_METHOD,
        ))),
        "upper" => Some(CachedMethod::simple(builtin_method_value(
            &STR_UPPER_METHOD,
        ))),
        "lower" => Some(CachedMethod::simple(builtin_method_value(
            &STR_LOWER_METHOD,
        ))),
        "replace" => Some(CachedMethod::simple(builtin_method_value(
            &STR_REPLACE_METHOD,
        ))),
        "split" => Some(CachedMethod::simple(builtin_method_value(
            &STR_SPLIT_METHOD,
        ))),
        "strip" => Some(CachedMethod::simple(builtin_method_value(
            &STR_STRIP_METHOD,
        ))),
        "lstrip" => Some(CachedMethod::simple(builtin_method_value(
            &STR_LSTRIP_METHOD,
        ))),
        "rstrip" => Some(CachedMethod::simple(builtin_method_value(
            &STR_RSTRIP_METHOD,
        ))),
        "join" => Some(CachedMethod::simple(builtin_method_value(&STR_JOIN_METHOD))),
        "isidentifier" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISIDENTIFIER_METHOD,
        ))),
        "startswith" => Some(CachedMethod::simple(builtin_method_value(
            &STR_STARTSWITH_METHOD,
        ))),
        "endswith" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ENDSWITH_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin set and frozenset methods backed by static builtin function objects.
pub fn resolve_set_method(type_id: TypeId, name: &str) -> Option<CachedMethod> {
    match (type_id, name) {
        (TypeId::SET, "add") => Some(CachedMethod::simple(builtin_method_value(&SET_ADD_METHOD))),
        (TypeId::SET, "pop") => Some(CachedMethod::simple(builtin_method_value(&SET_POP_METHOD))),
        (TypeId::SET, "copy") => Some(CachedMethod::simple(builtin_method_value(&SET_COPY_METHOD))),
        (TypeId::SET, "__contains__") => Some(CachedMethod::simple(builtin_method_value(
            &SET_CONTAINS_METHOD,
        ))),
        (TypeId::FROZENSET, "copy") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_COPY_METHOD,
        ))),
        (TypeId::FROZENSET, "__contains__") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_CONTAINS_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin bytearray methods backed by static builtin function objects.
pub fn resolve_bytearray_method(name: &str) -> Option<CachedMethod> {
    match name {
        "copy" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_COPY_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin generator methods backed by static builtin function objects.
pub fn resolve_generator_method(name: &str) -> Option<CachedMethod> {
    match name {
        "close" => Some(CachedMethod::simple(builtin_method_value(
            &GENERATOR_CLOSE_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin function descriptor methods backed by static builtin function objects.
pub fn resolve_function_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__get__" => Some(CachedMethod::simple(builtin_method_value(
            &FUNCTION_GET_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin property methods backed by static builtin function objects.
pub fn resolve_property_method(name: &str) -> Option<CachedMethod> {
    match name {
        "getter" => Some(CachedMethod::simple(builtin_method_value(
            &PROPERTY_GETTER_METHOD,
        ))),
        "setter" => Some(CachedMethod::simple(builtin_method_value(
            &PROPERTY_SETTER_METHOD,
        ))),
        "deleter" => Some(CachedMethod::simple(builtin_method_value(
            &PROPERTY_DELETER_METHOD,
        ))),
        _ => None,
    }
}

#[inline(always)]
fn builtin_method_value(method: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(method as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn list_append(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "append", args, 1)?;
    let list = expect_list_mut(args[0], "append")?;
    list.push(args[1]);
    Ok(Value::none())
}

#[inline]
fn list_extend(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "extend", args, 1)?;

    // Collect before taking a mutable borrow so `lst.extend(lst)` duplicates the
    // current contents once instead of aliasing the list during mutation.
    let items = collect_iterable_values(args[1])?;
    extend_list_with_values(args[0], items)
}

#[inline]
fn list_extend_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "extend", args, 1)?;
    let items = collect_iterable_values_with_vm(vm, args[1])?;
    extend_list_with_values(args[0], items)
}

#[inline]
fn extend_list_with_values(receiver: Value, items: Vec<Value>) -> Result<Value, BuiltinError> {
    let list = expect_list_mut(receiver, "extend")?;
    list.extend(items);
    Ok(Value::none())
}

#[inline]
fn list_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "copy", args, 0)?;
    let list = expect_list_ref(args[0], "copy")?;
    Ok(to_object_value(ListObject::from_slice(list.as_slice())))
}

#[inline]
fn deque_append(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("deque", "append", args, 1)?;
    let deque = expect_deque_mut(args[0], "append")?;
    deque.deque_mut().append(args[1]);
    Ok(Value::none())
}

#[inline]
fn deque_appendleft(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("deque", "appendleft", args, 1)?;
    let deque = expect_deque_mut(args[0], "appendleft")?;
    deque.deque_mut().appendleft(args[1]);
    Ok(Value::none())
}

#[inline]
fn deque_pop(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("deque", "pop", args, 0)?;
    let deque = expect_deque_mut(args[0], "pop")?;
    deque
        .deque_mut()
        .pop()
        .ok_or_else(|| BuiltinError::IndexError("pop from an empty deque".to_string()))
}

#[inline]
fn deque_popleft(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("deque", "popleft", args, 0)?;
    let deque = expect_deque_mut(args[0], "popleft")?;
    deque
        .deque_mut()
        .popleft()
        .ok_or_else(|| BuiltinError::IndexError("pop from an empty deque".to_string()))
}

#[inline]
fn dict_keys(args: &[Value]) -> Result<Value, BuiltinError> {
    dict_view(args, "keys", DictViewKind::Keys)
}

#[inline]
fn dict_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "__getitem__", args, 1)?;
    let dict = expect_dict_ref(args[0], "__getitem__")?;
    dict.get(args[1])
        .ok_or_else(|| BuiltinError::KeyError("key not found".to_string()))
}

#[inline]
fn dict_contains(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "__contains__", args, 1)?;
    let dict = expect_dict_ref(args[0], "__contains__")?;
    Ok(Value::bool(dict.get(args[1]).is_some()))
}

#[inline]
fn dict_setitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "__setitem__", args, 2)?;
    let dict = expect_dict_mut(args[0], "__setitem__")?;
    dict.set(args[1], args[2]);
    Ok(Value::none())
}

#[inline]
fn dict_delitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "__delitem__", args, 1)?;
    let dict = expect_dict_mut(args[0], "__delitem__")?;
    dict.remove(args[1])
        .map(|_| Value::none())
        .ok_or_else(|| BuiltinError::KeyError("key not found".to_string()))
}

#[inline]
fn dict_values(args: &[Value]) -> Result<Value, BuiltinError> {
    dict_view(args, "values", DictViewKind::Values)
}

#[inline]
fn dict_items(args: &[Value]) -> Result<Value, BuiltinError> {
    dict_view(args, "items", DictViewKind::Items)
}

#[inline]
fn dict_get(args: &[Value]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if !(1..=2).contains(&given) {
        return Err(BuiltinError::TypeError(format!(
            "dict.get() takes from 1 to 2 arguments ({given} given)"
        )));
    }

    let dict = expect_dict_ref(args[0], "get")?;
    let default = args.get(2).copied().unwrap_or_else(Value::none);
    Ok(dict.get(args[1]).unwrap_or(default))
}

#[inline]
fn dict_len(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "__len__", args, 0)?;
    let dict = expect_dict_ref(args[0], "__len__")?;
    Ok(Value::int(dict.len() as i64).expect("dict length should fit in Value::int"))
}

#[inline]
fn dict_pop(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        let given = args.len().saturating_sub(1);
        return Err(BuiltinError::TypeError(format!(
            "dict.pop() takes 1 or 2 arguments ({given} given)"
        )));
    }

    let dict = expect_dict_mut(args[0], "pop")?;
    if let Some(value) = dict.remove(args[1]) {
        return Ok(value);
    }

    if let Some(default) = args.get(2).copied() {
        return Ok(default);
    }

    Err(BuiltinError::KeyError("key not found".to_string()))
}

#[inline]
fn dict_setdefault(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        let given = args.len().saturating_sub(1);
        return Err(BuiltinError::TypeError(format!(
            "dict.setdefault() takes 1 or 2 arguments ({given} given)"
        )));
    }

    let dict = expect_dict_mut(args[0], "setdefault")?;
    let default = args.get(2).copied().unwrap_or_else(Value::none);
    Ok(dict.setdefault(args[1], default))
}

#[inline]
fn dict_clear(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "clear", args, 0)?;
    let dict = expect_dict_mut(args[0], "clear")?;
    dict.clear();
    Ok(Value::none())
}

#[inline]
fn dict_update_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        let given = args.len().saturating_sub(1);
        return Err(BuiltinError::TypeError(format!(
            "dict.update() takes at most 1 argument ({given} given)"
        )));
    }

    if args.len() == 1 {
        expect_dict_receiver(args[0], "update")?;
        return Ok(Value::none());
    }

    let entries = collect_dict_update_entries(vm, args[1])?;
    let dict = expect_dict_mut(args[0], "update")?;
    for (key, value) in entries {
        dict.set(key, value);
    }
    Ok(Value::none())
}

#[inline]
fn dict_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "copy", args, 0)?;
    let dict = expect_dict_ref(args[0], "copy")?;
    let mut copied = DictObject::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        copied.set(key, value);
    }
    Ok(to_object_value(copied))
}

#[inline]
fn mappingproxy_keys(args: &[Value]) -> Result<Value, BuiltinError> {
    mappingproxy_view(args, "keys", DictViewKind::Keys)
}

#[inline]
fn mappingproxy_values(args: &[Value]) -> Result<Value, BuiltinError> {
    mappingproxy_view(args, "values", DictViewKind::Values)
}

#[inline]
fn mappingproxy_items(args: &[Value]) -> Result<Value, BuiltinError> {
    mappingproxy_view(args, "items", DictViewKind::Items)
}

#[inline]
fn mappingproxy_get(args: &[Value]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if !(1..=2).contains(&given) {
        return Err(BuiltinError::TypeError(format!(
            "mappingproxy.get() takes from 1 to 2 arguments ({given} given)"
        )));
    }

    let proxy = expect_mapping_proxy_ref(args[0], "get")?;
    let default = args.get(2).copied().unwrap_or_else(Value::none);
    Ok(builtin_mapping_proxy_get_item_static(proxy, args[1])
        .map_err(runtime_error_to_builtin_error)?
        .unwrap_or(default))
}

#[inline]
fn mappingproxy_len(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("mappingproxy", "__len__", args, 0)?;
    let proxy = expect_mapping_proxy_ref(args[0], "__len__")?;
    let len = builtin_mapping_proxy_len(proxy).map_err(runtime_error_to_builtin_error)?;
    Ok(Value::int(len as i64).expect("mappingproxy length should fit in Value::int"))
}

#[inline]
fn mappingproxy_contains(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("mappingproxy", "__contains__", args, 1)?;
    let proxy = expect_mapping_proxy_ref(args[0], "__contains__")?;
    Ok(Value::bool(
        builtin_mapping_proxy_contains_key(proxy, args[1])
            .map_err(runtime_error_to_builtin_error)?,
    ))
}

#[inline]
fn mappingproxy_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("mappingproxy", "copy", args, 0)?;
    let proxy = expect_mapping_proxy_ref(args[0], "copy")?;
    let entries =
        builtin_mapping_proxy_entries_static(proxy).map_err(runtime_error_to_builtin_error)?;
    let mut copied = DictObject::with_capacity(entries.len());
    for (key, value) in entries {
        copied.set(key, value);
    }
    Ok(to_object_value(copied))
}

#[inline]
fn object_eq(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("object", "__eq__", args, 1)?;
    if args[0] == args[1] {
        Ok(Value::bool(true))
    } else {
        Ok(builtin_not_implemented_value())
    }
}

#[inline]
fn object_ne(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("object", "__ne__", args, 1)?;
    if args[0] == args[1] {
        Ok(Value::bool(false))
    } else {
        Ok(builtin_not_implemented_value())
    }
}

#[inline]
fn value_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    crate::builtins::builtin_repr(args)
}

#[inline]
fn value_str(args: &[Value]) -> Result<Value, BuiltinError> {
    crate::builtins::builtin_str(args)
}

#[inline]
fn value_format(args: &[Value]) -> Result<Value, BuiltinError> {
    crate::builtins::builtin_format(args)
}

fn value_reduce_ex(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        let given = args.len().saturating_sub(1);
        return Err(BuiltinError::TypeError(format!(
            "__reduce_ex__() takes exactly 1 argument ({given} given)"
        )));
    }

    Err(BuiltinError::NotImplemented(
        "pickle protocol support is not implemented yet".to_string(),
    ))
}

fn generic_dunder_method_value(
    type_id: TypeId,
    name: &'static str,
    function: fn(&[Value]) -> Result<Value, BuiltinError>,
) -> Value {
    let mut cache = GENERIC_DUNDER_METHODS
        .lock()
        .expect("generic dunder method cache lock poisoned");
    if let Some(ptr) = cache.get(&(type_id.raw(), name)) {
        return Value::object_ptr(*ptr as *const ());
    }

    let label = Arc::<str>::from(format!("{}.{}", type_id.name(), name));
    let method = Box::leak(Box::new(BuiltinFunctionObject::new(label, function)));
    let ptr = method as *mut BuiltinFunctionObject as usize;
    cache.insert((type_id.raw(), name), ptr);
    Value::object_ptr(ptr as *const ())
}

fn type_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        let given = args.len().saturating_sub(1);
        return Err(BuiltinError::TypeError(format!(
            "type.__repr__() takes exactly 0 arguments ({given} given)"
        )));
    }

    let rendered = format_type_repr(args[0])?;
    Ok(Value::string(intern(&rendered)))
}

fn format_type_repr(value: Value) -> Result<String, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("descriptor '__repr__' requires a type object".to_string())
    })?;

    if crate::ops::objects::extract_type_id(ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "descriptor '__repr__' requires a type object".to_string(),
        ));
    }

    if let Some(type_id) = crate::builtins::builtin_type_object_type_id(ptr) {
        let module = match type_id {
            TypeId::DEQUE => "collections",
            _ => "builtins",
        };
        return Ok(render_type_repr(module, type_id.name()));
    }

    let class = unsafe { &*(ptr as *const PyClassObject) };
    let module = class
        .get_attr(&intern("__module__"))
        .and_then(string_value_text)
        .unwrap_or_else(|| "__main__".to_string());
    let qualname = class
        .get_attr(&intern("__qualname__"))
        .and_then(string_value_text)
        .unwrap_or_else(|| class.name().as_str().to_string());
    Ok(render_type_repr(&module, &qualname))
}

#[inline]
fn render_type_repr(module: &str, qualname: &str) -> String {
    if module == "builtins" {
        format!("<class '{qualname}'>")
    } else {
        format!("<class '{module}.{qualname}'>")
    }
}

fn string_value_text(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        return interned_by_ptr(ptr as *const u8).map(|name| name.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    Some(
        unsafe { &*(ptr as *const StringObject) }
            .as_str()
            .to_string(),
    )
}

#[inline]
fn str_upper(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "upper", args, 0)?;
    with_str_receiver(args[0], "upper", |value| {
        if value.is_ascii() {
            let mut bytes = value.as_bytes().to_vec();
            let mut changed = false;
            for byte in &mut bytes {
                let upper = byte.to_ascii_uppercase();
                changed |= upper != *byte;
                *byte = upper;
            }

            return if changed {
                let upper = unsafe { String::from_utf8_unchecked(bytes) };
                Ok(Value::string(intern(&upper)))
            } else {
                Ok(args[0])
            };
        }

        let upper: String = value.chars().flat_map(char::to_uppercase).collect();
        if upper == value {
            Ok(args[0])
        } else {
            Ok(Value::string(intern(&upper)))
        }
    })
}

#[inline]
fn str_lower(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "lower", args, 0)?;
    with_str_receiver(args[0], "lower", |value| {
        if value.is_ascii() {
            let mut bytes = value.as_bytes().to_vec();
            let mut changed = false;
            for byte in &mut bytes {
                let lower = byte.to_ascii_lowercase();
                changed |= lower != *byte;
                *byte = lower;
            }

            return if changed {
                let lower = unsafe { String::from_utf8_unchecked(bytes) };
                Ok(Value::string(intern(&lower)))
            } else {
                Ok(args[0])
            };
        }

        let lower: String = value.chars().flat_map(char::to_lowercase).collect();
        if lower == value {
            Ok(args[0])
        } else {
            Ok(Value::string(intern(&lower)))
        }
    })
}

#[inline]
fn str_replace(args: &[Value]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if !(2..=3).contains(&given) {
        return Err(BuiltinError::TypeError(format!(
            "str.replace() takes from 2 to 3 arguments ({given} given)"
        )));
    }

    let old = expect_str_method_string_arg(args[1], "replace", 1)?;
    let new = expect_str_method_string_arg(args[2], "replace", 2)?;
    let count = parse_replace_count(args.get(3).copied())?;

    with_str_receiver(args[0], "replace", |value| {
        if count == Some(0) || old == new {
            return Ok(args[0]);
        }

        if old.is_empty() {
            let replaced = replace_empty_pattern(value, &new, count);
            return if replaced == value {
                Ok(args[0])
            } else {
                Ok(Value::string(intern(&replaced)))
            };
        }

        if !value.contains(&*old) {
            return Ok(args[0]);
        }

        let replaced = match count {
            Some(limit) => value.replacen(&old, &new, limit),
            None => value.replace(&old, &new),
        };
        if replaced == value {
            Ok(args[0])
        } else {
            Ok(Value::string(intern(&replaced)))
        }
    })
}

#[inline]
fn str_split(args: &[Value]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 2 {
        return Err(BuiltinError::TypeError(format!(
            "str.split() takes at most 2 arguments ({given} given)"
        )));
    }

    let separator = match args.get(1).copied() {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(expect_str_method_string_arg(value, "split", 1)?),
    };
    let maxsplit = parse_split_count(args.get(2).copied(), "split", 2)?;

    with_str_receiver(args[0], "split", |value| {
        let parts = match separator.as_deref() {
            Some(separator) => split_with_separator(value, separator, maxsplit)?,
            None => split_on_whitespace(value, maxsplit),
        };

        Ok(to_object_value(ListObject::from_slice(parts.as_slice())))
    })
}

#[inline]
fn str_strip(args: &[Value]) -> Result<Value, BuiltinError> {
    strip_method(args, "strip", StripDirection::Both)
}

#[inline]
fn str_lstrip(args: &[Value]) -> Result<Value, BuiltinError> {
    strip_method(args, "lstrip", StripDirection::Leading)
}

#[inline]
fn str_rstrip(args: &[Value]) -> Result<Value, BuiltinError> {
    strip_method(args, "rstrip", StripDirection::Trailing)
}

#[inline]
fn str_join(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "join", args, 1)?;
    let values = collect_iterable_values(args[1])?;
    str_join_values(args[0], values)
}

#[inline]
fn str_join_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "join", args, 1)?;
    let values = collect_iterable_values_with_vm(vm, args[1])?;
    str_join_values(args[0], values)
}

#[inline]
fn str_join_values(receiver: Value, values: Vec<Value>) -> Result<Value, BuiltinError> {
    let mut parts = Vec::with_capacity(values.len());

    for (index, value) in values.into_iter().enumerate() {
        let part = string_object_from_value(value).map_err(|_| {
            BuiltinError::TypeError(format!(
                "sequence item {index}: expected str instance, {} found",
                value.type_name()
            ))
        })?;
        parts.push(part);
    }

    with_str_receiver(receiver, "join", |separator| {
        Ok(to_object_value(StringObject::new(separator).join(&parts)))
    })
}

#[inline]
fn str_isidentifier(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "isidentifier", args, 0)?;
    with_str_receiver(args[0], "isidentifier", |value| {
        Ok(Value::bool(is_python_identifier(value)))
    })
}

#[inline]
fn str_startswith(args: &[Value]) -> Result<Value, BuiltinError> {
    affix_match(args, "startswith", |value, affix| value.starts_with(affix))
}

#[inline]
fn str_endswith(args: &[Value]) -> Result<Value, BuiltinError> {
    affix_match(args, "endswith", |value, affix| value.ends_with(affix))
}

#[inline]
fn is_python_identifier(value: &str) -> bool {
    let Some(first) = value.chars().next() else {
        return false;
    };

    if value.is_ascii() {
        return is_ascii_identifier(value.as_bytes());
    }

    if first != '_' && !UnicodeXID::is_xid_start(first) {
        return false;
    }

    value
        .chars()
        .skip(1)
        .all(|ch| ch == '_' || UnicodeXID::is_xid_continue(ch))
}

#[inline]
fn is_ascii_identifier(bytes: &[u8]) -> bool {
    let Some((&first, rest)) = bytes.split_first() else {
        return false;
    };

    if !(first == b'_' || first.is_ascii_alphabetic()) {
        return false;
    }

    rest.iter()
        .all(|byte| *byte == b'_' || byte.is_ascii_alphanumeric())
}

#[inline]
fn affix_match(
    args: &[Value],
    method_name: &'static str,
    matcher: impl Fn(&str, &str) -> bool,
) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "str.{method_name}() takes from 1 to 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    with_str_receiver(args[0], method_name, |value| {
        let (start, end) = normalize_slice_bounds(
            value,
            args.get(2).copied(),
            args.get(3).copied(),
            method_name,
        )?;
        let slice = &value[start..end];
        Ok(Value::bool(match_string_affixes(
            args[1],
            method_name,
            |candidate| matcher(slice, candidate),
        )?))
    })
}

#[inline]
fn match_string_affixes(
    affixes: Value,
    method_name: &'static str,
    mut matcher: impl FnMut(&str) -> bool,
) -> Result<bool, BuiltinError> {
    if let Ok(candidate) = string_object_from_value(affixes) {
        return Ok(matcher(candidate.as_str()));
    }

    let Some(ptr) = affixes.as_object_ptr() else {
        return Err(str_affix_type_error(method_name, affixes.type_name()));
    };
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::TUPLE {
        return Err(str_affix_type_error(method_name, affixes.type_name()));
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    for index in 0..tuple.len() {
        let candidate = tuple
            .get(index as i64)
            .expect("tuple index should be valid");
        let candidate = string_object_from_value(candidate)
            .map_err(|_| str_affix_type_error(method_name, candidate.type_name()))?;
        if matcher(candidate.as_str()) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[inline]
fn str_affix_type_error(method_name: &'static str, actual_type: &str) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "str.{method_name}() first arg must be str or a tuple of str, not {actual_type}"
    ))
}

#[inline]
fn normalize_slice_bounds(
    value: &str,
    start: Option<Value>,
    end: Option<Value>,
    method_name: &'static str,
) -> Result<(usize, usize), BuiltinError> {
    let char_len = value.chars().count();
    let start = clamp_slice_index(parse_slice_bound(start, 0, method_name)?, char_len);
    let end = clamp_slice_index(
        parse_slice_bound(end, char_len as isize, method_name)?,
        char_len,
    );
    let start = start.min(end);
    let end = end.max(start);
    Ok((
        char_index_to_byte_offset(value, start),
        char_index_to_byte_offset(value, end),
    ))
}

#[inline]
fn parse_slice_bound(
    bound: Option<Value>,
    default: isize,
    method_name: &'static str,
) -> Result<isize, BuiltinError> {
    let Some(bound) = bound else {
        return Ok(default);
    };

    bound.as_int().map(|value| value as isize).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "str.{method_name}() slice indices must be integers, not '{}'",
            bound.type_name()
        ))
    })
}

#[inline]
fn clamp_slice_index(index: isize, char_len: usize) -> usize {
    if index < 0 {
        char_len.saturating_sub(index.unsigned_abs())
    } else {
        usize::try_from(index).unwrap_or(usize::MAX).min(char_len)
    }
}

#[inline]
fn char_index_to_byte_offset(value: &str, index: usize) -> usize {
    if value.is_ascii() {
        return index.min(value.len());
    }

    if index == 0 {
        return 0;
    }

    value
        .char_indices()
        .nth(index)
        .map_or(value.len(), |(offset, _)| offset)
}

#[inline]
fn expect_str_method_string_arg(
    value: Value,
    method_name: &'static str,
    position: usize,
) -> Result<String, BuiltinError> {
    string_object_from_value(value)
        .map(|string: StringObject| string.as_str().to_string())
        .map_err(|_| {
            BuiltinError::TypeError(format!(
                "str.{method_name}() argument {position} must be str, not {}",
                value.type_name()
            ))
        })
}

#[inline]
fn parse_split_count(
    count: Option<Value>,
    method_name: &'static str,
    position: usize,
) -> Result<Option<usize>, BuiltinError> {
    let Some(count) = count else {
        return Ok(None);
    };

    if let Some(value) = count.as_int() {
        if value < 0 {
            return Ok(None);
        }
        return Ok(Some(usize::try_from(value).unwrap_or(usize::MAX)));
    }

    if let Some(value) = count.as_bool() {
        return Ok(Some(if value { 1 } else { 0 }));
    }

    Err(BuiltinError::TypeError(format!(
        "str.{method_name}() argument {position} must be int, not {}",
        count.type_name()
    )))
}

#[inline]
fn parse_replace_count(count: Option<Value>) -> Result<Option<usize>, BuiltinError> {
    parse_split_count(count, "replace", 3)
}

#[inline]
fn strip_method(
    args: &[Value],
    method_name: &'static str,
    direction: StripDirection,
) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 1 {
        return Err(BuiltinError::TypeError(format!(
            "str.{method_name}() takes at most 1 argument ({given} given)"
        )));
    }

    let chars = parse_strip_chars(args.get(1).copied(), method_name)?;

    with_str_receiver(args[0], method_name, |value| {
        let stripped = strip_text(value, chars.as_ref(), direction);
        if stripped.len() == value.len() {
            Ok(args[0])
        } else {
            Ok(Value::string(intern(stripped)))
        }
    })
}

#[inline]
fn parse_strip_chars(
    chars: Option<Value>,
    method_name: &'static str,
) -> Result<Option<StripChars>, BuiltinError> {
    let Some(chars) = chars else {
        return Ok(None);
    };

    if chars.is_none() {
        return Ok(None);
    }

    let chars = expect_str_method_string_arg(chars, method_name, 1)?;
    Ok(Some(StripChars::new(&chars)))
}

#[inline]
fn strip_text<'a>(
    value: &'a str,
    chars: Option<&StripChars>,
    direction: StripDirection,
) -> &'a str {
    let start = if matches!(direction, StripDirection::Leading | StripDirection::Both) {
        trim_start_index(value, chars)
    } else {
        0
    };
    if start == value.len() {
        return "";
    }

    let end = if matches!(direction, StripDirection::Trailing | StripDirection::Both) {
        trim_end_index(value, chars)
    } else {
        value.len()
    };

    if start >= end { "" } else { &value[start..end] }
}

#[inline]
fn trim_start_index(value: &str, chars: Option<&StripChars>) -> usize {
    for (idx, ch) in value.char_indices() {
        if !should_strip_char(ch, chars) {
            return idx;
        }
    }
    value.len()
}

#[inline]
fn trim_end_index(value: &str, chars: Option<&StripChars>) -> usize {
    for (idx, ch) in value.char_indices().rev() {
        if !should_strip_char(ch, chars) {
            return idx + ch.len_utf8();
        }
    }
    0
}

#[inline]
fn should_strip_char(ch: char, chars: Option<&StripChars>) -> bool {
    match chars {
        Some(chars) => chars.contains(ch),
        None => ch.is_whitespace(),
    }
}

#[inline]
fn split_with_separator(
    value: &str,
    separator: &str,
    maxsplit: Option<usize>,
) -> Result<Vec<Value>, BuiltinError> {
    if separator.is_empty() {
        return Err(BuiltinError::ValueError("empty separator".to_string()));
    }

    let mut parts = Vec::new();
    let limit = maxsplit.unwrap_or(usize::MAX);
    let mut start = 0usize;
    let mut splits = 0usize;

    while splits < limit {
        let Some(offset) = value[start..].find(separator) else {
            break;
        };

        let end = start + offset;
        parts.push(Value::string(intern(&value[start..end])));
        start = end + separator.len();
        splits += 1;
    }

    parts.push(Value::string(intern(&value[start..])));
    Ok(parts)
}

#[inline]
fn split_on_whitespace(value: &str, maxsplit: Option<usize>) -> Vec<Value> {
    let limit = maxsplit.unwrap_or(usize::MAX);
    let trimmed = value.trim_start_matches(char::is_whitespace);
    if trimmed.is_empty() {
        return Vec::new();
    }

    if limit == 0 {
        return vec![Value::string(intern(trimmed))];
    }

    let mut parts = Vec::new();
    let mut remaining = trimmed;
    let mut splits = 0usize;

    while !remaining.is_empty() {
        if splits == limit {
            parts.push(Value::string(intern(remaining)));
            break;
        }

        let Some((word_end, next_start)) = find_whitespace_run(remaining) else {
            parts.push(Value::string(intern(remaining)));
            break;
        };

        parts.push(Value::string(intern(&remaining[..word_end])));
        remaining = &remaining[next_start..];
        splits += 1;
    }

    parts
}

#[inline]
fn find_whitespace_run(value: &str) -> Option<(usize, usize)> {
    let mut chars = value.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if !ch.is_whitespace() {
            continue;
        }

        let mut next_start = value.len();
        while let Some((next_idx, next_char)) = chars.peek().copied() {
            if next_char.is_whitespace() {
                chars.next();
                continue;
            }
            next_start = next_idx;
            break;
        }
        return Some((idx, next_start));
    }
    None
}

#[inline]
fn replace_empty_pattern(value: &str, new: &str, count: Option<usize>) -> String {
    let max_insertions = value.chars().count() + 1;
    let mut remaining = count.unwrap_or(max_insertions).min(max_insertions);
    if remaining == 0 {
        return value.to_string();
    }

    let mut result = String::with_capacity(value.len() + new.len() * remaining);
    result.push_str(new);
    remaining -= 1;

    for ch in value.chars() {
        result.push(ch);
        if remaining > 0 {
            result.push_str(new);
            remaining -= 1;
        }
    }

    result
}

#[inline]
fn dict_view(
    args: &[Value],
    method_name: &'static str,
    kind: DictViewKind,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", method_name, args, 0)?;
    expect_dict_receiver(args[0], method_name)?;
    Ok(to_object_value(DictViewObject::new(kind, args[0])))
}

#[inline]
fn mappingproxy_view(
    args: &[Value],
    method_name: &'static str,
    kind: DictViewKind,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("mappingproxy", method_name, args, 0)?;
    expect_mapping_proxy_ref(args[0], method_name)?;
    Ok(to_object_value(DictViewObject::new(kind, args[0])))
}

fn collect_dict_update_entries(
    vm: &mut VirtualMachine,
    source: Value,
) -> Result<Vec<(Value, Value)>, BuiltinError> {
    if let Some(dict_ptr) = source.as_object_ptr() {
        let header = unsafe { &*(dict_ptr as *const ObjectHeader) };
        if header.type_id == TypeId::DICT {
            let dict = unsafe { &*(dict_ptr as *const DictObject) };
            return Ok(dict.iter().collect());
        }
        if header.type_id == TypeId::MAPPING_PROXY {
            let proxy = unsafe { &*(dict_ptr as *const MappingProxyObject) };
            return builtin_mapping_proxy_entries_static(proxy)
                .map_err(runtime_error_to_builtin_error);
        }
    }

    let items = collect_iterable_values_with_vm(vm, source)?;
    let mut entries = Vec::with_capacity(items.len());
    for item in items {
        entries.push(expect_dict_update_entry(item)?);
    }
    Ok(entries)
}

fn expect_dict_update_entry(item: Value) -> Result<(Value, Value), BuiltinError> {
    let Some(ptr) = item.as_object_ptr() else {
        return Err(BuiltinError::ValueError(
            "dictionary update sequence element is not a 2-item iterable".to_string(),
        ));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            expect_pair_from_slice(tuple.as_slice())
        }
        TypeId::LIST => {
            let list = unsafe { &*(ptr as *const ListObject) };
            expect_pair_from_slice(list.as_slice())
        }
        _ => Err(BuiltinError::ValueError(
            "dictionary update sequence element is not a 2-item iterable".to_string(),
        )),
    }
}

#[inline]
fn expect_pair_from_slice(values: &[Value]) -> Result<(Value, Value), BuiltinError> {
    if values.len() != 2 {
        return Err(BuiltinError::ValueError(format!(
            "dictionary update sequence element has length {}; 2 is required",
            values.len()
        )));
    }

    Ok((values[0], values[1]))
}

#[inline]
fn set_add(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "add", args, 1)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "add")?;
    set.add(args[1]);
    Ok(Value::none())
}

#[inline]
fn set_pop(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "pop", args, 0)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "pop")?;
    set.pop()
        .ok_or_else(|| BuiltinError::KeyError("pop from an empty set".to_string()))
}

#[inline]
fn set_contains(args: &[Value]) -> Result<Value, BuiltinError> {
    contains_for_set_type(args, TypeId::SET, "set", "__contains__")
}

#[inline]
fn frozenset_contains(args: &[Value]) -> Result<Value, BuiltinError> {
    contains_for_set_type(args, TypeId::FROZENSET, "frozenset", "__contains__")
}

#[inline]
fn set_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "copy", args, 0)?;
    let set = expect_set_receiver(args[0], TypeId::SET, "copy")?;
    Ok(to_object_value(set.clone()))
}

#[inline]
fn frozenset_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("frozenset", "copy", args, 0)?;
    expect_set_receiver(args[0], TypeId::FROZENSET, "copy")?;
    Ok(args[0])
}

#[inline]
fn bytearray_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("bytearray", "copy", args, 0)?;
    let bytearray = expect_bytearray_ref(args[0], "copy")?;
    Ok(to_object_value(bytearray.clone()))
}

#[inline]
fn contains_for_set_type(
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    Ok(Value::bool(set.contains(args[1])))
}

#[inline]
fn expect_method_arg_count(
    receiver_name: &'static str,
    method_name: &'static str,
    args: &[Value],
    explicit_count: usize,
) -> Result<(), BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given == explicit_count {
        return Ok(());
    }

    let noun = if explicit_count == 1 {
        "argument"
    } else {
        "arguments"
    };
    Err(BuiltinError::TypeError(format!(
        "{receiver_name}.{method_name}() takes exactly {explicit_count} {noun} ({given} given)"
    )))
}

#[inline]
fn collect_iterable_values(iterable: Value) -> Result<Vec<Value>, BuiltinError> {
    if let Some(iter) = get_iterator_mut(&iterable) {
        return Ok(iter.collect_remaining());
    }

    let mut iter = value_to_iterator(&iterable).map_err(BuiltinError::from)?;
    Ok(iter.collect_remaining())
}

#[derive(Copy, Clone)]
enum StripDirection {
    Leading,
    Trailing,
    Both,
}

struct StripChars {
    ascii_mask: [u64; 2],
    unicode: FxHashMap<char, ()>,
}

impl StripChars {
    #[inline]
    fn new(chars: &str) -> Self {
        let mut ascii_mask = [0u64; 2];
        let mut unicode = FxHashMap::default();

        for ch in chars.chars() {
            if ch.is_ascii() {
                let byte = ch as u8;
                let slot = usize::from(byte / 64);
                let bit = u32::from(byte % 64);
                ascii_mask[slot] |= 1u64 << bit;
            } else {
                unicode.insert(ch, ());
            }
        }

        Self {
            ascii_mask,
            unicode,
        }
    }

    #[inline]
    fn contains(&self, ch: char) -> bool {
        if ch.is_ascii() {
            let byte = ch as u8;
            let slot = usize::from(byte / 64);
            let bit = u32::from(byte % 64);
            return (self.ascii_mask[slot] & (1u64 << bit)) != 0;
        }

        self.unicode.contains_key(&ch)
    }
}

#[inline]
fn collect_iterable_values_with_vm(
    vm: &mut VirtualMachine,
    iterable: Value,
) -> Result<Vec<Value>, BuiltinError> {
    let iterator = ensure_iterator_value(vm, iterable).map_err(runtime_error_to_builtin_error)?;
    let mut values = Vec::new();

    loop {
        match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
            IterStep::Yielded(value) => values.push(value),
            IterStep::Exhausted => return Ok(values),
        }
    }
}

#[inline]
fn runtime_error_to_builtin_error(err: crate::error::RuntimeError) -> BuiltinError {
    let display = err.to_string();
    match err.kind {
        RuntimeErrorKind::TypeError { message } => BuiltinError::TypeError(message.to_string()),
        RuntimeErrorKind::UnsupportedOperandTypes { op, left, right } => BuiltinError::TypeError(
            format!("unsupported operand type(s) for {op}: '{left}' and '{right}'"),
        ),
        RuntimeErrorKind::NotCallable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not callable", type_name))
        }
        RuntimeErrorKind::NotIterable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not iterable", type_name))
        }
        RuntimeErrorKind::NotSubscriptable { type_name } => {
            BuiltinError::TypeError(format!("'{}' object is not subscriptable", type_name))
        }
        RuntimeErrorKind::AttributeError { type_name, attr } => BuiltinError::AttributeError(
            format!("'{}' object has no attribute '{}'", type_name, attr),
        ),
        RuntimeErrorKind::KeyError { key } => BuiltinError::KeyError(key.to_string()),
        RuntimeErrorKind::IndexError { index, length } => {
            BuiltinError::IndexError(format!("index {index} out of range for length {length}"))
        }
        RuntimeErrorKind::ValueError { message } => BuiltinError::ValueError(message.to_string()),
        RuntimeErrorKind::OverflowError { message } => {
            BuiltinError::OverflowError(message.to_string())
        }
        RuntimeErrorKind::StopIteration => BuiltinError::StopIteration,
        _ => BuiltinError::TypeError(display),
    }
}

#[inline]
fn with_str_receiver<R>(
    value: Value,
    method_name: &'static str,
    f: impl FnOnce(&str) -> Result<R, BuiltinError>,
) -> Result<R, BuiltinError> {
    let Some(string) = value_as_string_ref(value) else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'str.{method_name}' requires a 'str' object but received '{}'",
            value.type_name()
        )));
    };
    f(string.as_str())
}

#[inline]
fn string_object_from_value(value: Value) -> Result<StringObject, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("expected str instance".to_string()))?;
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError("expected str instance".to_string()))?;
        return Ok(StringObject::new(interned.as_str()));
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError("expected str instance".to_string()));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::STR {
        return Err(BuiltinError::TypeError("expected str instance".to_string()));
    }

    Ok(unsafe { &*(ptr as *const StringObject) }.clone())
}

#[inline]
fn expect_list_mut(
    value: Value,
    method_name: &'static str,
) -> Result<&'static mut ListObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'list.{method_name}' requires a 'list' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::LIST {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'list.{method_name}' requires a 'list' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(unsafe { &mut *(ptr as *mut ListObject) })
}

#[inline]
fn expect_list_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static ListObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'list.{method_name}' requires a 'list' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::LIST {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'list.{method_name}' requires a 'list' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(unsafe { &*(ptr as *const ListObject) })
}

#[inline]
fn expect_deque_mut(
    value: Value,
    method_name: &'static str,
) -> Result<&'static mut DequeObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'deque.{method_name}' requires a 'deque' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::DEQUE {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'deque.{method_name}' requires a 'deque' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(unsafe { &mut *(ptr as *mut DequeObject) })
}

#[inline]
fn expect_dict_receiver(value: Value, method_name: &'static str) -> Result<(), BuiltinError> {
    expect_dict_ref(value, method_name).map(|_| ())
}

#[inline]
fn expect_dict_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static DictObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'dict.{method_name}' requires a 'dict' object but received '{}'",
            value.type_name()
        )));
    };

    if let Some(dict) = dict_storage_ref_from_ptr(ptr) {
        return Ok(dict);
    }

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    Err(BuiltinError::TypeError(format!(
        "descriptor 'dict.{method_name}' requires a 'dict' object but received '{}'",
        header.type_id.name()
    )))
}

#[inline]
fn expect_dict_mut(
    value: Value,
    method_name: &'static str,
) -> Result<&'static mut DictObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'dict.{method_name}' requires a 'dict' object but received '{}'",
            value.type_name()
        )));
    };

    if let Some(dict) = dict_storage_mut_from_ptr(ptr) {
        return Ok(dict);
    }

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    Err(BuiltinError::TypeError(format!(
        "descriptor 'dict.{method_name}' requires a 'dict' object but received '{}'",
        header.type_id.name()
    )))
}

#[inline]
fn expect_mapping_proxy_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static MappingProxyObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'mappingproxy.{method_name}' requires a 'mappingproxy' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::MAPPING_PROXY {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'mappingproxy.{method_name}' requires a 'mappingproxy' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(unsafe { &*(ptr as *const MappingProxyObject) })
}

#[inline]
fn expect_set_receiver(
    value: Value,
    expected_type: TypeId,
    method_name: &'static str,
) -> Result<&'static SetObject, BuiltinError> {
    let receiver_name = expected_type.name();
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{receiver_name}.{method_name}' requires a '{receiver_name}' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != expected_type {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{receiver_name}.{method_name}' requires a '{receiver_name}' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(unsafe { &*(ptr as *const SetObject) })
}

#[inline]
fn expect_set_mut_receiver(
    value: Value,
    expected_type: TypeId,
    method_name: &'static str,
) -> Result<&'static mut SetObject, BuiltinError> {
    let receiver_name = expected_type.name();
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{receiver_name}.{method_name}' requires a '{receiver_name}' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != expected_type {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{receiver_name}.{method_name}' requires a '{receiver_name}' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(unsafe { &mut *(ptr as *mut SetObject) })
}

#[inline]
fn expect_bytearray_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static BytesObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'bytearray.{method_name}' requires a 'bytearray' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::BYTEARRAY {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'bytearray.{method_name}' requires a 'bytearray' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(unsafe { &*(ptr as *const BytesObject) })
}

#[inline]
fn to_object_value<T>(object: T) -> Value {
    let ptr = Box::leak(Box::new(object)) as *mut T as *const ();
    Value::object_ptr(ptr)
}

#[inline]
fn property_copy(
    receiver: Value,
    method_name: &'static str,
    getter_override: Option<Value>,
    setter_override: Option<Value>,
    deleter_override: Option<Value>,
) -> Result<Value, BuiltinError> {
    let descriptor = expect_property_receiver(receiver, method_name)?;
    Ok(to_object_value(PropertyDescriptor::new_full(
        select_property_accessor(descriptor.getter(), getter_override),
        select_property_accessor(descriptor.setter(), setter_override),
        select_property_accessor(descriptor.deleter(), deleter_override),
        descriptor.doc(),
    )))
}

#[inline]
fn select_property_accessor(
    current: Option<Value>,
    override_value: Option<Value>,
) -> Option<Value> {
    match override_value {
        Some(value) if !value.is_none() => Some(value),
        Some(_) | None => current,
    }
}

#[inline]
fn generator_close(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("generator", "close", args, 0)?;
    let generator = expect_generator_mut(args[0], "close")?;
    match prepare_close(generator) {
        CloseResult::Closed => Ok(Value::none()),
        CloseResult::RuntimeError(exception) => match exception.type_name.as_str() {
            "ValueError" => Err(BuiltinError::ValueError(exception.message)),
            "TypeError" => Err(BuiltinError::TypeError(exception.message)),
            _ => Err(BuiltinError::ValueError(format!(
                "{}: {}",
                exception.type_name, exception.message
            ))),
        },
        CloseResult::YieldedInFinally(_) => Err(BuiltinError::ValueError(
            "generator ignored GeneratorExit".to_string(),
        )),
    }
}

#[inline]
fn property_getter(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("property", "getter", args, 1)?;
    property_copy(args[0], "getter", Some(args[1]), None, None)
}

#[inline]
fn property_setter(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("property", "setter", args, 1)?;
    property_copy(args[0], "setter", None, Some(args[1]), None)
}

#[inline]
fn property_deleter(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("property", "deleter", args, 1)?;
    property_copy(args[0], "deleter", None, None, Some(args[1]))
}

#[inline]
fn expect_generator_mut(
    value: Value,
    method_name: &'static str,
) -> Result<&'static mut GeneratorObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'generator.{method_name}' requires a 'generator' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::GENERATOR {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'generator.{method_name}' requires a 'generator' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(unsafe { &mut *(ptr as *mut GeneratorObject) })
}

#[inline]
fn expect_property_receiver(
    value: Value,
    method_name: &'static str,
) -> Result<&'static PropertyDescriptor, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'property.{method_name}' requires a 'property' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::PROPERTY {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'property.{method_name}' requires a 'property' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(unsafe { &*(ptr as *const PropertyDescriptor) })
}

#[inline]
fn expect_function_receiver(
    value: Value,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'function.{method_name}' requires a 'function' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if !matches!(header.type_id, TypeId::FUNCTION | TypeId::CLOSURE) {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'function.{method_name}' requires a 'function' object but received '{}'",
            header.type_id.name()
        )));
    }

    Ok(value)
}

fn function_get(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '__get__' of 'function' object needs 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let function = expect_function_receiver(args[0], "__get__")?;
    let instance = args[1];
    if instance.is_none() {
        return Ok(function);
    }

    Ok(crate::ops::objects::bind_instance_attribute(
        function, instance,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::iterator_to_value;
    use prism_compiler::bytecode::CodeObject;
    use prism_core::intern::{intern, interned_by_ptr};
    use prism_runtime::object::class::PyClassObject;
    use prism_runtime::object::descriptor::BoundMethod;
    use prism_runtime::object::shape::Shape;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::object::views::MappingProxyObject;
    use prism_runtime::types::function::FunctionObject;
    use prism_runtime::types::iter::IteratorObject;
    use std::sync::Arc;

    fn boxed_list_value(list: ListObject) -> (Value, *mut ListObject) {
        let ptr = Box::into_raw(Box::new(list));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    fn list_values(ptr: *const ListObject) -> Vec<i64> {
        let list = unsafe { &*ptr };
        list.as_slice()
            .iter()
            .map(|value| value.as_int().expect("expected tagged int"))
            .collect()
    }

    fn string_value(value: Value) -> String {
        if value.is_string() {
            let ptr = value
                .as_string_object_ptr()
                .expect("tagged string should have a pointer");
            return interned_by_ptr(ptr as *const u8)
                .expect("tagged string pointer should resolve")
                .as_str()
                .to_string();
        }

        let ptr = value
            .as_object_ptr()
            .expect("string result should be an object");
        let string = unsafe { &*(ptr as *const StringObject) };
        string.as_str().to_string()
    }

    #[test]
    fn test_resolve_list_method_returns_builtin_for_append_extend_and_copy() {
        let append = resolve_list_method("append").expect("append should resolve");
        let extend = resolve_list_method("extend").expect("extend should resolve");
        let copy = resolve_list_method("copy").expect("copy should resolve");
        assert!(append.method.as_object_ptr().is_some());
        assert!(extend.method.as_object_ptr().is_some());
        assert!(copy.method.as_object_ptr().is_some());
        assert!(!append.is_descriptor);
        assert!(!extend.is_descriptor);
        assert!(!copy.is_descriptor);
    }

    #[test]
    fn test_list_append_mutates_list_and_returns_none() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::new());
        let result =
            list_append(&[list_value, Value::int(7).unwrap()]).expect("append should work");
        assert!(result.is_none());
        assert_eq!(list_values(list_ptr), vec![7]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_copy_returns_distinct_shallow_copy() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]));

        let copied = list_copy(&[list_value]).expect("copy should work");
        let copied_ptr = copied
            .as_object_ptr()
            .expect("list.copy should return a list object")
            as *mut ListObject;

        assert_eq!(list_values(list_ptr), vec![1, 2]);
        assert_eq!(list_values(copied_ptr), vec![1, 2]);
        assert_ne!(list_ptr, copied_ptr);

        unsafe {
            drop(Box::from_raw(copied_ptr));
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_resolve_deque_method_returns_builtin_methods() {
        let append = resolve_deque_method("append").expect("deque.append should resolve");
        let pop = resolve_deque_method("pop").expect("deque.pop should resolve");
        assert!(append.method.as_object_ptr().is_some());
        assert!(pop.method.as_object_ptr().is_some());
        assert!(!append.is_descriptor);
        assert!(!pop.is_descriptor);
    }

    #[test]
    fn test_deque_append_and_pop_round_trip() {
        let deque_ptr = Box::into_raw(Box::new(DequeObject::new()));
        let deque_value = Value::object_ptr(deque_ptr as *const ());

        let appended = deque_append(&[deque_value, Value::int(11).unwrap()])
            .expect("deque.append should succeed");
        assert!(appended.is_none());
        assert_eq!(unsafe { &*deque_ptr }.len(), 1);

        let popped = deque_pop(&[deque_value]).expect("deque.pop should succeed");
        assert_eq!(popped.as_int(), Some(11));
        assert!(unsafe { &*deque_ptr }.is_empty());

        unsafe {
            drop(Box::from_raw(deque_ptr));
        }
    }

    #[test]
    fn test_resolve_dict_method_returns_builtin_for_views() {
        let keys = resolve_dict_method("keys").expect("keys should resolve");
        let get = resolve_dict_method("get").expect("get should resolve");
        let len = resolve_dict_method("__len__").expect("__len__ should resolve");
        let contains = resolve_dict_method("__contains__").expect("__contains__ should resolve");
        let values = resolve_dict_method("values").expect("values should resolve");
        let items = resolve_dict_method("items").expect("items should resolve");
        let getitem = resolve_dict_method("__getitem__").expect("__getitem__ should resolve");
        let setitem = resolve_dict_method("__setitem__").expect("__setitem__ should resolve");
        let delitem = resolve_dict_method("__delitem__").expect("__delitem__ should resolve");
        let pop = resolve_dict_method("pop").expect("pop should resolve");
        let setdefault = resolve_dict_method("setdefault").expect("setdefault should resolve");
        let clear = resolve_dict_method("clear").expect("clear should resolve");
        let update = resolve_dict_method("update").expect("update should resolve");
        let copy = resolve_dict_method("copy").expect("copy should resolve");
        assert!(keys.method.as_object_ptr().is_some());
        assert!(get.method.as_object_ptr().is_some());
        assert!(len.method.as_object_ptr().is_some());
        assert!(contains.method.as_object_ptr().is_some());
        assert!(values.method.as_object_ptr().is_some());
        assert!(items.method.as_object_ptr().is_some());
        assert!(getitem.method.as_object_ptr().is_some());
        assert!(setitem.method.as_object_ptr().is_some());
        assert!(delitem.method.as_object_ptr().is_some());
        assert!(pop.method.as_object_ptr().is_some());
        assert!(setdefault.method.as_object_ptr().is_some());
        assert!(clear.method.as_object_ptr().is_some());
        assert!(update.method.as_object_ptr().is_some());
        assert!(copy.method.as_object_ptr().is_some());
        assert!(!keys.is_descriptor);
        assert!(!get.is_descriptor);
        assert!(!len.is_descriptor);
        assert!(!contains.is_descriptor);
        assert!(!values.is_descriptor);
        assert!(!items.is_descriptor);
        assert!(!getitem.is_descriptor);
        assert!(!setitem.is_descriptor);
        assert!(!delitem.is_descriptor);
        assert!(!pop.is_descriptor);
        assert!(!setdefault.is_descriptor);
        assert!(!clear.is_descriptor);
        assert!(!update.is_descriptor);
        assert!(!copy.is_descriptor);
    }

    #[test]
    fn test_dict_methods_accept_heap_dict_subclasses_with_native_backing() {
        let mut instance = ShapedObject::new_dict_backed(TypeId::from_raw(700), Shape::empty());
        instance
            .dict_backing_mut()
            .expect("dict backing should exist")
            .set(Value::string(intern("existing")), Value::int(1).unwrap());
        let ptr = Box::into_raw(Box::new(instance));
        let value = Value::object_ptr(ptr as *const ());

        dict_setitem(&[
            value,
            Value::string(intern("added")),
            Value::int(9).unwrap(),
        ])
        .expect("dict.__setitem__ should accept dict subclasses");
        assert_eq!(
            dict_contains(&[value, Value::string(intern("existing"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            dict_getitem(&[value, Value::string(intern("added"))]).unwrap(),
            Value::int(9).unwrap()
        );

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_resolve_mapping_proxy_method_returns_builtin_for_mapping_surface() {
        let keys = resolve_mapping_proxy_method("keys").expect("keys should resolve");
        let values = resolve_mapping_proxy_method("values").expect("values should resolve");
        let items = resolve_mapping_proxy_method("items").expect("items should resolve");
        let get = resolve_mapping_proxy_method("get").expect("get should resolve");
        let len = resolve_mapping_proxy_method("__len__").expect("__len__ should resolve");
        let contains =
            resolve_mapping_proxy_method("__contains__").expect("__contains__ should resolve");
        let copy = resolve_mapping_proxy_method("copy").expect("copy should resolve");

        for method in [keys, values, items, get, len, contains, copy] {
            assert!(method.method.as_object_ptr().is_some());
            assert!(!method.is_descriptor);
        }
    }

    #[test]
    fn test_mappingproxy_methods_cover_heap_class_proxies() {
        let class = Arc::new(PyClassObject::new_simple(intern("ProxyMapping")));
        class.set_attr(intern("token"), Value::int(7).unwrap());
        class.set_attr(intern("label"), Value::string(intern("ready")));

        let proxy_ptr = Box::into_raw(Box::new(MappingProxyObject::for_user_class(Arc::as_ptr(
            &class,
        ))));
        let proxy_value = Value::object_ptr(proxy_ptr as *const ());

        let keys = mappingproxy_keys(&[proxy_value]).expect("keys should succeed");
        let values = mappingproxy_values(&[proxy_value]).expect("values should succeed");
        let items = mappingproxy_items(&[proxy_value]).expect("items should succeed");
        for (value, expected_type) in [
            (keys, TypeId::DICT_KEYS),
            (values, TypeId::DICT_VALUES),
            (items, TypeId::DICT_ITEMS),
        ] {
            let ptr = value.as_object_ptr().expect("view should be object");
            let header = unsafe { &*(ptr as *const ObjectHeader) };
            assert_eq!(header.type_id, expected_type);
            unsafe {
                drop(Box::from_raw(ptr as *mut DictViewObject));
            }
        }

        assert_eq!(
            mappingproxy_get(&[proxy_value, Value::string(intern("token"))])
                .expect("get should succeed")
                .as_int(),
            Some(7)
        );
        assert_eq!(
            mappingproxy_get(&[
                proxy_value,
                Value::string(intern("missing")),
                Value::int(99).unwrap(),
            ])
            .expect("get should return provided default")
            .as_int(),
            Some(99)
        );
        assert_eq!(
            mappingproxy_len(&[proxy_value])
                .expect("__len__ should succeed")
                .as_int(),
            Some(2)
        );
        assert_eq!(
            mappingproxy_contains(&[proxy_value, Value::string(intern("label"))])
                .expect("__contains__ should succeed"),
            Value::bool(true)
        );

        let copy = mappingproxy_copy(&[proxy_value]).expect("copy should succeed");
        let copy_ptr = copy.as_object_ptr().expect("copy should be a dict");
        let copy_dict = unsafe { &*(copy_ptr as *const DictObject) };
        assert_eq!(
            copy_dict
                .get(Value::string(intern("token")))
                .expect("copied token should exist")
                .as_int(),
            Some(7)
        );

        unsafe {
            drop(Box::from_raw(copy_ptr as *mut DictObject));
            drop(Box::from_raw(proxy_ptr));
        }
    }

    #[test]
    fn test_resolve_object_method_returns_builtin_for_rich_comparisons() {
        let eq = resolve_object_method("__eq__").expect("__eq__ should resolve");
        let ne = resolve_object_method("__ne__").expect("__ne__ should resolve");
        assert!(eq.method.as_object_ptr().is_some());
        assert!(ne.method.as_object_ptr().is_some());
        assert!(!eq.is_descriptor);
        assert!(!ne.is_descriptor);
        assert!(resolve_object_method("__lt__").is_none());
    }

    #[test]
    fn test_resolve_object_method_returns_builtin_for_representation_dunders() {
        let repr_method = resolve_object_method("__repr__").expect("__repr__ should resolve");
        let str_method = resolve_object_method("__str__").expect("__str__ should resolve");
        let format_method = resolve_object_method("__format__").expect("__format__ should resolve");

        let repr_builtin = unsafe {
            &*(repr_method
                .method
                .as_object_ptr()
                .expect("__repr__ should be allocated")
                as *const BuiltinFunctionObject)
        };
        let str_builtin = unsafe {
            &*(str_method
                .method
                .as_object_ptr()
                .expect("__str__ should be allocated")
                as *const BuiltinFunctionObject)
        };
        let format_builtin = unsafe {
            &*(format_method
                .method
                .as_object_ptr()
                .expect("__format__ should be allocated")
                as *const BuiltinFunctionObject)
        };

        assert_eq!(repr_builtin.name(), "object.__repr__");
        assert_eq!(str_builtin.name(), "object.__str__");
        assert_eq!(format_builtin.name(), "object.__format__");
    }

    #[test]
    fn test_resolve_type_method_renders_builtin_type_repr() {
        let method = resolve_type_method("__repr__").expect("type.__repr__ should resolve");
        let builtin = unsafe {
            &*(method
                .method
                .as_object_ptr()
                .expect("type.__repr__ should be allocated")
                as *const BuiltinFunctionObject)
        };
        let rendered = builtin
            .call(&[crate::builtins::builtin_type_object_for_type_id(
                TypeId::INT,
            )])
            .expect("type.__repr__(int) should succeed");

        assert_eq!(string_value(rendered), "<class 'int'>");
    }

    #[test]
    fn test_numeric_and_string_resolvers_expose_representation_dunders() {
        for (resolver, owner) in [
            (
                resolve_int_method as fn(&str) -> Option<CachedMethod>,
                "int",
            ),
            (
                resolve_bool_method as fn(&str) -> Option<CachedMethod>,
                "bool",
            ),
            (
                resolve_float_method as fn(&str) -> Option<CachedMethod>,
                "float",
            ),
            (
                resolve_str_method as fn(&str) -> Option<CachedMethod>,
                "str",
            ),
        ] {
            let repr_builtin = unsafe {
                &*(resolver("__repr__")
                    .expect("__repr__ should resolve")
                    .method
                    .as_object_ptr()
                    .expect("__repr__ should be allocated")
                    as *const BuiltinFunctionObject)
            };
            let str_builtin = unsafe {
                &*(resolver("__str__")
                    .expect("__str__ should resolve")
                    .method
                    .as_object_ptr()
                    .expect("__str__ should be allocated")
                    as *const BuiltinFunctionObject)
            };
            let format_builtin = unsafe {
                &*(resolver("__format__")
                    .expect("__format__ should resolve")
                    .method
                    .as_object_ptr()
                    .expect("__format__ should be allocated")
                    as *const BuiltinFunctionObject)
            };

            assert_eq!(repr_builtin.name(), format!("{owner}.__repr__"));
            assert_eq!(str_builtin.name(), format!("{owner}.__str__"));
            assert_eq!(format_builtin.name(), format!("{owner}.__format__"));
        }
    }

    #[test]
    fn test_object_rich_comparisons_follow_identity_default() {
        let same = Value::int(7).unwrap();
        assert_eq!(
            object_eq(&[same, same]).expect("__eq__ should accept two operands"),
            Value::bool(true)
        );
        assert_eq!(
            object_ne(&[same, same]).expect("__ne__ should accept two operands"),
            Value::bool(false)
        );

        let lhs = Value::int(7).unwrap();
        let rhs = Value::int(8).unwrap();
        assert_eq!(
            object_eq(&[lhs, rhs]).expect("__eq__ should accept mismatched operands"),
            builtin_not_implemented_value()
        );
        assert_eq!(
            object_ne(&[lhs, rhs]).expect("__ne__ should accept mismatched operands"),
            builtin_not_implemented_value()
        );
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_upper() {
        let upper = resolve_str_method("upper").expect("upper should resolve");
        assert!(upper.method.as_object_ptr().is_some());
        assert!(!upper.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_lower() {
        let lower = resolve_str_method("lower").expect("lower should resolve");
        assert!(lower.method.as_object_ptr().is_some());
        assert!(!lower.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_replace() {
        let replace = resolve_str_method("replace").expect("replace should resolve");
        assert!(replace.method.as_object_ptr().is_some());
        assert!(!replace.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_split() {
        let split = resolve_str_method("split").expect("split should resolve");
        assert!(split.method.as_object_ptr().is_some());
        assert!(!split.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_strip_family() {
        for name in ["strip", "lstrip", "rstrip"] {
            let method = resolve_str_method(name).expect("strip-family method should resolve");
            assert!(method.method.as_object_ptr().is_some(), "{name} should resolve");
            assert!(!method.is_descriptor, "{name} should not be a descriptor");
        }
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_join() {
        let join = resolve_str_method("join").expect("join should resolve");
        assert!(join.method.as_object_ptr().is_some());
        assert!(!join.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_isidentifier() {
        let method = resolve_str_method("isidentifier").expect("isidentifier should resolve");
        assert!(method.method.as_object_ptr().is_some());
        assert!(!method.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_startswith_and_endswith() {
        let startswith = resolve_str_method("startswith").expect("startswith should resolve");
        let endswith = resolve_str_method("endswith").expect("endswith should resolve");
        assert!(startswith.method.as_object_ptr().is_some());
        assert!(endswith.method.as_object_ptr().is_some());
        assert!(!startswith.is_descriptor);
        assert!(!endswith.is_descriptor);
    }

    #[test]
    fn test_str_upper_returns_uppercased_string() {
        let result = str_upper(&[Value::string(prism_core::intern::intern("Path"))])
            .expect("upper should work");
        assert_eq!(string_value(result), "PATH");
    }

    #[test]
    fn test_str_lower_returns_lowercased_string_and_reuses_unchanged_receiver() {
        let mixed = Value::string(prism_core::intern::intern("Path"));
        let lowered = str_lower(&[mixed]).expect("lower should work");
        assert_eq!(string_value(lowered), "path");

        let already_lower = Value::string(prism_core::intern::intern("path"));
        let unchanged = str_lower(&[already_lower]).expect("lower should preserve lowercase");
        assert_eq!(unchanged, already_lower);
    }

    #[test]
    fn test_str_replace_replaces_occurrences_and_honors_count() {
        let replaced = str_replace(&[
            Value::string(intern("banana")),
            Value::string(intern("na")),
            Value::string(intern("NA")),
        ])
        .expect("replace should work");
        assert_eq!(string_value(replaced), "baNANA");

        let counted = str_replace(&[
            Value::string(intern("banana")),
            Value::string(intern("na")),
            Value::string(intern("NA")),
            Value::int(1).unwrap(),
        ])
        .expect("counted replace should work");
        assert_eq!(string_value(counted), "baNAna");
    }

    #[test]
    fn test_str_replace_supports_empty_pattern_and_bool_count() {
        let all = str_replace(&[
            Value::string(intern("abc")),
            Value::string(intern("")),
            Value::string(intern("-")),
        ])
        .expect("replace with empty pattern should work");
        assert_eq!(string_value(all), "-a-b-c-");

        let limited = str_replace(&[
            Value::string(intern("abc")),
            Value::string(intern("")),
            Value::string(intern("-")),
            Value::int(3).unwrap(),
        ])
        .expect("counted empty-pattern replace should work");
        assert_eq!(string_value(limited), "-a-b-c");

        let bool_count = str_replace(&[
            Value::string(intern("aaaa")),
            Value::string(intern("a")),
            Value::string(intern("b")),
            Value::bool(true),
        ])
        .expect("bool count should be treated as an int");
        assert_eq!(string_value(bool_count), "baaa");
    }

    #[test]
    fn test_str_replace_reuses_receiver_when_nothing_changes() {
        let receiver = Value::string(intern("banana"));

        let unchanged = str_replace(&[
            receiver,
            Value::string(intern("x")),
            Value::string(intern("y")),
        ])
        .expect("replace should succeed when there is nothing to replace");
        assert_eq!(unchanged, receiver);

        let zero_count = str_replace(&[
            receiver,
            Value::string(intern("a")),
            Value::string(intern("b")),
            Value::bool(false),
        ])
        .expect("replace with count=False should succeed");
        assert_eq!(zero_count, receiver);
    }

    #[test]
    fn test_str_replace_rejects_non_string_operands_and_non_int_count() {
        let non_string_old = str_replace(&[
            Value::string(intern("banana")),
            Value::int(1).unwrap(),
            Value::string(intern("b")),
        ])
        .expect_err("old pattern must be a string");
        assert!(
            non_string_old
                .to_string()
                .contains("str.replace() argument 1 must be str")
        );

        let non_string_new = str_replace(&[
            Value::string(intern("banana")),
            Value::string(intern("a")),
            Value::int(1).unwrap(),
        ])
        .expect_err("replacement must be a string");
        assert!(
            non_string_new
                .to_string()
                .contains("str.replace() argument 2 must be str")
        );

        let non_int_count = str_replace(&[
            Value::string(intern("banana")),
            Value::string(intern("a")),
            Value::string(intern("b")),
            Value::float(1.5),
        ])
        .expect_err("count must be integer-like");
        assert!(
            non_int_count
                .to_string()
                .contains("str.replace() argument 3 must be int")
        );
    }

    #[test]
    fn test_str_split_supports_explicit_separator_and_maxsplit() {
        let result = str_split(&[
            Value::string(intern("a::b::c")),
            Value::string(intern("::")),
            Value::int(1).unwrap(),
        ])
        .expect("split with separator should work");

        let result_ptr = result.as_object_ptr().expect("split should return a list");
        let list = unsafe { &*(result_ptr as *const ListObject) };
        let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
        assert_eq!(values, vec!["a".to_string(), "b::c".to_string()]);
    }

    #[test]
    fn test_str_split_supports_whitespace_splitting_and_none_separator() {
        let whitespace = str_split(&[
            Value::string(intern("  alpha   beta gamma  ")),
            Value::none(),
            Value::int(1).unwrap(),
        ])
        .expect("split with implicit whitespace should work");
        let whitespace_ptr = whitespace
            .as_object_ptr()
            .expect("whitespace split should return a list");
        let whitespace_list = unsafe { &*(whitespace_ptr as *const ListObject) };
        let values: Vec<String> = whitespace_list
            .as_slice()
            .iter()
            .copied()
            .map(string_value)
            .collect();
        assert_eq!(
            values,
            vec!["alpha".to_string(), "beta gamma  ".to_string()]
        );

        let zero = str_split(&[
            Value::string(intern("   keep  spacing")),
            Value::none(),
            Value::int(0).unwrap(),
        ])
        .expect("split with maxsplit=0 should work");
        let zero_ptr = zero
            .as_object_ptr()
            .expect("zero split should return a list");
        let zero_list = unsafe { &*(zero_ptr as *const ListObject) };
        let zero_values: Vec<String> = zero_list
            .as_slice()
            .iter()
            .copied()
            .map(string_value)
            .collect();
        assert_eq!(zero_values, vec!["keep  spacing".to_string()]);
    }

    #[test]
    fn test_str_split_rejects_empty_separator() {
        let err = str_split(&[Value::string(intern("abc")), Value::string(intern(""))])
            .expect_err("empty separator should fail");
        assert!(err.to_string().contains("empty separator"));
    }

    #[test]
    fn test_str_strip_family_supports_whitespace_none_and_explicit_char_sets() {
        let stripped = str_strip(&[Value::string(intern("  alpha  "))])
            .expect("strip should trim surrounding whitespace");
        assert_eq!(string_value(stripped), "alpha");

        let none_chars = str_lstrip(&[
            Value::string(intern("\n\t alpha")),
            Value::none(),
        ])
        .expect("lstrip should accept None as the default whitespace matcher");
        assert_eq!(string_value(none_chars), "alpha");

        let explicit = str_rstrip(&[
            Value::string(intern("path\\\\//")),
            Value::string(intern("/\\")),
        ])
        .expect("rstrip should support explicit trim character sets");
        assert_eq!(string_value(explicit), "path");
    }

    #[test]
    fn test_str_strip_family_reuses_receiver_when_nothing_changes_or_chars_are_empty() {
        let receiver = Value::string(intern("stable"));

        let unchanged = str_strip(&[receiver]).expect("strip should succeed");
        assert_eq!(unchanged, receiver);

        let empty_chars = str_rstrip(&[
            receiver,
            Value::string(intern("")),
        ])
        .expect("rstrip with empty char set should be a no-op");
        assert_eq!(empty_chars, receiver);
    }

    #[test]
    fn test_str_strip_family_rejects_non_string_char_sets() {
        let err = str_strip(&[
            Value::string(intern("value")),
            Value::int(1).unwrap(),
        ])
        .expect_err("strip chars must be strings or None");
        assert!(err.to_string().contains("str.strip() argument 1 must be str"));
    }

    #[test]
    fn test_str_join_concatenates_iterable_of_strings() {
        let parts = prism_runtime::types::tuple::TupleObject::from_slice(&[
            Value::string(intern("hits")),
            Value::string(intern("misses")),
            Value::string(intern("maxsize")),
        ]);
        let parts_ptr = Box::into_raw(Box::new(parts));
        let result = str_join(&[
            Value::string(intern(", ")),
            Value::object_ptr(parts_ptr as *const ()),
        ])
        .expect("join should work");
        assert_eq!(string_value(result), "hits, misses, maxsize");

        unsafe {
            drop(Box::from_raw(parts_ptr));
        }
    }

    #[test]
    fn test_str_join_rejects_non_string_items() {
        let parts = prism_runtime::types::tuple::TupleObject::from_slice(&[
            Value::string(intern("hits")),
            Value::int(1).unwrap(),
        ]);
        let parts_ptr = Box::into_raw(Box::new(parts));
        let err = str_join(&[
            Value::string(intern(", ")),
            Value::object_ptr(parts_ptr as *const ()),
        ])
        .expect_err("join should reject non-strings");
        assert!(err.to_string().contains("sequence item 1"));

        unsafe {
            drop(Box::from_raw(parts_ptr));
        }
    }

    #[test]
    fn test_str_isidentifier_accepts_ascii_and_unicode_identifiers() {
        assert_eq!(
            str_isidentifier(&[Value::string(intern("cache_info"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_isidentifier(&[Value::string(intern("_lru_cache_wrapper"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_isidentifier(&[Value::string(intern("cafe"))]).unwrap(),
            Value::bool(true)
        );

        let unicode =
            Value::object_ptr(Box::into_raw(Box::new(StringObject::new("πcache"))) as *const ());
        let unicode_result = str_isidentifier(&[unicode]).unwrap();
        assert_eq!(unicode_result, Value::bool(true));
        unsafe {
            drop(Box::from_raw(
                unicode.as_object_ptr().unwrap() as *mut StringObject
            ));
        }
    }

    #[test]
    fn test_str_isidentifier_rejects_invalid_identifier_forms() {
        for candidate in ["", "2cache", "cache-info", "cache info"] {
            assert_eq!(
                str_isidentifier(&[Value::string(intern(candidate))]).unwrap(),
                Value::bool(false),
                "{candidate:?} should not be a valid identifier"
            );
        }
    }

    #[test]
    fn test_str_startswith_supports_single_prefix_tuple_prefixes_and_bounds() {
        assert_eq!(
            str_startswith(&[
                Value::string(intern("CacheInfo")),
                Value::string(intern("Cache")),
            ])
            .unwrap(),
            Value::bool(true)
        );

        let prefixes = prism_runtime::types::tuple::TupleObject::from_slice(&[
            Value::string(intern("miss")),
            Value::string(intern("Cache")),
        ]);
        let prefixes_ptr = Box::into_raw(Box::new(prefixes));
        let tuple_result = str_startswith(&[
            Value::string(intern("CacheInfo")),
            Value::object_ptr(prefixes_ptr as *const ()),
        ])
        .unwrap();
        assert_eq!(tuple_result, Value::bool(true));

        let bounded = str_startswith(&[
            Value::string(intern("xxCacheInfo")),
            Value::string(intern("Cache")),
            Value::int(2).unwrap(),
            Value::int(7).unwrap(),
        ])
        .unwrap();
        assert_eq!(bounded, Value::bool(true));

        unsafe {
            drop(Box::from_raw(prefixes_ptr));
        }
    }

    #[test]
    fn test_str_endswith_supports_suffixes_and_negative_bounds() {
        assert_eq!(
            str_endswith(&[
                Value::string(intern("functools.py")),
                Value::string(intern(".py")),
            ])
            .unwrap(),
            Value::bool(true)
        );

        let bounded = str_endswith(&[
            Value::string(intern("prefix_suffix_tail")),
            Value::string(intern("suffix")),
            Value::int(7).unwrap(),
            Value::int(-5).unwrap(),
        ])
        .unwrap();
        assert_eq!(bounded, Value::bool(true));
    }

    #[test]
    fn test_str_startswith_rejects_non_string_affix_members() {
        let prefixes = prism_runtime::types::tuple::TupleObject::from_slice(&[
            Value::string(intern("Miss")),
            Value::int(1).unwrap(),
        ]);
        let prefixes_ptr = Box::into_raw(Box::new(prefixes));
        let err = str_startswith(&[
            Value::string(intern("CacheInfo")),
            Value::object_ptr(prefixes_ptr as *const ()),
        ])
        .expect_err("tuple prefixes should reject non-string members");
        assert!(
            err.to_string()
                .contains("first arg must be str or a tuple of str")
        );

        unsafe {
            drop(Box::from_raw(prefixes_ptr));
        }
    }

    #[test]
    fn test_dict_view_methods_return_specialized_view_objects() {
        let dict = Box::new(prism_runtime::types::dict::DictObject::new());
        let dict_ptr = Box::into_raw(dict);
        let dict_value = Value::object_ptr(dict_ptr as *const ());

        let keys = dict_keys(&[dict_value]).expect("keys should work");
        let values = dict_values(&[dict_value]).expect("values should work");
        let items = dict_items(&[dict_value]).expect("items should work");

        for (value, expected_type) in [
            (keys, TypeId::DICT_KEYS),
            (values, TypeId::DICT_VALUES),
            (items, TypeId::DICT_ITEMS),
        ] {
            let result_ptr = value
                .as_object_ptr()
                .expect("dict view should return an object");
            let header = unsafe { &*(result_ptr as *const ObjectHeader) };
            assert_eq!(header.type_id, expected_type);
            unsafe {
                drop(Box::from_raw(result_ptr as *mut DictViewObject));
            }
        }

        unsafe {
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_dict_item_methods_mutate_and_read_entries() {
        let mut dict = DictObject::new();
        let dict_ptr = Box::into_raw(Box::new(dict));
        let dict_value = Value::object_ptr(dict_ptr as *const ());
        let key = Value::string(intern("token"));
        let value = Value::int(7).unwrap();

        dict_setitem(&[dict_value, key, value]).expect("__setitem__ should work");
        assert_eq!(
            dict_getitem(&[dict_value, key])
                .expect("__getitem__ should work")
                .as_int(),
            Some(7)
        );
        assert_eq!(
            dict_pop(&[dict_value, key])
                .expect("pop should work")
                .as_int(),
            Some(7)
        );

        unsafe {
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_dict_copy_returns_distinct_mapping_with_same_entries() {
        let mut dict = DictObject::new();
        dict.set(Value::string(intern("a")), Value::int(1).unwrap());
        dict.set(Value::string(intern("b")), Value::int(2).unwrap());
        let dict_ptr = Box::into_raw(Box::new(dict));
        let dict_value = Value::object_ptr(dict_ptr as *const ());

        let copied = dict_copy(&[dict_value]).expect("dict.copy should work");
        let copied_ptr = copied
            .as_object_ptr()
            .expect("dict.copy should return a dict object")
            as *mut DictObject;

        let original = unsafe { &*dict_ptr };
        let duplicate = unsafe { &*copied_ptr };
        assert_eq!(original.len(), duplicate.len());
        assert_eq!(
            duplicate.get(Value::string(intern("a"))).unwrap().as_int(),
            Some(1)
        );
        assert_eq!(
            duplicate.get(Value::string(intern("b"))).unwrap().as_int(),
            Some(2)
        );
        assert_ne!(dict_ptr, copied_ptr);

        unsafe {
            drop(Box::from_raw(copied_ptr));
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_dict_get_returns_existing_and_default_values() {
        let mut dict = DictObject::new();
        dict.set(Value::string(intern("token")), Value::int(7).unwrap());
        let dict_ptr = Box::into_raw(Box::new(dict));
        let dict_value = Value::object_ptr(dict_ptr as *const ());

        assert_eq!(
            dict_get(&[dict_value, Value::string(intern("token"))])
                .expect("get should return existing value")
                .as_int(),
            Some(7)
        );
        assert_eq!(
            dict_get(&[
                dict_value,
                Value::string(intern("missing")),
                Value::int(42).unwrap(),
            ])
            .expect("get should return default")
            .as_int(),
            Some(42)
        );
        assert!(
            dict_get(&[dict_value, Value::string(intern("missing"))])
                .expect("get without default should return None")
                .is_none()
        );

        unsafe {
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_dict_len_returns_current_size() {
        let mut dict = DictObject::new();
        dict.set(Value::string(intern("a")), Value::int(1).unwrap());
        dict.set(Value::string(intern("b")), Value::int(2).unwrap());
        let dict_ptr = Box::into_raw(Box::new(dict));
        let dict_value = Value::object_ptr(dict_ptr as *const ());

        assert_eq!(
            dict_len(&[dict_value])
                .expect("__len__ should succeed")
                .as_int(),
            Some(2)
        );

        unsafe {
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_dict_delitem_and_clear_update_storage() {
        let mut dict = DictObject::new();
        dict.set(Value::string(intern("a")), Value::int(1).unwrap());
        dict.set(Value::string(intern("b")), Value::int(2).unwrap());
        let dict_ptr = Box::into_raw(Box::new(dict));
        let dict_value = Value::object_ptr(dict_ptr as *const ());

        dict_delitem(&[dict_value, Value::string(intern("a"))]).expect("__delitem__ should work");
        let dict_ref = unsafe { &*(dict_ptr as *const DictObject) };
        assert!(!dict_ref.contains_key(Value::string(intern("a"))));
        assert!(dict_ref.contains_key(Value::string(intern("b"))));

        dict_clear(&[dict_value]).expect("clear should work");
        let dict_ref = unsafe { &*(dict_ptr as *const DictObject) };
        assert!(dict_ref.is_empty());

        unsafe {
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_dict_setdefault_inserts_once_and_preserves_existing_value() {
        let mut dict = DictObject::new();
        let existing_key = Value::string(intern("existing"));
        dict.set(existing_key, Value::int_unchecked(1));
        let dict_ptr = Box::into_raw(Box::new(dict));
        let dict_value = Value::object_ptr(dict_ptr as *const ());

        let inserted = dict_setdefault(&[dict_value, Value::string(intern("new"))])
            .expect("setdefault should insert a None default");
        let existing = dict_setdefault(&[dict_value, existing_key, Value::int_unchecked(99)])
            .expect("setdefault should preserve the existing value");

        let dict_ref = unsafe { &*dict_ptr };
        assert!(inserted.is_none());
        assert!(
            dict_ref
                .get(Value::string(intern("new")))
                .unwrap()
                .is_none()
        );
        assert_eq!(existing.as_int(), Some(1));
        assert_eq!(dict_ref.get(existing_key).unwrap().as_int(), Some(1));

        unsafe {
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_dict_update_merges_other_dict_entries() {
        let mut target = DictObject::new();
        target.set(Value::string(intern("a")), Value::int(1).unwrap());
        let target_ptr = Box::into_raw(Box::new(target));
        let target_value = Value::object_ptr(target_ptr as *const ());

        let mut source = DictObject::new();
        source.set(Value::string(intern("b")), Value::int(2).unwrap());
        source.set(Value::string(intern("a")), Value::int(3).unwrap());
        let source_ptr = Box::into_raw(Box::new(source));
        let source_value = Value::object_ptr(source_ptr as *const ());

        let mut vm = VirtualMachine::new();
        dict_update_with_vm(&mut vm, &[target_value, source_value]).expect("update should work");

        let target_ref = unsafe { &*target_ptr };
        assert_eq!(
            target_ref.get(Value::string(intern("a"))).unwrap().as_int(),
            Some(3)
        );
        assert_eq!(
            target_ref.get(Value::string(intern("b"))).unwrap().as_int(),
            Some(2)
        );

        unsafe {
            drop(Box::from_raw(source_ptr));
            drop(Box::from_raw(target_ptr));
        }
    }

    #[test]
    fn test_dict_update_accepts_iterable_of_pairs() {
        let target_ptr = Box::into_raw(Box::new(DictObject::new()));
        let target_value = Value::object_ptr(target_ptr as *const ());

        let pair_a = Box::into_raw(Box::new(TupleObject::from_slice(&[
            Value::string(intern("left")),
            Value::int(10).unwrap(),
        ])));
        let pair_b = Box::into_raw(Box::new(TupleObject::from_slice(&[
            Value::string(intern("right")),
            Value::int(20).unwrap(),
        ])));
        let items_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[
            Value::object_ptr(pair_a as *const ()),
            Value::object_ptr(pair_b as *const ()),
        ])));
        let items_value = Value::object_ptr(items_ptr as *const ());

        let mut vm = VirtualMachine::new();
        dict_update_with_vm(&mut vm, &[target_value, items_value])
            .expect("iterable update should work");

        let target_ref = unsafe { &*target_ptr };
        assert_eq!(
            target_ref
                .get(Value::string(intern("left")))
                .unwrap()
                .as_int(),
            Some(10)
        );
        assert_eq!(
            target_ref
                .get(Value::string(intern("right")))
                .unwrap()
                .as_int(),
            Some(20)
        );

        unsafe {
            drop(Box::from_raw(items_ptr));
            drop(Box::from_raw(pair_b));
            drop(Box::from_raw(pair_a));
            drop(Box::from_raw(target_ptr));
        }
    }

    #[test]
    fn test_resolve_set_method_returns_builtin_for_membership_copy_and_pop() {
        let set_add = resolve_set_method(TypeId::SET, "add").expect("set.add should resolve");
        let set_pop = resolve_set_method(TypeId::SET, "pop").expect("set.pop should resolve");
        let set_copy = resolve_set_method(TypeId::SET, "copy").expect("set.copy should resolve");
        let set_contains =
            resolve_set_method(TypeId::SET, "__contains__").expect("set membership should resolve");
        let frozenset_copy =
            resolve_set_method(TypeId::FROZENSET, "copy").expect("frozenset.copy should resolve");
        let frozenset_contains = resolve_set_method(TypeId::FROZENSET, "__contains__")
            .expect("frozenset membership should resolve");
        assert!(set_add.method.as_object_ptr().is_some());
        assert!(set_pop.method.as_object_ptr().is_some());
        assert!(set_copy.method.as_object_ptr().is_some());
        assert!(set_contains.method.as_object_ptr().is_some());
        assert!(frozenset_copy.method.as_object_ptr().is_some());
        assert!(frozenset_contains.method.as_object_ptr().is_some());
        assert!(!set_add.is_descriptor);
        assert!(!set_pop.is_descriptor);
        assert!(!set_copy.is_descriptor);
        assert!(!set_contains.is_descriptor);
        assert!(!frozenset_copy.is_descriptor);
        assert!(!frozenset_contains.is_descriptor);
    }

    #[test]
    fn test_set_add_mutates_receiver_and_returns_none() {
        let set = SetObject::new();
        let ptr = Box::into_raw(Box::new(set));
        let value = Value::object_ptr(ptr as *const ());

        let result = set_add(&[value, Value::int(11).unwrap()]).expect("add should work");
        assert!(result.is_none());
        assert!(unsafe { &*ptr }.contains(Value::int(11).unwrap()));

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_set_pop_removes_and_returns_single_member() {
        let set = SetObject::from_slice(&[Value::int(11).unwrap()]);
        let ptr = Box::into_raw(Box::new(set));
        let value = Value::object_ptr(ptr as *const ());

        let popped = set_pop(&[value]).expect("set.pop should work");
        assert_eq!(popped.as_int(), Some(11));
        assert!(unsafe { &*ptr }.is_empty());

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_set_pop_errors_on_empty_set() {
        let set = SetObject::new();
        let ptr = Box::into_raw(Box::new(set));
        let value = Value::object_ptr(ptr as *const ());

        let error = set_pop(&[value]).expect_err("empty set.pop should fail");
        assert!(matches!(error, BuiltinError::KeyError(_)));
        assert_eq!(error.to_string(), "KeyError: pop from an empty set");

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_set_copy_returns_distinct_set() {
        let set = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let ptr = Box::into_raw(Box::new(set));
        let value = Value::object_ptr(ptr as *const ());

        let copied = set_copy(&[value]).expect("set.copy should work");
        let copied_ptr = copied
            .as_object_ptr()
            .expect("set.copy should return an object") as *mut SetObject;

        assert!(unsafe { &*copied_ptr }.contains(Value::int(1).unwrap()));
        assert!(unsafe { &*copied_ptr }.contains(Value::int(2).unwrap()));
        assert_ne!(ptr, copied_ptr);

        unsafe {
            drop(Box::from_raw(copied_ptr));
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_frozenset_contains_reports_membership() {
        let mut frozenset =
            SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        frozenset.header.type_id = TypeId::FROZENSET;
        let ptr = Box::into_raw(Box::new(frozenset));
        let value = Value::object_ptr(ptr as *const ());

        let present =
            frozenset_contains(&[value, Value::int(2).unwrap()]).expect("contains should work");
        let missing =
            frozenset_contains(&[value, Value::int(9).unwrap()]).expect("contains should work");
        assert_eq!(present.as_bool(), Some(true));
        assert_eq!(missing.as_bool(), Some(false));

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_frozenset_copy_returns_same_object() {
        let mut frozenset =
            SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        frozenset.header.type_id = TypeId::FROZENSET;
        let ptr = Box::into_raw(Box::new(frozenset));
        let value = Value::object_ptr(ptr as *const ());

        let copied = frozenset_copy(&[value]).expect("frozenset.copy should work");
        assert_eq!(copied, value);

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_resolve_bytearray_method_returns_builtin_for_copy() {
        let copy = resolve_bytearray_method("copy").expect("bytearray.copy should resolve");
        assert!(copy.method.as_object_ptr().is_some());
        assert!(!copy.is_descriptor);
    }

    #[test]
    fn test_bytearray_copy_returns_new_mutable_sequence() {
        let bytearray = BytesObject::bytearray_from_slice(b"ab");
        let ptr = Box::into_raw(Box::new(bytearray));
        let value = Value::object_ptr(ptr as *const ());

        let copied = bytearray_copy(&[value]).expect("bytearray.copy should work");
        let copied_ptr = copied
            .as_object_ptr()
            .expect("bytearray.copy should return an object")
            as *mut BytesObject;

        let copied_ref = unsafe { &*copied_ptr };
        assert!(copied_ref.is_bytearray());
        assert_eq!(copied_ref.as_bytes(), b"ab");
        assert_ne!(ptr, copied_ptr);

        unsafe {
            drop(Box::from_raw(copied_ptr));
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_list_extend_consumes_iterable_values() {
        let (list_value, list_ptr) =
            boxed_list_value(ListObject::from_slice(&[Value::int(1).unwrap()]));
        let iter_value = iterator_to_value(IteratorObject::from_values(vec![
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));

        let result = list_extend(&[list_value, iter_value]).expect("extend should work");
        assert!(result.is_none());
        assert_eq!(list_values(list_ptr), vec![1, 2, 3]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_extend_self_duplicates_once() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(4).unwrap(),
            Value::int(5).unwrap(),
        ]));

        list_extend(&[list_value, list_value]).expect("self-extend should work");
        assert_eq!(list_values(list_ptr), vec![4, 5, 4, 5]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_extend_rejects_non_iterable_values() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::new());
        let err =
            list_extend(&[list_value, Value::int(42).unwrap()]).expect_err("extend should fail");
        assert!(err.to_string().contains("not iterable"));

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_method_argument_validation_matches_method_shape() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::new());
        let append_err = list_append(&[list_value]).expect_err("append should require a value");
        assert!(append_err.to_string().contains("list.append()"));

        let extend_err = list_extend(&[list_value]).expect_err("extend should require an iterable");
        assert!(extend_err.to_string().contains("list.extend()"));

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_resolve_generator_method_returns_builtin_for_close() {
        let close = resolve_generator_method("close").expect("close should resolve");
        assert!(close.method.as_object_ptr().is_some());
        assert!(!close.is_descriptor);
    }

    #[test]
    fn test_resolve_function_method_binds_descriptor_surface() {
        let get = resolve_function_method("__get__").expect("__get__ should resolve");
        assert!(get.method.as_object_ptr().is_some());
        assert!(!get.is_descriptor);

        let builtin = unsafe {
            &*(get
                .method
                .as_object_ptr()
                .expect("__get__ should be allocated")
                as *const BuiltinFunctionObject)
        };

        let function = FunctionObject::new(
            Arc::new(CodeObject::new("demo", "<test>")),
            Arc::from("demo"),
            None,
            None,
        );
        let function_ptr = Box::into_raw(Box::new(function));
        let function_value = Value::object_ptr(function_ptr as *const ());
        let bound_builtin = builtin.bind(function_value);

        assert_eq!(
            bound_builtin
                .call(&[Value::none()])
                .expect("__get__(None) should return the function"),
            function_value
        );

        let bound_method = bound_builtin
            .call(&[Value::int(7).unwrap()])
            .expect("__get__(instance) should bind the function");
        let bound_ptr = bound_method
            .as_object_ptr()
            .expect("bound method should be heap allocated");
        assert_eq!(
            unsafe { &*(bound_ptr as *const ObjectHeader) }.type_id,
            TypeId::METHOD
        );

        let bound = unsafe { &*(bound_ptr as *const BoundMethod) };
        assert_eq!(bound.function(), function_value);
        assert_eq!(bound.instance(), Value::int(7).unwrap());

        unsafe {
            drop(Box::from_raw(function_ptr));
        }
    }

    #[test]
    fn test_resolve_property_method_returns_builtin_methods() {
        let getter = resolve_property_method("getter").expect("getter should resolve");
        let setter = resolve_property_method("setter").expect("setter should resolve");
        let deleter = resolve_property_method("deleter").expect("deleter should resolve");
        assert!(getter.method.as_object_ptr().is_some());
        assert!(setter.method.as_object_ptr().is_some());
        assert!(deleter.method.as_object_ptr().is_some());
        assert!(!getter.is_descriptor);
        assert!(!setter.is_descriptor);
        assert!(!deleter.is_descriptor);
    }

    #[test]
    fn test_property_copy_methods_replace_requested_accessor_only() {
        let getter = Value::int(1).unwrap();
        let setter = Value::int(2).unwrap();
        let deleter = Value::int(3).unwrap();
        let doc = Value::string(intern("doc"));
        let property =
            PropertyDescriptor::new_full(Some(getter), Some(setter), Some(deleter), Some(doc));
        let property_ptr = Box::into_raw(Box::new(property));
        let property_value = Value::object_ptr(property_ptr as *const ());

        let copied = property_setter(&[property_value, Value::int(22).unwrap()])
            .expect("setter copy should work");
        let copied_ptr = copied
            .as_object_ptr()
            .expect("property copy should produce an object");
        let copied_desc = unsafe { &*(copied_ptr as *const PropertyDescriptor) };
        assert_eq!(copied_desc.getter(), Some(getter));
        assert_eq!(copied_desc.setter(), Some(Value::int(22).unwrap()));
        assert_eq!(copied_desc.deleter(), Some(deleter));
        assert_eq!(copied_desc.doc(), Some(doc));

        let copied =
            property_getter(&[property_value, Value::none()]).expect("getter copy should work");
        let copied_ptr2 = copied
            .as_object_ptr()
            .expect("property copy should produce an object");
        let copied_desc = unsafe { &*(copied_ptr2 as *const PropertyDescriptor) };
        assert_eq!(copied_desc.getter(), Some(getter));
        assert_eq!(copied_desc.setter(), Some(setter));
        assert_eq!(copied_desc.deleter(), Some(deleter));

        unsafe {
            drop(Box::from_raw(copied_ptr as *mut PropertyDescriptor));
            drop(Box::from_raw(copied_ptr2 as *mut PropertyDescriptor));
            drop(Box::from_raw(property_ptr));
        }
    }

    #[test]
    fn test_generator_close_exhausts_created_generator() {
        let code = Arc::new(prism_compiler::bytecode::CodeObject::new("g", "<test>"));
        let generator = Box::new(GeneratorObject::new(code));
        let ptr = Box::into_raw(generator);
        let value = Value::object_ptr(ptr as *const ());

        let result = generator_close(&[value]).expect("close should succeed");
        assert!(result.is_none());
        assert!(unsafe { &*ptr }.is_exhausted());

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}
