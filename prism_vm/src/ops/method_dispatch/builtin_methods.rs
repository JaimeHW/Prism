//! Static builtin method objects for optimized method dispatch.
//!
//! These builtins are resolved directly by `LoadMethod` and invoked through the
//! existing `CallMethod` fast path, which keeps common container method calls on
//! the same optimized dispatch path as other builtins.

use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, EXCEPTION_TYPE_ID, ExceptionTypeObject, ExceptionValue,
    builtin_hash, builtin_mapping_proxy_contains_key, builtin_mapping_proxy_entries_static,
    builtin_mapping_proxy_get_item_static, builtin_mapping_proxy_len,
    builtin_not_implemented_value, get_exception_type, get_iterator_mut, iterator_to_value,
    value_to_iterator,
};
use crate::error::RuntimeError;
use crate::error::RuntimeErrorKind;
use crate::ops::calls::invoke_callable_value;
use crate::ops::exception::helpers::{
    extract_type_id_from_value, is_exception_class_value, is_exception_instance_value,
};
use crate::ops::iteration::{IterStep, ensure_iterator_value, next_step};
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::ops::objects::{
    alloc_heap_value, delete_attribute_value, dict_storage_mut_from_ptr, dict_storage_ref_from_ptr,
    get_attribute_value, list_storage_mut_from_ptr, list_storage_ref_from_ptr,
    object_getattribute_default, set_attribute_value, tuple_storage_ref_from_ptr,
};
use crate::stdlib::collections::deque::DequeObject;
use crate::stdlib::exceptions::ExceptionTypeId;
use crate::stdlib::generators::{CloseResult, GeneratorObject, prepare_close};
use crate::vm::GeneratorResumeOutcome;
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::{
    ClassMethodDescriptor, PropertyDescriptor, StaticMethodDescriptor,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{DictViewKind, DictViewObject, MappingProxyObject};
use prism_runtime::types::bytes::{BytesObject, value_as_bytes_ref};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::iter::IteratorObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::memoryview::{
    MemoryViewFormat, MemoryViewObject, value_as_memoryview_mut, value_as_memoryview_ref,
};
use prism_runtime::types::set::SetObject;
use prism_runtime::types::simd::search::{
    bytes_count as simd_bytes_count, bytes_find as simd_bytes_find,
};
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::string::{StringObject, StringValueRef, value_as_string_ref};
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::sync::{Arc, LazyLock, Mutex};
use unicode_xid::UnicodeXID;

use super::method_cache::CachedMethod;

mod text;

use text::char_index_to_byte_offset;
pub use text::resolve_str_method;
#[cfg(test)]
use text::*;

static LIST_ITER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.__iter__"), list_iter));
static LIST_LEN_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.__len__"), list_len));
static LIST_GETITEM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.__getitem__"), list_getitem));
static LIST_APPEND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.append"), list_append));
static LIST_EXTEND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("list.extend"), list_extend_with_vm));
static LIST_INSERT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.insert"), list_insert));
static LIST_REMOVE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.remove"), list_remove));
static LIST_POP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.pop"), list_pop));
static LIST_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.copy"), list_copy));
static LIST_CLEAR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.clear"), list_clear));
static LIST_COUNT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.count"), list_count));
static LIST_INDEX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.index"), list_index));
static LIST_REVERSE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.reverse"), list_reverse));
static LIST_SORT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm_kw(Arc::from("list.sort"), list_sort_with_vm));
static TUPLE_ITER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("tuple.__iter__"), tuple_iter));
static TUPLE_LEN_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("tuple.__len__"), tuple_len));
static TUPLE_GETITEM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("tuple.__getitem__"), tuple_getitem));
static TUPLE_COUNT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("tuple.count"), tuple_count));
static TUPLE_INDEX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("tuple.index"), tuple_index));
static DEQUE_APPEND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.append"), deque_append));
static DEQUE_APPENDLEFT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.appendleft"), deque_appendleft));
static DEQUE_POP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.pop"), deque_pop));
static DEQUE_POPLEFT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.popleft"), deque_popleft));
static DEQUE_REMOVE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.remove"), deque_remove));
static DEQUE_GETITEM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("deque.__getitem__"), deque_getitem));
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
static REGEX_PATTERN_FINDALL_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Pattern.findall"),
        crate::stdlib::re::builtin_pattern_findall,
    )
});
static REGEX_PATTERN_FINDITER_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Pattern.finditer"),
        crate::stdlib::re::builtin_pattern_finditer,
    )
});
static REGEX_PATTERN_SUB_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Pattern.sub"),
        crate::stdlib::re::builtin_pattern_sub,
    )
});
static REGEX_PATTERN_SUBN_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Pattern.subn"),
        crate::stdlib::re::builtin_pattern_subn,
    )
});
static REGEX_PATTERN_SPLIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Pattern.split"),
        crate::stdlib::re::builtin_pattern_split,
    )
});
static REGEX_MATCH_GROUP_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Match.group"),
        crate::stdlib::re::builtin_match_group,
    )
});
static REGEX_MATCH_GETITEM_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Match.__getitem__"),
        crate::stdlib::re::builtin_match_getitem,
    )
});
static REGEX_MATCH_GROUPS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Match.groups"),
        crate::stdlib::re::builtin_match_groups,
    )
});
static REGEX_MATCH_GROUPDICT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("Match.groupdict"),
        crate::stdlib::re::builtin_match_groupdict,
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
static DICT_POPITEM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dict.popitem"), dict_popitem));
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
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("object.__ne__"), object_ne));
static OBJECT_GETATTRIBUTE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("object.__getattribute__"), object_getattribute)
});
static OBJECT_SETATTR_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("object.__setattr__"), object_setattr)
});
static OBJECT_DELATTR_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("object.__delattr__"), object_delattr)
});
static OBJECT_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("object.__repr__"), value_repr));
static OBJECT_STR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("object.__str__"), value_str));
static OBJECT_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("object.__format__"), value_format));
static OBJECT_REDUCE_EX_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("object.__reduce_ex__"), value_reduce_ex)
});
static TYPE_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("type.__repr__"), type_repr));
static INT_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.__repr__"), value_repr));
static INT_STR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.__str__"), value_str));
static INT_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.__format__"), value_format));
static INT_ADD_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.__add__"), int_add));
static INT_INDEX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.__index__"), int_index));
static INT_BIT_LENGTH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.bit_length"), int_bit_length));
static INT_BIT_COUNT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("int.bit_count"), int_bit_count));
static INT_TO_BYTES_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("int.to_bytes"),
        crate::builtins::builtin_int_to_bytes,
    )
});
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
static BYTES_DECODE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.decode"), bytes_decode));
static BYTES_STARTSWITH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.startswith"), bytes_startswith));
static BYTES_ENDSWITH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.endswith"), bytes_endswith));
static BYTES_UPPER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.upper"), bytes_upper));
static BYTES_LOWER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.lower"), bytes_lower));
static BYTES_STRIP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.strip"), bytes_strip));
static BYTES_LSTRIP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.lstrip"), bytes_lstrip));
static BYTES_RSTRIP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.rstrip"), bytes_rstrip));
static BYTES_TRANSLATE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.translate"), bytes_translate));
static BYTES_JOIN_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("bytes.join"), bytes_join_with_vm));
static BYTES_FIND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.find"), bytes_find_method));
static BYTES_RFIND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.rfind"), bytes_rfind_method));
static BYTES_INDEX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.index"), bytes_index_method));
static BYTES_RINDEX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.rindex"), bytes_rindex_method));
static BYTES_COUNT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.count"), bytes_count_method));
static SET_ADD_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.add"), set_add));
static SET_REMOVE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.remove"), set_remove));
static SET_DISCARD_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.discard"), set_discard));
static SET_POP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.pop"), set_pop));
static SET_CLEAR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.clear"), set_clear));
static SET_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("set.update"), set_update_with_vm));
static SET_DIFFERENCE_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("set.difference_update"),
        set_difference_update_with_vm,
    )
});
static SET_INTERSECTION_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("set.intersection_update"),
        set_intersection_update_with_vm,
    )
});
static SET_SYMMETRIC_DIFFERENCE_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm(
            Arc::from("set.symmetric_difference_update"),
            set_symmetric_difference_update_with_vm,
        )
    });
static SET_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.copy"), set_copy));
static SET_UNION_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("set.union"), set_union_with_vm));
static SET_INTERSECTION_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.intersection"), set_intersection_with_vm)
});
static SET_DIFFERENCE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.difference"), set_difference_with_vm)
});
static SET_SYMMETRIC_DIFFERENCE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("set.symmetric_difference"),
        set_symmetric_difference_with_vm,
    )
});
static SET_ISDISJOINT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.isdisjoint"), set_isdisjoint_with_vm)
});
static SET_ISSUBSET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.issubset"), set_issubset_with_vm)
});
static SET_ISSUPERSET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("set.issuperset"), set_issuperset_with_vm)
});
static SET_CONTAINS_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("set.__contains__"), set_contains));
static FROZENSET_CONTAINS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("frozenset.__contains__"), frozenset_contains)
});
static FROZENSET_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("frozenset.copy"), frozenset_copy));
static FROZENSET_UNION_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("frozenset.union"), frozenset_union_with_vm)
});
static FROZENSET_INTERSECTION_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("frozenset.intersection"),
        frozenset_intersection_with_vm,
    )
});
static FROZENSET_DIFFERENCE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("frozenset.difference"),
        frozenset_difference_with_vm,
    )
});
static FROZENSET_SYMMETRIC_DIFFERENCE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm(
            Arc::from("frozenset.symmetric_difference"),
            frozenset_symmetric_difference_with_vm,
        )
    });
static FROZENSET_ISDISJOINT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("frozenset.isdisjoint"),
        frozenset_isdisjoint_with_vm,
    )
});
static FROZENSET_ISSUBSET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("frozenset.issubset"), frozenset_issubset_with_vm)
});
static FROZENSET_ISSUPERSET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("frozenset.issuperset"),
        frozenset_issuperset_with_vm,
    )
});
static BYTEARRAY_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.copy"), bytearray_copy));
static BYTEARRAY_EXTEND_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("bytearray.extend"), bytearray_extend_with_vm)
});
static BYTEARRAY_DECODE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.decode"), bytearray_decode));
static BYTEARRAY_STARTSWITH_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("bytearray.startswith"), bytearray_startswith)
});
static BYTEARRAY_ENDSWITH_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("bytearray.endswith"), bytearray_endswith)
});
static BYTEARRAY_UPPER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.upper"), bytearray_upper));
static BYTEARRAY_LOWER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.lower"), bytearray_lower));
static BYTEARRAY_STRIP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.strip"), bytearray_strip));
static BYTEARRAY_LSTRIP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.lstrip"), bytearray_lstrip));
static BYTEARRAY_RSTRIP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.rstrip"), bytearray_rstrip));
static BYTEARRAY_TRANSLATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("bytearray.translate"), bytearray_translate)
});
static BYTEARRAY_JOIN_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("bytearray.join"), bytearray_join_with_vm)
});
static BYTEARRAY_FIND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.find"), bytearray_find));
static BYTEARRAY_RFIND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.rfind"), bytearray_rfind));
static BYTEARRAY_INDEX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.index"), bytearray_index));
static BYTEARRAY_RINDEX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.rindex"), bytearray_rindex));
static BYTEARRAY_COUNT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytearray.count"), bytearray_count));
static MEMORYVIEW_TOBYTES_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("memoryview.tobytes"), memoryview_tobytes)
});
static MEMORYVIEW_TOLIST_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("memoryview.tolist"), memoryview_tolist));
static MEMORYVIEW_CAST_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("memoryview.cast"), memoryview_cast));
static MEMORYVIEW_RELEASE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("memoryview.release"), memoryview_release)
});
static MEMORYVIEW_ENTER_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("memoryview.__enter__"), memoryview_enter)
});
static MEMORYVIEW_EXIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("memoryview.__exit__"), memoryview_exit));
static ITERATOR_ITER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("iterator.__iter__"), iterator_iter));
static ITERATOR_NEXT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("iterator.__next__"), iterator_next));
static ITERATOR_LENGTH_HINT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("iterator.__length_hint__"), iterator_length_hint)
});
static GENERATOR_CLOSE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("generator.close"), generator_close));
static GENERATOR_THROW_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("generator.throw"), generator_throw));
static FUNCTION_GET_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("function.__get__"), function_get));
static CLASSMETHOD_GET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("classmethod.__get__"), classmethod_get)
});
static STATICMETHOD_GET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("staticmethod.__get__"), staticmethod_get)
});
static PROPERTY_GET_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("property.__get__"), property_get));
static PROPERTY_SET_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("property.__set__"), property_set));
static PROPERTY_DELETE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("property.__delete__"), property_delete)
});
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
        "__iter__" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_ITER_METHOD,
        ))),
        "__len__" => Some(CachedMethod::simple(builtin_method_value(&LIST_LEN_METHOD))),
        "__getitem__" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_GETITEM_METHOD,
        ))),
        "append" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_APPEND_METHOD,
        ))),
        "extend" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_EXTEND_METHOD,
        ))),
        "insert" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_INSERT_METHOD,
        ))),
        "remove" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_REMOVE_METHOD,
        ))),
        "pop" => Some(CachedMethod::simple(builtin_method_value(&LIST_POP_METHOD))),
        "copy" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_COPY_METHOD,
        ))),
        "clear" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_CLEAR_METHOD,
        ))),
        "count" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_COUNT_METHOD,
        ))),
        "index" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_INDEX_METHOD,
        ))),
        "reverse" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_REVERSE_METHOD,
        ))),
        "sort" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_SORT_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin tuple methods backed by native tuple storage.
pub fn resolve_tuple_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__iter__" => Some(CachedMethod::simple(builtin_method_value(
            &TUPLE_ITER_METHOD,
        ))),
        "__len__" => Some(CachedMethod::simple(builtin_method_value(
            &TUPLE_LEN_METHOD,
        ))),
        "__getitem__" => Some(CachedMethod::simple(builtin_method_value(
            &TUPLE_GETITEM_METHOD,
        ))),
        "count" => Some(CachedMethod::simple(builtin_method_value(
            &TUPLE_COUNT_METHOD,
        ))),
        "index" => Some(CachedMethod::simple(builtin_method_value(
            &TUPLE_INDEX_METHOD,
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
        "remove" => Some(CachedMethod::simple(builtin_method_value(
            &DEQUE_REMOVE_METHOD,
        ))),
        "__getitem__" => Some(CachedMethod::simple(builtin_method_value(
            &DEQUE_GETITEM_METHOD,
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
        "findall" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_PATTERN_FINDALL_METHOD,
        ))),
        "finditer" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_PATTERN_FINDITER_METHOD,
        ))),
        "sub" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_PATTERN_SUB_METHOD,
        ))),
        "subn" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_PATTERN_SUBN_METHOD,
        ))),
        "split" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_PATTERN_SPLIT_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin regex match methods backed by static builtin function objects.
pub fn resolve_regex_match_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__getitem__" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_MATCH_GETITEM_METHOD,
        ))),
        "group" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_MATCH_GROUP_METHOD,
        ))),
        "groups" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_MATCH_GROUPS_METHOD,
        ))),
        "groupdict" => Some(CachedMethod::simple(builtin_method_value(
            &REGEX_MATCH_GROUPDICT_METHOD,
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
        "popitem" => Some(CachedMethod::simple(builtin_method_value(
            &DICT_POPITEM_METHOD,
        ))),
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
        "__getattribute__" => Some(CachedMethod::simple(builtin_method_value(
            &OBJECT_GETATTRIBUTE_METHOD,
        ))),
        "__setattr__" => Some(CachedMethod::simple(builtin_method_value(
            &OBJECT_SETATTR_METHOD,
        ))),
        "__delattr__" => Some(CachedMethod::simple(builtin_method_value(
            &OBJECT_DELATTR_METHOD,
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
        "__add__" => Some(CachedMethod::simple(builtin_method_value(&INT_ADD_METHOD))),
        "__index__" => Some(CachedMethod::simple(builtin_method_value(
            &INT_INDEX_METHOD,
        ))),
        "bit_length" => Some(CachedMethod::simple(builtin_method_value(
            &INT_BIT_LENGTH_METHOD,
        ))),
        "bit_count" => Some(CachedMethod::simple(builtin_method_value(
            &INT_BIT_COUNT_METHOD,
        ))),
        "to_bytes" => Some(CachedMethod::simple(builtin_method_value(
            &INT_TO_BYTES_METHOD,
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
        _ => resolve_int_method(name),
    }
}

pub fn resolve_exception_method(name: &str) -> Option<CachedMethod> {
    crate::builtins::exception_method_value(name).map(CachedMethod::simple)
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
    if name == "__reduce_ex__" {
        return Some(CachedMethod::simple(builtin_method_value(
            &OBJECT_REDUCE_EX_METHOD,
        )));
    }

    let (name, function): (&'static str, fn(&[Value]) -> Result<Value, BuiltinError>) = match name {
        "__repr__" => ("__repr__", value_repr),
        "__str__" => ("__str__", value_str),
        "__format__" => ("__format__", value_format),
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

/// Resolve builtin bytes methods backed by static builtin function objects.
pub fn resolve_bytes_method(name: &str) -> Option<CachedMethod> {
    match name {
        "decode" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_DECODE_METHOD,
        ))),
        "startswith" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_STARTSWITH_METHOD,
        ))),
        "endswith" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_ENDSWITH_METHOD,
        ))),
        "upper" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_UPPER_METHOD,
        ))),
        "lower" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_LOWER_METHOD,
        ))),
        "strip" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_STRIP_METHOD,
        ))),
        "lstrip" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_LSTRIP_METHOD,
        ))),
        "rstrip" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_RSTRIP_METHOD,
        ))),
        "translate" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_TRANSLATE_METHOD,
        ))),
        "join" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_JOIN_METHOD,
        ))),
        "find" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_FIND_METHOD,
        ))),
        "rfind" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_RFIND_METHOD,
        ))),
        "index" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_INDEX_METHOD,
        ))),
        "rindex" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_RINDEX_METHOD,
        ))),
        "count" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_COUNT_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin set and frozenset methods backed by static builtin function objects.
pub fn resolve_set_method(type_id: TypeId, name: &str) -> Option<CachedMethod> {
    match (type_id, name) {
        (TypeId::SET, "add") => Some(CachedMethod::simple(builtin_method_value(&SET_ADD_METHOD))),
        (TypeId::SET, "remove") => Some(CachedMethod::simple(builtin_method_value(
            &SET_REMOVE_METHOD,
        ))),
        (TypeId::SET, "discard") => Some(CachedMethod::simple(builtin_method_value(
            &SET_DISCARD_METHOD,
        ))),
        (TypeId::SET, "pop") => Some(CachedMethod::simple(builtin_method_value(&SET_POP_METHOD))),
        (TypeId::SET, "clear") => Some(CachedMethod::simple(builtin_method_value(
            &SET_CLEAR_METHOD,
        ))),
        (TypeId::SET, "update") => Some(CachedMethod::simple(builtin_method_value(
            &SET_UPDATE_METHOD,
        ))),
        (TypeId::SET, "difference_update") => Some(CachedMethod::simple(builtin_method_value(
            &SET_DIFFERENCE_UPDATE_METHOD,
        ))),
        (TypeId::SET, "intersection_update") => Some(CachedMethod::simple(builtin_method_value(
            &SET_INTERSECTION_UPDATE_METHOD,
        ))),
        (TypeId::SET, "symmetric_difference_update") => Some(CachedMethod::simple(
            builtin_method_value(&SET_SYMMETRIC_DIFFERENCE_UPDATE_METHOD),
        )),
        (TypeId::SET, "copy") => Some(CachedMethod::simple(builtin_method_value(&SET_COPY_METHOD))),
        (TypeId::SET, "union") => Some(CachedMethod::simple(builtin_method_value(
            &SET_UNION_METHOD,
        ))),
        (TypeId::SET, "intersection") => Some(CachedMethod::simple(builtin_method_value(
            &SET_INTERSECTION_METHOD,
        ))),
        (TypeId::SET, "difference") => Some(CachedMethod::simple(builtin_method_value(
            &SET_DIFFERENCE_METHOD,
        ))),
        (TypeId::SET, "symmetric_difference") => Some(CachedMethod::simple(builtin_method_value(
            &SET_SYMMETRIC_DIFFERENCE_METHOD,
        ))),
        (TypeId::SET, "isdisjoint") => Some(CachedMethod::simple(builtin_method_value(
            &SET_ISDISJOINT_METHOD,
        ))),
        (TypeId::SET, "issubset") => Some(CachedMethod::simple(builtin_method_value(
            &SET_ISSUBSET_METHOD,
        ))),
        (TypeId::SET, "issuperset") => Some(CachedMethod::simple(builtin_method_value(
            &SET_ISSUPERSET_METHOD,
        ))),
        (TypeId::SET, "__contains__") => Some(CachedMethod::simple(builtin_method_value(
            &SET_CONTAINS_METHOD,
        ))),
        (TypeId::FROZENSET, "union") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_UNION_METHOD,
        ))),
        (TypeId::FROZENSET, "intersection") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_INTERSECTION_METHOD,
        ))),
        (TypeId::FROZENSET, "difference") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_DIFFERENCE_METHOD,
        ))),
        (TypeId::FROZENSET, "symmetric_difference") => Some(CachedMethod::simple(
            builtin_method_value(&FROZENSET_SYMMETRIC_DIFFERENCE_METHOD),
        )),
        (TypeId::FROZENSET, "isdisjoint") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_ISDISJOINT_METHOD,
        ))),
        (TypeId::FROZENSET, "issubset") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_ISSUBSET_METHOD,
        ))),
        (TypeId::FROZENSET, "issuperset") => Some(CachedMethod::simple(builtin_method_value(
            &FROZENSET_ISSUPERSET_METHOD,
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
        "extend" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_EXTEND_METHOD,
        ))),
        "decode" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_DECODE_METHOD,
        ))),
        "startswith" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_STARTSWITH_METHOD,
        ))),
        "endswith" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_ENDSWITH_METHOD,
        ))),
        "upper" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_UPPER_METHOD,
        ))),
        "lower" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_LOWER_METHOD,
        ))),
        "strip" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_STRIP_METHOD,
        ))),
        "lstrip" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_LSTRIP_METHOD,
        ))),
        "rstrip" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_RSTRIP_METHOD,
        ))),
        "translate" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_TRANSLATE_METHOD,
        ))),
        "join" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_JOIN_METHOD,
        ))),
        "find" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_FIND_METHOD,
        ))),
        "rfind" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_RFIND_METHOD,
        ))),
        "index" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_INDEX_METHOD,
        ))),
        "rindex" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_RINDEX_METHOD,
        ))),
        "count" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_COUNT_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin iterator methods backed by static builtin function objects.
pub fn resolve_iterator_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__iter__" => Some(CachedMethod::simple(builtin_method_value(
            &ITERATOR_ITER_METHOD,
        ))),
        "__next__" => Some(CachedMethod::simple(builtin_method_value(
            &ITERATOR_NEXT_METHOD,
        ))),
        "__length_hint__" => Some(CachedMethod::simple(builtin_method_value(
            &ITERATOR_LENGTH_HINT_METHOD,
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
        "throw" => Some(CachedMethod::simple(builtin_method_value(
            &GENERATOR_THROW_METHOD,
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

pub fn resolve_classmethod_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__get__" => Some(CachedMethod::simple(builtin_method_value(
            &CLASSMETHOD_GET_METHOD,
        ))),
        _ => None,
    }
}

pub fn resolve_staticmethod_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__get__" => Some(CachedMethod::simple(builtin_method_value(
            &STATICMETHOD_GET_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin memoryview methods backed by static builtin function objects.
pub fn resolve_memoryview_method(name: &str) -> Option<CachedMethod> {
    match name {
        "tobytes" => Some(CachedMethod::simple(builtin_method_value(
            &MEMORYVIEW_TOBYTES_METHOD,
        ))),
        "tolist" => Some(CachedMethod::simple(builtin_method_value(
            &MEMORYVIEW_TOLIST_METHOD,
        ))),
        "cast" => Some(CachedMethod::simple(builtin_method_value(
            &MEMORYVIEW_CAST_METHOD,
        ))),
        "release" => Some(CachedMethod::simple(builtin_method_value(
            &MEMORYVIEW_RELEASE_METHOD,
        ))),
        "__enter__" => Some(CachedMethod::simple(builtin_method_value(
            &MEMORYVIEW_ENTER_METHOD,
        ))),
        "__exit__" => Some(CachedMethod::simple(builtin_method_value(
            &MEMORYVIEW_EXIT_METHOD,
        ))),
        _ => None,
    }
}

/// Resolve builtin property methods backed by static builtin function objects.
pub fn resolve_property_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__get__" => Some(CachedMethod::simple(builtin_method_value(
            &PROPERTY_GET_METHOD,
        ))),
        "__set__" => Some(CachedMethod::simple(builtin_method_value(
            &PROPERTY_SET_METHOD,
        ))),
        "__delete__" => Some(CachedMethod::simple(builtin_method_value(
            &PROPERTY_DELETE_METHOD,
        ))),
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
fn ensure_hashable(value: Value) -> Result<(), BuiltinError> {
    builtin_hash(&[value]).map(|_| ())
}

#[inline]
fn list_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "__iter__", args, 0)?;
    let list = expect_list_ref(args[0], "__iter__")?;
    Ok(iterator_to_value(IteratorObject::from_values(
        list.as_slice().to_vec(),
    )))
}

#[inline]
fn list_len(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "__len__", args, 0)?;
    let list = expect_list_ref(args[0], "__len__")?;
    Ok(Value::int(i64::try_from(list.len()).unwrap_or(i64::MAX))
        .expect("list length should fit in tagged int"))
}

#[inline]
fn list_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "__getitem__", args, 1)?;
    let list = expect_list_ref(args[0], "__getitem__")?;

    if let Some(index) = args[1]
        .as_object_ptr()
        .and_then(slice_object_from_value_ptr)
    {
        let indices = index.indices(list.len());
        let mut values = Vec::with_capacity(indices.length);
        for index in indices.iter() {
            values.push(list.as_slice()[index]);
        }
        return Ok(to_object_value(ListObject::from_iter(values)));
    }

    let index = expect_integer_like_index(args[1])?;
    list.get(index)
        .ok_or_else(|| BuiltinError::IndexError("list index out of range".to_string()))
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
fn list_insert(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "insert", args, 2)?;
    let index = expect_integer_like_index(args[1])?;
    let list = expect_list_mut(args[0], "insert")?;
    list.insert(index, args[2]);
    Ok(Value::none())
}

#[inline]
fn list_remove(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "remove", args, 1)?;

    let remove_index = {
        let list = expect_list_ref(args[0], "remove")?;
        list.as_slice()
            .iter()
            .copied()
            .position(|item| crate::ops::comparison::values_equal(item, args[1]))
    };

    let Some(index) = remove_index else {
        return Err(BuiltinError::ValueError(
            "list.remove(x): x not in list".to_string(),
        ));
    };

    let list = expect_list_mut(args[0], "remove")?;
    list.remove(i64::try_from(index).expect("list.remove index should fit in i64"))
        .expect("validated list.remove index should be in bounds");
    Ok(Value::none())
}

#[inline]
fn list_pop(args: &[Value]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 1 {
        return Err(BuiltinError::TypeError(format!(
            "list.pop() takes at most 1 argument ({given} given)"
        )));
    }

    let list = expect_list_mut(args[0], "pop")?;
    if list.is_empty() {
        return Err(BuiltinError::IndexError("pop from empty list".to_string()));
    }

    if given == 0 {
        return list
            .pop()
            .ok_or_else(|| BuiltinError::IndexError("pop from empty list".to_string()));
    }

    let index = expect_integer_like_index(args[1])?;
    list.remove(index)
        .ok_or_else(|| BuiltinError::IndexError("pop index out of range".to_string()))
}

#[inline]
fn list_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "copy", args, 0)?;
    let list = expect_list_ref(args[0], "copy")?;
    Ok(to_object_value(ListObject::from_slice(list.as_slice())))
}

#[inline]
fn list_clear(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "clear", args, 0)?;
    let list = expect_list_mut(args[0], "clear")?;
    list.clear();
    Ok(Value::none())
}

#[inline]
fn list_count(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "count", args, 1)?;
    let list = expect_list_ref(args[0], "count")?;
    let count = list
        .as_slice()
        .iter()
        .copied()
        .filter(|item| crate::ops::comparison::values_equal(*item, args[1]))
        .count();
    Ok(Value::int(i64::try_from(count).unwrap_or(i64::MAX)).expect("count should fit"))
}

#[inline]
fn list_index(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_range("list", "index", args, 1, 3)?;
    let list = expect_list_ref(args[0], "index")?;
    let len = i64::try_from(list.len()).unwrap_or(i64::MAX);
    let start = if args.len() >= 3 {
        normalize_search_bound(expect_integer_like_index(args[2])?, len)
    } else {
        0
    };
    let stop = if args.len() >= 4 {
        normalize_search_bound(expect_integer_like_index(args[3])?, len)
    } else {
        len
    };

    for index in start..stop {
        let item = list
            .get(index)
            .expect("normalized list.index bounds should stay in range");
        if crate::ops::comparison::values_equal(item, args[1]) {
            return Ok(Value::int(index).expect("list.index result should fit"));
        }
    }

    Err(BuiltinError::ValueError(
        "list.index(x): x not in list".to_string(),
    ))
}

#[inline]
fn tuple_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("tuple", "__iter__", args, 0)?;
    let tuple = expect_tuple_ref(args[0], "__iter__")?;
    Ok(iterator_to_value(IteratorObject::from_values(
        tuple.as_slice().to_vec(),
    )))
}

#[inline]
fn tuple_len(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("tuple", "__len__", args, 0)?;
    let tuple = expect_tuple_ref(args[0], "__len__")?;
    Ok(Value::int(i64::try_from(tuple.len()).unwrap_or(i64::MAX))
        .expect("tuple length should fit in tagged int"))
}

#[inline]
fn tuple_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("tuple", "__getitem__", args, 1)?;
    let tuple = expect_tuple_ref(args[0], "__getitem__")?;

    if let Some(index) = args[1]
        .as_object_ptr()
        .and_then(slice_object_from_value_ptr)
    {
        let indices = index.indices(tuple.len());
        let mut values = Vec::with_capacity(indices.length);
        for index in indices.iter() {
            values.push(tuple.as_slice()[index]);
        }
        return Ok(to_object_value(TupleObject::from_vec(values)));
    }

    let index = expect_integer_like_index(args[1])?;
    tuple
        .get(index)
        .ok_or_else(|| BuiltinError::IndexError("tuple index out of range".to_string()))
}

#[inline]
fn tuple_count(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("tuple", "count", args, 1)?;
    let tuple = expect_tuple_ref(args[0], "count")?;
    let count = tuple
        .as_slice()
        .iter()
        .copied()
        .filter(|item| crate::ops::comparison::values_equal(*item, args[1]))
        .count();
    Ok(Value::int(i64::try_from(count).unwrap_or(i64::MAX)).expect("count should fit"))
}

#[inline]
fn tuple_index(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_range("tuple", "index", args, 1, 3)?;
    let tuple = expect_tuple_ref(args[0], "index")?;
    let len = i64::try_from(tuple.len()).unwrap_or(i64::MAX);
    let start = if args.len() >= 3 {
        normalize_search_bound(expect_integer_like_index(args[2])?, len)
    } else {
        0
    };
    let stop = if args.len() >= 4 {
        normalize_search_bound(expect_integer_like_index(args[3])?, len)
    } else {
        len
    };

    for index in start..stop {
        let item = tuple.as_slice()[index as usize];
        if crate::ops::comparison::values_equal(item, args[1]) {
            return Ok(Value::int(index).expect("tuple.index result should fit"));
        }
    }

    Err(BuiltinError::ValueError(
        "tuple.index(x): x not in tuple".to_string(),
    ))
}

#[inline]
fn slice_object_from_value_ptr(ptr: *const ()) -> Option<&'static SliceObject> {
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::SLICE).then(|| unsafe { &*(ptr as *const SliceObject) })
}

#[inline]
fn int_receiver_magnitude(value: Value, method_name: &'static str) -> Result<u64, BuiltinError> {
    if let Some(int_value) = value.as_int() {
        return Ok(int_value.unsigned_abs());
    }
    if let Some(bool_value) = value.as_bool() {
        return Ok(u64::from(bool_value));
    }

    Err(BuiltinError::TypeError(format!(
        "descriptor 'int.{method_name}' requires an 'int' object but received '{}'",
        value.type_name()
    )))
}

#[inline]
fn int_operand_bigint(value: Value) -> Option<num_bigint::BigInt> {
    if let Some(bool_value) = value.as_bool() {
        return Some(num_bigint::BigInt::from(i64::from(bool_value)));
    }

    value_to_bigint(value)
}

#[inline]
fn int_receiver_bigint(
    value: Value,
    method_name: &'static str,
) -> Result<num_bigint::BigInt, BuiltinError> {
    int_operand_bigint(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor 'int.{method_name}' requires an 'int' object but received '{}'",
            value.type_name()
        ))
    })
}

#[inline]
fn int_add(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("int", "__add__", args, 1)?;
    let left = int_receiver_bigint(args[0], "__add__")?;
    let Some(right) = int_operand_bigint(args[1]) else {
        return Ok(builtin_not_implemented_value());
    };

    Ok(bigint_to_value(left + right))
}

#[inline]
fn int_index(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("int", "__index__", args, 0)?;
    if let Some(value) = args[0].as_int() {
        return Value::int(value)
            .ok_or_else(|| BuiltinError::OverflowError("int.__index__ result overflowed".into()));
    }
    if let Some(value) = args[0].as_bool() {
        return Ok(Value::int(i64::from(value)).expect("bool index result should fit"));
    }

    Err(BuiltinError::TypeError(format!(
        "descriptor 'int.__index__' requires an 'int' object but received '{}'",
        args[0].type_name()
    )))
}

#[inline]
fn int_bit_length(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("int", "bit_length", args, 0)?;
    let magnitude = int_receiver_magnitude(args[0], "bit_length")?;
    let bits = if magnitude == 0 {
        0
    } else {
        u64::BITS - magnitude.leading_zeros()
    };
    Ok(Value::int(i64::from(bits)).expect("bit length should fit"))
}

#[inline]
fn int_bit_count(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("int", "bit_count", args, 0)?;
    let count = int_receiver_magnitude(args[0], "bit_count")?.count_ones();
    Ok(Value::int(i64::from(count)).expect("bit count should fit"))
}

#[inline]
fn normalize_search_bound(index: i64, len: i64) -> i64 {
    let normalized = if index < 0 {
        len.saturating_add(index)
    } else {
        index
    };
    normalized.clamp(0, len)
}

#[inline]
fn list_reverse(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "reverse", args, 0)?;
    let list = expect_list_mut(args[0], "reverse")?;
    list.reverse();
    Ok(Value::none())
}

#[inline]
fn list_sort_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "sort", args, 0)?;
    let (key_func, reverse) = parse_list_sort_keywords(vm, keywords)?;
    let values = {
        let list = expect_list_ref(args[0], "sort")?;
        list.as_slice().to_vec()
    };

    let mut decorated = Vec::with_capacity(values.len());
    for value in values {
        let sort_key = match key_func {
            Some(callable) => invoke_callable_value(vm, callable, &[value])
                .map_err(runtime_error_to_builtin_error)?,
            None => value,
        };
        decorated.push((sort_key, value));
    }

    let mut compare_error = None;
    decorated.sort_by(|(left_key, _), (right_key, _)| {
        if compare_error.is_some() {
            return Ordering::Equal;
        }

        match compare_sort_keys(vm, *left_key, *right_key) {
            Ok(ordering) => ordering,
            Err(err) => {
                compare_error = Some(err);
                Ordering::Equal
            }
        }
    });
    if let Some(err) = compare_error {
        return Err(err);
    }

    if reverse {
        decorated.reverse();
    }

    let list = expect_list_mut(args[0], "sort")?;
    list.clear();
    list.extend(decorated.into_iter().map(|(_, value)| value));
    Ok(Value::none())
}

#[inline]
fn parse_list_sort_keywords(
    vm: &mut VirtualMachine,
    keywords: &[(&str, Value)],
) -> Result<(Option<Value>, bool), BuiltinError> {
    let mut key = None;
    let mut key_seen = false;
    let mut reverse = false;
    let mut reverse_seen = false;

    for (name, value) in keywords {
        match *name {
            "key" => {
                if key_seen {
                    return Err(BuiltinError::TypeError(
                        "list.sort() got multiple values for keyword argument 'key'".to_string(),
                    ));
                }
                if !value.is_none() {
                    key = Some(*value);
                }
                key_seen = true;
            }
            "reverse" => {
                if reverse_seen {
                    return Err(BuiltinError::TypeError(
                        "list.sort() got multiple values for keyword argument 'reverse'"
                            .to_string(),
                    ));
                }
                reverse = crate::truthiness::try_is_truthy(vm, *value)
                    .map_err(runtime_error_to_builtin_error)?;
                reverse_seen = true;
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "list.sort() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    Ok((key, reverse))
}

#[inline]
fn compare_sort_keys(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
) -> Result<Ordering, BuiltinError> {
    if let Some(ordering) = primitive_sort_ordering(left, right) {
        return Ok(ordering);
    }

    let left_lt = rich_lt(vm, left, right)?;
    if left_lt == Some(true) {
        return Ok(Ordering::Less);
    }

    let right_lt = rich_lt(vm, right, left)?;
    if right_lt == Some(true) {
        return Ok(Ordering::Greater);
    }

    if left_lt.is_none() && right_lt.is_none() {
        return Err(BuiltinError::TypeError(format!(
            "'<' not supported between instances of '{}' and '{}'",
            left.type_name(),
            right.type_name()
        )));
    }

    Ok(Ordering::Equal)
}

#[inline]
fn primitive_sort_ordering(left: Value, right: Value) -> Option<Ordering> {
    if left == right {
        return Some(Ordering::Equal);
    }

    let left_numeric = numeric_sort_key(left);
    let right_numeric = numeric_sort_key(right);
    if let (Some(left), Some(right)) = (left_numeric, right_numeric) {
        return left.partial_cmp(&right);
    }

    match (value_as_string_ref(left), value_as_string_ref(right)) {
        (Some(left), Some(right)) => Some(left.as_str().cmp(right.as_str())),
        _ => None,
    }
}

#[inline]
fn numeric_sort_key(value: Value) -> Option<f64> {
    if let Some(boolean) = value.as_bool() {
        return Some(if boolean { 1.0 } else { 0.0 });
    }
    if let Some(integer) = value.as_int() {
        return Some(integer as f64);
    }
    value.as_float()
}

#[inline]
fn rich_lt(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
) -> Result<Option<bool>, BuiltinError> {
    match resolve_special_method(left, "__lt__") {
        Ok(target) => {
            let result = invoke_comparison_method(vm, target, right)?;
            if result == builtin_not_implemented_value() {
                return Ok(None);
            }
            crate::truthiness::try_is_truthy(vm, result)
                .map(Some)
                .map_err(runtime_error_to_builtin_error)
        }
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => Ok(None),
        Err(err) => Err(runtime_error_to_builtin_error(err)),
    }
}

#[inline]
fn invoke_comparison_method(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
    operand: Value,
) -> Result<Value, BuiltinError> {
    invoke_bound_method_with_arg(vm, &target, operand)
}

fn invoke_bound_method_with_arg(
    vm: &mut VirtualMachine,
    target: &BoundMethodTarget,
    arg: Value,
) -> Result<Value, BuiltinError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self, arg])
            .map_err(runtime_error_to_builtin_error),
        None => invoke_callable_value(vm, target.callable, &[arg])
            .map_err(runtime_error_to_builtin_error),
    }
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
fn deque_remove(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("deque", "remove", args, 1)?;
    let deque = expect_deque_mut(args[0], "remove")?;
    if deque.deque_mut().remove(&args[1]) {
        Ok(Value::none())
    } else {
        Err(BuiltinError::ValueError(
            "deque.remove(x): x not in deque".to_string(),
        ))
    }
}

#[inline]
fn deque_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("deque", "__getitem__", args, 1)?;
    let deque = expect_deque_ref(args[0], "__getitem__")?;
    let Some(raw_index) = args[1].as_int() else {
        return Err(BuiltinError::TypeError(
            "deque indices must be integers".to_string(),
        ));
    };
    let Some(index) = isize::try_from(raw_index).ok() else {
        return Err(BuiltinError::IndexError(format!(
            "index {raw_index} out of range for length {}",
            deque.len()
        )));
    };
    deque
        .deque()
        .get(index)
        .copied()
        .ok_or_else(|| BuiltinError::IndexError(format!("deque index out of range: {raw_index}")))
}

#[inline]
fn dict_keys(args: &[Value]) -> Result<Value, BuiltinError> {
    dict_view(args, "keys", DictViewKind::Keys)
}

#[inline]
fn dict_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "__getitem__", args, 1)?;
    let dict = expect_dict_ref(args[0], "__getitem__")?;
    ensure_hashable(args[1])?;
    dict.get(args[1])
        .ok_or_else(|| BuiltinError::KeyError("key not found".to_string()))
}

#[inline]
fn dict_contains(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "__contains__", args, 1)?;
    let dict = expect_dict_ref(args[0], "__contains__")?;
    ensure_hashable(args[1])?;
    Ok(Value::bool(dict.get(args[1]).is_some()))
}

#[inline]
fn dict_setitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "__setitem__", args, 2)?;
    let dict = expect_dict_mut(args[0], "__setitem__")?;
    ensure_hashable(args[1])?;
    dict.set(args[1], args[2]);
    Ok(Value::none())
}

#[inline]
fn dict_delitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "__delitem__", args, 1)?;
    let dict = expect_dict_mut(args[0], "__delitem__")?;
    ensure_hashable(args[1])?;
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
    ensure_hashable(args[1])?;
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
    ensure_hashable(args[1])?;
    if let Some(value) = dict.remove(args[1]) {
        return Ok(value);
    }

    if let Some(default) = args.get(2).copied() {
        return Ok(default);
    }

    Err(BuiltinError::KeyError("key not found".to_string()))
}

#[inline]
fn dict_popitem(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("dict", "popitem", args, 0)?;
    let dict = expect_dict_mut(args[0], "popitem")?;
    let (key, value) = dict
        .popitem()
        .ok_or_else(|| BuiltinError::KeyError("popitem(): dictionary is empty".to_string()))?;
    Ok(to_object_value(TupleObject::from_slice(&[key, value])))
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
    ensure_hashable(args[1])?;
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
        ensure_hashable(key)?;
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
fn object_ne(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("object", "__ne__", args, 1)?;
    match resolve_special_method(args[0], "__eq__") {
        Ok(eq) => {
            let result = invoke_comparison_method(vm, eq, args[1])?;
            if result != builtin_not_implemented_value() {
                return crate::truthiness::try_is_truthy(vm, result)
                    .map(|truthy| Value::bool(!truthy))
                    .map_err(runtime_error_to_builtin_error);
            }
        }
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {}
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    }

    if let Some(equal) = crate::ops::comparison::builtin_eq_fallback(args[0], args[1]) {
        return Ok(Value::bool(!equal));
    }

    Ok(builtin_not_implemented_value())
}

fn object_getattribute(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("object", "__getattribute__", args, 1)?;
    let Some(name) = value_as_string_ref(args[1]) else {
        return Err(BuiltinError::TypeError(
            "attribute name must be string".to_string(),
        ));
    };

    object_getattribute_default(vm, args[0], &intern(name.as_str()))
        .map_err(runtime_error_to_builtin_error)
}

fn object_setattr(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("object", "__setattr__", args, 2)?;
    let Some(name) = value_as_string_ref(args[1]) else {
        return Err(BuiltinError::TypeError(
            "attribute name must be string".to_string(),
        ));
    };

    set_attribute_value(vm, args[0], &intern(name.as_str()), args[2])
        .map_err(runtime_error_to_builtin_error)?;
    Ok(Value::none())
}

fn object_delattr(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("object", "__delattr__", args, 1)?;
    let Some(name) = value_as_string_ref(args[1]) else {
        return Err(BuiltinError::TypeError(
            "attribute name must be string".to_string(),
        ));
    };

    delete_attribute_value(vm, args[0], &intern(name.as_str()))
        .map_err(runtime_error_to_builtin_error)?;
    Ok(Value::none())
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

fn value_reduce_ex(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        let given = args.len().saturating_sub(1);
        return Err(BuiltinError::TypeError(format!(
            "__reduce_ex__() takes exactly 1 argument ({given} given)"
        )));
    }

    let reduce_name = intern("__reduce__");
    let reduce = match get_attribute_value(vm, args[0], &reduce_name) {
        Ok(reduce) => reduce,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return Err(BuiltinError::NotImplemented(
                "pickle protocol support is not implemented yet".to_string(),
            ));
        }
        Err(err) => return Err(crate::builtins::runtime_error_to_builtin_error(err)),
    };

    invoke_callable_value(vm, reduce, &[]).map_err(crate::builtins::runtime_error_to_builtin_error)
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
fn bytes_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    decode_bytes_method(args, "bytes", expect_bytes_ref)
}

#[inline]
fn bytearray_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    decode_bytes_method(args, "bytearray", expect_bytearray_ref)
}

#[inline]
fn bytes_startswith(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_affix_match(
        args,
        "bytes",
        "startswith",
        expect_bytes_ref,
        |value, affix| value.starts_with(affix),
    )
}

#[inline]
fn bytes_endswith(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_affix_match(
        args,
        "bytes",
        "endswith",
        expect_bytes_ref,
        |value, affix| value.ends_with(affix),
    )
}

#[inline]
fn bytes_upper(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_case_transform(
        args,
        "bytes",
        "upper",
        expect_bytes_ref,
        TypeId::BYTES,
        u8::to_ascii_uppercase,
    )
}

#[inline]
fn bytes_lower(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_case_transform(
        args,
        "bytes",
        "lower",
        expect_bytes_ref,
        TypeId::BYTES,
        u8::to_ascii_lowercase,
    )
}

#[inline]
fn bytes_strip(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_strip(
        args,
        "bytes",
        "strip",
        expect_bytes_ref,
        TypeId::BYTES,
        StripDirection::Both,
    )
}

#[inline]
fn bytes_lstrip(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_strip(
        args,
        "bytes",
        "lstrip",
        expect_bytes_ref,
        TypeId::BYTES,
        StripDirection::Leading,
    )
}

#[inline]
fn bytes_rstrip(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_strip(
        args,
        "bytes",
        "rstrip",
        expect_bytes_ref,
        TypeId::BYTES,
        StripDirection::Trailing,
    )
}

#[inline]
fn bytes_translate(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_translate(args, "bytes", expect_bytes_ref, TypeId::BYTES)
}

#[inline]
fn bytearray_startswith(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_affix_match(
        args,
        "bytearray",
        "startswith",
        expect_bytearray_ref,
        |value, affix| value.starts_with(affix),
    )
}

#[inline]
fn bytearray_endswith(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_affix_match(
        args,
        "bytearray",
        "endswith",
        expect_bytearray_ref,
        |value, affix| value.ends_with(affix),
    )
}

#[inline]
fn bytearray_upper(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_case_transform(
        args,
        "bytearray",
        "upper",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        u8::to_ascii_uppercase,
    )
}

#[inline]
fn bytearray_lower(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_case_transform(
        args,
        "bytearray",
        "lower",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        u8::to_ascii_lowercase,
    )
}

#[inline]
fn bytearray_strip(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_strip(
        args,
        "bytearray",
        "strip",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        StripDirection::Both,
    )
}

#[inline]
fn bytearray_lstrip(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_strip(
        args,
        "bytearray",
        "lstrip",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        StripDirection::Leading,
    )
}

#[inline]
fn bytearray_rstrip(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_strip(
        args,
        "bytearray",
        "rstrip",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        StripDirection::Trailing,
    )
}

#[inline]
fn bytearray_translate(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_translate(args, "bytearray", expect_bytearray_ref, TypeId::BYTEARRAY)
}

#[inline]
fn bytes_join_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_join_with_vm(vm, args, "bytes", expect_bytes_ref, TypeId::BYTES)
}

#[inline]
fn bytearray_join_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_join_with_vm(
        vm,
        args,
        "bytearray",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
    )
}

#[inline]
fn bytes_find_method(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_find_like(
        args,
        "bytes",
        "find",
        expect_bytes_ref,
        SearchDirection::Forward,
        MissingNeedle::ReturnMinusOne,
    )
}

#[inline]
fn bytes_rfind_method(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_find_like(
        args,
        "bytes",
        "rfind",
        expect_bytes_ref,
        SearchDirection::Reverse,
        MissingNeedle::ReturnMinusOne,
    )
}

#[inline]
fn bytes_index_method(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_find_like(
        args,
        "bytes",
        "index",
        expect_bytes_ref,
        SearchDirection::Forward,
        MissingNeedle::RaiseValueError,
    )
}

#[inline]
fn bytes_rindex_method(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_find_like(
        args,
        "bytes",
        "rindex",
        expect_bytes_ref,
        SearchDirection::Reverse,
        MissingNeedle::RaiseValueError,
    )
}

#[inline]
fn bytes_count_method(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_count(args, "bytes", expect_bytes_ref)
}

#[inline]
fn bytearray_find(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_find_like(
        args,
        "bytearray",
        "find",
        expect_bytearray_ref,
        SearchDirection::Forward,
        MissingNeedle::ReturnMinusOne,
    )
}

#[inline]
fn bytearray_rfind(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_find_like(
        args,
        "bytearray",
        "rfind",
        expect_bytearray_ref,
        SearchDirection::Reverse,
        MissingNeedle::ReturnMinusOne,
    )
}

#[inline]
fn bytearray_index(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_find_like(
        args,
        "bytearray",
        "index",
        expect_bytearray_ref,
        SearchDirection::Forward,
        MissingNeedle::RaiseValueError,
    )
}

#[inline]
fn bytearray_rindex(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_find_like(
        args,
        "bytearray",
        "rindex",
        expect_bytearray_ref,
        SearchDirection::Reverse,
        MissingNeedle::RaiseValueError,
    )
}

#[inline]
fn bytearray_count(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_count(args, "bytearray", expect_bytearray_ref)
}

#[inline]
fn decode_bytes_method(
    args: &[Value],
    receiver_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 2 {
        return Err(BuiltinError::TypeError(format!(
            "decode() takes at most 2 arguments ({given} given)"
        )));
    }

    let encoding = args
        .get(1)
        .copied()
        .map(|value| expect_method_string_arg(value, receiver_name, "decode", 1))
        .transpose()?;
    let errors = args
        .get(2)
        .copied()
        .map(|value| expect_method_string_arg(value, receiver_name, "decode", 2))
        .transpose()?;

    let bytes = receiver(args[0], "decode")?;
    crate::builtins::decode_bytes_to_value(bytes.as_bytes(), encoding.as_deref(), errors.as_deref())
}

#[inline]
fn byte_sequence_case_transform(
    args: &[Value],
    receiver_name: &'static str,
    method_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
    result_type: TypeId,
    map_byte: fn(&u8) -> u8,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 0)?;
    let bytes = receiver(args[0], method_name)?;
    let converted = bytes.as_bytes().iter().map(map_byte).collect();
    Ok(to_object_value(BytesObject::from_vec_with_type(
        converted,
        result_type,
    )))
}

fn byte_sequence_join_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
    receiver_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
    result_type: TypeId,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, "join", args, 1)?;
    let separator = receiver(args[0], "join")?.as_bytes();
    let values = collect_iterable_values_with_vm(vm, args[1])?;

    let mut parts = Vec::with_capacity(values.len());
    let mut total_len = 0usize;
    for (index, value) in values.into_iter().enumerate() {
        let part = bytes_like_join_part(value).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "sequence item {index}: expected a bytes-like object, {} found",
                value.type_name()
            ))
        })?;
        total_len = total_len
            .checked_add(part.len())
            .ok_or_else(|| BuiltinError::OverflowError("joined bytes are too long".to_string()))?;
        parts.push(part);
    }

    if parts.len() > 1 {
        let separators_len = separator
            .len()
            .checked_mul(parts.len() - 1)
            .ok_or_else(|| BuiltinError::OverflowError("joined bytes are too long".to_string()))?;
        total_len = total_len
            .checked_add(separators_len)
            .ok_or_else(|| BuiltinError::OverflowError("joined bytes are too long".to_string()))?;
    }

    let mut joined = Vec::with_capacity(total_len);
    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            joined.extend_from_slice(separator);
        }
        joined.extend_from_slice(part);
    }

    Ok(to_object_value(BytesObject::from_vec_with_type(
        joined,
        result_type,
    )))
}

fn bytes_like_join_part(value: Value) -> Option<Vec<u8>> {
    if let Some(bytes) = value_as_bytes_ref(value) {
        return Some(bytes.to_vec());
    }

    let view = value_as_memoryview_ref(value)?;
    if view.released() {
        return None;
    }
    Some(view.to_vec())
}

#[inline]
fn byte_sequence_find_like(
    args: &[Value],
    receiver_name: &'static str,
    method_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
    direction: SearchDirection,
    missing: MissingNeedle,
) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "{receiver_name}.{method_name}() takes from 1 to 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let bytes = receiver(args[0], method_name)?;
    let needle = byte_sequence_search_needle(args[1])?;
    let Some((start, end)) =
        normalize_byte_search_bounds(args, bytes.len(), receiver_name, method_name)?
    else {
        return byte_sequence_missing_result(missing);
    };

    let haystack = &bytes.as_bytes()[start..end];
    let offset = match direction {
        SearchDirection::Forward => simd_bytes_find(haystack, &needle),
        SearchDirection::Reverse => byte_sequence_rfind(haystack, &needle),
    };

    match offset {
        Some(index) => Ok(Value::int((start + index) as i64).expect("byte index should fit int")),
        None => byte_sequence_missing_result(missing),
    }
}

#[inline]
fn byte_sequence_count(
    args: &[Value],
    receiver_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "{receiver_name}.count() takes from 1 to 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let bytes = receiver(args[0], "count")?;
    let needle = byte_sequence_search_needle(args[1])?;
    let Some((start, end)) =
        normalize_byte_search_bounds(args, bytes.len(), receiver_name, "count")?
    else {
        return Ok(Value::int(0).unwrap());
    };

    let count = simd_bytes_count(&bytes.as_bytes()[start..end], &needle);
    Ok(Value::int(count as i64).expect("byte count should fit int"))
}

#[inline]
fn byte_sequence_search_needle(value: Value) -> Result<Vec<u8>, BuiltinError> {
    if value.as_int().is_some() || value.as_bool().is_some() {
        let byte = expect_integer_like_index(value)?;
        return u8::try_from(byte)
            .map(|byte| vec![byte])
            .map_err(|_| BuiltinError::ValueError("byte must be in range(0, 256)".to_string()));
    }

    bytes_like_argument_bytes(value).map_err(|err| match err {
        BuiltinError::TypeError(_) => BuiltinError::TypeError(format!(
            "argument should be integer or bytes-like object, not '{}'",
            value.type_name()
        )),
        other => other,
    })
}

#[inline]
fn normalize_byte_search_bounds(
    args: &[Value],
    len: usize,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Option<(usize, usize)>, BuiltinError> {
    let start = clamp_slice_index(
        parse_slice_bound(args.get(2).copied(), 0, receiver_name, method_name)?,
        len,
    );
    let end = clamp_slice_index(
        parse_slice_bound(
            args.get(3).copied(),
            len as isize,
            receiver_name,
            method_name,
        )?,
        len,
    );

    Ok((start <= end).then_some((start, end)))
}

#[inline]
fn byte_sequence_missing_result(missing: MissingNeedle) -> Result<Value, BuiltinError> {
    match missing {
        MissingNeedle::ReturnMinusOne => Ok(Value::int(-1).unwrap()),
        MissingNeedle::RaiseValueError => {
            Err(BuiltinError::ValueError("subsection not found".to_string()))
        }
    }
}

#[inline]
fn byte_sequence_rfind(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(haystack.len());
    }
    if needle.len() > haystack.len() {
        return None;
    }
    if needle.len() == 1 {
        return haystack.iter().rposition(|&byte| byte == needle[0]);
    }

    haystack
        .windows(needle.len())
        .rposition(|candidate| candidate == needle)
}

#[inline]
fn byte_sequence_strip(
    args: &[Value],
    receiver_name: &'static str,
    method_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
    result_type: TypeId,
    direction: StripDirection,
) -> Result<Value, BuiltinError> {
    expect_method_arg_range(receiver_name, method_name, args, 0, 1)?;

    let bytes = receiver(args[0], method_name)?;
    let strip_set = parse_byte_strip_set(args.get(1).copied())?;
    let data = bytes.as_bytes();
    let (start, end) = strip_byte_bounds(data, &strip_set, direction);

    if result_type == TypeId::BYTES && start == 0 && end == data.len() {
        return Ok(args[0]);
    }

    Ok(to_object_value(BytesObject::from_vec_with_type(
        data[start..end].to_vec(),
        result_type,
    )))
}

#[inline]
fn parse_byte_strip_set(chars: Option<Value>) -> Result<ByteSet, BuiltinError> {
    let Some(chars) = chars else {
        return Ok(ByteSet::ascii_whitespace());
    };

    if chars.is_none() {
        return Ok(ByteSet::ascii_whitespace());
    }

    bytes_like_argument_bytes(chars).map(|bytes| ByteSet::from_bytes(&bytes))
}

#[inline]
fn bytes_like_argument_bytes(value: Value) -> Result<Vec<u8>, BuiltinError> {
    if let Some(bytes) = value_as_bytes_ref(value) {
        return Ok(bytes.as_bytes().to_vec());
    }

    if let Some(view) = value_as_memoryview_ref(value) {
        ensure_memoryview_not_released(view)?;
        return Ok(view.to_vec());
    }

    if let Some(bytes) = crate::stdlib::array::value_as_array_bytes(value)? {
        return Ok(bytes);
    }

    Err(BuiltinError::TypeError(format!(
        "a bytes-like object is required, not '{}'",
        value.type_name()
    )))
}

#[inline]
fn strip_byte_bounds(
    data: &[u8],
    strip_set: &ByteSet,
    direction: StripDirection,
) -> (usize, usize) {
    let mut start = 0usize;
    let mut end = data.len();

    if matches!(direction, StripDirection::Leading | StripDirection::Both) {
        while start < end && strip_set.contains(data[start]) {
            start += 1;
        }
    }

    if matches!(direction, StripDirection::Trailing | StripDirection::Both) {
        while end > start && strip_set.contains(data[end - 1]) {
            end -= 1;
        }
    }

    (start, end)
}

#[inline]
fn byte_sequence_translate(
    args: &[Value],
    receiver_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
    result_type: TypeId,
) -> Result<Value, BuiltinError> {
    expect_method_arg_range(receiver_name, "translate", args, 1, 2)?;

    let bytes = receiver(args[0], "translate")?;
    let table = parse_byte_translate_table(args[1])?;
    let delete = parse_byte_delete_set(args.get(2).copied())?;
    let data = bytes.as_bytes();
    let mut translated = Vec::with_capacity(data.len());
    let mut changed = false;

    for &byte in data {
        if delete.contains(byte) {
            changed = true;
            continue;
        }

        let mapped = table
            .as_ref()
            .map_or(byte, |table| table[usize::from(byte)]);
        changed |= mapped != byte;
        translated.push(mapped);
    }

    if result_type == TypeId::BYTES && !changed {
        return Ok(args[0]);
    }

    Ok(to_object_value(BytesObject::from_vec_with_type(
        translated,
        result_type,
    )))
}

#[inline]
fn parse_byte_translate_table(table: Value) -> Result<Option<[u8; 256]>, BuiltinError> {
    if table.is_none() {
        return Ok(None);
    }

    let table_bytes = bytes_like_argument_bytes(table)?;
    if table_bytes.len() != 256 {
        return Err(BuiltinError::ValueError(
            "translation table must be 256 characters long".to_string(),
        ));
    }

    let mut translation = [0u8; 256];
    translation.copy_from_slice(&table_bytes);
    Ok(Some(translation))
}

#[inline]
fn parse_byte_delete_set(delete: Option<Value>) -> Result<ByteSet, BuiltinError> {
    delete.map_or_else(
        || Ok(ByteSet::empty()),
        |value| bytes_like_argument_bytes(value).map(|bytes| ByteSet::from_bytes(&bytes)),
    )
}

#[inline]
fn byte_sequence_affix_match(
    args: &[Value],
    receiver_name: &'static str,
    method_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
    matcher: impl Fn(&[u8], &[u8]) -> bool,
) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "{receiver_name}.{method_name}() takes from 1 to 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let bytes = receiver(args[0], method_name)?;
    let byte_len = bytes.len();
    let start = clamp_slice_index(
        parse_slice_bound(args.get(2).copied(), 0, receiver_name, method_name)?,
        byte_len,
    );
    let end = clamp_slice_index(
        parse_slice_bound(
            args.get(3).copied(),
            byte_len as isize,
            receiver_name,
            method_name,
        )?,
        byte_len,
    );
    let start = start.min(end);
    let end = end.max(start);
    let slice = &bytes.as_bytes()[start..end];

    Ok(Value::bool(match_byte_sequence_affixes(
        args[1],
        method_name,
        |candidate| matcher(slice, candidate),
    )?))
}

#[inline]
fn match_byte_sequence_affixes(
    affixes: Value,
    method_name: &'static str,
    mut matcher: impl FnMut(&[u8]) -> bool,
) -> Result<bool, BuiltinError> {
    if let Some(candidate) = byte_sequence_object_from_value(affixes) {
        return Ok(matcher(candidate.as_bytes()));
    }

    let Some(ptr) = affixes.as_object_ptr() else {
        return Err(byte_sequence_affix_type_error(
            method_name,
            affixes.type_name(),
        ));
    };
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::TUPLE {
        return Err(byte_sequence_affix_type_error(
            method_name,
            affixes.type_name(),
        ));
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    for index in 0..tuple.len() {
        let candidate = tuple
            .get(index as i64)
            .expect("tuple index should be valid");
        let Some(candidate) = byte_sequence_object_from_value(candidate) else {
            return Err(BuiltinError::TypeError(format!(
                "a bytes-like object is required, not '{}'",
                candidate.type_name()
            )));
        };
        if matcher(candidate.as_bytes()) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[inline]
fn byte_sequence_affix_type_error(method_name: &'static str, actual_type: &str) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "{method_name} first arg must be bytes or a tuple of bytes, not {actual_type}"
    ))
}

#[inline]
fn byte_sequence_object_from_value(value: Value) -> Option<&'static BytesObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::BYTES | TypeId::BYTEARRAY => Some(unsafe { &*(ptr as *const BytesObject) }),
        _ => None,
    }
}

fn normalize_slice_bounds(
    value: &str,
    start: Option<Value>,
    end: Option<Value>,
    method_name: &'static str,
) -> Result<(usize, usize), BuiltinError> {
    let bounds = normalize_slice_bounds_with_chars(value, start, end, method_name)?;
    Ok((bounds.start_byte, bounds.end_byte))
}

#[inline]
fn normalize_slice_bounds_with_chars(
    value: &str,
    start: Option<Value>,
    end: Option<Value>,
    method_name: &'static str,
) -> Result<SliceBounds, BuiltinError> {
    let char_len = value.chars().count();
    let start_char = clamp_slice_index(parse_slice_bound(start, 0, "str", method_name)?, char_len);
    let end_char = clamp_slice_index(
        parse_slice_bound(end, char_len as isize, "str", method_name)?,
        char_len,
    );
    let start_char = start_char.min(end_char);
    let end_char = end_char.max(start_char);
    Ok(SliceBounds {
        start_char,
        end_char,
        start_byte: char_index_to_byte_offset(value, start_char),
        end_byte: char_index_to_byte_offset(value, end_char),
    })
}

#[inline]
fn parse_slice_bound(
    bound: Option<Value>,
    default: isize,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<isize, BuiltinError> {
    let Some(bound) = bound else {
        return Ok(default);
    };

    bound.as_int().map(|value| value as isize).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{receiver_name}.{method_name}() slice indices must be integers, not '{}'",
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

#[derive(Clone, Copy)]
enum SearchDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Copy)]
enum MissingNeedle {
    ReturnMinusOne,
    RaiseValueError,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StringCaseProperty {
    Upper,
    Lower,
}

#[derive(Clone, Copy)]
struct SliceBounds {
    start_char: usize,
    end_char: usize,
    start_byte: usize,
    end_byte: usize,
}

#[inline]
fn expect_str_method_string_arg(
    value: Value,
    method_name: &'static str,
    position: usize,
) -> Result<String, BuiltinError> {
    expect_method_string_arg(value, "str", method_name, position)
}

#[inline]
fn expect_method_string_arg(
    value: Value,
    receiver_name: &'static str,
    method_name: &'static str,
    position: usize,
) -> Result<String, BuiltinError> {
    string_object_from_value(value)
        .map(|string: StringObject| string.as_str().to_string())
        .map_err(|_| {
            BuiltinError::TypeError(format!(
                "{receiver_name}.{method_name}() argument {position} must be str, not {}",
                value.type_name()
            ))
        })
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

    if let Some(entries) = collect_mapping_entries_with_vm(vm, source)? {
        return Ok(entries);
    }

    let items = collect_iterable_values_with_vm(vm, source)?;
    let mut entries = Vec::with_capacity(items.len());
    for item in items {
        entries.push(expect_dict_update_entry(item)?);
    }
    Ok(entries)
}

fn collect_mapping_entries_with_vm(
    vm: &mut VirtualMachine,
    source: Value,
) -> Result<Option<Vec<(Value, Value)>>, BuiltinError> {
    let keys = match resolve_special_method(source, "keys") {
        Ok(target) => {
            let value = match target.implicit_self {
                Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
                None => invoke_callable_value(vm, target.callable, &[]),
            }
            .map_err(runtime_error_to_builtin_error)?;
            collect_iterable_values_with_vm(vm, value)?
        }
        Err(err) if err.is_attribute_error() => return Ok(None),
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };

    let get_item =
        resolve_special_method(source, "__getitem__").map_err(runtime_error_to_builtin_error)?;
    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        entries.push((key, invoke_bound_method_with_arg(vm, &get_item, key)?));
    }
    Ok(Some(entries))
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
    ensure_hashable(args[1])?;
    set.add(args[1]);
    Ok(Value::none())
}

#[inline]
fn set_remove(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "remove", args, 1)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "remove")?;
    ensure_hashable(args[1])?;
    if set.remove(args[1]) {
        Ok(Value::none())
    } else {
        Err(BuiltinError::KeyError(args[1].to_string()))
    }
}

#[inline]
fn set_discard(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "discard", args, 1)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "discard")?;
    ensure_hashable(args[1])?;
    set.discard(args[1]);
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
fn set_clear(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "clear", args, 0)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "clear")?;
    set.clear();
    Ok(Value::none())
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
fn set_result_value(mut set: SetObject, result_type: TypeId) -> Value {
    set.header.type_id = result_type;
    to_object_value(set)
}

#[inline]
fn hashable_iterable_values_with_vm(
    vm: &mut VirtualMachine,
    iterable: Value,
) -> Result<Vec<Value>, BuiltinError> {
    let values = collect_iterable_values_with_vm(vm, iterable)?;
    for value in values.iter().copied() {
        ensure_hashable(value)?;
    }
    Ok(values)
}

#[inline]
fn iterable_to_hashable_set_with_vm(
    vm: &mut VirtualMachine,
    iterable: Value,
) -> Result<SetObject, BuiltinError> {
    Ok(SetObject::from_iter(hashable_iterable_values_with_vm(
        vm, iterable,
    )?))
}

#[inline]
fn set_update_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let set = expect_set_mut_receiver(
        *args
            .first()
            .ok_or_else(|| BuiltinError::TypeError("unbound set.update()".to_string()))?,
        TypeId::SET,
        "update",
    )?;

    for iterable in &args[1..] {
        for value in hashable_iterable_values_with_vm(vm, *iterable)? {
            set.add(value);
        }
    }

    Ok(Value::none())
}

#[inline]
fn set_difference_update_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let set = expect_set_mut_receiver(
        *args.first().ok_or_else(|| {
            BuiltinError::TypeError("unbound set.difference_update()".to_string())
        })?,
        TypeId::SET,
        "difference_update",
    )?;

    for iterable in &args[1..] {
        for value in hashable_iterable_values_with_vm(vm, *iterable)? {
            set.discard(value);
        }
    }

    Ok(Value::none())
}

#[inline]
fn set_intersection_update_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let set = expect_set_mut_receiver(
        *args.first().ok_or_else(|| {
            BuiltinError::TypeError("unbound set.intersection_update()".to_string())
        })?,
        TypeId::SET,
        "intersection_update",
    )?;

    for iterable in &args[1..] {
        let other = iterable_to_hashable_set_with_vm(vm, *iterable)?;
        set.intersection_update(&other);
    }

    Ok(Value::none())
}

#[inline]
fn set_symmetric_difference_update_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("set", "symmetric_difference_update", args, 1)?;
    let set = expect_set_mut_receiver(args[0], TypeId::SET, "symmetric_difference_update")?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    set.symmetric_difference_update(&other);
    Ok(Value::none())
}

#[inline]
fn set_union_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    let receiver = *args.first().ok_or_else(|| {
        BuiltinError::TypeError(format!("unbound {receiver_name}.{method_name}()"))
    })?;
    let mut result = expect_set_receiver(receiver, expected_type, method_name)?.clone();
    for iterable in &args[1..] {
        for value in hashable_iterable_values_with_vm(vm, *iterable)? {
            result.add(value);
        }
    }
    Ok(set_result_value(result, expected_type))
}

#[inline]
fn set_intersection_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    let receiver = *args.first().ok_or_else(|| {
        BuiltinError::TypeError(format!("unbound {receiver_name}.{method_name}()"))
    })?;
    let current = expect_set_receiver(receiver, expected_type, method_name)?;
    if args.len() == 1 {
        return if expected_type == TypeId::FROZENSET {
            Ok(receiver)
        } else {
            Ok(set_result_value(current.clone(), expected_type))
        };
    }

    let mut result = current.clone();
    for iterable in &args[1..] {
        let other = iterable_to_hashable_set_with_vm(vm, *iterable)?;
        result.intersection_update(&other);
    }
    Ok(set_result_value(result, expected_type))
}

#[inline]
fn set_difference_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    let receiver = *args.first().ok_or_else(|| {
        BuiltinError::TypeError(format!("unbound {receiver_name}.{method_name}()"))
    })?;
    let current = expect_set_receiver(receiver, expected_type, method_name)?;
    if args.len() == 1 {
        return if expected_type == TypeId::FROZENSET {
            Ok(receiver)
        } else {
            Ok(set_result_value(current.clone(), expected_type))
        };
    }

    let mut result = current.clone();
    for iterable in &args[1..] {
        let other = iterable_to_hashable_set_with_vm(vm, *iterable)?;
        result.difference_update(&other);
    }
    Ok(set_result_value(result, expected_type))
}

#[inline]
fn set_symmetric_difference_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    Ok(set_result_value(
        set.symmetric_difference(&other),
        expected_type,
    ))
}

#[inline]
fn set_isdisjoint_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    Ok(Value::bool(set.is_disjoint(&other)))
}

#[inline]
fn set_issubset_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    Ok(Value::bool(set.is_subset(&other)))
}

#[inline]
fn set_issuperset_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    expected_type: TypeId,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let set = expect_set_receiver(args[0], expected_type, method_name)?;
    let other = iterable_to_hashable_set_with_vm(vm, args[1])?;
    Ok(Value::bool(set.is_superset(&other)))
}

#[inline]
fn set_union_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    set_union_impl(vm, args, TypeId::SET, "set", "union")
}

#[inline]
fn frozenset_union_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    set_union_impl(vm, args, TypeId::FROZENSET, "frozenset", "union")
}

#[inline]
fn set_intersection_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_intersection_impl(vm, args, TypeId::SET, "set", "intersection")
}

#[inline]
fn frozenset_intersection_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_intersection_impl(vm, args, TypeId::FROZENSET, "frozenset", "intersection")
}

#[inline]
fn set_difference_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    set_difference_impl(vm, args, TypeId::SET, "set", "difference")
}

#[inline]
fn frozenset_difference_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_difference_impl(vm, args, TypeId::FROZENSET, "frozenset", "difference")
}

#[inline]
fn set_symmetric_difference_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_symmetric_difference_impl(vm, args, TypeId::SET, "set", "symmetric_difference")
}

#[inline]
fn frozenset_symmetric_difference_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_symmetric_difference_impl(
        vm,
        args,
        TypeId::FROZENSET,
        "frozenset",
        "symmetric_difference",
    )
}

#[inline]
fn set_isdisjoint_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    set_isdisjoint_impl(vm, args, TypeId::SET, "set", "isdisjoint")
}

#[inline]
fn frozenset_isdisjoint_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_isdisjoint_impl(vm, args, TypeId::FROZENSET, "frozenset", "isdisjoint")
}

#[inline]
fn set_issubset_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    set_issubset_impl(vm, args, TypeId::SET, "set", "issubset")
}

#[inline]
fn frozenset_issubset_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_issubset_impl(vm, args, TypeId::FROZENSET, "frozenset", "issubset")
}

#[inline]
fn set_issuperset_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    set_issuperset_impl(vm, args, TypeId::SET, "set", "issuperset")
}

#[inline]
fn frozenset_issuperset_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    set_issuperset_impl(vm, args, TypeId::FROZENSET, "frozenset", "issuperset")
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

fn bytearray_extend_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("bytearray", "extend", args, 1)?;
    let incoming = collect_bytearray_extend_data(vm, args[1])?;
    let bytearray = expect_bytearray_mut(args[0], "extend")?;
    bytearray.extend_from_slice(&incoming);
    Ok(Value::none())
}

fn memoryview_tobytes(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("memoryview", "tobytes", args, 0)?;
    let view = expect_memoryview_ref(args[0], "tobytes")?;
    ensure_memoryview_not_released(view)?;
    Ok(to_object_value(BytesObject::from_vec(view.to_vec())))
}

fn memoryview_tolist(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("memoryview", "tolist", args, 0)?;
    let view = expect_memoryview_ref(args[0], "tolist")?;
    ensure_memoryview_not_released(view)?;
    let values = view.to_values().ok_or_else(|| {
        BuiltinError::TypeError("memoryview item format is not supported".to_string())
    })?;
    Ok(to_object_value(ListObject::from_iter(values)))
}

fn memoryview_cast(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_range("memoryview", "cast", args, 1, 2)?;
    let view = expect_memoryview_ref(args[0], "cast")?;
    ensure_memoryview_not_released(view)?;
    let format = value_as_string_ref(args[1])
        .and_then(|text| MemoryViewFormat::parse(text.as_str()))
        .ok_or_else(|| {
            BuiltinError::ValueError(
                "memoryview: destination format must be a native single character format"
                    .to_string(),
            )
        })?;

    let shape = if args.len() == 3 && !args[2].is_none() {
        Some(parse_memoryview_cast_shape(
            args[2],
            view.nbytes(),
            format.item_size(),
        )?)
    } else {
        None
    };

    let casted = view.cast_with_shape(format, shape).ok_or_else(|| {
        BuiltinError::TypeError("memoryview: length is not a multiple of itemsize".to_string())
    })?;
    Ok(to_object_value(casted))
}

fn memoryview_release(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("memoryview", "release", args, 0)?;
    let view = expect_memoryview_mut(args[0], "release")?;
    view.release();
    Ok(Value::none())
}

fn memoryview_enter(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("memoryview", "__enter__", args, 0)?;
    let view = expect_memoryview_ref(args[0], "__enter__")?;
    ensure_memoryview_not_released(view)?;
    Ok(args[0])
}

fn memoryview_exit(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "memoryview.__exit__() takes exactly 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let view = expect_memoryview_mut(args[0], "__exit__")?;
    view.release();
    Ok(Value::none())
}

fn iterator_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("iterator", "__iter__", args, 0)?;
    Ok(args[0])
}

fn iterator_next(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("iterator", "__next__", args, 0)?;
    let iter = get_iterator_mut(&args[0]).ok_or_else(|| {
        BuiltinError::TypeError("'iterator' object is not an iterator".to_string())
    })?;
    iter.next().ok_or(BuiltinError::StopIteration)
}

fn iterator_length_hint(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("iterator", "__length_hint__", args, 0)?;
    let iter = get_iterator_mut(&args[0]).ok_or_else(|| {
        BuiltinError::TypeError("'iterator' object is not an iterator".to_string())
    })?;
    Ok(Value::int(iter.size_hint().unwrap_or(0) as i64)
        .expect("iterator length hint should fit in tagged int"))
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
    ensure_hashable(args[1])?;
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
fn expect_method_arg_range(
    receiver_name: &'static str,
    method_name: &'static str,
    args: &[Value],
    min_count: usize,
    max_count: usize,
) -> Result<(), BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given < min_count {
        let noun = if min_count == 1 {
            "argument"
        } else {
            "arguments"
        };
        return Err(BuiltinError::TypeError(format!(
            "{receiver_name}.{method_name}() takes at least {min_count} {noun} ({given} given)"
        )));
    }
    if given > max_count {
        let noun = if max_count == 1 {
            "argument"
        } else {
            "arguments"
        };
        return Err(BuiltinError::TypeError(format!(
            "{receiver_name}.{method_name}() takes at most {max_count} {noun} ({given} given)"
        )));
    }
    Ok(())
}

#[inline]
fn normalize_generator_throw_arguments(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<(Value, u16), BuiltinError> {
    let typ = args[0];
    let value = args.get(1).copied();
    let traceback = args.get(2).copied();

    let exception = if is_exception_instance_value(&typ) {
        if value.is_some_and(|separate| !separate.is_none()) {
            return Err(BuiltinError::TypeError(
                "instance exception may not have a separate value".to_string(),
            ));
        }
        typ
    } else if let Some(exception_type) = exception_type_from_value(typ) {
        match value {
            Some(arg) if is_exception_instance_value(&arg) => arg,
            Some(arg) if !arg.is_none() => {
                let constructor_args = [arg];
                exception_type.construct(&constructor_args)
            }
            _ => exception_type.construct(&[]),
        }
    } else if is_exception_class_value(&typ) {
        match value {
            Some(arg) if is_exception_instance_value(&arg) => arg,
            Some(arg) if !arg.is_none() => {
                invoke_callable_value(vm, typ, &[arg]).map_err(BuiltinError::Raised)?
            }
            _ => invoke_callable_value(vm, typ, &[]).map_err(BuiltinError::Raised)?,
        }
    } else {
        return Err(BuiltinError::TypeError(format!(
            "exceptions must be classes or instances deriving from BaseException, not {}",
            typ.type_name()
        )));
    };

    if let Some(traceback) = traceback {
        attach_generator_throw_traceback(vm, exception, traceback)?;
    }

    if !is_exception_instance_value(&exception) {
        return Err(BuiltinError::TypeError(
            "exceptions must derive from BaseException".to_string(),
        ));
    }
    let type_id = extract_type_id_from_value(&exception);
    Ok((exception, type_id))
}

fn attach_generator_throw_traceback(
    vm: &mut VirtualMachine,
    exception: Value,
    traceback: Value,
) -> Result<(), BuiltinError> {
    if let Some(exception_value) = unsafe { ExceptionValue::from_value_mut(exception) } {
        exception_value.replace_traceback(traceback).map_err(|_| {
            BuiltinError::TypeError("throw() third argument must be a traceback object".to_string())
        })?;
        return Ok(());
    }

    let traceback = normalize_generator_throw_traceback(traceback)?;
    set_attribute_value(vm, exception, &intern("__traceback__"), traceback)
        .map_err(BuiltinError::Raised)
}

fn normalize_generator_throw_traceback(traceback: Value) -> Result<Value, BuiltinError> {
    if traceback.is_none() {
        return Ok(traceback);
    }

    let Some(ptr) = traceback.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "throw() third argument must be a traceback object".to_string(),
        ));
    };
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::TRACEBACK {
        return Err(BuiltinError::TypeError(
            "throw() third argument must be a traceback object".to_string(),
        ));
    }
    Ok(traceback)
}

#[inline]
fn exception_type_from_value(value: Value) -> Option<&'static ExceptionTypeObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != EXCEPTION_TYPE_ID {
        return None;
    }
    Some(unsafe { &*(ptr as *const ExceptionTypeObject) })
}

#[inline]
fn exception_runtime_error_message(value: Value) -> Arc<str> {
    unsafe { ExceptionValue::from_value(value) }
        .map(|exception| {
            let message = exception.display_text();
            if message.is_empty() {
                Arc::<str>::from(exception.repr_text())
            } else {
                Arc::<str>::from(message)
            }
        })
        .unwrap_or_else(|| Arc::<str>::from("Uncaught exception"))
}

#[inline]
fn stop_iteration_runtime_error(value: Value) -> RuntimeError {
    let stop_iteration_type = get_exception_type("StopIteration")
        .expect("StopIteration exception type must exist in the builtin registry");
    let exception = if value.is_none() {
        stop_iteration_type.construct(&[])
    } else {
        let constructor_args = [value];
        stop_iteration_type.construct(&constructor_args)
    };
    RuntimeError::raised_exception(
        ExceptionTypeId::StopIteration.as_u8() as u16,
        exception,
        exception_runtime_error_message(exception),
    )
}

#[inline]
fn collect_iterable_values(iterable: Value) -> Result<Vec<Value>, BuiltinError> {
    if let Some(iter) = get_iterator_mut(&iterable) {
        return Ok(iter.collect_remaining());
    }

    let mut iter = value_to_iterator(&iterable).map_err(BuiltinError::from)?;
    Ok(iter.collect_remaining())
}

fn collect_bytearray_extend_data(
    vm: &mut VirtualMachine,
    iterable: Value,
) -> Result<Vec<u8>, BuiltinError> {
    if let Some(bytes) = value_as_bytes_ref(iterable) {
        return Ok(bytes.to_vec());
    }

    collect_iterable_values_with_vm(vm, iterable)?
        .into_iter()
        .map(bytearray_extend_byte)
        .collect()
}

#[inline]
fn bytearray_extend_byte(value: Value) -> Result<u8, BuiltinError> {
    let byte = expect_integer_like_index(value)?;
    u8::try_from(byte)
        .map_err(|_| BuiltinError::ValueError("byte must be in range(0, 256)".to_string()))
}

#[derive(Copy, Clone)]
enum StripDirection {
    Leading,
    Trailing,
    Both,
}

#[derive(Clone)]
struct ByteSet {
    mask: [u64; 4],
}

impl ByteSet {
    #[inline]
    fn empty() -> Self {
        Self { mask: [0u64; 4] }
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut mask = [0u64; 4];
        for &byte in bytes {
            let slot = usize::from(byte / 64);
            let bit = u32::from(byte % 64);
            mask[slot] |= 1u64 << bit;
        }
        Self { mask }
    }

    #[inline]
    fn ascii_whitespace() -> Self {
        Self::from_bytes(b" \t\n\r\x0b\x0c")
    }

    #[inline]
    fn contains(&self, byte: u8) -> bool {
        let slot = usize::from(byte / 64);
        let bit = u32::from(byte % 64);
        (self.mask[slot] & (1u64 << bit)) != 0
    }
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
    match err.into_kind() {
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
    let Some(list) = list_storage_mut_from_ptr(ptr) else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'list.{method_name}' requires a 'list' object but received '{}'",
            header.type_id.name()
        )));
    };

    Ok(list)
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
    let Some(list) = list_storage_ref_from_ptr(ptr) else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'list.{method_name}' requires a 'list' object but received '{}'",
            header.type_id.name()
        )));
    };

    Ok(list)
}

#[inline]
fn expect_tuple_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static TupleObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'tuple.{method_name}' requires a 'tuple' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    let Some(tuple) = tuple_storage_ref_from_ptr(ptr) else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'tuple.{method_name}' requires a 'tuple' object but received '{}'",
            header.type_id.name()
        )));
    };

    Ok(tuple)
}

#[inline]
fn expect_integer_like_index(value: Value) -> Result<i64, BuiltinError> {
    if let Some(index) = value.as_int() {
        return Ok(index);
    }

    if let Some(boolean) = value.as_bool() {
        return Ok(if boolean { 1 } else { 0 });
    }

    Err(BuiltinError::TypeError(format!(
        "'{}' object cannot be interpreted as an integer",
        value.type_name()
    )))
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
fn expect_deque_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static DequeObject, BuiltinError> {
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

    Ok(unsafe { &*(ptr as *const DequeObject) })
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
fn expect_bytearray_mut(
    value: Value,
    method_name: &'static str,
) -> Result<&'static mut BytesObject, BuiltinError> {
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

    Ok(unsafe { &mut *(ptr as *mut BytesObject) })
}

#[inline]
fn expect_memoryview_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static MemoryViewObject, BuiltinError> {
    value_as_memoryview_ref(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor 'memoryview.{method_name}' requires a 'memoryview' object but received '{}'",
            value.type_name()
        ))
    })
}

#[inline]
fn expect_memoryview_mut(
    value: Value,
    method_name: &'static str,
) -> Result<&'static mut MemoryViewObject, BuiltinError> {
    value_as_memoryview_mut(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor 'memoryview.{method_name}' requires a 'memoryview' object but received '{}'",
            value.type_name()
        ))
    })
}

#[inline]
fn ensure_memoryview_not_released(view: &MemoryViewObject) -> Result<(), BuiltinError> {
    if view.released() {
        Err(BuiltinError::ValueError(
            "operation forbidden on released memoryview object".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn parse_memoryview_cast_shape(
    shape_value: Value,
    nbytes: usize,
    item_size: usize,
) -> Result<Vec<usize>, BuiltinError> {
    let dims = if let Some(dim) = prism_runtime::types::int::value_to_i64(shape_value) {
        vec![dim]
    } else if let Some(ptr) = shape_value.as_object_ptr() {
        if let Some(tuple) = tuple_storage_ref_from_ptr(ptr) {
            tuple
                .as_slice()
                .iter()
                .copied()
                .map(memoryview_shape_dim)
                .collect::<Result<Vec<_>, _>>()?
        } else if let Some(list) = list_storage_ref_from_ptr(ptr) {
            list.as_slice()
                .iter()
                .copied()
                .map(memoryview_shape_dim)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            return Err(BuiltinError::TypeError(
                "shape must be an integer or a tuple/list of integers".to_string(),
            ));
        }
    } else {
        return Err(BuiltinError::TypeError(
            "shape must be an integer or a tuple/list of integers".to_string(),
        ));
    };

    let mut elements = 1usize;
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        if dim < 0 {
            return Err(BuiltinError::ValueError(
                "memoryview.cast(): elements of shape must be integers > 0".to_string(),
            ));
        }
        let dim = dim as usize;
        elements = elements
            .checked_mul(dim)
            .ok_or_else(|| BuiltinError::OverflowError("memoryview shape too large".to_string()))?;
        shape.push(dim);
    }
    let expected = elements
        .checked_mul(item_size)
        .ok_or_else(|| BuiltinError::OverflowError("memoryview shape too large".to_string()))?;
    if expected != nbytes {
        return Err(BuiltinError::TypeError(
            "memoryview: product(shape) * itemsize != buffer size".to_string(),
        ));
    }
    Ok(shape)
}

#[inline]
fn memoryview_shape_dim(value: Value) -> Result<i64, BuiltinError> {
    prism_runtime::types::int::value_to_i64(value).ok_or_else(|| {
        BuiltinError::TypeError("memoryview.cast(): elements of shape must be integers".to_string())
    })
}

#[inline]
fn expect_bytes_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static BytesObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'bytes.{method_name}' requires a 'bytes' object but received '{}'",
            value.type_name()
        )));
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::BYTES {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'bytes.{method_name}' requires a 'bytes' object but received '{}'",
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
fn generator_throw(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_range("generator", "throw", args, 1, 3)?;
    let (exception, type_id) = normalize_generator_throw_arguments(vm, &args[1..])?;
    let generator = expect_generator_mut(args[0], "throw")?;
    match vm.resume_generator_for_throw(generator, exception, type_id) {
        Ok(GeneratorResumeOutcome::Yielded(value)) => Ok(value),
        Ok(GeneratorResumeOutcome::Returned(value)) => {
            Err(BuiltinError::Raised(stop_iteration_runtime_error(value)))
        }
        Err(err) => Err(BuiltinError::Raised(err)),
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

fn expect_classmethod_receiver(
    value: Value,
    method_name: &'static str,
) -> Result<&'static ClassMethodDescriptor, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'classmethod.{method_name}' requires a 'classmethod' object but received '{}'",
            value.type_name()
        )));
    };
    if unsafe { &*(ptr as *const ObjectHeader) }.type_id != TypeId::CLASSMETHOD {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'classmethod.{method_name}' requires a 'classmethod' object but received '{}'",
            unsafe { &*(ptr as *const ObjectHeader) }.type_id.name()
        )));
    }

    Ok(unsafe { &*(ptr as *const ClassMethodDescriptor) })
}

fn expect_staticmethod_receiver(
    value: Value,
    method_name: &'static str,
) -> Result<&'static StaticMethodDescriptor, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'staticmethod.{method_name}' requires a 'staticmethod' object but received '{}'",
            value.type_name()
        )));
    };
    if unsafe { &*(ptr as *const ObjectHeader) }.type_id != TypeId::STATICMETHOD {
        return Err(BuiltinError::TypeError(format!(
            "descriptor 'staticmethod.{method_name}' requires a 'staticmethod' object but received '{}'",
            unsafe { &*(ptr as *const ObjectHeader) }.type_id.name()
        )));
    }

    Ok(unsafe { &*(ptr as *const StaticMethodDescriptor) })
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

fn classmethod_get(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '__get__' of 'classmethod' object needs 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let descriptor = expect_classmethod_receiver(args[0], "__get__")?;
    let instance = args[1];
    let owner = if args.len() == 3 && !args[2].is_none() {
        args[2]
    } else if !instance.is_none() {
        crate::builtins::builtin_type(&[instance])?
    } else {
        return Err(BuiltinError::TypeError(
            "__get__(None, None) is invalid".to_string(),
        ));
    };

    crate::ops::objects::bind_wrapped_classmethod_value(vm, descriptor.function(), owner)
        .map_err(crate::builtins::runtime_error_to_builtin_error)
}

fn staticmethod_get(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '__get__' of 'staticmethod' object needs 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let descriptor = expect_staticmethod_receiver(args[0], "__get__")?;
    Ok(descriptor.function())
}

fn property_get(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '__get__' of 'property' object needs 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let descriptor = expect_property_receiver(args[0], "__get__")?;
    let instance = args[1];
    if instance.is_none() {
        return Ok(args[0]);
    }

    let getter = descriptor
        .getter()
        .ok_or_else(|| BuiltinError::AttributeError("property has no getter".to_string()))?;
    invoke_callable_value(vm, getter, &[instance])
        .map_err(crate::builtins::runtime_error_to_builtin_error)
}

fn property_set(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '__set__' of 'property' object needs 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let descriptor = expect_property_receiver(args[0], "__set__")?;
    let setter = descriptor
        .setter()
        .ok_or_else(|| BuiltinError::AttributeError("property has no setter".to_string()))?;
    invoke_callable_value(vm, setter, &[args[1], args[2]])
        .map_err(crate::builtins::runtime_error_to_builtin_error)?;
    Ok(Value::none())
}

fn property_delete(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '__delete__' of 'property' object needs 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let descriptor = expect_property_receiver(args[0], "__delete__")?;
    let deleter = descriptor
        .deleter()
        .ok_or_else(|| BuiltinError::AttributeError("property has no deleter".to_string()))?;
    invoke_callable_value(vm, deleter, &[args[1]])
        .map_err(crate::builtins::runtime_error_to_builtin_error)?;
    Ok(Value::none())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::iterator_to_value;
    use crate::error::RuntimeErrorKind;
    use crate::stdlib::exceptions::ExceptionTypeId;
    use prism_code::CodeObject;
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

    fn boxed_tuple_value(tuple: TupleObject) -> (Value, *mut TupleObject) {
        let ptr = Box::into_raw(Box::new(tuple));
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

    fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
        unsafe {
            &*(value
                .as_object_ptr()
                .expect("builtin method should be materialized")
                as *const BuiltinFunctionObject)
        }
    }

    fn property_echo_getter(args: &[Value]) -> Result<Value, BuiltinError> {
        if args.len() != 1 {
            return Err(BuiltinError::TypeError(format!(
                "getter expected 1 argument, got {}",
                args.len()
            )));
        }
        Ok(args[0])
    }

    fn property_accepting_setter(args: &[Value]) -> Result<Value, BuiltinError> {
        if args.len() != 2 {
            return Err(BuiltinError::TypeError(format!(
                "setter expected 2 arguments, got {}",
                args.len()
            )));
        }
        Ok(Value::string(intern("setter return is ignored")))
    }

    fn property_accepting_deleter(args: &[Value]) -> Result<Value, BuiltinError> {
        if args.len() != 1 {
            return Err(BuiltinError::TypeError(format!(
                "deleter expected 1 argument, got {}",
                args.len()
            )));
        }
        Ok(Value::string(intern("deleter return is ignored")))
    }

    fn byte_values(value: Value) -> Vec<u8> {
        let ptr = value
            .as_object_ptr()
            .expect("byte result should be an object");
        let bytes = unsafe { &*(ptr as *const BytesObject) };
        bytes.as_bytes().to_vec()
    }

    fn assert_unicode_encode_error(err: BuiltinError, expected_message: &str) {
        match err {
            BuiltinError::Raised(runtime_err) => match runtime_err.kind() {
                RuntimeErrorKind::Exception { type_id, message } => {
                    assert_eq!(*type_id, ExceptionTypeId::UnicodeEncodeError.as_u8() as u16);
                    assert_eq!(message.as_ref(), expected_message);
                }
                kind => panic!("expected UnicodeEncodeError, got {kind:?}"),
            },
            other => panic!("expected UnicodeEncodeError, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_list_method_returns_builtin_for_sequence_protocol_and_mutators() {
        let iter = resolve_list_method("__iter__").expect("__iter__ should resolve");
        let len = resolve_list_method("__len__").expect("__len__ should resolve");
        let getitem = resolve_list_method("__getitem__").expect("__getitem__ should resolve");
        let append = resolve_list_method("append").expect("append should resolve");
        let extend = resolve_list_method("extend").expect("extend should resolve");
        let insert = resolve_list_method("insert").expect("insert should resolve");
        let remove = resolve_list_method("remove").expect("remove should resolve");
        let pop = resolve_list_method("pop").expect("pop should resolve");
        let copy = resolve_list_method("copy").expect("copy should resolve");
        let clear = resolve_list_method("clear").expect("clear should resolve");
        let reverse = resolve_list_method("reverse").expect("reverse should resolve");
        assert!(iter.method.as_object_ptr().is_some());
        assert!(len.method.as_object_ptr().is_some());
        assert!(getitem.method.as_object_ptr().is_some());
        assert!(append.method.as_object_ptr().is_some());
        assert!(extend.method.as_object_ptr().is_some());
        assert!(insert.method.as_object_ptr().is_some());
        assert!(remove.method.as_object_ptr().is_some());
        assert!(pop.method.as_object_ptr().is_some());
        assert!(copy.method.as_object_ptr().is_some());
        assert!(clear.method.as_object_ptr().is_some());
        assert!(reverse.method.as_object_ptr().is_some());
        assert!(!append.is_descriptor);
        assert!(!extend.is_descriptor);
        assert!(!insert.is_descriptor);
        assert!(!remove.is_descriptor);
        assert!(!pop.is_descriptor);
        assert!(!copy.is_descriptor);
        assert!(!clear.is_descriptor);
        assert!(!reverse.is_descriptor);
    }

    #[test]
    fn test_resolve_tuple_method_returns_builtin_for_sequence_protocol() {
        for name in ["__iter__", "__len__", "__getitem__", "count", "index"] {
            let method =
                resolve_tuple_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
            assert!(method.method.as_object_ptr().is_some());
            assert!(!method.is_descriptor);
        }
    }

    #[test]
    fn test_tuple_methods_use_native_storage() {
        let (tuple_value, tuple_ptr) = boxed_tuple_value(TupleObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(1).unwrap(),
        ]));

        assert_eq!(
            tuple_len(&[tuple_value])
                .expect("tuple.__len__ should succeed")
                .as_int(),
            Some(3)
        );
        assert_eq!(
            tuple_getitem(&[tuple_value, Value::int(-1).unwrap()])
                .expect("tuple.__getitem__ should accept negative indices")
                .as_int(),
            Some(1)
        );
        assert_eq!(
            tuple_count(&[tuple_value, Value::int(1).unwrap()])
                .expect("tuple.count should succeed")
                .as_int(),
            Some(2)
        );
        assert_eq!(
            tuple_index(&[tuple_value, Value::int(1).unwrap(), Value::int(1).unwrap(),])
                .expect("tuple.index should honor the start bound")
                .as_int(),
            Some(2)
        );

        let slice_ptr = Box::into_raw(Box::new(SliceObject::new(Some(0), Some(3), Some(2))));
        let sliced = tuple_getitem(&[tuple_value, Value::object_ptr(slice_ptr as *const ())])
            .expect("tuple.__getitem__ should accept slice objects");
        let sliced_ptr = sliced
            .as_object_ptr()
            .expect("tuple slice should return a tuple object")
            as *mut TupleObject;
        let sliced_tuple = unsafe { &*sliced_ptr };
        assert_eq!(sliced_tuple.len(), 2);
        assert_eq!(sliced_tuple.as_slice()[0].as_int(), Some(1));
        assert_eq!(sliced_tuple.as_slice()[1].as_int(), Some(1));

        let iter_value = tuple_iter(&[tuple_value]).expect("tuple.__iter__ should succeed");
        let iter = get_iterator_mut(&iter_value).expect("tuple.__iter__ should return iterator");
        assert_eq!(iter.next().and_then(|value| value.as_int()), Some(1));
        assert_eq!(iter.next().and_then(|value| value.as_int()), Some(2));
        assert_eq!(iter.next().and_then(|value| value.as_int()), Some(1));
        assert!(iter.next().is_none());

        unsafe {
            drop(Box::from_raw(
                iter_value.as_object_ptr().unwrap() as *mut IteratorObject
            ));
            drop(Box::from_raw(sliced_ptr));
            drop(Box::from_raw(slice_ptr));
            drop(Box::from_raw(tuple_ptr));
        }
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
    fn test_list_clear_removes_all_items_and_returns_none() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));

        let result = list_clear(&[list_value]).expect("clear should work");

        assert!(result.is_none());
        assert_eq!(list_values(list_ptr), Vec::<i64>::new());

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_methods_accept_heap_list_subclasses_with_native_backing() {
        let object = Box::into_raw(Box::new(ShapedObject::new_list_backed(
            TypeId::from_raw(512),
            Shape::empty(),
        )));
        let value = Value::object_ptr(object as *const ());

        list_append(&[value, Value::int(7).unwrap()]).expect("append should work on subclasses");
        list_append(&[value, Value::int(11).unwrap()]).expect("append should work on subclasses");
        let len = list_len(&[value]).expect("__len__ should work on subclasses");
        let first = list_getitem(&[value, Value::int(0).unwrap()])
            .expect("__getitem__ should work on subclasses");
        let iter = list_iter(&[value]).expect("__iter__ should work on subclasses");
        let copied = list_copy(&[value]).expect("copy should work on subclasses");
        list_clear(&[value]).expect("clear should work on subclasses");
        let copied_ptr = copied
            .as_object_ptr()
            .expect("list.copy should still return a concrete list")
            as *mut ListObject;
        let iter_ptr = iter
            .as_object_ptr()
            .expect("list.__iter__ should return an iterator")
            as *mut IteratorObject;
        let iter_ref = unsafe { &mut *iter_ptr };

        let backing = unsafe { &*object }
            .list_backing()
            .expect("list backing should exist");
        assert!(backing.as_slice().is_empty());
        assert_eq!(len.as_int(), Some(2));
        assert_eq!(first.as_int(), Some(7));
        assert_eq!(iter_ref.next().and_then(|value| value.as_int()), Some(7));
        assert_eq!(iter_ref.next().and_then(|value| value.as_int()), Some(11));
        assert!(iter_ref.next().is_none());
        assert_eq!(list_values(copied_ptr), vec![7, 11]);

        unsafe {
            drop(Box::from_raw(iter_ptr));
            drop(Box::from_raw(copied_ptr));
            drop(Box::from_raw(object));
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
    fn test_list_count_counts_matching_values() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(1).unwrap(),
            Value::int(3).unwrap(),
        ]));

        assert_eq!(
            list_count(&[list_value, Value::int(1).unwrap()])
                .expect("list.count should succeed")
                .as_int(),
            Some(2)
        );

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_index_honors_optional_bounds() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::string(intern("alpha")),
            Value::string(intern("beta")),
            Value::string(intern("alpha")),
            Value::string(intern("gamma")),
        ]));

        assert_eq!(
            list_index(&[list_value, Value::string(intern("alpha"))])
                .expect("list.index should find the first match")
                .as_int(),
            Some(0)
        );
        assert_eq!(
            list_index(&[
                list_value,
                Value::string(intern("alpha")),
                Value::int(1).unwrap(),
            ])
            .expect("list.index should honor the start bound")
            .as_int(),
            Some(2)
        );
        let err = list_index(&[
            list_value,
            Value::string(intern("alpha")),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ])
        .expect_err("list.index should honor the stop bound");
        assert_eq!(err.to_string(), "ValueError: list.index(x): x not in list");

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_remove_deletes_first_matching_value_and_returns_none() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));

        let result = list_remove(&[list_value, Value::int(2).unwrap()])
            .expect("remove should delete the first matching value");
        assert!(result.is_none());
        assert_eq!(list_values(list_ptr), vec![1, 2, 3]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_pop_without_index_returns_last_item() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));

        let popped = list_pop(&[list_value]).expect("pop should remove the tail element");
        assert_eq!(popped.as_int(), Some(3));
        assert_eq!(list_values(list_ptr), vec![1, 2]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_pop_with_index_supports_negative_offsets() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(10).unwrap(),
            Value::int(20).unwrap(),
            Value::int(30).unwrap(),
        ]));

        let popped =
            list_pop(&[list_value, Value::int(-2).unwrap()]).expect("pop should honor negatives");
        assert_eq!(popped.as_int(), Some(20));
        assert_eq!(list_values(list_ptr), vec![10, 30]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_pop_empty_and_out_of_range_match_cpython_messages() {
        let (empty_value, empty_ptr) = boxed_list_value(ListObject::new());
        let empty_err = list_pop(&[empty_value]).expect_err("empty pop should fail");
        match empty_err {
            BuiltinError::IndexError(message) => assert_eq!(message, "pop from empty list"),
            other => panic!("expected IndexError, got {other:?}"),
        }

        let indexed_empty_err = list_pop(&[empty_value, Value::int(0).unwrap()])
            .expect_err("indexed empty pop should fail");
        match indexed_empty_err {
            BuiltinError::IndexError(message) => assert_eq!(message, "pop from empty list"),
            other => panic!("expected IndexError, got {other:?}"),
        }

        let (list_value, list_ptr) =
            boxed_list_value(ListObject::from_slice(&[Value::int(1).unwrap()]));
        let range_err = list_pop(&[list_value, Value::int(4).unwrap()])
            .expect_err("out-of-range pop should fail");
        match range_err {
            BuiltinError::IndexError(message) => assert_eq!(message, "pop index out of range"),
            other => panic!("expected IndexError, got {other:?}"),
        }
        assert_eq!(list_values(list_ptr), vec![1]);

        unsafe {
            drop(Box::from_raw(list_ptr));
            drop(Box::from_raw(empty_ptr));
        }
    }

    #[test]
    fn test_list_insert_inserts_at_requested_position_and_returns_none() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(3).unwrap(),
        ]));

        let result = list_insert(&[list_value, Value::int(1).unwrap(), Value::int(2).unwrap()])
            .expect("insert should place the value before the requested index");
        assert!(result.is_none());
        assert_eq!(list_values(list_ptr), vec![1, 2, 3]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_insert_clamps_indices_to_list_bounds() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));

        list_insert(&[list_value, Value::int(-10).unwrap(), Value::int(1).unwrap()])
            .expect("insert should clamp large negative indices to the start");
        list_insert(&[list_value, Value::int(99).unwrap(), Value::int(4).unwrap()])
            .expect("insert should clamp large positive indices to the end");
        assert_eq!(list_values(list_ptr), vec![1, 2, 3, 4]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_insert_rejects_non_integer_indices() {
        let (list_value, list_ptr) =
            boxed_list_value(ListObject::from_slice(&[Value::int(1).unwrap()]));

        let err = list_insert(&[
            list_value,
            Value::string(intern("bad")),
            Value::int(2).unwrap(),
        ])
        .expect_err("insert should reject non-integer indices");
        match err {
            BuiltinError::TypeError(message) => {
                assert_eq!(message, "'str' object cannot be interpreted as an integer");
            }
            other => panic!("expected TypeError, got {other:?}"),
        }
        assert_eq!(list_values(list_ptr), vec![1]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_remove_uses_runtime_string_equality() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::string(intern("needle")),
            Value::string(intern("other")),
        ]));
        let needle_ptr = Box::into_raw(Box::new(StringObject::new("needle")));
        let needle = Value::object_ptr(needle_ptr as *const ());

        let result = list_remove(&[list_value, needle])
            .expect("remove should match strings across runtime representations");
        assert!(result.is_none());

        let remaining = unsafe { &*list_ptr }
            .as_slice()
            .iter()
            .copied()
            .map(string_value)
            .collect::<Vec<_>>();
        assert_eq!(remaining, vec!["other".to_string()]);

        unsafe {
            drop(Box::from_raw(needle_ptr));
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_remove_raises_value_error_when_item_is_missing() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]));

        let err = list_remove(&[list_value, Value::int(9).unwrap()])
            .expect_err("remove should fail when the value is absent");
        match err {
            BuiltinError::ValueError(message) => {
                assert_eq!(message, "list.remove(x): x not in list");
            }
            other => panic!("expected ValueError, got {other:?}"),
        }
        assert_eq!(list_values(list_ptr), vec![1, 2]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_reverse_mutates_in_place_and_returns_none() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));

        let result = list_reverse(&[list_value]).expect("reverse should work");
        assert!(result.is_none());
        assert_eq!(list_values(list_ptr), vec![3, 2, 1]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_sort_orders_numeric_values_in_place() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(3).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]));
        let mut vm = VirtualMachine::default();

        let result = list_sort_with_vm(&mut vm, &[list_value], &[]).expect("sort should succeed");
        assert!(result.is_none());
        assert_eq!(list_values(list_ptr), vec![1, 2, 3]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_sort_honors_reverse_keyword() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(3).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]));
        let mut vm = VirtualMachine::default();

        let result = list_sort_with_vm(&mut vm, &[list_value], &[("reverse", Value::bool(true))])
            .expect("sort(reverse=True) should succeed");
        assert!(result.is_none());
        assert_eq!(list_values(list_ptr), vec![3, 2, 1]);

        unsafe {
            drop(Box::from_raw(list_ptr));
        }
    }

    #[test]
    fn test_list_sort_honors_key_keyword_with_builtin_callable() {
        let (list_value, list_ptr) = boxed_list_value(ListObject::from_slice(&[
            Value::int(2).unwrap(),
            Value::int(0).unwrap(),
            Value::int(3).unwrap(),
            Value::int(1).unwrap(),
        ]));
        let mut vm = VirtualMachine::default();

        let result = list_sort_with_vm(
            &mut vm,
            &[list_value],
            &[(
                "key",
                crate::builtins::builtin_type_object_for_type_id(TypeId::BOOL),
            )],
        )
        .expect("sort(key=bool) should succeed");
        assert!(result.is_none());
        assert_eq!(list_values(list_ptr), vec![0, 2, 3, 1]);

        unsafe {
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
        let popitem = resolve_dict_method("popitem").expect("popitem should resolve");
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
        assert!(popitem.method.as_object_ptr().is_some());
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
        assert!(!popitem.is_descriptor);
        assert!(!setdefault.is_descriptor);
        assert!(!clear.is_descriptor);
        assert!(!update.is_descriptor);
        assert!(!copy.is_descriptor);
    }

    #[test]
    fn test_dict_popitem_returns_latest_entry_and_rejects_empty_dict() {
        let mut dict = DictObject::new();
        dict.set(Value::string(intern("alpha")), Value::int(1).unwrap());
        dict.set(Value::string(intern("beta")), Value::int(2).unwrap());
        let ptr = Box::into_raw(Box::new(dict));
        let value = Value::object_ptr(ptr as *const ());

        let popped = dict_popitem(&[value]).expect("popitem should succeed");
        let popped_ptr = popped
            .as_object_ptr()
            .expect("popitem should return a tuple") as *mut TupleObject;
        let tuple = unsafe { &*popped_ptr };
        assert_eq!(tuple.as_slice()[0], Value::string(intern("beta")));
        assert_eq!(tuple.as_slice()[1].as_int(), Some(2));
        assert_eq!(unsafe { &*ptr }.len(), 1);

        let second = dict_popitem(&[value]).expect("second popitem should also succeed");
        let second_ptr = second
            .as_object_ptr()
            .expect("second popitem should return a tuple")
            as *mut TupleObject;
        let err = dict_popitem(&[value]).expect_err("empty popitem should fail");
        assert_eq!(err.to_string(), "KeyError: popitem(): dictionary is empty");

        unsafe {
            drop(Box::from_raw(second_ptr));
            drop(Box::from_raw(popped_ptr));
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_dict_methods_reject_unhashable_keys() {
        let dict = DictObject::new();
        let dict_ptr = Box::into_raw(Box::new(dict));
        let dict_value = Value::object_ptr(dict_ptr as *const ());
        let key = to_object_value(ListObject::from_slice(&[Value::int(1).unwrap()]));

        for err in [
            dict_contains(&[dict_value, key]).unwrap_err(),
            dict_get(&[dict_value, key]).unwrap_err(),
            dict_setitem(&[dict_value, key, Value::int(1).unwrap()]).unwrap_err(),
        ] {
            assert!(err.to_string().contains("unhashable type: 'list'"));
        }

        unsafe {
            drop(Box::from_raw(
                key.as_object_ptr().unwrap() as *mut ListObject
            ));
            drop(Box::from_raw(dict_ptr));
        }
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
    fn test_object_attribute_mutator_wrappers_use_default_attribute_storage() {
        let mut vm = VirtualMachine::new();
        let object = crate::builtins::builtin_object(&[]).expect("object() should succeed");
        let object_ptr = object
            .as_object_ptr()
            .expect("object() should allocate a shaped object");

        object_setattr(
            &mut vm,
            &[
                object,
                Value::string(intern("token")),
                Value::int(42).expect("token should fit"),
            ],
        )
        .expect("object.__setattr__ should set default attributes");

        let shaped = unsafe { &*(object_ptr as *const ShapedObject) };
        assert_eq!(
            shaped
                .get_property("token")
                .expect("token should be stored")
                .as_int(),
            Some(42)
        );

        object_delattr(&mut vm, &[object, Value::string(intern("token"))])
            .expect("object.__delattr__ should delete default attributes");
        let shaped = unsafe { &*(object_ptr as *const ShapedObject) };
        assert!(shaped.get_property("token").is_none());
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
    fn test_resolve_int_method_exposes_bit_operations() {
        let bit_length = unsafe {
            &*(resolve_int_method("bit_length")
                .expect("bit_length should resolve")
                .method
                .as_object_ptr()
                .expect("bit_length should be allocated")
                as *const BuiltinFunctionObject)
        };
        let bit_count = unsafe {
            &*(resolve_int_method("bit_count")
                .expect("bit_count should resolve")
                .method
                .as_object_ptr()
                .expect("bit_count should be allocated")
                as *const BuiltinFunctionObject)
        };

        assert_eq!(bit_length.name(), "int.bit_length");
        assert_eq!(bit_count.name(), "int.bit_count");
    }

    #[test]
    fn test_resolve_int_method_exposes_add_wrapper() {
        let add = unsafe {
            &*(resolve_int_method("__add__")
                .expect("__add__ should resolve")
                .method
                .as_object_ptr()
                .expect("__add__ should be allocated")
                as *const BuiltinFunctionObject)
        };

        assert_eq!(add.name(), "int.__add__");
    }

    #[test]
    fn test_int_index_matches_python_descriptor_contract() {
        let int_result = int_index(&[Value::int(42).expect("value should fit")])
            .expect("int.__index__ should accept exact ints");
        assert_eq!(int_result.as_int(), Some(42));

        let bool_result =
            int_index(&[Value::bool(true)]).expect("bool should inherit int.__index__");
        assert_eq!(bool_result.as_int(), Some(1));

        let error = int_index(&[Value::none()])
            .expect_err("non-int receiver should fail descriptor validation");
        assert!(matches!(error, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_int_add_matches_python_descriptor_contract() {
        let result = int_add(&[
            bigint_to_value(num_bigint::BigInt::from(i64::MAX)),
            Value::int(1).expect("one should fit"),
        ])
        .expect("int.__add__ should accept integer operands");
        assert_eq!(
            prism_runtime::types::int::value_to_bigint(result),
            Some(num_bigint::BigInt::from(i64::MAX) + num_bigint::BigInt::from(1_i64))
        );

        let bool_result = int_add(&[Value::bool(true), Value::int(2).expect("two should fit")])
            .expect("bool should be accepted as an int receiver");
        assert_eq!(bool_result.as_int(), Some(3));

        let unsupported = int_add(&[Value::int(1).expect("one should fit"), Value::none()])
            .expect("unsupported rhs should return NotImplemented");
        assert_eq!(unsupported, builtin_not_implemented_value());

        let error = int_add(&[Value::none(), Value::int(1).expect("one should fit")])
            .expect_err("non-int receiver should fail descriptor validation");
        assert!(matches!(error, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_int_bit_operations_match_python_magnitude_rules() {
        let bit_length =
            int_bit_length(&[Value::int(-37).unwrap()]).expect("bit_length should accept ints");
        assert_eq!(bit_length.as_int(), Some(6));

        let bit_count =
            int_bit_count(&[Value::int(-37).unwrap()]).expect("bit_count should accept ints");
        assert_eq!(bit_count.as_int(), Some(3));

        let boolean = int_bit_length(&[Value::bool(true)]).expect("bool should inherit int APIs");
        assert_eq!(boolean.as_int(), Some(1));
    }

    #[test]
    fn test_resolve_exception_method_returns_builtin_for_with_traceback() {
        let with_traceback =
            resolve_exception_method("with_traceback").expect("with_traceback should resolve");
        let ptr = with_traceback
            .method
            .as_object_ptr()
            .expect("with_traceback should be allocated");
        let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "BaseException.with_traceback");
        assert!(!with_traceback.is_descriptor);
    }

    #[test]
    fn test_object_rich_comparisons_follow_identity_default() {
        let mut vm = VirtualMachine::new();
        let same = Value::int(7).unwrap();
        assert_eq!(
            object_eq(&[same, same]).expect("__eq__ should accept two operands"),
            Value::bool(true)
        );
        assert_eq!(
            object_ne(&mut vm, &[same, same]).expect("__ne__ should accept two operands"),
            Value::bool(false)
        );

        let lhs = Value::int(7).unwrap();
        let rhs = Value::int(8).unwrap();
        assert_eq!(
            object_eq(&[lhs, rhs]).expect("__eq__ should accept mismatched operands"),
            builtin_not_implemented_value()
        );
        assert_eq!(
            object_ne(&mut vm, &[lhs, rhs]).expect("__ne__ should accept mismatched operands"),
            Value::bool(true)
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
    fn test_resolve_str_method_returns_builtin_for_capitalize() {
        let capitalize = resolve_str_method("capitalize").expect("capitalize should resolve");
        assert!(capitalize.method.as_object_ptr().is_some());
        assert!(!capitalize.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_replace() {
        let replace = resolve_str_method("replace").expect("replace should resolve");
        assert!(replace.method.as_object_ptr().is_some());
        assert!(!replace.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_remove_affix_methods() {
        for name in ["removeprefix", "removesuffix"] {
            let method =
                resolve_str_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
            assert!(method.method.as_object_ptr().is_some());
            assert!(!method.is_descriptor);
        }
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_split() {
        let split = resolve_str_method("split").expect("split should resolve");
        assert!(split.method.as_object_ptr().is_some());
        assert!(!split.is_descriptor);
        let rsplit = resolve_str_method("rsplit").expect("rsplit should resolve");
        assert!(rsplit.method.as_object_ptr().is_some());
        assert!(!rsplit.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_splitlines() {
        let splitlines = resolve_str_method("splitlines").expect("splitlines should resolve");
        assert!(splitlines.method.as_object_ptr().is_some());
        assert!(!splitlines.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_expandtabs() {
        let expandtabs = resolve_str_method("expandtabs").expect("expandtabs should resolve");
        assert!(expandtabs.method.as_object_ptr().is_some());
        assert!(!expandtabs.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_strip_family() {
        for name in ["strip", "lstrip", "rstrip"] {
            let method = resolve_str_method(name).expect("strip-family method should resolve");
            assert!(
                method.method.as_object_ptr().is_some(),
                "{name} should resolve"
            );
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
    fn test_resolve_str_method_returns_builtin_for_format() {
        let format = resolve_str_method("format").expect("format should resolve");
        assert!(format.method.as_object_ptr().is_some());
        assert!(!format.is_descriptor);
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
        let partition = resolve_str_method("partition").expect("partition should resolve");
        let rpartition = resolve_str_method("rpartition").expect("rpartition should resolve");
        assert!(startswith.method.as_object_ptr().is_some());
        assert!(endswith.method.as_object_ptr().is_some());
        assert!(partition.method.as_object_ptr().is_some());
        assert!(rpartition.method.as_object_ptr().is_some());
        assert!(!startswith.is_descriptor);
        assert!(!endswith.is_descriptor);
        assert!(!partition.is_descriptor);
        assert!(!rpartition.is_descriptor);
    }

    #[test]
    fn test_resolve_str_method_returns_builtin_for_find_family_and_predicates() {
        for name in [
            "find",
            "rfind",
            "index",
            "rindex",
            "count",
            "translate",
            "isascii",
            "isalpha",
            "isdigit",
            "isalnum",
            "isspace",
            "isupper",
            "islower",
            "isdecimal",
            "isnumeric",
            "istitle",
        ] {
            let method =
                resolve_str_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
            assert!(method.method.as_object_ptr().is_some());
            assert!(!method.is_descriptor);
        }
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
    fn test_str_translate_applies_mapping_values_and_preserves_missing_chars() {
        let mut vm = VirtualMachine::new();
        let mut table = DictObject::new();
        table.set(
            Value::int('a' as i64).unwrap(),
            Value::int('A' as i64).unwrap(),
        );
        table.set(
            Value::int('.' as i64).unwrap(),
            Value::string(intern("\\.")),
        );
        table.set(Value::int('x' as i64).unwrap(), Value::none());
        let table_ptr = Box::into_raw(Box::new(table));
        let table_value = Value::object_ptr(table_ptr as *const ());

        let translated = str_translate(&mut vm, &[Value::string(intern("a.x!")), table_value])
            .expect("translate should apply integer, string, and deletion mappings");
        assert_eq!(string_value(translated), "A\\.!");

        unsafe {
            drop(Box::from_raw(table_ptr));
        }
    }

    #[test]
    fn test_str_translate_reuses_receiver_and_validates_mapping_results() {
        let mut vm = VirtualMachine::new();
        let empty_table_ptr = Box::into_raw(Box::new(DictObject::new()));
        let empty_table_value = Value::object_ptr(empty_table_ptr as *const ());
        let source = Value::string(intern("plain"));

        let unchanged = str_translate(&mut vm, &[source, empty_table_value])
            .expect("missing translations should leave characters unchanged");
        assert_eq!(unchanged, source);

        let mut invalid_table = DictObject::new();
        let invalid_replacement = to_object_value(ListObject::new());
        invalid_table.set(Value::int('p' as i64).unwrap(), invalid_replacement);
        let invalid_table_ptr = Box::into_raw(Box::new(invalid_table));
        let invalid_table_value = Value::object_ptr(invalid_table_ptr as *const ());
        let err = str_translate(&mut vm, &[source, invalid_table_value])
            .expect_err("unsupported mapping result should fail");
        assert_eq!(
            err.to_string(),
            "TypeError: character mapping must return integer, None or str"
        );

        unsafe {
            drop(Box::from_raw(
                invalid_replacement.as_object_ptr().unwrap() as *mut ListObject
            ));
            drop(Box::from_raw(
                invalid_table_value.as_object_ptr().unwrap() as *mut DictObject
            ));
            drop(Box::from_raw(
                empty_table_value.as_object_ptr().unwrap() as *mut DictObject
            ));
        }
    }

    #[test]
    fn test_str_capitalize_uppercases_first_character_and_lowercases_rest() {
        let result =
            str_capitalize(&[Value::string(intern("hELLO"))]).expect("capitalize should work");
        assert_eq!(string_value(result), "Hello");

        let unchanged = str_capitalize(&[Value::string(intern("Hello"))])
            .expect("capitalize should preserve canonical form");
        assert_eq!(unchanged, Value::string(intern("Hello")));
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
    fn test_str_replace_accepts_cpython_keyword_arguments() {
        let replaced = str_replace_kw(
            &[Value::string(intern("banana"))],
            &[
                ("old", Value::string(intern("na"))),
                ("new", Value::string(intern("NA"))),
                ("count", Value::int(1).unwrap()),
            ],
        )
        .expect("keyword replace should work");
        assert_eq!(string_value(replaced), "baNAna");

        let mixed = str_replace_kw(
            &[Value::string(intern("banana")), Value::string(intern("na"))],
            &[("new", Value::string(intern("NA")))],
        )
        .expect("mixed positional and keyword replace should work");
        assert_eq!(string_value(mixed), "baNANA");

        let duplicate = str_replace_kw(
            &[Value::string(intern("banana")), Value::string(intern("na"))],
            &[("old", Value::string(intern("a")))],
        )
        .expect_err("duplicate old argument should fail");
        assert!(duplicate.to_string().contains("multiple values"));
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
    fn test_str_remove_affix_methods_match_python_contract() {
        let without_prefix =
            str_removeprefix(&[Value::string(intern("spam")), Value::string(intern("sp"))])
                .expect("removeprefix should remove matching prefix");
        assert_eq!(string_value(without_prefix), "am");

        let without_suffix =
            str_removesuffix(&[Value::string(intern("spam")), Value::string(intern("am"))])
                .expect("removesuffix should remove matching suffix");
        assert_eq!(string_value(without_suffix), "sp");

        let full_prefix = str_removeprefix(&[
            Value::string(intern("abcde")),
            Value::string(intern("abcde")),
        ])
        .expect("removeprefix should handle full-string matches");
        assert_eq!(string_value(full_prefix), "");

        let full_suffix = str_removesuffix(&[
            Value::string(intern("abcde")),
            Value::string(intern("abcde")),
        ])
        .expect("removesuffix should handle full-string matches");
        assert_eq!(string_value(full_suffix), "");
    }

    #[test]
    fn test_str_remove_affix_reuses_receiver_for_noop_cases() {
        let receiver = Value::string(intern("spam"));

        let missing_prefix = str_removeprefix(&[receiver, Value::string(intern("python"))])
            .expect("missing prefix should be a no-op");
        assert_eq!(missing_prefix, receiver);

        let empty_prefix = str_removeprefix(&[receiver, Value::string(intern(""))])
            .expect("empty prefix should be a no-op");
        assert_eq!(empty_prefix, receiver);

        let missing_suffix = str_removesuffix(&[receiver, Value::string(intern("python"))])
            .expect("missing suffix should be a no-op");
        assert_eq!(missing_suffix, receiver);

        let empty_suffix = str_removesuffix(&[receiver, Value::string(intern(""))])
            .expect("empty suffix should be a no-op");
        assert_eq!(empty_suffix, receiver);
    }

    #[test]
    fn test_str_remove_affix_rejects_invalid_arguments() {
        let missing =
            str_removeprefix(&[Value::string(intern("hello"))]).expect_err("prefix is required");
        assert!(
            missing
                .to_string()
                .contains("str.removeprefix() takes exactly 1 argument")
        );

        let non_string =
            str_removesuffix(&[Value::string(intern("hello")), Value::int(42).unwrap()])
                .expect_err("suffix must be a string");
        assert!(
            non_string
                .to_string()
                .contains("str.removesuffix() argument 1 must be str")
        );

        let tuple_affix = to_object_value(TupleObject::from_slice(&[
            Value::string(intern("he")),
            Value::string(intern("l")),
        ]));
        let tuple_err = str_removeprefix(&[Value::string(intern("hello")), tuple_affix])
            .expect_err("tuple prefixes are not accepted");
        assert!(
            tuple_err
                .to_string()
                .contains("str.removeprefix() argument 1 must be str")
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
    fn test_str_split_accepts_sep_and_maxsplit_keywords() {
        let result = str_split_kw(
            &[Value::string(intern("3.12.0"))],
            &[
                ("sep", Value::string(intern("."))),
                ("maxsplit", Value::int(1).unwrap()),
            ],
        )
        .expect("split keywords should work");

        let result_ptr = result.as_object_ptr().expect("split should return a list");
        let list = unsafe { &*(result_ptr as *const ListObject) };
        let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
        assert_eq!(values, vec!["3".to_string(), "12.0".to_string()]);
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
    fn test_str_rsplit_supports_explicit_separator_and_maxsplit() {
        let result = str_rsplit(&[
            Value::string(intern("os.confstr")),
            Value::string(intern(".")),
            Value::int(1).unwrap(),
        ])
        .expect("rsplit with separator should work");

        let result_ptr = result.as_object_ptr().expect("rsplit should return a list");
        let list = unsafe { &*(result_ptr as *const ListObject) };
        let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
        assert_eq!(values, vec!["os".to_string(), "confstr".to_string()]);
    }

    #[test]
    fn test_str_rsplit_accepts_keywords_and_rejects_duplicates() {
        let result = str_rsplit_kw(
            &[Value::string(intern("a.b.c"))],
            &[
                ("sep", Value::string(intern("."))),
                ("maxsplit", Value::int(1).unwrap()),
            ],
        )
        .expect("rsplit keywords should work");

        let result_ptr = result.as_object_ptr().expect("rsplit should return a list");
        let list = unsafe { &*(result_ptr as *const ListObject) };
        let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
        assert_eq!(values, vec!["a.b".to_string(), "c".to_string()]);

        let duplicate = str_split_kw(
            &[Value::string(intern("a.b")), Value::string(intern("."))],
            &[("sep", Value::string(intern(",")))],
        )
        .expect_err("duplicate sep must be rejected");
        assert!(
            duplicate
                .to_string()
                .contains("multiple values for argument 'sep'")
        );
    }

    #[test]
    fn test_str_rsplit_preserves_left_whitespace_remainder() {
        let result = str_rsplit(&[
            Value::string(intern("  alpha   beta gamma  ")),
            Value::none(),
            Value::int(1).unwrap(),
        ])
        .expect("rsplit with implicit whitespace should work");

        let result_ptr = result.as_object_ptr().expect("rsplit should return a list");
        let list = unsafe { &*(result_ptr as *const ListObject) };
        let values: Vec<String> = list.as_slice().iter().copied().map(string_value).collect();
        assert_eq!(
            values,
            vec!["alpha   beta".to_string(), "gamma".to_string()]
        );
    }

    #[test]
    fn test_str_splitlines_handles_crlf_and_optional_keepends() {
        let without_keepends = str_splitlines(&[Value::string(intern("alpha\r\nbeta\n"))])
            .expect("splitlines should work");
        let without_ptr = without_keepends
            .as_object_ptr()
            .expect("splitlines should return a list");
        let without_list = unsafe { &*(without_ptr as *const ListObject) };
        let without_values: Vec<String> = without_list
            .iter()
            .map(|value| {
                string_object_from_value(*value)
                    .unwrap()
                    .as_str()
                    .to_string()
            })
            .collect();
        assert_eq!(
            without_values,
            vec!["alpha".to_string(), "beta".to_string()]
        );

        let with_keepends =
            str_splitlines(&[Value::string(intern("alpha\r\nbeta\n")), Value::bool(true)])
                .expect("splitlines keepends should work");
        let with_ptr = with_keepends
            .as_object_ptr()
            .expect("splitlines should return a list");
        let with_list = unsafe { &*(with_ptr as *const ListObject) };
        let with_values: Vec<String> = with_list
            .iter()
            .map(|value| {
                string_object_from_value(*value)
                    .unwrap()
                    .as_str()
                    .to_string()
            })
            .collect();
        assert_eq!(
            with_values,
            vec!["alpha\r\n".to_string(), "beta\n".to_string()]
        );
    }

    #[test]
    fn test_str_splitlines_returns_empty_list_for_empty_string() {
        let result = str_splitlines(&[Value::string(intern(""))]).expect("splitlines should work");
        let ptr = result
            .as_object_ptr()
            .expect("splitlines should return a list");
        let list = unsafe { &*(ptr as *const ListObject) };
        assert!(list.is_empty());
    }

    #[test]
    fn test_str_splitlines_accepts_keepends_keyword_argument() {
        let splitlines = builtin_from_value(
            resolve_str_method("splitlines")
                .expect("splitlines should resolve")
                .method,
        );
        let result = splitlines
            .call_with_keywords(
                &[Value::string(intern("a\r\nb"))],
                &[("keepends", Value::bool(true))],
            )
            .expect("splitlines keyword call should succeed");
        let list = unsafe {
            &*(result
                .as_object_ptr()
                .expect("splitlines should return a list") as *const ListObject)
        };
        let values: Vec<String> = list
            .iter()
            .map(|value| {
                string_object_from_value(*value)
                    .unwrap()
                    .as_str()
                    .to_string()
            })
            .collect();
        assert_eq!(values, vec!["a\r\n".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_str_expandtabs_expands_tabs_and_resets_columns_after_newlines() {
        let result = str_expandtabs(&[Value::string(intern("01\t0123\na\r\tb"))])
            .expect("expandtabs should work");
        assert_eq!(string_value(result), "01      0123\na\r        b");
    }

    #[test]
    fn test_str_expandtabs_supports_negative_and_boolean_tab_sizes() {
        let collapsed = str_expandtabs(&[Value::string(intern("a\tb")), Value::int(-1).unwrap()])
            .expect("negative tabsize should collapse tabs");
        assert_eq!(string_value(collapsed), "ab");

        let boolean = str_expandtabs(&[Value::string(intern("a\tb")), Value::bool(true)])
            .expect("bool tabsize should be treated as an integer");
        assert_eq!(string_value(boolean), "a b");
    }

    #[test]
    fn test_str_expandtabs_accepts_tabsize_keyword_argument() {
        let expandtabs = builtin_from_value(
            resolve_str_method("expandtabs")
                .expect("expandtabs should resolve")
                .method,
        );
        let result = expandtabs
            .call_with_keywords(
                &[Value::string(intern("a\tb"))],
                &[("tabsize", Value::int(4).unwrap())],
            )
            .expect("expandtabs keyword call should succeed");
        assert_eq!(string_value(result), "a   b");
    }

    #[test]
    fn test_str_expandtabs_reuses_receiver_when_no_tabs_are_present() {
        let receiver = Value::string(intern("stable"));
        let result = str_expandtabs(&[receiver]).expect("expandtabs should accept default tabsize");
        assert_eq!(result, receiver);
    }

    #[test]
    fn test_str_partition_matches_python_contract() {
        let value = str_partition(&[
            Value::string(intern("alpha.beta.gamma")),
            Value::string(intern(".")),
        ])
        .expect("partition should succeed");
        let tuple_ptr = value
            .as_object_ptr()
            .expect("partition should return a tuple") as *mut TupleObject;
        let tuple = unsafe { &*tuple_ptr };
        assert_eq!(string_value(tuple.as_slice()[0]), "alpha");
        assert_eq!(string_value(tuple.as_slice()[1]), ".");
        assert_eq!(string_value(tuple.as_slice()[2]), "beta.gamma");

        let missing = str_partition(&[Value::string(intern("alpha")), Value::string(intern("."))])
            .expect("partition should handle missing separator");
        let missing_ptr = missing
            .as_object_ptr()
            .expect("missing partition should return a tuple")
            as *mut TupleObject;
        let missing_tuple = unsafe { &*missing_ptr };
        assert_eq!(string_value(missing_tuple.as_slice()[0]), "alpha");
        assert_eq!(string_value(missing_tuple.as_slice()[1]), "");
        assert_eq!(string_value(missing_tuple.as_slice()[2]), "");

        let empty_separator =
            str_partition(&[Value::string(intern("alpha")), Value::string(intern(""))])
                .expect_err("partition should reject an empty separator");
        assert!(matches!(empty_separator, BuiltinError::ValueError(_)));

        unsafe {
            drop(Box::from_raw(tuple_ptr));
            drop(Box::from_raw(missing_ptr));
        }
    }

    #[test]
    fn test_str_rpartition_matches_python_contract() {
        let value = str_rpartition(&[
            Value::string(intern("alpha.beta.gamma")),
            Value::string(intern(".")),
        ])
        .expect("rpartition should succeed");
        let tuple_ptr = value
            .as_object_ptr()
            .expect("rpartition should return a tuple") as *mut TupleObject;
        let tuple = unsafe { &*tuple_ptr };
        assert_eq!(string_value(tuple.as_slice()[0]), "alpha.beta");
        assert_eq!(string_value(tuple.as_slice()[1]), ".");
        assert_eq!(string_value(tuple.as_slice()[2]), "gamma");

        let missing = str_rpartition(&[Value::string(intern("alpha")), Value::string(intern("."))])
            .expect("rpartition should handle missing separator");
        let missing_ptr = missing
            .as_object_ptr()
            .expect("missing rpartition should return a tuple")
            as *mut TupleObject;
        let missing_tuple = unsafe { &*missing_ptr };
        assert_eq!(string_value(missing_tuple.as_slice()[0]), "");
        assert_eq!(string_value(missing_tuple.as_slice()[1]), "");
        assert_eq!(string_value(missing_tuple.as_slice()[2]), "alpha");

        unsafe {
            drop(Box::from_raw(tuple_ptr));
            drop(Box::from_raw(missing_ptr));
        }
    }

    #[test]
    fn test_str_strip_family_supports_whitespace_none_and_explicit_char_sets() {
        let stripped = str_strip(&[Value::string(intern("  alpha  "))])
            .expect("strip should trim surrounding whitespace");
        assert_eq!(string_value(stripped), "alpha");

        let none_chars = str_lstrip(&[Value::string(intern("\n\t alpha")), Value::none()])
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

        let empty_chars = str_rstrip(&[receiver, Value::string(intern(""))])
            .expect("rstrip with empty char set should be a no-op");
        assert_eq!(empty_chars, receiver);
    }

    #[test]
    fn test_str_strip_family_rejects_non_string_char_sets() {
        let err = str_strip(&[Value::string(intern("value")), Value::int(1).unwrap()])
            .expect_err("strip chars must be strings or None");
        assert!(
            err.to_string()
                .contains("str.strip() argument 1 must be str")
        );
    }

    #[test]
    fn test_str_encode_resolves_and_supports_default_and_explicit_codecs() {
        assert!(resolve_str_method("encode").is_some());

        let default_encoded = str_encode(&[Value::string(intern("h\u{00e9}"))])
            .expect("encode should default to utf-8");
        assert_eq!(byte_values(default_encoded), "h\u{00e9}".as_bytes());

        let latin1_encoded = str_encode(&[
            Value::string(intern("\u{00e9}")),
            Value::string(intern("latin-1")),
        ])
        .expect("encode should support latin-1");
        assert_eq!(byte_values(latin1_encoded), vec![0xe9]);

        let ignored = str_encode(&[
            Value::string(intern("A\u{00e9}")),
            Value::string(intern("ascii")),
            Value::string(intern("ignore")),
        ])
        .expect("encode should honor ignore errors");
        assert_eq!(byte_values(ignored), b"A");
    }

    #[test]
    fn test_str_encode_raises_unicode_encode_error_for_strict_failures() {
        let err = str_encode(&[
            Value::string(intern("A\u{00e9}")),
            Value::string(intern("ascii")),
        ])
        .expect_err("strict ascii encoding should fail");

        assert_unicode_encode_error(
            err,
            "'ascii' codec can't encode character '\\xe9' in position 1: ordinal not in range(128)",
        );
    }

    #[test]
    fn test_str_encode_rejects_too_many_arguments() {
        let err = str_encode(&[
            Value::string(intern("abc")),
            Value::string(intern("utf-8")),
            Value::string(intern("strict")),
            Value::string(intern("extra")),
        ])
        .expect_err("encode should reject too many arguments");

        assert_eq!(
            err.to_string(),
            "TypeError: encode() takes at most 2 arguments (3 given)"
        );
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
    fn test_str_find_family_supports_bounds_and_missing_substrings() {
        assert_eq!(
            str_find(&[
                Value::string(intern("prefix_suffix_suffix")),
                Value::string(intern("suffix")),
            ])
            .unwrap(),
            Value::int(7).unwrap()
        );
        assert_eq!(
            str_rfind(&[
                Value::string(intern("prefix_suffix_suffix")),
                Value::string(intern("suffix")),
            ])
            .unwrap(),
            Value::int(14).unwrap()
        );
        assert_eq!(
            str_find(&[
                Value::string(intern("prefix_suffix_suffix")),
                Value::string(intern("suffix")),
                Value::int(8).unwrap(),
                Value::int(20).unwrap(),
            ])
            .unwrap(),
            Value::int(14).unwrap()
        );
        assert_eq!(
            str_find(&[
                Value::string(intern("prefix_suffix")),
                Value::string(intern("missing")),
            ])
            .unwrap(),
            Value::int(-1).unwrap()
        );
        assert_eq!(
            str_count(&[Value::string(intern("aaaa")), Value::string(intern("aa")),]).unwrap(),
            Value::int(2).unwrap()
        );
        assert_eq!(
            str_count(&[Value::string(intern("abc")), Value::string(intern("")),]).unwrap(),
            Value::int(4).unwrap()
        );

        let index_err = str_index(&[
            Value::string(intern("prefix_suffix")),
            Value::string(intern("missing")),
        ])
        .expect_err("index should raise when substring is absent");
        assert!(index_err.to_string().contains("substring not found"));

        let rindex_err = str_rindex(&[
            Value::string(intern("prefix_suffix")),
            Value::string(intern("missing")),
        ])
        .expect_err("rindex should raise when substring is absent");
        assert!(rindex_err.to_string().contains("substring not found"));
    }

    #[test]
    fn test_str_character_predicates_match_python_truthiness_rules() {
        assert_eq!(
            str_isalpha(&[Value::string(intern("Prism"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_isdigit(&[Value::string(intern("0123"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_isdecimal(&[Value::string(intern("0123"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_isnumeric(&[Value::string(intern("0123"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_isalnum(&[Value::string(intern("Prism3"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_isspace(&[Value::string(intern("\u{3000}"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_isupper(&[Value::string(intern("PRISM 3"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_islower(&[Value::string(intern("prism 3"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_istitle(&[Value::string(intern("Prism Runtime"))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_istitle(&[Value::string(intern("prism Runtime"))]).unwrap(),
            Value::bool(false)
        );
        assert_eq!(
            str_isalpha(&[Value::string(intern(""))]).unwrap(),
            Value::bool(false)
        );
        assert_eq!(
            str_isascii(&[Value::string(intern(""))]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            str_isascii(&[Value::string(intern("Prism123"))]).unwrap(),
            Value::bool(true)
        );
        let unicode =
            Value::object_ptr(Box::into_raw(Box::new(StringObject::new("πcache"))) as *const ());
        assert_eq!(str_isascii(&[unicode]).unwrap(), Value::bool(false));
        unsafe {
            drop(Box::from_raw(
                unicode.as_object_ptr().unwrap() as *mut StringObject
            ));
        }
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
    fn test_resolve_set_method_returns_builtin_for_mutation_and_membership_surface() {
        for name in [
            "add",
            "remove",
            "discard",
            "pop",
            "clear",
            "update",
            "difference_update",
            "intersection_update",
            "symmetric_difference_update",
            "copy",
            "union",
            "intersection",
            "difference",
            "symmetric_difference",
            "isdisjoint",
            "issubset",
            "issuperset",
            "__contains__",
        ] {
            let method = resolve_set_method(TypeId::SET, name)
                .unwrap_or_else(|| panic!("set.{name} should resolve"));
            assert!(
                method.method.as_object_ptr().is_some(),
                "set.{name} should be heap backed"
            );
            assert!(
                !method.is_descriptor,
                "set.{name} should not be a descriptor"
            );
        }

        for name in [
            "union",
            "intersection",
            "difference",
            "symmetric_difference",
            "isdisjoint",
            "issubset",
            "issuperset",
            "copy",
            "__contains__",
        ] {
            let method = resolve_set_method(TypeId::FROZENSET, name)
                .unwrap_or_else(|| panic!("frozenset.{name} should resolve"));
            assert!(method.method.as_object_ptr().is_some());
            assert!(!method.is_descriptor);
        }
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
    fn test_set_remove_discard_and_clear_follow_python_mutation_rules() {
        let set = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let ptr = Box::into_raw(Box::new(set));
        let value = Value::object_ptr(ptr as *const ());

        assert!(
            set_remove(&[value, Value::int(1).unwrap()])
                .expect("remove should succeed")
                .is_none()
        );
        assert!(!unsafe { &*ptr }.contains(Value::int(1).unwrap()));

        assert!(
            set_discard(&[value, Value::int(9).unwrap()])
                .expect("discard should ignore missing values")
                .is_none()
        );

        let error = set_remove(&[value, Value::int(9).unwrap()]).expect_err("remove should fail");
        assert!(matches!(error, BuiltinError::KeyError(_)));

        assert!(set_clear(&[value]).expect("clear should succeed").is_none());
        assert!(unsafe { &*ptr }.is_empty());

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
    fn test_set_update_family_consumes_iterables_with_vm_support() {
        let set = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let ptr = Box::into_raw(Box::new(set));
        let value = Value::object_ptr(ptr as *const ());
        let mut vm = VirtualMachine::new();

        let update_items = to_object_value(ListObject::from_slice(&[
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));
        set_update_with_vm(&mut vm, &[value, update_items]).expect("update should work");
        assert!(unsafe { &*ptr }.contains(Value::int(3).unwrap()));

        let difference_items = to_object_value(TupleObject::from_slice(&[
            Value::int(2).unwrap(),
            Value::int(7).unwrap(),
        ]));
        set_difference_update_with_vm(&mut vm, &[value, difference_items])
            .expect("difference_update should work");
        assert!(!unsafe { &*ptr }.contains(Value::int(2).unwrap()));

        let intersection_items = to_object_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(9).unwrap(),
        ]));
        set_intersection_update_with_vm(&mut vm, &[value, intersection_items])
            .expect("intersection_update should work");
        assert!(unsafe { &*ptr }.contains(Value::int(1).unwrap()));
        assert!(!unsafe { &*ptr }.contains(Value::int(3).unwrap()));

        let symmetric_items = to_object_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(5).unwrap(),
        ]));
        set_symmetric_difference_update_with_vm(&mut vm, &[value, symmetric_items])
            .expect("symmetric_difference_update should work");
        assert!(!unsafe { &*ptr }.contains(Value::int(1).unwrap()));
        assert!(unsafe { &*ptr }.contains(Value::int(5).unwrap()));

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_set_functional_methods_accept_iterables_and_preserve_receiver_type() {
        let set = SetObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let set_ptr = Box::into_raw(Box::new(set));
        let set_value = Value::object_ptr(set_ptr as *const ());
        let mut vm = VirtualMachine::new();

        let union_items = to_object_value(ListObject::from_slice(&[
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));
        let union =
            set_union_with_vm(&mut vm, &[set_value, union_items]).expect("set.union should work");
        let union_ptr = union
            .as_object_ptr()
            .expect("set.union should return a set") as *mut SetObject;
        assert_eq!(unsafe { &*set_ptr }.len(), 2);
        assert!(unsafe { &*union_ptr }.contains(Value::int(3).unwrap()));

        let difference_items = to_object_value(TupleObject::from_slice(&[Value::int(2).unwrap()]));
        let difference = set_difference_with_vm(&mut vm, &[set_value, difference_items])
            .expect("set.difference should work");
        let difference_ptr = difference
            .as_object_ptr()
            .expect("set.difference should return a set")
            as *mut SetObject;
        assert!(unsafe { &*difference_ptr }.contains(Value::int(1).unwrap()));
        assert!(!unsafe { &*difference_ptr }.contains(Value::int(2).unwrap()));

        let subset_items = to_object_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(9).unwrap(),
        ]));
        assert_eq!(
            set_issubset_with_vm(&mut vm, &[set_value, subset_items]).unwrap(),
            Value::bool(true)
        );

        unsafe {
            drop(Box::from_raw(union_ptr));
            drop(Box::from_raw(difference_ptr));
            drop(Box::from_raw(set_ptr));
        }

        let mut frozen = SetObject::from_slice(&[Value::int(4).unwrap(), Value::int(5).unwrap()]);
        frozen.header.type_id = TypeId::FROZENSET;
        let frozen_ptr = Box::into_raw(Box::new(frozen));
        let frozen_value = Value::object_ptr(frozen_ptr as *const ());
        let frozen_union_items = to_object_value(ListObject::from_slice(&[Value::int(6).unwrap()]));
        let frozen_union = frozenset_union_with_vm(&mut vm, &[frozen_value, frozen_union_items])
            .expect("frozenset.union should work");
        let frozen_union_ptr = frozen_union
            .as_object_ptr()
            .expect("frozenset.union should return a frozenset");
        assert_eq!(
            unsafe { &*(frozen_union_ptr as *const ObjectHeader) }.type_id,
            TypeId::FROZENSET
        );
        assert!(
            unsafe { &*(frozen_union_ptr as *const SetObject) }.contains(Value::int(6).unwrap())
        );

        unsafe {
            drop(Box::from_raw(frozen_union_ptr as *mut SetObject));
            drop(Box::from_raw(frozen_ptr));
        }
    }

    #[test]
    fn test_set_membership_and_mutation_methods_reject_unhashable_values() {
        let set = SetObject::new();
        let set_ptr = Box::into_raw(Box::new(set));
        let set_value = Value::object_ptr(set_ptr as *const ());
        let key = to_object_value(ListObject::from_slice(&[Value::int(1).unwrap()]));

        for err in [
            set_add(&[set_value, key]).unwrap_err(),
            set_contains(&[set_value, key]).unwrap_err(),
            set_discard(&[set_value, key]).unwrap_err(),
        ] {
            assert!(err.to_string().contains("unhashable type: 'list'"));
        }

        unsafe {
            drop(Box::from_raw(
                key.as_object_ptr().unwrap() as *mut ListObject
            ));
            drop(Box::from_raw(set_ptr));
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
    fn test_resolve_bytes_method_returns_builtin_for_decode() {
        let decode = resolve_bytes_method("decode").expect("bytes.decode should resolve");
        assert!(decode.method.as_object_ptr().is_some());
        assert!(!decode.is_descriptor);
    }

    #[test]
    fn test_resolve_bytes_method_returns_builtin_for_affix_checks() {
        for name in [
            "startswith",
            "endswith",
            "upper",
            "lower",
            "strip",
            "lstrip",
            "rstrip",
            "translate",
            "join",
            "find",
            "rfind",
            "index",
            "rindex",
            "count",
        ] {
            let method =
                resolve_bytes_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
            assert!(method.method.as_object_ptr().is_some());
            assert!(!method.is_descriptor);
        }
    }

    #[test]
    fn test_resolve_bytearray_method_returns_builtin_for_copy() {
        let copy = resolve_bytearray_method("copy").expect("bytearray.copy should resolve");
        assert!(copy.method.as_object_ptr().is_some());
        assert!(!copy.is_descriptor);
    }

    #[test]
    fn test_resolve_bytearray_method_returns_builtin_for_extend() {
        let extend = resolve_bytearray_method("extend").expect("bytearray.extend should resolve");
        assert!(extend.method.as_object_ptr().is_some());
        assert!(!extend.is_descriptor);
    }

    #[test]
    fn test_resolve_bytearray_method_returns_builtin_for_decode() {
        let decode = resolve_bytearray_method("decode").expect("bytearray.decode should resolve");
        assert!(decode.method.as_object_ptr().is_some());
        assert!(!decode.is_descriptor);
    }

    #[test]
    fn test_resolve_bytearray_method_returns_builtin_for_affix_checks() {
        for name in [
            "startswith",
            "endswith",
            "upper",
            "lower",
            "strip",
            "lstrip",
            "rstrip",
            "translate",
            "join",
            "find",
            "rfind",
            "index",
            "rindex",
            "count",
        ] {
            let method =
                resolve_bytearray_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
            assert!(method.method.as_object_ptr().is_some());
            assert!(!method.is_descriptor);
        }
    }

    #[test]
    fn test_resolve_iterator_method_returns_builtin_for_iter_and_next() {
        for name in ["__iter__", "__next__", "__length_hint__"] {
            let method =
                resolve_iterator_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
            assert!(method.method.as_object_ptr().is_some());
            assert!(!method.is_descriptor);
        }
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
    fn test_bytearray_extend_appends_bytes_and_integer_iterables() {
        let bytearray = BytesObject::bytearray_from_slice(b"ab");
        let ptr = Box::into_raw(Box::new(bytearray));
        let value = Value::object_ptr(ptr as *const ());
        let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"cd")));
        let bytes_value = Value::object_ptr(bytes_ptr as *const ());
        let iter_value = iterator_to_value(IteratorObject::from_values(vec![
            Value::int(255).unwrap(),
            Value::bool(true),
        ]));
        let mut vm = VirtualMachine::new();

        assert!(
            bytearray_extend_with_vm(&mut vm, &[value, bytes_value])
                .expect("bytearray.extend(bytes) should work")
                .is_none()
        );
        assert!(
            bytearray_extend_with_vm(&mut vm, &[value, iter_value])
                .expect("bytearray.extend(iterable) should work")
                .is_none()
        );
        assert_eq!(unsafe { &*ptr }.as_bytes(), b"abcd\xff\x01");

        unsafe {
            drop(Box::from_raw(bytes_ptr));
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_bytearray_extend_self_duplicates_once() {
        let bytearray = BytesObject::bytearray_from_slice(b"xy");
        let ptr = Box::into_raw(Box::new(bytearray));
        let value = Value::object_ptr(ptr as *const ());
        let mut vm = VirtualMachine::new();

        bytearray_extend_with_vm(&mut vm, &[value, value]).expect("self extend should work");
        assert_eq!(unsafe { &*ptr }.as_bytes(), b"xyxy");

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_bytearray_extend_validates_integer_range() {
        let bytearray = BytesObject::bytearray_from_slice(b"");
        let ptr = Box::into_raw(Box::new(bytearray));
        let value = Value::object_ptr(ptr as *const ());
        let iter_value = iterator_to_value(IteratorObject::from_values(vec![
            Value::int(1).unwrap(),
            Value::int(256).unwrap(),
        ]));
        let mut vm = VirtualMachine::new();

        let error = bytearray_extend_with_vm(&mut vm, &[value, iter_value])
            .expect_err("out-of-range byte should fail");
        assert_eq!(
            error.to_string(),
            "ValueError: byte must be in range(0, 256)"
        );
        assert_eq!(unsafe { &*ptr }.as_bytes(), b"");

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_bytes_decode_supports_utf8_surrogatepass() {
        let bytes = BytesObject::from_slice(&[0x41, 0xED, 0xB2, 0x80]);
        let ptr = Box::into_raw(Box::new(bytes));
        let value = Value::object_ptr(ptr as *const ());

        let decoded = bytes_decode(&[
            value,
            Value::string(intern("utf-8")),
            Value::string(intern("surrogatepass")),
        ])
        .expect("bytes.decode should support surrogatepass");

        let expected =
            prism_core::python_unicode::encode_python_code_point(0xDC80).expect("surrogate");
        assert_eq!(string_value(decoded), format!("A{expected}"));

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_bytearray_decode_supports_utf8_surrogatepass() {
        let bytearray = BytesObject::bytearray_from_slice(&[0x41, 0xED, 0xB2, 0x80]);
        let ptr = Box::into_raw(Box::new(bytearray));
        let value = Value::object_ptr(ptr as *const ());

        let decoded = bytearray_decode(&[
            value,
            Value::string(intern("utf-8")),
            Value::string(intern("surrogatepass")),
        ])
        .expect("bytearray.decode should support surrogatepass");

        let expected =
            prism_core::python_unicode::encode_python_code_point(0xDC80).expect("surrogate");
        assert_eq!(string_value(decoded), format!("A{expected}"));

        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_bytes_and_bytearray_case_methods_are_ascii_only_and_preserve_type() {
        let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"aZ09\xff")));
        let bytes_value = Value::object_ptr(bytes_ptr as *const ());
        let upper = bytes_upper(&[bytes_value]).expect("bytes.upper should work");
        let lower = bytes_lower(&[bytes_value]).expect("bytes.lower should work");

        let upper_ptr = upper.as_object_ptr().expect("upper should return bytes");
        let lower_ptr = lower.as_object_ptr().expect("lower should return bytes");
        let upper_bytes = unsafe { &*(upper_ptr as *const BytesObject) };
        let lower_bytes = unsafe { &*(lower_ptr as *const BytesObject) };
        assert!(!upper_bytes.is_bytearray());
        assert_eq!(upper_bytes.as_bytes(), b"AZ09\xff");
        assert_eq!(lower_bytes.as_bytes(), b"az09\xff");

        let bytearray_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"aZ")));
        let bytearray_value = Value::object_ptr(bytearray_ptr as *const ());
        let bytearray_upper =
            bytearray_upper(&[bytearray_value]).expect("bytearray.upper should work");
        let result_ptr = bytearray_upper
            .as_object_ptr()
            .expect("bytearray.upper should return an object");
        let result = unsafe { &*(result_ptr as *const BytesObject) };
        assert!(result.is_bytearray());
        assert_eq!(result.as_bytes(), b"AZ");

        unsafe {
            drop(Box::from_raw(result_ptr as *mut BytesObject));
            drop(Box::from_raw(bytearray_ptr));
            drop(Box::from_raw(lower_ptr as *mut BytesObject));
            drop(Box::from_raw(upper_ptr as *mut BytesObject));
            drop(Box::from_raw(bytes_ptr));
        }
    }

    #[test]
    fn test_bytes_and_bytearray_strip_methods_trim_ascii_and_custom_sets() {
        let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"\t==payload==\n")));
        let bytes_value = Value::object_ptr(bytes_ptr as *const ());
        let chars_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"=\n")));
        let chars_value = Value::object_ptr(chars_ptr as *const ());
        let bytearray_chars_ptr =
            Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"\t=")));
        let bytearray_chars_value = Value::object_ptr(bytearray_chars_ptr as *const ());

        let stripped = bytes_strip(&[bytes_value]).expect("bytes.strip should trim whitespace");
        assert_eq!(byte_values(stripped), b"==payload==");
        let stripped_ptr = stripped
            .as_object_ptr()
            .expect("bytes.strip should return bytes");

        let rstripped =
            bytes_rstrip(&[bytes_value, chars_value]).expect("bytes.rstrip should trim set");
        assert_eq!(byte_values(rstripped), b"\t==payload");
        let rstripped_ptr = rstripped
            .as_object_ptr()
            .expect("bytes.rstrip should return bytes");

        let lstripped =
            bytes_lstrip(&[bytes_value, bytearray_chars_value]).expect("bytes.lstrip should work");
        assert_eq!(byte_values(lstripped), b"payload==\n");
        let lstripped_ptr = lstripped
            .as_object_ptr()
            .expect("bytes.lstrip should return bytes");

        let bytearray_ptr =
            Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"xmutablex")));
        let bytearray_value = Value::object_ptr(bytearray_ptr as *const ());
        let bytearray_strip_chars_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"x")));
        let bytearray_strip_chars_value = Value::object_ptr(bytearray_strip_chars_ptr as *const ());
        let bytearray_stripped = bytearray_strip(&[bytearray_value, bytearray_strip_chars_value])
            .expect("bytearray.strip should trim custom bytes");
        let bytearray_stripped_ptr = bytearray_stripped
            .as_object_ptr()
            .expect("bytearray.strip should return bytearray");
        let bytearray_stripped_bytes = unsafe { &*(bytearray_stripped_ptr as *const BytesObject) };
        assert!(bytearray_stripped_bytes.is_bytearray());
        assert_eq!(bytearray_stripped_bytes.as_bytes(), b"mutable");

        let empty_chars_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"")));
        let empty_chars_value = Value::object_ptr(empty_chars_ptr as *const ());
        let unchanged =
            bytes_strip(&[bytes_value, empty_chars_value]).expect("empty strip set is a no-op");
        assert_eq!(unchanged, bytes_value);

        unsafe {
            drop(Box::from_raw(empty_chars_ptr));
            drop(Box::from_raw(bytearray_stripped_ptr as *mut BytesObject));
            drop(Box::from_raw(bytearray_strip_chars_ptr));
            drop(Box::from_raw(bytearray_ptr));
            drop(Box::from_raw(lstripped_ptr as *mut BytesObject));
            drop(Box::from_raw(rstripped_ptr as *mut BytesObject));
            drop(Box::from_raw(stripped_ptr as *mut BytesObject));
            drop(Box::from_raw(bytearray_chars_ptr));
            drop(Box::from_raw(chars_ptr));
            drop(Box::from_raw(bytes_ptr));
        }
    }

    #[test]
    fn test_bytes_and_bytearray_translate_apply_table_and_delete_set() {
        let mut table = (0_u8..=u8::MAX).collect::<Vec<_>>();
        table[usize::from(b'a')] = b'x';
        let table_ptr = Box::into_raw(Box::new(BytesObject::from_vec(table)));
        let table_value = Value::object_ptr(table_ptr as *const ());
        let delete_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"b")));
        let delete_value = Value::object_ptr(delete_ptr as *const ());
        let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"abcabc")));
        let bytes_value = Value::object_ptr(bytes_ptr as *const ());

        let translated = bytes_translate(&[bytes_value, table_value, delete_value])
            .expect("bytes.translate should apply a 256-byte table and deletions");
        assert_eq!(byte_values(translated), b"xcxc");
        let translated_ptr = translated
            .as_object_ptr()
            .expect("bytes.translate should return bytes");

        let deleted =
            bytes_translate(&[bytes_value, Value::none(), delete_value]).expect("delete-only mode");
        assert_eq!(byte_values(deleted), b"acac");
        let deleted_ptr = deleted
            .as_object_ptr()
            .expect("bytes.translate delete-only should return bytes");

        let unchanged = bytes_translate(&[bytes_value, Value::none()])
            .expect("identity table with no delete set should be a no-op");
        assert_eq!(unchanged, bytes_value);

        let bytearray_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"abba")));
        let bytearray_value = Value::object_ptr(bytearray_ptr as *const ());
        let bytearray_translated =
            bytearray_translate(&[bytearray_value, table_value, delete_value])
                .expect("bytearray.translate should preserve bytearray type");
        let bytearray_translated_ptr = bytearray_translated
            .as_object_ptr()
            .expect("bytearray.translate should return bytearray");
        let bytearray_translated_bytes =
            unsafe { &*(bytearray_translated_ptr as *const BytesObject) };
        assert!(bytearray_translated_bytes.is_bytearray());
        assert_eq!(bytearray_translated_bytes.as_bytes(), b"xx");

        unsafe {
            drop(Box::from_raw(bytearray_translated_ptr as *mut BytesObject));
            drop(Box::from_raw(bytearray_ptr));
            drop(Box::from_raw(deleted_ptr as *mut BytesObject));
            drop(Box::from_raw(translated_ptr as *mut BytesObject));
            drop(Box::from_raw(bytes_ptr));
            drop(Box::from_raw(delete_ptr));
            drop(Box::from_raw(table_ptr));
        }
    }

    #[test]
    fn test_bytes_and_bytearray_join_iterables_preserve_receiver_type() {
        let sep_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b",")));
        let sep_value = Value::object_ptr(sep_ptr as *const ());
        let left_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"a")));
        let left_value = Value::object_ptr(left_ptr as *const ());
        let right_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"b")));
        let right_value = Value::object_ptr(right_ptr as *const ());
        let list_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[left_value, right_value])));
        let list_value = Value::object_ptr(list_ptr as *const ());
        let mut vm = VirtualMachine::new();

        let joined = bytes_join_with_vm(&mut vm, &[sep_value, list_value])
            .expect("bytes.join should consume bytes-like iterable");
        let joined_ptr = joined
            .as_object_ptr()
            .expect("bytes.join should return bytes");
        let joined_bytes = unsafe { &*(joined_ptr as *const BytesObject) };
        assert!(!joined_bytes.is_bytearray());
        assert_eq!(joined_bytes.as_bytes(), b"a,b");

        let bytearray_sep_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"|")));
        let bytearray_sep_value = Value::object_ptr(bytearray_sep_ptr as *const ());
        let bytearray_joined = bytearray_join_with_vm(&mut vm, &[bytearray_sep_value, list_value])
            .expect("bytearray.join should consume bytes-like iterable");
        let bytearray_joined_ptr = bytearray_joined
            .as_object_ptr()
            .expect("bytearray.join should return bytearray");
        let bytearray_joined_bytes = unsafe { &*(bytearray_joined_ptr as *const BytesObject) };
        assert!(bytearray_joined_bytes.is_bytearray());
        assert_eq!(bytearray_joined_bytes.as_bytes(), b"a|b");

        unsafe {
            drop(Box::from_raw(bytearray_joined_ptr as *mut BytesObject));
            drop(Box::from_raw(bytearray_sep_ptr));
            drop(Box::from_raw(joined_ptr as *mut BytesObject));
            drop(Box::from_raw(list_ptr));
            drop(Box::from_raw(right_ptr));
            drop(Box::from_raw(left_ptr));
            drop(Box::from_raw(sep_ptr));
        }
    }

    #[test]
    fn test_bytes_affix_methods_accept_bytes_like_prefixes_and_bounds() {
        let bytes = BytesObject::from_slice(b"traceback.py");
        let bytes_ptr = Box::into_raw(Box::new(bytes));
        let bytes_value = Value::object_ptr(bytes_ptr as *const ());

        let prefix = BytesObject::bytearray_from_slice(b"trace");
        let prefix_ptr = Box::into_raw(Box::new(prefix));
        let prefix_value = Value::object_ptr(prefix_ptr as *const ());

        let suffix = BytesObject::from_slice(b".py");
        let suffix_ptr = Box::into_raw(Box::new(suffix));
        let suffix_value = Value::object_ptr(suffix_ptr as *const ());

        let prefixes = TupleObject::from_slice(&[prefix_value, suffix_value]);
        let prefixes_ptr = Box::into_raw(Box::new(prefixes));
        let prefixes_value = Value::object_ptr(prefixes_ptr as *const ());

        assert_eq!(
            bytes_startswith(&[bytes_value, prefixes_value]).unwrap(),
            Value::bool(true)
        );
        assert_eq!(
            bytes_endswith(&[
                bytes_value,
                suffix_value,
                Value::int(0).unwrap(),
                Value::int(12).unwrap(),
            ])
            .unwrap(),
            Value::bool(true)
        );

        let short_prefix = BytesObject::from_slice(b"tr");
        let short_prefix_ptr = Box::into_raw(Box::new(short_prefix));
        let short_prefix_value = Value::object_ptr(short_prefix_ptr as *const ());
        assert_eq!(
            bytearray_startswith(&[prefix_value, short_prefix_value]).unwrap(),
            Value::bool(true)
        );

        let error = bytes_startswith(&[bytes_value, Value::int(1).unwrap()])
            .expect_err("non-bytes affix should fail");
        assert!(
            error
                .to_string()
                .contains("startswith first arg must be bytes or a tuple of bytes")
        );

        unsafe {
            drop(Box::from_raw(short_prefix_ptr));
            drop(Box::from_raw(prefixes_ptr));
            drop(Box::from_raw(suffix_ptr));
            drop(Box::from_raw(prefix_ptr));
            drop(Box::from_raw(bytes_ptr));
        }
    }

    #[test]
    fn test_bytes_and_bytearray_search_methods_match_python_bounds_and_needles() {
        let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"abaaba")));
        let bytes_value = Value::object_ptr(bytes_ptr as *const ());
        let needle_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"ba")));
        let needle_value = Value::object_ptr(needle_ptr as *const ());
        let empty_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"")));
        let empty_value = Value::object_ptr(empty_ptr as *const ());
        let bytearray_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(&[
            0, 1, 2, 1, 0,
        ])));
        let bytearray_value = Value::object_ptr(bytearray_ptr as *const ());

        assert_eq!(
            bytes_find_method(&[bytes_value, needle_value])
                .expect("bytes.find should work")
                .as_int(),
            Some(1)
        );
        assert_eq!(
            bytes_rfind_method(&[bytes_value, needle_value])
                .expect("bytes.rfind should work")
                .as_int(),
            Some(4)
        );
        assert_eq!(
            bytes_count_method(&[bytes_value, needle_value])
                .expect("bytes.count should work")
                .as_int(),
            Some(2)
        );
        assert_eq!(
            bytes_find_method(&[
                bytes_value,
                needle_value,
                Value::int(4).unwrap(),
                Value::int(2).unwrap(),
            ])
            .expect("empty search interval should not be reordered")
            .as_int(),
            Some(-1)
        );
        assert_eq!(
            bytes_find_method(&[
                bytes_value,
                empty_value,
                Value::int(3).unwrap(),
                Value::int(3).unwrap(),
            ])
            .expect("empty needle should match a valid zero-length interval")
            .as_int(),
            Some(3)
        );
        assert_eq!(
            bytes_rfind_method(&[
                bytes_value,
                empty_value,
                Value::int(0).unwrap(),
                Value::int(4).unwrap(),
            ])
            .expect("rfind empty needle should return the clamped end")
            .as_int(),
            Some(4)
        );
        assert_eq!(
            bytearray_find(&[
                bytearray_value,
                Value::int(1).unwrap(),
                Value::int(2).unwrap(),
            ])
            .expect("bytearray.find should accept an integer needle")
            .as_int(),
            Some(3)
        );
        assert_eq!(
            bytearray_count(&[bytearray_value, Value::int(1).unwrap()])
                .expect("bytearray.count should accept an integer needle")
                .as_int(),
            Some(2)
        );

        let missing = bytes_index_method(&[bytes_value, Value::int(b'z' as i64).unwrap()])
            .expect_err("bytes.index should raise for a missing subsection");
        assert!(missing.to_string().contains("subsection not found"));

        let out_of_range = bytearray_find(&[bytearray_value, Value::int(256).unwrap()])
            .expect_err("integer needles must be valid bytes");
        assert_eq!(
            out_of_range.to_string(),
            "ValueError: byte must be in range(0, 256)"
        );

        let wrong_type =
            bytes_find_method(&[bytes_value, Value::string(intern("ba"))]).unwrap_err();
        assert!(
            wrong_type
                .to_string()
                .contains("argument should be integer or bytes-like object, not 'str'")
        );

        unsafe {
            drop(Box::from_raw(bytearray_ptr));
            drop(Box::from_raw(empty_ptr));
            drop(Box::from_raw(needle_ptr));
            drop(Box::from_raw(bytes_ptr));
        }
    }

    #[test]
    fn test_iterator_iter_and_next_follow_python_protocol() {
        let iter_value = iterator_to_value(IteratorObject::from_values(vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]));

        assert_eq!(iterator_iter(&[iter_value]).unwrap(), iter_value);
        assert_eq!(iterator_next(&[iter_value]).unwrap().as_int(), Some(1));
        assert_eq!(iterator_next(&[iter_value]).unwrap().as_int(), Some(2));
        assert!(matches!(
            iterator_next(&[iter_value]),
            Err(BuiltinError::StopIteration)
        ));
    }

    #[test]
    fn test_iterator_length_hint_reports_remaining_items() {
        let iter_value = iterator_to_value(IteratorObject::from_values(vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]));

        assert_eq!(
            iterator_length_hint(&[iter_value]).unwrap().as_int(),
            Some(3)
        );
        assert_eq!(iterator_next(&[iter_value]).unwrap().as_int(), Some(1));
        assert_eq!(
            iterator_length_hint(&[iter_value]).unwrap().as_int(),
            Some(2)
        );
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

        let pop_err = list_pop(&[list_value, Value::int(1).unwrap(), Value::int(2).unwrap()])
            .expect_err("pop should reject multiple indices");
        assert_eq!(
            pop_err.to_string(),
            "TypeError: list.pop() takes at most 1 argument (2 given)"
        );

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
    fn test_resolve_generator_method_returns_builtin_for_throw() {
        let throw = resolve_generator_method("throw").expect("throw should resolve");
        assert!(throw.method.as_object_ptr().is_some());
        assert!(!throw.is_descriptor);
    }

    #[test]
    fn test_resolve_regex_pattern_method_returns_builtin_for_findall() {
        let findall = resolve_regex_pattern_method("findall").expect("findall should resolve");
        assert!(findall.method.as_object_ptr().is_some());
        assert!(!findall.is_descriptor);
    }

    #[test]
    fn test_resolve_regex_pattern_method_returns_builtin_for_finditer() {
        let finditer = resolve_regex_pattern_method("finditer").expect("finditer should resolve");
        assert!(finditer.method.as_object_ptr().is_some());
        assert!(!finditer.is_descriptor);
    }

    #[test]
    fn test_resolve_regex_pattern_method_returns_builtin_for_sub() {
        let sub = resolve_regex_pattern_method("sub").expect("sub should resolve");
        assert!(sub.method.as_object_ptr().is_some());
        assert!(!sub.is_descriptor);
    }

    #[test]
    fn test_resolve_regex_match_method_returns_builtin_for_getitem() {
        let getitem =
            resolve_regex_match_method("__getitem__").expect("__getitem__ should resolve");
        assert!(getitem.method.as_object_ptr().is_some());
        assert!(!getitem.is_descriptor);
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
        let get = resolve_property_method("__get__").expect("__get__ should resolve");
        let set = resolve_property_method("__set__").expect("__set__ should resolve");
        let delete = resolve_property_method("__delete__").expect("__delete__ should resolve");
        let getter = resolve_property_method("getter").expect("getter should resolve");
        let setter = resolve_property_method("setter").expect("setter should resolve");
        let deleter = resolve_property_method("deleter").expect("deleter should resolve");
        assert!(get.method.as_object_ptr().is_some());
        assert!(set.method.as_object_ptr().is_some());
        assert!(delete.method.as_object_ptr().is_some());
        assert!(getter.method.as_object_ptr().is_some());
        assert!(setter.method.as_object_ptr().is_some());
        assert!(deleter.method.as_object_ptr().is_some());
        assert!(!get.is_descriptor);
        assert!(!set.is_descriptor);
        assert!(!delete.is_descriptor);
        assert!(!getter.is_descriptor);
        assert!(!setter.is_descriptor);
        assert!(!deleter.is_descriptor);
    }

    #[test]
    fn test_property_dunder_methods_follow_descriptor_protocol() {
        let getter_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
            Arc::from("property_test.getter"),
            property_echo_getter,
        )));
        let setter_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
            Arc::from("property_test.setter"),
            property_accepting_setter,
        )));
        let deleter_ptr = Box::into_raw(Box::new(BuiltinFunctionObject::new(
            Arc::from("property_test.deleter"),
            property_accepting_deleter,
        )));
        let getter = Value::object_ptr(getter_ptr as *const ());
        let setter = Value::object_ptr(setter_ptr as *const ());
        let deleter = Value::object_ptr(deleter_ptr as *const ());
        let property_ptr = Box::into_raw(Box::new(PropertyDescriptor::new_full(
            Some(getter),
            Some(setter),
            Some(deleter),
            None,
        )));
        let property_value = Value::object_ptr(property_ptr as *const ());
        let mut vm = VirtualMachine::new();

        assert_eq!(
            property_get(&mut vm, &[property_value, Value::none(), Value::none()])
                .expect("__get__(None, owner) should return the property object"),
            property_value
        );
        let instance = Value::int(99).unwrap();
        assert_eq!(
            property_get(&mut vm, &[property_value, instance])
                .expect("__get__(instance) should invoke fget"),
            instance
        );
        assert!(
            property_set(&mut vm, &[property_value, instance, Value::int(7).unwrap()])
                .expect("__set__ should invoke fset")
                .is_none()
        );
        assert!(
            property_delete(&mut vm, &[property_value, instance])
                .expect("__delete__ should invoke fdel")
                .is_none()
        );

        unsafe {
            drop(Box::from_raw(property_ptr));
            drop(Box::from_raw(deleter_ptr));
            drop(Box::from_raw(setter_ptr));
            drop(Box::from_raw(getter_ptr));
        }
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
        let code = Arc::new(prism_code::CodeObject::new("g", "<test>"));
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
