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
    try_length_hint, value_to_iterator,
};
use crate::error::RuntimeError;
use crate::error::RuntimeErrorKind;
use crate::ops::calls::invoke_callable_value;
use crate::ops::comparison::eq_result;
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
use prism_runtime::types::iter::{IteratorEmptyIterable, IteratorObject, IteratorReduction};
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

mod binary;
mod sets;
mod text;

pub use binary::{resolve_bytearray_method, resolve_bytes_method};
pub use sets::resolve_set_method;
use text::char_index_to_byte_offset;
pub use text::resolve_str_method;

static LIST_ITER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.__iter__"), list_iter));
static LIST_LEN_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.__len__"), list_len));
static LIST_GETITEM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.__getitem__"), list_getitem));
static LIST_ADD_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.__add__"), list_add));
static LIST_IADD_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("list.__iadd__"), list_iadd_with_vm));
static LIST_IMUL_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("list.__imul__"), list_imul));
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
static DICT_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("dict.update"), dict_update_with_vm_kw)
});
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
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("iterator.__next__"), iterator_next));
static ITERATOR_LENGTH_HINT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("iterator.__length_hint__"), iterator_length_hint)
});
static ITERATOR_REDUCE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("iterator.__reduce__"), iterator_reduce)
});
static ITERATOR_SETSTATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("iterator.__setstate__"), iterator_setstate)
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
static PROPERTY_SET_NAME_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("property.__set_name__"), property_set_name)
});
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
        "__add__" => Some(CachedMethod::simple(builtin_method_value(&LIST_ADD_METHOD))),
        "__iadd__" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_IADD_METHOD,
        ))),
        "__imul__" => Some(CachedMethod::simple(builtin_method_value(
            &LIST_IMUL_METHOD,
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
        "__reduce__" => Some(CachedMethod::simple(builtin_method_value(
            &ITERATOR_REDUCE_METHOD,
        ))),
        "__setstate__" => Some(CachedMethod::simple(builtin_method_value(
            &ITERATOR_SETSTATE_METHOD,
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
        "__set_name__" => Some(CachedMethod::simple(builtin_method_value(
            &PROPERTY_SET_NAME_METHOD,
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
    expect_list_ref(args[0], "__iter__")?;
    Ok(iterator_to_value(IteratorObject::from_list(args[0])))
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
fn list_add(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "__add__", args, 1)?;
    let left = expect_list_ref(args[0], "__add__")?;
    let right = args[1]
        .as_object_ptr()
        .and_then(list_storage_ref_from_ptr)
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "can only concatenate list (not \"{}\") to list",
                args[1].type_name()
            ))
        })?;
    Ok(to_object_value(left.concat(right)))
}

#[inline]
fn list_iadd_with_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "__iadd__", args, 1)?;

    let items = collect_iterable_values_with_vm(vm, args[1])?;
    let list = expect_list_mut(args[0], "__iadd__")?;
    list.extend(items);
    Ok(args[0])
}

#[inline]
fn list_imul(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("list", "__imul__", args, 1)?;
    let repeat = repeat_count_for_list(expect_integer_like_index(args[1])?)?;
    let list = expect_list_mut(args[0], "__imul__")?;
    list.repeat_in_place(repeat).ok_or_else(|| {
        BuiltinError::Raised(RuntimeError::memory_error("repeated sequence is too long"))
    })?;
    Ok(args[0])
}

#[inline]
fn repeat_count_for_list(count: i64) -> Result<usize, BuiltinError> {
    if count <= 0 {
        return Ok(0);
    }

    usize::try_from(count).map_err(|_| {
        BuiltinError::Raised(RuntimeError::memory_error("repeated sequence is too long"))
    })
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
    dict_update_with_vm_kw(vm, args, &[])
}

#[inline]
fn dict_update_with_vm_kw(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        let given = args.len().saturating_sub(1);
        return Err(BuiltinError::TypeError(format!(
            "dict.update() takes at most 1 argument ({given} given)"
        )));
    }

    if args.len() == 1 && keywords.is_empty() {
        expect_dict_receiver(args[0], "update")?;
        return Ok(Value::none());
    }

    let entries = if args.len() == 2 {
        collect_dict_update_entries(vm, args[1])?
    } else {
        Vec::new()
    };
    let dict = expect_dict_mut(args[0], "update")?;

    for (key, value) in entries {
        ensure_hashable(key)?;
        dict.set(key, value);
    }
    for &(name, value) in keywords {
        dict.set(Value::string(intern(name)), value);
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

fn iterator_next(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("iterator", "__next__", args, 0)?;
    match next_step(vm, args[0]).map_err(runtime_error_to_builtin_error)? {
        IterStep::Yielded(value) => Ok(value),
        IterStep::Exhausted => Err(BuiltinError::StopIteration),
    }
}

fn iterator_length_hint(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("iterator", "__length_hint__", args, 0)?;
    let iter = get_iterator_mut(&args[0]).ok_or_else(|| {
        BuiltinError::TypeError("'iterator' object is not an iterator".to_string())
    })?;
    Ok(Value::int(iter.size_hint().unwrap_or(0) as i64)
        .expect("iterator length hint should fit in tagged int"))
}

fn iterator_reduce(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("iterator", "__reduce__", args, 0)?;
    let reducer_builtin_name = {
        let iter = get_iterator_mut(&args[0]).ok_or_else(|| {
            BuiltinError::TypeError("'iterator' object is not an iterator".to_string())
        })?;
        iter.reduction_builtin_name()
    };
    let reducer_builtin = reduce_builtin(vm, reducer_builtin_name)?;
    let state = {
        let iter = get_iterator_mut(&args[0]).ok_or_else(|| {
            BuiltinError::TypeError("'iterator' object is not an iterator".to_string())
        })?;
        iter.reduction_state()
            .map_err(|err| BuiltinError::Raised(RuntimeError::from(err)))?
    };

    match state {
        IteratorReduction::Empty => reduce_tuple(vm, reducer_builtin, &[], None),
        IteratorReduction::EmptyIterable(kind) => {
            let iterable = empty_reduce_iterable(vm, kind)?;
            reduce_tuple(vm, reducer_builtin, &[iterable], None)
        }
        IteratorReduction::Iterable { iterable, state } => {
            reduce_tuple(vm, reducer_builtin, &[iterable], state)
        }
        IteratorReduction::ReversedIterable { iterable, state } => {
            reduce_tuple(vm, reducer_builtin, &[iterable], state)
        }
        IteratorReduction::CallSentinel { callable, sentinel } => {
            reduce_tuple(vm, reducer_builtin, &[callable, sentinel], None)
        }
        IteratorReduction::RemainingValues(values) => {
            let list = alloc_heap_value(
                vm,
                ListObject::from_iter(values),
                "iterator reduce remaining values",
            )
            .map_err(runtime_error_to_builtin_error)?;
            reduce_tuple(vm, reducer_builtin, &[list], None)
        }
        IteratorReduction::RequiresVm(reason) => Err(BuiltinError::NotImplemented(
            reason.to_string(),
        )),
    }
}

fn iterator_setstate(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("iterator", "__setstate__", args, 1)?;
    let state = expect_integer_like_index(args[1])?;
    let iter = get_iterator_mut(&args[0]).ok_or_else(|| {
        BuiltinError::TypeError("'iterator' object is not an iterator".to_string())
    })?;
    iter.set_state(state);
    Ok(Value::none())
}

fn empty_reduce_iterable(
    vm: &mut VirtualMachine,
    kind: IteratorEmptyIterable,
) -> Result<Value, BuiltinError> {
    match kind {
        IteratorEmptyIterable::Tuple => alloc_heap_value(
            vm,
            TupleObject::empty(),
            "iterator reduce empty tuple",
        )
        .map_err(runtime_error_to_builtin_error),
        IteratorEmptyIterable::List => alloc_heap_value(
            vm,
            ListObject::new(),
            "iterator reduce empty list",
        )
        .map_err(runtime_error_to_builtin_error),
        IteratorEmptyIterable::String => Ok(Value::string(intern(""))),
    }
}

fn reduce_builtin(vm: &mut VirtualMachine, name: &str) -> Result<Value, BuiltinError> {
    if let Some(module) = vm.cached_module("builtins") {
        if let Some(value) = lookup_dict_string_key_vm(vm, module.dict_value(), name)? {
            return Ok(value);
        }
        if let Some(value) = module.get_attr(name) {
            return Ok(value);
        }
    }

    vm.builtin_value(name)
        .ok_or_else(|| BuiltinError::AttributeError(format!("builtin '{}' is not available", name)))
}

fn lookup_dict_string_key_vm(
    vm: &mut VirtualMachine,
    dict_value: Value,
    name: &str,
) -> Result<Option<Value>, BuiltinError> {
    let target = Value::string(intern(name));
    let Some(ptr) = dict_value.as_object_ptr() else {
        return Ok(None);
    };
    if crate::ops::objects::extract_type_id(ptr) != TypeId::DICT {
        return Ok(None);
    }

    let dict = unsafe { &*(ptr as *const DictObject) };
    if let Some(value) = dict.get(target) {
        return Ok(Some(value));
    }

    let entries = dict.iter().collect::<Vec<_>>();
    for (key, value) in entries {
        if eq_result(vm, key, target).map_err(runtime_error_to_builtin_error)? {
            return Ok(Some(value));
        }
    }

    Ok(None)
}

fn reduce_tuple(
    vm: &mut VirtualMachine,
    callable: Value,
    call_args: &[Value],
    state: Option<i64>,
) -> Result<Value, BuiltinError> {
    let args_tuple = alloc_heap_value(
        vm,
        TupleObject::from_slice(call_args),
        "iterator reduce args tuple",
    )
    .map_err(runtime_error_to_builtin_error)?;

    let values = match state {
        Some(state) => {
            let state_value = bigint_to_value(state.into());
            [callable, args_tuple, state_value].to_vec()
        }
        None => [callable, args_tuple].to_vec(),
    };

    alloc_heap_value(
        vm,
        TupleObject::from_vec(values),
        "iterator reduce result tuple",
    )
    .map_err(runtime_error_to_builtin_error)
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
    let capacity = try_length_hint(vm, iterable, 0).map_err(runtime_error_to_builtin_error)?;
    let iterator = ensure_iterator_value(vm, iterable).map_err(runtime_error_to_builtin_error)?;
    let mut values = Vec::new();
    values.try_reserve(capacity).map_err(|_| {
        BuiltinError::Raised(RuntimeError::memory_error("length hint is too large"))
    })?;

    loop {
        match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
            IterStep::Yielded(value) => values.push(value),
            IterStep::Exhausted => return Ok(values),
        }
    }
}

#[inline]
fn runtime_error_to_builtin_error(err: crate::error::RuntimeError) -> BuiltinError {
    let kind = err.kind().clone();
    match kind {
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
        _ => BuiltinError::Raised(err),
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
    let getter = select_property_accessor(descriptor.getter(), getter_override);
    let setter = select_property_accessor(descriptor.setter(), setter_override);
    let deleter = select_property_accessor(descriptor.deleter(), deleter_override);
    let refresh_getter_doc = getter_override.is_some() && descriptor.doc_from_getter();
    let doc = if refresh_getter_doc {
        getter.and_then(crate::builtins::property_doc_from_accessor)
    } else {
        descriptor.doc()
    };
    let getter_doc = if refresh_getter_doc {
        doc.is_some()
    } else {
        descriptor.doc_from_getter()
    };

    Ok(to_object_value(
        PropertyDescriptor::new_full_with_doc_source(getter, setter, deleter, doc, getter_doc),
    ))
}

#[inline]
fn select_property_accessor(
    current: Option<Value>,
    override_value: Option<Value>,
) -> Option<Value> {
    match override_value {
        Some(value) if !value.is_none() => Some(value),
        Some(_) => None,
        None => current,
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
fn property_set_name(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "__set_name__() takes 2 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }
    expect_property_receiver(args[0], "__set_name__")?;
    Ok(Value::none())
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
