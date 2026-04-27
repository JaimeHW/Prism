//! Comparison opcode handlers.
//!
//! Handles all comparison operations: <, <=, ==, !=, >, >=, is, in.

use crate::VirtualMachine;
use crate::builtins::builtin_type_object_type_id;
use crate::dispatch::ControlFlow;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::iteration::{IterStep, ensure_iterator_value, next_step};
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::ops::protocols::{
    RichCompareOp, binary_special_method, rich_compare_bool, value_type_id,
};
use crate::python_numeric::{
    complex_like_parts, float_like_value, int_like_value, is_complex_value,
};
use crate::stdlib::_warnings::emit_bool_invert_deprecation_warning;
use crate::stdlib::collections::deque::DequeObject;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use prism_code::Instruction;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::global_class_bitmap;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{
    DictViewKind, DictViewObject, MappingProxyObject, MappingProxySource, UnionTypeObject,
};
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::list::value_as_list_ref;
use prism_runtime::types::range::RangeObject;
use prism_runtime::types::simd::search::{bytes_contains, str_contains};
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::value_as_tuple_ref;
use prism_runtime::types::{DictObject, ListObject, SetObject, TupleObject};
use rustc_hash::FxHashSet;
use std::cmp::Ordering;

#[inline]
fn with_string_value<R>(value: Value, f: impl FnOnce(&str) -> R) -> Option<R> {
    let string = value_as_string_ref(value)?;
    Some(f(string.as_str()))
}

#[inline]
fn string_values_equal(a: Value, b: Value) -> Option<bool> {
    with_string_value(a, |left| with_string_value(b, |right| left == right)).flatten()
}

#[inline]
fn compare_string_values(a: Value, b: Value) -> Option<Ordering> {
    with_string_value(a, |left| with_string_value(b, |right| left.cmp(right))).flatten()
}

#[inline]
fn bytes_values_equal(a: Value, b: Value) -> Option<bool> {
    let left = value_as_bytes_ref(a)?;
    let right = value_as_bytes_ref(b)?;
    Some(left == right)
}

#[inline]
fn compare_bytes_values(a: Value, b: Value) -> Option<Ordering> {
    let left = value_as_bytes_ref(a)?;
    let right = value_as_bytes_ref(b)?;
    Some(left.cmp(right))
}

#[inline]
fn numeric_value_to_bigint(value: Value) -> Option<BigInt> {
    if let Some(boolean) = value.as_bool() {
        return Some(BigInt::from(u8::from(boolean)));
    }

    value_to_bigint(value)
}

#[inline]
fn is_numeric_value(value: Value) -> bool {
    value.as_float().is_some() || numeric_value_to_bigint(value).is_some()
}

#[inline]
fn exact_integral_value_to_bigint(value: Value) -> Option<BigInt> {
    if matches!(value_type_id(value), TypeId::BOOL | TypeId::INT) {
        numeric_value_to_bigint(value)
    } else {
        None
    }
}

#[inline]
fn is_exact_numeric_fast_type(type_id: TypeId) -> bool {
    matches!(type_id, TypeId::BOOL | TypeId::INT | TypeId::FLOAT)
}

#[inline]
fn has_numeric_subtype_operand(left: Value, right: Value) -> bool {
    (is_numeric_value(left) && !is_exact_numeric_fast_type(value_type_id(left)))
        || (is_numeric_value(right) && !is_exact_numeric_fast_type(value_type_id(right)))
}

#[inline]
fn compare_f64_to_bigint(left: f64, right: &BigInt) -> Option<Ordering> {
    compare_bigint_to_f64(right, left).map(Ordering::reverse)
}

fn compare_bigint_to_f64(left: &BigInt, right: f64) -> Option<Ordering> {
    if right.is_nan() {
        return None;
    }
    if right.is_infinite() {
        return Some(if right.is_sign_positive() {
            Ordering::Less
        } else {
            Ordering::Greater
        });
    }

    let bits = right.to_bits();
    let sign_negative = (bits >> 63) != 0;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let mantissa_bits = bits & ((1u64 << 52) - 1);

    if exponent_bits == 0 && mantissa_bits == 0 {
        return Some(left.cmp(&BigInt::zero()));
    }

    let exponent = if exponent_bits == 0 {
        -1022
    } else {
        exponent_bits - 1023
    };
    let mantissa = if exponent_bits == 0 {
        BigInt::from(mantissa_bits)
    } else {
        BigInt::from((1u64 << 52) | mantissa_bits)
    };
    let signed_mantissa = if sign_negative { -mantissa } else { mantissa };

    if exponent >= 52 {
        let shifted = signed_mantissa << ((exponent - 52) as usize);
        return Some(left.cmp(&shifted));
    }

    let shift = (52 - exponent) as usize;
    Some((left << shift).cmp(&signed_mantissa))
}

#[inline]
fn compare_numeric_values(left: Value, right: Value) -> Option<Ordering> {
    if let (Some(left_int), Some(right_int)) = (
        numeric_value_to_bigint(left),
        numeric_value_to_bigint(right),
    ) {
        return Some(left_int.cmp(&right_int));
    }

    match (left.as_float(), right.as_float()) {
        (Some(left_float), Some(right_float)) => left_float.partial_cmp(&right_float),
        (Some(left_float), None) => {
            compare_f64_to_bigint(left_float, &numeric_value_to_bigint(right)?)
        }
        (None, Some(right_float)) => {
            compare_bigint_to_f64(&numeric_value_to_bigint(left)?, right_float)
        }
        (None, None) => None,
    }
}

#[inline]
fn value_as_bytes_ref(value: Value) -> Option<&'static [u8]> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            Some(unsafe { &*(ptr as *const BytesObject) }.as_bytes())
        }
        _ => None,
    }
}

#[inline]
fn contains_bytes_value(needle: Value, haystack: &[u8]) -> Result<bool, RuntimeError> {
    if let Some(value) = int_like_value(needle) {
        let byte = u8::try_from(value)
            .map_err(|_| RuntimeError::value_error("byte must be in range(0, 256)"))?;
        return Ok(haystack.contains(&byte));
    }

    if let Some(needle_bytes) = value_as_bytes_ref(needle) {
        return Ok(bytes_contains(haystack, needle_bytes));
    }

    Err(RuntimeError::type_error(
        "'in <bytes>' requires bytes-like object or integer as left operand",
    ))
}

#[inline]
fn union_member_type_ids(value: Value) -> Option<Vec<TypeId>> {
    if value.is_none() {
        return Some(vec![TypeId::NONE]);
    }

    let ptr = value.as_object_ptr()?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::TYPE => Some(vec![builtin_type_object_type_id(ptr).unwrap_or_else(
            || {
                let class = unsafe { &*(ptr as *const PyClassObject) };
                class.class_type_id()
            },
        )]),
        TypeId::UNION => {
            let union = unsafe { &*(ptr as *const UnionTypeObject) };
            Some(union.members().to_vec())
        }
        _ => crate::stdlib::typing::typing_marker_type_id(value).map(|type_id| vec![type_id]),
    }
}

#[derive(Clone, Copy)]
enum SetRelation {
    StrictSubset,
    Subset,
    StrictSuperset,
    Superset,
}

#[derive(Clone, Copy)]
pub(crate) enum SetLikeBinaryOp {
    Intersection,
    Union,
    Difference,
    SymmetricDifference,
}

#[inline]
fn compare_set_relation(left: Value, right: Value, relation: SetRelation) -> Option<bool> {
    let left_ptr = left.as_object_ptr()?;
    let right_ptr = right.as_object_ptr()?;

    let left_type = unsafe { (*(left_ptr as *const ObjectHeader)).type_id };
    let right_type = unsafe { (*(right_ptr as *const ObjectHeader)).type_id };

    if !matches!(left_type, TypeId::SET | TypeId::FROZENSET)
        || !matches!(right_type, TypeId::SET | TypeId::FROZENSET)
    {
        return None;
    }

    let left_set = unsafe { &*(left_ptr as *const SetObject) };
    let right_set = unsafe { &*(right_ptr as *const SetObject) };

    Some(match relation {
        SetRelation::StrictSubset => {
            left_set.len() < right_set.len() && left_set.is_subset(right_set)
        }
        SetRelation::Subset => left_set.is_subset(right_set),
        SetRelation::StrictSuperset => {
            left_set.len() > right_set.len() && left_set.is_superset(right_set)
        }
        SetRelation::Superset => left_set.is_superset(right_set),
    })
}

fn compare_set_like_relation(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    relation: SetRelation,
) -> Result<Option<bool>, RuntimeError> {
    if let Some(result) = compare_dict_items_view_relation(vm, left, right, relation)? {
        return Ok(Some(result));
    }

    if !is_dict_view_set_operand(left) && !is_dict_view_set_operand(right) {
        return Ok(None);
    }

    let Some(left_set) = materialize_set_like(left)? else {
        return Ok(None);
    };
    let Some(right_set) = materialize_set_like(right)? else {
        return Ok(None);
    };

    Ok(Some(match relation {
        SetRelation::StrictSubset => {
            left_set.len() < right_set.len() && left_set.is_subset(&right_set)
        }
        SetRelation::Subset => left_set.is_subset(&right_set),
        SetRelation::StrictSuperset => {
            left_set.len() > right_set.len() && left_set.is_superset(&right_set)
        }
        SetRelation::Superset => left_set.is_superset(&right_set),
    }))
}

#[inline]
pub(crate) fn set_binary_operands(
    left: Value,
    right: Value,
) -> Option<(&'static SetObject, &'static SetObject, TypeId)> {
    let left_ptr = left.as_object_ptr()?;
    let right_ptr = right.as_object_ptr()?;

    let left_type = unsafe { (*(left_ptr as *const ObjectHeader)).type_id };
    let right_type = unsafe { (*(right_ptr as *const ObjectHeader)).type_id };
    if !matches!(left_type, TypeId::SET | TypeId::FROZENSET)
        || !matches!(right_type, TypeId::SET | TypeId::FROZENSET)
    {
        return None;
    }

    let left_set = unsafe { &*(left_ptr as *const SetObject) };
    let right_set = unsafe { &*(right_ptr as *const SetObject) };
    Some((left_set, right_set, left_type))
}

pub(crate) fn dict_view_set_binary_result(
    left: Value,
    right: Value,
    op: SetLikeBinaryOp,
) -> Result<Option<Value>, RuntimeError> {
    if !is_dict_view_set_operand(left) && !is_dict_view_set_operand(right) {
        return Ok(None);
    }

    let Some(left_set) = materialize_set_like(left)? else {
        return Ok(None);
    };
    let Some(right_set) = materialize_set_like(right)? else {
        return Ok(None);
    };

    let result = match op {
        SetLikeBinaryOp::Intersection => left_set.intersection(&right_set),
        SetLikeBinaryOp::Union => left_set.union(&right_set),
        SetLikeBinaryOp::Difference => left_set.difference(&right_set),
        SetLikeBinaryOp::SymmetricDifference => left_set.symmetric_difference(&right_set),
    };
    Ok(Some(boxed_set_result(result, TypeId::SET)))
}

#[inline]
fn is_dict_view_set_operand(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    matches!(
        unsafe { (*(ptr as *const ObjectHeader)).type_id },
        TypeId::DICT_KEYS | TypeId::DICT_ITEMS
    )
}

fn materialize_set_like(value: Value) -> Result<Option<SetObject>, RuntimeError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };

    match unsafe { (*(ptr as *const ObjectHeader)).type_id } {
        TypeId::SET | TypeId::FROZENSET => Ok(Some(unsafe { &*(ptr as *const SetObject) }.clone())),
        TypeId::DICT_KEYS | TypeId::DICT_ITEMS => {
            let view = unsafe { &*(ptr as *const DictViewObject) };
            materialize_dict_view_set(view)
        }
        _ => Ok(None),
    }
}

fn materialize_dict_view_set(view: &DictViewObject) -> Result<Option<SetObject>, RuntimeError> {
    if view.kind() == DictViewKind::Values {
        return Ok(None);
    }

    let entries = dict_view_entries(view)?;
    let mut set = SetObject::with_capacity(entries.len());
    for (key, value) in entries {
        match view.kind() {
            DictViewKind::Keys => {
                set.add(key);
            }
            DictViewKind::Items => {
                set.add(crate::alloc_managed_value(TupleObject::from_slice(&[
                    key, value,
                ])));
            }
            DictViewKind::Values => unreachable!(),
        }
    }
    Ok(Some(set))
}

fn dict_view_entries(view: &DictViewObject) -> Result<Vec<(Value, Value)>, RuntimeError> {
    let Some(ptr) = view.dict().as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "invalid dictionary view backing object",
        ));
    };

    if let Some(dict) = crate::ops::objects::dict_storage_ref_from_ptr(ptr) {
        return Ok(dict.iter().collect());
    }

    match unsafe { (*(ptr as *const ObjectHeader)).type_id } {
        TypeId::MAPPING_PROXY => {
            let proxy = unsafe { &*(ptr as *const MappingProxyObject) };
            crate::builtins::builtin_mapping_proxy_entries_static(proxy)
        }
        _ => Err(RuntimeError::type_error(
            "invalid dictionary view backing object",
        )),
    }
}

fn exact_dict_items_view(value: Value) -> Option<(*const (), &'static DictObject)> {
    let view_ptr = value.as_object_ptr()?;
    if unsafe { (*(view_ptr as *const ObjectHeader)).type_id } != TypeId::DICT_ITEMS {
        return None;
    }

    let view = unsafe { &*(view_ptr as *const DictViewObject) };
    let dict_ptr = view.dict().as_object_ptr()?;
    if unsafe { (*(dict_ptr as *const ObjectHeader)).type_id } != TypeId::DICT {
        return None;
    }

    Some((dict_ptr, unsafe { &*(dict_ptr as *const DictObject) }))
}

fn compare_dict_items_view_relation(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    relation: SetRelation,
) -> Result<Option<bool>, RuntimeError> {
    let Some((_, left_dict)) = exact_dict_items_view(left) else {
        return Ok(None);
    };
    let Some((_, right_dict)) = exact_dict_items_view(right) else {
        return Ok(None);
    };

    Ok(Some(match relation {
        SetRelation::StrictSubset => {
            left_dict.len() < right_dict.len()
                && dict_items_subset_result(vm, left_dict, right_dict)?
        }
        SetRelation::Subset => dict_items_subset_result(vm, left_dict, right_dict)?,
        SetRelation::StrictSuperset => {
            left_dict.len() > right_dict.len()
                && dict_items_subset_result(vm, right_dict, left_dict)?
        }
        SetRelation::Superset => dict_items_subset_result(vm, right_dict, left_dict)?,
    }))
}

fn dict_items_subset_result(
    vm: &mut VirtualMachine,
    left: &DictObject,
    right: &DictObject,
) -> Result<bool, RuntimeError> {
    if left.len() > right.len() {
        return Ok(false);
    }

    let mut seen_pairs = FxHashSet::default();
    for (key, left_value) in left.iter().collect::<Vec<_>>() {
        let Some(right_value) = crate::ops::dict_access::dict_get_item(vm, right, key)? else {
            return Ok(false);
        };
        if !eq_result_inner(vm, left_value, right_value, &mut seen_pairs)? {
            return Ok(false);
        }
    }

    Ok(true)
}

#[inline]
fn supports_membership_iteration_fallback(type_id: TypeId) -> bool {
    matches!(
        type_id,
        TypeId::OBJECT
            | TypeId::TYPE
            | TypeId::ITERATOR
            | TypeId::ENUMERATE
            | TypeId::GENERATOR
            | TypeId::DEQUE
            | TypeId::DICT_KEYS
            | TypeId::DICT_VALUES
            | TypeId::DICT_ITEMS
    ) || crate::ops::attribute::is_user_defined_type(type_id)
}

#[inline]
fn dict_binary_operands(
    left: Value,
    right: Value,
) -> Option<(&'static DictObject, &'static DictObject)> {
    let left_ptr = left.as_object_ptr()?;
    let right_ptr = right.as_object_ptr()?;

    let left_type = unsafe { (*(left_ptr as *const ObjectHeader)).type_id };
    let right_type = unsafe { (*(right_ptr as *const ObjectHeader)).type_id };
    if left_type != TypeId::DICT || right_type != TypeId::DICT {
        return None;
    }

    let left_dict = unsafe { &*(left_ptr as *const DictObject) };
    let right_dict = unsafe { &*(right_ptr as *const DictObject) };
    Some((left_dict, right_dict))
}

#[inline]
fn exact_dict_operand(value: Value) -> Option<(*const (), &'static DictObject)> {
    let ptr = value.as_object_ptr()?;
    (unsafe { (*(ptr as *const ObjectHeader)).type_id } == TypeId::DICT)
        .then(|| (ptr, unsafe { &*(ptr as *const DictObject) }))
}

#[inline]
pub(crate) fn boxed_set_result(mut set: SetObject, result_type: TypeId) -> Value {
    set.header.type_id = result_type;
    crate::alloc_managed_value(set)
}

#[inline]
fn boxed_dict_result(dict: DictObject) -> Value {
    crate::alloc_managed_value(dict)
}

#[inline]
fn bitwise_int_result(
    op: &'static str,
    left: Value,
    right: Value,
    compute: impl FnOnce(BigInt, BigInt) -> BigInt,
) -> Result<Value, RuntimeError> {
    let x = numeric_value_to_bigint(left).ok_or_else(|| {
        RuntimeError::unsupported_operand(op, left.type_name(), right.type_name())
    })?;
    let y = numeric_value_to_bigint(right).ok_or_else(|| {
        RuntimeError::unsupported_operand(op, left.type_name(), right.type_name())
    })?;
    let result = compute(x, y);
    if left.is_bool() && right.is_bool() {
        return Ok(Value::bool(!result.is_zero()));
    }
    Ok(bigint_to_value(result))
}

#[inline]
fn ordered_object_pair(left: *const (), right: *const ()) -> (usize, usize) {
    let left = left as usize;
    let right = right as usize;
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn sequence_values_equal(
    left: &[Value],
    right: &[Value],
    seen_pairs: &mut FxHashSet<(usize, usize)>,
) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .copied()
            .zip(right.iter().copied())
            .all(|(left, right)| values_equal_inner(left, right, seen_pairs))
}

fn dict_values_equal(
    left: &DictObject,
    right: &DictObject,
    seen_pairs: &mut FxHashSet<(usize, usize)>,
) -> bool {
    left.len() == right.len()
        && left.iter().all(|(key, left_value)| {
            right
                .get(key)
                .is_some_and(|right_value| values_equal_inner(left_value, right_value, seen_pairs))
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeSequenceKind {
    List,
    Tuple,
}

#[derive(Clone, Copy)]
enum NativeSequenceRef {
    List(&'static ListObject),
    Tuple(&'static TupleObject),
}

impl NativeSequenceRef {
    #[inline]
    fn kind(self) -> NativeSequenceKind {
        match self {
            Self::List(_) => NativeSequenceKind::List,
            Self::Tuple(_) => NativeSequenceKind::Tuple,
        }
    }

    #[inline]
    fn len(self) -> usize {
        match self {
            Self::List(list) => list.len(),
            Self::Tuple(tuple) => tuple.len(),
        }
    }

    #[inline]
    fn get(self, index: usize) -> Option<Value> {
        match self {
            Self::List(list) => {
                if index < list.len() {
                    Some(unsafe { list.get_unchecked(index) })
                } else {
                    None
                }
            }
            Self::Tuple(tuple) => {
                if index < tuple.len() {
                    Some(unsafe { tuple.get_unchecked(index) })
                } else {
                    None
                }
            }
        }
    }
}

#[inline]
fn native_sequence_items(value: Value) -> Option<(NativeSequenceKind, &'static [Value])> {
    if let Some(list) = value_as_list_ref(value) {
        return Some((NativeSequenceKind::List, list.as_slice()));
    }

    value_as_tuple_ref(value).map(|tuple| (NativeSequenceKind::Tuple, tuple.as_slice()))
}

#[inline]
fn native_sequence_ref(value: Value) -> Option<NativeSequenceRef> {
    if let Some(list) = value_as_list_ref(value) {
        return Some(NativeSequenceRef::List(list));
    }

    value_as_tuple_ref(value).map(NativeSequenceRef::Tuple)
}

#[inline]
fn exact_slice_ref(value: Value) -> Option<&'static SliceObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::SLICE).then(|| unsafe { &*(ptr as *const SliceObject) })
}

#[inline]
fn exact_native_sequence_kind(value: Value) -> Option<NativeSequenceKind> {
    let ptr = value.as_object_ptr()?;
    match unsafe { (*(ptr as *const ObjectHeader)).type_id } {
        TypeId::LIST => Some(NativeSequenceKind::List),
        TypeId::TUPLE => Some(NativeSequenceKind::Tuple),
        _ => None,
    }
}

pub(crate) fn builtin_eq_fallback(a: Value, b: Value) -> Option<bool> {
    if let Some(ordering) = compare_numeric_values(a, b) {
        return Some(ordering.is_eq());
    }

    if is_numeric_value(a) && is_numeric_value(b) {
        return Some(false);
    }

    if let Some(equal) = string_values_equal(a, b) {
        return Some(equal);
    }

    if let Some(equal) = bytes_values_equal(a, b) {
        return Some(equal);
    }

    if let (Some((left_kind, _)), Some((right_kind, _))) =
        (native_sequence_items(a), native_sequence_items(b))
        && left_kind == right_kind
    {
        return Some(values_equal(a, b));
    }

    let (left_ptr, right_ptr) = (a.as_object_ptr()?, b.as_object_ptr()?);
    let left_type = unsafe { (*(left_ptr as *const ObjectHeader)).type_id };
    let right_type = unsafe { (*(right_ptr as *const ObjectHeader)).type_id };

    match (left_type, right_type) {
        (TypeId::LIST, TypeId::LIST)
        | (TypeId::TUPLE, TypeId::TUPLE)
        | (TypeId::DICT, TypeId::DICT)
        | (TypeId::DEQUE, TypeId::DEQUE)
        | (TypeId::SET | TypeId::FROZENSET, TypeId::SET | TypeId::FROZENSET) => {
            Some(values_equal(a, b))
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn eq_result(vm: &mut VirtualMachine, a: Value, b: Value) -> Result<bool, RuntimeError> {
    let mut seen_pairs = FxHashSet::default();
    eq_result_inner(vm, a, b, &mut seen_pairs)
}

fn eq_result_inner(
    vm: &mut VirtualMachine,
    a: Value,
    b: Value,
    seen_pairs: &mut FxHashSet<(usize, usize)>,
) -> Result<bool, RuntimeError> {
    if has_numeric_subtype_operand(a, b)
        && let Some(result) = rich_compare_bool(vm, a, b, RichCompareOp::Eq)?
    {
        return Ok(result);
    }

    if let Some(ordering) = compare_numeric_values(a, b) {
        return Ok(ordering.is_eq());
    }

    if is_numeric_value(a) && is_numeric_value(b) {
        return Ok(false);
    }

    if let Some(equal) = string_values_equal(a, b) {
        return Ok(equal);
    }

    if let Some(equal) = bytes_values_equal(a, b) {
        return Ok(equal);
    }

    if exact_slice_ref(a).is_some() || exact_slice_ref(b).is_some() {
        return slice_eq_result(vm, a, b, seen_pairs);
    }

    if exact_native_sequence_kind(a).is_some()
        && exact_native_sequence_kind(a) == exact_native_sequence_kind(b)
        && let Some(equal) = native_sequence_eq_result(vm, a, b, seen_pairs)?
    {
        return Ok(equal);
    }

    if let Some(equal) = dict_view_eq_result(vm, a, b, seen_pairs)? {
        return Ok(equal);
    }

    if let Some(equal) = mapping_eq_result(vm, a, b, seen_pairs)? {
        return Ok(equal);
    }

    if let (Some((left_ptr, left)), Some((right_ptr, right))) =
        (exact_dict_operand(a), exact_dict_operand(b))
    {
        let pair = ordered_object_pair(left_ptr, right_ptr);
        if !seen_pairs.insert(pair) {
            return Ok(true);
        }
        return dict_eq_result(vm, left, right, seen_pairs);
    }

    if let Some(result) = rich_compare_bool(vm, a, b, RichCompareOp::Eq)? {
        return Ok(result);
    }

    if let Some(equal) = native_sequence_eq_result(vm, a, b, seen_pairs)? {
        return Ok(equal);
    }

    Ok(values_equal(a, b))
}

fn slice_eq_result(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    seen_pairs: &mut FxHashSet<(usize, usize)>,
) -> Result<bool, RuntimeError> {
    if left.raw_bits() == right.raw_bits() {
        return Ok(true);
    }

    let Some(left_slice) = exact_slice_ref(left) else {
        return Ok(false);
    };
    let Some(right_slice) = exact_slice_ref(right) else {
        return Ok(false);
    };

    for (left_value, right_value) in [
        (left_slice.start_value(), right_slice.start_value()),
        (left_slice.stop_value(), right_slice.stop_value()),
        (left_slice.step_value(), right_slice.step_value()),
    ] {
        if left_value.raw_bits() == right_value.raw_bits() {
            continue;
        }
        if !eq_result_inner(vm, left_value, right_value, seen_pairs)? {
            return Ok(false);
        }
    }

    Ok(true)
}

#[inline]
pub(crate) fn ne_result(vm: &mut VirtualMachine, a: Value, b: Value) -> Result<bool, RuntimeError> {
    if let Some(ordering) = compare_numeric_values(a, b) {
        return Ok(!ordering.is_eq());
    }

    if is_numeric_value(a) && is_numeric_value(b) {
        return Ok(true);
    }

    if let Some(equal) = string_values_equal(a, b) {
        return Ok(!equal);
    }

    if let Some(equal) = bytes_values_equal(a, b) {
        return Ok(!equal);
    }

    let mut seen_pairs = FxHashSet::default();
    if exact_native_sequence_kind(a).is_some()
        && exact_native_sequence_kind(a) == exact_native_sequence_kind(b)
        && let Some(equal) = native_sequence_eq_result(vm, a, b, &mut seen_pairs)?
    {
        return Ok(!equal);
    }

    if let Some(equal) = dict_view_eq_result(vm, a, b, &mut seen_pairs)? {
        return Ok(!equal);
    }

    if let Some(equal) = mapping_eq_result(vm, a, b, &mut seen_pairs)? {
        return Ok(!equal);
    }

    if let (Some((left_ptr, left)), Some((right_ptr, right))) =
        (exact_dict_operand(a), exact_dict_operand(b))
    {
        let pair = ordered_object_pair(left_ptr, right_ptr);
        if !seen_pairs.insert(pair) {
            return Ok(false);
        }
        return dict_eq_result(vm, left, right, &mut seen_pairs).map(|equal| !equal);
    }

    if let Some(result) = rich_compare_bool(vm, a, b, RichCompareOp::Ne)? {
        return Ok(result);
    }

    if let Some(equal) = native_sequence_eq_result(vm, a, b, &mut seen_pairs)? {
        return Ok(!equal);
    }

    Ok(!values_equal(a, b))
}

fn dict_view_eq_result(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    seen_pairs: &mut FxHashSet<(usize, usize)>,
) -> Result<Option<bool>, RuntimeError> {
    if let (Some((left_ptr, left_dict)), Some((right_ptr, right_dict))) =
        (exact_dict_items_view(left), exact_dict_items_view(right))
    {
        let pair = ordered_object_pair(left_ptr, right_ptr);
        if !seen_pairs.insert(pair) {
            return Ok(Some(true));
        }
        return dict_eq_result(vm, left_dict, right_dict, seen_pairs).map(Some);
    }

    if !is_dict_view_set_operand(left) && !is_dict_view_set_operand(right) {
        return Ok(None);
    }

    let Some(left_set) = materialize_set_like(left)? else {
        return Ok(None);
    };
    let Some(right_set) = materialize_set_like(right)? else {
        return Ok(None);
    };

    Ok(Some(
        left_set.len() == right_set.len() && left_set.is_subset(&right_set),
    ))
}

fn dict_eq_result(
    vm: &mut VirtualMachine,
    left: &DictObject,
    right: &DictObject,
    seen_pairs: &mut FxHashSet<(usize, usize)>,
) -> Result<bool, RuntimeError> {
    if left.len() != right.len() {
        return Ok(false);
    }

    let entries = left.iter().collect::<Vec<_>>();
    for (key, left_value) in entries {
        let Some(right_value) = crate::ops::dict_access::dict_get_item(vm, right, key)? else {
            return Ok(false);
        };
        if !eq_result_inner(vm, left_value, right_value, seen_pairs)? {
            return Ok(false);
        }
    }

    Ok(true)
}

enum MappingCompareOperand {
    Borrowed {
        identity: *const (),
        dict: &'static DictObject,
    },
    Owned {
        identity: *const (),
        dict: DictObject,
    },
}

impl MappingCompareOperand {
    #[inline]
    fn identity(&self) -> *const () {
        match self {
            Self::Borrowed { identity, .. } | Self::Owned { identity, .. } => *identity,
        }
    }

    #[inline]
    fn dict(&self) -> &DictObject {
        match self {
            Self::Borrowed { dict, .. } => dict,
            Self::Owned { dict, .. } => dict,
        }
    }
}

#[inline]
fn exact_mapping_proxy_operand(value: Value) -> Option<(*const (), &'static MappingProxyObject)> {
    let ptr = value.as_object_ptr()?;
    (unsafe { (*(ptr as *const ObjectHeader)).type_id } == TypeId::MAPPING_PROXY)
        .then(|| (ptr, unsafe { &*(ptr as *const MappingProxyObject) }))
}

fn mapping_compare_operand(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Option<MappingCompareOperand>, RuntimeError> {
    if let Some((ptr, dict)) = exact_dict_operand(value) {
        return Ok(Some(MappingCompareOperand::Borrowed {
            identity: ptr,
            dict,
        }));
    }

    if let Some(ptr) = value.as_object_ptr()
        && is_dict_subclass_instance(ptr)
        && let Some(dict) = crate::ops::objects::dict_storage_ref_from_ptr(ptr)
    {
        return Ok(Some(MappingCompareOperand::Borrowed {
            identity: ptr,
            dict,
        }));
    }

    let Some((proxy_ptr, proxy)) = exact_mapping_proxy_operand(value) else {
        return Ok(None);
    };

    if let MappingProxySource::Dict(mapping) = proxy.source()
        && let Some(mapping_ptr) = mapping.as_object_ptr()
        && let Some(dict) = crate::ops::objects::dict_storage_ref_from_ptr(mapping_ptr)
    {
        return Ok(Some(MappingCompareOperand::Borrowed {
            identity: mapping_ptr,
            dict,
        }));
    }

    let entries = crate::builtins::builtin_mapping_proxy_entries_static(proxy)?;
    let mut dict = DictObject::with_capacity(entries.len());
    for (key, value) in entries {
        crate::ops::dict_access::dict_set_item(vm, &mut dict, key, value)?;
    }

    Ok(Some(MappingCompareOperand::Owned {
        identity: proxy_ptr,
        dict,
    }))
}

#[inline]
fn is_dict_subclass_instance(ptr: *const ()) -> bool {
    let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
    type_id.raw() >= TypeId::FIRST_USER_TYPE
        && global_class_bitmap(ClassId(type_id.raw()))
            .is_some_and(|bitmap| bitmap.is_subclass_of(TypeId::DICT))
}

fn mapping_eq_result(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    seen_pairs: &mut FxHashSet<(usize, usize)>,
) -> Result<Option<bool>, RuntimeError> {
    if !is_mapping_compare_candidate(left) && !is_mapping_compare_candidate(right) {
        return Ok(None);
    }

    let Some(left) = mapping_compare_operand(vm, left)? else {
        return Ok(None);
    };
    let Some(right) = mapping_compare_operand(vm, right)? else {
        return Ok(None);
    };

    let pair = ordered_object_pair(left.identity(), right.identity());
    if !seen_pairs.insert(pair) {
        return Ok(Some(true));
    }

    dict_eq_result(vm, left.dict(), right.dict(), seen_pairs).map(Some)
}

#[inline]
fn is_mapping_compare_candidate(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
    type_id == TypeId::DICT || type_id == TypeId::MAPPING_PROXY || is_dict_subclass_instance(ptr)
}

fn native_sequence_eq_result(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    seen_pairs: &mut FxHashSet<(usize, usize)>,
) -> Result<Option<bool>, RuntimeError> {
    let Some(left_sequence) = native_sequence_ref(left) else {
        return Ok(None);
    };
    let Some(right_sequence) = native_sequence_ref(right) else {
        return Ok(None);
    };
    if left_sequence.kind() != right_sequence.kind() {
        return Ok(None);
    }

    let (Some(left_ptr), Some(right_ptr)) = (left.as_object_ptr(), right.as_object_ptr()) else {
        return Ok(None);
    };
    let pair = ordered_object_pair(left_ptr, right_ptr);
    if !seen_pairs.insert(pair) {
        return Ok(Some(true));
    }

    sequence_eq_result(vm, left_sequence, right_sequence, seen_pairs).map(Some)
}

fn sequence_eq_result(
    vm: &mut VirtualMachine,
    left: NativeSequenceRef,
    right: NativeSequenceRef,
    seen_pairs: &mut FxHashSet<(usize, usize)>,
) -> Result<bool, RuntimeError> {
    if left.len() != right.len() {
        return Ok(false);
    }

    let mut index = 0usize;
    loop {
        let left_len = left.len();
        let right_len = right.len();
        if index >= left_len || index >= right_len {
            return Ok(left_len == right_len);
        }

        let Some(left_item) = left.get(index) else {
            return Ok(right.len() == index);
        };
        let Some(right_item) = right.get(index) else {
            return Ok(left.len() == index);
        };

        if left_item.raw_bits() == right_item.raw_bits() {
            index += 1;
            continue;
        }

        if !eq_result_inner(vm, left_item, right_item, seen_pairs)? {
            let left_len = left.len();
            let right_len = right.len();
            if index >= left_len || index >= right_len {
                return Ok(left_len == right_len);
            }
            return Ok(false);
        }

        index += 1;
    }
}

fn compare_sequence_result(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
    op: RichCompareOp,
) -> Result<Option<bool>, RuntimeError> {
    let (left_kind, left_items) = match native_sequence_items(left) {
        Some(sequence) => sequence,
        None => return Ok(None),
    };
    let (right_kind, right_items) = match native_sequence_items(right) {
        Some(sequence) => sequence,
        None => return Ok(None),
    };
    if left_kind != right_kind {
        return Ok(None);
    }

    for (&left_item, &right_item) in left_items.iter().zip(right_items.iter()) {
        if eq_result(vm, left_item, right_item)? {
            continue;
        }

        return compare_order_result(vm, left_item, right_item, op).map(Some);
    }

    let result = match op {
        RichCompareOp::Lt => left_items.len() < right_items.len(),
        RichCompareOp::Le => left_items.len() <= right_items.len(),
        RichCompareOp::Gt => left_items.len() > right_items.len(),
        RichCompareOp::Ge => left_items.len() >= right_items.len(),
        RichCompareOp::Eq | RichCompareOp::Ne => unreachable!(),
    };

    Ok(Some(result))
}

pub(crate) fn compare_order_result(
    vm: &mut VirtualMachine,
    a: Value,
    b: Value,
    op: RichCompareOp,
) -> Result<bool, RuntimeError> {
    debug_assert!(matches!(
        op,
        RichCompareOp::Lt | RichCompareOp::Le | RichCompareOp::Gt | RichCompareOp::Ge
    ));

    let set_relation = match op {
        RichCompareOp::Lt => Some(SetRelation::StrictSubset),
        RichCompareOp::Le => Some(SetRelation::Subset),
        RichCompareOp::Gt => Some(SetRelation::StrictSuperset),
        RichCompareOp::Ge => Some(SetRelation::Superset),
        RichCompareOp::Eq | RichCompareOp::Ne => None,
    };
    if let Some(relation) = set_relation
        && let Some(result) = compare_set_relation(a, b, relation)
    {
        return Ok(result);
    }
    if let Some(relation) = set_relation
        && let Some(result) = compare_set_like_relation(vm, a, b, relation)?
    {
        return Ok(result);
    }

    if let Some(ordering) = compare_string_values(a, b) {
        return Ok(match op {
            RichCompareOp::Lt => ordering.is_lt(),
            RichCompareOp::Le => !ordering.is_gt(),
            RichCompareOp::Gt => ordering.is_gt(),
            RichCompareOp::Ge => !ordering.is_lt(),
            RichCompareOp::Eq | RichCompareOp::Ne => unreachable!(),
        });
    }

    if let Some(ordering) = compare_bytes_values(a, b) {
        return Ok(match op {
            RichCompareOp::Lt => ordering.is_lt(),
            RichCompareOp::Le => !ordering.is_gt(),
            RichCompareOp::Gt => ordering.is_gt(),
            RichCompareOp::Ge => !ordering.is_lt(),
            RichCompareOp::Eq | RichCompareOp::Ne => unreachable!(),
        });
    }

    if let Some(ordering) = compare_numeric_values(a, b) {
        return Ok(match op {
            RichCompareOp::Lt => ordering.is_lt(),
            RichCompareOp::Le => !ordering.is_gt(),
            RichCompareOp::Gt => ordering.is_gt(),
            RichCompareOp::Ge => !ordering.is_lt(),
            RichCompareOp::Eq | RichCompareOp::Ne => unreachable!(),
        });
    }

    if is_numeric_value(a) && is_numeric_value(b) {
        return Ok(false);
    }

    if let Some(result) = rich_compare_bool(vm, a, b, op)? {
        return Ok(result);
    }

    if let Some(result) = compare_sequence_result(vm, a, b, op)? {
        return Ok(result);
    }

    let op = match op {
        RichCompareOp::Lt => "<",
        RichCompareOp::Le => "<=",
        RichCompareOp::Gt => ">",
        RichCompareOp::Ge => ">=",
        RichCompareOp::Eq | RichCompareOp::Ne => unreachable!(),
    };
    Err(RuntimeError::type_error(format!(
        "'{op}' not supported between instances of '{}' and '{}'",
        a.type_name(),
        b.type_name()
    )))
}

/// Return the ordering relation used by Python's stable sort operations.
///
/// CPython sort only asks whether `left < right`; equality is inferred when
/// neither side is less than the other. Keeping this helper beside the rich
/// comparison engine gives `sorted()` and `list.sort()` the same primitive fast
/// paths and tuple/list lexicographic semantics as the comparison opcodes.
#[inline]
pub(crate) fn compare_sort_ordering(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
) -> Result<Ordering, RuntimeError> {
    if left == right {
        return Ok(Ordering::Equal);
    }

    if compare_order_result(vm, left, right, RichCompareOp::Lt)? {
        return Ok(Ordering::Less);
    }
    if compare_order_result(vm, right, left, RichCompareOp::Lt)? {
        return Ok(Ordering::Greater);
    }
    Ok(Ordering::Equal)
}

#[inline]
fn contains_match(
    vm: &mut VirtualMachine,
    needle: Value,
    candidate: Value,
) -> Result<bool, RuntimeError> {
    if needle.raw_bits() == candidate.raw_bits() {
        return Ok(true);
    }

    // Membership compares each element against the searched value
    // (`candidate == needle`), which matters for asymmetric __eq__ methods.
    eq_result(vm, candidate, needle)
}

// =============================================================================
// Numeric Comparisons
// =============================================================================

/// Lt: dst = src1 < src2 (generic with speculative fast-path)
///
/// Uses O(1) speculation cache lookup for type-specialized comparison.
#[inline(always)]
pub fn lt(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_lt_float, spec_lt_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
    let site = ICSiteId::new(code_id, bc_offset);

    // =========================================================================
    // Speculative Fast Path (O(1) cache lookup)
    // =========================================================================
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_lt_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_lt_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // =========================================================================
    // Slow Path: Full type check + feedback recording
    // =========================================================================
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    match compare_order_result(vm, a, b, RichCompareOp::Lt) {
        Ok(result) => {
            vm.current_frame_mut()
                .set_reg(inst.dst().0, Value::bool(result));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// Le: dst = src1 <= src2 (generic with speculative fast-path)
#[inline(always)]
pub fn le(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_le_float, spec_le_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_le_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_le_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    match compare_order_result(vm, a, b, RichCompareOp::Le) {
        Ok(result) => {
            vm.current_frame_mut()
                .set_reg(inst.dst().0, Value::bool(result));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// Gt: dst = src1 > src2 (generic with speculative fast-path)
#[inline(always)]
pub fn gt(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_gt_float, spec_gt_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_gt_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_gt_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    match compare_order_result(vm, a, b, RichCompareOp::Gt) {
        Ok(result) => {
            vm.current_frame_mut()
                .set_reg(inst.dst().0, Value::bool(result));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// Ge: dst = src1 >= src2 (generic with speculative fast-path)
#[inline(always)]
pub fn ge(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_ge_float, spec_ge_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_ge_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_ge_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    match compare_order_result(vm, a, b, RichCompareOp::Ge) {
        Ok(result) => {
            vm.current_frame_mut()
                .set_reg(inst.dst().0, Value::bool(result));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

// =============================================================================
// Equality
// =============================================================================

/// Eq: dst = src1 == src2 (generic with speculative fast-path)
#[inline(always)]
pub fn eq(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_eq_float, spec_eq_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_eq_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_eq_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::StrStr => {
                if let Some(equal) = string_values_equal(a, b) {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, Value::bool(equal));
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    match eq_result(vm, a, b) {
        Ok(result) => {
            vm.current_frame_mut()
                .set_reg(inst.dst().0, Value::bool(result));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// Ne: dst = src1 != src2 (generic with speculative fast-path)
#[inline(always)]
pub fn ne(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_ne_float, spec_ne_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_ne_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_ne_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::StrStr => {
                if let Some(equal) = string_values_equal(a, b) {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, Value::bool(!equal));
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    match ne_result(vm, a, b) {
        Ok(result) => {
            vm.current_frame_mut()
                .set_reg(inst.dst().0, Value::bool(result));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

// =============================================================================
// Identity
// =============================================================================

#[inline(always)]
fn values_identical(a: Value, b: Value) -> bool {
    if a.is_none() && b.is_none() {
        true
    } else if a.is_bool() && b.is_bool() {
        a.as_bool() == b.as_bool()
    } else if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        x == y
    } else if a.is_string() && b.is_string() {
        a.as_string_object_ptr() == b.as_string_object_ptr()
    } else if a.is_object() && b.is_object() {
        a.as_object_ptr() == b.as_object_ptr()
    } else {
        false
    }
}

/// Is: dst = src1 is src2
#[inline(always)]
pub fn is(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    frame.set_reg(inst.dst().0, Value::bool(values_identical(a, b)));
    ControlFlow::Continue
}

/// IsNot: dst = src1 is not src2
#[inline(always)]
pub fn is_not(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    frame.set_reg(inst.dst().0, Value::bool(!values_identical(a, b)));
    ControlFlow::Continue
}

// =============================================================================
// Membership Testing
// =============================================================================

/// In: dst = src1 in src2
///
/// Tests if src1 is contained in src2. Dispatches based on container type:
/// - List/Tuple: O(n) linear search with value equality
/// - Set: O(1) hash-based lookup
/// - Dict: O(1) hash-based key lookup
/// - String: SIMD-accelerated substring search
/// - Range: O(1) arithmetic bounds check
///
/// # Performance Characteristics
///
/// - Set/Dict: O(1) average case via hash tables
/// - List/Tuple: O(n) with speculative fast paths for common types
/// - String: O(n*m) worst case, O(n) typical with SIMD
/// - Range: O(1) arithmetic (no iteration)
#[inline(always)]
pub fn in_op(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let needle = frame.get_reg(inst.src1().0);
    let container = frame.get_reg(inst.src2().0);

    match contains_value(vm, needle, container) {
        Ok(result) => {
            let frame = vm.current_frame_mut();
            frame.set_reg(inst.dst().0, Value::bool(result));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// NotIn: dst = src1 not in src2
///
/// Logical negation of the `in` operator.
#[inline(always)]
pub fn not_in(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let needle = frame.get_reg(inst.src1().0);
    let container = frame.get_reg(inst.src2().0);

    match contains_value(vm, needle, container) {
        Ok(result) => {
            let frame = vm.current_frame_mut();
            frame.set_reg(inst.dst().0, Value::bool(!result));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// Core containment check with type dispatch.
///
/// Returns Ok(bool) for valid containment checks,
/// Err(ControlFlow) for type errors.
///
/// # Type Dispatch
///
/// - List/Tuple: O(n) linear scan with value equality
/// - Set: O(1) hash-based lookup
/// - Dict: O(1) hash-based key lookup
/// - String: SIMD-accelerated substring search (~8-32 GB/s)
/// - Range: O(1) arithmetic containment check
///
/// # Performance
///
/// String containment uses SSE4.2 PCMPESTRI for needles ≤16 bytes,
/// AVX2 dual-byte filter for longer needles.
#[inline]
pub(crate) fn contains_value(
    vm: &mut VirtualMachine,
    needle: Value,
    container: Value,
) -> Result<bool, RuntimeError> {
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::object::views::MappingProxyObject;
    use prism_runtime::types::range::RangeObject;
    use prism_runtime::types::string::StringObject;
    use prism_runtime::types::tuple::TupleObject;
    use prism_runtime::types::{DictObject, ListObject, SetObject};

    if let Some(haystack) = value_as_string_ref(container) {
        let needle = value_as_string_ref(needle).ok_or_else(|| {
            RuntimeError::type_error("'in <string>' requires string as left operand")
        })?;
        return Ok(str_contains(haystack.as_str(), needle.as_str()));
    }

    if let Some(haystack) = value_as_bytes_ref(container) {
        return contains_bytes_value(needle, haystack);
    }

    // Fast path: check object pointer types
    if let Some(ptr) = container.as_object_ptr() {
        // Read the TypeId from the object header for safe dispatch
        // All Python objects start with ObjectHeader which contains type_id
        let header_ptr = ptr as *const prism_runtime::object::ObjectHeader;
        let type_id = unsafe { (*header_ptr).type_id };

        match type_id {
            // List: O(n) linear scan
            TypeId::LIST => {
                let list = unsafe { &*(ptr as *const ListObject) };
                for i in 0..list.len() {
                    if let Some(val) = list.get(i as i64) {
                        if contains_match(vm, needle, val)? {
                            return Ok(true);
                        }
                    }
                }
                return Ok(false);
            }

            // Tuple: O(n) linear scan
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                for i in 0..tuple.len() {
                    if let Some(val) = tuple.get(i as i64) {
                        if contains_match(vm, needle, val)? {
                            return Ok(true);
                        }
                    }
                }
                return Ok(false);
            }

            // Set/Frozenset: O(1) hash-based lookup
            TypeId::SET | TypeId::FROZENSET => {
                let set = unsafe { &*(ptr as *const SetObject) };
                return Ok(set.contains(needle));
            }

            // Dict: O(1) hash-based key lookup
            TypeId::DICT => {
                let dict = unsafe { &*(ptr as *const DictObject) };
                return crate::ops::dict_access::dict_contains_key(vm, dict, needle);
            }

            TypeId::MAPPING_PROXY => {
                let proxy = unsafe { &*(ptr as *const MappingProxyObject) };
                return crate::builtins::builtin_mapping_proxy_contains_key(proxy, needle);
            }

            TypeId::DICT_KEYS | TypeId::DICT_VALUES | TypeId::DICT_ITEMS => {
                let view = unsafe { &*(ptr as *const DictViewObject) };
                return dict_view_contains(vm, view, needle);
            }

            // String: SIMD-accelerated substring search
            TypeId::STR => {
                let haystack = unsafe { &*(ptr as *const StringObject) };

                // Needle must also be a string for substring search
                if let Some(needle_ptr) = needle.as_object_ptr() {
                    let needle_header = needle_ptr as *const prism_runtime::object::ObjectHeader;
                    let needle_type = unsafe { (*needle_header).type_id };

                    if needle_type == TypeId::STR {
                        let needle_str = unsafe { &*(needle_ptr as *const StringObject) };
                        // Use SIMD-accelerated search (~8-32 GB/s)
                        return Ok(haystack.contains(needle_str.as_str()));
                    }
                }

                // Non-string needle in string container is always false
                return Err(RuntimeError::type_error(
                    "'in <string>' requires string as left operand",
                ));
            }

            // Range: O(1) arithmetic containment
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };

                if let Some(value) = exact_integral_value_to_bigint(needle) {
                    return Ok(range.contains_bigint(&value));
                }

                if matches!(value_type_id(needle), TypeId::FLOAT | TypeId::COMPLEX)
                    && let Some(parts) = complex_like_parts(needle)
                {
                    if parts.imag == 0.0
                        && parts.real.is_finite()
                        && parts.real.fract() == 0.0
                        && parts.real >= i64::MIN as f64
                        && parts.real <= i64::MAX as f64
                    {
                        return Ok(range.contains(parts.real as i64));
                    }
                }

                for item in range.iter() {
                    if eq_result(vm, item, needle)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }

            // Other types: fall through to protocol lookup
            _ => {}
        }

        if supports_membership_iteration_fallback(type_id) {
            if let Some(result) = contains_via_special_method(vm, needle, container)? {
                return Ok(result);
            }

            if crate::ops::attribute::is_user_defined_type(type_id) {
                if let Some(dict) = crate::ops::objects::dict_storage_ref_from_ptr(ptr) {
                    return crate::ops::dict_access::dict_contains_key(vm, dict, needle);
                }
            }

            return contains_via_iteration(vm, needle, container);
        }

        return Err(RuntimeError::type_error(format!(
            "argument of type '{}' is not iterable",
            type_id.name()
        )));
    }

    // Inline types: integers, floats, bools cannot be containers
    Err(RuntimeError::type_error("argument of type is not iterable"))
}

fn dict_view_contains(
    vm: &mut VirtualMachine,
    view: &DictViewObject,
    needle: Value,
) -> Result<bool, RuntimeError> {
    match view.kind() {
        DictViewKind::Keys => dict_view_contains_key(vm, view, needle),
        DictViewKind::Values => {
            for (_, value) in dict_view_entries(view)? {
                if contains_match(vm, needle, value)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        DictViewKind::Items => dict_items_view_contains(vm, view, needle),
    }
}

fn dict_view_contains_key(
    vm: &mut VirtualMachine,
    view: &DictViewObject,
    needle: Value,
) -> Result<bool, RuntimeError> {
    let Some(ptr) = view.dict().as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "invalid dictionary view backing object",
        ));
    };

    if let Some(dict) = crate::ops::objects::dict_storage_ref_from_ptr(ptr) {
        return crate::ops::dict_access::dict_contains_key(vm, dict, needle);
    }

    match unsafe { (*(ptr as *const ObjectHeader)).type_id } {
        TypeId::MAPPING_PROXY => {
            let proxy = unsafe { &*(ptr as *const MappingProxyObject) };
            crate::builtins::builtin_mapping_proxy_contains_key(proxy, needle)
        }
        _ => Err(RuntimeError::type_error(
            "invalid dictionary view backing object",
        )),
    }
}

fn dict_items_view_contains(
    vm: &mut VirtualMachine,
    view: &DictViewObject,
    needle: Value,
) -> Result<bool, RuntimeError> {
    let Some((needle_key, needle_value)) = dict_item_candidate(needle) else {
        return Ok(false);
    };
    let Some(ptr) = view.dict().as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "invalid dictionary view backing object",
        ));
    };

    if let Some(dict) = crate::ops::objects::dict_storage_ref_from_ptr(ptr) {
        return match crate::ops::dict_access::dict_get_item(vm, dict, needle_key)? {
            Some(value) => contains_match(vm, needle_value, value),
            None => Ok(false),
        };
    }

    match unsafe { (*(ptr as *const ObjectHeader)).type_id } {
        TypeId::MAPPING_PROXY => {
            for (key, value) in dict_view_entries(view)? {
                if contains_match(vm, needle_key, key)? && contains_match(vm, needle_value, value)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Err(RuntimeError::type_error(
            "invalid dictionary view backing object",
        )),
    }
}

#[inline]
fn dict_item_candidate(value: Value) -> Option<(Value, Value)> {
    let tuple = value_as_tuple_ref(value)?;
    let values = tuple.as_slice();
    (values.len() == 2).then_some((values[0], values[1]))
}

#[inline]
fn contains_via_special_method(
    vm: &mut VirtualMachine,
    needle: Value,
    container: Value,
) -> Result<Option<bool>, RuntimeError> {
    match resolve_special_method(container, "__contains__") {
        Ok(target) => Ok(Some(call_contains_method(vm, target, needle)?)),
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => Ok(None),
        Err(err) => Err(err),
    }
}

#[inline]
fn call_contains_method(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
    needle: Value,
) -> Result<bool, RuntimeError> {
    let result = match target.implicit_self {
        Some(implicit_self) => {
            let args = [implicit_self, needle];
            invoke_callable_value(vm, target.callable, &args)?
        }
        None => {
            let args = [needle];
            invoke_callable_value(vm, target.callable, &args)?
        }
    };

    crate::truthiness::try_is_truthy(vm, result)
}

#[inline]
fn contains_via_iteration(
    vm: &mut VirtualMachine,
    needle: Value,
    container: Value,
) -> Result<bool, RuntimeError> {
    let iterator = ensure_iterator_value(vm, container)?;

    loop {
        match next_step(vm, iterator)? {
            IterStep::Yielded(value) if contains_match(vm, needle, value)? => return Ok(true),
            IterStep::Yielded(_) => {}
            IterStep::Exhausted => return Ok(false),
        }
    }
}

/// Value equality comparison with cross-type numerical equality.
///
/// Implements Python semantics:
/// - int == float if same value (1 == 1.0)
/// - bool == int (True == 1, False == 0)
/// - Object equality by reference (identity for objects)
#[inline]
pub(crate) fn values_equal(a: Value, b: Value) -> bool {
    let mut seen_pairs = FxHashSet::default();
    values_equal_inner(a, b, &mut seen_pairs)
}

fn values_equal_inner(a: Value, b: Value, seen_pairs: &mut FxHashSet<(usize, usize)>) -> bool {
    if a.raw_bits() == b.raw_bits() {
        return true;
    }

    if a == b {
        return true;
    }

    // None equality
    if a.is_none() && b.is_none() {
        return true;
    }

    // None is not equal to anything else
    if a.is_none() || b.is_none() {
        return false;
    }

    // Bool equality
    if a.is_bool() && b.is_bool() {
        return a.as_bool() == b.as_bool();
    }

    if let Some(ordering) = compare_numeric_values(a, b) {
        return ordering.is_eq();
    }

    if is_numeric_value(a) && is_numeric_value(b) {
        return false;
    }

    if let Some(equal) = string_values_equal(a, b) {
        return equal;
    }

    if let Some(equal) = bytes_values_equal(a, b) {
        return equal;
    }

    if is_complex_value(a) || is_complex_value(b) {
        let Some(left) = complex_like_parts(a) else {
            return false;
        };
        let Some(right) = complex_like_parts(b) else {
            return false;
        };
        return left.real == right.real && left.imag == right.imag;
    }

    if let (Some(pa), Some(pb)) = (a.as_object_ptr(), b.as_object_ptr()) {
        if pa == pb {
            return true;
        }

        let left_type = unsafe { (*(pa as *const ObjectHeader)).type_id };
        let right_type = unsafe { (*(pb as *const ObjectHeader)).type_id };

        if let (Some((left_kind, left_items)), Some((right_kind, right_items))) =
            (native_sequence_items(a), native_sequence_items(b))
            && left_kind == right_kind
        {
            let pair = ordered_object_pair(pa, pb);
            if !seen_pairs.insert(pair) {
                return true;
            }

            return sequence_values_equal(left_items, right_items, seen_pairs);
        }

        match (left_type, right_type) {
            (TypeId::SLICE, TypeId::SLICE) => {
                let left = unsafe { &*(pa as *const SliceObject) };
                let right = unsafe { &*(pb as *const SliceObject) };
                return values_equal_inner(left.start_value(), right.start_value(), seen_pairs)
                    && values_equal_inner(left.stop_value(), right.stop_value(), seen_pairs)
                    && values_equal_inner(left.step_value(), right.step_value(), seen_pairs);
            }
            (TypeId::RANGE, TypeId::RANGE) => {
                let left = unsafe { &*(pa as *const RangeObject) };
                let right = unsafe { &*(pb as *const RangeObject) };
                return left == right;
            }
            (TypeId::LIST, TypeId::LIST) => {
                let pair = ordered_object_pair(pa, pb);
                if !seen_pairs.insert(pair) {
                    return true;
                }

                let left = unsafe { &*(pa as *const ListObject) };
                let right = unsafe { &*(pb as *const ListObject) };
                return sequence_values_equal(left.as_slice(), right.as_slice(), seen_pairs);
            }
            (TypeId::TUPLE, TypeId::TUPLE) => {
                let pair = ordered_object_pair(pa, pb);
                if !seen_pairs.insert(pair) {
                    return true;
                }

                let left = unsafe { &*(pa as *const TupleObject) };
                let right = unsafe { &*(pb as *const TupleObject) };
                return sequence_values_equal(left.as_slice(), right.as_slice(), seen_pairs);
            }
            (TypeId::DICT, TypeId::DICT) => {
                let pair = ordered_object_pair(pa, pb);
                if !seen_pairs.insert(pair) {
                    return true;
                }

                let left = unsafe { &*(pa as *const DictObject) };
                let right = unsafe { &*(pb as *const DictObject) };
                return dict_values_equal(left, right, seen_pairs);
            }
            (TypeId::DEQUE, TypeId::DEQUE) => {
                let pair = ordered_object_pair(pa, pb);
                if !seen_pairs.insert(pair) {
                    return true;
                }

                let left = unsafe { &*(pa as *const DequeObject) };
                let right = unsafe { &*(pb as *const DequeObject) };
                return left.deque().len() == right.deque().len()
                    && left
                        .deque()
                        .iter()
                        .zip(right.deque().iter())
                        .all(|(left, right)| values_equal_inner(*left, *right, seen_pairs));
            }
            (TypeId::METHOD, TypeId::METHOD) => {
                let pair = ordered_object_pair(pa, pb);
                if !seen_pairs.insert(pair) {
                    return true;
                }

                let left = unsafe { &*(pa as *const BoundMethod) };
                let right = unsafe { &*(pb as *const BoundMethod) };
                return values_equal_inner(left.function(), right.function(), seen_pairs)
                    && values_equal_inner(left.instance(), right.instance(), seen_pairs);
            }
            (TypeId::SET | TypeId::FROZENSET, TypeId::SET | TypeId::FROZENSET) => {
                let left = unsafe { &*(pa as *const SetObject) };
                let right = unsafe { &*(pb as *const SetObject) };
                return left.len() == right.len() && left.is_subset(right);
            }
            _ => return false,
        }
    }

    false
}

// =============================================================================
// Logical/Bitwise
// =============================================================================

/// Not: dst = not src1
#[inline(always)]
pub fn not(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let a = vm.current_frame().get_reg(inst.src1().0);

    match crate::truthiness::try_is_truthy(vm, a) {
        Ok(result) => {
            vm.current_frame_mut()
                .set_reg(inst.dst().0, Value::bool(!result));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// BitwiseAnd: dst = src1 & src2
#[inline(always)]
pub fn bitwise_and(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (a, b, dst) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, inst.dst().0)
    };

    if let Some((left, right, result_type)) = set_binary_operands(a, b) {
        let value = boxed_set_result(left.intersection(right), result_type);
        vm.current_frame_mut().set_reg(dst, value);
        return ControlFlow::Continue;
    }
    match dict_view_set_binary_result(a, b, SetLikeBinaryOp::Intersection) {
        Ok(Some(value)) => {
            vm.current_frame_mut().set_reg(dst, value);
            return ControlFlow::Continue;
        }
        Ok(None) => {}
        Err(err) => return ControlFlow::Error(err),
    }

    match bitwise_int_result("&", a, b, |x, y| x & y) {
        Ok(value) => {
            vm.current_frame_mut().set_reg(dst, value);
            ControlFlow::Continue
        }
        Err(int_err) => {
            match try_binary_special_method_result(vm, dst, a, b, "__and__", "__rand__") {
                Ok(true) => ControlFlow::Continue,
                Ok(false) => ControlFlow::Error(int_err),
                Err(err) => ControlFlow::Error(err),
            }
        }
    }
}

/// BitwiseOr: dst = src1 | src2
#[inline(always)]
pub fn bitwise_or(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (a, b, dst) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, inst.dst().0)
    };

    if let (Some(mut members), Some(rhs_members)) =
        (union_member_type_ids(a), union_member_type_ids(b))
    {
        for member in rhs_members {
            if !members.contains(&member) {
                members.push(member);
            }
        }

        if members.len() == 1 {
            vm.current_frame_mut().set_reg(dst, a);
            return ControlFlow::Continue;
        }

        let union = UnionTypeObject::new(members);
        let value = match vm.allocator().alloc(union) {
            Some(ptr) => Value::object_ptr(ptr as *const ()),
            None => {
                return ControlFlow::Error(RuntimeError::internal(
                    "out of memory: failed to allocate union type",
                ));
            }
        };
        vm.current_frame_mut().set_reg(dst, value);
        return ControlFlow::Continue;
    }

    if let Some((left, right, result_type)) = set_binary_operands(a, b) {
        let value = boxed_set_result(left.union(right), result_type);
        vm.current_frame_mut().set_reg(dst, value);
        return ControlFlow::Continue;
    }
    match dict_view_set_binary_result(a, b, SetLikeBinaryOp::Union) {
        Ok(Some(value)) => {
            vm.current_frame_mut().set_reg(dst, value);
            return ControlFlow::Continue;
        }
        Ok(None) => {}
        Err(err) => return ControlFlow::Error(err),
    }

    if let Some((left, right)) = dict_binary_operands(a, b) {
        let mut merged = DictObject::with_capacity(left.len() + right.len());
        merged.update(left);
        merged.update(right);
        vm.current_frame_mut()
            .set_reg(dst, boxed_dict_result(merged));
        return ControlFlow::Continue;
    }

    match bitwise_int_result("|", a, b, |x, y| x | y) {
        Ok(value) => {
            vm.current_frame_mut().set_reg(dst, value);
            ControlFlow::Continue
        }
        Err(int_err) => {
            match try_binary_special_method_result(vm, dst, a, b, "__or__", "__ror__") {
                Ok(true) => ControlFlow::Continue,
                Ok(false) => ControlFlow::Error(int_err),
                Err(err) => ControlFlow::Error(err),
            }
        }
    }
}

/// BitwiseXor: dst = src1 ^ src2
#[inline(always)]
pub fn bitwise_xor(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (a, b, dst) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, inst.dst().0)
    };

    if let Some((left, right, result_type)) = set_binary_operands(a, b) {
        let value = boxed_set_result(left.symmetric_difference(right), result_type);
        vm.current_frame_mut().set_reg(dst, value);
        return ControlFlow::Continue;
    }
    match dict_view_set_binary_result(a, b, SetLikeBinaryOp::SymmetricDifference) {
        Ok(Some(value)) => {
            vm.current_frame_mut().set_reg(dst, value);
            return ControlFlow::Continue;
        }
        Ok(None) => {}
        Err(err) => return ControlFlow::Error(err),
    }

    match bitwise_int_result("^", a, b, |x, y| x ^ y) {
        Ok(value) => {
            vm.current_frame_mut().set_reg(dst, value);
            ControlFlow::Continue
        }
        Err(int_err) => {
            match try_binary_special_method_result(vm, dst, a, b, "__xor__", "__rxor__") {
                Ok(true) => ControlFlow::Continue,
                Ok(false) => ControlFlow::Error(int_err),
                Err(err) => ControlFlow::Error(err),
            }
        }
    }
}

/// BitwiseNot: dst = ~src1
#[inline(always)]
pub fn bitwise_not(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (a, dst) = {
        let frame = vm.current_frame();
        (frame.get_reg(inst.src1().0), inst.dst().0)
    };

    if a.is_bool()
        && let Err(err) = emit_bool_invert_deprecation_warning(vm)
    {
        return ControlFlow::Error(err);
    }

    match numeric_value_to_bigint(a) {
        Some(x) => {
            vm.current_frame_mut().set_reg(dst, bigint_to_value(!x));
            ControlFlow::Continue
        }
        None => ControlFlow::Error(RuntimeError::type_error(format!(
            "bad operand type for unary ~: '{}'",
            a.type_name()
        ))),
    }
}

/// Shl: dst = src1 << src2
#[inline(always)]
pub fn shl(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (a, b, dst) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, inst.dst().0)
    };

    match (numeric_value_to_bigint(a), numeric_value_to_bigint(b)) {
        (Some(_), Some(shift)) if shift.sign() == num_bigint::Sign::Minus => {
            ControlFlow::Error(RuntimeError::value_error("negative shift count"))
        }
        (Some(value), Some(shift)) => {
            let Some(shift_amount) = shift.to_usize() else {
                return ControlFlow::Error(RuntimeError::new(
                    crate::error::RuntimeErrorKind::OverflowError {
                        message: "shift count too large".into(),
                    },
                ));
            };
            vm.current_frame_mut()
                .set_reg(dst, bigint_to_value(value << shift_amount));
            ControlFlow::Continue
        }
        _ => match try_binary_special_method_result(vm, dst, a, b, "__lshift__", "__rlshift__") {
            Ok(true) => ControlFlow::Continue,
            Ok(false) => ControlFlow::Error(RuntimeError::unsupported_operand(
                "<<",
                a.type_name(),
                b.type_name(),
            )),
            Err(err) => ControlFlow::Error(err),
        },
    }
}

/// Shr: dst = src1 >> src2
#[inline(always)]
pub fn shr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (a, b, dst) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, inst.dst().0)
    };

    match (numeric_value_to_bigint(a), numeric_value_to_bigint(b)) {
        (Some(_), Some(shift)) if shift.sign() == num_bigint::Sign::Minus => {
            ControlFlow::Error(RuntimeError::value_error("negative shift count"))
        }
        (Some(value), Some(shift)) => {
            let Some(shift_amount) = shift.to_usize() else {
                let result = if value.sign() == num_bigint::Sign::Minus {
                    Value::int(-1).unwrap()
                } else {
                    Value::int(0).unwrap()
                };
                vm.current_frame_mut().set_reg(dst, result);
                return ControlFlow::Continue;
            };
            vm.current_frame_mut()
                .set_reg(dst, bigint_to_value(value >> shift_amount));
            ControlFlow::Continue
        }
        _ => match try_binary_special_method_result(vm, dst, a, b, "__rshift__", "__rrshift__") {
            Ok(true) => ControlFlow::Continue,
            Ok(false) => ControlFlow::Error(RuntimeError::unsupported_operand(
                ">>",
                a.type_name(),
                b.type_name(),
            )),
            Err(err) => ControlFlow::Error(err),
        },
    }
}

/// InPlaceBitwiseAnd: dst = src1; dst &= src2.
#[inline(always)]
pub fn inplace_bitwise_and(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    crate::ops::arithmetic::inplace_binary_with_fallback(vm, inst, "__iand__", bitwise_and)
}

/// InPlaceBitwiseOr: dst = src1; dst |= src2.
#[inline(always)]
pub fn inplace_bitwise_or(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    crate::ops::arithmetic::inplace_binary_with_fallback(vm, inst, "__ior__", bitwise_or)
}

/// InPlaceBitwiseXor: dst = src1; dst ^= src2.
#[inline(always)]
pub fn inplace_bitwise_xor(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    crate::ops::arithmetic::inplace_binary_with_fallback(vm, inst, "__ixor__", bitwise_xor)
}

/// InPlaceShl: dst = src1; dst <<= src2.
#[inline(always)]
pub fn inplace_shl(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    crate::ops::arithmetic::inplace_binary_with_fallback(vm, inst, "__ilshift__", shl)
}

/// InPlaceShr: dst = src1; dst >>= src2.
#[inline(always)]
pub fn inplace_shr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    crate::ops::arithmetic::inplace_binary_with_fallback(vm, inst, "__irshift__", shr)
}

#[inline]
fn try_binary_special_method_result(
    vm: &mut VirtualMachine,
    dst: u8,
    left: Value,
    right: Value,
    left_method: &'static str,
    right_method: &'static str,
) -> Result<bool, RuntimeError> {
    if let Some(value) = binary_special_method(vm, left, right, left_method, right_method)? {
        vm.current_frame_mut().set_reg(dst, value);
        return Ok(true);
    }

    Ok(false)
}
