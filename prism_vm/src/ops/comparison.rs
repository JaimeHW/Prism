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
use crate::ops::protocols::{RichCompareOp, rich_compare_bool};
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
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::UnionTypeObject;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::list::value_as_list_ref;
use prism_runtime::types::simd::search::{bytes_contains, str_contains};
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
fn contains_bytes_value(needle: Value, haystack: &[u8]) -> Result<bool, ControlFlow> {
    if let Some(value) = int_like_value(needle) {
        let byte = u8::try_from(value).map_err(|_| {
            ControlFlow::Error(RuntimeError::value_error("byte must be in range(0, 256)"))
        })?;
        return Ok(haystack.contains(&byte));
    }

    if let Some(needle_bytes) = value_as_bytes_ref(needle) {
        return Ok(bytes_contains(haystack, needle_bytes));
    }

    Err(ControlFlow::Error(RuntimeError::type_error(
        "'in <bytes>' requires bytes-like object or integer as left operand",
    )))
}

#[inline]
fn union_member_type_ids(value: Value) -> Option<Vec<TypeId>> {
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
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum SetRelation {
    StrictSubset,
    Subset,
    StrictSuperset,
    Superset,
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

#[inline]
fn supports_membership_iteration_fallback(type_id: TypeId) -> bool {
    matches!(
        type_id,
        TypeId::OBJECT
            | TypeId::TYPE
            | TypeId::ITERATOR
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

#[inline]
fn native_sequence_items(value: Value) -> Option<(NativeSequenceKind, &'static [Value])> {
    if let Some(list) = value_as_list_ref(value) {
        return Some((NativeSequenceKind::List, list.as_slice()));
    }

    value_as_tuple_ref(value).map(|tuple| (NativeSequenceKind::Tuple, tuple.as_slice()))
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
fn eq_result(vm: &mut VirtualMachine, a: Value, b: Value) -> Result<bool, RuntimeError> {
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

    if let Some(result) = rich_compare_bool(vm, a, b, RichCompareOp::Eq)? {
        return Ok(result);
    }

    Ok(values_equal(a, b))
}

#[inline]
fn ne_result(vm: &mut VirtualMachine, a: Value, b: Value) -> Result<bool, RuntimeError> {
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

    if let Some(result) = rich_compare_bool(vm, a, b, RichCompareOp::Ne)? {
        return Ok(result);
    }

    Ok(!values_equal(a, b))
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

fn compare_order_result(
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

    Err(RuntimeError::unsupported_operand(
        match op {
            RichCompareOp::Lt => "<",
            RichCompareOp::Le => "<=",
            RichCompareOp::Gt => ">",
            RichCompareOp::Ge => ">=",
            RichCompareOp::Eq | RichCompareOp::Ne => unreachable!(),
        },
        a.type_name(),
        b.type_name(),
    ))
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

    eq_result(vm, needle, candidate)
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
        Err(cf) => cf,
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
        Err(cf) => cf,
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
fn contains_value(
    vm: &mut VirtualMachine,
    needle: Value,
    container: Value,
) -> Result<bool, ControlFlow> {
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::object::views::MappingProxyObject;
    use prism_runtime::types::range::RangeObject;
    use prism_runtime::types::string::StringObject;
    use prism_runtime::types::tuple::TupleObject;
    use prism_runtime::types::{DictObject, ListObject, SetObject};

    if let Some(haystack) = value_as_string_ref(container) {
        let needle = value_as_string_ref(needle).ok_or_else(|| {
            ControlFlow::Error(RuntimeError::type_error(
                "'in <string>' requires string as left operand",
            ))
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
                        if contains_match(vm, needle, val).map_err(ControlFlow::Error)? {
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
                        if contains_match(vm, needle, val).map_err(ControlFlow::Error)? {
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
                return Ok(dict.get(needle).is_some());
            }

            TypeId::MAPPING_PROXY => {
                let proxy = unsafe { &*(ptr as *const MappingProxyObject) };
                return crate::builtins::builtin_mapping_proxy_contains_key(proxy, needle)
                    .map_err(ControlFlow::Error);
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
                return Err(ControlFlow::Error(RuntimeError::type_error(
                    "'in <string>' requires string as left operand",
                )));
            }

            // Range: O(1) arithmetic containment
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };

                // Only integers can be in a range
                if let Some(value) = needle.as_int() {
                    return Ok(range.contains(value));
                }

                // Float values check: Python allows 5.0 in range(10)
                if let Some(f) = needle.as_float() {
                    // Check if it's a whole number
                    if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                        return Ok(range.contains(f as i64));
                    }
                }

                // Non-numeric types are not in range
                return Ok(false);
            }

            // Other types: fall through to protocol lookup
            _ => {}
        }

        if supports_membership_iteration_fallback(type_id) {
            if let Some(result) =
                contains_via_special_method(vm, needle, container).map_err(ControlFlow::Error)?
            {
                return Ok(result);
            }

            if crate::ops::attribute::is_user_defined_type(type_id) {
                if let Some(dict) = crate::ops::objects::dict_storage_ref_from_ptr(ptr) {
                    return Ok(dict.get(needle).is_some());
                }
            }

            return contains_via_iteration(vm, needle, container).map_err(ControlFlow::Error);
        }

        return Err(ControlFlow::Error(RuntimeError::type_error(format!(
            "argument of type '{}' is not iterable",
            type_id.name()
        ))));
    }

    // Inline types: integers, floats, bools cannot be containers
    Err(ControlFlow::Error(RuntimeError::type_error(
        "argument of type is not iterable",
    )))
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

    let frame = vm.current_frame_mut();

    match bitwise_int_result("&", a, b, |x, y| x & y) {
        Ok(value) => {
            frame.set_reg(dst, value);
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
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
        Err(err) => ControlFlow::Error(err),
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

    let frame = vm.current_frame_mut();

    match bitwise_int_result("^", a, b, |x, y| x ^ y) {
        Ok(value) => {
            frame.set_reg(dst, value);
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
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
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

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
            frame.set_reg(inst.dst().0, bigint_to_value(value << shift_amount));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            "<<",
            a.type_name(),
            b.type_name(),
        )),
    }
}

/// Shr: dst = src1 >> src2
#[inline(always)]
pub fn shr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

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
                frame.set_reg(inst.dst().0, result);
                return ControlFlow::Continue;
            };
            frame.set_reg(inst.dst().0, bigint_to_value(value >> shift_amount));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            ">>",
            a.type_name(),
            b.type_name(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VirtualMachine;
    use crate::builtins::builtin_type_object_for_type_id;
    use num_bigint::BigInt;
    use prism_code::{CodeObject, Instruction, Opcode, Register};
    use prism_compiler::Compiler;
    use prism_core::intern::intern;
    use prism_parser::parse;
    use prism_runtime::object::shape::Shape;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::types::bytes::BytesObject;
    use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
    use prism_runtime::types::string::StringObject;
    use prism_runtime::types::{DictObject, ListObject, SetObject, TupleObject};
    use std::path::Path;
    use std::sync::Arc;

    fn vm_with_frame() -> VirtualMachine {
        let mut vm = VirtualMachine::new();
        let code = Arc::new(CodeObject::new("test_compare", "<test>"));
        vm.push_frame(code, 0).expect("frame push failed");
        vm
    }

    fn boxed_string_value(value: &str) -> (Value, *mut StringObject) {
        let ptr = Box::into_raw(Box::new(StringObject::new(value)));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    fn boxed_object_value<T>(value: T) -> (Value, *mut T) {
        let ptr = Box::into_raw(Box::new(value));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    fn promoted_int(value: BigInt) -> Value {
        bigint_to_value(value)
    }

    fn execute(source: &str) -> Result<Value, String> {
        execute_with_search_paths(source, &[])
    }

    fn binary_inst(opcode: Opcode) -> Instruction {
        Instruction::op_dss(opcode, Register::new(3), Register::new(1), Register::new(2))
    }

    fn execute_with_search_paths(source: &str, search_paths: &[&Path]) -> Result<Value, String> {
        let module = parse(source).map_err(|err| format!("parse error: {err:?}"))?;
        let code = Compiler::compile_module(&module, "<comparison-test>")
            .map_err(|err| format!("compile error: {err:?}"))?;

        let mut vm = VirtualMachine::new();
        for path in search_paths {
            let path = Arc::<str>::from(path.to_string_lossy().into_owned());
            vm.import_resolver.add_search_path(path);
        }

        vm.execute(Arc::new(code))
            .map_err(|err| format!("runtime error: {err:?}"))
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        drop(unsafe { Box::from_raw(ptr) });
    }

    #[test]
    fn test_string_values_equal_across_tagged_and_heap_representations() {
        let tagged = Value::string(intern("inter"));
        let (heap_same, heap_same_ptr) = boxed_string_value("inter");
        let (heap_other, heap_other_ptr) = boxed_string_value("inner");

        assert_eq!(string_values_equal(tagged, tagged), Some(true));
        assert_eq!(string_values_equal(tagged, heap_same), Some(true));
        assert_eq!(string_values_equal(heap_same, tagged), Some(true));
        assert_eq!(string_values_equal(heap_same, heap_other), Some(false));

        unsafe { drop_boxed(heap_same_ptr) };
        unsafe { drop_boxed(heap_other_ptr) };
    }

    #[test]
    fn test_values_equal_uses_string_contents() {
        let left = Value::string(intern("left"));
        let (right, right_ptr) = boxed_string_value("left");
        let (other, other_ptr) = boxed_string_value("right");

        assert!(values_equal(left, right));
        assert!(!values_equal(left, other));

        unsafe { drop_boxed(right_ptr) };
        unsafe { drop_boxed(other_ptr) };
    }

    #[test]
    fn test_values_equal_supports_heap_backed_ints() {
        let big = (BigInt::from(1_u8) << 80_u32) + BigInt::from(7_u8);
        let left = promoted_int(big.clone());
        let right = promoted_int(big.clone());
        let other = promoted_int(big + BigInt::from(1_u8));

        assert!(values_equal(left, right));
        assert!(!values_equal(left, other));
    }

    #[test]
    fn test_eq_and_lt_opcodes_support_heap_backed_ints() {
        let mut eq_vm = vm_with_frame();
        let big = (BigInt::from(1_u8) << 80_u32) + BigInt::from(5_u8);
        eq_vm
            .current_frame_mut()
            .set_reg(1, promoted_int(big.clone()));
        eq_vm
            .current_frame_mut()
            .set_reg(2, promoted_int(big.clone()));

        assert!(matches!(
            eq(&mut eq_vm, binary_inst(Opcode::Eq)),
            ControlFlow::Continue
        ));
        assert_eq!(eq_vm.current_frame().get_reg(3).as_bool(), Some(true));

        let mut lt_vm = vm_with_frame();
        lt_vm.current_frame_mut().set_reg(1, promoted_int(big));
        lt_vm.current_frame_mut().set_reg(
            2,
            promoted_int((BigInt::from(1_u8) << 80_u32) + BigInt::from(6_u8)),
        );

        assert!(matches!(
            lt(&mut lt_vm, binary_inst(Opcode::Lt)),
            ControlFlow::Continue
        ));
        assert_eq!(lt_vm.current_frame().get_reg(3).as_bool(), Some(true));
    }

    #[test]
    fn test_values_equal_uses_bytes_contents_across_bytes_and_bytearray() {
        let (left, left_ptr) = boxed_object_value(BytesObject::from_slice(b"01\n"));
        let (right, right_ptr) = boxed_object_value(BytesObject::from_slice(b"01\n"));
        let (bytearray, bytearray_ptr) =
            boxed_object_value(BytesObject::bytearray_from_slice(b"01\n"));
        let (other, other_ptr) = boxed_object_value(BytesObject::from_slice(b"00\n"));

        assert!(values_equal(left, right));
        assert!(values_equal(left, bytearray));
        assert!(!values_equal(left, other));

        unsafe { drop_boxed(left_ptr) };
        unsafe { drop_boxed(right_ptr) };
        unsafe { drop_boxed(bytearray_ptr) };
        unsafe { drop_boxed(other_ptr) };
    }

    #[test]
    fn test_execute_supports_bytes_equality_and_ordering() {
        let result = execute(
            r#"
assert b"abc" == b"abc"
assert b"abc" == bytearray(b"abc")
assert b"abc" < b"abd"
assert b"abd" > b"abc"
"#,
        );

        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_values_equal_uses_list_contents() {
        let (left, left_ptr) = boxed_object_value(ListObject::from_slice(&[
            Value::string(intern("1")),
            Value::string(intern("2")),
            Value::string(intern("3")),
        ]));
        let (right, right_ptr) = boxed_object_value(ListObject::from_slice(&[
            Value::string(intern("1")),
            Value::string(intern("2")),
            Value::string(intern("3")),
        ]));
        let (other, other_ptr) = boxed_object_value(ListObject::from_slice(&[
            Value::string(intern("1")),
            Value::string(intern("2")),
        ]));

        assert!(values_equal(left, right));
        assert!(!values_equal(left, other));

        unsafe { drop_boxed(left_ptr) };
        unsafe { drop_boxed(right_ptr) };
        unsafe { drop_boxed(other_ptr) };
    }

    #[test]
    fn test_values_equal_uses_tuple_contents() {
        let (left, left_ptr) = boxed_object_value(TupleObject::from_slice(&[
            Value::string(intern("1")),
            Value::string(intern("2")),
            Value::string(intern("3")),
        ]));
        let (right, right_ptr) = boxed_object_value(TupleObject::from_slice(&[
            Value::string(intern("1")),
            Value::string(intern("2")),
            Value::string(intern("3")),
        ]));
        let (other, other_ptr) = boxed_object_value(TupleObject::from_slice(&[
            Value::string(intern("1")),
            Value::string(intern("2")),
            Value::string(intern("4")),
        ]));

        assert!(values_equal(left, right));
        assert!(!values_equal(left, other));

        unsafe { drop_boxed(left_ptr) };
        unsafe { drop_boxed(right_ptr) };
        unsafe { drop_boxed(other_ptr) };
    }

    #[test]
    fn test_values_equal_uses_dict_and_set_contents() {
        let mut left_dict = DictObject::new();
        left_dict.set(Value::string(intern("a")), Value::int_unchecked(1));
        left_dict.set(Value::string(intern("b")), Value::int_unchecked(2));
        let mut right_dict = DictObject::new();
        right_dict.set(Value::string(intern("a")), Value::int_unchecked(1));
        right_dict.set(Value::string(intern("b")), Value::int_unchecked(2));
        let mut other_dict = DictObject::new();
        other_dict.set(Value::string(intern("a")), Value::int_unchecked(1));
        other_dict.set(Value::string(intern("b")), Value::int_unchecked(3));

        let (left_dict_value, left_dict_ptr) = boxed_object_value(left_dict);
        let (right_dict_value, right_dict_ptr) = boxed_object_value(right_dict);
        let (other_dict_value, other_dict_ptr) = boxed_object_value(other_dict);

        assert!(values_equal(left_dict_value, right_dict_value));
        assert!(!values_equal(left_dict_value, other_dict_value));

        let (left_set_value, left_set_ptr) = boxed_object_value(SetObject::from_slice(&[
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
        ]));
        let (right_set_value, right_set_ptr) = boxed_object_value(SetObject::from_slice(&[
            Value::int_unchecked(3),
            Value::int_unchecked(2),
            Value::int_unchecked(1),
        ]));
        let (other_set_value, other_set_ptr) = boxed_object_value(SetObject::from_slice(&[
            Value::int_unchecked(1),
            Value::int_unchecked(2),
        ]));

        assert!(values_equal(left_set_value, right_set_value));
        assert!(!values_equal(left_set_value, other_set_value));

        unsafe { drop_boxed(left_dict_ptr) };
        unsafe { drop_boxed(right_dict_ptr) };
        unsafe { drop_boxed(other_dict_ptr) };
        unsafe { drop_boxed(left_set_ptr) };
        unsafe { drop_boxed(right_set_ptr) };
        unsafe { drop_boxed(other_set_ptr) };
    }

    #[test]
    fn test_eq_opcode_supports_string_values() {
        let mut vm = vm_with_frame();
        let left = Value::string(intern("inter"));
        let (right, right_ptr) = boxed_string_value("inter");
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::Eq,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(eq(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

        unsafe { drop_boxed(right_ptr) };
    }

    #[test]
    fn test_is_and_is_not_treat_interned_strings_as_identical() {
        let mut is_vm = vm_with_frame();
        let left = Value::string(intern("decorator-name"));
        let right = Value::string(intern("decorator-name"));
        is_vm.current_frame_mut().set_reg(1, left);
        is_vm.current_frame_mut().set_reg(2, right);

        assert!(matches!(
            is(&mut is_vm, binary_inst(Opcode::Is)),
            ControlFlow::Continue
        ));
        assert_eq!(is_vm.current_frame().get_reg(3), Value::bool(true));

        let mut is_not_vm = vm_with_frame();
        is_not_vm.current_frame_mut().set_reg(1, left);
        is_not_vm.current_frame_mut().set_reg(2, right);

        assert!(matches!(
            is_not(&mut is_not_vm, binary_inst(Opcode::IsNot)),
            ControlFlow::Continue
        ));
        assert_eq!(is_not_vm.current_frame().get_reg(3), Value::bool(false));
    }

    #[test]
    fn test_ne_opcode_supports_string_values() {
        let mut vm = vm_with_frame();
        let left = Value::string(intern("inter"));
        let (right, right_ptr) = boxed_string_value("inner");
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::Ne,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(ne(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

        unsafe { drop_boxed(right_ptr) };
    }

    #[test]
    fn test_lt_opcode_supports_string_values() {
        let mut vm = vm_with_frame();
        let left = Value::string(intern("alpha"));
        let (right, right_ptr) = boxed_string_value("beta");
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::Lt,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(lt(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

        unsafe { drop_boxed(right_ptr) };
    }

    #[test]
    fn test_le_opcode_supports_string_values() {
        let mut vm = vm_with_frame();
        let (left, left_ptr) = boxed_string_value("beta");
        let right = Value::string(intern("beta"));
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::Le,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(le(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

        unsafe { drop_boxed(left_ptr) };
    }

    #[test]
    fn test_gt_opcode_supports_string_values() {
        let mut vm = vm_with_frame();
        let (left, left_ptr) = boxed_string_value("gamma");
        let right = Value::string(intern("beta"));
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::Gt,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(gt(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

        unsafe { drop_boxed(left_ptr) };
    }

    #[test]
    fn test_ge_opcode_supports_string_values() {
        let mut vm = vm_with_frame();
        let left = Value::string(intern("beta"));
        let (right, right_ptr) = boxed_string_value("beta");
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::Ge,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(ge(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));

        unsafe { drop_boxed(right_ptr) };
    }

    #[test]
    fn test_bitwise_or_on_type_objects_produces_union_type() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut()
            .set_reg(1, builtin_type_object_for_type_id(TypeId::INT));
        vm.current_frame_mut()
            .set_reg(2, builtin_type_object_for_type_id(TypeId::STR));

        let inst = Instruction::op_dss(
            Opcode::BitwiseOr,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(bitwise_or(&mut vm, inst), ControlFlow::Continue));

        let ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
        assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::UNION);
    }

    #[test]
    fn test_bitwise_and_on_sets_returns_intersection() {
        let mut vm = vm_with_frame();
        let (left, left_ptr) = boxed_object_value(SetObject::from_slice(&[
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
        ]));
        let (right, right_ptr) = boxed_object_value(SetObject::from_slice(&[
            Value::int_unchecked(2),
            Value::int_unchecked(3),
            Value::int_unchecked(4),
        ]));
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::BitwiseAnd,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(bitwise_and(&mut vm, inst), ControlFlow::Continue));

        let result_ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
        let result = unsafe { &*(result_ptr as *const SetObject) };
        assert_eq!(
            crate::ops::objects::extract_type_id(result_ptr),
            TypeId::SET
        );
        assert!(result.contains(Value::int_unchecked(2)));
        assert!(result.contains(Value::int_unchecked(3)));
        assert_eq!(result.len(), 2);

        unsafe { drop_boxed(left_ptr) };
        unsafe { drop_boxed(right_ptr) };
    }

    #[test]
    fn test_bitwise_or_on_frozenset_preserves_left_operand_type() {
        let mut vm = vm_with_frame();
        let mut left = SetObject::from_slice(&[Value::int_unchecked(1), Value::int_unchecked(2)]);
        left.header.type_id = TypeId::FROZENSET;
        let mut right = SetObject::from_slice(&[Value::int_unchecked(2), Value::int_unchecked(3)]);
        right.header.type_id = TypeId::FROZENSET;
        let (left, left_ptr) = boxed_object_value(left);
        let (right, right_ptr) = boxed_object_value(right);
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::BitwiseOr,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(bitwise_or(&mut vm, inst), ControlFlow::Continue));

        let result_ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
        let result = unsafe { &*(result_ptr as *const SetObject) };
        assert_eq!(
            crate::ops::objects::extract_type_id(result_ptr),
            TypeId::FROZENSET
        );
        assert!(result.contains(Value::int_unchecked(1)));
        assert!(result.contains(Value::int_unchecked(2)));
        assert!(result.contains(Value::int_unchecked(3)));
        assert_eq!(result.len(), 3);

        unsafe { drop_boxed(left_ptr) };
        unsafe { drop_boxed(right_ptr) };
    }

    #[test]
    fn test_bitwise_or_on_dicts_returns_merged_dict_with_right_overrides() {
        let mut vm = vm_with_frame();
        let mut left = DictObject::new();
        left.set(Value::int_unchecked(1), Value::int_unchecked(10));
        left.set(Value::int_unchecked(2), Value::int_unchecked(20));
        let mut right = DictObject::new();
        right.set(Value::int_unchecked(2), Value::int_unchecked(200));
        right.set(Value::int_unchecked(3), Value::int_unchecked(30));
        let (left, left_ptr) = boxed_object_value(left);
        let (right, right_ptr) = boxed_object_value(right);
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::BitwiseOr,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(bitwise_or(&mut vm, inst), ControlFlow::Continue));

        let result_ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
        let result = unsafe { &*(result_ptr as *const DictObject) };
        let items = result
            .iter()
            .map(|(key, value)| (key.as_int(), value.as_int()))
            .collect::<Vec<_>>();
        assert_eq!(
            items,
            vec![
                (Some(1), Some(10)),
                (Some(2), Some(200)),
                (Some(3), Some(30)),
            ]
        );

        unsafe { drop_boxed(left_ptr) };
        unsafe { drop_boxed(right_ptr) };
    }

    #[test]
    fn test_bitwise_xor_on_sets_returns_symmetric_difference() {
        let mut vm = vm_with_frame();
        let (left, left_ptr) = boxed_object_value(SetObject::from_slice(&[
            Value::int_unchecked(1),
            Value::int_unchecked(2),
            Value::int_unchecked(3),
        ]));
        let (right, right_ptr) = boxed_object_value(SetObject::from_slice(&[
            Value::int_unchecked(3),
            Value::int_unchecked(4),
        ]));
        vm.current_frame_mut().set_reg(1, left);
        vm.current_frame_mut().set_reg(2, right);

        let inst = Instruction::op_dss(
            Opcode::BitwiseXor,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(bitwise_xor(&mut vm, inst), ControlFlow::Continue));

        let result_ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
        let result = unsafe { &*(result_ptr as *const SetObject) };
        assert!(result.contains(Value::int_unchecked(1)));
        assert!(result.contains(Value::int_unchecked(2)));
        assert!(result.contains(Value::int_unchecked(4)));
        assert_eq!(result.len(), 3);

        unsafe { drop_boxed(left_ptr) };
        unsafe { drop_boxed(right_ptr) };
    }

    #[test]
    fn test_shl_promotes_large_results_to_heap_backed_ints() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::int(1).unwrap());
        vm.current_frame_mut().set_reg(2, Value::int(1000).unwrap());

        let inst = Instruction::op_dss(
            Opcode::Shl,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(shl(&mut vm, inst), ControlFlow::Continue));

        let result = vm.current_frame().get_reg(3);
        let expected = num_bigint::BigInt::from(1_u8) << 1000_u32;
        assert_eq!(value_to_bigint(result), Some(expected));
    }

    #[test]
    fn test_bitwise_ops_support_heap_backed_ints() {
        let all_ones = (BigInt::from(1_u8) << 128_u32) - BigInt::from(1_u8);
        let low_ones = (BigInt::from(1_u8) << 64_u32) - BigInt::from(1_u8);
        let high_bit = BigInt::from(1_u8) << 127_u32;

        let mut vm = vm_with_frame();
        vm.current_frame_mut()
            .set_reg(1, promoted_int(all_ones.clone()));
        vm.current_frame_mut()
            .set_reg(2, promoted_int(low_ones.clone()));

        assert!(matches!(
            bitwise_xor(&mut vm, binary_inst(Opcode::BitwiseXor)),
            ControlFlow::Continue
        ));
        assert_eq!(
            value_to_bigint(vm.current_frame().get_reg(3)),
            Some(all_ones.clone() ^ low_ones.clone())
        );

        vm.current_frame_mut()
            .set_reg(1, promoted_int(all_ones.clone()));
        vm.current_frame_mut()
            .set_reg(2, promoted_int(low_ones.clone()));
        assert!(matches!(
            bitwise_and(&mut vm, binary_inst(Opcode::BitwiseAnd)),
            ControlFlow::Continue
        ));
        assert_eq!(
            value_to_bigint(vm.current_frame().get_reg(3)),
            Some(low_ones.clone())
        );

        vm.current_frame_mut()
            .set_reg(1, promoted_int(high_bit.clone()));
        vm.current_frame_mut().set_reg(2, Value::int(3).unwrap());
        assert!(matches!(
            bitwise_or(&mut vm, binary_inst(Opcode::BitwiseOr)),
            ControlFlow::Continue
        ));
        assert_eq!(
            value_to_bigint(vm.current_frame().get_reg(3)),
            Some(high_bit.clone() | BigInt::from(3_u8))
        );

        vm.current_frame_mut()
            .set_reg(1, promoted_int(high_bit.clone()));
        let inst = Instruction::op_ds(Opcode::BitwiseNot, Register::new(3), Register::new(1));
        assert!(matches!(bitwise_not(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(
            value_to_bigint(vm.current_frame().get_reg(3)),
            Some(!high_bit)
        );
    }

    #[test]
    fn test_shift_ops_accept_bool_operands_as_ints() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(true));
        vm.current_frame_mut().set_reg(2, Value::int(6).unwrap());
        assert!(matches!(
            shl(&mut vm, binary_inst(Opcode::Shl)),
            ControlFlow::Continue
        ));
        let result = vm.current_frame().get_reg(3);
        assert_eq!(result.as_int(), Some(64));
        assert!(!result.is_bool());

        vm.current_frame_mut().set_reg(1, Value::int(8).unwrap());
        vm.current_frame_mut().set_reg(2, Value::bool(true));
        assert!(matches!(
            shr(&mut vm, binary_inst(Opcode::Shr)),
            ControlFlow::Continue
        ));
        let result = vm.current_frame().get_reg(3);
        assert_eq!(result.as_int(), Some(4));
        assert!(!result.is_bool());
    }

    #[test]
    fn test_contains_value_supports_heap_dict_subclasses_with_native_backing() {
        let mut vm = vm_with_frame();
        let mut instance = ShapedObject::new_dict_backed(TypeId::from_raw(600), Shape::empty());
        instance
            .dict_backing_mut()
            .expect("dict backing should exist")
            .set(Value::string(intern("present")), Value::int(1).unwrap());
        let (value, ptr) = boxed_object_value(instance);

        assert_eq!(
            contains_value(&mut vm, Value::string(intern("present")), value).unwrap(),
            true
        );
        assert_eq!(
            contains_value(&mut vm, Value::string(intern("missing")), value).unwrap(),
            false
        );

        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_contains_value_supports_interned_string_containment() {
        let mut vm = vm_with_frame();

        assert!(
            contains_value(
                &mut vm,
                Value::string(intern("a")),
                Value::string(intern("ab"))
            )
            .unwrap()
        );
        assert!(
            !contains_value(
                &mut vm,
                Value::string(intern("z")),
                Value::string(intern("ab"))
            )
            .unwrap()
        );
    }

    #[test]
    fn test_contains_value_supports_bytes_membership_protocol() {
        let mut vm = vm_with_frame();
        let (haystack, haystack_ptr) = boxed_object_value(BytesObject::from_slice(b"abc"));
        let (needle, needle_ptr) = boxed_object_value(BytesObject::from_slice(b"bc"));

        assert!(contains_value(&mut vm, Value::int_unchecked(97), haystack).unwrap());
        assert!(!contains_value(&mut vm, Value::int_unchecked(122), haystack).unwrap());
        assert!(contains_value(&mut vm, needle, haystack).unwrap());

        unsafe { drop_boxed(needle_ptr) };
        unsafe { drop_boxed(haystack_ptr) };
    }

    #[test]
    fn test_membership_uses_user_defined_contains_protocol() {
        let result = execute(
            r#"
class Bucket:
    def __contains__(self, item):
        return item == 42

bucket = Bucket()
assert 42 in bucket
assert 7 not in bucket
"#,
        );

        assert!(result.is_ok(), "membership protocol failed: {result:?}");
    }

    #[test]
    fn test_membership_falls_back_to_iterator_protocol() {
        let result = execute(
            r#"
class Bucket:
    def __iter__(self):
        return iter((1, 2, 3))

bucket = Bucket()
assert 2 in bucket
assert 5 not in bucket
"#,
        );

        assert!(result.is_ok(), "iterator fallback failed: {result:?}");
    }

    #[test]
    fn test_membership_supports_builtin_dict_key_views() {
        let result = execute(
            r#"
view = {"alpha": 1, "beta": 2}.keys()
assert "alpha" in view
assert "gamma" not in view
"#,
        );

        assert!(result.is_ok(), "dict_keys membership failed: {result:?}");
    }

    #[test]
    fn test_membership_supports_iterator_objects_and_consumes_progress() {
        let result = execute(
            r#"
it = iter([1, 2, 3])
assert 2 in it
assert list(it) == [3]
"#,
        );

        assert!(result.is_ok(), "iterator membership failed: {result:?}");
    }

    #[test]
    fn test_membership_treats_identical_nan_value_as_present() {
        let result = execute(
            r#"
needle = float("nan")
items = [needle]
assert needle in items
"#,
        );

        assert!(result.is_ok(), "nan membership identity failed: {result:?}");
    }

    #[test]
    fn test_bool_ordering_uses_int_subtype_semantics() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(false));
        vm.current_frame_mut().set_reg(2, Value::int(1).unwrap());

        assert!(matches!(
            lt(&mut vm, binary_inst(Opcode::Lt)),
            ControlFlow::Continue
        ));
        assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(true));
    }

    #[test]
    fn test_bitwise_and_with_bool_and_int_returns_int() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(true));
        vm.current_frame_mut().set_reg(2, Value::int(1).unwrap());

        assert!(matches!(
            bitwise_and(&mut vm, binary_inst(Opcode::BitwiseAnd)),
            ControlFlow::Continue
        ));
        let result = vm.current_frame().get_reg(3);
        assert_eq!(result.as_int(), Some(1));
        assert!(!result.is_bool());
    }

    #[test]
    fn test_bitwise_xor_with_bool_pair_preserves_bool_result_type() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(true));
        vm.current_frame_mut().set_reg(2, Value::bool(true));

        assert!(matches!(
            bitwise_xor(&mut vm, binary_inst(Opcode::BitwiseXor)),
            ControlFlow::Continue
        ));
        assert_eq!(vm.current_frame().get_reg(3).as_bool(), Some(false));
    }

    #[test]
    fn test_eq_and_ne_use_rich_comparison_protocols() {
        let result = execute(
            r#"
calls = []

class Left:
    def __eq__(self, other):
        calls.append("Left.__eq__")
        return NotImplemented

class Right:
    def __eq__(self, other):
        calls.append("Right.__eq__")
        return True

class OnlyEq:
    def __eq__(self, other):
        return self.value == other.value

class LeftNe:
    def __eq__(self, other):
        calls.append("LeftNe.__eq__")
        return NotImplemented

class RightNe:
    def __eq__(self, other):
        calls.append("RightNe.__eq__")
        return NotImplemented
    def __ne__(self, other):
        calls.append("RightNe.__ne__")
        return NotImplemented

a = Left()
b = Right()
assert a == b
assert calls == ["Left.__eq__", "Right.__eq__"]

x = OnlyEq()
y = OnlyEq()
x.value = 1
y.value = 1
assert (x != y) is False

calls = []
assert LeftNe() != RightNe()
assert calls == ["LeftNe.__eq__", "RightNe.__ne__"]
"#,
        );

        assert!(result.is_ok(), "rich equality protocol failed: {result:?}");
    }

    #[test]
    fn test_native_tuple_subclasses_use_tuple_comparison_semantics() {
        let result = execute(
            r#"
class Pair(tuple):
    pass

left = Pair((1, 2))
same = Pair((1, 2))
larger = Pair((1, 3))

assert left == same
assert not (left != same)
assert left == (1, 2)
assert (1, 2) == left
assert left < larger
assert larger > left
"#,
        );

        assert!(
            result.is_ok(),
            "tuple subclass comparison failed: {result:?}"
        );
    }

    #[test]
    fn test_ordering_uses_rich_comparison_protocols() {
        let result = execute(
            r#"
class Value:
    def __init__(self, value):
        self.value = value
    def __lt__(self, other):
        return self.value < other.value
    def __gt__(self, other):
        return self.value > other.value
    def __le__(self, other):
        return self.value <= other.value
    def __ge__(self, other):
        return self.value >= other.value

assert Value(1) < Value(2)
assert Value(2) > Value(1)
assert Value(2) >= Value(2)
assert Value(2) <= Value(2)
"#,
        );

        assert!(result.is_ok(), "rich ordering protocol failed: {result:?}");
    }

    #[test]
    fn test_membership_falls_back_to_sequence_getitem_protocol() {
        let result = execute(
            r#"
class Bucket:
    def __getitem__(self, index):
        return [1, 2, 3][index]

bucket = Bucket()
assert 2 in bucket
assert 5 not in bucket
"#,
        );

        assert!(
            result.is_ok(),
            "sequence membership fallback failed: {result:?}"
        );
    }

    #[test]
    fn test_tuple_ordering_uses_lexicographic_semantics() {
        let result = execute(
            r#"
assert (1, 2) < (1, 3)
assert (1, 2) <= (1, 2)
assert (2, 0) > (1, 99)
assert (2, 0) >= (2, 0)
"#,
        );

        assert!(result.is_ok(), "tuple ordering failed: {result:?}");
    }
}
