//! Core builtin functions (len, abs, min, max, sum, pow, etc.).

use super::{BuiltinError, BuiltinFunctionObject};
use crate::VirtualMachine;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::iteration::{IterStep, ensure_iterator_value, next_step};
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::python_numeric::int_like_value;
use crate::stdlib::collections::deque::DequeObject;
use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, ToPrimitive, Zero};
use prism_core::Value;
use prism_core::intern::intern;
use prism_core::python_unicode::{is_surrogate_carrier, python_char_escape};
use prism_runtime::object::descriptor::{ClassMethodDescriptor, StaticMethodDescriptor};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::MappingProxyObject;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::complex::ComplexObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::{IntObject, bigint_to_value, int_value_to_string, value_to_bigint};
use prism_runtime::types::list::ListObject;
use prism_runtime::types::memoryview::MemoryViewObject;
use prism_runtime::types::range::RangeObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::string::{object_ptr_as_string_ref, value_as_string_ref};
use prism_runtime::types::tuple::TupleObject;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

// =============================================================================
// len
// =============================================================================

/// Builtin len function.
///
/// Returns the length of an object (list, tuple, string, dict, set, range).
pub fn builtin_len(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "len() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let obj = args[0];

    match exact_len(obj).map_err(super::runtime_error_to_builtin_error)? {
        Some(len) => len_to_value(len, type_name_for_exact_len(obj)),
        None => Err(BuiltinError::TypeError(format!(
            "object of type '{}' has no len()",
            value_type_name(obj)
        ))),
    }
}

/// VM-aware len builtin that honors the Python `__len__` protocol.
pub fn builtin_len_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "len() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let len = try_len_value(vm, args[0]).map_err(super::runtime_error_to_builtin_error)?;
    len_to_value(len, "object")
}

#[inline]
fn len_to_value(len: usize, type_name: &str) -> Result<Value, BuiltinError> {
    if len > isize::MAX as usize {
        return Err(BuiltinError::OverflowError(format!(
            "{} length overflow",
            type_name
        )));
    }

    Ok(bigint_to_value(BigInt::from(len)))
}

pub(crate) fn try_len_value(vm: &mut VirtualMachine, value: Value) -> Result<usize, RuntimeError> {
    if let Some(len) = exact_len(value)? {
        return Ok(len);
    }

    let target = match resolve_special_method(value, "__len__") {
        Ok(target) => target,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return Err(no_len_runtime_error(value));
        }
        Err(err) => return Err(err),
    };

    let len_value = invoke_zero_arg_bound_method(vm, target)?;
    normalize_len_protocol_result(len_value)
}

pub(crate) fn try_length_hint(
    vm: &mut VirtualMachine,
    value: Value,
    default: usize,
) -> Result<usize, RuntimeError> {
    match try_len_value(vm, value) {
        Ok(len) => return Ok(len),
        Err(err) if runtime_error_is_type_error(&err) => {}
        Err(err) => return Err(err),
    }

    let target = match resolve_special_method(value, "__length_hint__") {
        Ok(target) => target,
        Err(err) if err.is_attribute_error() => return Ok(default),
        Err(err) => return Err(err),
    };

    let hint = match invoke_zero_arg_bound_method(vm, target) {
        Ok(value) => value,
        Err(err) if runtime_error_is_type_error(&err) => return Ok(default),
        Err(err) => return Err(err),
    };

    if hint == super::builtin_not_implemented_value() {
        return Ok(default);
    }

    normalize_length_hint_protocol_result(hint)
}

#[inline]
fn exact_len(value: Value) -> Result<Option<usize>, RuntimeError> {
    if let Some(string) = value_as_string_ref(value) {
        return Ok(Some(string.char_count()));
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };

    use crate::ops::objects::extract_type_id;
    let type_id = extract_type_id(ptr);
    if let Some(list) = crate::ops::objects::list_storage_ref_from_ptr(ptr) {
        return Ok(Some(list.len()));
    }
    if let Some(tuple) = crate::ops::objects::tuple_storage_ref_from_ptr(ptr) {
        return Ok(Some(tuple.len()));
    }
    match type_id {
        TypeId::DICT => Ok(Some(unsafe { &*(ptr as *const DictObject) }.len())),
        TypeId::MAPPING_PROXY => {
            let proxy = unsafe { &*(ptr as *const MappingProxyObject) };
            crate::builtins::builtin_mapping_proxy_len(proxy)
                .map(Some)
                .map_err(|err| RuntimeError::type_error(err.to_string()))
        }
        TypeId::SET | TypeId::FROZENSET => Ok(Some(unsafe { &*(ptr as *const SetObject) }.len())),
        TypeId::BYTES | TypeId::BYTEARRAY => {
            Ok(Some(unsafe { &*(ptr as *const BytesObject) }.len()))
        }
        TypeId::MEMORYVIEW => {
            let view = unsafe { &*(ptr as *const MemoryViewObject) };
            if view.released() {
                Err(RuntimeError::value_error(
                    "operation forbidden on released memoryview object",
                ))
            } else {
                Ok(Some(view.len()))
            }
        }
        TypeId::DEQUE => Ok(Some(unsafe { &*(ptr as *const DequeObject) }.len())),
        TypeId::RANGE => {
            let range = unsafe { &*(ptr as *const RangeObject) };
            range
                .try_len()
                .ok_or_else(|| {
                    RuntimeError::new(RuntimeErrorKind::OverflowError {
                        message: "range length overflow".into(),
                    })
                })
                .map(Some)
        }
        _ => Ok(None),
    }
}

#[inline]
fn type_name_for_exact_len(value: Value) -> &'static str {
    if value.is_string() {
        return "str";
    }

    let Some(ptr) = value.as_object_ptr() else {
        return value_type_name(value);
    };

    let type_id = crate::ops::objects::extract_type_id(ptr);
    let has_builtin_sequence_layout = type_id.raw() < TypeId::FIRST_USER_TYPE;
    if has_builtin_sequence_layout && crate::ops::objects::list_storage_ref_from_ptr(ptr).is_some()
    {
        return "list";
    }
    if has_builtin_sequence_layout && crate::ops::objects::tuple_storage_ref_from_ptr(ptr).is_some()
    {
        return "tuple";
    }

    match type_id {
        TypeId::DICT => "dict",
        TypeId::MAPPING_PROXY => "mappingproxy",
        TypeId::SET => "set",
        TypeId::FROZENSET => "frozenset",
        TypeId::BYTES => "bytes",
        TypeId::BYTEARRAY => "bytearray",
        TypeId::MEMORYVIEW => "memoryview",
        TypeId::DEQUE => "deque",
        TypeId::RANGE => "range",
        other => other.name(),
    }
}

#[inline]
fn no_len_runtime_error(value: Value) -> RuntimeError {
    RuntimeError::type_error(format!(
        "object of type '{}' has no len()",
        value_type_name(value)
    ))
}

#[inline]
fn invoke_zero_arg_bound_method(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
) -> Result<Value, RuntimeError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
}

#[inline]
fn normalize_len_protocol_result(value: Value) -> Result<usize, RuntimeError> {
    if let Some(flag) = value.as_bool() {
        return Ok(usize::from(flag));
    }

    if let Some(length) = value_to_bigint(value) {
        if length.sign() == Sign::Minus {
            return Err(RuntimeError::value_error("__len__() should return >= 0"));
        }

        return length.to_usize().ok_or_else(|| {
            RuntimeError::new(RuntimeErrorKind::OverflowError {
                message: "cannot fit 'int' into an index-sized integer".into(),
            })
        });
    }

    Err(RuntimeError::type_error(format!(
        "'{}' object cannot be interpreted as an integer",
        value_type_name(value)
    )))
}

#[inline]
fn normalize_length_hint_protocol_result(value: Value) -> Result<usize, RuntimeError> {
    if let Some(flag) = value.as_bool() {
        return Ok(usize::from(flag));
    }

    if let Some(length) = value_to_bigint(value) {
        if length.sign() == Sign::Minus {
            return Err(RuntimeError::value_error(
                "__length_hint__() should return >= 0",
            ));
        }

        return length.to_usize().ok_or_else(|| {
            RuntimeError::new(RuntimeErrorKind::OverflowError {
                message: "cannot fit 'int' into an index-sized integer".into(),
            })
        });
    }

    Err(RuntimeError::type_error(format!(
        "__length_hint__ must be an integer, not {}",
        value_type_name(value)
    )))
}

#[inline]
fn runtime_error_is_type_error(err: &RuntimeError) -> bool {
    match err.kind() {
        RuntimeErrorKind::TypeError { .. }
        | RuntimeErrorKind::UnsupportedOperandTypes { .. }
        | RuntimeErrorKind::NotCallable { .. }
        | RuntimeErrorKind::NotIterable { .. }
        | RuntimeErrorKind::NotSubscriptable { .. } => true,
        RuntimeErrorKind::Exception { type_id, .. } => {
            *type_id == crate::stdlib::exceptions::ExceptionTypeId::TypeError.as_u8() as u16
        }
        _ => false,
    }
}

// =============================================================================
// abs
// =============================================================================

/// Builtin abs function.
///
/// Returns the absolute value of a number.
pub fn builtin_abs(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "abs() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let arg = args[0];

    if let Some(i) = int_like_value(arg) {
        return Value::int(i.abs()).ok_or_else(|| {
            BuiltinError::OverflowError("integer absolute value overflow".to_string())
        });
    }

    if let Some(f) = arg.as_float() {
        return Ok(Value::float(f.abs()));
    }

    Err(BuiltinError::TypeError(
        "bad operand type for abs(): expected number".to_string(),
    ))
}

// =============================================================================
// min / max
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtremumKind {
    Min,
    Max,
}

impl ExtremumKind {
    #[inline]
    fn name(self) -> &'static str {
        match self {
            Self::Min => "min",
            Self::Max => "max",
        }
    }

    #[inline]
    fn should_replace(self, ordering: Ordering) -> bool {
        match self {
            Self::Min => ordering == Ordering::Less,
            Self::Max => ordering == Ordering::Greater,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ExtremumOptions {
    default: Option<Value>,
    key: Option<Value>,
}

/// Builtin min function.
///
/// Returns the smallest item in an iterable or the smallest of two or more arguments.
pub fn builtin_min(args: &[Value]) -> Result<Value, BuiltinError> {
    select_extreme_without_vm(args, ExtremumKind::Min)
}

/// Builtin max function.
///
/// Returns the largest item in an iterable or the largest of two or more arguments.
pub fn builtin_max(args: &[Value]) -> Result<Value, BuiltinError> {
    select_extreme_without_vm(args, ExtremumKind::Max)
}

/// VM-aware builtin min implementation with CPython-compatible keyword handling.
pub fn builtin_min_vm_kw(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    select_extreme_with_vm(vm, args, keywords, ExtremumKind::Min)
}

/// VM-aware builtin max implementation with CPython-compatible keyword handling.
pub fn builtin_max_vm_kw(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    select_extreme_with_vm(vm, args, keywords, ExtremumKind::Max)
}

fn select_extreme_without_vm(args: &[Value], kind: ExtremumKind) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "{} expected at least 1 argument, got 0",
            kind.name()
        )));
    }

    if args.len() == 1 {
        let mut iterator = super::iter_dispatch::value_to_iterator(&args[0])?;
        let Some(mut best) = iterator.next() else {
            return Err(BuiltinError::ValueError(format!(
                "{}() arg is an empty sequence",
                kind.name()
            )));
        };

        while let Some(candidate) = iterator.next() {
            let ordering = compare_extreme_values(candidate, best)?;
            if kind.should_replace(ordering) {
                best = candidate;
            }
        }

        return Ok(best);
    }

    let mut best = args[0];
    for &candidate in &args[1..] {
        let ordering = compare_extreme_values(candidate, best)?;
        if kind.should_replace(ordering) {
            best = candidate;
        }
    }
    Ok(best)
}

fn select_extreme_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
    kind: ExtremumKind,
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "{} expected at least 1 argument, got 0",
            kind.name()
        )));
    }

    let options = parse_extremum_keywords(kind, args.len(), keywords)?;

    if args.len() == 1 {
        return select_extreme_from_iterable(vm, args[0], options, kind);
    }

    select_extreme_from_slice(vm, args, options, kind)
}

fn parse_extremum_keywords(
    kind: ExtremumKind,
    positional_count: usize,
    keywords: &[(&str, Value)],
) -> Result<ExtremumOptions, BuiltinError> {
    let mut options = ExtremumOptions::default();
    let mut default_seen = false;
    let mut key_seen = false;

    for (name, value) in keywords {
        match *name {
            "default" => {
                if default_seen {
                    return Err(BuiltinError::TypeError(format!(
                        "{}() got multiple values for keyword argument 'default'",
                        kind.name()
                    )));
                }
                options.default = Some(*value);
                default_seen = true;
            }
            "key" => {
                if key_seen {
                    return Err(BuiltinError::TypeError(format!(
                        "{}() got multiple values for keyword argument 'key'",
                        kind.name()
                    )));
                }
                if !value.is_none() {
                    options.key = Some(*value);
                }
                key_seen = true;
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "{}() got an unexpected keyword argument '{}'",
                    kind.name(),
                    other
                )));
            }
        }
    }

    if positional_count > 1 && options.default.is_some() {
        return Err(BuiltinError::TypeError(format!(
            "Cannot specify a default for {}() with multiple positional arguments",
            kind.name()
        )));
    }

    Ok(options)
}

fn select_extreme_from_iterable(
    vm: &mut VirtualMachine,
    iterable: Value,
    options: ExtremumOptions,
    kind: ExtremumKind,
) -> Result<Value, BuiltinError> {
    let iterator =
        ensure_iterator_value(vm, iterable).map_err(super::runtime_error_to_builtin_error)?;
    let Some((mut best_value, mut best_key)) = next_extreme_candidate(vm, iterator, options.key)?
    else {
        return options.default.ok_or_else(|| {
            BuiltinError::ValueError(format!("{}() arg is an empty sequence", kind.name()))
        });
    };

    loop {
        let Some((candidate_value, candidate_key)) =
            next_extreme_candidate(vm, iterator, options.key)?
        else {
            return Ok(best_value);
        };

        let ordering = compare_extreme_values_with_vm(vm, candidate_key, best_key)?;
        if kind.should_replace(ordering) {
            best_value = candidate_value;
            best_key = candidate_key;
        }
    }
}

fn select_extreme_from_slice(
    vm: &mut VirtualMachine,
    values: &[Value],
    options: ExtremumOptions,
    kind: ExtremumKind,
) -> Result<Value, BuiltinError> {
    let mut best_value = values[0];
    let mut best_key = apply_extremum_key(vm, options.key, best_value)?;

    for &candidate_value in &values[1..] {
        let candidate_key = apply_extremum_key(vm, options.key, candidate_value)?;
        let ordering = compare_extreme_values_with_vm(vm, candidate_key, best_key)?;
        if kind.should_replace(ordering) {
            best_value = candidate_value;
            best_key = candidate_key;
        }
    }

    Ok(best_value)
}

fn next_extreme_candidate(
    vm: &mut VirtualMachine,
    iterator: Value,
    key: Option<Value>,
) -> Result<Option<(Value, Value)>, BuiltinError> {
    match next_step(vm, iterator).map_err(super::runtime_error_to_builtin_error)? {
        IterStep::Yielded(value) => {
            let key_value = apply_extremum_key(vm, key, value)?;
            Ok(Some((value, key_value)))
        }
        IterStep::Exhausted => Ok(None),
    }
}

#[inline]
fn apply_extremum_key(
    vm: &mut VirtualMachine,
    key: Option<Value>,
    value: Value,
) -> Result<Value, BuiltinError> {
    match key {
        Some(key_func) => invoke_callable_value(vm, key_func, &[value])
            .map_err(super::runtime_error_to_builtin_error),
        None => Ok(value),
    }
}

#[inline]
fn compare_extreme_values(left: Value, right: Value) -> Result<Ordering, BuiltinError> {
    primitive_extreme_ordering(left, right).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "'<' not supported between instances of '{}' and '{}'",
            left.type_name(),
            right.type_name()
        ))
    })
}

#[inline]
fn compare_extreme_values_with_vm(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
) -> Result<Ordering, BuiltinError> {
    if let Some(ordering) = primitive_extreme_ordering(left, right) {
        return Ok(ordering);
    }

    let left_lt = rich_extremum_lt(vm, left, right)?;
    if left_lt == Some(true) {
        return Ok(Ordering::Less);
    }

    let right_lt = rich_extremum_lt(vm, right, left)?;
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
fn primitive_extreme_ordering(left: Value, right: Value) -> Option<Ordering> {
    if left == right {
        return Some(Ordering::Equal);
    }

    if let Some(ordering) = compare_numeric_extreme_values(left, right) {
        return Some(ordering);
    }

    match (value_as_string_ref(left), value_as_string_ref(right)) {
        (Some(left), Some(right)) => Some(left.as_str().cmp(right.as_str())),
        _ => None,
    }
}

#[inline]
fn compare_numeric_extreme_values(left: Value, right: Value) -> Option<Ordering> {
    if let (Some(left_int), Some(right_int)) = (
        numeric_value_to_bigint(left),
        numeric_value_to_bigint(right),
    ) {
        return Some(left_int.cmp(&right_int));
    }

    match (numeric_value_to_f64(left), numeric_value_to_f64(right)) {
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
fn numeric_value_to_bigint(value: Value) -> Option<BigInt> {
    if let Some(boolean) = value.as_bool() {
        return Some(BigInt::from(u8::from(boolean)));
    }

    value_to_bigint(value)
}

#[inline]
fn numeric_value_to_f64(value: Value) -> Option<f64> {
    value.as_float()
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
fn rich_extremum_lt(
    vm: &mut VirtualMachine,
    left: Value,
    right: Value,
) -> Result<Option<bool>, BuiltinError> {
    match resolve_special_method(left, "__lt__") {
        Ok(target) => {
            let result = invoke_extremum_comparison_method(vm, target, right)?;
            if result == super::builtin_not_implemented_value() {
                return Ok(None);
            }
            crate::truthiness::try_is_truthy(vm, result)
                .map(Some)
                .map_err(super::runtime_error_to_builtin_error)
        }
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => Ok(None),
        Err(err) => Err(super::runtime_error_to_builtin_error(err)),
    }
}

#[inline]
fn invoke_extremum_comparison_method(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
    operand: Value,
) -> Result<Value, BuiltinError> {
    match target.implicit_self {
        Some(implicit_self) => {
            invoke_callable_value(vm, target.callable, &[implicit_self, operand])
                .map_err(super::runtime_error_to_builtin_error)
        }
        None => invoke_callable_value(vm, target.callable, &[operand])
            .map_err(super::runtime_error_to_builtin_error),
    }
}

// =============================================================================
// sum
// =============================================================================

/// Builtin sum function.
///
/// Sums the items of an iterable, left to right, with optional start value.
pub fn builtin_sum(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "sum expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let mut acc = if args.len() == 2 {
        NumericAccumulator::from_start(args[1])?
    } else {
        NumericAccumulator::SmallInt(0)
    };

    if let Some(iter) = super::iter_dispatch::get_iterator_mut(&args[0]) {
        while let Some(item) = iter.next() {
            acc.add(item)?;
        }
    } else {
        let mut iter =
            super::iter_dispatch::value_to_iterator(&args[0]).map_err(BuiltinError::from)?;
        while let Some(item) = iter.next() {
            acc.add(item)?;
        }
    }

    Ok(acc.into_value())
}

/// VM-aware sum builtin.
pub fn builtin_sum_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_sum_vm_kw(vm, args, &[])
}

/// VM-aware sum builtin with CPython-compatible keyword handling.
pub fn builtin_sum_vm_kw(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if keywords.is_empty() {
        return builtin_sum_vm_positional(vm, args);
    }

    let args = parse_sum_args(args, keywords)?;
    builtin_sum_vm_positional(vm, &args)
}

#[inline]
fn builtin_sum_vm_positional(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "sum expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let mut acc = if args.len() == 2 {
        NumericAccumulator::from_start(args[1])?
    } else {
        NumericAccumulator::SmallInt(0)
    };

    let iterator =
        ensure_iterator_value(vm, args[0]).map_err(super::runtime_error_to_builtin_error)?;
    loop {
        match next_step(vm, iterator).map_err(super::runtime_error_to_builtin_error)? {
            IterStep::Yielded(item) => acc.add(item)?,
            IterStep::Exhausted => return acc.into_value_in_vm(vm),
        }
    }
}

#[inline]
fn parse_sum_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<SmallVec<[Value; 2]>, BuiltinError> {
    let mut parsed = SmallVec::<[Value; 2]>::new();
    parsed.extend_from_slice(args);

    for &(name, value) in keywords {
        match name {
            "start" => {
                if parsed.len() >= 2 {
                    return Err(BuiltinError::TypeError(
                        "sum() got multiple values for argument 'start'".to_string(),
                    ));
                }
                parsed.push(value);
            }
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "sum() got an unexpected keyword argument '{name}'"
                )));
            }
        }
    }

    Ok(parsed)
}

#[derive(Clone)]
enum NumericAccumulator {
    SmallInt(i64),
    BigInt(BigInt),
    Float(f64),
}

impl NumericAccumulator {
    #[inline]
    fn from_start(value: Value) -> Result<Self, BuiltinError> {
        if value.is_string() || is_str_object(value) {
            return Err(BuiltinError::TypeError(
                "sum() can't sum strings [use ''.join(seq) instead]".to_string(),
            ));
        }

        if let Some(f) = value.as_float() {
            return Ok(Self::Float(f));
        }
        if let Some(bigint) = numeric_value_to_bigint(value) {
            if let Some(i) = bigint.to_i64()
                && Value::int(i).is_some()
            {
                return Ok(Self::SmallInt(i));
            }
            return Ok(Self::BigInt(bigint));
        }

        Err(BuiltinError::TypeError(format!(
            "sum() start value must be a number, not {}",
            value_type_name(value)
        )))
    }

    #[inline]
    fn add(&mut self, value: Value) -> Result<(), BuiltinError> {
        let rhs = if let Some(f) = value.as_float() {
            NumericAccumulator::Float(f)
        } else if value.is_string() || is_str_object(value) {
            return Err(BuiltinError::TypeError(
                "sum() can't sum strings [use ''.join(seq) instead]".to_string(),
            ));
        } else if let Some(bigint) = numeric_value_to_bigint(value) {
            if let Some(i) = bigint.to_i64()
                && Value::int(i).is_some()
            {
                NumericAccumulator::SmallInt(i)
            } else {
                NumericAccumulator::BigInt(bigint)
            }
        } else {
            return Err(BuiltinError::TypeError(format!(
                "unsupported operand type(s) for +: '{}' and '{}'",
                self.type_name(),
                value_type_name(value)
            )));
        };

        match (&*self, rhs) {
            (NumericAccumulator::SmallInt(lhs), NumericAccumulator::SmallInt(rhs_i)) => {
                if let Some(sum) = lhs.checked_add(rhs_i) {
                    *self = if Value::int(sum).is_some() {
                        NumericAccumulator::SmallInt(sum)
                    } else {
                        NumericAccumulator::BigInt(BigInt::from(sum))
                    };
                } else {
                    *self = NumericAccumulator::BigInt(BigInt::from(*lhs) + BigInt::from(rhs_i));
                }
            }
            (NumericAccumulator::SmallInt(lhs), NumericAccumulator::BigInt(rhs_big)) => {
                *self = NumericAccumulator::BigInt(BigInt::from(*lhs) + rhs_big);
            }
            (NumericAccumulator::BigInt(lhs_big), NumericAccumulator::SmallInt(rhs_i)) => {
                *self = NumericAccumulator::BigInt(lhs_big + BigInt::from(rhs_i));
            }
            (NumericAccumulator::BigInt(lhs_big), NumericAccumulator::BigInt(rhs_big)) => {
                *self = NumericAccumulator::BigInt(lhs_big + rhs_big);
            }
            (NumericAccumulator::SmallInt(lhs), NumericAccumulator::Float(rhs_f)) => {
                *self = NumericAccumulator::Float(*lhs as f64 + rhs_f);
            }
            (NumericAccumulator::Float(lhs), NumericAccumulator::SmallInt(rhs_i)) => {
                *self = NumericAccumulator::Float(*lhs + rhs_i as f64);
            }
            (NumericAccumulator::BigInt(lhs_big), NumericAccumulator::Float(rhs_f)) => {
                let lhs = bigint_to_finite_f64(lhs_big)?;
                *self = NumericAccumulator::Float(lhs + rhs_f);
            }
            (NumericAccumulator::Float(lhs), NumericAccumulator::BigInt(rhs_big)) => {
                let rhs = bigint_to_finite_f64(&rhs_big)?;
                *self = NumericAccumulator::Float(*lhs + rhs);
            }
            (NumericAccumulator::Float(lhs), NumericAccumulator::Float(rhs_f)) => {
                *self = NumericAccumulator::Float(*lhs + rhs_f);
            }
        }
        Ok(())
    }

    #[inline]
    fn type_name(&self) -> &'static str {
        match self {
            NumericAccumulator::SmallInt(_) | NumericAccumulator::BigInt(_) => "int",
            NumericAccumulator::Float(_) => "float",
        }
    }

    #[inline]
    fn into_value(self) -> Value {
        match self {
            NumericAccumulator::SmallInt(i) => Value::int(i).expect("small accumulator must fit"),
            NumericAccumulator::BigInt(value) => bigint_to_value(value),
            NumericAccumulator::Float(f) => Value::float(f),
        }
    }

    #[inline]
    fn into_value_in_vm(self, vm: &VirtualMachine) -> Result<Value, BuiltinError> {
        match self {
            NumericAccumulator::SmallInt(i) => {
                Ok(Value::int(i).expect("small accumulator must fit"))
            }
            NumericAccumulator::BigInt(value) => {
                bigint_to_value_in_vm(vm, value).map_err(super::runtime_error_to_builtin_error)
            }
            NumericAccumulator::Float(f) => Ok(Value::float(f)),
        }
    }
}

#[inline]
fn is_str_object(value: Value) -> bool {
    value
        .as_object_ptr()
        .and_then(object_ptr_as_string_ref)
        .is_some()
}

#[inline]
fn bigint_to_finite_f64(value: &BigInt) -> Result<f64, BuiltinError> {
    let converted = value.to_f64().ok_or_else(|| {
        BuiltinError::OverflowError("int too large to convert to float".to_string())
    })?;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(BuiltinError::OverflowError(
            "int too large to convert to float".to_string(),
        ))
    }
}

#[inline]
fn bigint_to_value_in_vm(vm: &VirtualMachine, value: BigInt) -> Result<Value, RuntimeError> {
    if let Some(i) = value.to_i64()
        && let Some(inline) = Value::int(i)
    {
        return Ok(inline);
    }

    vm.allocator()
        .alloc_value(IntObject::new(value))
        .ok_or_else(|| RuntimeError::internal("out of memory: failed to allocate int"))
}

#[inline]
fn value_type_name(value: Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.as_bool().is_some() {
        "bool"
    } else if value.as_int().is_some() {
        "int"
    } else if value.as_float().is_some() {
        "float"
    } else if value.is_string() {
        "str"
    } else if let Some(ptr) = value.as_object_ptr() {
        crate::ops::objects::extract_type_id(ptr).name()
    } else {
        "object"
    }
}

// =============================================================================
// pow
// =============================================================================

/// Builtin pow function.
///
/// pow(base, exp[, mod]) - Compute base**exp, optionally modulo mod.
pub fn builtin_pow(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_pow_impl(None, args)
}

/// VM-aware pow builtin that allocates promoted integer results in the managed heap.
pub fn builtin_pow_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_pow_impl(Some(vm), args)
}

fn builtin_pow_impl(vm: Option<&VirtualMachine>, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "pow expected 2 or 3 arguments, got {}",
            args.len()
        )));
    }

    let base = args[0];
    let exp = args[1];

    if let (Some(base_int), Some(exp_int)) =
        (numeric_value_to_bigint(base), numeric_value_to_bigint(exp))
    {
        if args.len() == 3 {
            let modulus = numeric_value_to_bigint(args[2]).ok_or_else(|| {
                BuiltinError::TypeError(
                    "pow() 3rd argument not allowed unless all arguments are integers".to_string(),
                )
            })?;

            let result = pow_bigint_with_modulus(base_int, exp_int, modulus)?;
            return bigint_result_to_value(vm, result);
        }

        if exp_int.is_negative() {
            if base_int.is_zero() {
                return Err(BuiltinError::Raised(
                    RuntimeError::zero_division_with_message(
                        "0.0 cannot be raised to a negative power",
                    ),
                ));
            }

            let base_float = bigint_to_finite_f64(&base_int)?;
            let exp_float = bigint_to_finite_f64(&exp_int)?;
            return Ok(Value::float(base_float.powf(exp_float)));
        }

        let result = pow_bigint_non_negative(&base_int, &exp_int);
        return bigint_result_to_value(vm, result);
    }

    if args.len() == 3 {
        return Err(BuiltinError::TypeError(
            "pow() 3rd argument not allowed unless all arguments are integers".to_string(),
        ));
    }

    if let (Some(b), Some(e)) = (base.as_float_coerce(), exp.as_float_coerce()) {
        let result = b.powf(e);
        return Ok(Value::float(result));
    }

    Err(BuiltinError::TypeError(
        "pow() arguments must be numeric".to_string(),
    ))
}

#[inline]
fn bigint_result_to_value(
    vm: Option<&VirtualMachine>,
    value: BigInt,
) -> Result<Value, BuiltinError> {
    match vm {
        Some(vm) => bigint_to_value_in_vm(vm, value).map_err(super::runtime_error_to_builtin_error),
        None => Ok(bigint_to_value(value)),
    }
}

fn pow_bigint_non_negative(base: &BigInt, exponent: &BigInt) -> BigInt {
    if let Some(exp_u32) = exponent.to_u32() {
        return base.pow(exp_u32);
    }

    let mut exponent = exponent.clone();
    let mut factor = base.clone();
    let mut result = BigInt::one();

    while !exponent.is_zero() {
        if (&exponent & BigInt::one()) == BigInt::one() {
            result *= &factor;
        }
        exponent >>= 1usize;
        if !exponent.is_zero() {
            factor = &factor * &factor;
        }
    }

    result
}

fn pow_bigint_with_modulus(
    base: BigInt,
    exponent: BigInt,
    modulus: BigInt,
) -> Result<BigInt, BuiltinError> {
    if modulus.is_zero() {
        return Err(BuiltinError::ValueError(
            "pow() 3rd argument cannot be 0".to_string(),
        ));
    }

    let modulus_negative = modulus.sign() == Sign::Minus;
    let modulus_abs = modulus.abs();
    let base = mod_floor_bigint(&base, &modulus_abs);

    let result = if exponent.is_negative() {
        let inverse = base.modinv(&modulus_abs).ok_or_else(|| {
            BuiltinError::ValueError("base is not invertible for the given modulus".to_string())
        })?;
        inverse.modpow(&(-exponent), &modulus_abs)
    } else {
        base.modpow(&exponent, &modulus_abs)
    };

    if modulus_negative && !result.is_zero() {
        Ok(result - modulus_abs)
    } else {
        Ok(result)
    }
}

fn mod_floor_bigint(value: &BigInt, modulus: &BigInt) -> BigInt {
    let remainder = value % modulus;
    if remainder.is_zero() || remainder.sign() == modulus.sign() {
        remainder
    } else {
        remainder + modulus
    }
}

// =============================================================================
// round
// =============================================================================

/// Builtin round function.
///
/// round(number[, ndigits]) - Round a number to a given precision.
pub fn builtin_round(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "round expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let ndigits = if args.len() == 2 {
        args[1].as_int().unwrap_or(0)
    } else {
        0
    };

    if let Some(i) = args[0].as_int() {
        // Rounding an integer with no ndigits returns the integer
        if ndigits >= 0 {
            return Ok(args[0]);
        }
        // Negative ndigits: round to nearest 10^(-ndigits)
        let factor = 10i64.pow((-ndigits) as u32);
        let rounded = ((i as f64 / factor as f64).round() * factor as f64) as i64;
        return Value::int(rounded)
            .ok_or_else(|| BuiltinError::OverflowError("integer overflow in round".to_string()));
    }

    if let Some(f) = args[0].as_float() {
        if ndigits == 0 {
            // Round to integer
            let rounded = f.round();
            if rounded >= i64::MIN as f64 && rounded <= i64::MAX as f64 {
                return Value::int(rounded as i64).ok_or_else(|| {
                    BuiltinError::OverflowError("integer overflow in round".to_string())
                });
            }
            return Ok(Value::float(rounded));
        }
        // Round to ndigits decimal places
        let factor = 10f64.powi(ndigits as i32);
        let rounded = (f * factor).round() / factor;
        return Ok(Value::float(rounded));
    }

    Err(BuiltinError::TypeError(
        "round() argument must be a number".to_string(),
    ))
}

// =============================================================================
// divmod
// =============================================================================

/// Builtin divmod function.
///
/// divmod(a, b) - Return (a // b, a % b) as a tuple.
pub fn builtin_divmod(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "divmod expected 2 arguments, got {}",
            args.len()
        )));
    }

    // Integer divmod (includes bool as int subclass semantics).
    if args[0].as_float().is_none() && args[1].as_float().is_none() {
        let a = if let Some(i) = args[0].as_int() {
            i
        } else if let Some(b) = args[0].as_bool() {
            if b { 1 } else { 0 }
        } else {
            return Err(BuiltinError::TypeError(
                "divmod() arguments must be numeric".to_string(),
            ));
        };
        let b = if let Some(i) = args[1].as_int() {
            i
        } else if let Some(b) = args[1].as_bool() {
            if b { 1 } else { 0 }
        } else {
            return Err(BuiltinError::TypeError(
                "divmod() arguments must be numeric".to_string(),
            ));
        };

        if b == 0 {
            return Err(BuiltinError::ValueError(
                "integer division or modulo by zero".to_string(),
            ));
        }

        let mut quotient = a / b;
        let mut remainder = a % b;
        if remainder != 0 && (remainder < 0) != (b < 0) {
            quotient -= 1;
            remainder += b;
        }
        let q = Value::int(quotient)
            .ok_or_else(|| BuiltinError::OverflowError("integer overflow in divmod".to_string()))?;
        let r = Value::int(remainder)
            .ok_or_else(|| BuiltinError::OverflowError("integer overflow in divmod".to_string()))?;
        return Ok(make_tuple2(q, r));
    }

    let a = if let Some(f) = args[0].as_float() {
        f
    } else if let Some(i) = args[0].as_int() {
        i as f64
    } else if let Some(b) = args[0].as_bool() {
        if b { 1.0 } else { 0.0 }
    } else {
        return Err(BuiltinError::TypeError(
            "divmod() arguments must be numeric".to_string(),
        ));
    };
    let b = if let Some(f) = args[1].as_float() {
        f
    } else if let Some(i) = args[1].as_int() {
        i as f64
    } else if let Some(b) = args[1].as_bool() {
        if b { 1.0 } else { 0.0 }
    } else {
        return Err(BuiltinError::TypeError(
            "divmod() arguments must be numeric".to_string(),
        ));
    };

    if b == 0.0 {
        return Err(BuiltinError::ValueError(
            "float divmod() by zero".to_string(),
        ));
    }

    // Python semantics: floor division quotient and remainder with sign of divisor.
    let quotient = (a / b).floor();
    let remainder = a - quotient * b;
    return Ok(make_tuple2(Value::float(quotient), Value::float(remainder)));
}

#[inline]
fn make_tuple2(a: Value, b: Value) -> Value {
    let tuple = TupleObject::from_slice(&[a, b]);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    Value::object_ptr(ptr)
}

// =============================================================================
// hash
// =============================================================================

/// Builtin hash function.
///
/// hash(object) - Return the hash value of the object.
pub fn builtin_hash(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "hash() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let hash = hash_value(args[0])?;
    Ok(bigint_to_value(BigInt::from(hash)))
}

#[inline]
fn hash_value(value: Value) -> Result<i64, BuiltinError> {
    if let Some(boolean) = value.as_bool() {
        return Ok(i64::from(boolean));
    }

    if let Some(integer) = value.as_int() {
        return Ok(normalize_python_hash(integer));
    }

    if let Some(float) = value.as_float() {
        return Ok(hash_float(float, value.raw_bits() as usize));
    }

    if let Some(integer) = value_to_bigint(value) {
        return Ok(hash_bigint(&integer));
    }

    if let Some(string) = value_as_string_ref(value) {
        return Ok(hash_with_default_hasher(&string.as_str()));
    }

    if value.is_none() {
        return Ok(hash_with_default_hasher(&value));
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("unhashable type".to_string()))?;
    let type_id = crate::ops::objects::extract_type_id(ptr);
    match type_id {
        TypeId::LIST | TypeId::DICT | TypeId::SET => Err(BuiltinError::TypeError(format!(
            "unhashable type: '{}'",
            type_id.name()
        ))),
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            hash_tuple(tuple)
        }
        TypeId::SLICE => {
            let slice = unsafe { &*(ptr as *const SliceObject) };
            let components = TupleObject::from_slice(&[
                slice.start_value(),
                slice.stop_value(),
                slice.step_value(),
            ]);
            hash_tuple(&components)
        }
        _ => Ok(hash_with_default_hasher(&(ptr as usize))),
    }
}

#[inline]
fn hash_tuple(tuple: &TupleObject) -> Result<i64, BuiltinError> {
    const XXPRIME_1: u64 = 11_400_714_785_074_694_791;
    const XXPRIME_2: u64 = 14_029_467_366_897_019_727;
    const XXPRIME_5: u64 = 2_870_177_450_012_600_261;
    const EMPTY_TUPLE_COMPAT_MANGLE: u64 = XXPRIME_5 ^ 3_527_539;

    let mut acc: u64 = XXPRIME_5;
    let len = tuple.len();
    for item in tuple.iter().copied() {
        let lane = hash_value(item)? as u64;
        acc = acc.wrapping_add(lane.wrapping_mul(XXPRIME_2));
        acc = acc.rotate_left(31);
        acc = acc.wrapping_mul(XXPRIME_1);
    }

    acc = acc.wrapping_add((len as u64) ^ EMPTY_TUPLE_COMPAT_MANGLE);
    if acc == u64::MAX {
        return Ok(1_546_275_796);
    }
    Ok(acc as i64)
}

#[inline]
fn hash_bigint(value: &BigInt) -> i64 {
    const HASH_MODULUS: u64 = (1u64 << 61) - 1;

    let modulus = BigInt::from(HASH_MODULUS);
    let magnitude = (value.abs() % modulus)
        .to_i64()
        .expect("hash modulus result always fits i64");
    let signed = if value.sign() == Sign::Minus {
        -magnitude
    } else {
        magnitude
    };
    normalize_python_hash(signed)
}

#[inline]
fn hash_float(value: f64, identity: usize) -> i64 {
    const HASH_BITS: i32 = 61;
    const HASH_MODULUS: u64 = (1u64 << HASH_BITS) - 1;
    const HASH_INF: i64 = 314_159;

    if !value.is_finite() {
        return if value.is_infinite() {
            if value.is_sign_positive() {
                HASH_INF
            } else {
                -HASH_INF
            }
        } else {
            hash_pointer(identity)
        };
    }

    let (mut mantissa, mut exponent) = libm::frexp(value);
    let sign = if mantissa < 0.0 {
        mantissa = -mantissa;
        -1
    } else {
        1
    };

    let mut acc = 0u64;
    while mantissa != 0.0 {
        acc = ((acc << 28) & HASH_MODULUS) | (acc >> (HASH_BITS as u32 - 28));
        mantissa *= 268_435_456.0;
        exponent -= 28;
        let lane = mantissa as u64;
        mantissa -= lane as f64;
        acc += lane;
        if acc >= HASH_MODULUS {
            acc -= HASH_MODULUS;
        }
    }

    let exponent = if exponent >= 0 {
        exponent % HASH_BITS
    } else {
        HASH_BITS - 1 - ((-1 - exponent) % HASH_BITS)
    } as u32;
    acc = ((acc << exponent) & HASH_MODULUS) | (acc >> (HASH_BITS as u32 - exponent));

    let signed = if sign < 0 { -(acc as i64) } else { acc as i64 };
    normalize_python_hash(signed)
}

#[inline]
fn hash_pointer(pointer: usize) -> i64 {
    let rotated = pointer.rotate_right(4);
    normalize_python_hash(rotated as isize as i64)
}

#[inline]
fn hash_with_default_hasher<T: Hash>(value: &T) -> i64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    normalize_python_hash(hasher.finish() as i64)
}

#[inline]
fn normalize_python_hash(hash: i64) -> i64 {
    if hash == -1 { -2 } else { hash }
}

// =============================================================================
// id
// =============================================================================

/// Builtin id function.
///
/// id(object) - Return the identity of an object (its memory address).
pub fn builtin_id(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "id() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let obj = args[0];

    // For objects, return pointer address
    if let Some(ptr) = obj.as_object_ptr() {
        return Value::int(ptr as i64)
            .ok_or_else(|| BuiltinError::OverflowError("id overflow".to_string()));
    }
    if let Some(ptr) = obj.as_string_object_ptr() {
        return Value::int(ptr as i64)
            .ok_or_else(|| BuiltinError::OverflowError("id overflow".to_string()));
    }

    // For primitives, compute a stable ID based on the actual value
    // Python semantics: identical primitive values may share ID (interning)
    if obj.is_none() {
        return Ok(Value::int(0x_DEAD_BEEF).unwrap());
    }
    if let Some(b) = obj.as_bool() {
        return Ok(Value::int(if b { 0x_1 } else { 0x_0 }).unwrap());
    }
    if let Some(i) = obj.as_int() {
        // Small integers get stable IDs (like CPython's -5 to 256 cache)
        return Value::int(i).ok_or_else(|| BuiltinError::OverflowError("id overflow".to_string()));
    }
    if let Some(f) = obj.as_float() {
        // Use float bits as ID
        return Value::int(f.to_bits() as i64)
            .ok_or_else(|| BuiltinError::OverflowError("id overflow".to_string()));
    }

    // Fallback
    Ok(Value::int(0).unwrap())
}

// =============================================================================
// callable
// =============================================================================

/// Builtin callable function.
///
/// callable(object) - Return True if the object appears callable.
pub fn builtin_callable(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "callable() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let obj = args[0];

    // Primitives are not callable
    if obj.is_none() || obj.is_bool() || obj.is_int() || obj.is_float() {
        return Ok(Value::bool(false));
    }
    if obj.is_string() {
        return Ok(Value::bool(false));
    }

    Ok(Value::bool(
        crate::ops::calls::value_supports_call_protocol(obj),
    ))
}

// =============================================================================
// repr / ascii
// =============================================================================

/// Builtin repr function.
///
/// repr(object) - Return a string containing a printable representation.
pub fn builtin_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "repr() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let repr = repr_value(args[0])?;
    Ok(Value::string(intern(&repr)))
}

/// VM-aware repr(object) that honors Python-level `__repr__` on heap classes.
pub fn builtin_repr_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "repr() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let repr = repr_value_vm(vm, args[0])?;
    Ok(Value::string(intern(&repr)))
}

/// Builtin ascii function.
///
/// ascii(object) - Like repr() but escape non-ASCII characters.
pub fn builtin_ascii(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "ascii() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let repr = repr_value(args[0])?;
    let ascii = escape_non_ascii(&repr);
    Ok(Value::string(intern(&ascii)))
}

/// VM-aware ascii(object) that honors Python-level `__repr__` on heap classes.
pub fn builtin_ascii_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "ascii() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let repr = repr_value_vm(vm, args[0])?;
    let ascii = escape_non_ascii(&repr);
    Ok(Value::string(intern(&ascii)))
}

const MAX_REPR_DEPTH: usize = 64;

#[derive(Default)]
struct ReprState {
    active: SmallVec<[usize; 16]>,
    depth: usize,
}

impl ReprState {
    fn repr_value(&mut self, value: Value) -> Result<String, BuiltinError> {
        if self.depth >= MAX_REPR_DEPTH {
            return Err(BuiltinError::Raised(RuntimeError::recursion_error(
                self.depth,
            )));
        }

        self.depth += 1;
        let result = self.repr_value_inner(value);
        self.depth -= 1;
        result
    }

    fn repr_value_inner(&mut self, value: Value) -> Result<String, BuiltinError> {
        if value.is_none() {
            return Ok("None".to_string());
        }
        if let Some(b) = value.as_bool() {
            return Ok(if b {
                "True".to_string()
            } else {
                "False".to_string()
            });
        }
        if let Some(int_repr) = int_value_to_string(value) {
            return Ok(int_repr);
        }
        if let Some(f) = value.as_float() {
            if f.fract() == 0.0 && f.is_finite() {
                return Ok(format!("{:.1}", f));
            }
            return Ok(f.to_string());
        }
        if let Some(string) = value_as_string_ref(value) {
            return Ok(quote_python_string(string.as_str()));
        }
        if let Some(text) = crate::builtins::exception_repr_text_for_value(value) {
            return Ok(text);
        }
        if let Some(text) = crate::stdlib::_thread::native_thread_object_repr(value) {
            return Ok(text);
        }

        let ptr = value
            .as_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("invalid object reference".to_string()))?;
        if crate::ops::objects::list_storage_ref_from_ptr(ptr).is_some() {
            return self.repr_list_like(ptr);
        }

        let type_id = crate::ops::objects::extract_type_id(ptr);
        match type_id {
            TypeId::BYTES => {
                let bytes = unsafe { &*(ptr as *const BytesObject) };
                Ok(quote_python_bytes(bytes.as_bytes()))
            }
            TypeId::BYTEARRAY => {
                let bytes = unsafe { &*(ptr as *const BytesObject) };
                Ok(format!(
                    "bytearray({})",
                    quote_python_bytes(bytes.as_bytes())
                ))
            }
            TypeId::MEMORYVIEW => {
                let view = unsafe { &*(ptr as *const MemoryViewObject) };
                Ok(format!(
                    "<memory at 0x{:x}; format {}; len {}>",
                    ptr as usize,
                    view.format_str(),
                    view.len()
                ))
            }
            TypeId::TUPLE => self.with_container(ptr, "(...)", |state| {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                state.repr_tuple(tuple)
            }),
            TypeId::DICT => self.with_container(ptr, "{...}", |state| {
                let dict = unsafe { &*(ptr as *const DictObject) };
                state.repr_dict(dict)
            }),
            TypeId::SET => self.with_container(ptr, "{...}", |state| {
                let set = unsafe { &*(ptr as *const SetObject) };
                state.repr_set(set)
            }),
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };
                Ok(range.to_string())
            }
            TypeId::SLICE => {
                let slice = unsafe { &*(ptr as *const SliceObject) };
                Ok(format!(
                    "slice({}, {}, {})",
                    self.repr_value(slice.start_value())?,
                    self.repr_value(slice.stop_value())?,
                    self.repr_value(slice.step_value())?
                ))
            }
            TypeId::COMPLEX => {
                let complex = unsafe { &*(ptr as *const ComplexObject) };
                Ok(complex.to_string())
            }
            TypeId::STATICMETHOD => {
                let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };
                Ok(format!(
                    "<staticmethod({})>",
                    self.repr_value(descriptor.function())?
                ))
            }
            TypeId::CLASSMETHOD => {
                let descriptor = unsafe { &*(ptr as *const ClassMethodDescriptor) };
                Ok(format!(
                    "<classmethod({})>",
                    self.repr_value(descriptor.function())?
                ))
            }
            TypeId::BUILTIN_FUNCTION => {
                let function = unsafe { &*(ptr as *const BuiltinFunctionObject) };
                Ok(format!(
                    "<built-in function {}>",
                    builtin_function_short_name(function.name())
                ))
            }
            _ => Ok(format!(
                "<{} object at 0x{:x}>",
                type_id.name(),
                ptr as usize
            )),
        }
    }

    fn with_container<F>(
        &mut self,
        ptr: *const (),
        recursive_repr: &'static str,
        render: F,
    ) -> Result<String, BuiltinError>
    where
        F: FnOnce(&mut Self) -> Result<String, BuiltinError>,
    {
        let key = ptr as usize;
        if self.active.contains(&key) {
            return Ok(recursive_repr.to_string());
        }

        self.active.push(key);
        let result = render(self);
        self.active.pop();
        result
    }

    fn repr_list_like(&mut self, root_ptr: *const ()) -> Result<String, BuiltinError> {
        #[derive(Clone, Copy)]
        struct ListReprFrame {
            ptr: *const (),
            index: usize,
        }

        let root_key = root_ptr as usize;
        if self.active.contains(&root_key) {
            return Ok("[...]".to_string());
        }

        let mut out = String::from("[");
        let mut frames = SmallVec::<[ListReprFrame; 16]>::new();
        self.active.push(root_key);
        frames.push(ListReprFrame {
            ptr: root_ptr,
            index: 0,
        });

        while let Some(frame) = frames.last_mut() {
            let list = crate::ops::objects::list_storage_ref_from_ptr(frame.ptr)
                .ok_or_else(|| BuiltinError::TypeError("invalid list reference".to_string()))?;

            if frame.index >= list.len() {
                out.push(']');
                frames.pop();
                self.active.pop();
                continue;
            }

            if frame.index > 0 {
                out.push_str(", ");
            }

            let item = unsafe { list.get_unchecked(frame.index) };
            frame.index += 1;

            let Some(child_ptr) = item.as_object_ptr() else {
                out.push_str(&self.repr_value(item)?);
                continue;
            };

            if crate::ops::objects::list_storage_ref_from_ptr(child_ptr).is_none() {
                out.push_str(&self.repr_value(item)?);
                continue;
            }

            let child_key = child_ptr as usize;
            if self.active.contains(&child_key) {
                out.push_str("[...]");
                continue;
            }

            if frames.len() >= MAX_REPR_DEPTH {
                return Err(BuiltinError::Raised(RuntimeError::recursion_error(
                    frames.len(),
                )));
            }

            out.push('[');
            self.active.push(child_key);
            frames.push(ListReprFrame {
                ptr: child_ptr,
                index: 0,
            });
        }

        Ok(out)
    }

    fn repr_tuple(&mut self, tuple: &TupleObject) -> Result<String, BuiltinError> {
        if tuple.is_empty() {
            return Ok("()".to_string());
        }

        let mut out = String::from("(");
        for (index, item) in tuple.iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            out.push_str(&self.repr_value(*item)?);
        }
        if tuple.len() == 1 {
            out.push(',');
        }
        out.push(')');
        Ok(out)
    }

    fn repr_dict(&mut self, dict: &DictObject) -> Result<String, BuiltinError> {
        let mut out = String::from("{");
        let mut first = true;
        for (key, value) in dict.iter() {
            if !first {
                out.push_str(", ");
            }
            first = false;
            out.push_str(&self.repr_value(key)?);
            out.push_str(": ");
            out.push_str(&self.repr_value(value)?);
        }
        out.push('}');
        Ok(out)
    }

    fn repr_set(&mut self, set: &SetObject) -> Result<String, BuiltinError> {
        if set.is_empty() {
            return Ok("set()".to_string());
        }

        let mut out = String::from("{");
        let mut first = true;
        for value in set.iter() {
            if !first {
                out.push_str(", ");
            }
            first = false;
            out.push_str(&self.repr_value(value)?);
        }
        out.push('}');
        Ok(out)
    }
}

fn repr_value(value: Value) -> Result<String, BuiltinError> {
    ReprState::default().repr_value(value)
}

#[inline]
fn builtin_function_short_name(name: &str) -> &str {
    name.rsplit_once('.')
        .map(|(_, short_name)| short_name)
        .unwrap_or(name)
}

fn repr_value_vm(vm: &mut VirtualMachine, value: Value) -> Result<String, BuiltinError> {
    if should_use_python_repr_protocol(value) {
        match resolve_special_method(value, "__repr__") {
            Ok(target) => {
                let rendered = invoke_zero_arg_bound_method(vm, target)
                    .map_err(super::runtime_error_to_builtin_error)?;
                let Some(text) = value_as_string_ref(rendered) else {
                    return Err(BuiltinError::TypeError(format!(
                        "__repr__ returned non-string (type {})",
                        value_type_name(rendered)
                    )));
                };
                return Ok(text.as_str().to_string());
            }
            Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {}
            Err(err) => return Err(super::runtime_error_to_builtin_error(err)),
        }
    }

    repr_value(value)
}

#[inline]
fn should_use_python_repr_protocol(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    crate::ops::objects::extract_type_id(ptr).raw() >= TypeId::FIRST_USER_TYPE
}

fn quote_python_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    out.push('\'');
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() || is_surrogate_carrier(c as u32) => {
                out.push_str(&escape_char(c));
            }
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

fn quote_python_bytes(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len() + 3);
    out.push('b');
    out.push('\'');
    for &byte in input {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\'' => out.push_str("\\'"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(byte as char),
            _ => {
                out.push_str("\\x");
                out.push(nibble_to_hex(byte >> 4));
                out.push(nibble_to_hex(byte & 0x0f));
            }
        }
    }
    out.push('\'');
    out
}

#[inline]
fn nibble_to_hex(nibble: u8) -> char {
    debug_assert!(nibble <= 0x0f);
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'a' + (nibble - 10)) as char,
    }
}

#[inline]
fn escape_non_ascii(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            out.push_str(&escape_char(ch));
        }
    }
    out
}

#[inline]
fn escape_char(ch: char) -> String {
    python_char_escape(ch)
}
