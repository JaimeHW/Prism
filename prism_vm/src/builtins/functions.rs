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
use prism_runtime::types::string::StringObject;
use prism_runtime::types::string::{object_ptr_as_string_ref, value_as_string_ref};
use prism_runtime::types::tuple::TupleObject;
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
    let len_i64 = i64::try_from(len)
        .map_err(|_| BuiltinError::OverflowError(format!("{} length overflow", type_name)))?;
    Value::int(len_i64)
        .ok_or_else(|| BuiltinError::OverflowError(format!("{} length overflow", type_name)))
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
    let mut signed = lower_48_signed(hash);
    if signed == -1 {
        signed = -2;
    }
    Value::int(signed).ok_or_else(|| BuiltinError::OverflowError("hash overflow".to_string()))
}

#[inline]
fn hash_value(value: Value) -> Result<u64, BuiltinError> {
    if value.is_none()
        || value.as_bool().is_some()
        || value.as_int().is_some()
        || value.as_float().is_some()
    {
        return Ok(hash_with_default_hasher(&value));
    }

    if let Some(string) = value_as_string_ref(value) {
        return Ok(hash_with_default_hasher(&string.as_str()));
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
        _ => Ok(hash_with_default_hasher(&(ptr as usize))),
    }
}

#[inline]
fn hash_tuple(tuple: &TupleObject) -> Result<u64, BuiltinError> {
    // CPython-inspired tuple hash combiner.
    let mut acc: u64 = 0x345678;
    let len = tuple.len();
    for (index, item) in tuple.iter().copied().enumerate() {
        let item_hash = hash_value(item)?;
        acc = (acc ^ item_hash).wrapping_mul(1_000_003);
        acc = acc.wrapping_add((index as u64).wrapping_mul(2).wrapping_add(82_520));
    }
    acc ^= len as u64;
    if acc == u64::MAX {
        acc = u64::MAX - 1;
    }
    Ok(acc)
}

#[inline]
fn hash_with_default_hasher<T: Hash>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[inline]
fn lower_48_signed(value: u64) -> i64 {
    let masked = value & ((1u64 << 48) - 1);
    if (masked & (1u64 << 47)) != 0 {
        masked as i64 - (1i64 << 48)
    } else {
        masked as i64
    }
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

    let repr = repr_value(args[0], 0)?;
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

    let repr = repr_value_vm(vm, args[0], 0)?;
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

    let repr = repr_value(args[0], 0)?;
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

    let repr = repr_value_vm(vm, args[0], 0)?;
    let ascii = escape_non_ascii(&repr);
    Ok(Value::string(intern(&ascii)))
}

const MAX_REPR_DEPTH: usize = 12;

fn repr_list_like(list: &ListObject, depth: usize) -> Result<String, BuiltinError> {
    let mut out = String::from("[");
    let mut first = true;
    for item in list.iter() {
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push_str(&repr_value(*item, depth + 1)?);
    }
    out.push(']');
    Ok(out)
}

fn repr_value(value: Value, depth: usize) -> Result<String, BuiltinError> {
    if depth >= MAX_REPR_DEPTH {
        return Ok("...".to_string());
    }

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
    if let Some(list) = crate::ops::objects::list_storage_ref_from_ptr(ptr) {
        return repr_list_like(list, depth);
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
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            if tuple.is_empty() {
                return Ok("()".to_string());
            }
            let mut out = String::from("(");
            for (index, item) in tuple.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&repr_value(*item, depth + 1)?);
            }
            if tuple.len() == 1 {
                out.push(',');
            }
            out.push(')');
            Ok(out)
        }
        TypeId::DICT => {
            let dict = unsafe { &*(ptr as *const DictObject) };
            let mut out = String::from("{");
            let mut first = true;
            for (key, value) in dict.iter() {
                if !first {
                    out.push_str(", ");
                }
                first = false;
                out.push_str(&repr_value(key, depth + 1)?);
                out.push_str(": ");
                out.push_str(&repr_value(value, depth + 1)?);
            }
            out.push('}');
            Ok(out)
        }
        TypeId::SET => {
            let set = unsafe { &*(ptr as *const SetObject) };
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
                out.push_str(&repr_value(value, depth + 1)?);
            }
            out.push('}');
            Ok(out)
        }
        TypeId::RANGE => {
            let range = unsafe { &*(ptr as *const RangeObject) };
            Ok(range.to_string())
        }
        TypeId::COMPLEX => {
            let complex = unsafe { &*(ptr as *const ComplexObject) };
            Ok(complex.to_string())
        }
        TypeId::STATICMETHOD => {
            let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            Ok(format!(
                "<staticmethod({})>",
                repr_value(descriptor.function(), depth + 1)?
            ))
        }
        TypeId::CLASSMETHOD => {
            let descriptor = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            Ok(format!(
                "<classmethod({})>",
                repr_value(descriptor.function(), depth + 1)?
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

#[inline]
fn builtin_function_short_name(name: &str) -> &str {
    name.rsplit_once('.')
        .map(|(_, short_name)| short_name)
        .unwrap_or(name)
}

fn repr_value_vm(
    vm: &mut VirtualMachine,
    value: Value,
    depth: usize,
) -> Result<String, BuiltinError> {
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

    repr_value(value, depth)
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::BuiltinFunctionObject;
    use crate::builtins::itertools::{builtin_iter, builtin_next};
    use crate::import::ModuleObject;
    use prism_core::intern::intern;
    use prism_core::python_unicode::encode_python_code_point;
    use prism_core::value::SMALL_INT_MAX;
    use prism_runtime::object::ObjectHeader;
    use prism_runtime::object::class::PyClassObject;
    use prism_runtime::object::descriptor::{ClassMethodDescriptor, StaticMethodDescriptor};
    use prism_runtime::object::shape::Shape;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::types::function::FunctionObject;
    use std::sync::Arc;

    fn boxed_value<T>(obj: T) -> (Value, *mut T) {
        let ptr = Box::into_raw(Box::new(obj));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        drop(unsafe { Box::from_raw(ptr) });
    }

    fn tagged_string_value_to_rust_string(value: Value) -> String {
        assert!(value.is_string(), "expected tagged string, got {value:?}");
        let ptr = value
            .as_string_object_ptr()
            .expect("tagged string pointer missing") as *const u8;
        prism_core::intern::interned_by_ptr(ptr)
            .expect("tagged string pointer not interned")
            .as_str()
            .to_string()
    }

    #[test]
    fn test_len_tagged_string() {
        let value = Value::string(intern("hello"));
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(5));
    }

    #[test]
    fn test_len_string_counts_unicode_scalar_values() {
        let tagged = Value::string(intern("tmpæ"));
        assert_eq!(builtin_len(&[tagged]).unwrap().as_int(), Some(4));

        let (heap, ptr) = boxed_value(StringObject::new("hé 🦀"));
        assert_eq!(builtin_len(&[heap]).unwrap().as_int(), Some(4));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_list_object() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(3));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_tuple_object() {
        let tuple = TupleObject::from_slice(&[
            Value::int(10).unwrap(),
            Value::int(20).unwrap(),
            Value::int(30).unwrap(),
            Value::int(40).unwrap(),
        ]);
        let (value, ptr) = boxed_value(tuple);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(4));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_tuple_backed_object() {
        let object = ShapedObject::new_tuple_backed(
            TypeId::OBJECT,
            Shape::empty(),
            TupleObject::from_slice(&[
                Value::int(1).unwrap(),
                Value::int(2).unwrap(),
                Value::int(3).unwrap(),
            ]),
        );
        let (value, ptr) = boxed_value(object);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(3));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_heap_list_subclass_uses_native_backing() {
        let mut object = ShapedObject::new_list_backed(TypeId::from_raw(512), Shape::empty());
        object
            .list_backing_mut()
            .expect("list backing should exist")
            .extend([Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (value, ptr) = boxed_value(object);

        let result = builtin_len(&[value]).unwrap();

        assert_eq!(result.as_int(), Some(2));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_dict_object() {
        let mut dict = DictObject::new();
        dict.set(Value::int(1).unwrap(), Value::int(11).unwrap());
        dict.set(Value::int(2).unwrap(), Value::int(22).unwrap());
        let (value, ptr) = boxed_value(dict);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(2));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_mappingproxy_object() {
        let class = Arc::new(PyClassObject::new_simple(intern("SizedProxy")));
        class.set_attr(intern("token"), Value::int(7).unwrap());
        class.set_attr(intern("label"), Value::string(intern("ready")));

        let proxy = MappingProxyObject::for_user_class(Arc::as_ptr(&class));
        let (value, ptr) = boxed_value(proxy);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(2));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_set_object() {
        let set = SetObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (value, ptr) = boxed_value(set);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(3));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_frozenset_object() {
        let mut set = SetObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]);
        set.header.type_id = TypeId::FROZENSET;
        let (value, ptr) = boxed_value(set);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(2));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_bytes_object() {
        let bytes = BytesObject::from_slice(b"hello");
        let (value, ptr) = boxed_value(bytes);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(5));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_bytearray_object() {
        let bytearray = BytesObject::bytearray_from_slice(&[1, 2, 3, 4]);
        let (value, ptr) = boxed_value(bytearray);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(4));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_range_object() {
        let range = RangeObject::new(0, 10, 2);
        let (value, ptr) = boxed_value(range);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(5));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_string_object() {
        let string = StringObject::new("runtime");
        let (value, ptr) = boxed_value(string);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(7));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_arity_error() {
        let result = builtin_len(&[]);
        assert!(matches!(result, Err(BuiltinError::TypeError(_))));
    }

    #[test]
    fn test_len_non_sized_type_error() {
        let result = builtin_len(&[Value::int(42).unwrap()]);
        assert!(matches!(result, Err(BuiltinError::TypeError(_))));
    }

    #[test]
    fn test_abs_int() {
        let result = builtin_abs(&[Value::int(-42).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(42));

        let result = builtin_abs(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_abs_bool_returns_int_result() {
        let result = builtin_abs(&[Value::bool(true)]).unwrap();
        assert_eq!(result.as_int(), Some(1));
        assert!(!result.is_bool());
    }

    #[test]
    fn test_abs_float() {
        let result = builtin_abs(&[Value::float(-3.14)]).unwrap();
        assert_eq!(result.as_float(), Some(3.14));
    }

    #[test]
    fn test_abs_error() {
        let result = builtin_abs(&[Value::none()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_min() {
        let result = builtin_min(&[
            Value::int(5).unwrap(),
            Value::int(3).unwrap(),
            Value::int(8).unwrap(),
        ])
        .unwrap();
        assert_eq!(result.as_int(), Some(3));
    }

    #[test]
    fn test_max() {
        let result = builtin_max(&[
            Value::int(5).unwrap(),
            Value::int(3).unwrap(),
            Value::int(8).unwrap(),
        ])
        .unwrap();
        assert_eq!(result.as_int(), Some(8));
    }

    #[test]
    fn test_max_iterable_uses_elements_instead_of_returning_the_iterable() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_max(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(3));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_max_iterable_supports_string_ordering() {
        let values = SetObject::from_slice(&[
            Value::string(intern("a")),
            Value::string(intern("c")),
            Value::string(intern("b")),
        ]);
        let (value, ptr) = boxed_value(values);
        let result = builtin_max(&[value]).unwrap();
        assert_eq!(value_as_string_ref(result).unwrap().as_str(), "c");
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_max_vm_supports_default_for_empty_iterable() {
        let mut vm = VirtualMachine::new();
        let list = ListObject::from_slice(&[]);
        let (value, ptr) = boxed_value(list);
        let fallback = Value::int(99).unwrap();
        let result = builtin_max_vm_kw(&mut vm, &[value], &[("default", fallback)]).unwrap();
        assert_eq!(result.as_int(), Some(99));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_max_vm_rejects_default_with_multiple_positional_arguments() {
        let mut vm = VirtualMachine::new();
        let error = builtin_max_vm_kw(
            &mut vm,
            &[Value::int(1).unwrap(), Value::int(2).unwrap()],
            &[("default", Value::int(3).unwrap())],
        )
        .unwrap_err();
        assert!(matches!(error, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_max_vm_honors_key_keyword() {
        let mut vm = VirtualMachine::new();
        let result = builtin_max_vm_kw(
            &mut vm,
            &[Value::int(0).unwrap(), Value::int(7).unwrap()],
            &[(
                "key",
                crate::builtins::builtin_type_object_for_type_id(TypeId::BOOL),
            )],
        )
        .unwrap();
        assert_eq!(result.as_int(), Some(7));
    }

    #[test]
    fn test_sum_int_list() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(10));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_float_list() {
        let list =
            ListObject::from_slice(&[Value::float(1.5), Value::float(2.0), Value::float(3.5)]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_float(), Some(7.0));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_mixed_numeric_promotes_to_float() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::float(2.5)]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_float(), Some(3.5));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_bool_items() {
        let list =
            ListObject::from_slice(&[Value::bool(true), Value::bool(false), Value::bool(true)]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(2));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_with_int_start() {
        let list = ListObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value, Value::int(10).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(15));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_with_float_start() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value, Value::float(0.5)]).unwrap();
        assert_eq!(result.as_float(), Some(3.5));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_range() {
        let range = RangeObject::new(1, 5, 1);
        let (value, ptr) = boxed_value(range);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(10));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_iterator_consumes_iterator_state() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (list_value, list_ptr) = boxed_value(list);
        let iter = builtin_iter(&[list_value]).unwrap();

        let result = builtin_sum(&[iter]).unwrap();
        assert_eq!(result.as_int(), Some(6));

        // Iterator should be exhausted after sum consumes it.
        let next_result = builtin_next(&[iter]);
        assert!(next_result.is_err());
        unsafe { drop_boxed(list_ptr) };
    }

    #[test]
    fn test_sum_non_iterable_error() {
        let err = builtin_sum(&[Value::int(42).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_sum_non_numeric_start_error() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let err = builtin_sum(&[value, Value::none()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_string_start_error() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let err = builtin_sum(&[value, Value::string(intern("x"))]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        let msg = err.to_string();
        assert!(msg.contains("can't sum strings"));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_string_item_error() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::string(intern("x"))]);
        let (value, ptr) = boxed_value(list);
        let err = builtin_sum(&[value]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        let msg = err.to_string();
        assert!(msg.contains("can't sum strings"));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_overflow_small_int_domain() {
        let list =
            ListObject::from_slice(&[Value::int(SMALL_INT_MAX).unwrap(), Value::int(1).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(
            value_to_bigint(result),
            Some(BigInt::from(SMALL_INT_MAX) + BigInt::from(1_i64))
        );
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_bigint_and_float_reports_exact_overflow_boundary() {
        let huge = bigint_to_value(BigInt::from(1_u8) << 10000_u32);
        let list = ListObject::from_slice(&[huge, Value::float(1.0)]);
        let (value, ptr) = boxed_value(list);
        let err = builtin_sum(&[value]).unwrap_err();
        assert!(matches!(err, BuiltinError::OverflowError(_)));
        assert!(
            err.to_string()
                .contains("int too large to convert to float"),
            "unexpected error: {err}"
        );
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_pow_int() {
        let result = builtin_pow(&[Value::int(2).unwrap(), Value::int(10).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(1024));
    }

    #[test]
    fn test_pow_mod() {
        let result = builtin_pow(&[
            Value::int(2).unwrap(),
            Value::int(10).unwrap(),
            Value::int(100).unwrap(),
        ])
        .unwrap();
        assert_eq!(result.as_int(), Some(24)); // 1024 % 100 = 24
    }

    #[test]
    fn test_min_prefers_exact_bigint_ordering_over_float_rounding() {
        let huge = bigint_to_value(BigInt::from(1_u8) << 80_u32);
        let result = builtin_min(&[huge, Value::float(2f64.powi(80))]).unwrap();
        assert_eq!(value_to_bigint(result), Some(BigInt::from(1_u8) << 80_u32));
    }

    #[test]
    fn test_max_prefers_exact_bigint_ordering_over_float_rounding() {
        let huge_plus_one = bigint_to_value((BigInt::from(1_u8) << 80_u32) + BigInt::from(1_u8));
        let result = builtin_max(&[huge_plus_one, Value::float(2f64.powi(80))]).unwrap();
        assert_eq!(
            value_to_bigint(result),
            Some((BigInt::from(1_u8) << 80_u32) + BigInt::from(1_u8))
        );
    }

    #[test]
    fn test_min_handles_negative_bigint_float_boundary_exactly() {
        let huge_negative = bigint_to_value(-((BigInt::from(1_u8) << 80_u32) + BigInt::from(1_u8)));
        let result = builtin_min(&[huge_negative, Value::float(-2f64.powi(80))]).unwrap();
        assert_eq!(
            value_to_bigint(result),
            Some(-((BigInt::from(1_u8) << 80_u32) + BigInt::from(1_u8)))
        );
    }

    #[test]
    fn test_pow_promotes_large_integer_results_to_bigint() {
        let result = builtin_pow(&[Value::int(2).unwrap(), Value::int(100).unwrap()]).unwrap();
        assert_eq!(value_to_bigint(result), Some(BigInt::from(1_u8) << 100_u32));
    }

    #[test]
    fn test_pow_negative_modular_exponent_uses_inverse() {
        let result = builtin_pow(&[
            Value::int(2).unwrap(),
            Value::int(-1).unwrap(),
            Value::int(5).unwrap(),
        ])
        .unwrap();
        assert_eq!(result.as_int(), Some(3));
    }

    #[test]
    fn test_pow_negative_modulus_matches_python_sign_rules() {
        let result = builtin_pow(&[
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(-5).unwrap(),
        ])
        .unwrap();
        assert_eq!(result.as_int(), Some(-2));

        let inverse = builtin_pow(&[
            Value::int(2).unwrap(),
            Value::int(-1).unwrap(),
            Value::int(-5).unwrap(),
        ])
        .unwrap();
        assert_eq!(inverse.as_int(), Some(-2));
    }

    #[test]
    fn test_pow_non_invertible_negative_modular_exponent_errors() {
        let err = builtin_pow(&[
            Value::int(2).unwrap(),
            Value::int(-1).unwrap(),
            Value::int(4).unwrap(),
        ])
        .unwrap_err();
        assert!(matches!(err, BuiltinError::ValueError(_)));
        assert!(
            err.to_string()
                .contains("base is not invertible for the given modulus"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_pow_zero_to_negative_power_raises_zero_division() {
        let err = builtin_pow(&[Value::int(0).unwrap(), Value::int(-1).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::Raised(_)));
        assert!(
            err.to_string()
                .contains("0.0 cannot be raised to a negative power"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_round_int() {
        let result = builtin_round(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_round_float() {
        let result = builtin_round(&[Value::float(3.7)]).unwrap();
        assert_eq!(result.as_int(), Some(4));
    }

    #[test]
    fn test_divmod_int_returns_tuple() {
        let result = builtin_divmod(&[Value::int(17).unwrap(), Value::int(5).unwrap()]).unwrap();
        let ptr = result.as_object_ptr().expect("divmod should return tuple");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.get(0).unwrap().as_int(), Some(3));
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(2));
    }

    #[test]
    fn test_divmod_int_negative_divisor() {
        let result = builtin_divmod(&[Value::int(7).unwrap(), Value::int(-3).unwrap()]).unwrap();
        let ptr = result.as_object_ptr().expect("divmod should return tuple");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.get(0).unwrap().as_int(), Some(-3));
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(-2));
    }

    #[test]
    fn test_divmod_float_returns_tuple() {
        let result = builtin_divmod(&[Value::float(7.5), Value::float(2.0)]).unwrap();
        let ptr = result.as_object_ptr().expect("divmod should return tuple");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.get(0).unwrap().as_float(), Some(3.0));
        assert_eq!(tuple.get(1).unwrap().as_float(), Some(1.5));
    }

    #[test]
    fn test_divmod_mixed_numeric() {
        let result = builtin_divmod(&[Value::int(7).unwrap(), Value::float(2.0)]).unwrap();
        let ptr = result.as_object_ptr().expect("divmod should return tuple");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.get(0).unwrap().as_float(), Some(3.0));
        assert_eq!(tuple.get(1).unwrap().as_float(), Some(1.0));
    }

    #[test]
    fn test_divmod_zero_errors() {
        let int_err = builtin_divmod(&[Value::int(1).unwrap(), Value::int(0).unwrap()]);
        assert!(matches!(int_err, Err(BuiltinError::ValueError(_))));

        let float_err = builtin_divmod(&[Value::float(1.0), Value::float(0.0)]);
        assert!(matches!(float_err, Err(BuiltinError::ValueError(_))));
    }

    #[test]
    fn test_repr_primitives() {
        assert_eq!(
            tagged_string_value_to_rust_string(builtin_repr(&[Value::none()]).unwrap()),
            "None"
        );
        assert_eq!(
            tagged_string_value_to_rust_string(builtin_repr(&[Value::bool(true)]).unwrap()),
            "True"
        );
        assert_eq!(
            tagged_string_value_to_rust_string(builtin_repr(&[Value::int(42).unwrap()]).unwrap()),
            "42"
        );
        assert_eq!(
            tagged_string_value_to_rust_string(builtin_repr(&[Value::float(1.5)]).unwrap()),
            "1.5"
        );
    }

    #[test]
    fn test_repr_tagged_string_escaping() {
        let value = Value::string(intern("a'b\n"));
        let repr = tagged_string_value_to_rust_string(builtin_repr(&[value]).unwrap());
        assert_eq!(repr, "'a\\'b\\n'");
    }

    #[test]
    fn test_repr_runtime_string() {
        let (value, ptr) = boxed_value(StringObject::new("runtime"));
        let repr = tagged_string_value_to_rust_string(builtin_repr(&[value]).unwrap());
        assert_eq!(repr, "'runtime'");
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_repr_and_ascii_escape_internal_surrogate_carriers_as_python_surrogates() {
        let surrogate =
            encode_python_code_point(0xDC80).expect("surrogate should map into carrier range");
        let text = format!("A{surrogate}");
        let (value, ptr) = boxed_value(StringObject::from_string(text));

        let repr = tagged_string_value_to_rust_string(builtin_repr(&[value]).unwrap());
        assert_eq!(repr, "'A\\udc80'");

        let ascii = tagged_string_value_to_rust_string(builtin_ascii(&[value]).unwrap());
        assert_eq!(ascii, "'A\\udc80'");

        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_repr_containers() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (list_value, list_ptr) = boxed_value(list);
        let list_repr = tagged_string_value_to_rust_string(builtin_repr(&[list_value]).unwrap());
        assert_eq!(list_repr, "[1, 2]");
        unsafe { drop_boxed(list_ptr) };

        let tuple = TupleObject::from_slice(&[Value::int(1).unwrap()]);
        let (tuple_value, tuple_ptr) = boxed_value(tuple);
        let tuple_repr = tagged_string_value_to_rust_string(builtin_repr(&[tuple_value]).unwrap());
        assert_eq!(tuple_repr, "(1,)");
        unsafe { drop_boxed(tuple_ptr) };

        let mut dict = DictObject::new();
        dict.set(Value::int(1).unwrap(), Value::int(2).unwrap());
        let (dict_value, dict_ptr) = boxed_value(dict);
        let dict_repr = tagged_string_value_to_rust_string(builtin_repr(&[dict_value]).unwrap());
        assert_eq!(dict_repr, "{1: 2}");
        unsafe { drop_boxed(dict_ptr) };

        let mut set = SetObject::new();
        set.add(Value::int(3).unwrap());
        let (set_value, set_ptr) = boxed_value(set);
        let set_repr = tagged_string_value_to_rust_string(builtin_repr(&[set_value]).unwrap());
        assert_eq!(set_repr, "{3}");
        unsafe { drop_boxed(set_ptr) };
    }

    #[test]
    fn test_repr_range_object() {
        let (range_value, range_ptr) = boxed_value(RangeObject::new(1, 6, 2));
        let repr = tagged_string_value_to_rust_string(builtin_repr(&[range_value]).unwrap());
        assert_eq!(repr, "range(1, 6, 2)");
        unsafe { drop_boxed(range_ptr) };
    }

    #[test]
    fn test_repr_complex_object() {
        let (complex_value, complex_ptr) = boxed_value(ComplexObject::new(1.0, 0.0));
        let repr = tagged_string_value_to_rust_string(builtin_repr(&[complex_value]).unwrap());
        assert_eq!(repr, "(1+0j)");
        unsafe { drop_boxed(complex_ptr) };
    }

    #[test]
    fn test_repr_bytes_and_bytearray_objects() {
        let (bytes_value, bytes_ptr) = boxed_value(BytesObject::from_slice(b"a'\n\\"));
        let bytes_repr = tagged_string_value_to_rust_string(builtin_repr(&[bytes_value]).unwrap());
        assert_eq!(bytes_repr, "b'a\\'\\n\\\\'");
        unsafe { drop_boxed(bytes_ptr) };

        let (bytearray_value, bytearray_ptr) =
            boxed_value(BytesObject::bytearray_from_slice(&[0, 65, 255]));
        let bytearray_repr =
            tagged_string_value_to_rust_string(builtin_repr(&[bytearray_value]).unwrap());
        assert_eq!(bytearray_repr, "bytearray(b'\\x00A\\xff')");
        unsafe { drop_boxed(bytearray_ptr) };
    }

    #[test]
    fn test_repr_exception_uses_exception_type_and_args() {
        let exc = crate::builtins::get_exception_type("ValueError")
            .expect("ValueError should exist")
            .construct(&[Value::string(intern("boom"))]);
        let repr = tagged_string_value_to_rust_string(builtin_repr(&[exc]).unwrap());

        assert_eq!(repr, "ValueError('boom')");
    }

    #[test]
    fn test_repr_staticmethod_and_classmethod_wrappers() {
        let function = FunctionObject::new(
            Arc::new(prism_code::CodeObject::new("demo", "<test>")),
            Arc::from("demo"),
            None,
            None,
        );
        let (function_value, function_ptr) = boxed_value(function);
        let inner_repr = repr_value(function_value, 0).expect("function repr should succeed");

        let (staticmethod_value, staticmethod_ptr) =
            boxed_value(StaticMethodDescriptor::new(function_value));
        let staticmethod_repr =
            tagged_string_value_to_rust_string(builtin_repr(&[staticmethod_value]).unwrap());
        assert_eq!(staticmethod_repr, format!("<staticmethod({inner_repr})>"));

        let (classmethod_value, classmethod_ptr) =
            boxed_value(ClassMethodDescriptor::new(function_value));
        let classmethod_repr =
            tagged_string_value_to_rust_string(builtin_repr(&[classmethod_value]).unwrap());
        assert_eq!(classmethod_repr, format!("<classmethod({inner_repr})>"));

        unsafe {
            drop_boxed(classmethod_ptr);
            drop_boxed(staticmethod_ptr);
            drop_boxed(function_ptr);
        }
    }

    #[test]
    fn test_repr_builtin_function_uses_cpython_display_name() {
        fn sample_builtin(_args: &[Value]) -> Result<Value, BuiltinError> {
            Ok(Value::none())
        }

        let (function_value, function_ptr) = boxed_value(BuiltinFunctionObject::new(
            Arc::from("time.sleep"),
            sample_builtin,
        ));
        let repr = tagged_string_value_to_rust_string(builtin_repr(&[function_value]).unwrap());

        assert_eq!(repr, "<built-in function sleep>");
        unsafe { drop_boxed(function_ptr) };
    }

    #[test]
    fn test_ascii_non_ascii_escaping() {
        let tagged = Value::string(intern("hé"));
        let tagged_ascii = tagged_string_value_to_rust_string(builtin_ascii(&[tagged]).unwrap());
        assert_eq!(tagged_ascii, "'h\\xe9'");

        let (runtime, ptr) = boxed_value(StringObject::new("漢"));
        let runtime_ascii = tagged_string_value_to_rust_string(builtin_ascii(&[runtime]).unwrap());
        assert_eq!(runtime_ascii, "'\\u6f22'");
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_repr_ascii_arity_errors() {
        let repr_err = builtin_repr(&[]);
        assert!(matches!(repr_err, Err(BuiltinError::TypeError(_))));

        let ascii_err = builtin_ascii(&[]);
        assert!(matches!(ascii_err, Err(BuiltinError::TypeError(_))));
    }

    #[test]
    fn test_id_distinguishes_interned_string_values() {
        let platform = Value::string(intern("platform"));
        let uname_result = Value::string(intern("uname_result"));

        let platform_id = builtin_id(&[platform]).unwrap().as_int().unwrap();
        let same_platform_id = builtin_id(&[Value::string(intern("platform"))])
            .unwrap()
            .as_int()
            .unwrap();
        let uname_result_id = builtin_id(&[uname_result]).unwrap().as_int().unwrap();

        assert_ne!(platform_id, 0);
        assert_eq!(platform_id, same_platform_id);
        assert_ne!(platform_id, uname_result_id);
    }

    #[test]
    fn test_hash_int() {
        let result = builtin_hash(&[Value::int(42).unwrap()]).unwrap();
        assert!(result.as_int().is_some());
    }

    #[test]
    fn test_hash_int_float_equivalence() {
        let int_hash = builtin_hash(&[Value::int(42).unwrap()]).unwrap();
        let float_hash = builtin_hash(&[Value::float(42.0)]).unwrap();
        assert_eq!(int_hash.as_int(), float_hash.as_int());
    }

    #[test]
    fn test_hash_tagged_string_by_content() {
        let a = builtin_hash(&[Value::string(intern("alpha"))]).unwrap();
        let b = builtin_hash(&[Value::string(intern("alpha"))]).unwrap();
        assert_eq!(a.as_int(), b.as_int());
    }

    #[test]
    fn test_hash_runtime_string_matches_tagged_string() {
        let tagged = builtin_hash(&[Value::string(intern("runtime"))]).unwrap();
        let (runtime_value, ptr) = boxed_value(StringObject::new("runtime"));
        let runtime = builtin_hash(&[runtime_value]).unwrap();
        assert_eq!(tagged.as_int(), runtime.as_int());
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_hash_module_object_uses_identity_hash() {
        let module = Arc::new(ModuleObject::new("abc"));
        let value = Value::object_ptr(Arc::as_ptr(&module) as *const ());
        let hash = builtin_hash(&[value]).expect("module objects should be hashable");
        assert!(hash.as_int().is_some());
    }

    #[test]
    fn test_hash_tuple_by_contents() {
        let tuple1 = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (tuple1_value, tuple1_ptr) = boxed_value(tuple1);
        let hash1 = builtin_hash(&[tuple1_value]).unwrap();

        let tuple2 = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (tuple2_value, tuple2_ptr) = boxed_value(tuple2);
        let hash2 = builtin_hash(&[tuple2_value]).unwrap();

        assert_eq!(hash1.as_int(), hash2.as_int());
        unsafe { drop_boxed(tuple1_ptr) };
        unsafe { drop_boxed(tuple2_ptr) };
    }

    #[test]
    fn test_hash_tuple_unhashable_member_error() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let (list_value, list_ptr) = boxed_value(list);
        let tuple = TupleObject::from_slice(&[list_value]);
        let (tuple_value, tuple_ptr) = boxed_value(tuple);

        let err = builtin_hash(&[tuple_value]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("unhashable type"));

        unsafe { drop_boxed(tuple_ptr) };
        unsafe { drop_boxed(list_ptr) };
    }

    #[test]
    fn test_hash_unhashable_containers_error() {
        let (list_value, list_ptr) = boxed_value(ListObject::new());
        let list_err = builtin_hash(&[list_value]).unwrap_err();
        assert!(list_err.to_string().contains("unhashable type: 'list'"));
        unsafe { drop_boxed(list_ptr) };

        let (dict_value, dict_ptr) = boxed_value(DictObject::new());
        let dict_err = builtin_hash(&[dict_value]).unwrap_err();
        assert!(dict_err.to_string().contains("unhashable type: 'dict'"));
        unsafe { drop_boxed(dict_ptr) };

        let (set_value, set_ptr) = boxed_value(SetObject::new());
        let set_err = builtin_hash(&[set_value]).unwrap_err();
        assert!(set_err.to_string().contains("unhashable type: 'set'"));
        unsafe { drop_boxed(set_ptr) };
    }

    #[test]
    fn test_callable() {
        let result = builtin_callable(&[Value::int(42).unwrap()]).unwrap();
        assert!(!result.is_truthy());

        let result = builtin_callable(&[Value::none()]).unwrap();
        assert!(!result.is_truthy());
    }

    fn dummy_builtin(_args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(Value::none())
    }

    #[test]
    fn test_callable_builtin_function_true() {
        let builtin = BuiltinFunctionObject::new("dummy".into(), dummy_builtin);
        let (value, ptr) = boxed_value(builtin);
        let result = builtin_callable(&[value]).unwrap();
        assert!(result.is_truthy());
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_callable_type_object_true() {
        #[repr(C)]
        struct DummyTypeObject {
            header: ObjectHeader,
        }

        let dummy = DummyTypeObject {
            header: ObjectHeader::new(TypeId::TYPE),
        };
        let (value, ptr) = boxed_value(dummy);
        let result = builtin_callable(&[value]).unwrap();
        assert!(result.is_truthy());
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_callable_non_callable_object_false() {
        let (value, ptr) = boxed_value(ListObject::new());
        let result = builtin_callable(&[value]).unwrap();
        assert!(!result.is_truthy());
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_callable_string_false() {
        let result = builtin_callable(&[Value::string(intern("name"))]).unwrap();
        assert!(!result.is_truthy());
    }
}
