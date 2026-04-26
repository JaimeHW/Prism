//! Bytes and bytearray builtin method resolution and implementations.

use super::*;

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

#[inline]
pub(super) fn bytes_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    decode_bytes_method(args, "bytes", expect_bytes_ref)
}

#[inline]
pub(super) fn bytearray_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    decode_bytes_method(args, "bytearray", expect_bytearray_ref)
}

#[inline]
pub(super) fn bytes_startswith(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_affix_match(
        args,
        "bytes",
        "startswith",
        expect_bytes_ref,
        |value, affix| value.starts_with(affix),
    )
}

#[inline]
pub(super) fn bytes_endswith(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_affix_match(
        args,
        "bytes",
        "endswith",
        expect_bytes_ref,
        |value, affix| value.ends_with(affix),
    )
}

#[inline]
pub(super) fn bytes_upper(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytes_lower(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytes_strip(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytes_lstrip(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytes_rstrip(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytes_translate(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_translate(args, "bytes", expect_bytes_ref, TypeId::BYTES)
}

#[inline]
pub(super) fn bytearray_startswith(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_affix_match(
        args,
        "bytearray",
        "startswith",
        expect_bytearray_ref,
        |value, affix| value.starts_with(affix),
    )
}

#[inline]
pub(super) fn bytearray_endswith(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_affix_match(
        args,
        "bytearray",
        "endswith",
        expect_bytearray_ref,
        |value, affix| value.ends_with(affix),
    )
}

#[inline]
pub(super) fn bytearray_upper(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytearray_lower(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytearray_strip(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytearray_lstrip(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytearray_rstrip(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytearray_translate(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_translate(args, "bytearray", expect_bytearray_ref, TypeId::BYTEARRAY)
}

#[inline]
pub(super) fn bytes_join_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    byte_sequence_join_with_vm(vm, args, "bytes", expect_bytes_ref, TypeId::BYTES)
}

#[inline]
pub(super) fn bytearray_join_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    byte_sequence_join_with_vm(
        vm,
        args,
        "bytearray",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
    )
}

#[inline]
pub(super) fn bytes_find_method(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytes_rfind_method(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytes_index_method(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytes_rindex_method(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytes_count_method(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_count(args, "bytes", expect_bytes_ref)
}

#[inline]
pub(super) fn bytearray_find(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytearray_rfind(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytearray_index(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytearray_rindex(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn bytearray_count(args: &[Value]) -> Result<Value, BuiltinError> {
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

#[inline]
pub(super) fn bytearray_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("bytearray", "copy", args, 0)?;
    let bytearray = expect_bytearray_ref(args[0], "copy")?;
    Ok(to_object_value(bytearray.clone()))
}

pub(super) fn bytearray_extend_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("bytearray", "extend", args, 1)?;
    let incoming = collect_bytearray_extend_data(vm, args[1])?;
    let bytearray = expect_bytearray_mut(args[0], "extend")?;
    bytearray.extend_from_slice(&incoming);
    Ok(Value::none())
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
