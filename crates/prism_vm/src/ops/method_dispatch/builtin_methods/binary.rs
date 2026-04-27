//! Bytes and bytearray builtin method resolution and implementations.

use super::*;

#[derive(Clone, Copy)]
enum SplitDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Copy)]
enum ReturnIdentity {
    WhenUnchanged,
    Never,
}

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
static BYTES_REPLACE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("bytes.replace"), bytes_replace_kw));
static BYTES_SPLIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("bytes.split"), bytes_split_kw));
static BYTES_RSPLIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("bytes.rsplit"), bytes_rsplit_kw));
static BYTES_PARTITION_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.partition"), bytes_partition));
static BYTES_RPARTITION_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("bytes.rpartition"), bytes_rpartition));

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
static BYTEARRAY_REPLACE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("bytearray.replace"), bytearray_replace_kw)
});
static BYTEARRAY_SPLIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("bytearray.split"), bytearray_split_kw)
});
static BYTEARRAY_RSPLIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("bytearray.rsplit"), bytearray_rsplit_kw)
});
static BYTEARRAY_PARTITION_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("bytearray.partition"), bytearray_partition)
});
static BYTEARRAY_RPARTITION_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("bytearray.rpartition"), bytearray_rpartition)
});

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
        "replace" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_REPLACE_METHOD,
        ))),
        "split" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_SPLIT_METHOD,
        ))),
        "rsplit" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_RSPLIT_METHOD,
        ))),
        "partition" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_PARTITION_METHOD,
        ))),
        "rpartition" => Some(CachedMethod::simple(builtin_method_value(
            &BYTES_RPARTITION_METHOD,
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
        "replace" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_REPLACE_METHOD,
        ))),
        "split" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_SPLIT_METHOD,
        ))),
        "rsplit" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_RSPLIT_METHOD,
        ))),
        "partition" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_PARTITION_METHOD,
        ))),
        "rpartition" => Some(CachedMethod::simple(builtin_method_value(
            &BYTEARRAY_RPARTITION_METHOD,
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
pub(super) fn bytes_replace(args: &[Value]) -> Result<Value, BuiltinError> {
    bytes_replace_kw(args, &[])
}

#[inline]
pub(super) fn bytes_replace_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    byte_sequence_replace_kw(
        args,
        keywords,
        "bytes",
        expect_bytes_ref,
        TypeId::BYTES,
        ReturnIdentity::WhenUnchanged,
    )
}

#[inline]
pub(super) fn bytes_split(args: &[Value]) -> Result<Value, BuiltinError> {
    bytes_split_kw(args, &[])
}

#[inline]
pub(super) fn bytes_split_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    byte_sequence_split_kw(
        args,
        keywords,
        "bytes",
        "split",
        expect_bytes_ref,
        TypeId::BYTES,
        SplitDirection::Forward,
    )
}

#[inline]
pub(super) fn bytes_rsplit(args: &[Value]) -> Result<Value, BuiltinError> {
    bytes_rsplit_kw(args, &[])
}

#[inline]
pub(super) fn bytes_rsplit_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    byte_sequence_split_kw(
        args,
        keywords,
        "bytes",
        "rsplit",
        expect_bytes_ref,
        TypeId::BYTES,
        SplitDirection::Reverse,
    )
}

#[inline]
pub(super) fn bytes_partition(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_partition(
        args,
        "bytes",
        "partition",
        expect_bytes_ref,
        TypeId::BYTES,
        SearchDirection::Forward,
    )
}

#[inline]
pub(super) fn bytes_rpartition(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_partition(
        args,
        "bytes",
        "rpartition",
        expect_bytes_ref,
        TypeId::BYTES,
        SearchDirection::Reverse,
    )
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
pub(super) fn bytearray_replace(args: &[Value]) -> Result<Value, BuiltinError> {
    bytearray_replace_kw(args, &[])
}

#[inline]
pub(super) fn bytearray_replace_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    byte_sequence_replace_kw(
        args,
        keywords,
        "bytearray",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        ReturnIdentity::Never,
    )
}

#[inline]
pub(super) fn bytearray_split(args: &[Value]) -> Result<Value, BuiltinError> {
    bytearray_split_kw(args, &[])
}

#[inline]
pub(super) fn bytearray_split_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    byte_sequence_split_kw(
        args,
        keywords,
        "bytearray",
        "split",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        SplitDirection::Forward,
    )
}

#[inline]
pub(super) fn bytearray_rsplit(args: &[Value]) -> Result<Value, BuiltinError> {
    bytearray_rsplit_kw(args, &[])
}

#[inline]
pub(super) fn bytearray_rsplit_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    byte_sequence_split_kw(
        args,
        keywords,
        "bytearray",
        "rsplit",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        SplitDirection::Reverse,
    )
}

#[inline]
pub(super) fn bytearray_partition(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_partition(
        args,
        "bytearray",
        "partition",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        SearchDirection::Forward,
    )
}

#[inline]
pub(super) fn bytearray_rpartition(args: &[Value]) -> Result<Value, BuiltinError> {
    byte_sequence_partition(
        args,
        "bytearray",
        "rpartition",
        expect_bytearray_ref,
        TypeId::BYTEARRAY,
        SearchDirection::Reverse,
    )
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
        SearchDirection::Reverse => simd_bytes_rfind(haystack, &needle),
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

fn byte_sequence_replace_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
    receiver_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
    result_type: TypeId,
    identity: ReturnIdentity,
) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 3 {
        return Err(BuiltinError::TypeError(format!(
            "{receiver_name}.replace() takes from 2 to 3 arguments ({given} given)"
        )));
    }

    let (old_arg, new_arg, count_arg) =
        bind_byte_replace_keyword_args(args, keywords, receiver_name)?;
    let old = old_arg
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "{receiver_name}.replace() missing required argument 'old'"
            ))
        })
        .and_then(bytes_like_argument_bytes)?;
    let new = new_arg
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "{receiver_name}.replace() missing required argument 'new'"
            ))
        })
        .and_then(bytes_like_argument_bytes)?;
    let count = parse_byte_count(count_arg, receiver_name, "replace", 3)?;
    let bytes = receiver(args[0], "replace")?;
    let data = bytes.as_bytes();

    if count == Some(0) || old == new {
        return unchanged_byte_sequence(args[0], data, result_type, identity);
    }

    if old.is_empty() {
        let replaced = replace_empty_byte_pattern(data, &new, count)?;
        return if replaced == data {
            unchanged_byte_sequence(args[0], data, result_type, identity)
        } else {
            Ok(to_object_value(BytesObject::from_vec_with_type(
                replaced,
                result_type,
            )))
        };
    }

    let replacements = count_non_overlapping_limited(data, &old, count.unwrap_or(usize::MAX));
    if replacements == 0 {
        return unchanged_byte_sequence(args[0], data, result_type, identity);
    }

    let replaced = replace_non_empty_byte_pattern(data, &old, &new, replacements)?;
    Ok(to_object_value(BytesObject::from_vec_with_type(
        replaced,
        result_type,
    )))
}

fn byte_sequence_split_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
    receiver_name: &'static str,
    method_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
    result_type: TypeId,
    direction: SplitDirection,
) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 2 {
        return Err(BuiltinError::TypeError(format!(
            "{receiver_name}.{method_name}() takes at most 2 arguments ({given} given)"
        )));
    }

    let (separator_arg, maxsplit_arg) =
        bind_byte_split_keyword_args(args, keywords, receiver_name, method_name)?;
    let separator = match separator_arg {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(bytes_like_argument_bytes(value)?),
    };
    let maxsplit = parse_byte_count(maxsplit_arg, receiver_name, method_name, 2)?;
    let bytes = receiver(args[0], method_name)?;
    let data = bytes.as_bytes();

    let parts = match (separator.as_deref(), direction) {
        (Some(separator), SplitDirection::Forward) => {
            split_byte_sequence(data, separator, maxsplit, result_type)?
        }
        (Some(separator), SplitDirection::Reverse) => {
            rsplit_byte_sequence(data, separator, maxsplit, result_type)?
        }
        (None, SplitDirection::Forward) => {
            split_byte_sequence_whitespace(data, maxsplit, result_type)
        }
        (None, SplitDirection::Reverse) => {
            rsplit_byte_sequence_whitespace(data, maxsplit, result_type)
        }
    };

    Ok(to_object_value(ListObject::from_slice(parts.as_slice())))
}

fn byte_sequence_partition(
    args: &[Value],
    receiver_name: &'static str,
    method_name: &'static str,
    receiver: fn(Value, &'static str) -> Result<&'static BytesObject, BuiltinError>,
    result_type: TypeId,
    direction: SearchDirection,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count(receiver_name, method_name, args, 1)?;
    let bytes = receiver(args[0], method_name)?;
    let separator = bytes_like_argument_bytes(args[1])?;
    if separator.is_empty() {
        return Err(BuiltinError::ValueError("empty separator".to_string()));
    }

    let data = bytes.as_bytes();
    let found = match direction {
        SearchDirection::Forward => simd_bytes_find(data, &separator),
        SearchDirection::Reverse => simd_bytes_rfind(data, &separator),
    };

    let values = match (direction, found) {
        (SearchDirection::Forward, Some(index)) | (SearchDirection::Reverse, Some(index)) => [
            byte_sequence_slice_value(&data[..index], result_type),
            byte_sequence_slice_value(&data[index..index + separator.len()], result_type),
            byte_sequence_slice_value(&data[index + separator.len()..], result_type),
        ],
        (SearchDirection::Forward, None) => [
            byte_sequence_slice_value(data, result_type),
            byte_sequence_slice_value(&[], result_type),
            byte_sequence_slice_value(&[], result_type),
        ],
        (SearchDirection::Reverse, None) => [
            byte_sequence_slice_value(&[], result_type),
            byte_sequence_slice_value(&[], result_type),
            byte_sequence_slice_value(data, result_type),
        ],
    };

    Ok(to_object_value(TupleObject::from_slice(&values)))
}

#[inline]
fn normalize_byte_search_bounds(
    args: &[Value],
    len: usize,
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<Option<(usize, usize)>, BuiltinError> {
    let raw_start = parse_slice_bound(args.get(2).copied(), 0, receiver_name, method_name)?;
    let len_bound = isize::try_from(len).unwrap_or(isize::MAX);
    if raw_start > len_bound {
        return Ok(None);
    }

    let raw_end = parse_slice_bound(
        args.get(3).copied(),
        len as isize,
        receiver_name,
        method_name,
    )?;
    let start = clamp_slice_index(raw_start, len);
    let end = clamp_slice_index(raw_end, len);

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

    if result_type == TypeId::BYTES
        && start == 0
        && end == data.len()
        && value_has_exact_type(args[0], TypeId::BYTES)
    {
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

    if result_type == TypeId::BYTES && !changed && value_has_exact_type(args[0], TypeId::BYTES) {
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

fn bind_byte_replace_keyword_args(
    args: &[Value],
    keywords: &[(&str, Value)],
    receiver_name: &'static str,
) -> Result<(Option<Value>, Option<Value>, Option<Value>), BuiltinError> {
    let mut old = args.get(1).copied();
    let mut new = args.get(2).copied();
    let mut count = args.get(3).copied();

    for (name, value) in keywords {
        match *name {
            "old" => {
                if old.is_some() {
                    return Err(BuiltinError::TypeError(
                        "replace() got multiple values for argument 'old'".to_string(),
                    ));
                }
                old = Some(*value);
            }
            "new" => {
                if new.is_some() {
                    return Err(BuiltinError::TypeError(
                        "replace() got multiple values for argument 'new'".to_string(),
                    ));
                }
                new = Some(*value);
            }
            "count" => {
                if count.is_some() {
                    return Err(BuiltinError::TypeError(
                        "replace() got multiple values for argument 'count'".to_string(),
                    ));
                }
                count = Some(*value);
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "{receiver_name}.replace() got an unexpected keyword argument '{other}'"
                )));
            }
        }
    }

    Ok((old, new, count))
}

fn bind_byte_split_keyword_args(
    args: &[Value],
    keywords: &[(&str, Value)],
    receiver_name: &'static str,
    method_name: &'static str,
) -> Result<(Option<Value>, Option<Value>), BuiltinError> {
    let mut separator = args.get(1).copied();
    let mut maxsplit = args.get(2).copied();

    for (name, value) in keywords {
        match *name {
            "sep" => {
                if separator.is_some() {
                    return Err(BuiltinError::TypeError(format!(
                        "{method_name}() got multiple values for argument 'sep'"
                    )));
                }
                separator = Some(*value);
            }
            "maxsplit" => {
                if maxsplit.is_some() {
                    return Err(BuiltinError::TypeError(format!(
                        "{method_name}() got multiple values for argument 'maxsplit'"
                    )));
                }
                maxsplit = Some(*value);
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "{receiver_name}.{method_name}() got an unexpected keyword argument '{other}'"
                )));
            }
        }
    }

    Ok((separator, maxsplit))
}

#[inline]
fn parse_byte_count(
    count: Option<Value>,
    receiver_name: &'static str,
    method_name: &'static str,
    position: usize,
) -> Result<Option<usize>, BuiltinError> {
    let Some(count) = count else {
        return Ok(None);
    };

    if let Some(value) = count.as_bool() {
        return Ok(Some(if value { 1 } else { 0 }));
    }

    if let Some(value) = value_to_saturated_i64(count) {
        if value < 0 {
            return Ok(None);
        }
        return Ok(Some(usize::try_from(value).unwrap_or(usize::MAX)));
    }

    Err(BuiltinError::TypeError(format!(
        "{receiver_name}.{method_name}() argument {position} must be int, not {}",
        count.type_name()
    )))
}

fn unchanged_byte_sequence(
    receiver: Value,
    data: &[u8],
    result_type: TypeId,
    identity: ReturnIdentity,
) -> Result<Value, BuiltinError> {
    if result_type == TypeId::BYTES
        && matches!(identity, ReturnIdentity::WhenUnchanged)
        && value_has_exact_type(receiver, TypeId::BYTES)
    {
        return Ok(receiver);
    }

    Ok(to_object_value(BytesObject::from_vec_with_type(
        data.to_vec(),
        result_type,
    )))
}

fn replace_empty_byte_pattern(
    data: &[u8],
    replacement: &[u8],
    count: Option<usize>,
) -> Result<Vec<u8>, BuiltinError> {
    let insertions = count.unwrap_or(data.len().saturating_add(1)).min(
        data.len()
            .checked_add(1)
            .ok_or_else(|| BuiltinError::OverflowError("result too large".to_string()))?,
    );
    if insertions == 0 {
        return Ok(data.to_vec());
    }

    let replacement_bytes = replacement
        .len()
        .checked_mul(insertions)
        .ok_or_else(|| BuiltinError::OverflowError("result too large".to_string()))?;
    let capacity = data
        .len()
        .checked_add(replacement_bytes)
        .ok_or_else(|| BuiltinError::OverflowError("result too large".to_string()))?;
    let mut result = Vec::with_capacity(capacity);
    let mut remaining = insertions;

    result.extend_from_slice(replacement);
    remaining -= 1;
    for &byte in data {
        result.push(byte);
        if remaining > 0 {
            result.extend_from_slice(replacement);
            remaining -= 1;
        }
    }

    Ok(result)
}

fn count_non_overlapping_limited(data: &[u8], needle: &[u8], limit: usize) -> usize {
    if limit == 0 {
        return 0;
    }

    let mut count = 0usize;
    let mut start = 0usize;
    while count < limit {
        let Some(offset) = simd_bytes_find(&data[start..], needle) else {
            break;
        };
        count += 1;
        start += offset + needle.len();
    }
    count
}

fn replace_non_empty_byte_pattern(
    data: &[u8],
    old: &[u8],
    new: &[u8],
    replacements: usize,
) -> Result<Vec<u8>, BuiltinError> {
    let capacity = if new.len() >= old.len() {
        data.len()
            .checked_add(
                new.len()
                    .checked_sub(old.len())
                    .and_then(|delta| delta.checked_mul(replacements))
                    .ok_or_else(|| BuiltinError::OverflowError("result too large".to_string()))?,
            )
            .ok_or_else(|| BuiltinError::OverflowError("result too large".to_string()))?
    } else {
        data.len() - (old.len() - new.len()) * replacements
    };
    let mut result = Vec::with_capacity(capacity);
    let mut start = 0usize;
    let mut remaining = replacements;

    while remaining > 0 {
        let offset = simd_bytes_find(&data[start..], old)
            .expect("pre-counted replacements must be discoverable");
        let index = start + offset;
        result.extend_from_slice(&data[start..index]);
        result.extend_from_slice(new);
        start = index + old.len();
        remaining -= 1;
    }
    result.extend_from_slice(&data[start..]);
    Ok(result)
}

fn split_byte_sequence(
    data: &[u8],
    separator: &[u8],
    maxsplit: Option<usize>,
    result_type: TypeId,
) -> Result<Vec<Value>, BuiltinError> {
    if separator.is_empty() {
        return Err(BuiltinError::ValueError("empty separator".to_string()));
    }

    let limit = maxsplit.unwrap_or(usize::MAX);
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut splits = 0usize;
    while splits < limit {
        let Some(offset) = simd_bytes_find(&data[start..], separator) else {
            break;
        };
        let index = start + offset;
        parts.push(byte_sequence_slice_value(&data[start..index], result_type));
        start = index + separator.len();
        splits += 1;
    }
    parts.push(byte_sequence_slice_value(&data[start..], result_type));
    Ok(parts)
}

fn rsplit_byte_sequence(
    data: &[u8],
    separator: &[u8],
    maxsplit: Option<usize>,
    result_type: TypeId,
) -> Result<Vec<Value>, BuiltinError> {
    if separator.is_empty() {
        return Err(BuiltinError::ValueError("empty separator".to_string()));
    }

    let limit = maxsplit.unwrap_or(usize::MAX);
    if limit == 0 {
        return Ok(vec![byte_sequence_slice_value(data, result_type)]);
    }

    let mut parts = Vec::new();
    let mut end = data.len();
    let mut splits = 0usize;
    while splits < limit {
        let Some(index) = simd_bytes_rfind(&data[..end], separator) else {
            break;
        };
        parts.push(byte_sequence_slice_value(
            &data[index + separator.len()..end],
            result_type,
        ));
        end = index;
        splits += 1;
    }
    parts.push(byte_sequence_slice_value(&data[..end], result_type));
    parts.reverse();
    Ok(parts)
}

fn split_byte_sequence_whitespace(
    data: &[u8],
    maxsplit: Option<usize>,
    result_type: TypeId,
) -> Vec<Value> {
    let mut start = trim_ascii_whitespace_start(data, 0);
    if start == data.len() {
        return Vec::new();
    }

    let limit = maxsplit.unwrap_or(usize::MAX);
    if limit == 0 {
        return vec![byte_sequence_slice_value(&data[start..], result_type)];
    }

    let mut parts = Vec::new();
    let mut splits = 0usize;
    while start < data.len() {
        if splits == limit {
            parts.push(byte_sequence_slice_value(&data[start..], result_type));
            break;
        }

        let end = next_ascii_whitespace(data, start);
        parts.push(byte_sequence_slice_value(&data[start..end], result_type));
        start = trim_ascii_whitespace_start(data, end);
        splits += 1;
    }
    parts
}

fn rsplit_byte_sequence_whitespace(
    data: &[u8],
    maxsplit: Option<usize>,
    result_type: TypeId,
) -> Vec<Value> {
    let mut end = trim_ascii_whitespace_end(data, data.len());
    if end == 0 {
        return Vec::new();
    }

    let limit = maxsplit.unwrap_or(usize::MAX);
    if limit == 0 {
        return vec![byte_sequence_slice_value(&data[..end], result_type)];
    }

    let mut parts = Vec::new();
    let mut splits = 0usize;
    while end > 0 && splits < limit {
        let mut word_start = end;
        while word_start > 0 && !is_ascii_byte_whitespace(data[word_start - 1]) {
            word_start -= 1;
        }

        parts.push(byte_sequence_slice_value(
            &data[word_start..end],
            result_type,
        ));

        end = trim_ascii_whitespace_end(data, word_start);
        splits += 1;
    }

    if end > 0 {
        let start = if splits == limit {
            0
        } else {
            trim_ascii_whitespace_start(data, 0)
        };
        if start < end {
            parts.push(byte_sequence_slice_value(&data[start..end], result_type));
        }
    }
    parts.reverse();
    parts
}

#[inline]
fn byte_sequence_slice_value(data: &[u8], result_type: TypeId) -> Value {
    to_object_value(BytesObject::from_vec_with_type(data.to_vec(), result_type))
}

#[inline]
fn value_has_exact_type(value: Value, expected: TypeId) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    header.type_id == expected
}

#[inline]
fn trim_ascii_whitespace_start(data: &[u8], mut index: usize) -> usize {
    while index < data.len() && is_ascii_byte_whitespace(data[index]) {
        index += 1;
    }
    index
}

#[inline]
fn trim_ascii_whitespace_end(data: &[u8], mut end: usize) -> usize {
    while end > 0 && is_ascii_byte_whitespace(data[end - 1]) {
        end -= 1;
    }
    end
}

#[inline]
fn next_ascii_whitespace(data: &[u8], mut index: usize) -> usize {
    while index < data.len() && !is_ascii_byte_whitespace(data[index]) {
        index += 1;
    }
    index
}

#[inline]
fn previous_ascii_whitespace(data: &[u8], start: usize, mut end: usize) -> Option<usize> {
    while end > start {
        end -= 1;
        if is_ascii_byte_whitespace(data[end]) {
            return Some(end);
        }
    }
    None
}

#[inline]
fn is_ascii_byte_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
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
