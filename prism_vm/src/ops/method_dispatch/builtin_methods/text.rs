//! String builtin method resolution and implementations.

use super::*;

static STR_REPR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.__repr__"), value_repr));
static STR_STR_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.__str__"), value_str));
static STR_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.__format__"), value_format));
static STR_FORMAT_CALL_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("str.format"),
        crate::stdlib::_string::builtin_str_format_method,
    )
});
static STR_ENCODE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.encode"), str_encode));

static STR_UPPER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.upper"), str_upper));
static STR_LOWER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.lower"), str_lower));
static STR_CAPITALIZE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.capitalize"), str_capitalize));
static STR_REPLACE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("str.replace"), str_replace_kw));
static STR_REMOVEPREFIX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.removeprefix"), str_removeprefix));
static STR_REMOVESUFFIX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.removesuffix"), str_removesuffix));
static STR_SPLIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("str.split"), str_split_kw));
static STR_RSPLIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("str.rsplit"), str_rsplit_kw));
static STR_SPLITLINES_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("str.splitlines"), str_splitlines_kw));
static STR_EXPANDTABS_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("str.expandtabs"), str_expandtabs_kw));
static STR_TRANSLATE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("str.translate"), str_translate));
static STR_FIND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.find"), str_find));
static STR_RFIND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.rfind"), str_rfind));
static STR_INDEX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.index"), str_index));
static STR_RINDEX_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.rindex"), str_rindex));
static STR_COUNT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.count"), str_count));
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
static STR_ISASCII_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.isascii"), str_isascii));
static STR_ISALPHA_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.isalpha"), str_isalpha));
static STR_ISDIGIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.isdigit"), str_isdigit));
static STR_ISALNUM_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.isalnum"), str_isalnum));
static STR_ISSPACE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.isspace"), str_isspace));
static STR_ISUPPER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.isupper"), str_isupper));
static STR_ISLOWER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.islower"), str_islower));
static STR_ISDECIMAL_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.isdecimal"), str_isdecimal));
static STR_ISNUMERIC_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.isnumeric"), str_isnumeric));
static STR_ISTITLE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.istitle"), str_istitle));
static STR_STARTSWITH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.startswith"), str_startswith));
static STR_ENDSWITH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.endswith"), str_endswith));
static STR_PARTITION_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.partition"), str_partition));
static STR_RPARTITION_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("str.rpartition"), str_rpartition));

pub fn resolve_str_method(name: &str) -> Option<CachedMethod> {
    match name {
        "__repr__" => Some(CachedMethod::simple(builtin_method_value(&STR_REPR_METHOD))),
        "__str__" => Some(CachedMethod::simple(builtin_method_value(&STR_STR_METHOD))),
        "__format__" => Some(CachedMethod::simple(builtin_method_value(
            &STR_FORMAT_METHOD,
        ))),
        "format" => Some(CachedMethod::simple(builtin_method_value(
            &STR_FORMAT_CALL_METHOD,
        ))),
        "encode" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ENCODE_METHOD,
        ))),
        "upper" => Some(CachedMethod::simple(builtin_method_value(
            &STR_UPPER_METHOD,
        ))),
        "lower" => Some(CachedMethod::simple(builtin_method_value(
            &STR_LOWER_METHOD,
        ))),
        "capitalize" => Some(CachedMethod::simple(builtin_method_value(
            &STR_CAPITALIZE_METHOD,
        ))),
        "replace" => Some(CachedMethod::simple(builtin_method_value(
            &STR_REPLACE_METHOD,
        ))),
        "removeprefix" => Some(CachedMethod::simple(builtin_method_value(
            &STR_REMOVEPREFIX_METHOD,
        ))),
        "removesuffix" => Some(CachedMethod::simple(builtin_method_value(
            &STR_REMOVESUFFIX_METHOD,
        ))),
        "split" => Some(CachedMethod::simple(builtin_method_value(
            &STR_SPLIT_METHOD,
        ))),
        "rsplit" => Some(CachedMethod::simple(builtin_method_value(
            &STR_RSPLIT_METHOD,
        ))),
        "splitlines" => Some(CachedMethod::simple(builtin_method_value(
            &STR_SPLITLINES_METHOD,
        ))),
        "expandtabs" => Some(CachedMethod::simple(builtin_method_value(
            &STR_EXPANDTABS_METHOD,
        ))),
        "translate" => Some(CachedMethod::simple(builtin_method_value(
            &STR_TRANSLATE_METHOD,
        ))),
        "find" => Some(CachedMethod::simple(builtin_method_value(&STR_FIND_METHOD))),
        "rfind" => Some(CachedMethod::simple(builtin_method_value(
            &STR_RFIND_METHOD,
        ))),
        "index" => Some(CachedMethod::simple(builtin_method_value(
            &STR_INDEX_METHOD,
        ))),
        "rindex" => Some(CachedMethod::simple(builtin_method_value(
            &STR_RINDEX_METHOD,
        ))),
        "count" => Some(CachedMethod::simple(builtin_method_value(
            &STR_COUNT_METHOD,
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
        "isascii" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISASCII_METHOD,
        ))),
        "isalpha" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISALPHA_METHOD,
        ))),
        "isdigit" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISDIGIT_METHOD,
        ))),
        "isalnum" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISALNUM_METHOD,
        ))),
        "isspace" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISSPACE_METHOD,
        ))),
        "isupper" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISUPPER_METHOD,
        ))),
        "islower" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISLOWER_METHOD,
        ))),
        "isdecimal" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISDECIMAL_METHOD,
        ))),
        "isnumeric" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISNUMERIC_METHOD,
        ))),
        "istitle" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ISTITLE_METHOD,
        ))),
        "startswith" => Some(CachedMethod::simple(builtin_method_value(
            &STR_STARTSWITH_METHOD,
        ))),
        "endswith" => Some(CachedMethod::simple(builtin_method_value(
            &STR_ENDSWITH_METHOD,
        ))),
        "partition" => Some(CachedMethod::simple(builtin_method_value(
            &STR_PARTITION_METHOD,
        ))),
        "rpartition" => Some(CachedMethod::simple(builtin_method_value(
            &STR_RPARTITION_METHOD,
        ))),
        _ => None,
    }
}

#[inline]
pub(super) fn str_encode(args: &[Value]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 2 {
        return Err(BuiltinError::TypeError(format!(
            "encode() takes at most 2 arguments ({given} given)"
        )));
    }

    let encoding = args
        .get(1)
        .copied()
        .map(|value| expect_method_string_arg(value, "str", "encode", 1))
        .transpose()?;
    let errors = args
        .get(2)
        .copied()
        .map(|value| expect_method_string_arg(value, "str", "encode", 2))
        .transpose()?;

    with_str_receiver(args[0], "encode", |value| {
        crate::builtins::encode_text_to_value(
            value,
            encoding.as_deref(),
            errors.as_deref(),
            TypeId::BYTES,
        )
    })
}

#[inline]
pub(super) fn str_upper(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn str_lower(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn str_capitalize(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "capitalize", args, 0)?;
    with_str_receiver(args[0], "capitalize", |value| {
        if value.is_empty() {
            return Ok(args[0]);
        }

        if value.is_ascii() {
            let mut bytes = value.as_bytes().to_vec();
            let mut changed = false;
            if let Some(first) = bytes.first_mut() {
                let upper = first.to_ascii_uppercase();
                changed |= upper != *first;
                *first = upper;
            }
            for byte in bytes.iter_mut().skip(1) {
                let lower = byte.to_ascii_lowercase();
                changed |= lower != *byte;
                *byte = lower;
            }

            return if changed {
                let capitalized = unsafe { String::from_utf8_unchecked(bytes) };
                Ok(Value::string(intern(&capitalized)))
            } else {
                Ok(args[0])
            };
        }

        let mut chars = value.chars();
        let Some(first) = chars.next() else {
            return Ok(args[0]);
        };
        let mut capitalized = String::with_capacity(value.len());
        capitalized.extend(first.to_uppercase());
        for ch in chars {
            capitalized.extend(ch.to_lowercase());
        }

        if capitalized == value {
            Ok(args[0])
        } else {
            Ok(Value::string(intern(&capitalized)))
        }
    })
}

#[inline]
pub(super) fn str_replace(args: &[Value]) -> Result<Value, BuiltinError> {
    str_replace_kw(args, &[])
}

pub(super) fn str_replace_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if !(2..=3).contains(&given) {
        if keywords.is_empty() {
            return Err(BuiltinError::TypeError(format!(
                "str.replace() takes from 2 to 3 arguments ({given} given)"
            )));
        }
        if given > 3 {
            return Err(BuiltinError::TypeError(format!(
                "str.replace() takes from 2 to 3 arguments ({given} given)"
            )));
        }
    }

    let (old_arg, new_arg, count_arg) = bind_replace_keyword_args(args, keywords)?;
    let old = expect_str_method_string_arg(
        old_arg.ok_or_else(|| {
            BuiltinError::TypeError("str.replace() missing required argument 'old'".to_string())
        })?,
        "replace",
        1,
    )?;
    let new = expect_str_method_string_arg(
        new_arg.ok_or_else(|| {
            BuiltinError::TypeError("str.replace() missing required argument 'new'".to_string())
        })?,
        "replace",
        2,
    )?;
    let count = parse_replace_count(count_arg)?;

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
pub(super) fn str_removeprefix(args: &[Value]) -> Result<Value, BuiltinError> {
    remove_affix(args, "removeprefix", RemoveAffixDirection::Prefix)
}

#[inline]
pub(super) fn str_removesuffix(args: &[Value]) -> Result<Value, BuiltinError> {
    remove_affix(args, "removesuffix", RemoveAffixDirection::Suffix)
}

#[inline]
pub(super) fn str_split(args: &[Value]) -> Result<Value, BuiltinError> {
    str_split_kw(args, &[])
}

#[inline]
pub(super) fn str_split_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 2 {
        return Err(BuiltinError::TypeError(format!(
            "str.split() takes at most 2 arguments ({given} given)"
        )));
    }

    let (separator_arg, maxsplit_arg) =
        bind_split_keyword_args(args, keywords, "split", "str.split")?;
    let separator = match separator_arg {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(expect_str_method_string_arg(value, "split", 1)?),
    };
    let maxsplit = parse_split_count(maxsplit_arg, "split", 2)?;

    with_str_receiver(args[0], "split", |value| {
        let parts = match separator.as_deref() {
            Some(separator) => split_with_separator(value, separator, maxsplit)?,
            None => split_on_whitespace(value, maxsplit),
        };

        Ok(to_object_value(ListObject::from_slice(parts.as_slice())))
    })
}

#[inline]
pub(super) fn str_rsplit(args: &[Value]) -> Result<Value, BuiltinError> {
    str_rsplit_kw(args, &[])
}

#[inline]
pub(super) fn str_rsplit_kw(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 2 {
        return Err(BuiltinError::TypeError(format!(
            "str.rsplit() takes at most 2 arguments ({given} given)"
        )));
    }

    let (separator_arg, maxsplit_arg) =
        bind_split_keyword_args(args, keywords, "rsplit", "str.rsplit")?;
    let separator = match separator_arg {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(expect_str_method_string_arg(value, "rsplit", 1)?),
    };
    let maxsplit = parse_split_count(maxsplit_arg, "rsplit", 2)?;

    with_str_receiver(args[0], "rsplit", |value| {
        let parts = match separator.as_deref() {
            Some(separator) => rsplit_with_separator(value, separator, maxsplit)?,
            None => rsplit_on_whitespace(value, maxsplit),
        };

        Ok(to_object_value(ListObject::from_slice(parts.as_slice())))
    })
}

#[inline]
pub(super) fn str_splitlines(args: &[Value]) -> Result<Value, BuiltinError> {
    str_splitlines_kw(args, &[])
}

#[inline]
pub(super) fn str_expandtabs(args: &[Value]) -> Result<Value, BuiltinError> {
    str_expandtabs_kw(args, &[])
}

#[inline]
fn str_expandtabs_kw(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let positional = args.len().saturating_sub(1);
    let total = positional + keywords.len();
    if total > 1 {
        return Err(BuiltinError::TypeError(format!(
            "expandtabs() takes at most 1 argument ({total} given)"
        )));
    }

    let mut tabsize_arg = args.get(1).copied();
    for (name, value) in keywords {
        if *name != "tabsize" {
            return Err(BuiltinError::TypeError(format!(
                "expandtabs() got an unexpected keyword argument '{}'",
                name
            )));
        }
        tabsize_arg = Some(*value);
    }

    let tabsize = tabsize_arg
        .map(expect_integer_like_index)
        .transpose()?
        .unwrap_or(8);
    with_str_receiver(args[0], "expandtabs", |value| {
        if !value.contains('\t') {
            return Ok(args[0]);
        }

        let expanded = expand_tabs(value, tabsize);
        if expanded == value {
            Ok(args[0])
        } else {
            Ok(Value::string(intern(&expanded)))
        }
    })
}

#[inline]
fn str_splitlines_kw(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if given > 1 {
        return Err(BuiltinError::TypeError(format!(
            "str.splitlines() takes at most 1 argument ({given} given)"
        )));
    }

    let mut keepends_arg = args.get(1).copied();
    for (name, value) in keywords {
        if *name != "keepends" {
            return Err(BuiltinError::TypeError(format!(
                "splitlines() got an unexpected keyword argument '{}'",
                name
            )));
        }
        if keepends_arg.is_some() {
            return Err(BuiltinError::TypeError(
                "splitlines() got multiple values for argument 'keepends'".to_string(),
            ));
        }
        keepends_arg = Some(*value);
    }

    let keepends = parse_keepends_flag(keepends_arg, "splitlines", 1)?;
    with_str_receiver(args[0], "splitlines", |value| {
        Ok(to_object_value(ListObject::from_slice(
            split_lines(value, keepends).as_slice(),
        )))
    })
}

#[inline]
pub(super) fn str_translate(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "translate", args, 1)?;
    let table = args[1];

    with_str_receiver(args[0], "translate", |value| {
        let mut translated: Option<String> = None;

        for (byte_index, ch) in value.char_indices() {
            let Some(replacement) = str_translate_lookup(vm, table, ch as u32)? else {
                if let Some(output) = translated.as_mut() {
                    output.push(ch);
                }
                continue;
            };

            let action = str_translate_action(replacement)?;
            if let Some(output) = translated.as_mut() {
                push_str_translate_action(output, &action);
            } else if !str_translate_action_is_unchanged(&action, ch) {
                let mut output = String::with_capacity(value.len());
                output.push_str(&value[..byte_index]);
                push_str_translate_action(&mut output, &action);
                translated = Some(output);
            }
        }

        match translated {
            Some(output) => {
                alloc_heap_value(vm, StringObject::new(&output), "str.translate result")
                    .map_err(runtime_error_to_builtin_error)
            }
            None => Ok(args[0]),
        }
    })
}

#[inline]
fn str_translate_lookup(
    vm: &mut VirtualMachine,
    table: Value,
    ordinal: u32,
) -> Result<Option<Value>, BuiltinError> {
    let key = Value::int(ordinal as i64).expect("Unicode ordinal should fit in tagged int");

    if let Some(ptr) = table.as_object_ptr() {
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        match header.type_id {
            TypeId::DICT => {
                let dict = unsafe { &*(ptr as *const DictObject) };
                return Ok(dict.get(key));
            }
            TypeId::MAPPING_PROXY => {
                let proxy = unsafe { &*(ptr as *const MappingProxyObject) };
                return builtin_mapping_proxy_get_item_static(proxy, key)
                    .map_err(runtime_error_to_builtin_error);
            }
            _ => {}
        }
    }

    let get_item =
        resolve_special_method(table, "__getitem__").map_err(runtime_error_to_builtin_error)?;
    match invoke_bound_method_with_arg(vm, &get_item, key) {
        Ok(value) => Ok(Some(value)),
        Err(BuiltinError::KeyError(_) | BuiltinError::IndexError(_)) => Ok(None),
        Err(err) => Err(err),
    }
}

enum StrTranslateAction {
    Delete,
    Char(char),
    Text(StringValueRef<'static>),
}

#[inline]
fn str_translate_action(value: Value) -> Result<StrTranslateAction, BuiltinError> {
    if value.is_none() {
        return Ok(StrTranslateAction::Delete);
    }

    if let Some(text) = value_as_string_ref(value) {
        return Ok(StrTranslateAction::Text(text));
    }

    if let Some(codepoint) = str_translate_codepoint(value)? {
        let Some(ch) = char::from_u32(codepoint) else {
            return Err(BuiltinError::ValueError(
                "character mapping must be in range(0x110000)".to_string(),
            ));
        };
        return Ok(StrTranslateAction::Char(ch));
    }

    Err(BuiltinError::TypeError(
        "character mapping must return integer, None or str".to_string(),
    ))
}

#[inline]
fn str_translate_codepoint(value: Value) -> Result<Option<u32>, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(Some(if flag { 1 } else { 0 }));
    }

    let Some(integer) = value_to_bigint(value) else {
        return Ok(None);
    };
    let Some(codepoint) = integer.to_u32() else {
        return Err(BuiltinError::ValueError(
            "character mapping must be in range(0x110000)".to_string(),
        ));
    };
    if codepoint > 0x10ffff {
        return Err(BuiltinError::ValueError(
            "character mapping must be in range(0x110000)".to_string(),
        ));
    }
    Ok(Some(codepoint))
}

#[inline]
fn push_str_translate_action(output: &mut String, action: &StrTranslateAction) {
    match action {
        StrTranslateAction::Delete => {}
        StrTranslateAction::Char(ch) => output.push(*ch),
        StrTranslateAction::Text(text) => output.push_str(text.as_str()),
    }
}

#[inline]
fn str_translate_action_is_unchanged(action: &StrTranslateAction, original: char) -> bool {
    match action {
        StrTranslateAction::Delete => false,
        StrTranslateAction::Char(ch) => *ch == original,
        StrTranslateAction::Text(text) => {
            let mut chars = text.as_str().chars();
            chars.next() == Some(original) && chars.next().is_none()
        }
    }
}

#[inline]
pub(super) fn str_find(args: &[Value]) -> Result<Value, BuiltinError> {
    string_find_like(
        args,
        "find",
        SearchDirection::Forward,
        MissingNeedle::ReturnMinusOne,
    )
}

#[inline]
pub(super) fn str_rfind(args: &[Value]) -> Result<Value, BuiltinError> {
    string_find_like(
        args,
        "rfind",
        SearchDirection::Reverse,
        MissingNeedle::ReturnMinusOne,
    )
}

#[inline]
pub(super) fn str_index(args: &[Value]) -> Result<Value, BuiltinError> {
    string_find_like(
        args,
        "index",
        SearchDirection::Forward,
        MissingNeedle::RaiseValueError,
    )
}

#[inline]
pub(super) fn str_rindex(args: &[Value]) -> Result<Value, BuiltinError> {
    string_find_like(
        args,
        "rindex",
        SearchDirection::Reverse,
        MissingNeedle::RaiseValueError,
    )
}

#[inline]
pub(super) fn str_count(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "str.count() takes from 1 to 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let needle = expect_str_method_string_arg(args[1], "count", 1)?;
    with_str_receiver(args[0], "count", |value| {
        let bounds = normalize_slice_bounds_with_chars(
            value,
            args.get(2).copied(),
            args.get(3).copied(),
            "count",
        )?;
        let slice = &value[bounds.start_byte..bounds.end_byte];
        let count = if needle.is_empty() {
            bounds.end_char - bounds.start_char + 1
        } else {
            slice.matches(&needle).count()
        };
        Ok(Value::int(count as i64).expect("substring count should fit in tagged int"))
    })
}

#[inline]
pub(super) fn str_strip(args: &[Value]) -> Result<Value, BuiltinError> {
    strip_method(args, "strip", StripDirection::Both)
}

#[inline]
pub(super) fn str_lstrip(args: &[Value]) -> Result<Value, BuiltinError> {
    strip_method(args, "lstrip", StripDirection::Leading)
}

#[inline]
pub(super) fn str_rstrip(args: &[Value]) -> Result<Value, BuiltinError> {
    strip_method(args, "rstrip", StripDirection::Trailing)
}

#[inline]
pub(super) fn str_join(args: &[Value]) -> Result<Value, BuiltinError> {
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
pub(super) fn str_isidentifier(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "isidentifier", args, 0)?;
    with_str_receiver(args[0], "isidentifier", |value| {
        Ok(Value::bool(is_python_identifier(value)))
    })
}

#[inline]
pub(super) fn str_isascii(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "isascii", args, 0)?;
    with_str_receiver(args[0], "isascii", |value| {
        Ok(Value::bool(value.is_ascii()))
    })
}

#[inline]
pub(super) fn str_isalpha(args: &[Value]) -> Result<Value, BuiltinError> {
    str_char_property(args, "isalpha", |ch| ch.is_alphabetic())
}

#[inline]
pub(super) fn str_isdigit(args: &[Value]) -> Result<Value, BuiltinError> {
    str_char_property(args, "isdigit", |ch| ch.to_digit(10).is_some())
}

#[inline]
pub(super) fn str_isalnum(args: &[Value]) -> Result<Value, BuiltinError> {
    str_char_property(args, "isalnum", |ch| ch.is_alphanumeric())
}

#[inline]
pub(super) fn str_isspace(args: &[Value]) -> Result<Value, BuiltinError> {
    str_char_property(args, "isspace", |ch| ch.is_whitespace())
}

#[inline]
pub(super) fn str_isupper(args: &[Value]) -> Result<Value, BuiltinError> {
    str_cased_property(args, "isupper", StringCaseProperty::Upper)
}

#[inline]
pub(super) fn str_islower(args: &[Value]) -> Result<Value, BuiltinError> {
    str_cased_property(args, "islower", StringCaseProperty::Lower)
}

#[inline]
pub(super) fn str_isdecimal(args: &[Value]) -> Result<Value, BuiltinError> {
    str_char_property(args, "isdecimal", |ch| ch.to_digit(10).is_some())
}

#[inline]
pub(super) fn str_isnumeric(args: &[Value]) -> Result<Value, BuiltinError> {
    str_char_property(args, "isnumeric", |ch| ch.is_numeric())
}

#[inline]
pub(super) fn str_istitle(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", "istitle", args, 0)?;
    with_str_receiver(args[0], "istitle", |value| {
        Ok(Value::bool(is_titlecase_string(value)))
    })
}

#[inline]
pub(super) fn str_startswith(args: &[Value]) -> Result<Value, BuiltinError> {
    affix_match(args, "startswith", |value, affix| value.starts_with(affix))
}

#[inline]
pub(super) fn str_endswith(args: &[Value]) -> Result<Value, BuiltinError> {
    affix_match(args, "endswith", |value, affix| value.ends_with(affix))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RemoveAffixDirection {
    Prefix,
    Suffix,
}

#[inline]
fn remove_affix(
    args: &[Value],
    method_name: &'static str,
    direction: RemoveAffixDirection,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", method_name, args, 1)?;
    let affix = expect_str_method_string_arg(args[1], method_name, 1)?;

    with_str_receiver(args[0], method_name, |value| {
        if affix.is_empty() {
            return Ok(args[0]);
        }

        let stripped = match direction {
            RemoveAffixDirection::Prefix => value.strip_prefix(&affix),
            RemoveAffixDirection::Suffix => value.strip_suffix(&affix),
        };
        match stripped {
            Some(rest) => Ok(Value::string(intern(rest))),
            None => Ok(args[0]),
        }
    })
}

#[inline]
pub(super) fn str_partition(args: &[Value]) -> Result<Value, BuiltinError> {
    str_partition_impl(args, "partition", PartitionDirection::Forward)
}

#[inline]
pub(super) fn str_rpartition(args: &[Value]) -> Result<Value, BuiltinError> {
    str_partition_impl(args, "rpartition", PartitionDirection::Reverse)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PartitionDirection {
    Forward,
    Reverse,
}

#[inline]
fn str_partition_impl(
    args: &[Value],
    method_name: &'static str,
    direction: PartitionDirection,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", method_name, args, 1)?;
    let separator = expect_str_method_string_arg(args[1], method_name, 1)?;
    if separator.is_empty() {
        return Err(BuiltinError::ValueError("empty separator".to_string()));
    }

    with_str_receiver(args[0], method_name, |value| {
        let parts = match direction {
            PartitionDirection::Forward => {
                if let Some(index) = value.find(&separator) {
                    (
                        Value::string(intern(&value[..index])),
                        Value::string(intern(&separator)),
                        Value::string(intern(&value[index + separator.len()..])),
                    )
                } else {
                    (
                        Value::string(intern(value)),
                        Value::string(intern("")),
                        Value::string(intern("")),
                    )
                }
            }
            PartitionDirection::Reverse => {
                if let Some(index) = value.rfind(&separator) {
                    (
                        Value::string(intern(&value[..index])),
                        Value::string(intern(&separator)),
                        Value::string(intern(&value[index + separator.len()..])),
                    )
                } else {
                    (
                        Value::string(intern("")),
                        Value::string(intern("")),
                        Value::string(intern(value)),
                    )
                }
            }
        };
        Ok(to_object_value(TupleObject::from_slice(&[
            parts.0, parts.1, parts.2,
        ])))
    })
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
fn string_find_like(
    args: &[Value],
    method_name: &'static str,
    direction: SearchDirection,
    missing: MissingNeedle,
) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "str.{method_name}() takes from 1 to 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let needle = expect_str_method_string_arg(args[1], method_name, 1)?;
    with_str_receiver(args[0], method_name, |value| {
        let bounds = normalize_slice_bounds_with_chars(
            value,
            args.get(2).copied(),
            args.get(3).copied(),
            method_name,
        )?;
        let slice = &value[bounds.start_byte..bounds.end_byte];
        let match_index = match direction {
            SearchDirection::Forward => slice.find(&needle),
            SearchDirection::Reverse => slice.rfind(&needle),
        }
        .map(|byte_offset| bounds.start_char + slice[..byte_offset].chars().count());

        match match_index {
            Some(index) => {
                Ok(Value::int(index as i64).expect("substring index should fit in tagged int"))
            }
            None => match missing {
                MissingNeedle::ReturnMinusOne => Ok(Value::int(-1).unwrap()),
                MissingNeedle::RaiseValueError => {
                    Err(BuiltinError::ValueError("substring not found".to_string()))
                }
            },
        }
    })
}

#[inline]
fn str_char_property(
    args: &[Value],
    method_name: &'static str,
    predicate: impl Fn(char) -> bool,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", method_name, args, 0)?;
    with_str_receiver(args[0], method_name, |value| {
        Ok(Value::bool(
            !value.is_empty() && value.chars().all(&predicate),
        ))
    })
}

#[inline]
fn str_cased_property(
    args: &[Value],
    method_name: &'static str,
    expected_case: StringCaseProperty,
) -> Result<Value, BuiltinError> {
    expect_method_arg_count("str", method_name, args, 0)?;
    with_str_receiver(args[0], method_name, |value| {
        Ok(Value::bool(matches_cased_property(value, expected_case)))
    })
}

#[inline]

fn matches_cased_property(value: &str, expected_case: StringCaseProperty) -> bool {
    let mut saw_cased = false;

    for ch in value.chars() {
        if ch.is_uppercase() {
            saw_cased = true;
            if expected_case == StringCaseProperty::Lower {
                return false;
            }
        } else if ch.is_lowercase() {
            saw_cased = true;
            if expected_case == StringCaseProperty::Upper {
                return false;
            }
        }
    }

    saw_cased
}

#[inline]
fn is_titlecase_string(value: &str) -> bool {
    let mut saw_cased = false;
    let mut expect_titlecase = true;

    for ch in value.chars() {
        let is_upper = ch.is_uppercase();
        let is_lower = ch.is_lowercase();
        if !is_upper && !is_lower {
            expect_titlecase = true;
            continue;
        }

        saw_cased = true;
        if expect_titlecase {
            if !is_upper {
                return false;
            }
            expect_titlecase = false;
        } else if !is_lower {
            return false;
        }
    }

    saw_cased
}

#[inline]
pub(super) fn char_index_to_byte_offset(value: &str, index: usize) -> usize {
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
fn bind_split_keyword_args(
    args: &[Value],
    keywords: &[(&str, Value)],
    method_name: &'static str,
    qualified_name: &'static str,
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
                    "{qualified_name}() got an unexpected keyword argument '{other}'"
                )));
            }
        }
    }

    Ok((separator, maxsplit))
}

#[inline]
fn bind_replace_keyword_args(
    args: &[Value],
    keywords: &[(&str, Value)],
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
                    "str.replace() got an unexpected keyword argument '{other}'"
                )));
            }
        }
    }

    Ok((old, new, count))
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
fn parse_keepends_flag(
    keepends: Option<Value>,
    method_name: &'static str,
    position: usize,
) -> Result<bool, BuiltinError> {
    let Some(keepends) = keepends else {
        return Ok(false);
    };

    if let Some(value) = keepends.as_bool() {
        return Ok(value);
    }
    if let Some(value) = keepends.as_int() {
        return Ok(value != 0);
    }
    if keepends.is_none() {
        return Ok(false);
    }

    Err(BuiltinError::TypeError(format!(
        "str.{method_name}() argument {position} must be bool or int, not {}",
        keepends.type_name()
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

fn rsplit_with_separator(
    value: &str,
    separator: &str,
    maxsplit: Option<usize>,
) -> Result<Vec<Value>, BuiltinError> {
    if separator.is_empty() {
        return Err(BuiltinError::ValueError("empty separator".to_string()));
    }

    let limit = maxsplit.unwrap_or(usize::MAX);
    if limit == 0 {
        return Ok(vec![Value::string(intern(value))]);
    }

    let mut parts = Vec::new();
    let mut end = value.len();
    let mut splits = 0usize;
    while splits < limit {
        let Some(offset) = value[..end].rfind(separator) else {
            break;
        };
        parts.push(Value::string(intern(&value[offset + separator.len()..end])));
        end = offset;
        splits += 1;
    }

    parts.push(Value::string(intern(&value[..end])));
    parts.reverse();
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

fn rsplit_on_whitespace(value: &str, maxsplit: Option<usize>) -> Vec<Value> {
    let start = value
        .char_indices()
        .find_map(|(index, ch)| (!ch.is_whitespace()).then_some(index))
        .unwrap_or(value.len());
    let mut end = trim_end_whitespace_index(value, value.len());
    if start >= end {
        return Vec::new();
    }

    let limit = maxsplit.unwrap_or(usize::MAX);
    if limit == 0 {
        return vec![Value::string(intern(&value[start..end]))];
    }

    let mut parts = Vec::new();
    let mut splits = 0usize;
    while splits < limit {
        end = trim_end_whitespace_index(value, end);
        if end <= start {
            break;
        }

        let mut word_start = end;
        while let Some((index, ch)) = previous_char(value, word_start) {
            if ch.is_whitespace() {
                break;
            }
            word_start = index;
        }

        let remainder_end = trim_end_whitespace_index(value, word_start);
        if remainder_end <= start {
            break;
        }

        parts.push(Value::string(intern(&value[word_start..end])));
        end = remainder_end;
        splits += 1;
    }

    parts.push(Value::string(intern(&value[start..end])));
    parts.reverse();
    parts
}

#[inline]
fn previous_char(value: &str, end: usize) -> Option<(usize, char)> {
    value[..end].char_indices().next_back()
}

#[inline]
fn trim_end_whitespace_index(value: &str, mut end: usize) -> usize {
    while let Some((index, ch)) = previous_char(value, end) {
        if !ch.is_whitespace() {
            break;
        }
        end = index;
    }
    end
}

#[inline]
fn split_lines(value: &str, keepends: bool) -> Vec<Value> {
    if value.is_empty() {
        return Vec::new();
    }

    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut chars = value.char_indices().peekable();

    while let Some((idx, ch)) = chars.next() {
        let Some(boundary_end) = splitlines_boundary_end(idx, ch, chars.peek().copied()) else {
            continue;
        };

        if ch == '\r' && matches!(chars.peek(), Some((_, '\n'))) {
            chars.next();
        }

        let part_end = if keepends { boundary_end } else { idx };
        parts.push(Value::string(intern(&value[start..part_end])));
        start = boundary_end;
    }

    if start < value.len() {
        parts.push(Value::string(intern(&value[start..])));
    }

    parts
}

#[inline]
fn splitlines_boundary_end(idx: usize, ch: char, next: Option<(usize, char)>) -> Option<usize> {
    match ch {
        '\n' | '\u{000b}' | '\u{000c}' | '\u{001c}' | '\u{001d}' | '\u{001e}' | '\u{0085}'
        | '\u{2028}' | '\u{2029}' => Some(idx + ch.len_utf8()),
        '\r' => Some(match next {
            Some((next_idx, '\n')) => next_idx + '\n'.len_utf8(),
            _ => idx + ch.len_utf8(),
        }),
        _ => None,
    }
}

#[inline]
fn expand_tabs(value: &str, tabsize: i64) -> String {
    let tabsize = tabsize.max(0) as usize;
    let mut expanded = String::with_capacity(value.len());
    let mut column = 0usize;

    for ch in value.chars() {
        match ch {
            '\t' => {
                if tabsize == 0 {
                    continue;
                }

                let spaces = tabsize - (column % tabsize);
                for _ in 0..spaces {
                    expanded.push(' ');
                }
                column += spaces;
            }
            '\n' | '\r' => {
                expanded.push(ch);
                column = 0;
            }
            _ => {
                expanded.push(ch);
                column += 1;
            }
        }
    }

    expanded
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
