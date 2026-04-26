use super::*;

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
        let method = resolve_bytes_method(name).unwrap_or_else(|| panic!("{name} should resolve"));
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
        .expect("bytearray.copy should return an object") as *mut BytesObject;

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

    let expected = prism_core::python_unicode::encode_python_code_point(0xDC80).expect("surrogate");
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

    let expected = prism_core::python_unicode::encode_python_code_point(0xDC80).expect("surrogate");
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
    let bytearray_upper = bytearray_upper(&[bytearray_value]).expect("bytearray.upper should work");
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
    let bytearray_chars_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"\t=")));
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

    let bytearray_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"xmutablex")));
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
    let bytearray_translated = bytearray_translate(&[bytearray_value, table_value, delete_value])
        .expect("bytearray.translate should preserve bytearray type");
    let bytearray_translated_ptr = bytearray_translated
        .as_object_ptr()
        .expect("bytearray.translate should return bytearray");
    let bytearray_translated_bytes = unsafe { &*(bytearray_translated_ptr as *const BytesObject) };
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

    let wrong_type = bytes_find_method(&[bytes_value, Value::string(intern("ba"))]).unwrap_err();
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
    let err = list_extend(&[list_value, Value::int(42).unwrap()]).expect_err("extend should fail");
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
    let getitem = resolve_regex_match_method("__getitem__").expect("__getitem__ should resolve");
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
            .expect("__get__ should be allocated") as *const BuiltinFunctionObject)
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
