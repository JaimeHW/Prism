use super::*;

fn bytes_value(value: &[u8]) -> Value {
    leak_object_value(BytesObject::from_slice(value))
}

#[test]
fn test_fnmatch_module_exposes_core_api() {
    let module = FnmatchModule::new();
    assert!(module.get_attr("fnmatch").is_ok());
    assert!(module.get_attr("fnmatchcase").is_ok());
    assert!(module.get_attr("filter").is_ok());
    assert!(module.get_attr("translate").is_ok());
}

#[test]
fn test_fnmatchcase_matches_shell_patterns() {
    let result = builtin_fnmatchcase(&[
        Value::string(intern("test_alpha.py")),
        Value::string(intern("test_*.py")),
    ])
    .expect("fnmatchcase should succeed");
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_fnmatch_applies_windows_normalization() {
    let result = builtin_fnmatch(&[
        Value::string(intern("A/Path.TXT")),
        Value::string(intern("a\\*.txt")),
    ])
    .expect("fnmatch should succeed");
    assert_eq!(result.as_bool(), Some(cfg!(windows)));
}

#[test]
fn test_translate_returns_regex_wrapper() {
    let result =
        builtin_translate(&[Value::string(intern("file?.py"))]).expect("translate should work");
    assert_eq!(
        interned_by_ptr(result.as_string_object_ptr().unwrap() as *const u8)
            .unwrap()
            .as_str(),
        r"(?s:file.\.py)\Z"
    );
}

#[test]
fn test_filter_returns_matching_names() {
    let names = ListObject::from_slice(&[
        Value::string(intern("alpha.py")),
        Value::string(intern("beta.txt")),
        Value::string(intern("gamma.py")),
    ]);
    let value = builtin_filter(&[leak_object_value(names), Value::string(intern("*.py"))])
        .expect("filter should succeed");
    let ptr = value.as_object_ptr().expect("filter should return a list");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert_eq!(list.len(), 2);
}

#[test]
fn test_fnmatch_requires_match_from_start_of_name() {
    let result = builtin_fnmatch(&[
        Value::string(intern("\nfoo")),
        Value::string(intern("foo*")),
    ])
    .expect("fnmatch should succeed");
    assert_eq!(result.as_bool(), Some(false));
}

#[test]
fn test_fnmatchcase_handles_descending_ranges_without_panicking() {
    let matched = builtin_fnmatchcase(&[
        Value::string(intern("axb")),
        Value::string(intern("a[z-^]b")),
    ])
    .expect("fnmatchcase should succeed");
    assert_eq!(matched.as_bool(), Some(false));
}

#[test]
fn test_translate_matches_cpython_public_formatting() {
    let translate =
        builtin_translate(&[Value::string(intern("**a*a****a"))]).expect("translate should work");
    assert_eq!(
        interned_by_ptr(translate.as_string_object_ptr().unwrap() as *const u8)
            .unwrap()
            .as_str(),
        r"(?s:(?>.*?a)(?>.*?a).*a)\Z"
    );
}

#[test]
fn test_fnmatch_supports_bytes_and_rejects_mixed_types() {
    let matched = builtin_fnmatch(&[bytes_value(b"test\xff"), bytes_value(b"te*\xff")])
        .expect("bytes fnmatch should succeed");
    assert_eq!(matched.as_bool(), Some(true));

    let err = builtin_fnmatch(&[Value::string(intern("test")), bytes_value(b"*")])
        .expect_err("mixed string and bytes patterns should fail");
    assert!(matches!(err, BuiltinError::TypeError(_)));
}

#[test]
fn test_filter_preserves_bytes_entries() {
    let names = ListObject::from_slice(&[
        bytes_value(b"alpha.py"),
        bytes_value(b"beta.txt"),
        bytes_value(b"gamma.py"),
    ]);
    let value = builtin_filter(&[leak_object_value(names), bytes_value(b"*.py")])
        .expect("filter should succeed");
    let ptr = value.as_object_ptr().expect("filter should return a list");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert_eq!(list.len(), 2);

    let first_ptr = list.as_slice()[0]
        .as_object_ptr()
        .expect("bytes entry should be an object");
    let first = unsafe { &*(first_ptr as *const BytesObject) };
    assert_eq!(first.as_bytes(), b"alpha.py");
}
