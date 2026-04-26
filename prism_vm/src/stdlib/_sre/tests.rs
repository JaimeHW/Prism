use super::*;
use crate::stdlib::re::{
    RegexFlags, builtin_match_group, builtin_pattern_match, pattern_attr_value,
};
use prism_core::intern::interned_by_ptr;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;

fn list_value(values: &[Value]) -> Value {
    Value::object_ptr(
        Box::into_raw(Box::new(ListObject::from_iter(values.iter().copied()))) as *const (),
    )
}

fn dict_value() -> Value {
    Value::object_ptr(Box::into_raw(Box::new(DictObject::new())) as *const ())
}

fn tuple_value(values: &[Value]) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(TupleObject::from_vec(values.to_vec()))) as *const ())
}

fn python_string_value(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        return interned_by_ptr(ptr as *const u8).map(|text| text.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::STR {
        return None;
    }

    Some(
        unsafe { &*(ptr as *const prism_runtime::types::string::StringObject) }
            .as_str()
            .to_string(),
    )
}

#[test]
fn test_sre_module_exposes_expected_surface() {
    let module = SreModule::new();
    assert_eq!(module.name(), "_sre");
    assert_eq!(module.get_attr("MAGIC").unwrap().as_int(), Some(MAGIC));
    assert_eq!(
        module.get_attr("CODESIZE").unwrap().as_int(),
        Some(CODESIZE)
    );
    assert_eq!(
        module.get_attr("MAXREPEAT").unwrap().as_int(),
        Some(MAXREPEAT)
    );
    assert_eq!(
        module.get_attr("MAXGROUPS").unwrap().as_int(),
        Some(MAXGROUPS)
    );
    for name in [
        "compile",
        "template",
        "getcodesize",
        "ascii_iscased",
        "unicode_iscased",
        "ascii_tolower",
        "unicode_tolower",
    ] {
        assert!(module.get_attr(name).unwrap().as_object_ptr().is_some());
    }
}

#[test]
fn test_sre_compile_bridge_returns_native_pattern_and_honors_cpython_flags() {
    let mut vm = VirtualMachine::new();
    let pattern = sre_compile(
        &mut vm,
        &[
            Value::string(intern("hello")),
            Value::int(RegexFlags::IGNORECASE as i64).unwrap(),
            list_value(&[]),
            Value::int(0).unwrap(),
            dict_value(),
            tuple_value(&[]),
        ],
    )
    .expect("_sre.compile should succeed");

    let ptr = pattern
        .as_object_ptr()
        .expect("_sre.compile should return an object");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::REGEX_PATTERN);
    assert_eq!(
        pattern_attr_value(&mut vm, pattern, &intern("flags"))
            .expect("pattern attribute lookup should succeed")
            .expect("flags attribute should exist")
            .as_int(),
        Some((RegexFlags::IGNORECASE | RegexFlags::UNICODE) as i64)
    );

    let matched = builtin_pattern_match(&mut vm, &[pattern, Value::string(intern("HELLO"))])
        .expect("compiled pattern should support match()");
    let group = builtin_match_group(&mut vm, &[matched]).expect("group() should succeed");
    assert_eq!(python_string_value(group).as_deref(), Some("HELLO"));
}

#[test]
fn test_sre_template_requires_list_and_returns_original_template() {
    let template = list_value(&[
        Value::string(intern("prefix")),
        Value::int(1).unwrap(),
        Value::string(intern("suffix")),
    ]);

    assert_eq!(
        sre_template(&[Value::none(), template]).expect("template() should accept lists"),
        template
    );
    assert!(matches!(
        sre_template(&[Value::none(), Value::int(1).unwrap()]),
        Err(BuiltinError::TypeError(_))
    ));
}

#[test]
fn test_sre_case_helpers_follow_cpython_contract() {
    assert_eq!(
        sre_ascii_iscased(&[Value::int('A' as i64).unwrap()])
            .unwrap()
            .as_bool(),
        Some(true)
    );
    assert_eq!(
        sre_ascii_iscased(&[Value::int('1' as i64).unwrap()])
            .unwrap()
            .as_bool(),
        Some(false)
    );
    assert_eq!(
        sre_unicode_iscased(&[Value::int(0x00C4).unwrap()])
            .unwrap()
            .as_bool(),
        Some(true)
    );
    assert_eq!(
        sre_ascii_tolower(&[Value::int('Z' as i64).unwrap()])
            .unwrap()
            .as_int(),
        Some('z' as i64)
    );
    assert_eq!(
        sre_unicode_tolower(&[Value::int(0x00C4).unwrap()])
            .unwrap()
            .as_int(),
        Some(0x00E4)
    );
}
