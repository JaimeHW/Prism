use super::*;
use crate::VirtualMachine;
use crate::builtins::get_iterator_mut;
use prism_core::intern::intern;

fn string_value(value: Value) -> String {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .expect("tagged string should provide a pointer");
        return prism_core::intern::interned_by_ptr(ptr as *const u8)
            .expect("tagged string pointer should resolve")
            .as_str()
            .to_string();
    }

    let ptr = value.as_object_ptr().expect("expected string object");
    unsafe { &*(ptr as *const StringObject) }
        .as_str()
        .to_string()
}

fn bytes_value(value: Value) -> Vec<u8> {
    let ptr = value.as_object_ptr().expect("expected bytes object");
    unsafe { &*(ptr as *const BytesObject) }.as_bytes().to_vec()
}

fn bytes_object_value(bytes: &[u8]) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(BytesObject::from_slice(bytes))) as *const ())
}

fn tuple_values(value: Value) -> Vec<Value> {
    let ptr = value.as_object_ptr().expect("expected tuple object");
    unsafe { &*(ptr as *const TupleObject) }.as_slice().to_vec()
}

fn list_values(value: Value) -> Vec<Value> {
    let ptr = value.as_object_ptr().expect("expected list object");
    unsafe { &*(ptr as *const ListObject) }.as_slice().to_vec()
}

fn dict_entries(value: Value) -> Vec<(Value, Value)> {
    let ptr = value.as_object_ptr().expect("expected dict object");
    unsafe { &*(ptr as *const DictObject) }.iter().collect()
}

fn pattern_flags(vm: &mut VirtualMachine, pattern: Value) -> i64 {
    pattern_attr_value(vm, pattern, &intern("flags"))
        .expect("pattern attribute lookup should succeed")
        .expect("flags attribute should exist")
        .as_int()
        .expect("flags should be returned as a small int")
}

fn exhaust_nursery(vm: &VirtualMachine) {
    while vm.allocator().alloc(DictObject::new()).is_some() {}
}

#[test]
fn test_pattern_search_allocates_match_after_full_nursery() {
    let mut vm = VirtualMachine::new();
    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))]).expect("compile should succeed");

    exhaust_nursery(&vm);

    let searched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc123def"))])
        .expect("search should allocate a match after nursery exhaustion");
    assert!(!searched.is_none());

    let group = builtin_match_group(&mut vm, &[searched]).expect("group should allocate");
    assert_eq!(string_value(group), "123");
}

#[test]
fn test_compile_string_pattern_and_match_group() {
    let mut vm = VirtualMachine::new();
    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))]).expect("compile should succeed");
    let matched = builtin_pattern_match(&mut vm, &[pattern, Value::string(intern("abc123def"))])
        .expect("pattern.match should succeed");
    assert!(matched.is_none(), "match() should anchor at the start");

    let searched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc123def"))])
        .expect("pattern.search should succeed");
    let group = builtin_match_group(&mut vm, &[searched]).expect("group() should succeed");
    assert_eq!(string_value(group), "123");
}

#[test]
fn test_compile_string_pattern_exposes_default_unicode_flag() {
    let mut vm = VirtualMachine::new();
    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\w+"))]).expect("compile should succeed");

    assert_eq!(pattern_flags(&mut vm, pattern), RegexFlags::UNICODE as i64);
}

#[test]
fn test_compile_string_pattern_ascii_flag_suppresses_default_unicode() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[
            Value::string(intern(r"\w+")),
            Value::int(RegexFlags::ASCII as i64).unwrap(),
        ],
    )
    .expect("compile should succeed");

    assert_eq!(pattern_flags(&mut vm, pattern), RegexFlags::ASCII as i64);
}

#[test]
fn test_compile_bytes_pattern_does_not_add_unicode_flag() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(&mut vm, &[bytes_object_value(br"\w+")])
        .expect("bytes compile should succeed");

    assert_eq!(pattern_flags(&mut vm, pattern), 0);
}

#[test]
fn test_parse_flags_accepts_int_backed_heap_values() {
    let raw = RegexFlags::IGNORECASE | RegexFlags::MULTILINE;
    let object = prism_runtime::object::shaped_object::ShapedObject::new_int_backed(
        TypeId::from_raw(TypeId::FIRST_USER_TYPE),
        prism_runtime::object::shape::Shape::empty(),
        num_bigint::BigInt::from(raw),
    );
    let ptr = Box::into_raw(Box::new(object));
    let value = Value::object_ptr(ptr as *const ());

    let flags = parse_flags(Some(value)).expect("int-backed flags should parse");
    unsafe { drop(Box::from_raw(ptr)) };

    assert_eq!(flags.bits(), raw);
}

#[test]
fn test_parse_flags_rejects_values_larger_than_sre_flags() {
    let value = prism_runtime::types::int::bigint_to_value(num_bigint::BigInt::from(
        u64::from(u32::MAX) + 1,
    ));

    assert!(matches!(
        parse_flags(Some(value)),
        Err(BuiltinError::OverflowError(_))
    ));
}

#[test]
fn test_compile_bytes_pattern_and_match_group() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::object_ptr(
            Box::into_raw(Box::new(BytesObject::from_slice(br"\w+"))) as *const (),
        )],
    )
    .expect("bytes compile should succeed");
    let searched = builtin_pattern_search(
        &mut vm,
        &[
            pattern,
            Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(b"abc123"))) as *const ()
            ),
        ],
    )
    .expect("bytes search should succeed");
    let group = builtin_match_group(&mut vm, &[searched]).expect("group() should succeed");
    assert_eq!(bytes_value(group), b"abc123");
}

#[test]
fn test_compile_bytes_pattern_uses_fancy_engine_for_lookahead() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[bytes_object_value(br"\n(?![ \t])|\r(?![ \t\n])")],
    )
    .expect("bytes lookahead pattern should compile");

    let legal = builtin_pattern_search(&mut vm, &[pattern, bytes_object_value(b"ok\n value")])
        .expect("bytes search should run");
    assert!(legal.is_none(), "continuation whitespace should not match");

    let pattern = builtin_compile(
        &mut vm,
        &[bytes_object_value(br"\n(?![ \t])|\r(?![ \t\n])")],
    )
    .expect("bytes lookahead pattern should compile");
    let illegal = builtin_pattern_search(&mut vm, &[pattern, bytes_object_value(b"bad\nHeader")])
        .expect("bytes search should run");
    let group = builtin_match_group(&mut vm, &[illegal]).expect("group() should succeed");
    assert_eq!(bytes_value(group), b"\n");
}

#[test]
fn test_escape_accepts_bytes() {
    let mut vm = VirtualMachine::new();
    let escaped = builtin_escape(
        &mut vm,
        &[Value::object_ptr(
            Box::into_raw(Box::new(BytesObject::from_slice(b"a+b"))) as *const (),
        )],
    )
    .expect("escape should succeed");
    assert_eq!(bytes_value(escaped), br"a\+b");
}

#[test]
fn test_match_groups_returns_capture_tuple() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"(\d+)-(\w+)"))])
        .expect("compile should succeed");
    let matched = builtin_pattern_search(
        &mut vm,
        &[pattern, Value::string(intern("prefix 123-word suffix"))],
    )
    .expect("search should succeed");

    let groups = builtin_match_groups(&mut vm, &[matched]).expect("groups() should succeed");
    let values = tuple_values(groups);
    assert_eq!(values.len(), 2);
    assert_eq!(string_value(values[0]), "123");
    assert_eq!(string_value(values[1]), "word");
}

#[test]
fn test_match_groups_uses_default_for_unmatched_captures() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"(a)?(b)"))])
        .expect("compile should succeed");
    let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("b"))])
        .expect("search should succeed");

    let groups = builtin_match_groups(&mut vm, &[matched, Value::string(intern("<missing>"))])
        .expect("groups() should succeed");
    let values = tuple_values(groups);
    assert_eq!(values.len(), 2);
    assert_eq!(string_value(values[0]), "<missing>");
    assert_eq!(string_value(values[1]), "b");
}

#[test]
fn test_match_groupdict_returns_named_groups() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::string(intern(r"(?P<lhs>\w+)=(?P<rhs>\w+)"))],
    )
    .expect("compile should succeed");
    let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("alpha=beta"))])
        .expect("search should succeed");

    let groupdict =
        builtin_match_groupdict(&mut vm, &[matched]).expect("groupdict() should succeed");
    let entries = dict_entries(groupdict);
    assert_eq!(entries.len(), 2);

    let lhs = entries
        .iter()
        .find(|(key, _)| {
            prism_runtime::types::string::value_as_string_ref(*key)
                .is_some_and(|key| key.as_str() == "lhs")
        })
        .map(|(_, value)| *value)
        .expect("lhs entry should exist");
    let rhs = entries
        .iter()
        .find(|(key, _)| {
            prism_runtime::types::string::value_as_string_ref(*key)
                .is_some_and(|key| key.as_str() == "rhs")
        })
        .map(|(_, value)| *value)
        .expect("rhs entry should exist");

    assert_eq!(string_value(lhs), "alpha");
    assert_eq!(string_value(rhs), "beta");
}

#[test]
fn test_match_group_accepts_named_and_multiple_selectors() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::string(intern(r"(?P<lhs>\w+)=(?P<rhs>\w+)"))],
    )
    .expect("compile should succeed");
    let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("alpha=beta"))])
        .expect("search should succeed");

    let lhs = builtin_match_group(&mut vm, &[matched, Value::string(intern("lhs"))])
        .expect("named group lookup should succeed");
    assert_eq!(string_value(lhs), "alpha");

    let values = builtin_match_group(
        &mut vm,
        &[
            matched,
            Value::string(intern("lhs")),
            Value::int(2).expect("group index should fit"),
        ],
    )
    .expect("mixed group lookup should succeed");
    let values = tuple_values(values);
    assert_eq!(values.len(), 2);
    assert_eq!(string_value(values[0]), "alpha");
    assert_eq!(string_value(values[1]), "beta");
}

#[test]
fn test_match_positions_accept_named_groups() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::string(intern(r"(?P<indent>\s+)(?P<name>\w+)"))],
    )
    .expect("compile should succeed");
    let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("   prism"))])
        .expect("search should succeed");

    let start = builtin_match_start(&[matched, Value::string(intern("name"))])
        .expect("named start should succeed");
    let end = builtin_match_end(&[matched, Value::string(intern("name"))])
        .expect("named end should succeed");
    let span = builtin_match_span(&mut vm, &[matched, Value::string(intern("name"))])
        .expect("named span should succeed");

    assert_eq!(start.as_int(), Some(3));
    assert_eq!(end.as_int(), Some(8));
    let span = tuple_values(span);
    assert_eq!(span[0].as_int(), Some(3));
    assert_eq!(span[1].as_int(), Some(8));
}

#[test]
fn test_pattern_groupindex_exposes_named_capture_mapping() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::string(intern(r"(?P<first>a)(?P<second>b)"))],
    )
    .expect("compile should succeed");

    let groupindex = pattern_attr_value(&mut vm, pattern, &intern("groupindex"))
        .expect("groupindex lookup should succeed")
        .expect("groupindex attribute should exist");
    let entries = dict_entries(groupindex);
    assert_eq!(entries.len(), 2);

    let first = entries
        .iter()
        .find(|(key, _)| string_value(*key) == "first")
        .map(|(_, value)| *value)
        .expect("first group should exist");
    let second = entries
        .iter()
        .find(|(key, _)| string_value(*key) == "second")
        .map(|(_, value)| *value)
        .expect("second group should exist");

    assert_eq!(first.as_int(), Some(1));
    assert_eq!(second.as_int(), Some(2));
}

#[test]
fn test_match_lastindex_and_lastgroup_attributes_follow_cpython() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::string(intern(r"(?P<word>\w+)(?:-(\d+))?"))],
    )
    .expect("compile should succeed");

    let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc"))])
        .expect("search should succeed");

    let lastindex = match_attr_value(&mut vm, matched, &intern("lastindex"))
        .expect("lastindex lookup should succeed")
        .expect("lastindex attribute should exist");
    let lastgroup = match_attr_value(&mut vm, matched, &intern("lastgroup"))
        .expect("lastgroup lookup should succeed")
        .expect("lastgroup attribute should exist");

    assert_eq!(lastindex.as_int(), Some(1));
    assert_eq!(string_value(lastgroup), "word");

    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\w+"))]).expect("compile should succeed");
    let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc"))])
        .expect("search should succeed");

    let lastindex = match_attr_value(&mut vm, matched, &intern("lastindex"))
        .expect("lastindex lookup should succeed")
        .expect("lastindex attribute should exist");
    let lastgroup = match_attr_value(&mut vm, matched, &intern("lastgroup"))
        .expect("lastgroup lookup should succeed")
        .expect("lastgroup attribute should exist");

    assert!(lastindex.is_none());
    assert!(lastgroup.is_none());
}

#[test]
fn test_match_getitem_indexes_capture_groups_like_group() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::string(intern(r"(?P<word>\w+)-(?P<num>\d+)"))],
    )
    .expect("compile should succeed");
    let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc-123"))])
        .expect("search should succeed");

    let whole = builtin_match_getitem(
        &mut vm,
        &[
            matched,
            Value::int(0).expect("index should fit in Value::int"),
        ],
    )
    .expect("getitem should return whole match");
    let first = builtin_match_getitem(
        &mut vm,
        &[
            matched,
            Value::int(1).expect("index should fit in Value::int"),
        ],
    )
    .expect("getitem should return numeric group");
    let named = builtin_match_getitem(&mut vm, &[matched, Value::string(intern("num"))])
        .expect("getitem should return named group");

    assert_eq!(string_value(whole), "abc-123");
    assert_eq!(string_value(first), "abc");
    assert_eq!(string_value(named), "123");

    let pattern = builtin_compile(
        &mut vm,
        &[Value::object_ptr(
            Box::into_raw(Box::new(BytesObject::from_slice(
                br"(?P<word>\w+)-(?P<num>\d+)",
            ))) as *const (),
        )],
    )
    .expect("bytes compile should succeed");
    let matched = builtin_pattern_search(
        &mut vm,
        &[
            pattern,
            Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(b"abc-123"))) as *const ()
            ),
        ],
    )
    .expect("bytes search should succeed");

    let whole = builtin_match_getitem(
        &mut vm,
        &[
            matched,
            Value::int(0).expect("index should fit in Value::int"),
        ],
    )
    .expect("bytes getitem should return whole match");
    let named = builtin_match_getitem(&mut vm, &[matched, Value::string(intern("num"))])
        .expect("bytes getitem should return named group");

    assert_eq!(bytes_value(whole), b"abc-123");
    assert_eq!(bytes_value(named), b"123");
}

#[test]
fn test_pattern_match_accepts_pos_and_rebases_span() {
    let mut vm = VirtualMachine::new();
    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))]).expect("compile should succeed");

    let matched = builtin_pattern_match(
        &mut vm,
        &[
            pattern,
            Value::string(intern("abc123def")),
            Value::int(3).expect("pos should fit in Value::int"),
        ],
    )
    .expect("pattern.match should accept pos");

    let group = builtin_match_group(&mut vm, &[matched]).expect("group() should succeed");
    assert_eq!(string_value(group), "123");
    assert_eq!(
        builtin_match_start(&[matched])
            .expect("start() should succeed")
            .as_int(),
        Some(3)
    );
    assert_eq!(
        builtin_match_end(&[matched])
            .expect("end() should succeed")
            .as_int(),
        Some(6)
    );
}

#[test]
fn test_pattern_fullmatch_accepts_pos_and_endpos() {
    let mut vm = VirtualMachine::new();
    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))]).expect("compile should succeed");

    let matched = builtin_pattern_fullmatch(
        &mut vm,
        &[
            pattern,
            Value::string(intern("abc123def")),
            Value::int(3).expect("pos should fit in Value::int"),
            Value::int(6).expect("endpos should fit in Value::int"),
        ],
    )
    .expect("pattern.fullmatch should accept bounds");

    let group = builtin_match_group(&mut vm, &[matched]).expect("group() should succeed");
    assert_eq!(string_value(group), "123");
    let span = builtin_match_span(&mut vm, &[matched]).expect("span() should succeed");
    let span = tuple_values(span);
    assert_eq!(span[0].as_int(), Some(3));
    assert_eq!(span[1].as_int(), Some(6));
}

#[test]
fn test_bytes_match_named_groups_support_group_and_groupdict() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::object_ptr(
            Box::into_raw(Box::new(BytesObject::from_slice(
                br"(?P<lhs>\w+)=(?P<rhs>\w+)",
            ))) as *const (),
        )],
    )
    .expect("bytes compile should succeed");
    let matched = builtin_pattern_search(
        &mut vm,
        &[
            pattern,
            Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(b"left=right"))) as *const (),
            ),
        ],
    )
    .expect("bytes search should succeed");

    let lhs = builtin_match_group(&mut vm, &[matched, Value::string(intern("lhs"))])
        .expect("bytes named group lookup should succeed");
    assert_eq!(bytes_value(lhs), b"left");

    let groupdict =
        builtin_match_groupdict(&mut vm, &[matched]).expect("bytes groupdict should succeed");
    let entries = dict_entries(groupdict);
    assert_eq!(entries.len(), 2);

    let lhs = entries
        .iter()
        .find(|(key, _)| string_value(*key) == "lhs")
        .map(|(_, value)| *value)
        .expect("lhs entry should exist");
    let rhs = entries
        .iter()
        .find(|(key, _)| string_value(*key) == "rhs")
        .map(|(_, value)| *value)
        .expect("rhs entry should exist");
    assert_eq!(bytes_value(lhs), b"left");
    assert_eq!(bytes_value(rhs), b"right");

    let lastindex = match_attr_value(&mut vm, matched, &intern("lastindex"))
        .expect("lastindex lookup should succeed")
        .expect("lastindex attribute should exist");
    let lastgroup = match_attr_value(&mut vm, matched, &intern("lastgroup"))
        .expect("lastgroup lookup should succeed")
        .expect("lastgroup attribute should exist");

    assert_eq!(lastindex.as_int(), Some(2));
    assert_eq!(string_value(lastgroup), "rhs");
}

#[test]
fn test_bytes_pattern_search_accepts_pos_and_rebases_span() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::object_ptr(
            Box::into_raw(Box::new(BytesObject::from_slice(br"\d+"))) as *const (),
        )],
    )
    .expect("bytes compile should succeed");

    let matched = builtin_pattern_search(
        &mut vm,
        &[
            pattern,
            Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(b"abc123def456"))) as *const (),
            ),
            Value::int(6).expect("pos should fit in Value::int"),
        ],
    )
    .expect("bytes search should accept pos");

    let group = builtin_match_group(&mut vm, &[matched]).expect("group() should succeed");
    assert_eq!(bytes_value(group), b"456");
    assert_eq!(
        builtin_match_start(&[matched])
            .expect("start() should succeed")
            .as_int(),
        Some(9)
    );
    assert_eq!(
        builtin_match_end(&[matched])
            .expect("end() should succeed")
            .as_int(),
        Some(12)
    );
}

#[test]
fn test_pattern_findall_returns_full_matches_without_groups() {
    let mut vm = VirtualMachine::new();
    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))]).expect("compile should succeed");

    let matches = builtin_pattern_findall(&mut vm, &[pattern, Value::string(intern("a1b22"))])
        .expect("findall should succeed");
    let values = list_values(matches);
    assert_eq!(values.len(), 2);
    assert_eq!(string_value(values[0]), "1");
    assert_eq!(string_value(values[1]), "22");
}

#[test]
fn test_pattern_findall_returns_capture_values_for_single_group() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"([a-z]+)?=(\d+)"))])
        .expect("compile should succeed");
    let matches =
        builtin_pattern_findall(&mut vm, &[pattern, Value::string(intern("foo=1 =2 bar=3"))])
            .expect("findall should succeed");

    let values = list_values(matches);
    assert_eq!(values.len(), 3);

    let first = tuple_values(values[0]);
    let second = tuple_values(values[1]);
    let third = tuple_values(values[2]);
    assert_eq!(
        first.into_iter().map(string_value).collect::<Vec<_>>(),
        vec!["foo".to_string(), "1".to_string()]
    );
    assert_eq!(
        second.into_iter().map(string_value).collect::<Vec<_>>(),
        vec!["".to_string(), "2".to_string()]
    );
    assert_eq!(
        third.into_iter().map(string_value).collect::<Vec<_>>(),
        vec!["bar".to_string(), "3".to_string()]
    );
}

#[test]
fn test_pattern_findall_returns_plain_strings_for_one_group() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"(\d+)"))])
        .expect("compile should succeed");

    let matches =
        builtin_pattern_findall(&mut vm, &[pattern, Value::string(intern("x7 y88 z999"))])
            .expect("findall should succeed");
    let values = list_values(matches);
    assert_eq!(values.len(), 3);
    assert_eq!(string_value(values[0]), "7");
    assert_eq!(string_value(values[1]), "88");
    assert_eq!(string_value(values[2]), "999");
}

#[test]
fn test_pattern_findall_returns_bytes_for_bytes_patterns() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(
        &mut vm,
        &[Value::object_ptr(
            Box::into_raw(Box::new(BytesObject::from_slice(br"(\w+)-(\d+)"))) as *const (),
        )],
    )
    .expect("bytes compile should succeed");

    let matches = builtin_pattern_findall(
        &mut vm,
        &[
            pattern,
            Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(b"abc-1 def-22"))) as *const (),
            ),
        ],
    )
    .expect("bytes findall should succeed");
    let values = list_values(matches);
    assert_eq!(values.len(), 2);

    let first = tuple_values(values[0]);
    let second = tuple_values(values[1]);
    assert_eq!(bytes_value(first[0]), b"abc");
    assert_eq!(bytes_value(first[1]), b"1");
    assert_eq!(bytes_value(second[0]), b"def");
    assert_eq!(bytes_value(second[1]), b"22");
}

#[test]
fn test_pattern_findall_accepts_pos_and_endpos() {
    let mut vm = VirtualMachine::new();
    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))]).expect("compile should succeed");

    let matches = builtin_pattern_findall(
        &mut vm,
        &[
            pattern,
            Value::string(intern("a1 b22 c333")),
            Value::int(3).expect("pos should fit in Value::int"),
            Value::int(7).expect("endpos should fit in Value::int"),
        ],
    )
    .expect("findall should accept bounds");
    let values = list_values(matches);
    assert_eq!(values.len(), 1);
    assert_eq!(string_value(values[0]), "22");
}

#[test]
fn test_module_findall_entrypoint_honors_flags() {
    let mut vm = VirtualMachine::new();
    let matches = builtin_findall(
        &mut vm,
        &[
            Value::string(intern(r"^hello")),
            Value::string(intern("Hello\nhello")),
            Value::int(RegexFlags::IGNORECASE as i64).unwrap(),
        ],
    )
    .expect("module findall should succeed");
    let values = list_values(matches);
    assert_eq!(values.len(), 1);
    assert_eq!(string_value(values[0]), "Hello");
}

#[test]
fn test_pattern_finditer_returns_iterator_of_match_objects() {
    let mut vm = VirtualMachine::new();
    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))]).expect("compile should succeed");

    let iterator =
        builtin_pattern_finditer(&mut vm, &[pattern, Value::string(intern("a1 b22 c333"))])
            .expect("finditer should succeed");
    let iter = get_iterator_mut(&iterator).expect("finditer should return an iterator");

    let first = builtin_match_group(&mut vm, &[iter.next().expect("first match should exist")])
        .expect("group should work");
    let second = builtin_match_group(&mut vm, &[iter.next().expect("second match should exist")])
        .expect("group should work");
    let third = builtin_match_group(&mut vm, &[iter.next().expect("third match should exist")])
        .expect("group should work");
    assert_eq!(string_value(first), "1");
    assert_eq!(string_value(second), "22");
    assert_eq!(string_value(third), "333");
    assert!(iter.next().is_none());
}

#[test]
fn test_pattern_finditer_accepts_pos_and_rebases_spans() {
    let mut vm = VirtualMachine::new();
    let pattern =
        builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))]).expect("compile should succeed");

    let iterator = builtin_pattern_finditer(
        &mut vm,
        &[
            pattern,
            Value::string(intern("a1 b22 c333")),
            Value::int(3).expect("pos should fit in Value::int"),
        ],
    )
    .expect("finditer should accept pos");
    let iter = get_iterator_mut(&iterator).expect("finditer should return an iterator");

    let first = iter.next().expect("first match should exist");
    let second = iter.next().expect("second match should exist");
    assert!(iter.next().is_none());

    let first_group = builtin_match_group(&mut vm, &[first]).expect("group() should work");
    let second_group = builtin_match_group(&mut vm, &[second]).expect("group() should work");
    assert_eq!(string_value(first_group), "22");
    assert_eq!(string_value(second_group), "333");
    assert_eq!(
        builtin_match_start(&[first])
            .expect("start() should succeed")
            .as_int(),
        Some(4)
    );
    assert_eq!(
        builtin_match_start(&[second])
            .expect("start() should succeed")
            .as_int(),
        Some(8)
    );
}

#[test]
fn test_module_finditer_entrypoint_returns_iterable_matches() {
    let mut vm = VirtualMachine::new();
    let iterator = builtin_finditer(
        &mut vm,
        &[
            Value::string(intern(r"[a-z]+")),
            Value::string(intern("ab  cd")),
        ],
    )
    .expect("module finditer should succeed");
    let iter = get_iterator_mut(&iterator).expect("finditer should return an iterator");

    let first = builtin_match_group(&mut vm, &[iter.next().expect("first match should exist")])
        .expect("group should work");
    let second = builtin_match_group(&mut vm, &[iter.next().expect("second match should exist")])
        .expect("group should work");
    assert_eq!(string_value(first), "ab");
    assert_eq!(string_value(second), "cd");
    assert!(iter.next().is_none());
}

#[test]
fn test_module_sub_entrypoint_replaces_matches() {
    let mut vm = VirtualMachine::new();
    let result = builtin_sub(
        &mut vm,
        &[
            Value::string(intern(r"\d+")),
            Value::string(intern("X")),
            Value::string(intern("a1b22c333")),
        ],
    )
    .expect("module sub should succeed");
    assert_eq!(string_value(result), "aXbXcX");
}

#[test]
fn test_module_sub_expands_python_numeric_backreferences() {
    let mut vm = VirtualMachine::new();
    let result = builtin_sub(
        &mut vm,
        &[
            Value::string(intern(r#"\\([\\\$"'`])"#)),
            Value::string(intern(r"\1")),
            Value::string(intern(r#"\$\`\\\'\""#)),
        ],
    )
    .expect("module sub should expand Python replacement backrefs");
    assert_eq!(string_value(result), "$`\\'\"");
}

#[test]
fn test_module_sub_escapes_literal_dollars_for_rust_regex_backend() {
    let mut vm = VirtualMachine::new();
    let result = builtin_sub(
        &mut vm,
        &[
            Value::string(intern(r"(a)(b)")),
            Value::string(intern(r"\2-\1-$")),
            Value::string(intern("ab")),
        ],
    )
    .expect("module sub should preserve literal dollars");
    assert_eq!(string_value(result), "b-a-$");
}

#[test]
fn test_module_subn_entrypoint_returns_result_and_count() {
    let mut vm = VirtualMachine::new();
    let result = builtin_subn(
        &mut vm,
        &[
            Value::string(intern(r"\d+")),
            Value::string(intern("X")),
            Value::string(intern("a1b22c333")),
            Value::int(2).expect("count should fit"),
        ],
    )
    .expect("module subn should succeed");
    let values = tuple_values(result);
    assert_eq!(values.len(), 2);
    assert_eq!(string_value(values[0]), "aXbXc333");
    assert_eq!(values[1].as_int(), Some(2));
}

#[test]
fn test_module_split_entrypoint_returns_parts() {
    let mut vm = VirtualMachine::new();
    let result = builtin_split(
        &mut vm,
        &[
            Value::string(intern(r",\s*")),
            Value::string(intern("a, b,  c")),
        ],
    )
    .expect("module split should succeed");
    let values = list_values(result);
    assert_eq!(values.len(), 3);
    assert_eq!(string_value(values[0]), "a");
    assert_eq!(string_value(values[1]), "b");
    assert_eq!(string_value(values[2]), "c");
}

#[test]
fn test_pattern_sub_entrypoint_returns_replaced_text() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
        .expect("pattern compile should succeed");
    let result = builtin_pattern_sub(
        &mut vm,
        &[
            pattern,
            Value::string(intern("X")),
            Value::string(intern("a1b22c333")),
        ],
    )
    .expect("pattern sub should succeed");
    assert_eq!(string_value(result), "aXbXcX");
}

#[test]
fn test_pattern_subn_entrypoint_returns_result_and_count() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
        .expect("pattern compile should succeed");
    let result = builtin_pattern_subn(
        &mut vm,
        &[
            pattern,
            Value::string(intern("X")),
            Value::string(intern("a1b22c333")),
            Value::int(2).expect("count should fit"),
        ],
    )
    .expect("pattern subn should succeed");
    let values = tuple_values(result);
    assert_eq!(values.len(), 2);
    assert_eq!(string_value(values[0]), "aXbXc333");
    assert_eq!(values[1].as_int(), Some(2));
}

#[test]
fn test_pattern_split_entrypoint_returns_parts() {
    let mut vm = VirtualMachine::new();
    let pattern = builtin_compile(&mut vm, &[Value::string(intern(r",\s*"))])
        .expect("pattern compile should succeed");
    let result = builtin_pattern_split(&mut vm, &[pattern, Value::string(intern("a, b,  c"))])
        .expect("pattern split should succeed");
    let values = list_values(result);
    assert_eq!(values.len(), 3);
    assert_eq!(string_value(values[0]), "a");
    assert_eq!(string_value(values[1]), "b");
    assert_eq!(string_value(values[2]), "c");
}

#[test]
fn test_module_sub_entrypoint_supports_bytes_patterns() {
    let mut vm = VirtualMachine::new();
    let result = builtin_sub(
        &mut vm,
        &[
            Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(br"\d+"))) as *const ()
            ),
            Value::object_ptr(Box::into_raw(Box::new(BytesObject::from_slice(b"X"))) as *const ()),
            Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(b"a1b22c333"))) as *const (),
            ),
        ],
    )
    .expect("module bytes sub should succeed");
    assert_eq!(bytes_value(result), b"aXbXcX");
}
