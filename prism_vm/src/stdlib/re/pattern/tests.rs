use super::*;

#[test]
fn test_compile() {
    let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
    assert_eq!(pattern.pattern(), r"\d+");
    assert_eq!(pattern.engine_kind(), EngineKind::Standard);
}

#[test]
fn test_compile_with_flags() {
    let flags = RegexFlags::new(RegexFlags::IGNORECASE);
    let pattern = CompiledPattern::compile(r"hello", flags).unwrap();
    assert!(pattern.is_match("HELLO"));
}

#[test]
fn test_match() {
    let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
    assert!(pattern.match_("123abc").is_some());
    assert!(pattern.match_("abc123").is_none());
}

#[test]
fn test_search() {
    let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
    let m = pattern.search("abc123def").unwrap();
    assert_eq!(m.as_str(), "123");
    assert_eq!(m.start(), 3);
}

#[test]
fn test_fullmatch() {
    let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
    assert!(pattern.fullmatch("123").is_some());
    assert!(pattern.fullmatch("123abc").is_none());
}

#[test]
fn test_findall() {
    let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
    let matches = pattern.findall("a1b22c333");
    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].as_str(), "1");
    assert_eq!(matches[1].as_str(), "22");
    assert_eq!(matches[2].as_str(), "333");
}

#[test]
fn test_findall_with_groups() {
    let pattern = CompiledPattern::compile_default(r"(\d+)-(\d+)").unwrap();
    let strings = pattern.findall_strings("1-2 3-4 5-6");
    assert_eq!(strings.len(), 3);
    assert_eq!(strings[0], vec!["1", "2"]);
}

#[test]
fn test_sub() {
    let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
    assert_eq!(pattern.sub("X", "a1b2c3"), "aXb2c3");
}

#[test]
fn test_sub_n() {
    let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
    assert_eq!(pattern.sub_n("X", "a1b2c3", 0), "aXbXcX");
    assert_eq!(pattern.sub_n("X", "a1b2c3", 2), "aXbXc3");
}

#[test]
fn test_split() {
    let pattern = CompiledPattern::compile_default(r",\s*").unwrap();
    let parts = pattern.split("a, b,  c");
    assert_eq!(parts, vec!["a", "b", "c"]);
}

#[test]
fn test_split_n() {
    let pattern = CompiledPattern::compile_default(r",").unwrap();
    let parts = pattern.split_n("a,b,c,d", 2);
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[2], "c,d");
}

#[test]
fn test_display() {
    let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
    let s = pattern.to_string();
    assert!(s.contains("re.compile"));
    assert!(s.contains(r"\d+"));
}

#[test]
fn test_clone() {
    let pattern = CompiledPattern::compile_default(r"\d+").unwrap();
    let cloned = pattern.clone();
    assert_eq!(cloned.pattern(), pattern.pattern());
}

#[test]
fn test_groups_count() {
    let pattern = CompiledPattern::compile_default(r"(\d+)-(\d+)").unwrap();
    assert_eq!(pattern.groups(), 3); // Full match + 2 groups
}

#[test]
fn test_backreference_pattern() {
    let pattern = CompiledPattern::compile_default(r"(.)\1").unwrap();
    assert_eq!(pattern.engine_kind(), EngineKind::Fancy);
    assert!(pattern.is_match("aa"));
    assert!(!pattern.is_match("ab"));
}
