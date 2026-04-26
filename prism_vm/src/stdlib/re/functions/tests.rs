use super::*;

#[test]
fn test_compile() {
    let pattern = compile(r"\d+", 0).unwrap();
    assert!(pattern.is_match("123"));
}

#[test]
fn test_match() {
    let m = match_default(r"\d+", "123abc").unwrap();
    assert!(m.is_some());
    assert_eq!(m.unwrap().as_str(), "123");

    let m = match_default(r"\d+", "abc123").unwrap();
    assert!(m.is_none());
}

#[test]
fn test_search() {
    let m = search_default(r"\d+", "abc123def").unwrap();
    assert!(m.is_some());
    assert_eq!(m.unwrap().as_str(), "123");
}

#[test]
fn test_fullmatch() {
    assert!(fullmatch_default(r"\d+", "123").unwrap().is_some());
    assert!(fullmatch_default(r"\d+", "123abc").unwrap().is_none());
}

#[test]
fn test_findall() {
    let matches = findall_default(r"\d+", "a1b22c333").unwrap();
    assert_eq!(matches.len(), 3);
}

#[test]
fn test_findall_strings() {
    let strings = findall_strings(r"\d+", "a1b22c333", 0).unwrap();
    assert_eq!(strings.len(), 3);
    assert_eq!(strings[0], vec!["1"]);
    assert_eq!(strings[1], vec!["22"]);
    assert_eq!(strings[2], vec!["333"]);
}

#[test]
fn test_sub() {
    assert_eq!(sub_default(r"\d+", "X", "a1b2c3").unwrap(), "aXbXcX");
}

#[test]
fn test_sub_count() {
    assert_eq!(sub(r"\d+", "X", "a1b2c3", 2, 0).unwrap(), "aXbXc3");
}

#[test]
fn test_subn() {
    let (result, count) = subn_default(r"\d+", "X", "a1b2c3").unwrap();
    assert_eq!(result, "aXbXcX");
    assert_eq!(count, 3);
}

#[test]
fn test_split() {
    let parts = split_default(r",\s*", "a, b,  c").unwrap();
    assert_eq!(parts, vec!["a", "b", "c"]);
}

#[test]
fn test_split_maxsplit() {
    let parts = split(r",", "a,b,c,d", 2, 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[2], "c,d");
}

#[test]
fn test_escape() {
    assert_eq!(escape(r"a.b*c?"), r"a\.b\*c\?");
    assert_eq!(escape(r"hello"), r"hello");
    assert_eq!(escape(r"[test]"), r"\[test\]");
}

#[test]
fn test_flags() {
    // Case insensitive
    let m = match_(r"hello", "HELLO", RegexFlags::IGNORECASE).unwrap();
    assert!(m.is_some());

    // Without flag
    let m = match_(r"hello", "HELLO", 0).unwrap();
    assert!(m.is_none());
}

#[test]
fn test_multiline() {
    let matches = findall(r"^\d+", "1\n2\n3", RegexFlags::MULTILINE).unwrap();
    assert_eq!(matches.len(), 3);
}

#[test]
fn test_dotall() {
    let m = search(r"a.b", "a\nb", RegexFlags::DOTALL).unwrap();
    assert!(m.is_some());

    let m = search(r"a.b", "a\nb", 0).unwrap();
    assert!(m.is_none());
}

#[test]
fn test_error_invalid_pattern() {
    let result = compile(r"[invalid", 0);
    assert!(result.is_err());
}

#[test]
fn test_groups_in_findall() {
    let strings = findall_strings(r"(\d+)-(\d+)", "1-2 3-4 5-6", 0).unwrap();
    assert_eq!(strings.len(), 3);
    assert_eq!(strings[0], vec!["1", "2"]);
    assert_eq!(strings[1], vec!["3", "4"]);
    assert_eq!(strings[2], vec!["5", "6"]);
}

#[test]
fn test_backreference() {
    // Should use fancy engine
    let m = search_default(r"(.)\1", "hello").unwrap();
    assert!(m.is_some());
    assert_eq!(m.unwrap().as_str(), "ll");
}
