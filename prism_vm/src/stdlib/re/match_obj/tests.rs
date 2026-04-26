use super::*;

fn create_test_match() -> Match {
    let text = Arc::from("hello world");
    let groups = vec![
        Some(0..5),  // group 0: "hello"
        Some(6..11), // group 1: "world"
    ];
    let mut named_groups = FxHashMap::default();
    named_groups.insert(Arc::from("word"), 1);
    Match::new(text, 0..5, groups, named_groups)
}

#[test]
fn test_as_str() {
    let m = create_test_match();
    assert_eq!(m.as_str(), "hello");
}

#[test]
fn test_group() {
    let m = create_test_match();
    assert_eq!(m.group(0), Some("hello"));
    assert_eq!(m.group(1), Some("world"));
    assert_eq!(m.group(2), None);
}

#[test]
fn test_groups() {
    let m = create_test_match();
    let groups = m.groups();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], Some("world"));
}

#[test]
fn test_start_end_span() {
    let m = create_test_match();
    assert_eq!(m.start(), 0);
    assert_eq!(m.end(), 5);
    assert_eq!(m.span(), (0, 5));
}

#[test]
fn test_start_end_group() {
    let m = create_test_match();
    assert_eq!(m.start_group(0), Some(0));
    assert_eq!(m.end_group(0), Some(5));
    assert_eq!(m.start_group(1), Some(6));
    assert_eq!(m.end_group(1), Some(11));
}

#[test]
fn test_groupdict() {
    let m = create_test_match();
    let dict = m.groupdict();
    assert_eq!(dict.get(&Arc::from("word")), Some(&Some("world")));
}

#[test]
fn test_lastindex() {
    let m = create_test_match();
    assert_eq!(m.lastindex(), Some(1));
}

#[test]
fn test_lastgroup() {
    let m = create_test_match();
    assert_eq!(m.lastgroup(), Some("word"));
}

#[test]
fn test_display() {
    let m = create_test_match();
    let s = m.to_string();
    assert!(s.contains("span=(0, 5)"));
    assert!(s.contains("match='hello'"));
}

#[test]
fn test_from_regex_captures() {
    let re = regex::Regex::new(r"(\d+)-(\d+)").unwrap();
    let caps = re.captures("123-456").unwrap();
    let m = Match::from_captures(&caps, "123-456");
    assert_eq!(m.as_str(), "123-456");
    assert_eq!(m.group(1), Some("123"));
    assert_eq!(m.group(2), Some("456"));
}
