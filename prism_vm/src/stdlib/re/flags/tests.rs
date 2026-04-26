use super::*;

#[test]
fn test_flag_constants() {
    assert_eq!(RegexFlags::TEMPLATE, 1);
    assert_eq!(RegexFlags::IGNORECASE, 2);
    assert_eq!(RegexFlags::LOCALE, 4);
    assert_eq!(RegexFlags::MULTILINE, 8);
    assert_eq!(RegexFlags::DOTALL, 16);
    assert_eq!(RegexFlags::UNICODE, 32);
    assert_eq!(RegexFlags::VERBOSE, 64);
    assert_eq!(RegexFlags::DEBUG, 128);
    assert_eq!(RegexFlags::ASCII, 256);
}

#[test]
fn test_flag_contains() {
    let flags = RegexFlags::new(RegexFlags::IGNORECASE | RegexFlags::MULTILINE);
    assert!(flags.is_case_insensitive());
    assert!(flags.is_multiline());
    assert!(!flags.is_dotall());
}

#[test]
fn test_flag_union() {
    let a = RegexFlags::new(RegexFlags::IGNORECASE);
    let b = RegexFlags::new(RegexFlags::MULTILINE);
    let c = a.union(b);
    assert!(c.is_case_insensitive());
    assert!(c.is_multiline());
}

#[test]
fn test_flag_display() {
    let flags = RegexFlags::new(RegexFlags::IGNORECASE | RegexFlags::DOTALL);
    let s = flags.to_string();
    assert!(s.contains("IGNORECASE"));
    assert!(s.contains("DOTALL"));
}
