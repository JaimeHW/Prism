use super::*;

#[test]
fn test_abspath_absolute() {
    #[cfg(windows)]
    let p = abspath("C:\\Windows");
    #[cfg(not(windows))]
    let p = abspath("/usr");
    assert!(Path::new(&p).is_absolute());
}

#[test]
fn test_abspath_relative() {
    let p = abspath("foo");
    assert!(Path::new(&p).is_absolute());
}

#[test]
fn test_normpath_dot() {
    let p = normpath("./foo/./bar");
    assert!(!p.contains("./"));
}

#[test]
fn test_normpath_dotdot() {
    let p = normpath("foo/bar/../baz");
    assert!(!p.contains(".."));
    assert!(p.contains("baz"));
}

#[test]
fn test_normpath_empty() {
    let p = normpath("");
    assert_eq!(p, ".");
}

#[test]
fn test_realpath_curdir() {
    let p = realpath(".").unwrap();
    assert!(Path::new(&p).is_absolute());
}

#[test]
fn test_realpath_nonexistent() {
    assert!(realpath("/nonexistent_12345").is_err());
}

#[test]
fn test_expanduser_no_tilde() {
    assert_eq!(expanduser("/foo/bar"), "/foo/bar");
}

#[test]
fn test_expanduser_tilde() {
    let p = expanduser("~/foo");
    assert!(!p.starts_with('~') || home_dir().is_none());
}
