use super::*;

#[test]
fn test_new() {
    let e = OsError::new(2, "Not found");
    assert_eq!(e.code, 2);
    assert_eq!(e.message, "Not found");
    assert!(e.path.is_none());
}

#[test]
fn test_with_path() {
    let e = OsError::with_path(2, "Not found", "/foo");
    assert_eq!(e.path, Some("/foo".to_string()));
}

#[test]
fn test_file_not_found() {
    let e = OsError::file_not_found("/missing");
    assert_eq!(e.code, 2);
    assert!(e.to_string().contains("/missing"));
}

#[test]
fn test_permission_denied() {
    let e = OsError::permission_denied("/secret");
    assert_eq!(e.code, 13);
}

#[test]
fn test_display_with_path() {
    let e = OsError::with_path(2, "Not found", "/foo");
    let s = e.to_string();
    assert!(s.contains("Errno 2"));
    assert!(s.contains("/foo"));
}

#[test]
fn test_display_without_path() {
    let e = OsError::new(1, "Error");
    let s = e.to_string();
    assert!(s.contains("Errno 1"));
    assert!(!s.contains("'"));
}

#[test]
fn test_from_io_error() {
    let io_e = io::Error::new(io::ErrorKind::NotFound, "not found");
    let e = OsError::from_io_error(&io_e, "/path");
    assert!(e.path.is_some());
}
