use super::*;

#[test]
fn test_getpid_nonzero() {
    assert!(getpid() > 0);
}

#[test]
fn test_getpid_consistent() {
    let p1 = getpid();
    let p2 = getpid();
    assert_eq!(p1, p2);
}

#[test]
fn test_getprocessname() {
    let name = getprocessname();
    // Should have some name
    assert!(name.is_some());
}
