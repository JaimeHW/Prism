use super::*;

#[test]
fn test_raw_handle() {
    let value = 42i64;
    let handle = RawHandle::new(&value as *const _ as *const ());
    assert!(!handle.is_null());
}

#[test]
fn test_null_handle() {
    let handle = RawHandle::null();
    assert!(handle.is_null());
}

#[test]
fn test_handle_scope() {
    let mut scope = HandleScope::new();
    assert!(scope.is_empty());

    let value = 42i64;
    unsafe {
        let _handle: GcHandle<i64> = scope.create_handle(&value);
    }

    assert_eq!(scope.len(), 1);
}

#[test]
fn test_handle_scope_clear() {
    let mut scope = HandleScope::new();

    let value = 42i64;
    unsafe {
        let _h1: GcHandle<i64> = scope.create_handle(&value);
        let _h2: GcHandle<i64> = scope.create_handle(&value);
    }
    assert_eq!(scope.len(), 2);

    scope.clear();
    assert!(scope.is_empty());
}
