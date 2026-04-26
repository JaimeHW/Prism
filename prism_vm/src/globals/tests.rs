use super::*;

#[test]
fn test_global_scope_basic() {
    let mut globals = GlobalScope::new();

    // Set and get
    globals.set("x".into(), Value::int(42).unwrap());
    assert_eq!(globals.get("x").unwrap().as_int(), Some(42));

    // Not found
    assert!(globals.get("y").is_none());
}

#[test]
fn test_global_scope_delete() {
    let mut globals = GlobalScope::new();

    globals.set("x".into(), Value::int(10).unwrap());
    assert!(globals.contains("x"));

    let old = globals.delete("x");
    assert_eq!(old.unwrap().as_int(), Some(10));
    assert!(!globals.contains("x"));
}

#[test]
fn test_global_scope_overwrite() {
    let mut globals = GlobalScope::new();

    globals.set("x".into(), Value::int(1).unwrap());
    globals.set("x".into(), Value::int(2).unwrap());

    assert_eq!(globals.get("x").unwrap().as_int(), Some(2));
    assert_eq!(globals.len(), 1);
}
