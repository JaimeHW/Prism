use super::*;
use crate::trace::tracer::CountingTracer;

#[test]
fn test_root_set_creation() {
    let roots = RootSet::new();
    assert_eq!(roots.handle_count(), 0);
    assert_eq!(roots.global_count(), 0);
}

#[test]
fn test_global_roots() {
    let roots = RootSet::new();

    roots.add_global(Value::int(42).unwrap());
    roots.add_global(Value::bool(true));

    assert_eq!(roots.global_count(), 2);

    let mut tracer = CountingTracer::new();
    roots.trace(&mut tracer);

    assert_eq!(tracer.value_count, 2);
}

#[test]
fn test_clear_globals() {
    let roots = RootSet::new();

    roots.add_global(Value::int(1).unwrap());
    roots.add_global(Value::int(2).unwrap());

    roots.clear_globals();
    assert_eq!(roots.global_count(), 0);
}
