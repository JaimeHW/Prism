use super::*;

// A simple traceable type for testing
struct TestObject {
    value: i32,
}

unsafe impl Trace for TestObject {
    fn trace(&self, _tracer: &mut dyn crate::trace::Tracer) {}
}

#[test]
fn test_gc_ref_creation() {
    let mut obj = TestObject { value: 42 };
    let gc_ref = unsafe { GcRef::from_raw(&mut obj) };

    assert_eq!(gc_ref.value, 42);
}

#[test]
fn test_gc_ref_deref_mut() {
    let mut obj = TestObject { value: 42 };
    let mut gc_ref = unsafe { GcRef::from_raw(&mut obj) };

    gc_ref.value = 100;
    assert_eq!(gc_ref.value, 100);
}
