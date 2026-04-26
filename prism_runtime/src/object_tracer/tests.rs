use super::*;
use crate::gc_dispatch::init_gc_dispatch;
use crate::types::list::ListObject;
use prism_core::Value;
use std::mem;

#[test]
fn test_runtime_object_tracer_creation() {
    let tracer = RuntimeObjectTracer::new();
    // Should be zero-sized
    assert_eq!(mem::size_of_val(&tracer), 0);
}

#[test]
fn test_trace_list_through_tracer() {
    init_gc_dispatch();

    let tracer = RuntimeObjectTracer::new();
    let mut list = ListObject::new();
    list.push(Value::int(42).unwrap());

    struct CountingTracer {
        value_count: usize,
    }
    impl Tracer for CountingTracer {
        fn trace_value(&mut self, _value: Value) {
            self.value_count += 1;
        }
        fn trace_ptr(&mut self, _ptr: *const ()) {}
    }

    let mut counting = CountingTracer { value_count: 0 };

    // Get pointer to list (the header is embedded in the ListObject)
    let ptr = &list as *const ListObject as *const ();

    unsafe {
        tracer.trace_object(ptr, &mut counting);
    }

    // Should trace 1 value
    assert_eq!(counting.value_count, 1);
}

#[test]
fn test_size_through_tracer() {
    init_gc_dispatch();

    let tracer = RuntimeObjectTracer::new();
    let list = ListObject::new();

    let ptr = &list as *const ListObject as *const ();

    let size = unsafe { tracer.size_of_object(ptr) };
    assert_eq!(size, mem::size_of::<ListObject>());
}

#[test]
fn test_null_pointer_handling() {
    let tracer = RuntimeObjectTracer::new();

    struct PanicTracer;
    impl Tracer for PanicTracer {
        fn trace_value(&mut self, _value: Value) {
            panic!("Should not be called");
        }
        fn trace_ptr(&mut self, _ptr: *const ()) {
            panic!("Should not be called");
        }
    }

    let mut panic_tracer = PanicTracer;

    // Should handle null pointer gracefully
    unsafe {
        tracer.trace_object(std::ptr::null(), &mut panic_tracer);
        assert_eq!(tracer.size_of_object(std::ptr::null()), 0);
    }
}
