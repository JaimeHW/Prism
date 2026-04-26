use super::*;

#[test]
fn test_noop_tracer() {
    let tracer = NoopObjectTracer;
    unsafe {
        tracer.trace_object(std::ptr::null(), &mut NoopTracerImpl);
        assert_eq!(tracer.size_of_object(std::ptr::null()), 0);
    }
}

struct NoopTracerImpl;
impl Tracer for NoopTracerImpl {
    fn trace_value(&mut self, _value: prism_core::Value) {}
    fn trace_ptr(&mut self, _ptr: *const ()) {}
}
