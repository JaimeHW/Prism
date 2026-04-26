use super::*;

#[test]
fn test_container_trace() {
    struct CountingTracer {
        count: usize,
    }
    impl Tracer for CountingTracer {
        fn trace_value(&mut self, _value: Value) {
            self.count += 1;
        }
        fn trace_ptr(&mut self, _ptr: *const ()) {
            self.count += 1;
        }
    }

    let mut tracer = CountingTracer { count: 0 };

    let values: Vec<Value> = vec![Value::none(), Value::bool(true), Value::int(42).unwrap()];
    values.trace(&mut tracer);

    assert_eq!(tracer.count, 3);
}
