use super::*;
use prism_core::Value;

/// Test tracer that counts traced values and pointers
struct CountingTracer {
    value_count: usize,
    ptr_count: usize,
}

impl CountingTracer {
    fn new() -> Self {
        Self {
            value_count: 0,
            ptr_count: 0,
        }
    }
}

impl Tracer for CountingTracer {
    fn trace_value(&mut self, _value: Value) {
        self.value_count += 1;
    }

    fn trace_ptr(&mut self, _ptr: *const ()) {
        self.ptr_count += 1;
    }
}

#[test]
fn test_string_object_trace() {
    let mut tracer = CountingTracer::new();
    let string = StringObject::new("hello world");
    string.trace(&mut tracer);

    // StringObject is a leaf type - no values traced
    assert_eq!(tracer.value_count, 0);
    assert_eq!(tracer.ptr_count, 0);
}

#[test]
fn test_range_object_trace() {
    let mut tracer = CountingTracer::new();
    let range = RangeObject::new(0, 10, 1);
    range.trace(&mut tracer);

    // RangeObject is a leaf type - no values traced
    assert_eq!(tracer.value_count, 0);
    assert_eq!(tracer.ptr_count, 0);
}

#[test]
fn test_bytes_object_trace() {
    let mut tracer = CountingTracer::new();
    let bytes = BytesObject::from_slice(b"abc");
    bytes.trace(&mut tracer);

    // BytesObject is a leaf type - no values traced
    assert_eq!(tracer.value_count, 0);
    assert_eq!(tracer.ptr_count, 0);
}

#[test]
fn test_list_object_trace() {
    let mut tracer = CountingTracer::new();
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    list.trace(&mut tracer);

    // Should trace 3 values
    assert_eq!(tracer.value_count, 3);
}

#[test]
fn test_tuple_object_trace() {
    let mut tracer = CountingTracer::new();
    let tuple =
        TupleObject::from_slice(&[Value::int(1).unwrap(), Value::none(), Value::bool(true)]);
    tuple.trace(&mut tracer);

    // Should trace 3 values
    assert_eq!(tracer.value_count, 3);
}

#[test]
fn test_dict_object_trace() {
    let mut tracer = CountingTracer::new();
    let mut dict = DictObject::new();
    dict.set(Value::int(1).unwrap(), Value::int(100).unwrap());
    dict.set(Value::int(2).unwrap(), Value::int(200).unwrap());
    dict.trace(&mut tracer);

    // Should trace 2 keys + 2 values = 4 values
    assert_eq!(tracer.value_count, 4);
}

#[test]
fn test_set_object_trace() {
    let mut tracer = CountingTracer::new();
    let set = SetObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    set.trace(&mut tracer);

    // Should trace 3 values
    assert_eq!(tracer.value_count, 3);
}

#[test]
fn test_closure_env_trace() {
    let mut tracer = CountingTracer::new();
    let env = ClosureEnv::from_values(
        vec![Value::int(1).unwrap(), Value::int(2).unwrap()].into_boxed_slice(),
        None,
    );
    env.trace(&mut tracer);

    // Should trace 2 captured values
    assert_eq!(tracer.value_count, 2);
}

#[test]
fn test_closure_env_with_parent_trace() {
    use std::sync::Arc;

    let mut tracer = CountingTracer::new();

    let parent = Arc::new(ClosureEnv::from_values(
        vec![Value::int(10).unwrap()].into_boxed_slice(),
        None,
    ));

    let child = ClosureEnv::from_values(
        vec![Value::int(1).unwrap(), Value::int(2).unwrap()].into_boxed_slice(),
        Some(parent),
    );

    child.trace(&mut tracer);

    // Should trace 2 child values + 1 parent value = 3
    assert_eq!(tracer.value_count, 3);
}

#[test]
fn test_closure_env_size_accounts_for_overflow_storage() {
    let inline = ClosureEnv::with_unbound_cells(ClosureEnv::INLINE_LIMIT);
    let overflow = ClosureEnv::with_unbound_cells(ClosureEnv::INLINE_LIMIT + 1);

    assert_eq!(inline.size_of(), std::mem::size_of::<ClosureEnv>());
    assert_eq!(
        overflow.size_of(),
        std::mem::size_of::<ClosureEnv>()
            + overflow.len() * std::mem::size_of::<std::sync::Arc<crate::types::Cell>>()
    );
}

#[test]
fn test_empty_list_trace() {
    let mut tracer = CountingTracer::new();
    let list = ListObject::new();
    list.trace(&mut tracer);

    assert_eq!(tracer.value_count, 0);
}

#[test]
fn test_empty_dict_trace() {
    let mut tracer = CountingTracer::new();
    let dict = DictObject::new();
    dict.trace(&mut tracer);

    assert_eq!(tracer.value_count, 0);
}

#[test]
fn test_object_header_trace() {
    use crate::object::type_obj::TypeId;

    let mut tracer = CountingTracer::new();
    let header = ObjectHeader::new(TypeId::LIST);
    header.trace(&mut tracer);

    // ObjectHeader is a leaf type
    assert_eq!(tracer.value_count, 0);
    assert_eq!(tracer.ptr_count, 0);
}

#[test]
fn test_descriptor_objects_trace_retained_values() {
    let mut tracer = CountingTracer::new();

    StaticMethodDescriptor::new(Value::int_unchecked(1)).trace(&mut tracer);
    ClassMethodDescriptor::new(Value::int_unchecked(2)).trace(&mut tracer);
    PropertyDescriptor::new_full(
        Some(Value::int_unchecked(3)),
        Some(Value::int_unchecked(4)),
        Some(Value::int_unchecked(5)),
        Some(Value::int_unchecked(6)),
    )
    .trace(&mut tracer);
    BoundMethod::new(Value::int_unchecked(7), Value::int_unchecked(8)).trace(&mut tracer);

    assert_eq!(tracer.value_count, 8);
}

#[test]
fn test_size_of_list() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);

    let size = list.size_of();
    // Should be at least header + 3 values
    assert!(size >= std::mem::size_of::<ListObject>());
}

#[test]
fn test_size_of_string() {
    let short = StringObject::new("hello");
    let long = StringObject::new(&"a".repeat(100));

    // Long string should report larger size
    assert!(long.size_of() > short.size_of());
}
