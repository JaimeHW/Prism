use super::*;
use crate::types::bytes::BytesObject;
use crate::types::list::ListObject;
use prism_core::Value;

#[test]
fn test_dispatch_table_initialization() {
    init_gc_dispatch();

    // Verify all built-in types are registered
    let table = dispatch_table();

    // ListObject should have a valid entry
    let list_entry = table.get(TypeId::LIST);
    assert!(list_entry.size as usize != (size_noop as SizeFn) as usize);
}

#[test]
fn test_trace_list_object() {
    init_gc_dispatch();

    let list = ListObject::new();
    let ptr = &list as *const ListObject as *const ();

    // Create a counting tracer
    struct CountingTracer {
        count: usize,
    }
    impl Tracer for CountingTracer {
        fn trace_value(&mut self, value: Value) {
            if value.as_object_ptr().is_some() {
                self.count += 1;
            }
        }
        fn trace_ptr(&mut self, ptr: *const ()) {
            if !ptr.is_null() {
                self.count += 1;
            }
        }
    }

    let mut tracer = CountingTracer { count: 0 };

    unsafe {
        trace_object(ptr, TypeId::LIST, &mut tracer);
    }

    // Empty list traces no children
    assert_eq!(tracer.count, 0);
}

#[test]
fn test_size_of_list() {
    init_gc_dispatch();

    let list = ListObject::new();
    let ptr = &list as *const ListObject as *const ();

    let size = unsafe { size_of_object(ptr, TypeId::LIST) };
    assert_eq!(size, mem::size_of::<ListObject>());
}

#[test]
fn test_size_of_dict() {
    init_gc_dispatch();

    let dict = DictObject::new();
    let ptr = &dict as *const DictObject as *const ();

    let size = unsafe { size_of_object(ptr, TypeId::DICT) };
    assert_eq!(size, mem::size_of::<DictObject>());
}

#[test]
fn test_size_of_bytes() {
    init_gc_dispatch();

    let bytes = BytesObject::from_slice(b"hello");
    let ptr = &bytes as *const BytesObject as *const ();

    let size = unsafe { size_of_object(ptr, TypeId::BYTES) };
    assert_eq!(size, mem::size_of::<BytesObject>() + 5);
}

#[test]
fn test_noop_for_unknown_type() {
    init_gc_dispatch();

    // Type ID 100 is not registered
    let entry = dispatch_table().get(TypeId(100));

    // Should return noop functions
    unsafe {
        let size = (entry.size)(std::ptr::null());
        assert_eq!(size, 0);
    }
}

#[test]
fn test_dispatch_table_all_types() {
    init_gc_dispatch();

    let table = dispatch_table();

    // Verify known types have non-noop entries
    let type_ids = [
        TypeId::STR,
        TypeId::BYTES,
        TypeId::BYTEARRAY,
        TypeId::LIST,
        TypeId::TUPLE,
        TypeId::DICT,
        TypeId::SET,
        TypeId::FUNCTION,
        TypeId::METHOD,
        TypeId::CLASSMETHOD,
        TypeId::STATICMETHOD,
        TypeId::PROPERTY,
        TypeId::RANGE,
        TypeId::ITERATOR,
    ];

    for type_id in type_ids {
        let entry = table.get(type_id);
        // Size function should not be noop
        assert!(
            entry.size as usize != (size_noop as SizeFn) as usize,
            "Type {:?} should have non-noop size function",
            type_id
        );
    }
}

#[test]
fn test_trace_list_with_elements() {
    init_gc_dispatch();

    let mut list = ListObject::new();
    list.push(Value::int(42).unwrap());
    list.push(Value::bool(true));
    list.push(Value::none());

    let ptr = &list as *const ListObject as *const ();

    struct CountingTracer {
        value_count: usize,
    }
    impl Tracer for CountingTracer {
        fn trace_value(&mut self, _value: Value) {
            self.value_count += 1;
        }
        fn trace_ptr(&mut self, _ptr: *const ()) {}
    }

    let mut tracer = CountingTracer { value_count: 0 };

    unsafe {
        trace_object(ptr, TypeId::LIST, &mut tracer);
    }

    // Should trace 3 values
    assert_eq!(tracer.value_count, 3);
}
