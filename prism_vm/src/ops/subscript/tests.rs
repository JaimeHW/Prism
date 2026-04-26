use super::*;
use prism_code::{CodeObject, Instruction, Opcode, Register};
use prism_core::intern::intern;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::Shape;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, builtin_class_mro, class_id_to_type_id, register_global_class,
};
use std::sync::Arc;

fn register_test_class(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(TypeId::from_raw(class_id.0));
    }

    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

fn register_dict_subclass(name: &str) -> Arc<PyClassObject> {
    let class = PyClassObject::new(intern(name), &[ClassId(TypeId::DICT.raw())], |id| {
        (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
            builtin_class_mro(class_id_to_type_id(id))
                .into_iter()
                .collect()
        })
    })
    .expect("dict subclass should build");
    register_test_class(class)
}

fn dict_backed_instance_value(class: &Arc<PyClassObject>) -> (*mut ShapedObject, Value) {
    let ptr = Box::into_raw(Box::new(ShapedObject::new_dict_backed(
        class.class_type_id(),
        class.instance_shape().clone(),
    )));
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn tuple_backed_object_value(items: &[Value]) -> (*mut ShapedObject, Value) {
    let ptr = Box::into_raw(Box::new(ShapedObject::new_tuple_backed(
        TypeId::OBJECT,
        Shape::empty(),
        TupleObject::from_slice(items),
    )));
    (ptr, Value::object_ptr(ptr as *const ()))
}

fn vm_with_frame() -> VirtualMachine {
    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(CodeObject::new("sub", "<test>")), 0)
        .expect("frame push failed");
    vm
}

fn exhaust_nursery(vm: &VirtualMachine) {
    while vm.allocator().alloc(DictObject::new()).is_some() {}
}

#[test]
fn test_finish_subscr_allocates_after_full_nursery() {
    let mut vm = vm_with_frame();
    exhaust_nursery(&vm);

    assert!(matches!(
        finish_subscr(
            &mut vm,
            1,
            SubscriptResult::AllocBytes(BytesObject::from_slice(b"abc"))
        ),
        ControlFlow::Continue
    ));
    let bytes_ptr = vm
        .current_frame()
        .get_reg(1)
        .as_object_ptr()
        .expect("bytes slice should allocate");
    assert_eq!(
        unsafe { &*(bytes_ptr as *const BytesObject) }.as_bytes(),
        b"abc"
    );

    assert!(matches!(
        finish_subscr(
            &mut vm,
            2,
            SubscriptResult::AllocString(StringObject::from_string("slice".to_string()))
        ),
        ControlFlow::Continue
    ));
    let string_ptr = vm
        .current_frame()
        .get_reg(2)
        .as_object_ptr()
        .expect("string slice should allocate");
    assert_eq!(
        unsafe { &*(string_ptr as *const StringObject) }.as_str(),
        "slice"
    );
}

// ==========================================================================
// List Slice Tests
// ==========================================================================

#[test]
fn test_list_slice_forward() {
    let list = ListObject::from_iter(vec![
        Value::int_unchecked(0),
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
        Value::int_unchecked(4),
    ]);
    let slice = SliceObject::start_stop(1, 4);
    let result = list_slice(&list, &slice);

    assert_eq!(result.len(), 3);
    assert_eq!(result.get(0).unwrap().as_int(), Some(1));
    assert_eq!(result.get(1).unwrap().as_int(), Some(2));
    assert_eq!(result.get(2).unwrap().as_int(), Some(3));
}

#[test]
fn test_list_slice_step() {
    let list = ListObject::from_iter(vec![
        Value::int_unchecked(0),
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
        Value::int_unchecked(4),
        Value::int_unchecked(5),
    ]);
    let slice = SliceObject::full(0, 6, 2);
    let result = list_slice(&list, &slice);

    assert_eq!(result.len(), 3);
    assert_eq!(result.get(0).unwrap().as_int(), Some(0));
    assert_eq!(result.get(1).unwrap().as_int(), Some(2));
    assert_eq!(result.get(2).unwrap().as_int(), Some(4));
}

#[test]
fn test_list_slice_reverse() {
    let list = ListObject::from_iter(vec![
        Value::int_unchecked(0),
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
        Value::int_unchecked(4),
    ]);
    let slice = SliceObject::new(None, None, Some(-1));
    let result = list_slice(&list, &slice);

    assert_eq!(result.len(), 5);
    assert_eq!(result.get(0).unwrap().as_int(), Some(4));
    assert_eq!(result.get(1).unwrap().as_int(), Some(3));
    assert_eq!(result.get(2).unwrap().as_int(), Some(2));
    assert_eq!(result.get(3).unwrap().as_int(), Some(1));
    assert_eq!(result.get(4).unwrap().as_int(), Some(0));
}

#[test]
fn test_list_slice_empty() {
    let list = ListObject::from_iter(vec![
        Value::int_unchecked(0),
        Value::int_unchecked(1),
        Value::int_unchecked(2),
    ]);
    let slice = SliceObject::start_stop(5, 10); // Out of bounds
    let result = list_slice(&list, &slice);

    assert_eq!(result.len(), 0);
}

// ==========================================================================
// Tuple Slice Tests
// ==========================================================================

#[test]
fn test_tuple_slice_forward() {
    let tuple = TupleObject::from_vec(vec![
        Value::int_unchecked(10),
        Value::int_unchecked(20),
        Value::int_unchecked(30),
        Value::int_unchecked(40),
    ]);
    let slice = SliceObject::start_stop(0, 2);
    let result = tuple_slice(&tuple, &slice);

    assert_eq!(result.len(), 2);
    assert_eq!(result.get(0).unwrap().as_int(), Some(10));
    assert_eq!(result.get(1).unwrap().as_int(), Some(20));
}

#[test]
fn test_tuple_slice_step() {
    let tuple = TupleObject::from_vec(vec![
        Value::int_unchecked(0),
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
        Value::int_unchecked(4),
        Value::int_unchecked(5),
    ]);
    let slice = SliceObject::full(1, 6, 2);
    let result = tuple_slice(&tuple, &slice);

    assert_eq!(result.len(), 3);
    assert_eq!(result.get(0).unwrap().as_int(), Some(1));
    assert_eq!(result.get(1).unwrap().as_int(), Some(3));
    assert_eq!(result.get(2).unwrap().as_int(), Some(5));
}

#[test]
fn test_tuple_slice_reverse() {
    let tuple = TupleObject::from_vec(vec![
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]);
    let slice = SliceObject::new(None, None, Some(-1));
    let result = tuple_slice(&tuple, &slice);

    assert_eq!(result.len(), 3);
    assert_eq!(result.get(0).unwrap().as_int(), Some(3));
    assert_eq!(result.get(1).unwrap().as_int(), Some(2));
    assert_eq!(result.get(2).unwrap().as_int(), Some(1));
}

#[test]
fn test_tuple_backed_object_integer_subscript() {
    let (ptr, value) = tuple_backed_object_value(&[
        Value::int_unchecked(10),
        Value::int_unchecked(20),
        Value::int_unchecked(30),
    ]);

    let result = subscr_integer(value, -1).expect("tuple-backed object should index");
    match result {
        Some(SubscriptResult::Value(value)) => assert_eq!(value.as_int(), Some(30)),
        Some(_) => panic!("expected direct tuple item result"),
        None => panic!("expected tuple-backed integer fast path"),
    }

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_tuple_backed_object_slice_subscript() {
    let (ptr, value) = tuple_backed_object_value(&[
        Value::int_unchecked(0),
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]);
    let slice = SliceObject::full(1, 4, 2);

    let result = subscr_slice(value, &slice).expect("tuple-backed object should slice");
    match result {
        Some(SubscriptResult::AllocTuple(tuple)) => {
            assert_eq!(
                tuple.as_slice(),
                &[Value::int_unchecked(1), Value::int_unchecked(3)]
            );
        }
        Some(_) => panic!("expected allocated tuple slice"),
        None => panic!("expected tuple-backed slice fast path"),
    }

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

// ==========================================================================
// String Slice Tests
// ==========================================================================

#[test]
fn test_string_slice_forward() {
    let string = StringObject::new("hello");
    let slice = SliceObject::start_stop(1, 4);
    let result = string_slice(&string, &slice);

    assert_eq!(result.as_str(), "ell");
}

#[test]
fn test_string_slice_step() {
    let string = StringObject::new("abcdef");
    let slice = SliceObject::full(0, 6, 2);
    let result = string_slice(&string, &slice);

    assert_eq!(result.as_str(), "ace");
}

#[test]
fn test_string_slice_reverse() {
    let string = StringObject::new("hello");
    let slice = SliceObject::new(None, None, Some(-1));
    let result = string_slice(&string, &slice);

    assert_eq!(result.as_str(), "olleh");
}

#[test]
fn test_string_slice_unicode() {
    let string = StringObject::new("héllo");
    let slice = SliceObject::start_stop(0, 3);
    let result = string_slice(&string, &slice);

    assert_eq!(result.as_str(), "hél");
}

#[test]
fn test_string_slice_empty() {
    let string = StringObject::new("test");
    let slice = SliceObject::start_stop(10, 20);
    let result = string_slice(&string, &slice);

    assert_eq!(result.as_str(), "");
}

#[test]
fn test_range_slice_reverse_returns_range() {
    let range = RangeObject::from_stop(5);
    let range_ptr = Box::leak(Box::new(range)) as *mut RangeObject as *const ();
    let slice = SliceObject::new(None, None, Some(-1));
    let result =
        subscr_slice(Value::object_ptr(range_ptr), &slice).expect("range slicing should work");

    match result {
        Some(SubscriptResult::AllocRange(range)) => {
            assert_eq!(
                range.to_vec(),
                vec![
                    Value::int_unchecked(4),
                    Value::int_unchecked(3),
                    Value::int_unchecked(2),
                    Value::int_unchecked(1),
                    Value::int_unchecked(0),
                ]
            );
        }
        Some(_) => panic!("expected range slice result"),
        None => panic!("expected range slice fast path"),
    }
}

#[test]
fn test_tagged_string_integer_subscript_forward() {
    let result = subscr_integer(Value::string(intern("hello")), 1)
        .expect("tagged string indexing should succeed");

    match result {
        Some(SubscriptResult::AllocString(string)) => assert_eq!(string.as_str(), "e"),
        Some(SubscriptResult::Value(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocBytes(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocList(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocTuple(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocRange(_)) => panic!("expected allocated string result"),
        None => panic!("expected integer fast path"),
    }
}

#[test]
fn test_tagged_string_integer_subscript_negative_index() {
    let result = subscr_integer(Value::string(intern("hello")), -1)
        .expect("tagged string negative indexing should succeed");

    match result {
        Some(SubscriptResult::AllocString(string)) => assert_eq!(string.as_str(), "o"),
        Some(SubscriptResult::Value(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocBytes(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocList(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocTuple(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocRange(_)) => panic!("expected allocated string result"),
        None => panic!("expected integer fast path"),
    }
}

#[test]
fn test_tagged_string_slice_forward() {
    let slice = SliceObject::start_stop(1, 4);
    let result = subscr_slice(Value::string(intern("hello")), &slice)
        .expect("tagged string slicing should succeed");

    match result {
        Some(SubscriptResult::AllocString(string)) => assert_eq!(string.as_str(), "ell"),
        Some(SubscriptResult::Value(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocBytes(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocList(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocTuple(_)) => panic!("expected allocated string result"),
        Some(SubscriptResult::AllocRange(_)) => panic!("expected allocated string result"),
        None => panic!("expected slice fast path"),
    }
}

#[test]
fn test_bytes_integer_subscript_returns_int_value() {
    let bytes = BytesObject::from_slice(b"abc");
    let bytes_ptr = Box::leak(Box::new(bytes)) as *mut BytesObject as *const ();
    let result =
        subscr_integer(Value::object_ptr(bytes_ptr), 1).expect("bytes indexing should succeed");

    match result {
        Some(SubscriptResult::Value(value)) => {
            assert_eq!(value.as_int(), Some(i64::from(b'b')))
        }
        Some(SubscriptResult::AllocBytes(_)) => panic!("expected integer value result"),
        Some(SubscriptResult::AllocString(_)) => panic!("expected integer value result"),
        Some(SubscriptResult::AllocList(_)) => panic!("expected integer value result"),
        Some(SubscriptResult::AllocTuple(_)) => panic!("expected integer value result"),
        Some(SubscriptResult::AllocRange(_)) => panic!("expected integer value result"),
        None => panic!("expected integer fast path"),
    }
}

#[test]
fn test_bytearray_slice_preserves_concrete_type() {
    let bytes = BytesObject::bytearray_from_slice(b"abcd");
    let bytes_ptr = Box::leak(Box::new(bytes)) as *mut BytesObject as *const ();
    let slice = SliceObject::new(None, None, Some(-1));
    let result = subscr_slice(Value::object_ptr(bytes_ptr), &slice)
        .expect("bytearray slicing should succeed");

    match result {
        Some(SubscriptResult::AllocBytes(bytes)) => {
            assert!(bytes.is_bytearray());
            assert_eq!(bytes.as_bytes(), b"dcba");
        }
        Some(SubscriptResult::Value(_)) => panic!("expected allocated byte sequence result"),
        Some(SubscriptResult::AllocString(_)) => {
            panic!("expected allocated byte sequence result")
        }
        Some(SubscriptResult::AllocList(_)) => {
            panic!("expected allocated byte sequence result")
        }
        Some(SubscriptResult::AllocTuple(_)) => {
            panic!("expected allocated byte sequence result")
        }
        Some(SubscriptResult::AllocRange(_)) => {
            panic!("expected allocated byte sequence result")
        }
        None => panic!("expected slice fast path"),
    }
}

#[test]
fn test_binary_subscr_reads_heap_dict_subclass_native_storage() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("sub", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");

    let class = register_dict_subclass("BinaryDictSubclass");
    let (instance_ptr, instance_value) = dict_backed_instance_value(&class);
    unsafe {
        (*instance_ptr)
            .dict_backing_mut()
            .expect("dict subclass should expose native dict storage")
            .set(Value::string(intern("answer")), Value::int_unchecked(42));
    }

    vm.current_frame_mut().set_reg(1, instance_value);
    vm.current_frame_mut()
        .set_reg(2, Value::string(intern("answer")));

    let inst = Instruction::op_dss(
        Opcode::GetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(
        binary_subscr(&mut vm, inst),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(3).as_int(), Some(42));
}

#[test]
fn test_binary_subscr_on_type_object_produces_generic_alias() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("sub", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");
    vm.current_frame_mut().set_reg(
        1,
        crate::builtins::builtin_type_object_for_type_id(TypeId::LIST),
    );
    vm.current_frame_mut().set_reg(
        2,
        crate::builtins::builtin_type_object_for_type_id(TypeId::INT),
    );

    let inst = Instruction::op_dss(
        Opcode::GetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(
        binary_subscr(&mut vm, inst),
        ControlFlow::Continue
    ));
    let ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::GENERIC_ALIAS);
}

#[test]
fn test_binary_subscr_on_mapping_proxy_returns_descriptor_view() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("sub", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");

    let mapping = crate::builtins::builtin_type_attribute_value(
        &mut vm,
        TypeId::DICT,
        &prism_core::intern::intern("__dict__"),
    )
    .expect("mapping proxy allocation should succeed")
    .expect("dict type should expose __dict__");

    vm.current_frame_mut().set_reg(1, mapping);
    vm.current_frame_mut()
        .set_reg(2, Value::string(prism_core::intern::intern("fromkeys")));

    let inst = Instruction::op_dss(
        Opcode::GetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(
        binary_subscr(&mut vm, inst),
        ControlFlow::Continue
    ));
    let ptr = vm.current_frame().get_reg(3).as_object_ptr().unwrap();
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::CLASSMETHOD_DESCRIPTOR);
}

#[test]
fn test_store_subscr_assigns_list_slice_from_iterable() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("store", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");

    let list = ListObject::from_iter(vec![
        Value::int_unchecked(0),
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]);
    let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let slice = SliceObject::start_stop(1, 3);
    let slice_ptr = Box::leak(Box::new(slice)) as *mut SliceObject as *const ();
    let replacement =
        ListObject::from_iter(vec![Value::int_unchecked(10), Value::int_unchecked(11)]);
    let replacement_ptr = Box::leak(Box::new(replacement)) as *mut ListObject as *const ();

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(list_ptr));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(replacement_ptr));
    vm.current_frame_mut()
        .set_reg(3, Value::object_ptr(slice_ptr));

    let inst = Instruction::op_dss(
        Opcode::SetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(store_subscr(&mut vm, inst), ControlFlow::Continue));

    let stored = unsafe { &*(list_ptr as *const ListObject) };
    assert_eq!(
        stored.as_slice(),
        &[
            Value::int_unchecked(0),
            Value::int_unchecked(10),
            Value::int_unchecked(11),
            Value::int_unchecked(3),
        ]
    );
}

#[test]
fn test_store_subscr_updates_bytearray_index_and_slice() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("store", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");

    let bytearray = BytesObject::bytearray_from_slice(b"MYAAAAAA");
    let bytearray_ptr = Box::leak(Box::new(bytearray)) as *mut BytesObject as *const ();
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(bytearray_ptr));
    vm.current_frame_mut().set_reg(2, Value::int_unchecked(90));
    vm.current_frame_mut().set_reg(3, Value::int_unchecked(0));

    let item_inst = Instruction::op_dss(
        Opcode::SetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(
        store_subscr(&mut vm, item_inst),
        ControlFlow::Continue
    ));

    let slice = SliceObject::new(Some(-6), None, None);
    let slice_ptr = Box::leak(Box::new(slice)) as *mut SliceObject as *const ();
    let replacement = BytesObject::from_slice(b"======");
    let replacement_ptr = Box::leak(Box::new(replacement)) as *mut BytesObject as *const ();
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(replacement_ptr));
    vm.current_frame_mut()
        .set_reg(3, Value::object_ptr(slice_ptr));

    let slice_inst = Instruction::op_dss(
        Opcode::SetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(
        store_subscr(&mut vm, slice_inst),
        ControlFlow::Continue
    ));

    let stored = unsafe { &*(bytearray_ptr as *const BytesObject) };
    assert_eq!(stored.as_bytes(), b"ZY======");
}

#[test]
fn test_store_subscr_rejects_mismatched_bytearray_extended_slice_assignment() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("store", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");

    let bytearray = BytesObject::bytearray_from_slice(b"abcdef");
    let bytearray_ptr = Box::leak(Box::new(bytearray)) as *mut BytesObject as *const ();
    let slice = SliceObject::full(0, 6, 2);
    let slice_ptr = Box::leak(Box::new(slice)) as *mut SliceObject as *const ();
    let replacement = BytesObject::from_slice(b"xy");
    let replacement_ptr = Box::leak(Box::new(replacement)) as *mut BytesObject as *const ();

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(bytearray_ptr));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(replacement_ptr));
    vm.current_frame_mut()
        .set_reg(3, Value::object_ptr(slice_ptr));

    let inst = Instruction::op_dss(
        Opcode::SetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    match store_subscr(&mut vm, inst) {
        ControlFlow::Error(error) => match error.kind() {
            crate::error::RuntimeErrorKind::ValueError { message } => {
                assert!(message.contains("extended slice"));
            }
            other => panic!("expected ValueError, got {:?}", other),
        },
        other => panic!("expected error, got {:?}", other),
    }
}

#[test]
fn test_store_subscr_updates_heap_dict_subclass_native_storage() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("store", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");

    let class = register_dict_subclass("StoreDictSubclass");
    let (instance_ptr, instance_value) = dict_backed_instance_value(&class);

    vm.current_frame_mut().set_reg(1, instance_value);
    vm.current_frame_mut().set_reg(2, Value::int_unchecked(99));
    vm.current_frame_mut()
        .set_reg(3, Value::string(intern("token")));

    let inst = Instruction::op_dss(
        Opcode::SetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(store_subscr(&mut vm, inst), ControlFlow::Continue));

    let stored = unsafe {
        (*instance_ptr)
            .dict_backing()
            .expect("dict subclass should expose native dict storage")
            .get(Value::string(intern("token")))
    };
    assert_eq!(stored.as_ref().and_then(Value::as_int), Some(99));
}

#[test]
fn test_store_subscr_rejects_mismatched_extended_slice_assignment() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("store", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");

    let list = ListObject::from_iter(vec![
        Value::int_unchecked(0),
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]);
    let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let slice = SliceObject::full(0, 4, 2);
    let slice_ptr = Box::leak(Box::new(slice)) as *mut SliceObject as *const ();
    let replacement = ListObject::from_iter(vec![Value::int_unchecked(10)]);
    let replacement_ptr = Box::leak(Box::new(replacement)) as *mut ListObject as *const ();

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(list_ptr));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(replacement_ptr));
    vm.current_frame_mut()
        .set_reg(3, Value::object_ptr(slice_ptr));

    let inst = Instruction::op_dss(
        Opcode::SetItem,
        Register::new(3),
        Register::new(1),
        Register::new(2),
    );
    match store_subscr(&mut vm, inst) {
        ControlFlow::Error(error) => match error.kind() {
            crate::error::RuntimeErrorKind::ValueError { message } => {
                assert!(message.contains("extended slice"));
            }
            other => panic!("expected ValueError, got {:?}", other),
        },
        other => panic!("expected error, got {:?}", other),
    }
}

#[test]
fn test_delete_subscr_removes_list_slice() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("delete", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");

    let list = ListObject::from_iter(vec![
        Value::int_unchecked(0),
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
        Value::int_unchecked(4),
    ]);
    let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let slice = SliceObject::full(0, 5, 2);
    let slice_ptr = Box::leak(Box::new(slice)) as *mut SliceObject as *const ();

    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(list_ptr));
    vm.current_frame_mut()
        .set_reg(2, Value::object_ptr(slice_ptr));

    let inst = Instruction::op_dss(
        Opcode::DelItem,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(
        delete_subscr(&mut vm, inst),
        ControlFlow::Continue
    ));

    let stored = unsafe { &*(list_ptr as *const ListObject) };
    assert_eq!(
        stored.as_slice(),
        &[Value::int_unchecked(1), Value::int_unchecked(3)]
    );
}

#[test]
fn test_delete_subscr_removes_heap_dict_subclass_native_storage_entry() {
    let mut vm = VirtualMachine::new();
    let code = std::sync::Arc::new(prism_code::CodeObject::new("delete", "<test>"));
    vm.push_frame(code, 0).expect("frame push failed");

    let class = register_dict_subclass("DeleteDictSubclass");
    let (instance_ptr, instance_value) = dict_backed_instance_value(&class);
    unsafe {
        (*instance_ptr)
            .dict_backing_mut()
            .expect("dict subclass should expose native dict storage")
            .set(Value::string(intern("victim")), Value::int_unchecked(7));
    }

    vm.current_frame_mut().set_reg(1, instance_value);
    vm.current_frame_mut()
        .set_reg(2, Value::string(intern("victim")));

    let inst = Instruction::op_dss(
        Opcode::DelItem,
        Register::new(0),
        Register::new(1),
        Register::new(2),
    );
    assert!(matches!(
        delete_subscr(&mut vm, inst),
        ControlFlow::Continue
    ));

    let stored = unsafe {
        (*instance_ptr)
            .dict_backing()
            .expect("dict subclass should expose native dict storage")
            .get(Value::string(intern("victim")))
    };
    assert!(stored.is_none());
}
