use super::*;

fn assert_interned_string(value: Value, expected: &str) {
    let ptr = value
        .as_string_object_ptr()
        .expect("iterator should yield an interned string") as *const u8;
    let actual = prism_core::intern::interned_by_ptr(ptr)
        .expect("string pointer should resolve through the interner");
    assert_eq!(actual.as_str(), expected);
}

// -------------------------------------------------------------------------
// Type Detection Tests
// -------------------------------------------------------------------------

#[test]
fn test_get_type_id_none() {
    let value = Value::none();
    assert!(get_type_id(&value).is_none());
}

#[test]
fn test_get_type_id_int() {
    let value = Value::int(42).unwrap();
    assert!(get_type_id(&value).is_none());
}

#[test]
fn test_get_type_id_float() {
    let value = Value::float(3.14);
    assert!(get_type_id(&value).is_none());
}

#[test]
fn test_get_type_id_list() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);
    assert_eq!(get_type_id(&value), Some(TypeId::LIST));
}

#[test]
fn test_get_type_id_tuple() {
    let tuple = TupleObject::from_slice(&[Value::int(1).unwrap()]);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    let value = Value::object_ptr(ptr);
    assert_eq!(get_type_id(&value), Some(TypeId::TUPLE));
}

#[test]
fn test_get_type_id_dict() {
    let dict = DictObject::new();
    let ptr = Box::leak(Box::new(dict)) as *mut DictObject as *const ();
    let value = Value::object_ptr(ptr);
    assert_eq!(get_type_id(&value), Some(TypeId::DICT));
}

#[test]
fn test_get_type_id_range() {
    let range = RangeObject::from_stop(10);
    let ptr = Box::leak(Box::new(range)) as *mut RangeObject as *const ();
    let value = Value::object_ptr(ptr);
    assert_eq!(get_type_id(&value), Some(TypeId::RANGE));
}

// -------------------------------------------------------------------------
// Not Iterable Tests
// -------------------------------------------------------------------------

#[test]
fn test_iter_none_not_iterable() {
    let value = Value::none();
    let result = value_to_iterator(&value);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, IterError::NotIterable(_)));
    assert!(err.to_string().contains("NoneType"));
}

#[test]
fn test_iter_int_not_iterable() {
    let value = Value::int(42).unwrap();
    let result = value_to_iterator(&value);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("int"));
}

#[test]
fn test_iter_float_not_iterable() {
    let value = Value::float(3.14);
    let result = value_to_iterator(&value);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("float"));
}

#[test]
fn test_iter_bool_not_iterable() {
    let value = Value::bool(true);
    let result = value_to_iterator(&value);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("bool"));
}

// -------------------------------------------------------------------------
// List Iterator Tests
// -------------------------------------------------------------------------

#[test]
fn test_iter_list_empty() {
    let list = ListObject::new();
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("list should be iterable");
    assert!(iter.next().is_none());
    assert!(iter.is_exhausted());
}

#[test]
fn test_iter_list_single() {
    let list = ListObject::from_slice(&[Value::int(42).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("list should be iterable");
    assert_eq!(iter.next().unwrap().as_int(), Some(42));
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_list_multiple() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("list should be iterable");
    assert_eq!(iter.next().unwrap().as_int(), Some(1));
    assert_eq!(iter.next().unwrap().as_int(), Some(2));
    assert_eq!(iter.next().unwrap().as_int(), Some(3));
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_heap_list_subclass_uses_native_backing() {
    let object = Box::into_raw(Box::new(
        prism_runtime::object::shaped_object::ShapedObject::new_list_backed(
            TypeId::from_raw(512),
            prism_runtime::object::shape::Shape::empty(),
        ),
    ));
    unsafe { &mut *object }
        .list_backing_mut()
        .expect("list backing should exist")
        .extend([Value::int(5).unwrap(), Value::int(8).unwrap()]);
    let value = Value::object_ptr(object as *const ());

    let mut iter = value_to_iterator(&value).expect("list subclass should be iterable");

    assert_eq!(iter.next().unwrap().as_int(), Some(5));
    assert_eq!(iter.next().unwrap().as_int(), Some(8));
    assert!(iter.next().is_none());

    unsafe {
        drop(Box::from_raw(object));
    }
}

#[test]
fn test_iter_list_collect_remaining() {
    let list = ListObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).unwrap();
    iter.next(); // consume first

    let remaining = iter.collect_remaining();
    assert_eq!(remaining.len(), 2);
    assert_eq!(remaining[0].as_int(), Some(20));
    assert_eq!(remaining[1].as_int(), Some(30));
}

#[test]
fn test_iter_list_observes_mutation_after_creation() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject;
    let value = Value::object_ptr(ptr as *const ());

    let mut iter = value_to_iterator(&value).expect("list should be iterable");
    assert_eq!(iter.next().unwrap().as_int(), Some(1));

    unsafe { &mut *ptr }.push(Value::int(3).unwrap());

    assert_eq!(iter.size_hint(), Some(2));
    assert_eq!(iter.next().unwrap().as_int(), Some(2));
    assert_eq!(iter.next().unwrap().as_int(), Some(3));
    assert!(iter.next().is_none());
}

// -------------------------------------------------------------------------
// Tuple Iterator Tests
// -------------------------------------------------------------------------

#[test]
fn test_iter_tuple_empty() {
    let tuple = TupleObject::empty();
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("tuple should be iterable");
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_tuple_single() {
    let tuple = TupleObject::from_slice(&[Value::int(99).unwrap()]);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("tuple should be iterable");
    assert_eq!(iter.next().unwrap().as_int(), Some(99));
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_tuple_heterogeneous() {
    let tuple = TupleObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::float(2.5),
        Value::none(),
        Value::bool(true),
    ]);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).unwrap();
    assert_eq!(iter.next().unwrap().as_int(), Some(1));
    assert_eq!(iter.next().unwrap().as_float(), Some(2.5));
    assert!(iter.next().unwrap().is_none());
    assert!(iter.next().unwrap().is_truthy());
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_tuple_backed_object() {
    let object = ShapedObject::new_tuple_backed(
        TypeId::OBJECT,
        Shape::empty(),
        TupleObject::from_slice(&[Value::int(4).unwrap(), Value::int(9).unwrap()]),
    );
    let ptr = Box::into_raw(Box::new(object));
    let value = Value::object_ptr(ptr as *const ());

    let mut iter = value_to_iterator(&value).expect("tuple-backed object should be iterable");
    assert_eq!(iter.next().unwrap().as_int(), Some(4));
    assert_eq!(iter.next().unwrap().as_int(), Some(9));
    assert!(iter.next().is_none());

    unsafe {
        drop(Box::from_raw(ptr));
    }
}

#[test]
fn test_iter_string_unicode_chars() {
    let string = StringObject::from_string("aé🙂".to_string());
    let ptr = Box::leak(Box::new(string)) as *mut StringObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("string should be iterable");
    assert_interned_string(iter.next().unwrap(), "a");
    assert_interned_string(iter.next().unwrap(), "é");
    assert_interned_string(iter.next().unwrap(), "🙂");
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_tagged_interned_string_unicode_chars() {
    let value = Value::string(prism_core::intern::intern("aé🙂"));

    let mut iter = value_to_iterator(&value).expect("tagged string should be iterable");
    assert_interned_string(iter.next().unwrap(), "a");
    assert_interned_string(iter.next().unwrap(), "é");
    assert_interned_string(iter.next().unwrap(), "🙂");
    assert!(iter.next().is_none());
}

// -------------------------------------------------------------------------
// Range Iterator Tests
// -------------------------------------------------------------------------

#[test]
fn test_iter_range_simple() {
    let range = RangeObject::from_stop(5);
    let ptr = Box::leak(Box::new(range)) as *mut RangeObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("range should be iterable");
    let values: Vec<i64> = std::iter::from_fn(|| iter.next().and_then(|v| v.as_int())).collect();
    assert_eq!(values, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_iter_range_with_start() {
    let range = RangeObject::new(2, 7, 1);
    let ptr = Box::leak(Box::new(range)) as *mut RangeObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).unwrap();
    let values: Vec<i64> = std::iter::from_fn(|| iter.next().and_then(|v| v.as_int())).collect();
    assert_eq!(values, vec![2, 3, 4, 5, 6]);
}

#[test]
fn test_iter_range_with_step() {
    let range = RangeObject::new(0, 10, 2);
    let ptr = Box::leak(Box::new(range)) as *mut RangeObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).unwrap();
    let values: Vec<i64> = std::iter::from_fn(|| iter.next().and_then(|v| v.as_int())).collect();
    assert_eq!(values, vec![0, 2, 4, 6, 8]);
}

#[test]
fn test_iter_range_negative_step() {
    let range = RangeObject::new(5, 0, -1);
    let ptr = Box::leak(Box::new(range)) as *mut RangeObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).unwrap();
    let values: Vec<i64> = std::iter::from_fn(|| iter.next().and_then(|v| v.as_int())).collect();
    assert_eq!(values, vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_iter_range_empty() {
    let range = RangeObject::new(5, 5, 1);
    let ptr = Box::leak(Box::new(range)) as *mut RangeObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).unwrap();
    assert!(iter.next().is_none());
}

// -------------------------------------------------------------------------
// Dict Iterator Tests
// -------------------------------------------------------------------------

#[test]
fn test_iter_dict_empty() {
    let dict = DictObject::new();
    let ptr = Box::leak(Box::new(dict)) as *mut DictObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("dict should be iterable");
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_dict_yields_keys() {
    let mut dict = DictObject::new();
    dict.set(Value::int(1).unwrap(), Value::int(100).unwrap());
    dict.set(Value::int(2).unwrap(), Value::int(200).unwrap());
    let ptr = Box::leak(Box::new(dict)) as *mut DictObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("dict should be iterable");
    let mut keys: Vec<i64> = Vec::new();
    while let Some(k) = iter.next() {
        keys.push(k.as_int().unwrap());
    }
    keys.sort(); // Order not guaranteed
    assert_eq!(keys, vec![1, 2]);
}

#[test]
fn test_iter_dict_view_variants_yield_backing_dict_contents() {
    let mut dict = DictObject::new();
    dict.set(Value::int(3).unwrap(), Value::int(30).unwrap());
    dict.set(Value::int(4).unwrap(), Value::int(40).unwrap());
    let dict_ptr = Box::into_raw(Box::new(dict));
    let dict_value = Value::object_ptr(dict_ptr as *const ());
    let views = [
        (
            DictViewObject::new(DictViewKind::Keys, dict_value),
            vec![3, 4],
        ),
        (
            DictViewObject::new(DictViewKind::Values, dict_value),
            vec![30, 40],
        ),
    ];

    for (view, expected) in views {
        let view_ptr = Box::into_raw(Box::new(view));
        let view_value = Value::object_ptr(view_ptr as *const ());
        let mut iter = value_to_iterator(&view_value).expect("dict view should be iterable");
        let mut ints = Vec::new();
        while let Some(value) = iter.next() {
            ints.push(value.as_int().expect("dict view should yield ints"));
        }
        ints.sort_unstable();
        assert_eq!(ints, expected);

        unsafe {
            drop(Box::from_raw(view_ptr));
        }
    }

    let items_view_ptr = Box::into_raw(Box::new(DictViewObject::new(
        DictViewKind::Items,
        dict_value,
    )));
    let items_view_value = Value::object_ptr(items_view_ptr as *const ());
    let mut iter =
        value_to_iterator(&items_view_value).expect("dict items view should be iterable");
    let mut pairs = Vec::new();
    while let Some(value) = iter.next() {
        let tuple_ptr = value
            .as_object_ptr()
            .expect("dict items should yield tuple objects");
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        pairs.push((
            tuple.as_slice()[0].as_int().expect("key should be an int"),
            tuple.as_slice()[1]
                .as_int()
                .expect("value should be an int"),
        ));
        unsafe {
            drop(Box::from_raw(tuple_ptr as *mut TupleObject));
        }
    }
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(3, 30), (4, 40)]);

    unsafe {
        drop(Box::from_raw(items_view_ptr));
        drop(Box::from_raw(dict_ptr));
    }
}

#[test]
fn test_iter_dict_view_variants_support_heap_dict_subclass_backing() {
    let mut instance = ShapedObject::new_dict_backed(TypeId::from_raw(600), Shape::empty());
    instance
        .dict_backing_mut()
        .expect("dict backing should exist")
        .set(Value::int(5).unwrap(), Value::int(50).unwrap());
    instance
        .dict_backing_mut()
        .expect("dict backing should exist")
        .set(Value::int(6).unwrap(), Value::int(60).unwrap());

    let instance_ptr = Box::into_raw(Box::new(instance));
    let instance_value = Value::object_ptr(instance_ptr as *const ());
    let views = [
        (
            DictViewObject::new(DictViewKind::Keys, instance_value),
            vec![5, 6],
        ),
        (
            DictViewObject::new(DictViewKind::Values, instance_value),
            vec![50, 60],
        ),
    ];

    for (view, expected) in views {
        let view_ptr = Box::into_raw(Box::new(view));
        let view_value = Value::object_ptr(view_ptr as *const ());
        let mut iter =
            value_to_iterator(&view_value).expect("dict subclass view should be iterable");
        let mut ints = Vec::new();
        while let Some(value) = iter.next() {
            ints.push(
                value
                    .as_int()
                    .expect("dict subclass view should yield ints"),
            );
        }
        ints.sort_unstable();
        assert_eq!(ints, expected);

        unsafe {
            drop(Box::from_raw(view_ptr));
        }
    }

    let items_view_ptr = Box::into_raw(Box::new(DictViewObject::new(
        DictViewKind::Items,
        instance_value,
    )));
    let items_view_value = Value::object_ptr(items_view_ptr as *const ());
    let mut iter =
        value_to_iterator(&items_view_value).expect("dict subclass items view should be iterable");
    let mut pairs = Vec::new();
    while let Some(value) = iter.next() {
        let tuple_ptr = value
            .as_object_ptr()
            .expect("dict items should yield tuple objects");
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        pairs.push((
            tuple.as_slice()[0].as_int().expect("key should be an int"),
            tuple.as_slice()[1]
                .as_int()
                .expect("value should be an int"),
        ));
        unsafe {
            drop(Box::from_raw(tuple_ptr as *mut TupleObject));
        }
    }
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(5, 50), (6, 60)]);

    unsafe {
        drop(Box::from_raw(items_view_ptr));
        drop(Box::from_raw(instance_ptr));
    }
}

#[test]
fn test_iter_mappingproxy_yields_heap_class_keys() {
    let class = std::sync::Arc::new(PyClassObject::new_simple(prism_core::intern::intern(
        "IterProxy",
    )));
    class.set_attr(prism_core::intern::intern("alpha"), Value::int(1).unwrap());
    class.set_attr(prism_core::intern::intern("beta"), Value::int(2).unwrap());

    let proxy_ptr = Box::into_raw(Box::new(MappingProxyObject::for_user_class(
        std::sync::Arc::as_ptr(&class),
    )));
    let proxy_value = Value::object_ptr(proxy_ptr as *const ());

    let mut iter = value_to_iterator(&proxy_value).expect("mappingproxy should be iterable");
    let mut keys = Vec::new();
    while let Some(value) = iter.next() {
        let ptr = value
            .as_string_object_ptr()
            .expect("mappingproxy keys should be interned strings");
        keys.push(
            prism_core::intern::interned_by_ptr(ptr as *const u8)
                .expect("interned string pointer should resolve")
                .as_str()
                .to_string(),
        );
    }
    keys.sort();
    assert_eq!(keys, vec!["alpha".to_string(), "beta".to_string()]);

    unsafe {
        drop(Box::from_raw(proxy_ptr));
    }
}

#[test]
fn test_iter_dict_view_variants_support_mappingproxy_backing() {
    let class = std::sync::Arc::new(PyClassObject::new_simple(prism_core::intern::intern(
        "ProxyBackedViews",
    )));
    class.set_attr(prism_core::intern::intern("token"), Value::int(11).unwrap());
    class.set_attr(prism_core::intern::intern("count"), Value::int(22).unwrap());

    let proxy_ptr = Box::into_raw(Box::new(MappingProxyObject::for_user_class(
        std::sync::Arc::as_ptr(&class),
    )));
    let proxy_value = Value::object_ptr(proxy_ptr as *const ());

    let keys_view_ptr = Box::into_raw(Box::new(DictViewObject::new(
        DictViewKind::Keys,
        proxy_value,
    )));
    let values_view_ptr = Box::into_raw(Box::new(DictViewObject::new(
        DictViewKind::Values,
        proxy_value,
    )));
    let items_view_ptr = Box::into_raw(Box::new(DictViewObject::new(
        DictViewKind::Items,
        proxy_value,
    )));

    let mut key_iter = value_to_iterator(&Value::object_ptr(keys_view_ptr as *const ()))
        .expect("mappingproxy keys view should be iterable");
    let mut keys = Vec::new();
    while let Some(value) = key_iter.next() {
        let ptr = value
            .as_string_object_ptr()
            .expect("mappingproxy keys view should yield strings");
        keys.push(
            prism_core::intern::interned_by_ptr(ptr as *const u8)
                .expect("interned string pointer should resolve")
                .as_str()
                .to_string(),
        );
    }
    keys.sort();
    assert_eq!(keys, vec!["count".to_string(), "token".to_string()]);

    let mut value_iter = value_to_iterator(&Value::object_ptr(values_view_ptr as *const ()))
        .expect("mappingproxy values view should be iterable");
    let mut values = Vec::new();
    while let Some(value) = value_iter.next() {
        values.push(value.as_int().expect("mappingproxy values should be ints"));
    }
    values.sort_unstable();
    assert_eq!(values, vec![11, 22]);

    let mut item_iter = value_to_iterator(&Value::object_ptr(items_view_ptr as *const ()))
        .expect("mappingproxy items view should be iterable");
    let mut pairs = Vec::new();
    while let Some(value) = item_iter.next() {
        let tuple_ptr = value
            .as_object_ptr()
            .expect("mappingproxy items should yield tuple objects");
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        let key_ptr = tuple.as_slice()[0]
            .as_string_object_ptr()
            .expect("tuple key should be a string");
        let key = prism_core::intern::interned_by_ptr(key_ptr as *const u8)
            .expect("interned string pointer should resolve")
            .as_str()
            .to_string();
        let value = tuple.as_slice()[1]
            .as_int()
            .expect("tuple value should be an int");
        pairs.push((key, value));
        unsafe {
            drop(Box::from_raw(tuple_ptr as *mut TupleObject));
        }
    }
    pairs.sort();
    assert_eq!(
        pairs,
        vec![("count".to_string(), 22), ("token".to_string(), 11)]
    );

    unsafe {
        drop(Box::from_raw(keys_view_ptr));
        drop(Box::from_raw(values_view_ptr));
        drop(Box::from_raw(items_view_ptr));
        drop(Box::from_raw(proxy_ptr));
    }
}

// -------------------------------------------------------------------------
// Set Iterator Tests
// -------------------------------------------------------------------------

#[test]
fn test_iter_set_empty() {
    let set = SetObject::new();
    let ptr = Box::leak(Box::new(set)) as *mut SetObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("set should be iterable");
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_set_yields_values() {
    let mut set = SetObject::new();
    set.add(Value::int(10).unwrap());
    set.add(Value::int(20).unwrap());
    set.add(Value::int(30).unwrap());
    let ptr = Box::leak(Box::new(set)) as *mut SetObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("set should be iterable");
    let mut values: Vec<i64> = Vec::new();
    while let Some(v) = iter.next() {
        values.push(v.as_int().unwrap());
    }
    values.sort();
    assert_eq!(values, vec![10, 20, 30]);
}

// -------------------------------------------------------------------------
// Bytes Iterator Tests
// -------------------------------------------------------------------------

#[test]
fn test_iter_bytes_yields_ints() {
    let bytes = BytesObject::from_slice(&[0, 65, 255]);
    let ptr = Box::leak(Box::new(bytes)) as *mut BytesObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("bytes should be iterable");
    assert_eq!(iter.next().unwrap().as_int(), Some(0));
    assert_eq!(iter.next().unwrap().as_int(), Some(65));
    assert_eq!(iter.next().unwrap().as_int(), Some(255));
    assert!(iter.next().is_none());
}

#[test]
fn test_iter_bytearray_yields_ints() {
    let bytearray = BytesObject::bytearray_from_slice(&[1, 2, 3]);
    let ptr = Box::leak(Box::new(bytearray)) as *mut BytesObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).expect("bytearray should be iterable");
    assert_eq!(iter.next().unwrap().as_int(), Some(1));
    assert_eq!(iter.next().unwrap().as_int(), Some(2));
    assert_eq!(iter.next().unwrap().as_int(), Some(3));
    assert!(iter.next().is_none());
}

// -------------------------------------------------------------------------
// Iterator-to-Value Round Trip Tests
// -------------------------------------------------------------------------

#[test]
fn test_iterator_to_value_and_back() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let list_value = Value::object_ptr(ptr);

    let iter = value_to_iterator(&list_value).unwrap();
    let iter_value = iterator_to_value(iter);

    // Verify we can get the iterator back
    let iter_obj = get_iterator_mut(&iter_value);
    assert!(iter_obj.is_some());
}

#[test]
fn test_value_to_iterator_accepts_iterator_values() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let list_value = Value::object_ptr(ptr);

    let iter_value = iterator_to_value(value_to_iterator(&list_value).unwrap());
    let mut proxy = value_to_iterator(&iter_value).expect("iterator values should remain iterable");

    assert_eq!(proxy.next().unwrap().as_int(), Some(1));
    assert_eq!(proxy.next().unwrap().as_int(), Some(2));

    let underlying = get_iterator_mut(&iter_value).expect("iterator value should remain mutable");
    assert_eq!(underlying.next().unwrap().as_int(), Some(3));
    assert!(proxy.next().is_none());
}

#[test]
fn test_is_iterator() {
    let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let list_value = Value::object_ptr(ptr);

    // List is not an iterator
    assert!(!is_iterator(&list_value));

    // Convert to iterator
    let iter = value_to_iterator(&list_value).unwrap();
    let iter_value = iterator_to_value(iter);

    // Now it's an iterator
    assert!(is_iterator(&iter_value));
}

// -------------------------------------------------------------------------
// Size Hint Tests
// -------------------------------------------------------------------------

#[test]
fn test_iter_size_hint_list() {
    let list = ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]);
    let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
    let value = Value::object_ptr(ptr);

    let mut iter = value_to_iterator(&value).unwrap();
    assert_eq!(iter.size_hint(), Some(3));
    iter.next();
    assert_eq!(iter.size_hint(), Some(2));
    iter.next();
    iter.next();
    assert_eq!(iter.size_hint(), Some(0));
}

#[test]
fn test_iter_size_hint_range() {
    let range = RangeObject::from_stop(100);
    let ptr = Box::leak(Box::new(range)) as *mut RangeObject as *const ();
    let value = Value::object_ptr(ptr);

    let iter = value_to_iterator(&value).unwrap();
    assert_eq!(iter.size_hint(), Some(100));
}

// -------------------------------------------------------------------------
// Error Message Tests
// -------------------------------------------------------------------------

#[test]
fn test_iter_error_not_iterable_message() {
    let err = IterError::NotIterable("NoneType".into());
    assert_eq!(err.to_string(), "'NoneType' object is not iterable");
}

#[test]
fn test_iter_error_invalid_object() {
    let err = IterError::InvalidObject;
    assert_eq!(err.to_string(), "invalid object reference");
}

#[test]
fn test_iter_error_into_builtin_error() {
    let err = IterError::NotIterable("int".into());
    let builtin_err: BuiltinError = err.into();
    match builtin_err {
        BuiltinError::TypeError(msg) => {
            assert!(msg.contains("int"));
            assert!(msg.contains("not iterable"));
        }
        _ => panic!("Expected TypeError"),
    }
}

// -------------------------------------------------------------------------
// Type Name Tests
// -------------------------------------------------------------------------

#[test]
fn test_get_value_type_name() {
    assert_eq!(get_value_type_name(&Value::none()), "NoneType");
    assert_eq!(get_value_type_name(&Value::bool(true)), "bool");
    assert_eq!(get_value_type_name(&Value::int(1).unwrap()), "int");
    assert_eq!(get_value_type_name(&Value::float(1.0)), "float");
}
