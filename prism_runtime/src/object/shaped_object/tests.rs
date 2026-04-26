use super::*;
use crate::object::shape::ShapeRegistry;
use num_bigint::BigInt;

fn intern(s: &str) -> InternedString {
    prism_core::intern::intern(s)
}

fn val(i: i64) -> Value {
    Value::int(i).unwrap()
}

// -------------------------------------------------------------------------
// InlineSlots Tests
// -------------------------------------------------------------------------

#[test]
fn test_inline_slots_new() {
    let slots = InlineSlots::new();
    assert_eq!(slots.used(), 0);
}

#[test]
fn test_inline_slots_get_set() {
    let mut slots = InlineSlots::new();
    slots.set(0, val(42));
    assert_eq!(slots.get(0), val(42));
    assert_eq!(slots.used(), 1);
}

#[test]
fn test_inline_slots_multiple() {
    let mut slots = InlineSlots::new();
    slots.set(0, val(1));
    slots.set(1, val(2));
    slots.set(2, val(3));

    assert_eq!(slots.get(0), val(1));
    assert_eq!(slots.get(1), val(2));
    assert_eq!(slots.get(2), val(3));
    assert_eq!(slots.used(), 3);
}

#[test]
fn test_inline_slots_overwrite() {
    let mut slots = InlineSlots::new();
    slots.set(0, val(1));
    slots.set(0, val(2));
    assert_eq!(slots.get(0), val(2));
}

#[test]
fn test_new_dict_backed_allocates_native_mapping_storage() {
    let mut object = ShapedObject::new_dict_backed(TypeId::from_raw(512), Shape::empty());
    assert!(object.has_dict_backing());

    let key = Value::string(intern("member"));
    let value = val(7);
    object
        .dict_backing_mut()
        .expect("dict backing should exist")
        .set(key, value);

    assert_eq!(
        object
            .dict_backing()
            .expect("dict backing should exist")
            .get(key),
        Some(value)
    );
}

#[test]
fn test_instance_dict_materializes_shape_properties_and_is_stable() {
    let registry = ShapeRegistry::new();
    let mut object = ShapedObject::new(TypeId::from_raw(612), registry.empty_shape());
    object.set_property(intern("alpha"), val(11), &registry);
    object.set_property(intern("beta"), val(22), &registry);

    assert!(!object.has_instance_dict());
    let dict_value = object.ensure_instance_dict_value();
    assert!(object.has_instance_dict());
    assert_eq!(object.ensure_instance_dict_value(), dict_value);

    let dict_ptr = dict_value
        .as_object_ptr()
        .expect("instance dict should be an object");
    let dict = unsafe { &*(dict_ptr as *const DictObject) };
    assert_eq!(dict.get(Value::string(intern("alpha"))), Some(val(11)));
    assert_eq!(dict.get(Value::string(intern("beta"))), Some(val(22)));
}

#[test]
fn test_instance_dict_can_alias_external_mapping() {
    let mut object = ShapedObject::new(TypeId::from_raw(613), Shape::empty());
    let mut dict = Box::new(DictObject::new());
    dict.set(Value::string(intern("external")), val(33));
    let dict_ptr = Box::into_raw(dict);
    let dict_value = Value::object_ptr(dict_ptr as *const ());

    object.set_instance_dict_value(dict_value);
    assert_eq!(object.instance_dict_value(), Some(dict_value));
    assert_eq!(
        unsafe { &*dict_ptr }.get(Value::string(intern("external"))),
        Some(val(33))
    );

    unsafe { drop(Box::from_raw(dict_ptr)) };
}

#[test]
fn test_new_list_backed_allocates_native_sequence_storage() {
    let mut object = ShapedObject::new_list_backed(TypeId::from_raw(513), Shape::empty());
    assert!(object.has_list_backing());

    let value = val(11);
    object
        .list_backing_mut()
        .expect("list backing should exist")
        .push(value);

    assert_eq!(
        object
            .list_backing()
            .expect("list backing should exist")
            .get(0),
        Some(value)
    );
}

#[test]
fn test_new_tuple_backed_preserves_native_tuple_storage() {
    let mut object = ShapedObject::new_tuple_backed(
        TypeId::OBJECT,
        Shape::empty(),
        TupleObject::from_slice(&[val(3), val(5)]),
    );

    assert!(object.has_tuple_backing());
    let tuple = object.tuple_backing().expect("tuple backing should exist");
    assert_eq!(tuple.len(), 2);
    assert_eq!(tuple.get(0), Some(val(3)));
    assert_eq!(tuple.get(1), Some(val(5)));

    object.set_tuple_backing(TupleObject::from_slice(&[val(8)]));
    let tuple = object.tuple_backing().expect("tuple backing should exist");
    assert_eq!(tuple.len(), 1);
    assert_eq!(tuple.get(0), Some(val(8)));
}

#[test]
fn test_new_string_backed_preserves_native_string_storage() {
    let object = ShapedObject::new_string_backed(
        TypeId::from_raw(514),
        Shape::empty(),
        StringObject::new("value"),
    );
    assert!(object.has_string_backing());
    assert_eq!(
        object
            .string_backing()
            .expect("string backing should exist")
            .as_str(),
        "value"
    );
}

#[test]
fn test_new_bytes_backed_preserves_native_byte_storage() {
    let object = ShapedObject::new_bytes_backed(
        TypeId::from_raw(515),
        Shape::empty(),
        BytesObject::from_slice(b"value"),
    );

    assert!(object.has_bytes_backing());
    assert_eq!(
        object
            .bytes_backing()
            .expect("bytes backing should exist")
            .as_bytes(),
        b"value"
    );
}

#[test]
fn test_new_int_backed_preserves_native_integer_storage() {
    let integer = BigInt::from(1_u8) << 90_u32;
    let object =
        ShapedObject::new_int_backed(TypeId::from_raw(515), Shape::empty(), integer.clone());

    assert!(object.has_int_backing());
    assert_eq!(
        object.int_backing().expect("integer backing should exist"),
        &integer
    );
}

#[test]
fn test_inline_slots_iter() {
    let mut slots = InlineSlots::new();
    slots.set(0, val(10));
    slots.set(1, val(20));

    let collected: Vec<_> = slots.iter().collect();
    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0], (0, &val(10)));
    assert_eq!(collected[1], (1, &val(20)));
}

// -------------------------------------------------------------------------
// OverflowStorage Tests
// -------------------------------------------------------------------------

#[test]
fn test_overflow_new() {
    let overflow = OverflowStorage::new();
    assert!(overflow.is_empty());
    assert_eq!(overflow.len(), 0);
}

#[test]
fn test_overflow_get_set() {
    let mut overflow = OverflowStorage::new();
    let name = intern("prop");
    overflow.set(name.clone(), val(42));

    assert!(overflow.contains(&name));
    assert_eq!(overflow.get(&name), Some(&val(42)));
    assert_eq!(overflow.len(), 1);
}

#[test]
fn test_overflow_remove() {
    let mut overflow = OverflowStorage::new();
    let name = intern("prop");
    overflow.set(name.clone(), val(42));

    let removed = overflow.remove(&name);
    assert_eq!(removed, Some(val(42)));
    assert!(!overflow.contains(&name));
}

// -------------------------------------------------------------------------
// ShapedObject Tests - Basic
// -------------------------------------------------------------------------

#[test]
fn test_shaped_object_new() {
    let registry = ShapeRegistry::new();
    let obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    assert!(obj.shape().is_empty());
    assert!(obj.is_inline_only());
    assert_eq!(obj.property_count(), 0);
}

#[test]
fn test_shaped_object_set_get() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("x"), val(10), &registry);
    assert_eq!(obj.get_property("x"), Some(val(10)));
}

#[test]
fn test_shaped_object_multiple_properties() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("a"), val(1), &registry);
    obj.set_property(intern("b"), val(2), &registry);
    obj.set_property(intern("c"), val(3), &registry);

    assert_eq!(obj.get_property("a"), Some(val(1)));
    assert_eq!(obj.get_property("b"), Some(val(2)));
    assert_eq!(obj.get_property("c"), Some(val(3)));
    assert_eq!(obj.property_count(), 3);
}

#[test]
fn test_shaped_object_update_property() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("x"), val(10), &registry);
    let shape_before = obj.shape_id();

    obj.set_property(intern("x"), val(20), &registry);
    let shape_after = obj.shape_id();

    // Shape should not change when updating existing property
    assert_eq!(shape_before, shape_after);
    assert_eq!(obj.get_property("x"), Some(val(20)));
}

#[test]
fn test_shaped_object_shape_transitions() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    let shape0 = obj.shape_id();
    obj.set_property(intern("x"), val(1), &registry);
    let shape1 = obj.shape_id();
    obj.set_property(intern("y"), val(2), &registry);
    let shape2 = obj.shape_id();

    // Each new property should create a shape transition
    assert_ne!(shape0, shape1);
    assert_ne!(shape1, shape2);
}

// -------------------------------------------------------------------------
// ShapedObject Tests - Inline Cache Path
// -------------------------------------------------------------------------

#[test]
fn test_shaped_object_cached_access() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("x"), val(42), &registry);

    // Simulate IC: lookup slot once, then use cached access
    let slot_index = obj.shape().lookup("x").unwrap();
    let cached_value = obj.get_property_cached(slot_index);

    assert_eq!(cached_value, val(42));
}

#[test]
fn test_shaped_object_cached_set() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("x"), val(1), &registry);
    let slot_index = obj.shape().lookup("x").unwrap();

    // Cached set (IC fast path)
    obj.set_property_cached(slot_index, val(99));

    assert_eq!(obj.get_property("x"), Some(val(99)));
}

#[test]
fn test_shaped_object_interned_access() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    let name = intern("prop");
    obj.set_property(name.clone(), val(100), &registry);

    assert_eq!(obj.get_property_interned(&name), Some(val(100)));
}

// -------------------------------------------------------------------------
// ShapedObject Tests - Property Operations
// -------------------------------------------------------------------------

#[test]
fn test_shaped_object_has_property() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    assert!(!obj.has_property("x"));
    obj.set_property(intern("x"), val(1), &registry);
    assert!(obj.has_property("x"));
}

#[test]
fn test_shaped_object_delete_property() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("x"), val(1), &registry);
    assert!(obj.delete_property("x"));

    // Attribute must be absent after deletion.
    assert_eq!(obj.get_property("x"), None);
    assert!(!obj.has_property("x"));
}

#[test]
fn test_shaped_object_delete_nonexistent() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    assert!(!obj.delete_property("nonexistent"));
}

#[test]
fn test_shaped_object_delete_twice_returns_false_second_time() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("x"), val(1), &registry);
    assert!(obj.delete_property("x"));
    assert!(!obj.delete_property("x"));
}

#[test]
fn test_shaped_object_none_value_is_not_deleted() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("x"), Value::none(), &registry);
    assert!(obj.has_property("x"));
    assert_eq!(obj.get_property("x"), Some(Value::none()));
}

#[test]
fn test_shaped_object_property_names() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("first"), val(1), &registry);
    obj.set_property(intern("second"), val(2), &registry);
    obj.set_property(intern("third"), val(3), &registry);

    let names = obj.property_names();
    assert_eq!(names.len(), 3);
    assert_eq!(names[0].as_str(), "first");
    assert_eq!(names[1].as_str(), "second");
    assert_eq!(names[2].as_str(), "third");
}

#[test]
fn test_shaped_object_property_names_exclude_deleted() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("first"), val(1), &registry);
    obj.set_property(intern("second"), val(2), &registry);
    assert!(obj.delete_property("first"));

    let names = obj.property_names();
    assert_eq!(names.len(), 1);
    assert_eq!(names[0].as_str(), "second");
    assert_eq!(obj.property_count(), 1);
}

// -------------------------------------------------------------------------
// ShapedObject Tests - Overflow Storage
// -------------------------------------------------------------------------

#[test]
fn test_shaped_object_inline_limit() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    // Fill inline slots
    for i in 0..MAX_INLINE_SLOTS {
        obj.set_property(intern(&format!("prop{}", i)), val(i as i64), &registry);
    }

    assert!(obj.is_inline_only());

    // Add one more to trigger overflow
    obj.set_property(
        intern(&format!("prop{}", MAX_INLINE_SLOTS)),
        val(MAX_INLINE_SLOTS as i64),
        &registry,
    );

    assert!(!obj.is_inline_only());
}

#[test]
fn test_shaped_object_overflow_access() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    // Fill inline + overflow
    for i in 0..(MAX_INLINE_SLOTS + 3) {
        obj.set_property(intern(&format!("prop{}", i)), val(i as i64), &registry);
    }

    // Verify all properties accessible
    for i in 0..(MAX_INLINE_SLOTS + 3) {
        assert_eq!(
            obj.get_property(&format!("prop{}", i)),
            Some(val(i as i64)),
            "Failed to get prop{}",
            i
        );
    }
}

#[test]
fn test_shaped_object_readd_deleted_overflow_property_reuses_shape() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    for i in 0..=MAX_INLINE_SLOTS {
        obj.set_property(intern(&format!("prop{}", i)), val(i as i64), &registry);
    }

    let overflow_name = format!("prop{}", MAX_INLINE_SLOTS);
    assert!(obj.delete_property(&overflow_name));
    assert_eq!(obj.get_property(&overflow_name), None);
    let shape_before = obj.shape_id();

    obj.set_property(intern(&overflow_name), val(999), &registry);
    let shape_after = obj.shape_id();

    assert_eq!(shape_before, shape_after);
    assert_eq!(obj.get_property(&overflow_name), Some(val(999)));
}

// -------------------------------------------------------------------------
// ShapedObject Tests - Shape Sharing
// -------------------------------------------------------------------------

#[test]
fn test_shaped_object_shape_sharing() {
    let registry = ShapeRegistry::new();
    let mut obj1 = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
    let mut obj2 = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    // Add same properties in same order
    obj1.set_property(intern("x"), val(1), &registry);
    obj1.set_property(intern("y"), val(2), &registry);

    obj2.set_property(intern("x"), val(10), &registry);
    obj2.set_property(intern("y"), val(20), &registry);

    // Should share the same shape
    assert_eq!(obj1.shape_id(), obj2.shape_id());

    // But have different values
    assert_eq!(obj1.get_property("x"), Some(val(1)));
    assert_eq!(obj2.get_property("x"), Some(val(10)));
}

#[test]
fn test_shaped_object_different_order_different_shape() {
    let registry = ShapeRegistry::new();
    let mut obj1 = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
    let mut obj2 = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    // Add same properties but different order
    obj1.set_property(intern("x"), val(1), &registry);
    obj1.set_property(intern("y"), val(2), &registry);

    obj2.set_property(intern("y"), val(2), &registry);
    obj2.set_property(intern("x"), val(1), &registry);

    // Different shapes due to different order
    assert_ne!(obj1.shape_id(), obj2.shape_id());

    // But both properties work
    assert_eq!(obj1.get_property("x"), Some(val(1)));
    assert_eq!(obj2.get_property("x"), Some(val(1)));
}

// -------------------------------------------------------------------------
// ShapedObject Tests - Property Flags
// -------------------------------------------------------------------------

#[test]
fn test_shaped_object_property_flags() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property_with_flags(
        intern("readonly"),
        val(42),
        PropertyFlags::read_only(),
        &registry,
    );

    // Property exists
    assert_eq!(obj.get_property("readonly"), Some(val(42)));

    // Check flags via shape
    let desc = obj.shape().get_descriptor("readonly").unwrap();
    assert!(!desc.is_writable());
    assert!(desc.is_enumerable());
}

#[test]
fn test_shaped_object_hidden_property() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property_with_flags(
        intern("_internal"),
        val(99),
        PropertyFlags::hidden(),
        &registry,
    );

    let desc = obj.shape().get_descriptor("_internal").unwrap();
    assert!(!desc.is_enumerable());
    assert!(desc.is_writable());
}

#[test]
fn test_shaped_object_same_property_name_different_flags_gets_distinct_shapes() {
    let registry = ShapeRegistry::new();
    let mut writable = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
    let mut read_only = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    writable.set_property(intern("shared"), val(1), &registry);
    read_only.set_property_with_flags(
        intern("shared"),
        val(2),
        PropertyFlags::read_only(),
        &registry,
    );

    assert_ne!(writable.shape_id(), read_only.shape_id());
    assert!(
        writable
            .shape()
            .get_descriptor("shared")
            .expect("writable descriptor missing")
            .is_writable()
    );
    assert!(
        !read_only
            .shape()
            .get_descriptor("shared")
            .expect("read-only descriptor missing")
            .is_writable()
    );
}

// -------------------------------------------------------------------------
// ShapedObject Tests - Iterator
// -------------------------------------------------------------------------

#[test]
fn test_shaped_object_iter_properties() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("a"), val(1), &registry);
    obj.set_property(intern("b"), val(2), &registry);

    let props: Vec<_> = obj.iter_properties().collect();
    assert_eq!(props.len(), 2);
}

// -------------------------------------------------------------------------
// ShapedObject Tests - Type Integration
// -------------------------------------------------------------------------

#[test]
fn test_shaped_object_type_id() {
    let registry = ShapeRegistry::new();
    let obj = ShapedObject::new(TypeId::DICT, registry.empty_shape());

    assert_eq!(obj.header().type_id, TypeId::DICT);
}

#[test]
fn test_shaped_object_pyobject_trait() {
    let registry = ShapeRegistry::new();
    let obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    // Test PyObject trait
    assert_eq!(obj.type_id(), TypeId::OBJECT);
}

// -------------------------------------------------------------------------
// Edge Cases
// -------------------------------------------------------------------------

#[test]
fn test_shaped_object_empty_property_name() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern(""), val(42), &registry);
    assert_eq!(obj.get_property(""), Some(val(42)));
}

#[test]
fn test_shaped_object_unicode_property() {
    let registry = ShapeRegistry::new();
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    obj.set_property(intern("名前"), val(1), &registry);
    obj.set_property(intern("🚀"), val(2), &registry);

    assert_eq!(obj.get_property("名前"), Some(val(1)));
    assert_eq!(obj.get_property("🚀"), Some(val(2)));
}

#[test]
fn test_shaped_object_nonexistent_property() {
    let registry = ShapeRegistry::new();
    let obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

    assert_eq!(obj.get_property("nonexistent"), None);
}
