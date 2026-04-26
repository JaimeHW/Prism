use super::*;
use prism_core::intern::intern;

// =========================================================================
// Construction Tests
// =========================================================================

#[test]
fn test_instance_new() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let instance = PyInstanceObject::new(class_id, type_id);

    assert_eq!(instance.class_id(), class_id);
    assert_eq!(instance.shape_id(), EMPTY_SHAPE_ID);
    assert!(!instance.has_overflow());
}

#[test]
fn test_instance_new_with_overflow() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let instance = PyInstanceObject::new_with_overflow(class_id, type_id);

    assert_eq!(instance.class_id(), class_id);
    assert!(instance.has_overflow());
}

#[test]
fn test_instance_new_slotted_small() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let instance = PyInstanceObject::new_slotted(class_id, type_id, 3);

    assert!(!instance.has_overflow());
}

#[test]
fn test_instance_new_slotted_large() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let instance = PyInstanceObject::new_slotted(class_id, type_id, 8);

    assert!(instance.has_overflow());
}

// =========================================================================
// Inline Slot Tests
// =========================================================================

#[test]
fn test_inline_slot_access() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new(class_id, type_id);

    // Set and get inline slots
    instance.set_inline_slot(0, Value::int_unchecked(42));
    instance.set_inline_slot(1, Value::int_unchecked(100));

    assert_eq!(instance.get_inline_slot(0).as_int().unwrap(), 42);
    assert_eq!(instance.get_inline_slot(1).as_int().unwrap(), 100);
}

#[test]
fn test_inline_slot_checked() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new(class_id, type_id);

    assert!(instance.set_inline_slot_checked(0, Value::int_unchecked(1)));
    assert!(instance.set_inline_slot_checked(3, Value::int_unchecked(4)));
    assert!(!instance.set_inline_slot_checked(4, Value::int_unchecked(5)));

    assert!(instance.get_inline_slot_checked(0).is_some());
    assert!(instance.get_inline_slot_checked(4).is_none());
}

#[test]
fn test_all_inline_slots() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new(class_id, type_id);

    for i in 0..INLINE_SLOT_COUNT {
        instance.set_inline_slot(i, Value::int_unchecked(i as i64));
    }

    for i in 0..INLINE_SLOT_COUNT {
        assert_eq!(instance.get_inline_slot(i).as_int().unwrap(), i as i64);
    }
}

// =========================================================================
// Overflow Storage Tests
// =========================================================================

#[test]
fn test_ensure_overflow() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new(class_id, type_id);

    assert!(!instance.has_overflow());
    instance.ensure_overflow();
    assert!(instance.has_overflow());
}

#[test]
fn test_overflow_slot_access() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new_with_overflow(class_id, type_id);

    assert!(instance.set_overflow_slot(0, Value::int_unchecked(99)));
    assert_eq!(instance.get_overflow_slot(0).unwrap().as_int().unwrap(), 99);
}

#[test]
fn test_unified_attr_access() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new_with_overflow(class_id, type_id);

    // Inline slots (offset 0-3)
    for i in 0..INLINE_SLOT_COUNT {
        assert!(instance.set_attr_by_offset(i, Value::int_unchecked(i as i64)));
    }

    // Overflow slots (offset 4+)
    for i in 0..4 {
        let offset = INLINE_SLOT_COUNT + i;
        assert!(instance.set_attr_by_offset(offset, Value::int_unchecked(100 + i as i64)));
    }

    // Verify inline
    for i in 0..INLINE_SLOT_COUNT {
        assert_eq!(
            instance.get_attr_by_offset(i).unwrap().as_int().unwrap(),
            i as i64
        );
    }

    // Verify overflow
    for i in 0..4 {
        let offset = INLINE_SLOT_COUNT + i;
        assert_eq!(
            instance
                .get_attr_by_offset(offset)
                .unwrap()
                .as_int()
                .unwrap(),
            100 + i as i64
        );
    }
}

// =========================================================================
// Shape ID Tests
// =========================================================================

#[test]
fn test_shape_id() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new(class_id, type_id);

    assert_eq!(instance.shape_id(), EMPTY_SHAPE_ID);

    let new_shape = allocate_shape_id();
    instance.set_shape_id(new_shape);
    assert_eq!(instance.shape_id(), new_shape);
}

#[test]
fn test_shape_id_allocation() {
    let id1 = allocate_shape_id();
    let id2 = allocate_shape_id();
    let id3 = allocate_shape_id();

    assert!(id1 < id2);
    assert!(id2 < id3);
}

// =========================================================================
// __dict__ Tests
// =========================================================================

#[test]
fn test_dict_creation() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new(class_id, type_id);

    assert!(!instance.has_dict());

    let dict = instance.get_or_create_dict();
    dict.insert(intern("x"), Value::int_unchecked(1));

    assert!(instance.has_dict());
}

#[test]
fn test_dict_access() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new(class_id, type_id);

    let dict = instance.get_or_create_dict();
    dict.insert(intern("foo"), Value::int_unchecked(42));
    dict.insert(intern("bar"), Value::int_unchecked(99));

    let read_dict = instance.get_dict().unwrap();
    assert_eq!(read_dict.get(&intern("foo")).unwrap().as_int().unwrap(), 42);
    assert_eq!(read_dict.get(&intern("bar")).unwrap().as_int().unwrap(), 99);
}

// =========================================================================
// OverflowStorage Tests
// =========================================================================

#[test]
fn test_overflow_storage_slots() {
    let mut storage = OverflowStorage::new_slots();

    assert!(storage.set_slot(0, Value::int_unchecked(10)));
    assert!(storage.set_slot(5, Value::int_unchecked(50)));

    assert_eq!(storage.get_slot(0).unwrap().as_int().unwrap(), 10);
    assert_eq!(storage.get_slot(5).unwrap().as_int().unwrap(), 50);
}

#[test]
fn test_overflow_storage_dict() {
    let mut storage = OverflowStorage::new_dict();
    let name = intern("attr");

    storage.set_dict(name.clone(), Value::int_unchecked(123));
    assert_eq!(storage.get_dict(&name).unwrap().as_int().unwrap(), 123);
}

#[test]
fn test_overflow_storage_clone() {
    let mut storage = OverflowStorage::new_slots();
    storage.set_slot(0, Value::int_unchecked(42));

    let cloned = storage.clone();
    assert_eq!(cloned.get_slot(0).unwrap().as_int().unwrap(), 42);
}

// =========================================================================
// InstanceOverflow Tests
// =========================================================================

#[test]
fn test_instance_overflow_new_dict() {
    let overflow = InstanceOverflow::new_dict();
    // Should have dict-based storage
    assert!(matches!(overflow.storage, OverflowStorage::Dict(_)));
}

// =========================================================================
// PyObject Trait Tests
// =========================================================================

#[test]
fn test_pyobject_trait() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let instance = PyInstanceObject::new(class_id, type_id);

    assert_eq!(instance.header().type_id, type_id);
}

// =========================================================================
// Specialization Hint Tests
// =========================================================================

#[test]
fn test_spec_hint_inline_only() {
    let hint = InstanceSpecHint::from_attr_count(3, false);
    assert_eq!(hint, InstanceSpecHint::InlineOnly);
}

#[test]
fn test_spec_hint_fixed_slots() {
    let hint = InstanceSpecHint::from_attr_count(10, true);
    assert_eq!(hint, InstanceSpecHint::FixedSlots);
}

#[test]
fn test_spec_hint_dynamic() {
    let hint = InstanceSpecHint::from_attr_count(5, false);
    assert_eq!(hint, InstanceSpecHint::Dynamic);
}

// =========================================================================
// Memory Layout Tests
// =========================================================================

#[test]
fn test_instance_size() {
    let size = std::mem::size_of::<PyInstanceObject>();
    assert_eq!(size, 64, "PyInstanceObject should be 64 bytes (cache line)");
}

#[test]
fn test_instance_alignment() {
    let align = std::mem::align_of::<PyInstanceObject>();
    assert_eq!(align, 64, "PyInstanceObject should be 64-byte aligned");
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_inline_slot_default_none() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let instance = PyInstanceObject::new(class_id, type_id);

    for i in 0..INLINE_SLOT_COUNT {
        assert!(instance.get_inline_slot(i).is_none());
    }
}

#[test]
fn test_overflow_boundary() {
    let class_id = ClassId(100);
    let type_id = TypeId::from_raw(100);
    let mut instance = PyInstanceObject::new_with_overflow(class_id, type_id);

    // Set value at inline boundary
    assert!(instance.set_attr_by_offset(INLINE_SLOT_COUNT - 1, Value::int_unchecked(1)));
    // Set value at overflow boundary
    assert!(instance.set_attr_by_offset(INLINE_SLOT_COUNT, Value::int_unchecked(2)));

    assert_eq!(
        instance
            .get_attr_by_offset(INLINE_SLOT_COUNT - 1)
            .unwrap()
            .as_int()
            .unwrap(),
        1
    );
    assert_eq!(
        instance
            .get_attr_by_offset(INLINE_SLOT_COUNT)
            .unwrap()
            .as_int()
            .unwrap(),
        2
    );
}
