use super::*;
use prism_core::intern::intern;

#[test]
fn test_slot_descriptor_creation() {
    let name = intern("x");
    let slot = SlotDescriptor::read_write(name.clone(), 0, 16);

    assert_eq!(slot.name(), &name);
    assert_eq!(slot.index(), 0);
    assert_eq!(slot.offset(), 16);
    assert!(slot.is_readable());
    assert!(slot.is_writable());
}

#[test]
fn test_slot_descriptor_read_only() {
    let name = intern("readonly");
    let slot = SlotDescriptor::read_only(name, 0, 16);

    assert!(slot.is_readable());
    assert!(!slot.is_writable());
    assert_eq!(slot.access(), SlotAccess::ReadOnly);
}

#[test]
fn test_slot_descriptor_kind() {
    let slot = SlotDescriptor::read_write(intern("x"), 0, 16);
    assert_eq!(slot.kind(), DescriptorKind::Slot);
}

#[test]
fn test_slot_descriptor_flags() {
    let slot = SlotDescriptor::read_write(intern("x"), 0, 16);
    let flags = slot.flags();

    assert!(flags.contains(DescriptorFlags::SLOT));
    assert!(flags.contains(DescriptorFlags::DATA_DESCRIPTOR));
    assert!(flags.contains(DescriptorFlags::HAS_GET));
    assert!(flags.contains(DescriptorFlags::HAS_SET));
}

#[test]
fn test_slot_descriptor_read_only_flags() {
    let slot = SlotDescriptor::read_only(intern("x"), 0, 16);
    let flags = slot.flags();

    assert!(flags.contains(DescriptorFlags::HAS_GET));
    assert!(!flags.contains(DescriptorFlags::HAS_SET));
}

#[test]
fn test_compute_offset() {
    // First slot should be at offset 16 (after ObjectHeader)
    assert_eq!(SlotDescriptor::compute_offset(0), 16);
    // Second slot at 24
    assert_eq!(SlotDescriptor::compute_offset(1), 24);
    // Third slot at 32
    assert_eq!(SlotDescriptor::compute_offset(2), 32);
}

#[test]
fn test_slot_collection_creation() {
    let names = vec![intern("x"), intern("y"), intern("z")];
    let collection = SlotCollection::from_names(&names);

    assert_eq!(collection.len(), 3);
    assert!(!collection.is_empty());
}

#[test]
fn test_slot_collection_get_by_name() {
    let names = vec![intern("x"), intern("y")];
    let collection = SlotCollection::from_names(&names);

    let x_slot = collection.get_by_name(&intern("x"));
    assert!(x_slot.is_some());
    assert_eq!(x_slot.unwrap().index(), 0);

    let y_slot = collection.get_by_name(&intern("y"));
    assert!(y_slot.is_some());
    assert_eq!(y_slot.unwrap().index(), 1);

    let z_slot = collection.get_by_name(&intern("z"));
    assert!(z_slot.is_none());
}

#[test]
fn test_slot_collection_offsets() {
    let names = vec![intern("a"), intern("b"), intern("c")];
    let collection = SlotCollection::from_names(&names);

    assert_eq!(collection.get(0).unwrap().offset(), 16);
    assert_eq!(collection.get(1).unwrap().offset(), 24);
    assert_eq!(collection.get(2).unwrap().offset(), 32);
}

#[test]
fn test_slot_collection_total_size() {
    let names = vec![intern("x"), intern("y")];
    let collection = SlotCollection::from_names(&names);

    // 2 slots * 8 bytes = 16 bytes, starting at offset 16
    // So total size should be 16 + 8 = 24 for last slot + 8 = 32
    assert_eq!(collection.total_size(), 32);
}

#[test]
fn test_slot_set_readonly_error() {
    let slot = SlotDescriptor::read_only(intern("x"), 0, 16);
    let result = slot.set(Value::int_unchecked(0), Value::int_unchecked(42));
    assert!(result.is_err());
}

#[test]
fn test_slot_get_writeonly_error() {
    let slot = SlotDescriptor::new(intern("x"), 0, 16, SlotAccess::WriteOnly);
    let result = slot.get(Some(Value::int_unchecked(0)), Value::none());
    assert!(result.is_err());
}

#[test]
fn test_slot_class_access() {
    let slot = SlotDescriptor::read_write(intern("x"), 0, 16);
    // When accessed through class (obj=None), should return descriptor
    let result = slot.get(None, Value::none());
    assert!(result.is_ok());
}

#[test]
fn test_slot_collection_iter() {
    let names = vec![intern("a"), intern("b")];
    let collection = SlotCollection::from_names(&names);

    let indices: Vec<u16> = collection.iter().map(|s| s.index()).collect();
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn test_slot_add_slot() {
    let mut collection = SlotCollection::new();

    let idx1 = collection.add_slot(intern("first"), SlotAccess::ReadWrite);
    assert_eq!(idx1, 0);

    let idx2 = collection.add_slot(intern("second"), SlotAccess::ReadOnly);
    assert_eq!(idx2, 1);

    assert_eq!(collection.len(), 2);
    assert!(collection.get(1).unwrap().is_readable());
    assert!(!collection.get(1).unwrap().is_writable());
}
