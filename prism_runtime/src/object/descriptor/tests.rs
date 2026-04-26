use super::*;

#[test]
fn test_descriptor_flags_data_descriptor() {
    let flags = DescriptorFlags::HAS_GET | DescriptorFlags::HAS_SET;
    assert!(flags.contains(DescriptorFlags::HAS_GET));
    assert!(flags.contains(DescriptorFlags::HAS_SET));
    assert!(flags.intersects(DescriptorFlags::HAS_SET | DescriptorFlags::HAS_DELETE));
}

#[test]
fn test_descriptor_flags_non_data_descriptor() {
    let flags = DescriptorFlags::HAS_GET;
    assert!(flags.contains(DescriptorFlags::HAS_GET));
    assert!(!flags.contains(DescriptorFlags::HAS_SET));
    assert!(!flags.intersects(DescriptorFlags::HAS_SET | DescriptorFlags::HAS_DELETE));
}

#[test]
fn test_descriptor_kind_is_data() {
    assert!(DescriptorKind::Property.is_data_descriptor());
    assert!(DescriptorKind::Slot.is_data_descriptor());
    assert!(!DescriptorKind::Method.is_data_descriptor());
    assert!(!DescriptorKind::StaticMethod.is_data_descriptor());
}

#[test]
fn test_descriptor_kind_binds() {
    assert!(DescriptorKind::Method.binds_to_instance());
    assert!(DescriptorKind::Property.binds_to_instance());
    assert!(!DescriptorKind::StaticMethod.binds_to_instance());
    assert!(!DescriptorKind::ClassMethod.binds_to_instance());
}
