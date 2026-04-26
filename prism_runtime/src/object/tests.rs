use super::*;

#[test]
fn test_header_size() {
    // Header should be 16 bytes
    assert_eq!(std::mem::size_of::<ObjectHeader>(), 16);
}

#[test]
fn test_gc_flags() {
    let flags = GcFlags::new();
    assert_eq!(flags.color(), GcColor::White);
    assert!(!flags.is_pinned());

    let flags = flags.set_color(GcColor::Gray);
    assert_eq!(flags.color(), GcColor::Gray);

    let flags = flags.set_pinned(true);
    assert!(flags.is_pinned());
    assert_eq!(flags.color(), GcColor::Gray);
}

#[test]
fn test_object_header() {
    let header = ObjectHeader::new(TypeId::LIST);
    assert_eq!(header.type_id, TypeId::LIST);
    assert_eq!(header.gc_color(), GcColor::White);
    assert!(!header.has_hash());
}
