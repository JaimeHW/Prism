use super::*;

#[test]
fn test_safepoint_creation() {
    let sp = SafePoint::new(0x10, 0b0011, 0b00001111);
    assert_eq!(sp.code_offset, 0x10);
    assert_eq!(sp.register_bitmap, 0b0011);
    assert_eq!(sp.stack_bitmap, 0b00001111);
}

#[test]
fn test_safepoint_live_checks() {
    let sp = SafePoint::new(0, 0b0101, 0b10010001);

    // Register checks (0b0101 = bits 0 and 2)
    assert!(sp.is_register_live(0));
    assert!(!sp.is_register_live(1));
    assert!(sp.is_register_live(2));
    assert!(!sp.is_register_live(3));

    // Stack checks (0b10010001 = bits 0, 4, 7)
    assert!(sp.is_stack_slot_live(0));
    assert!(!sp.is_stack_slot_live(1));
    assert!(sp.is_stack_slot_live(4));
    assert!(sp.is_stack_slot_live(7));
}

#[test]
fn test_live_bitmap_iter() {
    let sp = SafePoint::new(0, 0b1010, 0b10100101);

    let regs: Vec<u8> = sp.live_registers().collect();
    assert_eq!(regs, vec![1, 3]);

    let slots: Vec<u8> = sp.live_stack_slots().collect();
    assert_eq!(slots, vec![0, 2, 5, 7]);
}

#[test]
fn test_stackmap_lookup() {
    let map = StackMap::new(
        0x1000,
        0x100,
        64,
        vec![
            SafePoint::new(0x10, 0b0001, 0b0001),
            SafePoint::new(0x30, 0b0010, 0b0010),
            SafePoint::new(0x50, 0b0100, 0b0100),
        ],
    );

    // Exact match
    let sp = map.lookup_offset(0x30).unwrap();
    assert_eq!(sp.register_bitmap, 0b0010);

    // Before first safepoint
    assert!(map.lookup_offset(0x05).is_none());

    // Between safepoints (returns predecessor)
    let sp = map.lookup_offset(0x40).unwrap();
    assert_eq!(sp.code_offset, 0x30);

    // Address lookup
    let sp = map.lookup_address(0x1050).unwrap();
    assert_eq!(sp.code_offset, 0x50);

    // Out of range
    assert!(map.lookup_address(0x2000).is_none());
}

#[test]
fn test_stackmap_builder() {
    let mut builder = StackMapBuilder::new();
    builder.add_safepoint(0x20, 0b0001, 0b0001);
    builder.add_safepoint(0x10, 0b0010, 0b0010); // Unsorted input
    builder.add_safepoint(0x30, 0b0100, 0b0100);

    let map = builder.finish(0x1000, 0x100, 48);

    // Should be sorted
    assert_eq!(map.safepoints()[0].code_offset, 0x10);
    assert_eq!(map.safepoints()[1].code_offset, 0x20);
    assert_eq!(map.safepoints()[2].code_offset, 0x30);
}

#[test]
fn test_registry_lookup() {
    let registry = StackMapRegistry::new();

    // Insert multiple maps
    registry.insert(StackMap::new(
        0x1000,
        0x100,
        64,
        vec![SafePoint::new(0x10, 0b0001, 0b0001)],
    ));
    registry.insert(StackMap::new(
        0x2000,
        0x200,
        128,
        vec![SafePoint::new(0x20, 0b0010, 0b0010)],
    ));

    // Lookup in first map
    let result = registry.lookup(0x1010).unwrap();
    assert_eq!(result.safepoint.register_bitmap, 0b0001);
    assert_eq!(result.frame_size, 64);

    // Lookup in second map
    let result = registry.lookup(0x2020).unwrap();
    assert_eq!(result.safepoint.register_bitmap, 0b0010);
    assert_eq!(result.frame_size, 128);

    // Not found
    assert!(registry.lookup(0x3000).is_none());
}

#[test]
fn test_compact_array_lookup() {
    let maps = vec![
        StackMap::new(
            0x1000,
            0x100,
            64,
            vec![SafePoint::new(0x10, 0b0001, 0b0001)],
        ),
        StackMap::new(
            0x2000,
            0x200,
            128,
            vec![
                SafePoint::new(0x10, 0b0010, 0b0010),
                SafePoint::new(0x30, 0b0100, 0b0100),
            ],
        ),
    ];

    let compact = CompactStackMapArray::from_maps(maps);
    assert_eq!(compact.len(), 2);

    // Lookup
    let result = compact.lookup(0x1010).unwrap();
    assert_eq!(result.safepoint.register_bitmap, 0b0001);

    let result = compact.lookup(0x2025).unwrap();
    assert_eq!(result.safepoint.code_offset, 0x10); // Predecessor

    let result = compact.lookup(0x2030).unwrap();
    assert_eq!(result.safepoint.code_offset, 0x30);
}

#[test]
fn test_safepoint_counts() {
    let sp = SafePoint::new(0, 0b10101010, 0xFF00FF00FF00FF00);
    assert_eq!(sp.live_register_count(), 4);
    assert_eq!(sp.live_stack_slot_count(), 32);
}
