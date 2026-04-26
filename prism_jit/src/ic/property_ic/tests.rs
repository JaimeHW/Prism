use super::*;

// -------------------------------------------------------------------------
// SlotInfo Tests
// -------------------------------------------------------------------------

#[test]
fn test_slot_info_inline() {
    let info = SlotInfo::inline(5, PropertyFlags::default());
    assert_eq!(info.offset, 5);
    assert!(info.is_inline);
    assert!(info.is_writable());
    assert!(info.is_data());
}

#[test]
fn test_slot_info_dictionary() {
    let info = SlotInfo::dictionary(10, PropertyFlags::read_only());
    assert_eq!(info.offset, 10);
    assert!(!info.is_inline);
    assert!(!info.is_writable());
}

// -------------------------------------------------------------------------
// MonoPropertyData Tests
// -------------------------------------------------------------------------

#[test]
fn test_mono_property_data_size() {
    assert_eq!(std::mem::size_of::<MonoPropertyData>(), 16);
    assert_eq!(std::mem::align_of::<MonoPropertyData>(), 16);
}

#[test]
fn test_mono_property_data_new() {
    let data = MonoPropertyData::new(ShapeId(42), 5, PropertyFlags::default());
    assert_eq!(data.shape_id, ShapeId(42));
    assert_eq!(data.slot_offset, 5);
    assert!(data.matches(ShapeId(42)));
    assert!(!data.matches(ShapeId(1)));
}

#[test]
fn test_mono_property_data_flags() {
    let data = MonoPropertyData::new(ShapeId(1), 0, PropertyFlags::read_only());
    let flags = data.property_flags();
    assert!(!flags.contains(PropertyFlags::WRITABLE));
    assert!(flags.contains(PropertyFlags::ENUMERABLE));
}

// -------------------------------------------------------------------------
// PolyPropertyEntry Tests
// -------------------------------------------------------------------------

#[test]
fn test_poly_entry_size() {
    assert_eq!(std::mem::size_of::<PolyPropertyEntry>(), 12);
}

#[test]
fn test_poly_entry_new() {
    let entry = PolyPropertyEntry::new(ShapeId(10), 3, PropertyFlags::default());
    assert_eq!(entry.shape_id, ShapeId(10));
    assert_eq!(entry.slot_offset, 3);
    assert!(!entry.is_empty());
    assert!(entry.matches(ShapeId(10)));
}

#[test]
fn test_poly_entry_empty() {
    let entry = PolyPropertyEntry::empty();
    assert!(entry.is_empty());
    assert!(entry.matches(ShapeId(0))); // Empty matches shape 0
}

#[test]
fn test_poly_entry_touch() {
    let mut entry = PolyPropertyEntry::new(ShapeId(1), 0, PropertyFlags::default());
    assert_eq!(entry.access_count, 0);
    entry.touch();
    assert_eq!(entry.access_count, 1);
    entry.touch();
    assert_eq!(entry.access_count, 2);
}

// -------------------------------------------------------------------------
// PolyPropertyData Tests
// -------------------------------------------------------------------------

#[test]
fn test_poly_data_new() {
    let data = PolyPropertyData::new();
    assert!(data.is_empty());
    assert!(!data.is_full());
    assert_eq!(data.len(), 0);
}

#[test]
fn test_poly_data_lookup_empty() {
    let data = PolyPropertyData::new();
    assert!(data.lookup(ShapeId(1)).is_none());
}

#[test]
fn test_poly_data_add_and_lookup() {
    let mut data = PolyPropertyData::new();

    assert!(data.try_add(ShapeId(1), 5, PropertyFlags::default()));
    assert_eq!(data.len(), 1);

    let result = data.lookup(ShapeId(1));
    assert!(result.is_some());
    let (offset, _) = result.unwrap();
    assert_eq!(offset, 5);
}

#[test]
fn test_poly_data_multiple_entries() {
    let mut data = PolyPropertyData::new();

    data.try_add(ShapeId(1), 0, PropertyFlags::default());
    data.try_add(ShapeId(2), 1, PropertyFlags::default());
    data.try_add(ShapeId(3), 2, PropertyFlags::default());

    assert_eq!(data.len(), 3);
    assert!(data.lookup(ShapeId(1)).is_some());
    assert!(data.lookup(ShapeId(2)).is_some());
    assert!(data.lookup(ShapeId(3)).is_some());
    assert!(data.lookup(ShapeId(4)).is_none());
}

#[test]
fn test_poly_data_full() {
    let mut data = PolyPropertyData::new();

    for i in 0..POLY_IC_ENTRIES {
        assert!(data.try_add(ShapeId(i as u32 + 1), i as u16, PropertyFlags::default()));
    }

    assert!(data.is_full());
    assert!(!data.try_add(ShapeId(99), 99, PropertyFlags::default()));
}

#[test]
fn test_poly_data_replace_lru() {
    let mut data = PolyPropertyData::new();

    // Fill the cache
    for i in 0..POLY_IC_ENTRIES {
        data.try_add(ShapeId(i as u32 + 1), i as u16, PropertyFlags::default());
    }

    // Touch all except the first
    for i in 1..POLY_IC_ENTRIES {
        data.lookup_and_touch(ShapeId(i as u32 + 1));
        data.lookup_and_touch(ShapeId(i as u32 + 1));
    }

    // Replace LRU (should be index 0)
    data.replace_lru(ShapeId(99), 99, PropertyFlags::default());

    // Shape 1 should be gone
    assert!(data.lookup(ShapeId(1)).is_none());
    // Shape 99 should be present
    assert!(data.lookup(ShapeId(99)).is_some());
}

#[test]
fn test_poly_data_clear() {
    let mut data = PolyPropertyData::new();
    data.try_add(ShapeId(1), 0, PropertyFlags::default());
    data.try_add(ShapeId(2), 1, PropertyFlags::default());

    data.clear();

    assert!(data.is_empty());
    assert!(data.lookup(ShapeId(1)).is_none());
}

// -------------------------------------------------------------------------
// PropertyIc Tests
// -------------------------------------------------------------------------

#[test]
fn test_property_ic_new() {
    let ic = PropertyIc::new();
    assert_eq!(ic.state(), IcState::Uninitialized);
    assert_eq!(ic.hits(), 0);
    assert_eq!(ic.misses(), 0);
}

#[test]
fn test_property_ic_lookup_uninitialized() {
    let mut ic = PropertyIc::new();
    assert!(ic.lookup(ShapeId(1)).is_none());
    assert_eq!(ic.misses(), 1);
}

#[test]
fn test_property_ic_update_to_mono() {
    let mut ic = PropertyIc::new();

    ic.update(ShapeId(1), 5, PropertyFlags::default());

    assert_eq!(ic.state(), IcState::Monomorphic);

    let result = ic.lookup(ShapeId(1));
    assert!(result.is_some());
    assert_eq!(result.unwrap().offset, 5);
    assert_eq!(ic.hits(), 1);
}

#[test]
fn test_property_ic_mono_miss() {
    let mut ic = PropertyIc::new();
    ic.update(ShapeId(1), 0, PropertyFlags::default());

    // Lookup different shape
    assert!(ic.lookup(ShapeId(2)).is_none());
    assert_eq!(ic.misses(), 1);
}

#[test]
fn test_property_ic_mono_to_poly() {
    let mut ic = PropertyIc::new();

    ic.update(ShapeId(1), 0, PropertyFlags::default());
    assert_eq!(ic.state(), IcState::Monomorphic);

    ic.update(ShapeId(2), 1, PropertyFlags::default());
    assert_eq!(ic.state(), IcState::Polymorphic);

    // Both shapes should be accessible
    assert!(ic.lookup(ShapeId(1)).is_some());
    assert!(ic.lookup(ShapeId(2)).is_some());
}

#[test]
fn test_property_ic_poly_to_mega() {
    let mut ic = PropertyIc::new();

    // Add more shapes than POLY_IC_ENTRIES
    for i in 0..(POLY_IC_ENTRIES + 2) {
        ic.update(ShapeId(i as u32 + 1), i as u16, PropertyFlags::default());
    }

    assert_eq!(ic.state(), IcState::Megamorphic);
}

#[test]
fn test_property_ic_reset() {
    let mut ic = PropertyIc::new();

    ic.update(ShapeId(1), 0, PropertyFlags::default());
    ic.lookup(ShapeId(1));
    ic.lookup(ShapeId(2));

    ic.reset();

    assert_eq!(ic.state(), IcState::Uninitialized);
    assert_eq!(ic.hits(), 0);
    assert_eq!(ic.misses(), 0);
}

#[test]
fn test_property_ic_hit_rate() {
    let mut ic = PropertyIc::new();
    ic.update(ShapeId(1), 0, PropertyFlags::default());

    // 3 hits, 1 miss
    ic.lookup(ShapeId(1));
    ic.lookup(ShapeId(1));
    ic.lookup(ShapeId(1));
    ic.lookup(ShapeId(2)); // miss

    assert!((ic.hit_rate() - 0.75).abs() < 0.001);
}

#[test]
fn test_property_ic_hit_rate_empty() {
    let ic = PropertyIc::new();
    assert_eq!(ic.hit_rate(), 0.0);
}

// -------------------------------------------------------------------------
// State Transition Tests
// -------------------------------------------------------------------------

#[test]
fn test_state_transition_sequence() {
    let mut ic = PropertyIc::new();

    // Uninitialized → Monomorphic
    ic.update(ShapeId(1), 0, PropertyFlags::default());
    assert_eq!(ic.state(), IcState::Monomorphic);

    // Monomorphic → Polymorphic (different shape)
    ic.update(ShapeId(2), 1, PropertyFlags::default());
    assert_eq!(ic.state(), IcState::Polymorphic);

    // Add more to fill polymorphic cache
    ic.update(ShapeId(3), 2, PropertyFlags::default());
    ic.update(ShapeId(4), 3, PropertyFlags::default());
    assert_eq!(ic.state(), IcState::Polymorphic);

    // Polymorphic → Megamorphic (exceeds capacity)
    ic.update(ShapeId(5), 4, PropertyFlags::default());
    assert_eq!(ic.state(), IcState::Megamorphic);

    // Should stay megamorphic
    ic.update(ShapeId(6), 5, PropertyFlags::default());
    assert_eq!(ic.state(), IcState::Megamorphic);
}

#[test]
fn test_same_shape_update_no_transition() {
    let mut ic = PropertyIc::new();

    ic.update(ShapeId(1), 0, PropertyFlags::default());
    assert_eq!(ic.state(), IcState::Monomorphic);

    // Same shape shouldn't cause transition
    ic.update(ShapeId(1), 0, PropertyFlags::default());
    assert_eq!(ic.state(), IcState::Monomorphic);
}

// -------------------------------------------------------------------------
// Edge Case Tests
// -------------------------------------------------------------------------

#[test]
fn test_poly_data_lookup_and_touch() {
    let mut data = PolyPropertyData::new();
    data.try_add(ShapeId(1), 0, PropertyFlags::default());
    data.try_add(ShapeId(2), 1, PropertyFlags::default());

    // Touch shape 2 multiple times
    data.lookup_and_touch(ShapeId(2));
    data.lookup_and_touch(ShapeId(2));
    data.lookup_and_touch(ShapeId(2));

    assert_eq!(data.entries[1].access_count, 3);
    assert_eq!(data.entries[0].access_count, 0);
}

#[test]
fn test_property_ic_megamorphic_lookup_always_misses() {
    let mut ic = PropertyIc::new();
    ic.force_state(IcState::Megamorphic);

    // Megamorphic ICs don't use inline cache
    assert!(ic.lookup(ShapeId(1)).is_none());
    assert!(ic.lookup(ShapeId(2)).is_none());
    assert_eq!(ic.misses(), 2);
}

#[test]
fn test_property_ic_force_state() {
    let mut ic = PropertyIc::new();

    ic.force_state(IcState::Polymorphic);
    assert_eq!(ic.state(), IcState::Polymorphic);

    ic.force_state(IcState::Uninitialized);
    assert_eq!(ic.state(), IcState::Uninitialized);
}
