use super::*;

// -------------------------------------------------------------------------
// IcState Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_state_from_u8() {
    assert_eq!(IcState::from_u8(0), IcState::Uninitialized);
    assert_eq!(IcState::from_u8(1), IcState::Monomorphic);
    assert_eq!(IcState::from_u8(2), IcState::Polymorphic);
    assert_eq!(IcState::from_u8(3), IcState::Megamorphic);
    assert_eq!(IcState::from_u8(255), IcState::Megamorphic); // Invalid → mega
}

#[test]
fn test_ic_state_has_feedback() {
    assert!(!IcState::Uninitialized.has_feedback());
    assert!(IcState::Monomorphic.has_feedback());
    assert!(IcState::Polymorphic.has_feedback());
    assert!(IcState::Megamorphic.has_feedback());
}

#[test]
fn test_ic_state_is_monomorphic() {
    assert!(!IcState::Uninitialized.is_monomorphic());
    assert!(IcState::Monomorphic.is_monomorphic());
    assert!(!IcState::Polymorphic.is_monomorphic());
    assert!(!IcState::Megamorphic.is_monomorphic());
}

#[test]
fn test_ic_state_can_specialize() {
    assert!(IcState::Uninitialized.can_specialize());
    assert!(IcState::Monomorphic.can_specialize());
    assert!(IcState::Polymorphic.can_specialize());
    assert!(!IcState::Megamorphic.can_specialize());
}

// -------------------------------------------------------------------------
// IcKind Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_kind_is_property_op() {
    assert!(IcKind::GetProperty.is_property_op());
    assert!(IcKind::SetProperty.is_property_op());
    assert!(IcKind::DelProperty.is_property_op());
    assert!(!IcKind::GetItem.is_property_op());
    assert!(!IcKind::Call.is_property_op());
}

#[test]
fn test_ic_kind_is_item_op() {
    assert!(IcKind::GetItem.is_item_op());
    assert!(IcKind::SetItem.is_item_op());
    assert!(!IcKind::GetProperty.is_item_op());
    assert!(!IcKind::Call.is_item_op());
}

// -------------------------------------------------------------------------
// IcSiteHeader Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_site_header_new() {
    let header = IcSiteHeader::new(100, IcKind::GetProperty, 0);
    assert_eq!(header.bytecode_offset, 100);
    assert_eq!(header.kind, IcKind::GetProperty);
    assert_eq!(header.state(), IcState::Uninitialized);
    assert_eq!(header.miss_count(), 0);
    assert_eq!(header.data_index, 0);
}

#[test]
fn test_ic_site_header_size() {
    // Ensure header is exactly 16 bytes for cache alignment
    assert_eq!(std::mem::size_of::<IcSiteHeader>(), 16);
    assert_eq!(std::mem::align_of::<IcSiteHeader>(), 16);
}

#[test]
fn test_ic_site_header_transition() {
    let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

    // Forward transitions should succeed
    assert!(header.transition_to(IcState::Monomorphic));
    assert_eq!(header.state(), IcState::Monomorphic);

    assert!(header.transition_to(IcState::Polymorphic));
    assert_eq!(header.state(), IcState::Polymorphic);

    assert!(header.transition_to(IcState::Megamorphic));
    assert_eq!(header.state(), IcState::Megamorphic);

    // Backward transitions should fail
    assert!(!header.transition_to(IcState::Polymorphic));
    assert_eq!(header.state(), IcState::Megamorphic);

    assert!(!header.transition_to(IcState::Uninitialized));
    assert_eq!(header.state(), IcState::Megamorphic);
}

#[test]
fn test_ic_site_header_skip_transition() {
    let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

    // Can skip directly to megamorphic
    assert!(header.transition_to(IcState::Megamorphic));
    assert_eq!(header.state(), IcState::Megamorphic);
}

#[test]
fn test_ic_site_header_record_miss() {
    let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

    assert_eq!(header.miss_count(), 0);
    assert_eq!(header.record_miss(), 1);
    assert_eq!(header.miss_count(), 1);
    assert_eq!(header.record_miss(), 2);
    assert_eq!(header.miss_count(), 2);
}

#[test]
fn test_ic_site_header_miss_saturation() {
    let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

    // Miss count should saturate at 255
    for _ in 0..300 {
        header.record_miss();
    }
    assert_eq!(header.miss_count(), 255);
}

#[test]
fn test_ic_site_header_reset() {
    let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

    header.transition_to(IcState::Monomorphic);
    header.record_miss();
    header.record_miss();

    header.reset();

    assert_eq!(header.state(), IcState::Uninitialized);
    assert_eq!(header.miss_count(), 0);
}

// -------------------------------------------------------------------------
// IcManager Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_manager_new() {
    let mgr = IcManager::new(ShapeVersion::new(0));
    assert!(mgr.is_empty());
    assert_eq!(mgr.len(), 0);
}

#[test]
fn test_ic_manager_with_capacity() {
    let mgr = IcManager::with_capacity(16, ShapeVersion::new(0));
    assert!(mgr.is_empty());
}

#[test]
fn test_ic_manager_alloc_property() {
    let mut mgr = IcManager::new(ShapeVersion::new(0));

    let idx = mgr.alloc_property_ic(100, IcKind::GetProperty);
    assert_eq!(idx, Some(0));
    assert_eq!(mgr.len(), 1);

    let site = mgr.get(0).unwrap();
    assert_eq!(site.header.bytecode_offset, 100);
    assert_eq!(site.header.kind, IcKind::GetProperty);
    assert!(site.property_ic().is_some());
    assert!(site.call_ic().is_none());
}

#[test]
fn test_ic_manager_alloc_call() {
    let mut mgr = IcManager::new(ShapeVersion::new(0));

    let idx = mgr.alloc_call_ic(200);
    assert_eq!(idx, Some(0));
    assert_eq!(mgr.len(), 1);

    let site = mgr.get(0).unwrap();
    assert_eq!(site.header.bytecode_offset, 200);
    assert_eq!(site.header.kind, IcKind::Call);
    assert!(site.property_ic().is_none());
    assert!(site.call_ic().is_some());
}

#[test]
fn test_ic_manager_multiple_allocs() {
    let mut mgr = IcManager::new(ShapeVersion::new(0));

    let idx1 = mgr.alloc_property_ic(10, IcKind::GetProperty);
    let idx2 = mgr.alloc_property_ic(20, IcKind::SetProperty);
    let idx3 = mgr.alloc_call_ic(30);

    assert_eq!(idx1, Some(0));
    assert_eq!(idx2, Some(1));
    assert_eq!(idx3, Some(2));
    assert_eq!(mgr.len(), 3);
}

#[test]
fn test_ic_manager_get_out_of_bounds() {
    let mgr = IcManager::new(ShapeVersion::new(0));
    assert!(mgr.get(0).is_none());
    assert!(mgr.get(100).is_none());
}

#[test]
fn test_ic_manager_hit_miss_counters() {
    let mgr = IcManager::new(ShapeVersion::new(0));

    mgr.record_hit();
    mgr.record_hit();
    mgr.record_miss();

    let stats = mgr.stats();
    assert_eq!(stats.hits, 2);
    assert_eq!(stats.misses, 1);
}

#[test]
fn test_ic_manager_stats() {
    let mut mgr = IcManager::new(ShapeVersion::new(0));

    mgr.alloc_property_ic(10, IcKind::GetProperty);
    mgr.alloc_property_ic(20, IcKind::SetProperty);

    // Transition one to monomorphic
    mgr.get_mut(0)
        .unwrap()
        .header
        .transition_to(IcState::Monomorphic);

    let stats = mgr.stats();
    assert_eq!(stats.total_sites, 2);
    assert_eq!(stats.monomorphic, 1);
    assert_eq!(stats.polymorphic, 0);
    assert_eq!(stats.megamorphic, 0);
}

#[test]
fn test_ic_manager_reset_all() {
    let mut mgr = IcManager::new(ShapeVersion::new(0));
    mgr.alloc_property_ic(10, IcKind::GetProperty);
    mgr.alloc_property_ic(20, IcKind::SetProperty);

    // Transition to various states
    mgr.get_mut(0)
        .unwrap()
        .header
        .transition_to(IcState::Monomorphic);
    mgr.get_mut(1)
        .unwrap()
        .header
        .transition_to(IcState::Polymorphic);

    mgr.record_hit();
    mgr.record_miss();

    // Reset all
    mgr.reset_all(ShapeVersion::new(1));

    let stats = mgr.stats();
    assert_eq!(stats.monomorphic, 0);
    assert_eq!(stats.polymorphic, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(mgr.shape_version().value(), 1);

    // All sites should be uninitialized
    for site in mgr.iter() {
        assert_eq!(site.header.state(), IcState::Uninitialized);
    }
}

#[test]
fn test_ic_stats_hit_rate() {
    let stats = IcStats {
        hits: 80,
        misses: 20,
        ..Default::default()
    };
    assert!((stats.hit_rate() - 0.8).abs() < 0.001);

    let empty_stats = IcStats::default();
    assert_eq!(empty_stats.hit_rate(), 0.0);
}

#[test]
fn test_ic_manager_capacity_limit() {
    let mut mgr = IcManager::new(ShapeVersion::new(0));

    // Allocate up to limit
    for i in 0..MAX_IC_SITES_PER_FUNCTION {
        let result = mgr.alloc_property_ic(i as u32, IcKind::GetProperty);
        assert!(result.is_some(), "Failed at index {}", i);
    }

    // Next allocation should fail
    assert!(mgr.alloc_property_ic(999, IcKind::GetProperty).is_none());
    assert!(mgr.alloc_call_ic(999).is_none());
}

#[test]
fn test_ic_site_reset() {
    let mut mgr = IcManager::new(ShapeVersion::new(0));
    mgr.alloc_property_ic(10, IcKind::GetProperty);

    let site = mgr.get_mut(0).unwrap();
    site.header.transition_to(IcState::Monomorphic);
    site.header.record_miss();

    site.reset();

    assert_eq!(site.header.state(), IcState::Uninitialized);
    assert_eq!(site.header.miss_count(), 0);
}

// -------------------------------------------------------------------------
// Concurrent Access Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_header_concurrent_transitions() {
    use std::sync::Arc;
    use std::thread;

    let header = Arc::new(IcSiteHeader::new(0, IcKind::GetProperty, 0));
    let mut handles = vec![];

    // Spawn threads that all try to transition
    for _ in 0..10 {
        let h = Arc::clone(&header);
        handles.push(thread::spawn(move || {
            h.transition_to(IcState::Monomorphic);
            h.transition_to(IcState::Polymorphic);
            h.transition_to(IcState::Megamorphic);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should end up at Megamorphic
    assert_eq!(header.state(), IcState::Megamorphic);
}

#[test]
fn test_ic_manager_concurrent_hit_miss() {
    use std::sync::Arc;
    use std::thread;

    let mgr = Arc::new(IcManager::new(ShapeVersion::new(0)));
    let mut handles = vec![];

    for _ in 0..10 {
        let m = Arc::clone(&mgr);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                m.record_hit();
                m.record_miss();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let stats = mgr.stats();
    assert_eq!(stats.hits, 10 * 1000);
    assert_eq!(stats.misses, 10 * 1000);
}
