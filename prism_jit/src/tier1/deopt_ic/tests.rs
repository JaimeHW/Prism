use super::*;
use std::mem;

// -------------------------------------------------------------------------
// IcDeoptContext Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_deopt_context_size() {
    // Must be exactly 16 bytes for cache efficiency
    assert_eq!(mem::size_of::<IcDeoptContext>(), 16);
}

#[test]
fn test_ic_deopt_context_alignment() {
    // Must be 16-byte aligned
    assert_eq!(mem::align_of::<IcDeoptContext>(), 16);
}

#[test]
fn test_ic_deopt_context_new_property() {
    let ctx = IcDeoptContext::new_property(
        5,
        ShapeId(42),
        10,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );

    assert_eq!(ctx.ic_site_idx, 5);
    assert_eq!(ctx.observed_shape, ShapeId(42));
    assert_eq!(ctx.slot_offset, 10);
    assert_eq!(ctx.kind, IcKind::GetProperty);
    assert!(ctx.is_inline);
    assert!(ctx.is_property_op());
    assert!(!ctx.is_call_op());
}

#[test]
fn test_ic_deopt_context_new_call() {
    let ctx = IcDeoptContext::new_call(3, ShapeId(100));

    assert_eq!(ctx.ic_site_idx, 3);
    assert_eq!(ctx.observed_shape, ShapeId(100));
    assert_eq!(ctx.kind, IcKind::Call);
    assert!(!ctx.is_property_op());
    assert!(ctx.is_call_op());
}

#[test]
fn test_ic_deopt_context_empty() {
    let ctx = IcDeoptContext::empty();

    assert_eq!(ctx.ic_site_idx, 0);
    assert_eq!(ctx.observed_shape, ShapeId::EMPTY);
    assert_eq!(ctx.slot_offset, 0);
}

#[test]
fn test_ic_deopt_context_default() {
    let ctx = IcDeoptContext::default();
    assert_eq!(ctx.ic_site_idx, 0);
}

#[test]
fn test_ic_deopt_context_set_attr() {
    let ctx = IcDeoptContext::new_property(
        1,
        ShapeId(10),
        5,
        PropertyFlags::WRITABLE,
        IcKind::SetProperty,
        true,
    );

    assert_eq!(ctx.kind, IcKind::SetProperty);
    assert!(ctx.is_property_op());
}

// -------------------------------------------------------------------------
// IcUpdateResult Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_update_result_success() {
    let result = IcUpdateResult::Updated {
        old_state: IcState::Uninitialized,
        new_state: IcState::Monomorphic,
    };

    assert!(result.is_success());
    assert!(result.transitioned());
}

#[test]
fn test_ic_update_result_no_transition() {
    let result = IcUpdateResult::Updated {
        old_state: IcState::Monomorphic,
        new_state: IcState::Monomorphic,
    };

    assert!(result.is_success());
    assert!(!result.transitioned());
}

#[test]
fn test_ic_update_result_site_not_found() {
    let result = IcUpdateResult::SiteNotFound;

    assert!(!result.is_success());
    assert!(!result.transitioned());
}

#[test]
fn test_ic_update_result_already_mega() {
    let result = IcUpdateResult::AlreadyMegamorphic;

    assert!(!result.is_success());
    assert!(!result.transitioned());
}

#[test]
fn test_ic_update_result_kind_mismatch() {
    let result = IcUpdateResult::KindMismatch;

    assert!(!result.is_success());
    assert!(!result.transitioned());
}

// -------------------------------------------------------------------------
// IcDeoptStats Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_deopt_stats_new() {
    let stats = IcDeoptStats::new();

    assert_eq!(stats.total(), 0);
    assert_eq!(stats.get_attr_count(), 0);
    assert_eq!(stats.set_attr_count(), 0);
    assert_eq!(stats.call_count(), 0);
}

#[test]
fn test_ic_deopt_stats_record_get_attr() {
    let stats = IcDeoptStats::new();
    let result = IcUpdateResult::Updated {
        old_state: IcState::Uninitialized,
        new_state: IcState::Monomorphic,
    };

    stats.record(IcKind::GetProperty, &result);

    assert_eq!(stats.total(), 1);
    assert_eq!(stats.get_attr_count(), 1);
    assert_eq!(stats.transition_count(), 1);
}

#[test]
fn test_ic_deopt_stats_record_set_attr() {
    let stats = IcDeoptStats::new();
    let result = IcUpdateResult::Updated {
        old_state: IcState::Monomorphic,
        new_state: IcState::Polymorphic,
    };

    stats.record(IcKind::SetProperty, &result);

    assert_eq!(stats.total(), 1);
    assert_eq!(stats.set_attr_count(), 1);
    assert_eq!(stats.transition_count(), 1);
}

#[test]
fn test_ic_deopt_stats_record_call() {
    let stats = IcDeoptStats::new();
    let result = IcUpdateResult::Updated {
        old_state: IcState::Uninitialized,
        new_state: IcState::Monomorphic,
    };

    stats.record(IcKind::Call, &result);

    assert_eq!(stats.total(), 1);
    assert_eq!(stats.call_count(), 1);
}

#[test]
fn test_ic_deopt_stats_record_already_mega() {
    let stats = IcDeoptStats::new();
    let result = IcUpdateResult::AlreadyMegamorphic;

    stats.record(IcKind::GetProperty, &result);

    assert_eq!(stats.total(), 1);
    assert_eq!(stats.already_mega_count(), 1);
    assert_eq!(stats.transition_count(), 0);
}

#[test]
fn test_ic_deopt_stats_record_invalid_site() {
    let stats = IcDeoptStats::new();
    let result = IcUpdateResult::SiteNotFound;

    stats.record(IcKind::GetProperty, &result);

    assert_eq!(stats.total(), 1);
    assert_eq!(stats.invalid_site_count(), 1);
}

#[test]
fn test_ic_deopt_stats_snapshot() {
    let stats = IcDeoptStats::new();

    // Record various events
    stats.record(
        IcKind::GetProperty,
        &IcUpdateResult::Updated {
            old_state: IcState::Uninitialized,
            new_state: IcState::Monomorphic,
        },
    );
    stats.record(IcKind::SetProperty, &IcUpdateResult::AlreadyMegamorphic);
    stats.record(IcKind::Call, &IcUpdateResult::SiteNotFound);

    let snapshot = stats.snapshot();

    assert_eq!(snapshot.total, 3);
    assert_eq!(snapshot.get_attr, 1);
    assert_eq!(snapshot.set_attr, 1);
    assert_eq!(snapshot.call, 1);
    assert_eq!(snapshot.transitions, 1);
    assert_eq!(snapshot.already_mega, 1);
    assert_eq!(snapshot.invalid_sites, 1);
}

#[test]
fn test_ic_deopt_stats_concurrent() {
    use std::sync::Arc;
    use std::thread;

    let stats = Arc::new(IcDeoptStats::new());
    let mut handles = vec![];

    for _ in 0..10 {
        let s = Arc::clone(&stats);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                s.record(
                    IcKind::GetProperty,
                    &IcUpdateResult::Updated {
                        old_state: IcState::Uninitialized,
                        new_state: IcState::Monomorphic,
                    },
                );
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(stats.total(), 1000);
    assert_eq!(stats.get_attr_count(), 1000);
    assert_eq!(stats.transition_count(), 1000);
}

// -------------------------------------------------------------------------
// Handle IC Deopt Tests
// -------------------------------------------------------------------------

#[test]
fn test_handle_ic_deopt_site_not_found() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let context = IcDeoptContext::new_property(
        999, // Non-existent site
        ShapeId(42),
        5,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );

    let result = handle_ic_deopt(&mut manager, &context, None);

    assert_eq!(result, IcUpdateResult::SiteNotFound);
}

#[test]
fn test_handle_ic_deopt_get_attr_updates_ic() {
    let mut manager = IcManager::new(ShapeVersion::current());

    // Allocate a GetAttr IC site
    let site_idx = manager.alloc_property_ic(0, IcKind::GetProperty).unwrap();

    let context = IcDeoptContext::new_property(
        site_idx,
        ShapeId(42),
        5,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );

    let result = handle_ic_deopt(&mut manager, &context, None);

    // Should have transitioned to monomorphic
    assert!(result.is_success());
    assert!(result.transitioned());

    if let IcUpdateResult::Updated {
        old_state,
        new_state,
    } = result
    {
        assert_eq!(old_state, IcState::Uninitialized);
        assert_eq!(new_state, IcState::Monomorphic);
    }
}

#[test]
fn test_handle_ic_deopt_set_attr_updates_ic() {
    let mut manager = IcManager::new(ShapeVersion::current());

    // Allocate a SetAttr IC site
    let site_idx = manager.alloc_property_ic(0, IcKind::SetProperty).unwrap();

    let context = IcDeoptContext::new_property(
        site_idx,
        ShapeId(100),
        3,
        PropertyFlags::WRITABLE,
        IcKind::SetProperty,
        true,
    );

    let result = handle_ic_deopt(&mut manager, &context, None);

    assert!(result.is_success());
    assert!(result.transitioned());
}

#[test]
fn test_handle_ic_deopt_already_megamorphic() {
    let mut manager = IcManager::new(ShapeVersion::current());

    // Allocate and force to megamorphic
    let site_idx = manager.alloc_property_ic(0, IcKind::GetProperty).unwrap();
    if let Some(site) = manager.get_mut(site_idx) {
        if let Some(ic) = site.property_ic_mut() {
            ic.force_state(IcState::Megamorphic);
        }
    }

    let context = IcDeoptContext::new_property(
        site_idx,
        ShapeId(42),
        5,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );

    let result = handle_ic_deopt(&mut manager, &context, None);

    assert_eq!(result, IcUpdateResult::AlreadyMegamorphic);
}

#[test]
fn test_handle_ic_deopt_kind_mismatch() {
    let mut manager = IcManager::new(ShapeVersion::current());

    // Allocate a GetAttr IC site but send SetAttr context
    let site_idx = manager.alloc_call_ic(0).unwrap();

    let context = IcDeoptContext::new_property(
        site_idx,
        ShapeId(42),
        5,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );

    let result = handle_ic_deopt(&mut manager, &context, None);

    assert_eq!(result, IcUpdateResult::KindMismatch);
}

#[test]
fn test_handle_ic_deopt_with_stats() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let stats = IcDeoptStats::new();

    let site_idx = manager.alloc_property_ic(0, IcKind::GetProperty).unwrap();

    let context = IcDeoptContext::new_property(
        site_idx,
        ShapeId(50),
        8,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );

    let result = handle_ic_deopt(&mut manager, &context, Some(&stats));

    assert!(result.is_success());
    assert_eq!(stats.total(), 1);
    assert_eq!(stats.get_attr_count(), 1);
    assert_eq!(stats.transition_count(), 1);
}

#[test]
fn test_handle_ic_deopt_polymorphic_transition() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let site_idx = manager.alloc_property_ic(0, IcKind::GetProperty).unwrap();

    // First deopt - should go to monomorphic
    let ctx1 = IcDeoptContext::new_property(
        site_idx,
        ShapeId(1),
        0,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );
    let result1 = handle_ic_deopt(&mut manager, &ctx1, None);
    assert!(result1.transitioned());

    // Second deopt with different shape - should go to polymorphic
    let ctx2 = IcDeoptContext::new_property(
        site_idx,
        ShapeId(2),
        1,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );
    let result2 = handle_ic_deopt(&mut manager, &ctx2, None);

    if let IcUpdateResult::Updated { new_state, .. } = result2 {
        assert_eq!(new_state, IcState::Polymorphic);
    }
}

// -------------------------------------------------------------------------
// Batch Handler Tests
// -------------------------------------------------------------------------

#[test]
fn test_handle_ic_deopts_batch_empty() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let results = handle_ic_deopts_batch(&mut manager, &[], None);

    assert!(results.is_empty());
}

#[test]
fn test_handle_ic_deopts_batch_multiple() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let stats = IcDeoptStats::new();

    // Allocate multiple IC sites
    let site1 = manager.alloc_property_ic(0, IcKind::GetProperty).unwrap();
    let site2 = manager.alloc_property_ic(4, IcKind::SetProperty).unwrap();

    let contexts = vec![
        IcDeoptContext::new_property(
            site1,
            ShapeId(10),
            0,
            PropertyFlags::default(),
            IcKind::GetProperty,
            true,
        ),
        IcDeoptContext::new_property(
            site2,
            ShapeId(20),
            1,
            PropertyFlags::WRITABLE,
            IcKind::SetProperty,
            true,
        ),
    ];

    let results = handle_ic_deopts_batch(&mut manager, &contexts, Some(&stats));

    assert_eq!(results.len(), 2);
    assert!(results[0].is_success());
    assert!(results[1].is_success());
    assert_eq!(stats.total(), 2);
    assert_eq!(stats.get_attr_count(), 1);
    assert_eq!(stats.set_attr_count(), 1);
}

// -------------------------------------------------------------------------
// Builder Tests
// -------------------------------------------------------------------------

#[test]
fn test_ic_deopt_context_builder() {
    let ctx = IcDeoptContextBuilder::new()
        .ic_site(7)
        .shape(ShapeId(99))
        .offset(15)
        .flags(PropertyFlags::WRITABLE | PropertyFlags::ENUMERABLE)
        .kind(IcKind::SetProperty)
        .inline(false)
        .build();

    assert_eq!(ctx.ic_site_idx, 7);
    assert_eq!(ctx.observed_shape, ShapeId(99));
    assert_eq!(ctx.slot_offset, 15);
    assert!(ctx.flags.contains(PropertyFlags::WRITABLE));
    assert!(ctx.flags.contains(PropertyFlags::ENUMERABLE));
    assert_eq!(ctx.kind, IcKind::SetProperty);
    assert!(!ctx.is_inline);
}

#[test]
fn test_ic_deopt_context_builder_default() {
    let ctx = IcDeoptContextBuilder::new().build();

    assert_eq!(ctx.ic_site_idx, 0);
    assert_eq!(ctx.observed_shape, ShapeId::EMPTY);
    assert_eq!(ctx.slot_offset, 0);
}

// -------------------------------------------------------------------------
// Call IC Tests
// -------------------------------------------------------------------------

#[test]
fn test_handle_ic_deopt_call() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let site_idx = manager.alloc_call_ic(0).unwrap();

    let context = IcDeoptContext::new_call(site_idx, ShapeId(42));

    let result = handle_ic_deopt(&mut manager, &context, None);

    assert!(result.is_success());
    assert!(result.transitioned());
}

// -------------------------------------------------------------------------
// Edge Case Tests
// -------------------------------------------------------------------------

#[test]
fn test_handle_ic_deopt_zero_slot_offset() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let site_idx = manager.alloc_property_ic(0, IcKind::GetProperty).unwrap();

    let context = IcDeoptContext::new_property(
        site_idx,
        ShapeId(1),
        0, // First slot
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );

    let result = handle_ic_deopt(&mut manager, &context, None);
    assert!(result.is_success());
}

#[test]
fn test_handle_ic_deopt_max_slot_offset() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let site_idx = manager.alloc_property_ic(0, IcKind::GetProperty).unwrap();

    let context = IcDeoptContext::new_property(
        site_idx,
        ShapeId(1),
        u16::MAX, // Maximum slot offset
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );

    let result = handle_ic_deopt(&mut manager, &context, None);
    assert!(result.is_success());
}

#[test]
fn test_handle_ic_deopt_empty_shape() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let site_idx = manager.alloc_property_ic(0, IcKind::GetProperty).unwrap();

    let context = IcDeoptContext::new_property(
        site_idx,
        ShapeId::EMPTY,
        0,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );

    let result = handle_ic_deopt(&mut manager, &context, None);
    assert!(result.is_success());
}

#[test]
fn test_multiple_transitions_to_megamorphic() {
    let mut manager = IcManager::new(ShapeVersion::current());
    let stats = IcDeoptStats::new();
    let site_idx = manager.alloc_property_ic(0, IcKind::GetProperty).unwrap();

    // Force the IC to megamorphic state directly
    if let Some(site) = manager.get_mut(site_idx) {
        if let Some(ic) = site.property_ic_mut() {
            ic.force_state(IcState::Megamorphic);
        }
    }

    // Now try to update - should return AlreadyMegamorphic
    let ctx = IcDeoptContext::new_property(
        site_idx,
        ShapeId(100),
        50,
        PropertyFlags::default(),
        IcKind::GetProperty,
        true,
    );
    let result = handle_ic_deopt(&mut manager, &ctx, Some(&stats));

    assert_eq!(result, IcUpdateResult::AlreadyMegamorphic);
    assert_eq!(stats.already_mega_count(), 1);
}
