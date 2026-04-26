use super::*;

// -------------------------------------------------------------------------
// StackAllocConfig Tests
// -------------------------------------------------------------------------

#[test]
fn test_config_default() {
    let config = StackAllocConfig::default();

    assert_eq!(config.max_object_size, 4096);
    assert_eq!(config.max_total_stack, 65536);
    assert_eq!(config.alignment, 8);
    assert!(config.allow_arg_escape);
}

#[test]
fn test_config_conservative() {
    let config = StackAllocConfig::conservative();

    assert_eq!(config.max_object_size, 1024);
    assert!(!config.allow_arg_escape);
}

#[test]
fn test_config_aggressive() {
    let config = StackAllocConfig::aggressive();

    assert_eq!(config.max_object_size, 16384);
    assert_eq!(config.alignment, 16);
}

// -------------------------------------------------------------------------
// StackSlot Tests
// -------------------------------------------------------------------------

#[test]
fn test_stack_slot_new() {
    let slot = StackSlot::new(-64, 32, 8, NodeId::new(5));

    assert_eq!(slot.offset, -64);
    assert_eq!(slot.size, 32);
    assert_eq!(slot.alignment, 8);
    assert_eq!(slot.allocation, NodeId::new(5));
}

#[test]
fn test_stack_slot_end_offset() {
    let slot = StackSlot::new(-64, 32, 8, NodeId::new(0));
    assert_eq!(slot.end_offset(), -32);

    let slot2 = StackSlot::new(0, 16, 8, NodeId::new(0));
    assert_eq!(slot2.end_offset(), 16);
}

#[test]
fn test_stack_slot_overlaps() {
    let slot1 = StackSlot::new(-64, 32, 8, NodeId::new(0));
    let slot2 = StackSlot::new(-48, 16, 8, NodeId::new(1));

    // -64 to -32 and -48 to -32 overlap
    assert!(slot1.overlaps(&slot2));
    assert!(slot2.overlaps(&slot1));

    let slot3 = StackSlot::new(-32, 16, 8, NodeId::new(2));
    // -64 to -32 and -32 to -16 don't overlap (adjacent)
    assert!(!slot1.overlaps(&slot3));
}

#[test]
fn test_stack_slot_no_overlap() {
    let slot1 = StackSlot::new(-100, 20, 8, NodeId::new(0));
    let slot2 = StackSlot::new(-50, 20, 8, NodeId::new(1));

    // -100 to -80 and -50 to -30 don't overlap
    assert!(!slot1.overlaps(&slot2));
}

// -------------------------------------------------------------------------
// StackFrameLayout Tests
// -------------------------------------------------------------------------

#[test]
fn test_frame_layout_new() {
    let layout = StackFrameLayout::new();

    assert!(layout.slots().is_empty());
    assert_eq!(layout.total_size(), 0);
    assert_eq!(layout.frame_size(), 0);
}

#[test]
fn test_frame_layout_allocate() {
    let mut layout = StackFrameLayout::new();

    let slot1 = layout.allocate(32, 8, NodeId::new(0));
    assert_eq!(slot1.size, 32);
    assert!(slot1.offset < 0); // Stack grows down

    let slot2 = layout.allocate(64, 8, NodeId::new(1));
    assert_eq!(slot2.size, 64);
    assert!(slot2.offset < slot1.offset); // Lower in stack

    assert_eq!(layout.total_size(), 96);
    assert_eq!(layout.slots().len(), 2);
}

#[test]
fn test_frame_layout_get_slot() {
    let mut layout = StackFrameLayout::new();

    layout.allocate(32, 8, NodeId::new(5));

    let slot = layout.get_slot(NodeId::new(5));
    assert!(slot.is_some());
    assert_eq!(slot.unwrap().size, 32);

    assert!(layout.get_slot(NodeId::new(99)).is_none());
}

#[test]
fn test_frame_layout_can_allocate() {
    let mut layout = StackFrameLayout::new();
    let config = StackAllocConfig {
        max_total_stack: 100,
        ..Default::default()
    };

    assert!(layout.can_allocate(50, &config));

    layout.allocate(60, 8, NodeId::new(0));

    assert!(layout.can_allocate(40, &config));
    assert!(!layout.can_allocate(50, &config));
}

#[test]
fn test_frame_layout_max_alignment() {
    let mut layout = StackFrameLayout::new();

    layout.allocate(32, 8, NodeId::new(0));
    assert_eq!(layout.max_alignment(), 8);

    layout.allocate(32, 16, NodeId::new(1));
    assert_eq!(layout.max_alignment(), 16);
}

#[test]
fn test_frame_layout_frame_size() {
    let mut layout = StackFrameLayout::new();

    layout.allocate(32, 8, NodeId::new(0));
    layout.allocate(64, 8, NodeId::new(1));

    // Frame size should be positive
    assert!(layout.frame_size() >= 96);
}

// -------------------------------------------------------------------------
// StackAllocResult Tests
// -------------------------------------------------------------------------

#[test]
fn test_result_success() {
    let slot = StackSlot::new(-32, 32, 8, NodeId::new(0));
    let result = StackAllocResult::success(NodeId::new(0), slot);

    assert!(result.success);
    assert!(result.slot.is_some());
    assert!(result.failure_reason.is_none());
}

#[test]
fn test_result_failure() {
    let result = StackAllocResult::failure(NodeId::new(0), StackAllocFailure::TooLarge);

    assert!(!result.success);
    assert!(result.slot.is_none());
    assert_eq!(result.failure_reason, Some(StackAllocFailure::TooLarge));
}

// -------------------------------------------------------------------------
// ObjectSizeEstimator Tests
// -------------------------------------------------------------------------

#[test]
fn test_estimator_new() {
    let estimator = ObjectSizeEstimator::new(64);
    assert_eq!(estimator.default_size, 64);
}

#[test]
fn test_estimator_register_type() {
    let mut estimator = ObjectSizeEstimator::new(64);
    estimator.register_type_size(100, 256);

    assert_eq!(estimator.type_sizes.get(&100), Some(&256));
}

#[test]
fn test_estimator_unknown_allocation() {
    let estimator = ObjectSizeEstimator::new(64);
    let graph = Graph::new();

    // Non-existent node
    assert!(estimator.estimate_size(&graph, NodeId::new(999)).is_none());
}

// -------------------------------------------------------------------------
// StackAllocator Tests
// -------------------------------------------------------------------------

#[test]
fn test_allocator_new() {
    let allocator = StackAllocator::new();
    assert_eq!(allocator.stats().stack_allocated, 0);
}

#[test]
fn test_allocator_with_config() {
    let config = StackAllocConfig {
        max_object_size: 1024,
        ..Default::default()
    };
    let allocator = StackAllocator::with_config(config);

    assert_eq!(allocator.config.max_object_size, 1024);
}

#[test]
fn test_allocator_reset() {
    let mut allocator = StackAllocator::new();

    // Simulate some state
    allocator.stats.stack_allocated = 5;

    allocator.reset();

    assert_eq!(allocator.stats().stack_allocated, 0);
    assert!(allocator.layout().slots().is_empty());
}

#[test]
fn test_allocator_arg_escape_disallowed() {
    let config = StackAllocConfig {
        allow_arg_escape: false,
        ..Default::default()
    };
    let mut allocator = StackAllocator::with_config(config);
    let mut graph = Graph::new();

    let result = allocator.try_stack_allocate(&mut graph, NodeId::new(0), true);

    assert!(!result.success);
    assert_eq!(result.failure_reason, Some(StackAllocFailure::GlobalEscape));
}

// -------------------------------------------------------------------------
// BatchStackAllocator Tests
// -------------------------------------------------------------------------

#[test]
fn test_batch_allocator_new() {
    let allocator = BatchStackAllocator::new();
    assert!(allocator.results.is_empty());
}

#[test]
fn test_batch_allocator_with_config() {
    let config = StackAllocConfig::conservative();
    let allocator = BatchStackAllocator::with_config(config);

    assert_eq!(allocator.allocator.config.max_object_size, 1024);
}

#[test]
fn test_batch_allocator_empty_batch() {
    let mut allocator = BatchStackAllocator::new();
    let mut graph = Graph::new();

    let results = allocator.process(&mut graph, &[]);
    assert!(results.is_empty());
}

// -------------------------------------------------------------------------
// StackAllocStats Tests
// -------------------------------------------------------------------------

#[test]
fn test_stats_default() {
    let stats = StackAllocStats::default();

    assert_eq!(stats.stack_allocated, 0);
    assert_eq!(stats.failed, 0);
    assert_eq!(stats.total_bytes, 0);
    assert!(stats.failures_by_reason.is_empty());
}

// -------------------------------------------------------------------------
// Integration Tests
// -------------------------------------------------------------------------

#[test]
fn test_multiple_allocations() {
    let mut layout = StackFrameLayout::new();

    // Allocate several objects
    let slots: Vec<_> = (0..5)
        .map(|i| layout.allocate(32, 8, NodeId::new(i)))
        .collect();

    // Verify no overlaps
    for i in 0..slots.len() {
        for j in (i + 1)..slots.len() {
            assert!(!slots[i].overlaps(&slots[j]));
        }
    }

    assert_eq!(layout.total_size(), 160);
}

#[test]
fn test_different_alignments() {
    let mut layout = StackFrameLayout::new();

    layout.allocate(16, 8, NodeId::new(0));
    layout.allocate(32, 16, NodeId::new(1));
    layout.allocate(64, 32, NodeId::new(2));

    assert_eq!(layout.max_alignment(), 32);
}

#[test]
fn test_failure_reasons() {
    let config = StackAllocConfig {
        max_object_size: 100,
        max_total_stack: 200,
        ..Default::default()
    };
    let mut allocator = StackAllocator::with_config(config);
    let mut graph = Graph::new();

    // This should fail with unknown size
    let result = allocator.try_stack_allocate(&mut graph, NodeId::new(0), false);
    assert!(!result.success);

    assert_eq!(allocator.stats().failed, 1);
}
