use super::*;

#[test]
fn test_osr_stub_info_from_entry() {
    let mut desc = OsrStateDescriptor::new();
    desc.set_frame_size(64);
    desc.set_callee_saved_count(2);
    desc.add_local_mapping(ValueLocation::register(0));
    desc.add_local_mapping(ValueLocation::stack(-8));
    desc.add_local_mapping(ValueLocation::dead());
    desc.add_local_mapping(ValueLocation::constant(42));

    let entry = OsrEntry::new(100, 200, desc);
    let info = OsrStubInfo::from_entry(&entry);

    assert_eq!(info.jit_target_offset, 200);
    assert_eq!(info.frame_size, 64);
    assert_eq!(info.callee_saved_count, 2);
    // 3 live values (constant is also considered live for capture)
    assert_eq!(info.materializations.len(), 3);
}

#[test]
fn test_osr_stub_info_estimated_size() {
    let info = OsrStubInfo {
        jit_target_offset: 100,
        frame_size: 32,
        callee_saved_count: 1,
        materializations: vec![Materialization {
            source_local_idx: 0,
            destination: ValueLocation::register(0),
        }],
    };

    let size = info.estimated_size();
    assert!(size > 0);
    assert!(size < 1000); // Reasonable bound
}

#[test]
fn test_osr_stub_cache() {
    let mut cache = OsrStubCache::new();
    assert!(cache.is_empty());

    let info = OsrStubInfo {
        jit_target_offset: 100,
        frame_size: 32,
        callee_saved_count: 0,
        materializations: vec![],
    };

    cache.insert(1, 50, info);
    assert_eq!(cache.len(), 1);
    assert!(cache.get(1, 50).is_some());
    assert!(cache.get(1, 60).is_none());

    cache.remove(1, 50);
    assert!(cache.is_empty());
}

#[test]
fn test_osr_stub_cache_clear_for_code() {
    let mut cache = OsrStubCache::new();

    let info = OsrStubInfo {
        jit_target_offset: 100,
        frame_size: 32,
        callee_saved_count: 0,
        materializations: vec![],
    };

    cache.insert(1, 50, info.clone());
    cache.insert(1, 100, info.clone());
    cache.insert(2, 50, info);

    assert_eq!(cache.len(), 3);

    cache.clear_for_code(1);
    assert_eq!(cache.len(), 1);
    assert!(cache.get(2, 50).is_some());
}

#[test]
fn test_osr_exit_builder() {
    let mut builder = OsrExitBuilder::new();
    builder.capture(0, ValueLocation::register(0));
    builder.capture(1, ValueLocation::stack(-8));

    assert_eq!(builder.count(), 2);
}

#[test]
fn test_osr_exit_builder_from_descriptor() {
    let mut desc = OsrStateDescriptor::new();
    desc.add_local_mapping(ValueLocation::register(0));
    desc.add_local_mapping(ValueLocation::dead());
    desc.add_local_mapping(ValueLocation::stack(-16));

    let builder = OsrExitBuilder::from_descriptor(&desc);
    assert_eq!(builder.count(), 2); // Only 2 live values
}

#[test]
fn test_register_and_stack_destinations() {
    let info = OsrStubInfo {
        jit_target_offset: 100,
        frame_size: 64,
        callee_saved_count: 0,
        materializations: vec![
            Materialization {
                source_local_idx: 0,
                destination: ValueLocation::register(0),
            },
            Materialization {
                source_local_idx: 1,
                destination: ValueLocation::stack(-8),
            },
            Materialization {
                source_local_idx: 2,
                destination: ValueLocation::register(1),
            },
        ],
    };

    let regs: Vec<_> = info.register_destinations().collect();
    assert_eq!(regs.len(), 2);
    assert_eq!(regs[0], (0, 0));
    assert_eq!(regs[1], (2, 1));

    let stacks: Vec<_> = info.stack_destinations().collect();
    assert_eq!(stacks.len(), 1);
    assert_eq!(stacks[0], (1, -8));
}
