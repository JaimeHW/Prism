use super::*;

fn dummy_code_ptr() -> *const u8 {
    0x10000usize as *const u8
}

#[test]
fn test_compiled_entry_creation() {
    let entry = CompiledEntry::new(1, dummy_code_ptr(), 100);
    assert_eq!(entry.code_id, 1);
    assert_eq!(entry.code_size(), 100);
    assert_eq!(entry.tier(), 1);
    assert_eq!(entry.return_abi(), ReturnAbi::RawValueBits);
    assert!(entry.stack_map().is_none());
    assert!(entry.deopt_sites().is_empty());
}

#[test]
fn test_compiled_entry_builder_pattern() {
    let entry = CompiledEntry::new(1, dummy_code_ptr(), 100)
        .with_entry_offset(16)
        .with_tier(2)
        .with_return_abi(ReturnAbi::EncodedExitReason)
        .with_deopt_sites(vec![
            DeoptSite {
                code_offset: 12,
                bc_offset: 3,
            },
            DeoptSite {
                code_offset: 24,
                bc_offset: 8,
            },
        ]);
    assert_eq!(entry.tier(), 2);
    assert_eq!(entry.return_abi(), ReturnAbi::EncodedExitReason);
    assert_eq!(entry.entry_point() as usize, dummy_code_ptr() as usize + 16);
    assert_eq!(entry.lookup_deopt_bc_offset(12), Some(3));
    assert_eq!(entry.lookup_deopt_bc_offset(20), Some(3));
    assert_eq!(entry.lookup_deopt_bc_offset_by_index(1), Some(8));
}

#[test]
fn test_code_cache_insert_lookup() {
    let cache = CodeCache::new(1024 * 1024);

    let entry = CompiledEntry::new(42, dummy_code_ptr(), 100);
    assert!(cache.insert(entry).is_none());

    let found = cache.lookup(42);
    assert!(found.is_some());
    assert_eq!(found.unwrap().code_id, 42);

    assert!(cache.lookup(999).is_none());
}

#[test]
fn test_code_cache_remove() {
    let cache = CodeCache::new(1024 * 1024);

    let entry = CompiledEntry::new(42, dummy_code_ptr(), 100);
    cache.insert(entry);

    assert_eq!(cache.total_size(), 100);

    let removed = cache.remove(42);
    assert!(removed.is_some());
    assert_eq!(cache.total_size(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_code_cache_stats() {
    let cache = CodeCache::new(1024 * 1024);

    // Miss
    cache.lookup(1);
    // Insert
    cache.insert(CompiledEntry::new(1, dummy_code_ptr(), 50));
    // Hit
    cache.lookup(1);
    // Miss
    cache.lookup(2);

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 2);
    assert_eq!(stats.insertions, 1);
}

#[test]
fn test_code_cache_evicts_when_capacity_exceeded() {
    let cache = CodeCache::new(150);

    cache.insert(CompiledEntry::new(1, dummy_code_ptr(), 100));
    cache.insert(CompiledEntry::new(
        2,
        (dummy_code_ptr() as usize + 64) as *const u8,
        100,
    ));

    assert!(cache.total_size() <= 150);
    assert_eq!(cache.len(), 1);

    let stats = cache.stats();
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.insertions, 2);
}

#[test]
fn test_find_by_ip() {
    let cache = CodeCache::new(1024 * 1024);

    let ptr = 0x20000usize as *const u8;
    cache.insert(CompiledEntry::new(1, ptr, 100));

    // Within range
    let found = cache.find_by_ip(0x20050);
    assert!(found.is_some());

    // Out of range
    assert!(cache.find_by_ip(0x10000).is_none());
    assert!(cache.find_by_ip(0x20100).is_none());
}

#[test]
fn test_code_cache_tracks_stack_map_registry_lifecycle() {
    use crate::gc::SafePoint;

    let cache = CodeCache::new(1024 * 1024);
    let code_start = dummy_code_ptr() as usize;
    let map = StackMap::new(
        code_start,
        100,
        48,
        vec![SafePoint::new(0x10, 0b0010, 0b0101)],
    );

    cache.insert(CompiledEntry::new(99, dummy_code_ptr(), 100).with_stack_map(map));

    let lookup = cache
        .lookup_stack_map(code_start + 0x10)
        .expect("stack map lookup should resolve registered entry");
    assert_eq!(lookup.frame_size, 48);
    assert_eq!(lookup.safepoint.register_bitmap, 0b0010);
    assert_eq!(lookup.safepoint.stack_bitmap, 0b0101);

    cache.remove(99);
    assert!(cache.lookup_stack_map(code_start + 0x10).is_none());
}
