use super::*;

// ════════════════════════════════════════════════════════════════════════
// YieldPointEntry Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_yield_point_entry_new() {
    let entry = YieldPointEntry::new(5, 100);
    assert_eq!(entry.resume_idx, 5);
    assert_eq!(entry.pc, 100);
}

// ════════════════════════════════════════════════════════════════════════
// ResumeTable Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_resume_table_new() {
    let table = ResumeTable::new();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert!(table.is_dense());
}

#[test]
fn test_resume_table_insert_get() {
    let mut table = ResumeTable::new();

    table.insert(0, 100);
    table.insert(1, 200);
    table.insert(2, 300);

    assert_eq!(table.len(), 3);
    assert_eq!(table.get_pc(0), Some(100));
    assert_eq!(table.get_pc(1), Some(200));
    assert_eq!(table.get_pc(2), Some(300));
    assert_eq!(table.get_pc(3), None);
}

#[test]
fn test_resume_table_contains() {
    let mut table = ResumeTable::new();

    table.insert(5, 500);

    assert!(table.contains(5));
    assert!(!table.contains(0));
    assert!(!table.contains(6));
}

#[test]
fn test_resume_table_sparse_insert() {
    let mut table = ResumeTable::new();

    // Insert with gaps
    table.insert(0, 100);
    table.insert(5, 500);
    table.insert(10, 1000);

    // Should still be dense with gaps
    assert!(table.is_dense());
    assert_eq!(table.get_pc(0), Some(100));
    assert_eq!(table.get_pc(1), None); // Gap
    assert_eq!(table.get_pc(5), Some(500));
    assert_eq!(table.get_pc(10), Some(1000));
}

#[test]
fn test_resume_table_iter() {
    let mut table = ResumeTable::new();

    table.insert(0, 100);
    table.insert(2, 300);
    table.insert(4, 500);

    let entries: Vec<_> = table.iter().collect();

    assert_eq!(entries.len(), 3);
    assert!(entries.contains(&YieldPointEntry::new(0, 100)));
    assert!(entries.contains(&YieldPointEntry::new(2, 300)));
    assert!(entries.contains(&YieldPointEntry::new(4, 500)));
}

#[test]
fn test_resume_table_overwrite() {
    let mut table = ResumeTable::new();

    table.insert(0, 100);
    table.insert(0, 200);

    assert_eq!(table.get_pc(0), Some(200));
}

// ════════════════════════════════════════════════════════════════════════
// InlineResumeCache Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_inline_cache_new() {
    let cache = InlineResumeCache::new();
    assert!(cache.lookup(0x12345678).is_none());
}

#[test]
fn test_inline_cache_insert_lookup() {
    let mut cache = InlineResumeCache::new();
    let table = Arc::new(ResumeTable::new());

    cache.insert(0xDEADBEEF, Arc::clone(&table));

    let result = cache.lookup(0xDEADBEEF);
    assert!(result.is_some());
}

#[test]
fn test_inline_cache_miss() {
    let mut cache = InlineResumeCache::new();
    let table = Arc::new(ResumeTable::new());

    cache.insert(0xDEADBEEF, table);

    // Different key should miss
    assert!(cache.lookup(0xCAFEBABE).is_none());
}

#[test]
fn test_inline_cache_invalidate() {
    let mut cache = InlineResumeCache::new();
    let table = Arc::new(ResumeTable::new());

    cache.insert(0xDEADBEEF, table);
    assert!(cache.lookup(0xDEADBEEF).is_some());

    cache.invalidate(0xDEADBEEF);
    assert!(cache.lookup(0xDEADBEEF).is_none());
}

#[test]
fn test_inline_cache_clear() {
    let mut cache = InlineResumeCache::new();

    for i in 0..4 {
        cache.insert(i * 1000, Arc::new(ResumeTable::new()));
    }

    cache.clear();

    for i in 0..4 {
        assert!(cache.lookup(i * 1000).is_none());
    }
}

#[test]
fn test_inline_cache_collision() {
    let mut cache = InlineResumeCache::new();

    // Insert multiple entries that might collide
    for i in 0..10usize {
        let table = Arc::new(ResumeTable::new());
        cache.insert(i * 0x1000, table);
    }

    // At least some entries should be present
    // (exact behavior depends on hash collisions)
}

// ════════════════════════════════════════════════════════════════════════
// ResumeTableCache Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_cache_new() {
    let cache = ResumeTableCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cache_insert_yield_point() {
    let mut cache = ResumeTableCache::new();

    cache.insert_yield_point(0x1000, 0, 100);
    cache.insert_yield_point(0x1000, 1, 200);
    cache.insert_yield_point(0x1000, 2, 300);

    let table = cache.lookup(0x1000).expect("Should find table");
    assert_eq!(table.len(), 3);
    assert_eq!(table.get_pc(0), Some(100));
    assert_eq!(table.get_pc(1), Some(200));
    assert_eq!(table.get_pc(2), Some(300));
}

#[test]
fn test_cache_multiple_code_objects() {
    let mut cache = ResumeTableCache::new();

    cache.insert_yield_point(0x1000, 0, 100);
    cache.insert_yield_point(0x2000, 0, 200);
    cache.insert_yield_point(0x3000, 0, 300);

    assert_eq!(cache.len(), 3);

    // Verify each lookup independently to avoid borrow conflicts
    let pc1 = cache.lookup(0x1000).expect("Should find table 1").get_pc(0);
    assert_eq!(pc1, Some(100));

    let pc2 = cache.lookup(0x2000).expect("Should find table 2").get_pc(0);
    assert_eq!(pc2, Some(200));

    let pc3 = cache.lookup(0x3000).expect("Should find table 3").get_pc(0);
    assert_eq!(pc3, Some(300));
}

#[test]
fn test_cache_lookup_miss() {
    let mut cache = ResumeTableCache::new();

    cache.insert_yield_point(0x1000, 0, 100);

    assert!(cache.lookup(0x2000).is_none());
}

#[test]
fn test_cache_remove() {
    let mut cache = ResumeTableCache::new();

    cache.insert_yield_point(0x1000, 0, 100);
    cache.insert_yield_point(0x2000, 0, 200);

    cache.remove(0x1000);

    assert!(cache.lookup(0x1000).is_none());
    assert!(cache.lookup(0x2000).is_some());
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_cache_clear() {
    let mut cache = ResumeTableCache::new();

    cache.insert_yield_point(0x1000, 0, 100);
    cache.insert_yield_point(0x2000, 0, 200);

    cache.clear();

    assert!(cache.is_empty());
    assert!(cache.lookup(0x1000).is_none());
}

#[test]
fn test_cache_stats() {
    let mut cache = ResumeTableCache::new();

    cache.insert_yield_point(0x1000, 0, 100);

    // First lookup - inline miss, then promote
    let _ = cache.lookup(0x1000);
    // Second lookup - should be inline hit
    let _ = cache.lookup(0x1000);

    let stats = cache.stats();
    assert!(stats.lookups >= 2);
    assert!(stats.inline_hits >= 1);
}

#[test]
fn test_cache_inline_promotion() {
    let mut cache = ResumeTableCache::new();

    // Insert into overflow
    cache.insert_yield_point(0x1000, 0, 100);

    // First lookup promotes to inline
    let _ = cache.lookup(0x1000);

    // Second lookup should hit inline
    let _ = cache.lookup(0x1000);

    let stats = cache.stats();
    assert!(stats.inline_hits > 0);
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_resume_table_zero_pc() {
    let mut table = ResumeTable::new();

    table.insert(0, 0); // PC can be 0

    assert_eq!(table.get_pc(0), Some(0));
}

#[test]
fn test_resume_table_max_u32_resume_idx() {
    let mut table = ResumeTable::new();

    // This should trigger conversion to sparse
    table.insert(u32::MAX, 100);

    assert_eq!(table.get_pc(u32::MAX), Some(100));
    assert!(!table.is_dense());
}

#[test]
fn test_inline_cache_zero_ptr() {
    let mut cache = InlineResumeCache::new();
    let table = Arc::new(ResumeTable::new());

    // Zero is valid as a key
    cache.insert(0, table);
    assert!(cache.lookup(0).is_some());
}

#[test]
fn test_cache_get_or_create() {
    let mut cache = ResumeTableCache::new();

    // First call creates
    let _table1 = cache.get_or_create(0x1000);
    assert_eq!(cache.len(), 1);

    // Second call returns existing
    let _table2 = cache.get_or_create(0x1000);
    assert_eq!(cache.len(), 1); // Still 1
}

// ════════════════════════════════════════════════════════════════════════
// Performance Characteristics Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_inline_cache_entries_power_of_two() {
    assert!(INLINE_CACHE_ENTRIES.is_power_of_two());
}

#[test]
fn test_yield_point_entry_size() {
    assert_eq!(
        std::mem::size_of::<YieldPointEntry>(),
        8,
        "YieldPointEntry should be 8 bytes"
    );
}

#[test]
fn test_dense_storage_efficiency() {
    let mut table = ResumeTable::new();

    // Fill densely
    for i in 0..100u32 {
        table.insert(i, i * 10);
    }

    // Should still be dense
    assert!(table.is_dense());
    assert_eq!(table.len(), 100);
}

#[test]
fn test_sparse_conversion_threshold() {
    let mut table = ResumeTable::new();

    // Insert beyond dense threshold
    table.insert(MAX_DENSE_ENTRIES as u32, 100);

    // Should have converted to sparse
    assert!(!table.is_dense());
}
