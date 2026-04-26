use super::*;

// ════════════════════════════════════════════════════════════════════════
// InlineFrameStorage Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_inline_storage_new() {
    let storage = InlineFrameStorage::new();
    assert_eq!(storage.count(), 0);
    assert_eq!(storage.liveness(), 0);
    assert_eq!(storage.resume_index(), 0);
    assert_eq!(storage.capacity(), INLINE_REGISTER_CAPACITY);
}

#[test]
fn test_inline_storage_store_load() {
    let mut storage = InlineFrameStorage::new();

    storage.store(0, Value::int(42).unwrap());
    storage.store(1, Value::bool(true));
    storage.store(2, Value::none());

    assert_eq!(storage.count(), 3);

    unsafe {
        assert_eq!(storage.load(0).as_int(), Some(42));
        assert_eq!(storage.load(1).as_bool(), Some(true));
        assert!(storage.load(2).is_none());
    }
}

#[test]
fn test_inline_storage_metadata() {
    let mut storage = InlineFrameStorage::new();

    storage.set_metadata(0b1010_1010, 123);

    assert_eq!(storage.liveness(), 0b1010_1010);
    assert_eq!(storage.resume_index(), 123);
}

#[test]
fn test_inline_storage_clear() {
    let mut storage = InlineFrameStorage::new();

    storage.store(0, Value::int(1).unwrap());
    storage.store(1, Value::int(2).unwrap());
    storage.set_metadata(0xFF, 99);

    storage.clear();

    assert_eq!(storage.count(), 0);
    assert_eq!(storage.liveness(), 0);
    assert_eq!(storage.resume_index(), 0);
}

#[test]
fn test_inline_storage_cache_line_aligned() {
    assert_eq!(
        std::mem::align_of::<InlineFrameStorage>(),
        64,
        "InlineFrameStorage should be cache-line aligned"
    );
}

// ════════════════════════════════════════════════════════════════════════
// HeapFrameStorage Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_heap_storage_new() {
    let storage = HeapFrameStorage::with_capacity(16);
    assert_eq!(storage.count(), 0);
    assert!(storage.capacity() >= 16);
    assert_eq!(storage.liveness(), 0);
    assert_eq!(storage.resume_index(), 0);
}

#[test]
fn test_heap_storage_store_load() {
    let mut storage = HeapFrameStorage::with_capacity(4);

    storage.store(Value::int(10).unwrap());
    storage.store(Value::int(20).unwrap());
    storage.store(Value::int(30).unwrap());

    assert_eq!(storage.count(), 3);
    assert_eq!(storage.load(0).unwrap().as_int(), Some(10));
    assert_eq!(storage.load(1).unwrap().as_int(), Some(20));
    assert_eq!(storage.load(2).unwrap().as_int(), Some(30));
    assert!(storage.load(3).is_none());
}

#[test]
fn test_heap_storage_metadata() {
    let mut storage = HeapFrameStorage::with_capacity(4);

    storage.set_metadata(0xDEADBEEF, 456);

    assert_eq!(storage.liveness(), 0xDEADBEEF);
    assert_eq!(storage.resume_index(), 456);
}

#[test]
fn test_heap_storage_clear() {
    let mut storage = HeapFrameStorage::with_capacity(4);

    storage.store(Value::int(1).unwrap());
    storage.store(Value::int(2).unwrap());
    storage.set_metadata(0xFF, 99);

    storage.clear();

    assert_eq!(storage.count(), 0);
    assert_eq!(storage.liveness(), 0);
    assert_eq!(storage.resume_index(), 0);
}

#[test]
fn test_heap_storage_ensure_capacity() {
    let mut storage = HeapFrameStorage::with_capacity(4);
    let initial_cap = storage.capacity();

    storage.ensure_capacity(32);

    assert!(storage.capacity() >= 32);
    assert!(storage.capacity() >= initial_cap);
}

// ════════════════════════════════════════════════════════════════════════
// PooledFrame Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pooled_frame_inline() {
    let frame = PooledFrame::Inline(Box::new(InlineFrameStorage::new()));

    assert!(frame.is_inline());
    assert_eq!(frame.capacity(), INLINE_REGISTER_CAPACITY);
    assert_eq!(frame.count(), 0);
}

#[test]
fn test_pooled_frame_heap() {
    let frame = PooledFrame::Heap(Box::new(HeapFrameStorage::with_capacity(32)));

    assert!(!frame.is_inline());
    assert!(frame.capacity() >= 32);
    assert_eq!(frame.count(), 0);
}

#[test]
fn test_pooled_frame_store_restore_inline() {
    let mut frame = PooledFrame::Inline(Box::new(InlineFrameStorage::new()));

    let registers = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
    ];
    let liveness = 0b1011u64; // Registers 0, 1, 3 are live

    frame.set_metadata(liveness, 42);
    frame.store_registers(&registers, liveness);

    let mut restored = vec![Value::none(); 4];
    frame.restore_registers(&mut restored, liveness);

    assert_eq!(restored[0].as_int(), Some(1));
    assert_eq!(restored[1].as_int(), Some(2));
    assert!(restored[2].is_none()); // Was not live
    assert_eq!(restored[3].as_int(), Some(4));
}

#[test]
fn test_pooled_frame_store_restore_heap() {
    let mut frame = PooledFrame::Heap(Box::new(HeapFrameStorage::with_capacity(16)));

    let registers: Vec<Value> = (0..12).map(|i| Value::int(i).unwrap()).collect();
    let liveness = 0b1010_1010_1010u64;

    frame.set_metadata(liveness, 99);
    frame.store_registers(&registers, liveness);

    let mut restored = vec![Value::none(); 12];
    frame.restore_registers(&mut restored, liveness);

    // Check only even-indexed (live) registers
    for i in (1..12).step_by(2) {
        assert_eq!(restored[i].as_int(), Some(i as i64));
    }
}

// ════════════════════════════════════════════════════════════════════════
// GeneratorFramePool Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_pool_new() {
    let pool = GeneratorFramePool::new();
    assert!(pool.is_empty());
    assert_eq!(pool.len(), 0);
}

#[test]
fn test_pool_with_capacity() {
    let pool = GeneratorFramePool::with_capacity(8, 4);
    let stats = pool.stats();

    assert_eq!(stats.small_pool_size, 8);
    assert_eq!(stats.large_pool_size, 4);
    assert_eq!(stats.heap_allocations, 12);
}

#[test]
fn test_pool_allocate_small() {
    let mut pool = GeneratorFramePool::new();

    let frame = pool.allocate(4);
    assert!(frame.is_inline());
    assert!(frame.capacity() >= 4);

    let stats = pool.stats();
    assert_eq!(stats.allocations, 1);
    assert_eq!(stats.pool_misses, 1);
    assert_eq!(stats.heap_allocations, 1);
}

#[test]
fn test_pool_allocate_large() {
    let mut pool = GeneratorFramePool::new();

    let frame = pool.allocate(16);
    assert!(!frame.is_inline());
    assert!(frame.capacity() >= 16);

    let stats = pool.stats();
    assert_eq!(stats.allocations, 1);
    assert_eq!(stats.pool_misses, 1);
}

#[test]
fn test_pool_release_reuse() {
    let mut pool = GeneratorFramePool::new();

    let frame1 = pool.allocate(4);
    let ptr1 = frame1.as_ptr();

    pool.release(frame1);

    let frame2 = pool.allocate(4);
    let ptr2 = frame2.as_ptr();

    assert_eq!(ptr1, ptr2, "Released frame should be reused");

    let stats = pool.stats();
    assert_eq!(stats.allocations, 2);
    assert_eq!(stats.releases, 1);
    assert_eq!(stats.pool_hits, 1);
    assert_eq!(stats.pool_misses, 1);
}

#[test]
fn test_pool_separate_lists() {
    let mut pool = GeneratorFramePool::new();

    // Allocate and release small frame
    let small = pool.allocate(4);
    pool.release(small);

    // Allocate large - should not reuse small
    let large = pool.allocate(16);
    assert!(!large.is_inline());

    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 0); // Large allocation was a miss
    assert_eq!(stats.pool_misses, 2);
}

#[test]
fn test_pool_stats_tracking() {
    let mut pool = GeneratorFramePool::new();

    // Various operations
    let f1 = pool.allocate(4);
    let f2 = pool.allocate(4);
    let f3 = pool.allocate(16);

    pool.release(f1);
    pool.release(f2);

    let _f4 = pool.allocate(4);
    let _f5 = pool.allocate(4);

    let stats = pool.stats();
    assert_eq!(stats.allocations, 5);
    assert_eq!(stats.releases, 2);
    assert_eq!(stats.pool_hits, 2);
}

#[test]
fn test_pool_max_size_limit() {
    let mut pool = GeneratorFramePool::new();

    // Fill pool beyond limit
    let mut frames = Vec::new();
    for _ in 0..(MAX_SMALL_POOL_SIZE + 10) {
        frames.push(pool.allocate(4));
    }

    // Release all
    for frame in frames {
        pool.release(frame);
    }

    // Pool should be capped at max size
    let stats = pool.stats();
    assert!(stats.small_pool_size <= MAX_SMALL_POOL_SIZE);
    assert!(stats.evictions > 0);
}

#[test]
fn test_pool_clear() {
    let mut pool = GeneratorFramePool::with_capacity(8, 4);
    assert!(!pool.is_empty());

    pool.clear();

    assert!(pool.is_empty());
    let stats = pool.stats();
    assert_eq!(stats.small_pool_size, 0);
    assert_eq!(stats.large_pool_size, 0);
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_allocate_boundary_size() {
    let mut pool = GeneratorFramePool::new();

    // Exactly at inline limit
    let frame = pool.allocate(INLINE_REGISTER_CAPACITY);
    assert!(frame.is_inline());

    // One over limit
    let frame = pool.allocate(INLINE_REGISTER_CAPACITY + 1);
    assert!(!frame.is_inline());
}

#[test]
fn test_allocate_zero_registers() {
    let mut pool = GeneratorFramePool::new();

    let frame = pool.allocate(0);
    assert!(frame.is_inline()); // Should still use inline
}

#[test]
fn test_store_restore_empty() {
    let mut frame = PooledFrame::Inline(Box::new(InlineFrameStorage::new()));

    let registers: Vec<Value> = vec![];
    let liveness = 0u64;

    frame.store_registers(&registers, liveness);

    let mut restored: Vec<Value> = vec![];
    frame.restore_registers(&mut restored, liveness);

    assert_eq!(frame.count(), 0);
}

#[test]
fn test_store_restore_all_live() {
    let mut frame = PooledFrame::Heap(Box::new(HeapFrameStorage::with_capacity(8)));

    let registers: Vec<Value> = (0..8).map(|i| Value::int(i).unwrap()).collect();
    let liveness = 0xFFu64; // All 8 live

    frame.store_registers(&registers, liveness);

    let mut restored = vec![Value::none(); 8];
    frame.restore_registers(&mut restored, liveness);

    for i in 0..8 {
        assert_eq!(restored[i].as_int(), Some(i as i64));
    }
}

#[test]
fn test_pooled_frame_clear() {
    let mut frame = PooledFrame::Heap(Box::new(HeapFrameStorage::with_capacity(8)));

    frame.store_registers(&[Value::int(1).unwrap(), Value::int(2).unwrap()], 0b11);
    frame.set_metadata(0xFF, 42);

    frame.clear();

    assert_eq!(frame.count(), 0);
    assert_eq!(frame.liveness(), 0);
    assert_eq!(frame.resume_index(), 0);
}

// ════════════════════════════════════════════════════════════════════════
// Performance Characteristics Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_inline_storage_size() {
    // Ensure inline storage fits in reasonable size
    let size = std::mem::size_of::<InlineFrameStorage>();
    assert!(
        size <= 128,
        "InlineFrameStorage should be compact: {} bytes",
        size
    );
}

#[test]
fn test_heap_storage_overhead() {
    let size = std::mem::size_of::<HeapFrameStorage>();
    // Vec (24) + u64 (8) + u32 (4) + padding
    assert!(
        size <= 48,
        "HeapFrameStorage overhead should be minimal: {} bytes",
        size
    );
}

#[test]
fn test_pooled_frame_size() {
    let size = std::mem::size_of::<PooledFrame>();
    // Should be 8 bytes (pointer) + 8 bytes (tag + padding) at most
    assert!(
        size <= 16,
        "PooledFrame enum should be compact: {} bytes",
        size
    );
}
