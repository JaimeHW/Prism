use super::*;

// -------------------------------------------------------------------------
// BufferSizeClass Tests
// -------------------------------------------------------------------------

#[test]
fn test_size_class_sizes() {
    assert_eq!(BufferSizeClass::Small.size(), 1024);
    assert_eq!(BufferSizeClass::Medium.size(), 8192);
    assert_eq!(BufferSizeClass::Large.size(), 65536);
}

#[test]
fn test_size_class_for_size() {
    assert_eq!(BufferSizeClass::for_size(100), BufferSizeClass::Small);
    assert_eq!(BufferSizeClass::for_size(1024), BufferSizeClass::Small);
    assert_eq!(BufferSizeClass::for_size(1025), BufferSizeClass::Medium);
    assert_eq!(BufferSizeClass::for_size(8192), BufferSizeClass::Medium);
    assert_eq!(BufferSizeClass::for_size(8193), BufferSizeClass::Large);
    assert_eq!(BufferSizeClass::for_size(100000), BufferSizeClass::Large);
}

// -------------------------------------------------------------------------
// PooledBuffer Tests
// -------------------------------------------------------------------------

#[test]
fn test_pooled_buffer_creation() {
    let buf = PooledBuffer::new(BufferSizeClass::Small);
    assert_eq!(buf.capacity(), SMALL_BUFFER_SIZE);
    assert_eq!(buf.size_class(), BufferSizeClass::Small);
}

#[test]
fn test_pooled_buffer_as_slice() {
    let buf = PooledBuffer::new(BufferSizeClass::Small);
    assert_eq!(buf.as_slice().len(), SMALL_BUFFER_SIZE);
}

#[test]
fn test_pooled_buffer_mutability() {
    let mut buf = PooledBuffer::new(BufferSizeClass::Small);
    buf.as_mut_slice()[0] = 42;
    assert_eq!(buf.as_slice()[0], 42);
}

#[test]
fn test_pooled_buffer_deref() {
    let buf = PooledBuffer::new(BufferSizeClass::Small);
    let slice: &[u8] = &*buf;
    assert_eq!(slice.len(), SMALL_BUFFER_SIZE);
}

#[test]
fn test_pooled_buffer_clear() {
    let mut buf = PooledBuffer::new(BufferSizeClass::Small);
    buf.as_mut_slice()[0] = 255;
    buf.clear();
    assert_eq!(buf.as_slice()[0], 0);
}

// -------------------------------------------------------------------------
// BufferPool Tests
// -------------------------------------------------------------------------

#[test]
fn test_pool_acquire_fresh() {
    let mut pool = BufferPool::new();
    let buf = pool.acquire(BufferSizeClass::Medium);
    assert_eq!(buf.capacity(), DEFAULT_BUFFER_SIZE);
}

#[test]
fn test_pool_clear() {
    let mut pool = BufferPool::new();

    // Pre-populate with a buffer
    let buf = Vec::with_capacity(DEFAULT_BUFFER_SIZE);
    pool.medium.push(buf);

    assert_eq!(pool.pooled_count(), 1);
    pool.clear();
    assert_eq!(pool.pooled_count(), 0);
}

#[test]
fn test_pool_max_buffers() {
    let mut pool = BufferPool::new();

    // Fill the pool to max
    for _ in 0..MAX_POOLED_BUFFERS {
        let mut buf = Vec::with_capacity(DEFAULT_BUFFER_SIZE);
        buf.resize(DEFAULT_BUFFER_SIZE, 0);
        pool.return_buffer(buf, BufferSizeClass::Medium);
    }

    assert_eq!(pool.medium.len(), MAX_POOLED_BUFFERS);

    // Try to add one more - should be dropped
    let mut extra = Vec::with_capacity(DEFAULT_BUFFER_SIZE);
    extra.resize(DEFAULT_BUFFER_SIZE, 0);
    pool.return_buffer(extra, BufferSizeClass::Medium);

    // Still at max
    assert_eq!(pool.medium.len(), MAX_POOLED_BUFFERS);
}

#[test]
fn test_pool_wrong_capacity_rejected() {
    let mut pool = BufferPool::new();

    // Buffer with wrong capacity shouldn't be pooled
    let buf = Vec::with_capacity(100); // Wrong size
    pool.return_buffer(buf, BufferSizeClass::Medium);

    assert_eq!(pool.medium.len(), 0);
}

// -------------------------------------------------------------------------
// Thread-Local API Tests
// -------------------------------------------------------------------------

#[test]
fn test_acquire_buffer_api() {
    let buf = acquire_buffer(BufferSizeClass::Small);
    assert_eq!(buf.capacity(), SMALL_BUFFER_SIZE);
}

#[test]
fn test_acquire_buffer_sized_api() {
    let buf = acquire_buffer_sized(5000);
    assert_eq!(buf.capacity(), DEFAULT_BUFFER_SIZE);
}

#[test]
fn test_clear_pool_api() {
    // Acquire and drop to populate pool
    let _ = acquire_buffer(BufferSizeClass::Medium);

    clear_pool();

    let (count, _) = pool_stats();
    assert_eq!(count, 0);
}

// -------------------------------------------------------------------------
// Pool Statistics Tests
// -------------------------------------------------------------------------

#[test]
fn test_pooled_bytes_calculation() {
    let mut pool = BufferPool::new();

    let mut small = Vec::with_capacity(SMALL_BUFFER_SIZE);
    small.resize(SMALL_BUFFER_SIZE, 0);
    pool.return_buffer(small, BufferSizeClass::Small);

    assert_eq!(pool.pooled_bytes(), SMALL_BUFFER_SIZE);
}

// -------------------------------------------------------------------------
// Stress Tests
// -------------------------------------------------------------------------

#[test]
fn test_many_acquisitions() {
    for _ in 0..100 {
        let buf = acquire_buffer(BufferSizeClass::Medium);
        assert_eq!(buf.capacity(), DEFAULT_BUFFER_SIZE);
    }
}

#[test]
fn test_mixed_size_classes() {
    let bufs: Vec<_> = [
        BufferSizeClass::Small,
        BufferSizeClass::Medium,
        BufferSizeClass::Large,
        BufferSizeClass::Small,
        BufferSizeClass::Medium,
    ]
    .iter()
    .map(|&class| acquire_buffer(class))
    .collect();

    assert_eq!(bufs[0].capacity(), SMALL_BUFFER_SIZE);
    assert_eq!(bufs[1].capacity(), DEFAULT_BUFFER_SIZE);
    assert_eq!(bufs[2].capacity(), LARGE_BUFFER_SIZE);
}
