use super::*;

#[test]
fn test_tlab_default() {
    let tlab = Tlab::with_default_size();
    assert_eq!(tlab.size(), DEFAULT_TLAB_SIZE);
    assert!(tlab.is_empty());
    assert_eq!(tlab.free(), 0);
}

#[test]
fn test_tlab_config_default() {
    let config = TlabConfig::default();
    assert_eq!(config.initial_size, DEFAULT_TLAB_SIZE);
    assert_eq!(config.max_size, MAX_TLAB_SIZE);
}

#[test]
fn test_tlab_size_clamping() {
    // Too small
    let tlab = Tlab::new(100);
    assert_eq!(tlab.size(), MIN_TLAB_SIZE);

    // Too large
    let tlab = Tlab::new(10 * 1024 * 1024);
    assert_eq!(tlab.size(), MAX_TLAB_SIZE);

    // Just right
    let tlab = Tlab::new(32 * 1024);
    assert_eq!(tlab.size(), 32 * 1024);
}

#[test]
fn test_tlab_alloc_fast() {
    let mut tlab = Tlab::new(1024);

    // Simulate refill with a mock region
    let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
    let region = unsafe { std::alloc::alloc_zeroed(layout) };
    let end = unsafe { region.add(1024) };
    tlab.refill_from_region(region, end);

    // First allocation
    let ptr1 = tlab.alloc_fast(64).expect("Alloc 1 failed");
    assert_eq!(tlab.used(), 64);

    // Second allocation
    let ptr2 = tlab.alloc_fast(64).expect("Alloc 2 failed");
    assert_eq!(tlab.used(), 128);

    // Pointers should be consecutive
    assert_eq!(ptr2.as_ptr() as usize - ptr1.as_ptr() as usize, 64);

    // Cleanup
    unsafe { std::alloc::dealloc(region, layout) };
}

#[test]
fn test_tlab_exhaustion() {
    let mut tlab = Tlab::new(MIN_TLAB_SIZE);

    // Simulate refill with a small region
    let layout = std::alloc::Layout::from_size_align(128, 8).unwrap();
    let region = unsafe { std::alloc::alloc_zeroed(layout) };
    let end = unsafe { region.add(128) };
    tlab.refill_from_region(region, end);

    // Fill the TLAB
    let _ = tlab.alloc_fast(64);
    let _ = tlab.alloc_fast(64);

    // Should fail - TLAB is full
    assert!(tlab.alloc_fast(1).is_none());

    // Cleanup
    unsafe { std::alloc::dealloc(region, layout) };
}

#[test]
fn test_tlab_alignment() {
    let mut tlab = Tlab::new(MIN_TLAB_SIZE);

    let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
    let region = unsafe { std::alloc::alloc_zeroed(layout) };
    let end = unsafe { region.add(1024) };
    tlab.refill_from_region(region, end);

    // Allocate odd size
    let ptr1 = tlab.alloc_fast(7).expect("Alloc failed");
    assert_eq!(tlab.used(), 8); // Should be aligned to 8

    // Next allocation should be 8-byte aligned
    let ptr2 = tlab.alloc_fast(1).expect("Alloc failed");
    assert_eq!(ptr2.as_ptr() as usize % 8, 0);

    // Cleanup
    unsafe { std::alloc::dealloc(region, layout) };
}

#[test]
fn test_tlab_contains() {
    let mut tlab = Tlab::new(MIN_TLAB_SIZE);

    let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
    let region = unsafe { std::alloc::alloc_zeroed(layout) };
    let end = unsafe { region.add(1024) };
    tlab.refill_from_region(region, end);

    let ptr = tlab.alloc_fast(64).expect("Alloc failed");
    assert!(tlab.contains(ptr.as_ptr() as *const ()));

    // Outside the region
    let outside = std::ptr::null::<()>();
    assert!(!tlab.contains(outside));

    // Cleanup
    unsafe { std::alloc::dealloc(region, layout) };
}

#[test]
fn test_tlab_refill_statistics() {
    let mut tlab = Tlab::new(MIN_TLAB_SIZE);
    assert_eq!(tlab.refill_count(), 0);

    let layout = std::alloc::Layout::from_size_align(128, 8).unwrap();
    let region = unsafe { std::alloc::alloc_zeroed(layout) };
    let end = unsafe { region.add(128) };

    // First refill
    tlab.refill_from_region(region, end);
    assert_eq!(tlab.refill_count(), 1);

    // Allocate
    let _ = tlab.alloc_fast(64);
    assert_eq!(tlab.total_bytes_allocated(), 64);

    // Cleanup
    unsafe { std::alloc::dealloc(region, layout) };
}

#[test]
fn test_tlab_retire() {
    let mut tlab = Tlab::new(MIN_TLAB_SIZE);

    let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
    let region = unsafe { std::alloc::alloc_zeroed(layout) };
    let end = unsafe { region.add(1024) };
    tlab.refill_from_region(region, end);

    // Allocate some
    let _ = tlab.alloc_fast(256);

    // Retire
    let (start, used_end) = tlab.retire().expect("Should have region");
    assert_eq!(start, region);
    assert_eq!(used_end as usize - region as usize, 256);

    // TLAB should be empty after retire
    assert!(tlab.is_empty());

    // Cleanup
    unsafe { std::alloc::dealloc(region, layout) };
}

#[test]
fn test_tlab_stats() {
    let mut tlab = Tlab::new(MIN_TLAB_SIZE);

    let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
    let region = unsafe { std::alloc::alloc_zeroed(layout) };
    let end = unsafe { region.add(1024) };
    tlab.refill_from_region(region, end);

    let _ = tlab.alloc_fast(128);

    let stats = tlab.stats();
    assert_eq!(stats.current_used, 128);
    assert_eq!(stats.current_free, 1024 - 128);
    assert_eq!(stats.total_allocated, 128);
    assert_eq!(stats.refill_count, 1);

    // Cleanup
    unsafe { std::alloc::dealloc(region, layout) };
}

#[test]
fn test_tlab_waste_tracking() {
    let mut tlab = Tlab::new(MIN_TLAB_SIZE);

    let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
    let region1 = unsafe { std::alloc::alloc_zeroed(layout) };
    let end1 = unsafe { region1.add(1024) };
    tlab.refill_from_region(region1, end1);

    // Allocate half
    let _ = tlab.alloc_fast(512);

    // Refill again - should report waste
    let region2 = unsafe { std::alloc::alloc_zeroed(layout) };
    let end2 = unsafe { region2.add(1024) };
    let waste = tlab.refill_from_region(region2, end2);
    assert_eq!(waste, 512); // The unused portion

    // Cleanup
    unsafe { std::alloc::dealloc(region1, layout) };
    unsafe { std::alloc::dealloc(region2, layout) };
}
