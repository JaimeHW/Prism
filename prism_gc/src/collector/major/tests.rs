use super::*;
use crate::config::GcConfig;
use crate::roots::RawHandle;
use crate::trace::NoopObjectTracer;

#[test]
fn test_major_collector_creation() {
    let collector = MajorCollector::new();
    assert!(collector.worklist.is_empty());
    assert!(collector.marked.is_empty());
}

#[test]
fn test_major_collection_empty() {
    let mut collector = MajorCollector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();

    let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

    // Empty heap, nothing to free
    assert_eq!(result.bytes_freed, 0);
    assert_eq!(result.objects_freed, 0);
    assert_eq!(result.objects_marked, 0);
}

#[test]
fn test_major_collection_roots_only() {
    let mut collector = MajorCollector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();

    let result = collector.collect_roots_only(&mut heap, &roots);

    assert_eq!(result.bytes_freed, 0);
    assert_eq!(result.objects_marked, 0);
}

#[test]
fn test_major_collection_reclaims_unrooted_old_space_block() {
    let mut collector = MajorCollector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();

    heap.alloc_tenured(128)
        .expect("old-space allocation should succeed");

    let result = collector.collect_roots_only(&mut heap, &roots);

    assert_eq!(result.bytes_freed, 128);
    assert_eq!(result.live_bytes, 0);
    assert_eq!(heap.old_space().usage(), 0);
}

#[test]
fn test_major_collection_preserves_rooted_old_space_block() {
    let mut collector = MajorCollector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();

    let ptr = heap
        .alloc_tenured(128)
        .expect("old-space allocation should succeed");
    roots.register_handle(RawHandle::new(ptr.as_ptr() as *const ()));

    let result = collector.collect_roots_only(&mut heap, &roots);

    assert_eq!(result.bytes_freed, 0);
    assert_eq!(result.live_bytes, 128);
    assert_eq!(heap.old_space().usage(), 128);
    assert_eq!(result.objects_marked, 1);
}

#[test]
fn test_mark_gray() {
    let mut collector = MajorCollector::new();

    // First mark should succeed
    assert!(collector.mark_gray(0x1000 as *const ()));
    assert!(collector.is_marked(0x1000 as *const ()));
    assert_eq!(collector.marked_count(), 1);

    // Second mark should return false (already marked)
    assert!(!collector.mark_gray(0x1000 as *const ()));
    assert_eq!(collector.marked_count(), 1);

    // Null should not be marked
    assert!(!collector.mark_gray(std::ptr::null()));
    assert_eq!(collector.marked_count(), 1);
}

#[test]
fn test_worklist_processing() {
    let mut collector = MajorCollector::new();

    // Mark several objects
    collector.mark_gray(0x1000 as *const ());
    collector.mark_gray(0x2000 as *const ());
    collector.mark_gray(0x3000 as *const ());

    // Worklist should have 3 entries
    assert_eq!(collector.worklist.len(), 3);

    // Pop and verify order (FIFO)
    assert_eq!(collector.worklist.pop_front(), Some(0x1000 as *const ()));
    assert_eq!(collector.worklist.pop_front(), Some(0x2000 as *const ()));
    assert_eq!(collector.worklist.pop_front(), Some(0x3000 as *const ()));
}
