use super::*;
use crate::config::GcConfig;
use crate::trace::NoopObjectTracer;

#[test]
fn test_minor_collector_creation() {
    let collector = MinorCollector::new();
    assert_eq!(collector.promotion_age, 2);
}

#[test]
fn test_minor_collector_custom_promotion_age() {
    let collector = MinorCollector::with_promotion_age(5);
    assert_eq!(collector.promotion_age(), 5);
}

#[test]
fn test_minor_collection_empty() {
    let mut collector = MinorCollector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();

    let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

    assert_eq!(result.bytes_freed, 0);
    assert_eq!(result.objects_promoted, 0);
    assert_eq!(result.live_bytes, 0);
}

#[test]
fn test_minor_collection_preserves_allocated_nursery() {
    let mut collector = MinorCollector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();
    let ptr = heap.alloc(64).expect("nursery allocation should succeed");

    let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

    assert_eq!(result.bytes_freed, 0);
    assert_eq!(result.live_bytes, 64);
    assert!(heap.nursery().in_from_space(ptr.as_ptr() as *const ()));
}

#[test]
fn test_minor_collection_roots_only_preserves_allocated_nursery() {
    let mut collector = MinorCollector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();
    let ptr = heap.alloc(128).expect("nursery allocation should succeed");

    let result = collector.collect_roots_only(&mut heap, &roots);

    assert_eq!(result.bytes_freed, 0);
    assert_eq!(result.objects_promoted, 0);
    assert_eq!(result.live_bytes, 128);
    assert!(heap.nursery().in_from_space(ptr.as_ptr() as *const ()));
}
