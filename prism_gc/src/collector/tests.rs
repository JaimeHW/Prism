use super::*;
use crate::config::GcConfig;
use crate::trace::NoopObjectTracer;

#[test]
fn test_collector_creation() {
    let collector = Collector::new();
    assert_eq!(collector.promotion_age(), 2);
}

#[test]
fn test_collector_custom_promotion_age() {
    let collector = Collector::with_promotion_age(5);
    assert_eq!(collector.promotion_age(), 5);
}

#[test]
fn test_collect_minor_empty() {
    let mut collector = Collector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();

    let result = collector.collect_minor(&mut heap, &roots, &NoopObjectTracer);

    assert_eq!(result.collection_type, CollectionType::Minor);
    assert_eq!(result.bytes_freed, 0);
}

#[test]
fn test_collect_major_empty() {
    let mut collector = Collector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();

    let result = collector.collect_major(&mut heap, &roots, &NoopObjectTracer);

    assert_eq!(result.collection_type, CollectionType::Major);
    assert_eq!(result.bytes_freed, 0);
}

#[test]
fn test_collect_roots_only() {
    let mut collector = Collector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();

    let minor_result = collector.collect_minor_roots_only(&mut heap, &roots);
    let major_result = collector.collect_major_roots_only(&mut heap, &roots);

    assert_eq!(minor_result.collection_type, CollectionType::Minor);
    assert_eq!(major_result.collection_type, CollectionType::Major);
}

#[test]
fn test_auto_select_collection() {
    let mut collector = Collector::new();
    let mut heap = GcHeap::new(GcConfig::default());
    let roots = RootSet::new();

    // With empty heap, should choose minor
    let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
    assert_eq!(result.collection_type, CollectionType::Minor);
}
