use super::*;

#[test]
fn test_stats_recording() {
    let stats = GcStats::new();

    stats.record_allocation(1024);
    stats.record_allocation(2048);

    assert_eq!(stats.bytes_allocated.load(Ordering::Relaxed), 3072);
    assert_eq!(stats.objects_allocated.load(Ordering::Relaxed), 2);
}

#[test]
fn test_gc_timing() {
    let stats = GcStats::new();

    stats.record_minor_gc(Duration::from_micros(100));
    stats.record_minor_gc(Duration::from_micros(200));

    assert_eq!(stats.minor_collections.load(Ordering::Relaxed), 2);
    assert_eq!(stats.avg_minor_pause(), Duration::from_micros(150));
}

#[test]
fn test_format_bytes() {
    assert_eq!(format_bytes(512), "512 bytes");
    assert_eq!(format_bytes(2048), "2.00 KB");
    assert_eq!(format_bytes(1048576), "1.00 MB");
    assert_eq!(format_bytes(1073741824), "1.00 GB");
}
