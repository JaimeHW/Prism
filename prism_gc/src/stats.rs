//! GC statistics and metrics.
//!
//! Tracks allocation rates, collection times, and memory usage
//! for monitoring and tuning.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Statistics about garbage collection activity.
#[derive(Debug)]
pub struct GcStats {
    // =========================================================================
    // Allocation Statistics
    // =========================================================================
    /// Total bytes allocated since start.
    pub bytes_allocated: AtomicU64,
    /// Total objects allocated since start.
    pub objects_allocated: AtomicU64,
    /// Current live bytes (after last GC).
    pub live_bytes: AtomicU64,
    /// Current live objects (after last GC).
    pub live_objects: AtomicU64,

    // =========================================================================
    // Collection Statistics
    // =========================================================================
    /// Number of minor (nursery) collections.
    pub minor_collections: AtomicU64,
    /// Number of major (full) collections.
    pub major_collections: AtomicU64,
    /// Total time spent in minor GC (nanoseconds).
    pub minor_gc_time_ns: AtomicU64,
    /// Total time spent in major GC (nanoseconds).
    pub major_gc_time_ns: AtomicU64,

    // =========================================================================
    // Promotion Statistics
    // =========================================================================
    /// Total bytes promoted from nursery to tenured.
    pub bytes_promoted: AtomicU64,
    /// Total objects promoted from nursery to tenured.
    pub objects_promoted: AtomicU64,

    // =========================================================================
    // Memory Statistics
    // =========================================================================
    /// Current nursery usage in bytes.
    pub nursery_usage: AtomicU64,
    /// Current tenured usage in bytes.
    pub tenured_usage: AtomicU64,
    /// Current large object space usage in bytes.
    pub large_object_usage: AtomicU64,
}

impl GcStats {
    /// Create new empty statistics.
    pub const fn new() -> Self {
        Self {
            bytes_allocated: AtomicU64::new(0),
            objects_allocated: AtomicU64::new(0),
            live_bytes: AtomicU64::new(0),
            live_objects: AtomicU64::new(0),
            minor_collections: AtomicU64::new(0),
            major_collections: AtomicU64::new(0),
            minor_gc_time_ns: AtomicU64::new(0),
            major_gc_time_ns: AtomicU64::new(0),
            bytes_promoted: AtomicU64::new(0),
            objects_promoted: AtomicU64::new(0),
            nursery_usage: AtomicU64::new(0),
            tenured_usage: AtomicU64::new(0),
            large_object_usage: AtomicU64::new(0),
        }
    }

    /// Record an allocation.
    #[inline]
    pub fn record_allocation(&self, size: usize) {
        self.bytes_allocated
            .fetch_add(size as u64, Ordering::Relaxed);
        self.objects_allocated.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a minor GC.
    pub fn record_minor_gc(&self, duration: Duration) {
        self.minor_collections.fetch_add(1, Ordering::Relaxed);
        self.minor_gc_time_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record a major GC.
    pub fn record_major_gc(&self, duration: Duration) {
        self.major_collections.fetch_add(1, Ordering::Relaxed);
        self.major_gc_time_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record promotion from nursery to tenured.
    #[inline]
    pub fn record_promotion(&self, bytes: usize) {
        self.bytes_promoted
            .fetch_add(bytes as u64, Ordering::Relaxed);
        self.objects_promoted.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total GC time.
    pub fn total_gc_time(&self) -> Duration {
        let minor_ns = self.minor_gc_time_ns.load(Ordering::Relaxed);
        let major_ns = self.major_gc_time_ns.load(Ordering::Relaxed);
        Duration::from_nanos(minor_ns + major_ns)
    }

    /// Get average minor GC pause time.
    pub fn avg_minor_pause(&self) -> Duration {
        let count = self.minor_collections.load(Ordering::Relaxed);
        if count == 0 {
            return Duration::ZERO;
        }
        let total_ns = self.minor_gc_time_ns.load(Ordering::Relaxed);
        Duration::from_nanos(total_ns / count)
    }

    /// Get average major GC pause time.
    pub fn avg_major_pause(&self) -> Duration {
        let count = self.major_collections.load(Ordering::Relaxed);
        if count == 0 {
            return Duration::ZERO;
        }
        let total_ns = self.major_gc_time_ns.load(Ordering::Relaxed);
        Duration::from_nanos(total_ns / count)
    }

    /// Get total heap usage in bytes.
    pub fn heap_usage(&self) -> u64 {
        self.nursery_usage.load(Ordering::Relaxed)
            + self.tenured_usage.load(Ordering::Relaxed)
            + self.large_object_usage.load(Ordering::Relaxed)
    }

    /// Get allocation rate (bytes per second) since start.
    pub fn allocation_rate(&self, elapsed: Duration) -> f64 {
        let bytes = self.bytes_allocated.load(Ordering::Relaxed) as f64;
        let seconds = elapsed.as_secs_f64();
        if seconds > 0.0 {
            bytes / seconds
        } else {
            0.0
        }
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.bytes_allocated.store(0, Ordering::Relaxed);
        self.objects_allocated.store(0, Ordering::Relaxed);
        self.live_bytes.store(0, Ordering::Relaxed);
        self.live_objects.store(0, Ordering::Relaxed);
        self.minor_collections.store(0, Ordering::Relaxed);
        self.major_collections.store(0, Ordering::Relaxed);
        self.minor_gc_time_ns.store(0, Ordering::Relaxed);
        self.major_gc_time_ns.store(0, Ordering::Relaxed);
        self.bytes_promoted.store(0, Ordering::Relaxed);
        self.objects_promoted.store(0, Ordering::Relaxed);
    }

    /// Print a summary of GC statistics.
    pub fn print_summary(&self) {
        eprintln!("=== GC Statistics ===");
        eprintln!(
            "Allocations: {} objects, {} bytes",
            self.objects_allocated.load(Ordering::Relaxed),
            format_bytes(self.bytes_allocated.load(Ordering::Relaxed))
        );
        eprintln!(
            "Live: {} objects, {} bytes",
            self.live_objects.load(Ordering::Relaxed),
            format_bytes(self.live_bytes.load(Ordering::Relaxed))
        );
        eprintln!(
            "Collections: {} minor, {} major",
            self.minor_collections.load(Ordering::Relaxed),
            self.major_collections.load(Ordering::Relaxed)
        );
        eprintln!(
            "GC Time: {:?} total ({:?} avg minor, {:?} avg major)",
            self.total_gc_time(),
            self.avg_minor_pause(),
            self.avg_major_pause()
        );
        eprintln!(
            "Promotions: {} objects, {} bytes",
            self.objects_promoted.load(Ordering::Relaxed),
            format_bytes(self.bytes_promoted.load(Ordering::Relaxed))
        );
    }
}

impl Default for GcStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Format bytes in human-readable form.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Timer for measuring GC phases.
pub struct GcTimer {
    start: Instant,
    label: &'static str,
}

impl GcTimer {
    /// Start a new timer with the given label.
    pub fn start(label: &'static str) -> Self {
        Self {
            start: Instant::now(),
            label,
        }
    }

    /// Stop the timer and return the elapsed duration.
    pub fn stop(self) -> Duration {
        let elapsed = self.start.elapsed();
        #[cfg(feature = "trace")]
        eprintln!("GC {}: {:?}", self.label, elapsed);
        elapsed
    }
}

#[cfg(test)]
mod tests {
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
}
