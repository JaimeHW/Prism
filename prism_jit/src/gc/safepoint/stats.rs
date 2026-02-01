//! Performance statistics for GC safepoints.
//!
//! Tracks timing, frequency, and efficiency of safepoint operations.
//!
//! # Metrics
//!
//! - Stop-the-world request count
//! - Time to achieve full stop
//! - Total pause duration
//! - Per-thread wait times

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

// =============================================================================
// SafepointStats
// =============================================================================

/// Statistics for safepoint operations.
///
/// All counters are updated atomically for thread safety.
#[derive(Debug)]
pub struct SafepointStats {
    /// Total number of stop-the-world requests.
    stw_requests: AtomicU64,

    /// Number of successful stop-the-world operations.
    stw_achieved: AtomicU64,

    /// Total time spent achieving stop-the-world (nanoseconds).
    stw_achieve_time_ns: AtomicU64,

    /// Total pause duration (nanoseconds).
    stw_pause_time_ns: AtomicU64,

    /// Maximum stop-the-world time seen (nanoseconds).
    max_stw_time_ns: AtomicU64,

    /// Minimum stop-the-world time seen (nanoseconds).
    min_stw_time_ns: AtomicU64,

    /// Number of safepoint polls executed.
    polls_executed: AtomicU64,

    /// Number of safepoint traps triggered.
    traps_triggered: AtomicU64,
}

impl SafepointStats {
    /// Create new zeroed statistics.
    pub fn new() -> Self {
        SafepointStats {
            stw_requests: AtomicU64::new(0),
            stw_achieved: AtomicU64::new(0),
            stw_achieve_time_ns: AtomicU64::new(0),
            stw_pause_time_ns: AtomicU64::new(0),
            max_stw_time_ns: AtomicU64::new(0),
            min_stw_time_ns: AtomicU64::new(u64::MAX),
            polls_executed: AtomicU64::new(0),
            traps_triggered: AtomicU64::new(0),
        }
    }

    /// Record a stop-the-world request.
    #[inline]
    pub fn record_stw_request(&self) {
        self.stw_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record successful stop-the-world achievement.
    #[inline]
    pub fn record_stw_achieved(&self, time: Duration) {
        self.stw_achieved.fetch_add(1, Ordering::Relaxed);
        let nanos = time.as_nanos() as u64;
        self.stw_achieve_time_ns.fetch_add(nanos, Ordering::Relaxed);
    }

    /// Record total stop-the-world duration.
    #[inline]
    pub fn record_stw_duration(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        self.stw_pause_time_ns.fetch_add(nanos, Ordering::Relaxed);

        // Update max
        let mut current = self.max_stw_time_ns.load(Ordering::Relaxed);
        while nanos > current {
            match self.max_stw_time_ns.compare_exchange_weak(
                current,
                nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }

        // Update min
        current = self.min_stw_time_ns.load(Ordering::Relaxed);
        while nanos < current {
            match self.min_stw_time_ns.compare_exchange_weak(
                current,
                nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }

    /// Record a safepoint poll execution.
    #[inline]
    pub fn record_poll(&self) {
        self.polls_executed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a safepoint trap.
    #[inline]
    pub fn record_trap(&self) {
        self.traps_triggered.fetch_add(1, Ordering::Relaxed);
    }

    // =========================================================================
    // Getters
    // =========================================================================

    /// Get total stop-the-world requests.
    #[inline]
    pub fn stw_requests(&self) -> u64 {
        self.stw_requests.load(Ordering::Relaxed)
    }

    /// Get successful stop-the-world count.
    #[inline]
    pub fn stw_achieved(&self) -> u64 {
        self.stw_achieved.load(Ordering::Relaxed)
    }

    /// Get total polls executed.
    #[inline]
    pub fn polls_executed(&self) -> u64 {
        self.polls_executed.load(Ordering::Relaxed)
    }

    /// Get total traps triggered.
    #[inline]
    pub fn traps_triggered(&self) -> u64 {
        self.traps_triggered.load(Ordering::Relaxed)
    }

    /// Get average time to achieve stop-the-world.
    #[inline]
    pub fn avg_achieve_time(&self) -> Duration {
        let total = self.stw_achieve_time_ns.load(Ordering::Relaxed);
        let count = self.stw_achieved.load(Ordering::Relaxed);
        if count == 0 {
            return Duration::ZERO;
        }
        Duration::from_nanos(total / count)
    }

    /// Get average pause duration.
    #[inline]
    pub fn avg_pause_time(&self) -> Duration {
        let total = self.stw_pause_time_ns.load(Ordering::Relaxed);
        let count = self.stw_achieved.load(Ordering::Relaxed);
        if count == 0 {
            return Duration::ZERO;
        }
        Duration::from_nanos(total / count)
    }

    /// Get maximum pause time.
    #[inline]
    pub fn max_pause_time(&self) -> Duration {
        let nanos = self.max_stw_time_ns.load(Ordering::Relaxed);
        if nanos == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos(nanos)
        }
    }

    /// Get minimum pause time.
    #[inline]
    pub fn min_pause_time(&self) -> Duration {
        let nanos = self.min_stw_time_ns.load(Ordering::Relaxed);
        if nanos == u64::MAX {
            Duration::ZERO
        } else {
            Duration::from_nanos(nanos)
        }
    }

    /// Get total pause time.
    #[inline]
    pub fn total_pause_time(&self) -> Duration {
        Duration::from_nanos(self.stw_pause_time_ns.load(Ordering::Relaxed))
    }

    /// Calculate trap rate (traps per poll).
    #[inline]
    pub fn trap_rate(&self) -> f64 {
        let polls = self.polls_executed.load(Ordering::Relaxed);
        let traps = self.traps_triggered.load(Ordering::Relaxed);
        if polls == 0 {
            return 0.0;
        }
        traps as f64 / polls as f64
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.stw_requests.store(0, Ordering::Relaxed);
        self.stw_achieved.store(0, Ordering::Relaxed);
        self.stw_achieve_time_ns.store(0, Ordering::Relaxed);
        self.stw_pause_time_ns.store(0, Ordering::Relaxed);
        self.max_stw_time_ns.store(0, Ordering::Relaxed);
        self.min_stw_time_ns.store(u64::MAX, Ordering::Relaxed);
        self.polls_executed.store(0, Ordering::Relaxed);
        self.traps_triggered.store(0, Ordering::Relaxed);
    }
}

impl Default for SafepointStats {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SafepointStats {
    fn clone(&self) -> Self {
        SafepointStats {
            stw_requests: AtomicU64::new(self.stw_requests.load(Ordering::Relaxed)),
            stw_achieved: AtomicU64::new(self.stw_achieved.load(Ordering::Relaxed)),
            stw_achieve_time_ns: AtomicU64::new(self.stw_achieve_time_ns.load(Ordering::Relaxed)),
            stw_pause_time_ns: AtomicU64::new(self.stw_pause_time_ns.load(Ordering::Relaxed)),
            max_stw_time_ns: AtomicU64::new(self.max_stw_time_ns.load(Ordering::Relaxed)),
            min_stw_time_ns: AtomicU64::new(self.min_stw_time_ns.load(Ordering::Relaxed)),
            polls_executed: AtomicU64::new(self.polls_executed.load(Ordering::Relaxed)),
            traps_triggered: AtomicU64::new(self.traps_triggered.load(Ordering::Relaxed)),
        }
    }
}

// =============================================================================
// Display Implementation
// =============================================================================

impl std::fmt::Display for SafepointStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Safepoint Statistics:")?;
        writeln!(f, "  STW Requests:    {}", self.stw_requests())?;
        writeln!(f, "  STW Achieved:    {}", self.stw_achieved())?;
        writeln!(f, "  Avg Achieve:     {:?}", self.avg_achieve_time())?;
        writeln!(f, "  Avg Pause:       {:?}", self.avg_pause_time())?;
        writeln!(f, "  Max Pause:       {:?}", self.max_pause_time())?;
        writeln!(f, "  Min Pause:       {:?}", self.min_pause_time())?;
        writeln!(f, "  Total Pause:     {:?}", self.total_pause_time())?;
        writeln!(f, "  Polls Executed:  {}", self.polls_executed())?;
        writeln!(f, "  Traps Triggered: {}", self.traps_triggered())?;
        writeln!(f, "  Trap Rate:       {:.6}", self.trap_rate())?;
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_new() {
        let stats = SafepointStats::new();
        assert_eq!(stats.stw_requests(), 0);
        assert_eq!(stats.stw_achieved(), 0);
        assert_eq!(stats.polls_executed(), 0);
        assert_eq!(stats.traps_triggered(), 0);
    }

    #[test]
    fn test_stats_record_request() {
        let stats = SafepointStats::new();
        stats.record_stw_request();
        stats.record_stw_request();
        assert_eq!(stats.stw_requests(), 2);
    }

    #[test]
    fn test_stats_record_achieved() {
        let stats = SafepointStats::new();
        stats.record_stw_achieved(Duration::from_micros(100));
        stats.record_stw_achieved(Duration::from_micros(200));
        assert_eq!(stats.stw_achieved(), 2);
        assert_eq!(stats.avg_achieve_time(), Duration::from_micros(150));
    }

    #[test]
    fn test_stats_record_duration() {
        let stats = SafepointStats::new();
        stats.record_stw_achieved(Duration::ZERO);
        stats.record_stw_duration(Duration::from_millis(10));
        stats.record_stw_achieved(Duration::ZERO);
        stats.record_stw_duration(Duration::from_millis(20));

        assert_eq!(stats.min_pause_time(), Duration::from_millis(10));
        assert_eq!(stats.max_pause_time(), Duration::from_millis(20));
    }

    #[test]
    fn test_stats_record_polls() {
        let stats = SafepointStats::new();
        for _ in 0..100 {
            stats.record_poll();
        }
        stats.record_trap();

        assert_eq!(stats.polls_executed(), 100);
        assert_eq!(stats.traps_triggered(), 1);
        assert!((stats.trap_rate() - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_stats_reset() {
        let stats = SafepointStats::new();
        stats.record_stw_request();
        stats.record_poll();
        stats.reset();

        assert_eq!(stats.stw_requests(), 0);
        assert_eq!(stats.polls_executed(), 0);
    }

    #[test]
    fn test_stats_clone() {
        let stats = SafepointStats::new();
        stats.record_stw_request();
        stats.record_stw_request();

        let cloned = stats.clone();
        assert_eq!(cloned.stw_requests(), 2);
    }

    #[test]
    fn test_stats_display() {
        let stats = SafepointStats::new();
        stats.record_stw_request();

        let display = format!("{}", stats);
        assert!(display.contains("Safepoint Statistics"));
        assert!(display.contains("STW Requests"));
    }
}
