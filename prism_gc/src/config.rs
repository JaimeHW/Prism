//! GC configuration parameters.
//!
//! All sizes and thresholds are tunable for different workloads.
//! Default values are optimized for typical Python programs.

/// Configuration for the garbage collector.
///
/// # Example
///
/// ```ignore
/// use prism_gc::GcConfig;
///
/// // High-throughput configuration for batch processing
/// let config = GcConfig {
///     nursery_size: 16 * 1024 * 1024,  // 16MB nursery
///     promotion_age: 3,                 // Promote after 3 survivals
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct GcConfig {
    // =========================================================================
    // Nursery (Young Generation)
    // =========================================================================
    /// Size of each nursery semi-space in bytes.
    ///
    /// Total nursery memory is 2x this value (from-space + to-space).
    /// Larger nurseries reduce minor GC frequency but increase pause times.
    ///
    /// Default: 4MB (8MB total for both semi-spaces)
    pub nursery_size: usize,

    /// Number of minor GC survivals before promoting to tenured.
    ///
    /// Objects that survive this many minor collections are promoted
    /// to the old generation. Lower values reduce nursery pressure
    /// but may promote short-lived objects prematurely.
    ///
    /// Default: 2
    pub promotion_age: u8,

    // =========================================================================
    // Tenured (Old Generation)
    // =========================================================================
    /// Initial size of the old generation in bytes.
    ///
    /// The old generation grows dynamically as objects are promoted.
    ///
    /// Default: 16MB
    pub initial_old_size: usize,

    /// Maximum old generation size before OOM.
    ///
    /// Set to 0 for unlimited (bounded only by system memory).
    ///
    /// Default: 0 (unlimited)
    pub max_old_size: usize,

    /// Growth factor when old generation needs more space.
    ///
    /// When the old generation fills up, it grows by this factor.
    ///
    /// Default: 1.5
    pub old_growth_factor: f64,

    /// Fragmentation threshold for triggering compaction.
    ///
    /// If fragmentation exceeds this ratio, a compacting GC is performed.
    /// Set to 1.0 to disable compaction.
    ///
    /// Default: 0.3 (30% fragmentation triggers compaction)
    pub compaction_threshold: f64,

    // =========================================================================
    // Large Object Space
    // =========================================================================
    /// Minimum size for large object allocation.
    ///
    /// Objects larger than this are allocated directly in the large
    /// object space to avoid copying overhead during minor GC.
    ///
    /// Default: 8KB
    pub large_object_threshold: usize,

    // =========================================================================
    // Collection Triggers
    // =========================================================================
    /// Bytes allocated in nursery before triggering minor GC.
    ///
    /// This is effectively the nursery size, but can be set lower
    /// to trigger more frequent collections for testing.
    ///
    /// Default: Same as nursery_size
    pub minor_gc_trigger: usize,

    /// Old generation usage ratio that triggers major GC.
    ///
    /// When old gen usage exceeds this ratio, a major collection
    /// is scheduled.
    ///
    /// Default: 0.75 (75% usage triggers major GC)
    pub major_gc_threshold: f64,

    // =========================================================================
    // Block Sizes
    // =========================================================================
    /// Size of memory blocks in the old generation.
    ///
    /// Blocks are the unit of allocation and sweeping in the old gen.
    ///
    /// Default: 16KB
    pub block_size: usize,

    /// Card table granularity for write barriers.
    ///
    /// Each card covers this many bytes of heap. Smaller cards
    /// give more precise tracking but use more memory.
    ///
    /// Default: 512 bytes
    pub card_size: usize,

    // =========================================================================
    // Concurrent Collection (Phase 2)
    // =========================================================================
    /// Enable concurrent marking.
    ///
    /// When enabled, marking runs concurrently with the mutator,
    /// reducing pause times at the cost of throughput.
    ///
    /// Default: false (not yet implemented)
    pub concurrent_marking: bool,

    /// Number of concurrent marker threads.
    ///
    /// Default: Number of CPUs minus 1, minimum 1
    pub marker_threads: usize,

    // =========================================================================
    // Debugging
    // =========================================================================
    /// Enable GC tracing output.
    ///
    /// Prints detailed information about each collection.
    ///
    /// Default: false
    pub trace: bool,

    /// Verify heap integrity after each collection.
    ///
    /// Expensive but useful for debugging GC bugs.
    ///
    /// Default: false (enabled in debug builds)
    pub verify_heap: bool,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            // Nursery
            nursery_size: 4 * 1024 * 1024, // 4MB
            promotion_age: 2,

            // Tenured
            initial_old_size: 16 * 1024 * 1024, // 16MB
            max_old_size: 0,                    // Unlimited
            old_growth_factor: 1.5,
            compaction_threshold: 0.3,

            // Large objects
            large_object_threshold: 8 * 1024, // 8KB

            // Triggers
            minor_gc_trigger: 4 * 1024 * 1024, // Same as nursery
            major_gc_threshold: 0.75,

            // Blocks
            block_size: 16 * 1024, // 16KB
            card_size: 512,        // 512 bytes per card

            // Concurrent (Phase 2)
            concurrent_marking: false,
            marker_threads: num_cpus().saturating_sub(1).max(1),

            // Debugging
            trace: false,
            verify_heap: cfg!(debug_assertions),
        }
    }
}

impl GcConfig {
    /// Create a configuration optimized for low memory usage.
    pub fn low_memory() -> Self {
        Self {
            nursery_size: 1024 * 1024,         // 1MB
            initial_old_size: 4 * 1024 * 1024, // 4MB
            large_object_threshold: 4 * 1024,  // 4KB
            ..Default::default()
        }
    }

    /// Create a configuration optimized for high throughput.
    pub fn high_throughput() -> Self {
        Self {
            nursery_size: 16 * 1024 * 1024,     // 16MB
            initial_old_size: 64 * 1024 * 1024, // 64MB
            promotion_age: 3,
            major_gc_threshold: 0.85,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for low latency.
    pub fn low_latency() -> Self {
        Self {
            nursery_size: 2 * 1024 * 1024, // 2MB (smaller = faster minor GC)
            concurrent_marking: true,
            ..Default::default()
        }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.nursery_size < 64 * 1024 {
            return Err(ConfigError::NurseryTooSmall);
        }
        if self.block_size < 4096 {
            return Err(ConfigError::BlockTooSmall);
        }
        if self.card_size < 64 || !self.card_size.is_power_of_two() {
            return Err(ConfigError::InvalidCardSize);
        }
        if self.promotion_age == 0 {
            return Err(ConfigError::InvalidPromotionAge);
        }
        Ok(())
    }
}

/// Configuration validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    /// Nursery size is too small (minimum 64KB).
    NurseryTooSmall,
    /// Block size is too small (minimum 4KB).
    BlockTooSmall,
    /// Card size must be a power of two, minimum 64.
    InvalidCardSize,
    /// Promotion age must be at least 1.
    InvalidPromotionAge,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::NurseryTooSmall => write!(f, "nursery size must be at least 64KB"),
            ConfigError::BlockTooSmall => write!(f, "block size must be at least 4KB"),
            ConfigError::InvalidCardSize => {
                write!(f, "card size must be a power of two, minimum 64")
            }
            ConfigError::InvalidPromotionAge => write!(f, "promotion age must be at least 1"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Get the number of available CPUs.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        assert!(GcConfig::default().validate().is_ok());
    }

    #[test]
    fn test_preset_configs_are_valid() {
        assert!(GcConfig::low_memory().validate().is_ok());
        assert!(GcConfig::high_throughput().validate().is_ok());
        assert!(GcConfig::low_latency().validate().is_ok());
    }

    #[test]
    fn test_invalid_nursery_size() {
        let config = GcConfig {
            nursery_size: 1024,
            ..Default::default()
        };
        assert_eq!(config.validate(), Err(ConfigError::NurseryTooSmall));
    }

    #[test]
    fn test_invalid_card_size() {
        let config = GcConfig {
            card_size: 100, // Not power of two
            ..Default::default()
        };
        assert_eq!(config.validate(), Err(ConfigError::InvalidCardSize));
    }
}
