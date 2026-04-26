//! Deoptimization Statistics.
//!
//! Tracks deoptimization sites for adaptive optimization decisions.
//! Sites that deopt frequently can be patched to unconditionally bail out,
//! and functions with frequent deopts can be recompiled with different
//! optimization strategies.

use super::state::DeoptReason;
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// =============================================================================
// Deopt Site Key
// =============================================================================

/// Key identifying a unique deoptimization site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeoptSiteKey {
    /// Code object ID.
    pub code_id: u64,
    /// Bytecode offset of the deopt point.
    pub bc_offset: u32,
}

impl DeoptSiteKey {
    /// Create a new site key.
    #[inline]
    pub const fn new(code_id: u64, bc_offset: u32) -> Self {
        Self { code_id, bc_offset }
    }
}

// =============================================================================
// Deopt Site
// =============================================================================

/// Statistics for a single deoptimization site.
#[derive(Debug)]
pub struct DeoptSite {
    /// Site key.
    pub key: DeoptSiteKey,
    /// Total deopt count.
    count: AtomicU32,
    /// Count by reason.
    by_reason: [AtomicU32; 12],
    /// Last deopt timestamp (TSC or nanos).
    last_timestamp: AtomicU64,
    /// Whether guard has been patched.
    patched: AtomicU32, // Used as AtomicBool
}

impl DeoptSite {
    /// Create a new deopt site.
    #[inline]
    pub fn new(key: DeoptSiteKey) -> Self {
        const ZERO: AtomicU32 = AtomicU32::new(0);
        Self {
            key,
            count: AtomicU32::new(0),
            by_reason: [ZERO; 12],
            last_timestamp: AtomicU64::new(0),
            patched: AtomicU32::new(0),
        }
    }

    /// Record a deoptimization at this site.
    #[inline]
    pub fn record(&self, reason: DeoptReason, timestamp: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);

        let reason_idx = reason as usize;
        if reason_idx < self.by_reason.len() {
            self.by_reason[reason_idx].fetch_add(1, Ordering::Relaxed);
        }

        self.last_timestamp.store(timestamp, Ordering::Relaxed);
    }

    /// Get total deopt count.
    #[inline]
    pub fn total_count(&self) -> u32 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get count for a specific reason.
    #[inline]
    pub fn count_for_reason(&self, reason: DeoptReason) -> u32 {
        let reason_idx = reason as usize;
        if reason_idx < self.by_reason.len() {
            self.by_reason[reason_idx].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Get the dominant deopt reason.
    pub fn dominant_reason(&self) -> Option<DeoptReason> {
        let mut max_count = 0;
        let mut max_reason = None;

        for i in 0..self.by_reason.len() {
            let count = self.by_reason[i].load(Ordering::Relaxed);
            if count > max_count {
                max_count = count;
                max_reason = DeoptReason::from_u8(i as u8);
            }
        }

        max_reason
    }

    /// Check if guard is patched.
    #[inline]
    pub fn is_patched(&self) -> bool {
        self.patched.load(Ordering::Relaxed) != 0
    }

    /// Mark guard as patched.
    #[inline]
    pub fn set_patched(&self) {
        self.patched.store(1, Ordering::Relaxed);
    }

    /// Get last deopt timestamp.
    #[inline]
    pub fn last_timestamp(&self) -> u64 {
        self.last_timestamp.load(Ordering::Relaxed)
    }
}

// =============================================================================
// Deopt Stats
// =============================================================================

/// Global deoptimization statistics.
///
/// Thread-safe for concurrent recording from multiple execution threads.
#[derive(Debug, Default)]
pub struct DeoptStats {
    /// Per-site statistics.
    sites: RwLock<HashMap<DeoptSiteKey, DeoptSite>>,
    /// Total deopts across all sites.
    total_deopts: AtomicU64,
    /// Total deopts by reason.
    by_reason: [AtomicU64; 12],
}

impl DeoptStats {
    /// Create new statistics tracker.
    #[inline]
    pub fn new() -> Self {
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            sites: RwLock::new(HashMap::new()),
            total_deopts: AtomicU64::new(0),
            by_reason: [ZERO; 12],
        }
    }

    /// Record a deoptimization.
    pub fn record(&self, code_id: u64, bc_offset: u32, reason: DeoptReason) {
        let key = DeoptSiteKey::new(code_id, bc_offset);
        let timestamp = Self::current_timestamp();

        // Update global counters
        self.total_deopts.fetch_add(1, Ordering::Relaxed);
        let reason_idx = reason as usize;
        if reason_idx < self.by_reason.len() {
            self.by_reason[reason_idx].fetch_add(1, Ordering::Relaxed);
        }

        // Update per-site stats
        {
            let sites = self.sites.read().unwrap();
            if let Some(site) = sites.get(&key) {
                site.record(reason, timestamp);
                return;
            }
        }

        // Site not found, create it
        {
            let mut sites = self.sites.write().unwrap();
            // Double-check after acquiring write lock
            if let Some(site) = sites.get(&key) {
                site.record(reason, timestamp);
            } else {
                let site = DeoptSite::new(key);
                site.record(reason, timestamp);
                sites.insert(key, site);
            }
        }
    }

    /// Get site statistics.
    #[inline]
    pub fn get_site(&self, code_id: u64, bc_offset: u32) -> Option<SiteSnapshot> {
        let key = DeoptSiteKey::new(code_id, bc_offset);
        let sites = self.sites.read().unwrap();
        sites.get(&key).map(|site| SiteSnapshot {
            total_count: site.total_count(),
            dominant_reason: site.dominant_reason(),
            is_patched: site.is_patched(),
        })
    }

    /// Get total deopt count.
    #[inline]
    pub fn total_count(&self) -> u64 {
        self.total_deopts.load(Ordering::Relaxed)
    }

    /// Get count for a specific reason.
    #[inline]
    pub fn count_for_reason(&self, reason: DeoptReason) -> u64 {
        let reason_idx = reason as usize;
        if reason_idx < self.by_reason.len() {
            self.by_reason[reason_idx].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Get number of tracked sites.
    #[inline]
    pub fn site_count(&self) -> usize {
        self.sites.read().unwrap().len()
    }

    /// Mark a site as patched.
    pub fn mark_patched(&self, code_id: u64, bc_offset: u32) -> bool {
        let key = DeoptSiteKey::new(code_id, bc_offset);
        let sites = self.sites.read().unwrap();
        if let Some(site) = sites.get(&key) {
            site.set_patched();
            true
        } else {
            false
        }
    }

    /// Get sites that exceed deopt threshold.
    pub fn hot_sites(&self, threshold: u32) -> Vec<DeoptSiteKey> {
        let sites = self.sites.read().unwrap();
        sites
            .iter()
            .filter(|(_, site)| site.total_count() >= threshold && !site.is_patched())
            .map(|(key, _)| *key)
            .collect()
    }

    /// Get current timestamp.
    #[inline]
    fn current_timestamp() -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: rdtsc is always available on x86-64
            unsafe { std::arch::x86_64::_rdtsc() }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::time::Instant::now().elapsed().as_nanos() as u64
        }
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.sites.write().unwrap().clear();
        self.total_deopts.store(0, Ordering::Relaxed);
        for counter in &self.by_reason {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

/// Snapshot of site statistics (for returning from methods).
#[derive(Debug, Clone)]
pub struct SiteSnapshot {
    /// Total deopt count.
    pub total_count: u32,
    /// Dominant deopt reason.
    pub dominant_reason: Option<DeoptReason>,
    /// Whether guard is patched.
    pub is_patched: bool,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
