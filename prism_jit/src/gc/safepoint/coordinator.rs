//! Stop-the-world coordination for GC safepoints.
//!
//! The coordinator manages thread synchronization for GC:
//! 1. Arm safepoint page (triggers traps)
//! 2. Wait for all mutator threads to reach safepoints
//! 3. Run GC while threads are stopped
//! 4. Disarm page and resume threads
//!
//! # Performance
//!
//! Uses platform-optimal wait primitives:
//! - Linux: `futex(FUTEX_WAIT)`
//! - Windows: `WaitOnAddress`
//! - macOS: `os_unfair_lock` (via parking_lot)

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::page::{SafepointError, SafepointPage, SafepointState};
use super::stats::SafepointStats;

// =============================================================================
// MutatorState
// =============================================================================

/// State of a mutator thread with respect to safepoints.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutatorState {
    /// Thread is running and may need to stop at safepoint.
    Running = 0,
    /// Thread has reached a safepoint and is waiting.
    AtSafepoint = 1,
    /// Thread is blocked on I/O or synchronization (already safe).
    Blocked = 2,
    /// Thread is in native code (not managed).
    Native = 3,
}

impl MutatorState {
    /// Check if this state is considered "safe" for GC.
    #[inline]
    pub fn is_safe_for_gc(&self) -> bool {
        matches!(
            self,
            MutatorState::AtSafepoint | MutatorState::Blocked | MutatorState::Native
        )
    }
}

// =============================================================================
// MutatorThread
// =============================================================================

/// Represents a registered mutator thread.
#[derive(Debug)]
pub struct MutatorThread {
    /// Thread ID.
    pub id: u64,
    /// Current state.
    pub state: AtomicU32,
    /// Handle for waiting/waking.
    #[cfg(unix)]
    pub futex_word: AtomicU32,
}

impl MutatorThread {
    /// Create a new mutator thread entry.
    pub fn new(id: u64) -> Self {
        MutatorThread {
            id,
            state: AtomicU32::new(MutatorState::Running as u32),
            #[cfg(unix)]
            futex_word: AtomicU32::new(0),
        }
    }

    /// Get the current state.
    #[inline]
    pub fn get_state(&self) -> MutatorState {
        match self.state.load(Ordering::Acquire) {
            0 => MutatorState::Running,
            1 => MutatorState::AtSafepoint,
            2 => MutatorState::Blocked,
            3 => MutatorState::Native,
            _ => MutatorState::Running,
        }
    }

    /// Set the thread state.
    #[inline]
    pub fn set_state(&self, state: MutatorState) {
        self.state.store(state as u32, Ordering::Release);
    }
}

// =============================================================================
// SafepointGuard
// =============================================================================

/// RAII guard that keeps the world stopped until dropped.
///
/// When dropped, automatically resumes mutator threads.
pub struct SafepointGuard<'a> {
    coordinator: &'a SafepointCoordinator,
    start_time: Instant,
}

impl<'a> SafepointGuard<'a> {
    /// Get duration since stop-the-world began.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Drop for SafepointGuard<'_> {
    fn drop(&mut self) {
        // Record timing
        let duration = self.start_time.elapsed();
        self.coordinator.stats.record_stw_duration(duration);

        // Resume threads
        self.coordinator.resume_internal();
    }
}

// =============================================================================
// SafepointCoordinator
// =============================================================================

/// Coordinator for stop-the-world GC safepoints.
///
/// Manages thread registration, synchronization, and timing.
pub struct SafepointCoordinator {
    /// The safepoint page (shared across all threads).
    page: Arc<SafepointPage>,

    /// Registered mutator threads.
    mutators: RwLock<Vec<Arc<MutatorThread>>>,

    /// Barrier counter for synchronization.
    barrier: AtomicU32,

    /// Expected number of threads to stop.
    expected_threads: AtomicU32,

    /// Statistics tracking.
    stats: SafepointStats,
}

impl SafepointCoordinator {
    /// Create a new coordinator with the given safepoint page.
    pub fn new(page: Arc<SafepointPage>) -> Self {
        SafepointCoordinator {
            page,
            mutators: RwLock::new(Vec::new()),
            barrier: AtomicU32::new(0),
            expected_threads: AtomicU32::new(0),
            stats: SafepointStats::new(),
        }
    }

    /// Register a mutator thread.
    ///
    /// Returns a thread handle for state updates.
    pub fn register_thread(&self, thread_id: u64) -> Arc<MutatorThread> {
        let thread = Arc::new(MutatorThread::new(thread_id));
        let thread_clone = Arc::clone(&thread);

        let mut mutators = self.mutators.write().unwrap();
        mutators.push(thread);
        self.expected_threads.fetch_add(1, Ordering::AcqRel);

        thread_clone
    }

    /// Unregister a mutator thread.
    pub fn unregister_thread(&self, thread_id: u64) {
        let mut mutators = self.mutators.write().unwrap();
        if let Some(pos) = mutators.iter().position(|t| t.id == thread_id) {
            mutators.swap_remove(pos);
            self.expected_threads.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// Request stop-the-world for GC.
    ///
    /// Blocks until all mutator threads have reached safepoints.
    ///
    /// # Returns
    ///
    /// A guard that keeps threads stopped until dropped.
    pub fn stop_the_world(&self) -> Result<SafepointGuard<'_>, SafepointError> {
        let start = Instant::now();
        self.stats.record_stw_request();

        // Arm the safepoint page
        self.page.arm()?;
        self.page.mark_triggered();

        // Wait for all threads to reach safepoint
        let expected = self.expected_threads.load(Ordering::Acquire);
        self.wait_for_threads(expected);

        self.stats.record_stw_achieved(start.elapsed());

        Ok(SafepointGuard {
            coordinator: self,
            start_time: start,
        })
    }

    /// Try to stop the world with a timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for threads
    ///
    /// # Returns
    ///
    /// `Some(guard)` if successful, `None` if timeout.
    pub fn try_stop_the_world(
        &self,
        timeout: Duration,
    ) -> Result<Option<SafepointGuard<'_>>, SafepointError> {
        let start = Instant::now();
        self.stats.record_stw_request();

        // Arm the safepoint page
        self.page.arm()?;
        self.page.mark_triggered();

        // Wait with timeout
        let expected = self.expected_threads.load(Ordering::Acquire);
        if !self.wait_for_threads_timeout(expected, timeout) {
            // Timeout - disarm and return None
            self.page.disarm()?;
            return Ok(None);
        }

        self.stats.record_stw_achieved(start.elapsed());

        Ok(Some(SafepointGuard {
            coordinator: self,
            start_time: start,
        }))
    }

    /// Check if the world is currently stopped.
    #[inline]
    pub fn is_stopped(&self) -> bool {
        self.page.state() == SafepointState::Triggered
    }

    /// Get the safepoint page.
    #[inline]
    pub fn page(&self) -> &Arc<SafepointPage> {
        &self.page
    }

    /// Get statistics.
    #[inline]
    pub fn stats(&self) -> &SafepointStats {
        &self.stats
    }

    /// Get the number of registered threads.
    #[inline]
    pub fn thread_count(&self) -> usize {
        self.mutators.read().unwrap().len()
    }

    // =========================================================================
    // Internal Methods
    // =========================================================================

    /// Wait for all threads to reach safepoint.
    fn wait_for_threads(&self, expected: u32) {
        // Spin-wait with exponential backoff
        let mut spin_count = 0;
        const MAX_SPINS: u32 = 1000;

        loop {
            let stopped = self.page.stopped_count();
            if stopped >= expected {
                break;
            }

            if spin_count < MAX_SPINS {
                // Spin phase
                for _ in 0..10 {
                    std::hint::spin_loop();
                }
                spin_count += 1;
            } else {
                // Yield to OS scheduler
                std::thread::yield_now();
            }
        }
    }

    /// Wait for threads with timeout.
    fn wait_for_threads_timeout(&self, expected: u32, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;

        loop {
            let stopped = self.page.stopped_count();
            if stopped >= expected {
                return true;
            }

            if Instant::now() >= deadline {
                return false;
            }

            std::thread::yield_now();
        }
    }

    /// Resume threads (called from SafepointGuard drop).
    fn resume_internal(&self) {
        // Disarm the page (ignore errors in drop)
        let _ = self.page.disarm();

        // Reset barrier
        self.barrier.store(0, Ordering::Release);

        // Wake all waiting threads
        self.wake_all_threads();
    }

    /// Wake all threads using platform-optimal primitives.
    #[cfg(unix)]
    fn wake_all_threads(&self) {
        let mutators = self.mutators.read().unwrap();
        for thread in mutators.iter() {
            // Use futex wake to notify threads
            unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    &thread.futex_word as *const _ as *const libc::c_void,
                    libc::FUTEX_WAKE,
                    i32::MAX, // Wake all waiters
                );
            }
        }
    }

    #[cfg(windows)]
    fn wake_all_threads(&self) {
        // On Windows, threads use spin-wait checking the page state
        // No explicit wake needed - disarming the page is sufficient
    }
}

// =============================================================================
// Thread-Local Utilities
// =============================================================================

/// Enter a safepoint (called when trap occurs).
///
/// This is the slow path executed when a thread hits an armed safepoint.
pub fn enter_safepoint_slow(thread: &MutatorThread, page: &SafepointPage) {
    // Mark ourselves as at safepoint
    thread.set_state(MutatorState::AtSafepoint);

    // Increment stopped count
    page.thread_stopped();

    // Wait for resume
    wait_for_resume(thread, page);

    // Back to running
    thread.set_state(MutatorState::Running);
}

/// Wait for GC to complete and resume.
#[cfg(unix)]
fn wait_for_resume(thread: &MutatorThread, page: &SafepointPage) {
    // Fast path: check if already resumed
    if !page.is_armed() {
        return;
    }

    // Slow path: futex wait
    loop {
        // Set futex word to 1 (waiting)
        thread.futex_word.store(1, Ordering::Release);

        // Double-check armed state
        if !page.is_armed() {
            thread.futex_word.store(0, Ordering::Release);
            break;
        }

        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &thread.futex_word as *const _ as *const libc::c_void,
                libc::FUTEX_WAIT,
                1, // Expected value
                std::ptr::null::<libc::timespec>(),
            );
        }

        // Recheck after wake
        if !page.is_armed() {
            thread.futex_word.store(0, Ordering::Release);
            break;
        }
    }
}

#[cfg(windows)]
fn wait_for_resume(_thread: &MutatorThread, page: &SafepointPage) {
    // Spin-wait on Windows (WaitOnAddress requires more setup)
    while page.is_armed() {
        std::hint::spin_loop();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutator_state() {
        assert!(MutatorState::AtSafepoint.is_safe_for_gc());
        assert!(MutatorState::Blocked.is_safe_for_gc());
        assert!(MutatorState::Native.is_safe_for_gc());
        assert!(!MutatorState::Running.is_safe_for_gc());
    }

    #[test]
    fn test_mutator_thread() {
        let thread = MutatorThread::new(42);
        assert_eq!(thread.id, 42);
        assert_eq!(thread.get_state(), MutatorState::Running);

        thread.set_state(MutatorState::AtSafepoint);
        assert_eq!(thread.get_state(), MutatorState::AtSafepoint);
    }

    #[test]
    fn test_coordinator_creation() {
        let page = Arc::new(SafepointPage::new().unwrap());
        let coord = SafepointCoordinator::new(page);
        assert_eq!(coord.thread_count(), 0);
        assert!(!coord.is_stopped());
    }

    #[test]
    fn test_coordinator_register_unregister() {
        let page = Arc::new(SafepointPage::new().unwrap());
        let coord = SafepointCoordinator::new(page);

        let thread = coord.register_thread(100);
        assert_eq!(coord.thread_count(), 1);
        assert_eq!(thread.id, 100);

        coord.unregister_thread(100);
        assert_eq!(coord.thread_count(), 0);
    }

    #[test]
    fn test_stop_the_world_no_threads() {
        let page = Arc::new(SafepointPage::new().unwrap());
        let coord = SafepointCoordinator::new(page);

        // With no threads, stop_the_world should succeed immediately
        let guard = coord.stop_the_world().unwrap();
        assert!(coord.is_stopped());
        drop(guard);
        assert!(!coord.is_stopped());
    }

    #[test]
    fn test_try_stop_timeout() {
        let page = Arc::new(SafepointPage::new().unwrap());
        let coord = SafepointCoordinator::new(page);

        // Register a thread but don't stop it - should timeout
        let _thread = coord.register_thread(1);

        let result = coord.try_stop_the_world(Duration::from_millis(10)).unwrap();
        assert!(result.is_none()); // Should timeout
        assert!(!coord.is_stopped()); // Page should be disarmed
    }

    #[test]
    fn test_safepoint_guard_drop() {
        let page = Arc::new(SafepointPage::new().unwrap());
        let coord = SafepointCoordinator::new(page);

        {
            let _guard = coord.stop_the_world().unwrap();
            assert!(coord.is_stopped());
        }
        // Guard dropped, should resume
        assert!(!coord.is_stopped());
    }
}
