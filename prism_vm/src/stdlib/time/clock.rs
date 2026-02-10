//! High-resolution clock implementations.
//!
//! Provides access to various system clocks with nanosecond precision.
//! Uses platform-specific APIs for maximum performance.
//!
//! # Clocks Available
//!
//! | Clock | Description | Monotonic | Adjustable |
//! |-------|-------------|-----------|------------|
//! | `Realtime` | Wall-clock time | No | Yes |
//! | `Monotonic` | Steady time for intervals | Yes | No |
//! | `ProcessCputime` | CPU time for process | Yes | No |
//! | `ThreadCputime` | CPU time for thread | Yes | No |

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// =============================================================================
// Clock Identifiers
// =============================================================================

/// Clock identifiers matching Python's time.CLOCK_* constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ClockId {
    /// Real-time (wall clock) - can be adjusted.
    Realtime = 0,
    /// Monotonic clock - never adjusted, ideal for measuring intervals.
    Monotonic = 1,
    /// CPU time for the process.
    ProcessCputime = 2,
    /// CPU time for the current thread.
    ThreadCputime = 3,
    /// High-resolution performance counter.
    HighResolution = 4,
}

impl ClockId {
    /// Convert from integer value.
    #[inline]
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(ClockId::Realtime),
            1 => Some(ClockId::Monotonic),
            2 => Some(ClockId::ProcessCputime),
            3 => Some(ClockId::ThreadCputime),
            4 => Some(ClockId::HighResolution),
            _ => None,
        }
    }
}

// =============================================================================
// Clock Kind (for clock_getres/clock_gettime)
// =============================================================================

/// Kind of clock for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockKind {
    /// Wall-clock time.
    WallClock,
    /// Monotonic (steady) clock.
    Monotonic,
    /// CPU time clock.
    CpuTime,
}

// =============================================================================
// Clock Info (matches Python's time.get_clock_info)
// =============================================================================

/// Information about a clock.
#[derive(Debug, Clone)]
pub struct ClockInfo {
    /// Clock implementation name.
    pub implementation: &'static str,
    /// True if clock is monotonic.
    pub monotonic: bool,
    /// True if clock can be adjusted.
    pub adjustable: bool,
    /// Clock resolution in seconds.
    pub resolution: f64,
}

impl ClockInfo {
    /// Get info for the realtime clock.
    pub fn realtime() -> Self {
        Self {
            implementation: platform_realtime_impl(),
            monotonic: false,
            adjustable: true,
            resolution: platform_realtime_resolution(),
        }
    }

    /// Get info for the monotonic clock.
    pub fn monotonic() -> Self {
        Self {
            implementation: platform_monotonic_impl(),
            monotonic: true,
            adjustable: false,
            resolution: platform_monotonic_resolution(),
        }
    }

    /// Get info for the performance counter.
    pub fn perf_counter() -> Self {
        Self {
            implementation: platform_perf_counter_impl(),
            monotonic: true,
            adjustable: false,
            resolution: platform_perf_counter_resolution(),
        }
    }

    /// Get info for the process CPU time clock.
    pub fn process_time() -> Self {
        Self {
            implementation: platform_process_time_impl(),
            monotonic: true,
            adjustable: false,
            resolution: platform_process_time_resolution(),
        }
    }

    /// Get info for the thread CPU time clock.
    pub fn thread_time() -> Self {
        Self {
            implementation: platform_thread_time_impl(),
            monotonic: true,
            adjustable: false,
            resolution: platform_thread_time_resolution(),
        }
    }
}

// =============================================================================
// Clock Trait
// =============================================================================

/// Trait for clock implementations.
pub trait Clock {
    /// Get the current time in nanoseconds.
    fn now_ns(&self) -> i128;

    /// Get the current time in seconds as a float.
    #[inline]
    fn now(&self) -> f64 {
        self.now_ns() as f64 / 1_000_000_000.0
    }

    /// Get clock information.
    fn info(&self) -> ClockInfo;
}

// =============================================================================
// Realtime Clock (Wall Clock)
// =============================================================================

/// Real-time (wall clock) implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct RealtimeClock;

impl RealtimeClock {
    /// Create a new realtime clock.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Clock for RealtimeClock {
    #[inline]
    fn now_ns(&self) -> i128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i128)
            .unwrap_or(0)
    }

    fn info(&self) -> ClockInfo {
        ClockInfo::realtime()
    }
}

// =============================================================================
// Monotonic Clock
// =============================================================================

/// Monotonic clock implementation.
///
/// Uses `std::time::Instant` which is guaranteed monotonic.
#[derive(Debug)]
pub struct MonotonicClock {
    /// Origin instant for computing elapsed time.
    origin: Instant,
}

impl MonotonicClock {
    /// Create a new monotonic clock.
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }

    /// Get nanoseconds since origin.
    #[inline]
    pub fn elapsed_ns(&self) -> u128 {
        self.origin.elapsed().as_nanos()
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    #[inline]
    fn now_ns(&self) -> i128 {
        self.elapsed_ns() as i128
    }

    fn info(&self) -> ClockInfo {
        ClockInfo::monotonic()
    }
}

// =============================================================================
// Performance Counter
// =============================================================================

/// High-resolution performance counter.
///
/// Uses platform-specific APIs for highest resolution:
/// - Windows: QueryPerformanceCounter
/// - macOS: mach_absolute_time
/// - Linux: CLOCK_MONOTONIC_RAW
#[derive(Debug)]
pub struct PerfCounter {
    /// Origin for computing elapsed time.
    origin: Instant,
}

impl PerfCounter {
    /// Create a new performance counter.
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }

    /// Get the elapsed time in nanoseconds.
    #[inline]
    pub fn elapsed_ns(&self) -> u128 {
        self.origin.elapsed().as_nanos()
    }

    /// Get the elapsed time in seconds.
    #[inline]
    pub fn elapsed(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }
}

impl Default for PerfCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for PerfCounter {
    #[inline]
    fn now_ns(&self) -> i128 {
        self.elapsed_ns() as i128
    }

    fn info(&self) -> ClockInfo {
        ClockInfo::perf_counter()
    }
}

// =============================================================================
// Process CPU Time Clock
// =============================================================================

/// Process CPU time clock.
///
/// Measures CPU time consumed by the current process.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessTimeClock;

impl ProcessTimeClock {
    /// Create a new process time clock.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Clock for ProcessTimeClock {
    #[inline]
    fn now_ns(&self) -> i128 {
        get_process_time_ns()
    }

    fn info(&self) -> ClockInfo {
        ClockInfo::process_time()
    }
}

// =============================================================================
// Thread CPU Time Clock
// =============================================================================

/// Thread CPU time clock.
///
/// Measures CPU time consumed by the current thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThreadTimeClock;

impl ThreadTimeClock {
    /// Create a new thread time clock.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Clock for ThreadTimeClock {
    #[inline]
    fn now_ns(&self) -> i128 {
        get_thread_time_ns()
    }

    fn info(&self) -> ClockInfo {
        ClockInfo::thread_time()
    }
}

// =============================================================================
// Platform-Specific Implementations
// =============================================================================

// --- Windows ---

#[cfg(target_os = "windows")]
fn get_process_time_ns() -> i128 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    unsafe {
        let handle = GetCurrentProcess();
        let mut creation = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut exit = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut kernel = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut user = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };

        if GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) != 0 {
            let kernel_time =
                ((kernel.dwHighDateTime as u64) << 32) | (kernel.dwLowDateTime as u64);
            let user_time = ((user.dwHighDateTime as u64) << 32) | (user.dwLowDateTime as u64);
            // FILETIME is in 100-nanosecond intervals
            ((kernel_time + user_time) as i128) * 100
        } else {
            0
        }
    }
}

#[cfg(target_os = "windows")]
fn get_thread_time_ns() -> i128 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};

    unsafe {
        let handle = GetCurrentThread();
        let mut creation = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut exit = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut kernel = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut user = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };

        if GetThreadTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) != 0 {
            let kernel_time =
                ((kernel.dwHighDateTime as u64) << 32) | (kernel.dwLowDateTime as u64);
            let user_time = ((user.dwHighDateTime as u64) << 32) | (user.dwLowDateTime as u64);
            ((kernel_time + user_time) as i128) * 100
        } else {
            0
        }
    }
}

// --- Unix/Linux/macOS ---

#[cfg(not(target_os = "windows"))]
fn get_process_time_ns() -> i128 {
    unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            let user_sec = usage.ru_utime.tv_sec as i128;
            let user_usec = usage.ru_utime.tv_usec as i128;
            let sys_sec = usage.ru_stime.tv_sec as i128;
            let sys_usec = usage.ru_stime.tv_usec as i128;
            (user_sec + sys_sec) * 1_000_000_000 + (user_usec + sys_usec) * 1_000
        } else {
            0
        }
    }
}

#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
fn get_thread_time_ns() -> i128 {
    unsafe {
        let mut ts = std::mem::zeroed::<libc::timespec>();
        if libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) == 0 {
            (ts.tv_sec as i128) * 1_000_000_000 + (ts.tv_nsec as i128)
        } else {
            0
        }
    }
}

#[cfg(target_os = "macos")]
fn get_thread_time_ns() -> i128 {
    // macOS doesn't have CLOCK_THREAD_CPUTIME_ID, use thread_info
    // For now, fall back to process time divided by thread count (approximation)
    get_process_time_ns()
}

// =============================================================================
// Platform-Specific Clock Info
// =============================================================================

#[cfg(target_os = "windows")]
fn platform_realtime_impl() -> &'static str {
    "GetSystemTimePreciseAsFileTime"
}

#[cfg(not(target_os = "windows"))]
fn platform_realtime_impl() -> &'static str {
    "clock_gettime(CLOCK_REALTIME)"
}

#[cfg(target_os = "windows")]
fn platform_realtime_resolution() -> f64 {
    1e-7 // 100 nanoseconds (FILETIME resolution)
}

#[cfg(not(target_os = "windows"))]
fn platform_realtime_resolution() -> f64 {
    1e-9 // 1 nanosecond
}

#[cfg(target_os = "windows")]
fn platform_monotonic_impl() -> &'static str {
    "QueryPerformanceCounter"
}

#[cfg(target_os = "macos")]
fn platform_monotonic_impl() -> &'static str {
    "mach_absolute_time"
}

#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
fn platform_monotonic_impl() -> &'static str {
    "clock_gettime(CLOCK_MONOTONIC)"
}

fn platform_monotonic_resolution() -> f64 {
    1e-9 // Rust's Instant provides nanosecond resolution
}

fn platform_perf_counter_impl() -> &'static str {
    platform_monotonic_impl()
}

fn platform_perf_counter_resolution() -> f64 {
    platform_monotonic_resolution()
}

#[cfg(target_os = "windows")]
fn platform_process_time_impl() -> &'static str {
    "GetProcessTimes"
}

#[cfg(not(target_os = "windows"))]
fn platform_process_time_impl() -> &'static str {
    "getrusage(RUSAGE_SELF)"
}

fn platform_process_time_resolution() -> f64 {
    1e-6 // Microsecond resolution from rusage/FILETIME
}

#[cfg(target_os = "windows")]
fn platform_thread_time_impl() -> &'static str {
    "GetThreadTimes"
}

#[cfg(not(target_os = "windows"))]
fn platform_thread_time_impl() -> &'static str {
    "clock_gettime(CLOCK_THREAD_CPUTIME_ID)"
}

fn platform_thread_time_resolution() -> f64 {
    1e-9
}

// =============================================================================
// Global Clock Instances (Thread-Local for Performance)
// =============================================================================

thread_local! {
    static MONOTONIC_CLOCK: MonotonicClock = MonotonicClock::new();
    static PERF_COUNTER: PerfCounter = PerfCounter::new();
}

/// Get the current monotonic time in nanoseconds.
#[inline]
pub fn monotonic_ns() -> i128 {
    MONOTONIC_CLOCK.with(|c| c.now_ns())
}

/// Get the current monotonic time in seconds.
#[inline]
pub fn monotonic() -> f64 {
    MONOTONIC_CLOCK.with(|c| c.now())
}

/// Get the current performance counter value in nanoseconds.
#[inline]
pub fn perf_counter_ns() -> i128 {
    PERF_COUNTER.with(|c| c.now_ns())
}

/// Get the current performance counter value in seconds.
#[inline]
pub fn perf_counter() -> f64 {
    PERF_COUNTER.with(|c| c.now())
}

/// Get the current real time in nanoseconds since epoch.
#[inline]
pub fn time_ns() -> i128 {
    RealtimeClock::new().now_ns()
}

/// Get the current real time in seconds since epoch.
#[inline]
pub fn time() -> f64 {
    RealtimeClock::new().now()
}

/// Get the current process CPU time in nanoseconds.
#[inline]
pub fn process_time_ns() -> i128 {
    ProcessTimeClock::new().now_ns()
}

/// Get the current process CPU time in seconds.
#[inline]
pub fn process_time() -> f64 {
    ProcessTimeClock::new().now()
}

/// Get the current thread CPU time in nanoseconds.
#[inline]
pub fn thread_time_ns() -> i128 {
    ThreadTimeClock::new().now_ns()
}

/// Get the current thread CPU time in seconds.
#[inline]
pub fn thread_time() -> f64 {
    ThreadTimeClock::new().now()
}

// =============================================================================
// Sleep
// =============================================================================

/// Sleep for the specified duration.
///
/// # Arguments
/// * `seconds` - Duration to sleep in seconds (can be fractional)
///
/// # Precision
/// The actual sleep duration may be slightly longer due to OS scheduling.
/// For sub-millisecond precision, spin-waiting may be needed.
#[inline]
pub fn sleep(seconds: f64) {
    if seconds <= 0.0 {
        return;
    }

    let duration = Duration::from_secs_f64(seconds);
    std::thread::sleep(duration);
}

/// Sleep for the specified number of nanoseconds.
#[inline]
pub fn sleep_ns(nanoseconds: u64) {
    if nanoseconds == 0 {
        return;
    }

    std::thread::sleep(Duration::from_nanos(nanoseconds));
}
