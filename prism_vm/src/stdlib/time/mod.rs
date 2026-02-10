//! Python `time` module implementation.
//!
//! Provides high-performance time functions with nanosecond precision.
//! All functions are designed for zero-allocation hot paths where possible.
//!
//! # Performance Characteristics
//!
//! | Function | Time Complexity | Allocations |
//! |----------|-----------------|-------------|
//! | `time()` | O(1) | 0 |
//! | `time_ns()` | O(1) | 0 |
//! | `monotonic()` | O(1) | 0 |
//! | `perf_counter()` | O(1) | 0 |
//! | `sleep()` | O(1) | 0 |
//! | `gmtime()` | O(1) | 1 (struct_time) |
//! | `localtime()` | O(1) | 1 (struct_time) |
//! | `strftime()` | O(n) | 1 (string) |
//!
//! # Thread Safety
//!
//! All functions are thread-safe and reentrant.

mod clock;
mod format;
mod functions;
mod struct_time;

#[cfg(test)]
mod tests;

pub use clock::{Clock, ClockId, ClockInfo, ClockKind};
pub use format::{strftime, strptime};
pub use functions::*;
pub use struct_time::StructTime;

use super::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use std::sync::Arc;

// =============================================================================
// Time Module Constants
// =============================================================================

/// Timezone name for UTC.
pub const TIMEZONE_UTC: &str = "UTC";

/// Days per week.
pub const DAYS_PER_WEEK: i32 = 7;

/// Hours per day.
pub const HOURS_PER_DAY: i32 = 24;

/// Minutes per hour.
pub const MINUTES_PER_HOUR: i32 = 60;

/// Seconds per minute.
pub const SECONDS_PER_MINUTE: i32 = 60;

/// Seconds per hour.
pub const SECONDS_PER_HOUR: i32 = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;

/// Seconds per day.
pub const SECONDS_PER_DAY: i32 = SECONDS_PER_HOUR * HOURS_PER_DAY;

/// Days in each month (non-leap year).
pub const DAYS_IN_MONTH: [i32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Days in each month (leap year).
pub const DAYS_IN_MONTH_LEAP: [i32; 12] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Unix epoch day of week (Thursday = 4).
pub const EPOCH_WEEKDAY: i32 = 4;

/// Unix epoch year.
pub const EPOCH_YEAR: i32 = 1970;

// =============================================================================
// Time Module
// =============================================================================

/// The `time` module implementation.
#[derive(Debug, Clone)]
pub struct TimeModule {
    /// Cached attribute names for fast lookup.
    attrs: Vec<Arc<str>>,
}

impl TimeModule {
    /// Create a new time module instance.
    pub fn new() -> Self {
        let attrs = vec![
            Arc::from("time"),
            Arc::from("time_ns"),
            Arc::from("sleep"),
            Arc::from("monotonic"),
            Arc::from("monotonic_ns"),
            Arc::from("perf_counter"),
            Arc::from("perf_counter_ns"),
            Arc::from("process_time"),
            Arc::from("process_time_ns"),
            Arc::from("thread_time"),
            Arc::from("thread_time_ns"),
            Arc::from("gmtime"),
            Arc::from("localtime"),
            Arc::from("mktime"),
            Arc::from("strftime"),
            Arc::from("strptime"),
            Arc::from("asctime"),
            Arc::from("ctime"),
            Arc::from("timezone"),
            Arc::from("altzone"),
            Arc::from("daylight"),
            Arc::from("tzname"),
            Arc::from("struct_time"),
            // Clock constants (Python 3.3+)
            Arc::from("CLOCK_REALTIME"),
            Arc::from("CLOCK_MONOTONIC"),
            Arc::from("CLOCK_PROCESS_CPUTIME_ID"),
            Arc::from("CLOCK_THREAD_CPUTIME_ID"),
        ];

        Self { attrs }
    }
}

impl Default for TimeModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TimeModule {
    fn name(&self) -> &str {
        "time"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            // Time retrieval functions - return error until callable system is ready
            "time" | "time_ns" | "monotonic" | "monotonic_ns" | "perf_counter"
            | "perf_counter_ns" | "process_time" | "process_time_ns" | "thread_time"
            | "thread_time_ns" | "sleep" | "gmtime" | "localtime" | "mktime" | "strftime"
            | "strptime" | "asctime" | "ctime" | "get_clock_info" | "struct_time" => {
                // TODO: Return actual function objects when callable system is ready
                Err(ModuleError::AttributeError(format!(
                    "time.{} is not yet callable as an object",
                    name
                )))
            }

            // Timezone info - integer constants
            "timezone" => Ok(Value::int_unchecked(get_timezone_offset())),
            "altzone" => Ok(Value::int_unchecked(get_altzone_offset())),
            "daylight" => Ok(Value::int_unchecked(if has_daylight_saving() {
                1
            } else {
                0
            })),

            // tzname requires tuple/string - not yet supported
            "tzname" => Err(ModuleError::AttributeError(
                "time.tzname is not yet accessible (tuple support pending)".to_string(),
            )),

            // Clock constants
            "CLOCK_REALTIME" => Ok(Value::int_unchecked(ClockId::Realtime as i64)),
            "CLOCK_MONOTONIC" => Ok(Value::int_unchecked(ClockId::Monotonic as i64)),
            "CLOCK_PROCESS_CPUTIME_ID" => Ok(Value::int_unchecked(ClockId::ProcessCputime as i64)),
            "CLOCK_THREAD_CPUTIME_ID" => Ok(Value::int_unchecked(ClockId::ThreadCputime as i64)),

            _ => Err(ModuleError::AttributeError(format!(
                "module 'time' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

// =============================================================================
// Timezone Helpers (Platform-Specific)
// =============================================================================

/// Get the local timezone offset in seconds west of UTC.
#[cfg(target_os = "windows")]
fn get_timezone_offset() -> i64 {
    use std::mem::MaybeUninit;
    use windows_sys::Win32::System::Time::{GetTimeZoneInformation, TIME_ZONE_INFORMATION};

    unsafe {
        let mut tzi = MaybeUninit::<TIME_ZONE_INFORMATION>::uninit();
        GetTimeZoneInformation(tzi.as_mut_ptr());
        let tzi = tzi.assume_init();
        // Windows returns bias in minutes, we need seconds
        (tzi.Bias as i64) * 60
    }
}

#[cfg(not(target_os = "windows"))]
fn get_timezone_offset() -> i64 {
    // Unix: use tm_gmtoff from localtime
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Get local time and compute offset
    let local = unsafe {
        let t = now as libc::time_t;
        let mut tm = std::mem::zeroed::<libc::tm>();
        libc::localtime_r(&t, &mut tm);
        tm
    };

    // tm_gmtoff is seconds east of UTC, we return west
    -(local.tm_gmtoff as i64)
}

/// Get alternate timezone offset (DST offset).
fn get_altzone_offset() -> i64 {
    // Simplified: assume 1 hour DST offset
    get_timezone_offset() - 3600
}

/// Check if the system observes daylight saving time.
fn has_daylight_saving() -> bool {
    // Simplified check
    #[cfg(target_os = "windows")]
    {
        use std::mem::MaybeUninit;
        use windows_sys::Win32::System::Time::{GetTimeZoneInformation, TIME_ZONE_INFORMATION};

        unsafe {
            let mut tzi = MaybeUninit::<TIME_ZONE_INFORMATION>::uninit();
            let result = GetTimeZoneInformation(tzi.as_mut_ptr());
            result != 0 // Non-zero means DST is active or defined
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        unsafe { libc::daylight != 0 }
    }
}

/// Get timezone names (standard and DST).
fn get_tzname() -> (String, String) {
    #[cfg(target_os = "windows")]
    {
        use std::mem::MaybeUninit;
        use windows_sys::Win32::System::Time::{GetTimeZoneInformation, TIME_ZONE_INFORMATION};

        unsafe {
            let mut tzi = MaybeUninit::<TIME_ZONE_INFORMATION>::uninit();
            GetTimeZoneInformation(tzi.as_mut_ptr());
            let tzi = tzi.assume_init();

            let std_name = String::from_utf16_lossy(&tzi.StandardName)
                .trim_end_matches('\0')
                .to_string();
            let dst_name = String::from_utf16_lossy(&tzi.DaylightName)
                .trim_end_matches('\0')
                .to_string();

            (std_name, dst_name)
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        unsafe {
            let tzname_ptr = libc::tzname;
            let std_name = if !(*tzname_ptr).is_null() {
                std::ffi::CStr::from_ptr(*tzname_ptr)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "UTC".to_string()
            };
            let dst_name = if !(*tzname_ptr.add(1)).is_null() {
                std::ffi::CStr::from_ptr(*tzname_ptr.add(1))
                    .to_string_lossy()
                    .into_owned()
            } else {
                "UTC".to_string()
            };
            (std_name, dst_name)
        }
    }
}
