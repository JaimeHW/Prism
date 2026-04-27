//! High-level time module functions.
//!
//! These functions provide the Python-compatible API for the time module.

use super::ModuleError;
use super::clock::{self, ClockInfo};
use super::format;
use super::struct_time::StructTime;

// =============================================================================
// Time Retrieval Functions
// =============================================================================

/// Return the time in seconds since the epoch as a floating point number.
#[inline]
pub fn time() -> f64 {
    clock::time()
}

/// Return the time in nanoseconds since the epoch as an integer.
#[inline]
pub fn time_ns() -> i128 {
    clock::time_ns()
}

/// Return the value of a monotonic clock in seconds.
#[inline]
pub fn monotonic() -> f64 {
    clock::monotonic()
}

/// Return the value of a monotonic clock in nanoseconds.
#[inline]
pub fn monotonic_ns() -> i128 {
    clock::monotonic_ns()
}

/// Return the value of a performance counter in seconds.
#[inline]
pub fn perf_counter() -> f64 {
    clock::perf_counter()
}

/// Return the value of a performance counter in nanoseconds.
#[inline]
pub fn perf_counter_ns() -> i128 {
    clock::perf_counter_ns()
}

/// Return the value of the sum of the system and user CPU time in seconds.
#[inline]
pub fn process_time() -> f64 {
    clock::process_time()
}

/// Return the value of the sum of the system and user CPU time in nanoseconds.
#[inline]
pub fn process_time_ns() -> i128 {
    clock::process_time_ns()
}

/// Return the value of the current thread's CPU time in seconds.
#[inline]
pub fn thread_time() -> f64 {
    clock::thread_time()
}

/// Return the value of the current thread's CPU time in nanoseconds.
#[inline]
pub fn thread_time_ns() -> i128 {
    clock::thread_time_ns()
}

// =============================================================================
// Sleep
// =============================================================================

/// Suspend execution for the given number of seconds.
#[inline]
pub fn sleep(secs: f64) {
    clock::sleep(secs)
}

// =============================================================================
// Time Conversion Functions
// =============================================================================

/// Convert a time in seconds since the epoch to a struct_time in UTC.
pub fn gmtime(secs: Option<f64>) -> StructTime {
    let timestamp = secs.unwrap_or_else(time) as i64;
    StructTime::from_timestamp_utc(timestamp)
}

/// Convert a time in seconds since the epoch to a struct_time in local time.
pub fn localtime(secs: Option<f64>) -> StructTime {
    let timestamp = secs.unwrap_or_else(time) as i64;
    StructTime::from_timestamp_local(timestamp)
}

/// Convert a struct_time in local time to seconds since the epoch.
pub fn mktime(t: &StructTime) -> f64 {
    t.to_timestamp() as f64
}

/// Convert a struct_time to a string of the form "Sun Jun 20 23:21:05 1993".
pub fn asctime(t: Option<&StructTime>) -> Result<String, ModuleError> {
    let t = match t {
        Some(t) => *t,
        None => localtime(None),
    };
    format::strftime("%c", &t).map_err(|e| ModuleError::ValueError(e.to_string()))
}

/// Convert time in seconds to a string of the form "Sun Jun 20 23:21:05 1993".
pub fn ctime(secs: Option<f64>) -> Result<String, ModuleError> {
    let t = localtime(secs);
    asctime(Some(&t))
}

/// Format a struct_time according to the format string.
pub fn strftime_fn(format: &str, t: Option<&StructTime>) -> Result<String, ModuleError> {
    let t = match t {
        Some(t) => *t,
        None => localtime(None),
    };
    format::strftime(format, &t).map_err(|e| ModuleError::ValueError(e.to_string()))
}

/// Parse a string representing a time according to a format.
pub fn strptime_fn(string: &str, format: &str) -> Result<StructTime, ModuleError> {
    format::strptime(string, format).map_err(|e| ModuleError::ValueError(e.to_string()))
}

// =============================================================================
// Clock Info Functions
// =============================================================================

/// Get information about a clock.
pub fn get_clock_info(name: &str) -> Result<ClockInfo, ModuleError> {
    match name {
        "time" => Ok(ClockInfo::realtime()),
        "monotonic" => Ok(ClockInfo::monotonic()),
        "perf_counter" => Ok(ClockInfo::perf_counter()),
        "process_time" => Ok(ClockInfo::process_time()),
        "thread_time" => Ok(ClockInfo::thread_time()),
        _ => Err(ModuleError::ValueError(format!("unknown clock: {}", name))),
    }
}
