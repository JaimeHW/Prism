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
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::string::StringObject;
use std::sync::Arc;
use std::sync::LazyLock;

static TIME_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("time.time"), time_builtin));
static TIME_NS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("time.time_ns"), time_ns_builtin));
static MONOTONIC_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("time.monotonic"), monotonic_builtin));
static MONOTONIC_NS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("time.monotonic_ns"), monotonic_ns_builtin)
});
static PERF_COUNTER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("time.perf_counter"), perf_counter_builtin)
});
static PERF_COUNTER_NS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("time.perf_counter_ns"), perf_counter_ns_builtin)
});
static PROCESS_TIME_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("time.process_time"), process_time_builtin)
});
static PROCESS_TIME_NS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("time.process_time_ns"), process_time_ns_builtin)
});
static THREAD_TIME_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("time.thread_time"), thread_time_builtin)
});
static THREAD_TIME_NS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("time.thread_time_ns"), thread_time_ns_builtin)
});
static SLEEP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("time.sleep"), sleep_builtin));
static GET_CLOCK_INFO_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("time.get_clock_info"), get_clock_info_builtin)
});

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
            Arc::from("get_clock_info"),
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
            "time" => Ok(builtin_value(&TIME_FUNCTION)),
            "time_ns" => Ok(builtin_value(&TIME_NS_FUNCTION)),
            "monotonic" => Ok(builtin_value(&MONOTONIC_FUNCTION)),
            "monotonic_ns" => Ok(builtin_value(&MONOTONIC_NS_FUNCTION)),
            "perf_counter" => Ok(builtin_value(&PERF_COUNTER_FUNCTION)),
            "perf_counter_ns" => Ok(builtin_value(&PERF_COUNTER_NS_FUNCTION)),
            "process_time" => Ok(builtin_value(&PROCESS_TIME_FUNCTION)),
            "process_time_ns" => Ok(builtin_value(&PROCESS_TIME_NS_FUNCTION)),
            "thread_time" => Ok(builtin_value(&THREAD_TIME_FUNCTION)),
            "thread_time_ns" => Ok(builtin_value(&THREAD_TIME_NS_FUNCTION)),
            "sleep" => Ok(builtin_value(&SLEEP_FUNCTION)),
            "get_clock_info" => Ok(builtin_value(&GET_CLOCK_INFO_FUNCTION)),
            "gmtime" | "localtime" | "mktime" | "strftime" | "strptime" | "asctime" | "ctime"
            | "struct_time" => Err(ModuleError::AttributeError(format!(
                "time.{} is not yet callable as an object",
                name
            ))),

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

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn expect_no_args(args: &[Value], fn_name: &str) -> Result<(), BuiltinError> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes 0 positional arguments but {} were given",
            args.len()
        )))
    }
}

fn int_value(value: i128, context: &str) -> Result<Value, BuiltinError> {
    let value = i64::try_from(value).map_err(|_| {
        BuiltinError::OverflowError(format!("{context} result exceeds supported range"))
    })?;
    Value::int(value).ok_or_else(|| {
        BuiltinError::OverflowError(format!("{context} result exceeds supported range"))
    })
}

fn value_to_string(value: Value, context: &str) -> Result<String, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be str")))?;
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be str")))?;
        return Ok(interned.as_str().to_string());
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be str")))?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return Err(BuiltinError::TypeError(format!("{context} must be str")));
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(string.as_str().to_string())
}

fn time_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "time")?;
    Ok(Value::float(time()))
}

fn time_ns_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "time_ns")?;
    int_value(time_ns(), "time_ns")
}

fn monotonic_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "monotonic")?;
    Ok(Value::float(monotonic()))
}

fn monotonic_ns_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "monotonic_ns")?;
    int_value(monotonic_ns(), "monotonic_ns")
}

fn perf_counter_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "perf_counter")?;
    Ok(Value::float(perf_counter()))
}

fn perf_counter_ns_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "perf_counter_ns")?;
    int_value(perf_counter_ns(), "perf_counter_ns")
}

fn process_time_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "process_time")?;
    Ok(Value::float(process_time()))
}

fn process_time_ns_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "process_time_ns")?;
    int_value(process_time_ns(), "process_time_ns")
}

fn thread_time_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "thread_time")?;
    Ok(Value::float(thread_time()))
}

fn thread_time_ns_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "thread_time_ns")?;
    int_value(thread_time_ns(), "thread_time_ns")
}

fn sleep_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "sleep() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let seconds = if let Some(value) = args[0].as_float() {
        value
    } else if let Some(value) = args[0].as_int() {
        value as f64
    } else {
        return Err(BuiltinError::TypeError(
            "sleep() argument must be int or float".to_string(),
        ));
    };

    if seconds.is_sign_negative() {
        return Err(BuiltinError::ValueError(
            "sleep length must be non-negative".to_string(),
        ));
    }

    sleep(seconds);
    Ok(Value::none())
}

fn get_clock_info_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "get_clock_info() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let name = value_to_string(args[0], "clock name")?;
    let info = get_clock_info(&name).map_err(|err| BuiltinError::ValueError(err.to_string()))?;
    let registry = shape_registry();
    let mut object = Box::new(ShapedObject::with_empty_shape(registry.empty_shape()));
    object.set_property(
        intern("implementation"),
        Value::string(intern(info.implementation)),
        registry,
    );
    object.set_property(intern("monotonic"), Value::bool(info.monotonic), registry);
    object.set_property(intern("adjustable"), Value::bool(info.adjustable), registry);
    object.set_property(
        intern("resolution"),
        Value::float(info.resolution),
        registry,
    );
    Ok(Value::object_ptr(Box::into_raw(object) as *const ()))
}
