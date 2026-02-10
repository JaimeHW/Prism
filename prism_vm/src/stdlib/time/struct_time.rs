//! Python `time.struct_time` implementation.
//!
//! Provides the struct_time type which represents a time tuple.
//! This is a named tuple with 9 elements matching Python's time.struct_time.
//!
//! # Layout
//!
//! ```text
//! Index  Attribute  Values
//! 0      tm_year    e.g. 2024
//! 1      tm_mon     range [1, 12]
//! 2      tm_mday    range [1, 31]
//! 3      tm_hour    range [0, 23]
//! 4      tm_min     range [0, 59]
//! 5      tm_sec     range [0, 61] (60/61 for leap seconds)
//! 6      tm_wday    range [0, 6] (Monday = 0)
//! 7      tm_yday    range [1, 366]
//! 8      tm_isdst   0, 1 or -1
//! ```

use super::{DAYS_IN_MONTH, DAYS_IN_MONTH_LEAP, EPOCH_WEEKDAY, EPOCH_YEAR, SECONDS_PER_DAY};
use std::fmt;

// =============================================================================
// StructTime
// =============================================================================

/// Python's `time.struct_time` - a named tuple representing time.
///
/// Immutable, hashable, and comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct StructTime {
    /// Year (e.g., 2024).
    pub tm_year: i32,
    /// Month [1, 12].
    pub tm_mon: i32,
    /// Day of month [1, 31].
    pub tm_mday: i32,
    /// Hour [0, 23].
    pub tm_hour: i32,
    /// Minute [0, 59].
    pub tm_min: i32,
    /// Second [0, 61] (60/61 for leap seconds).
    pub tm_sec: i32,
    /// Day of week [0, 6] (Monday = 0).
    pub tm_wday: i32,
    /// Day of year [1, 366].
    pub tm_yday: i32,
    /// DST flag: 0=no, 1=yes, -1=unknown.
    pub tm_isdst: i32,
    /// Timezone abbreviation (optional, for strptime).
    pub tm_zone: Option<&'static str>,
    /// UTC offset in seconds (optional).
    pub tm_gmtoff: Option<i32>,
}

impl StructTime {
    /// Create a new struct_time with all fields.
    #[inline]
    pub const fn new(
        tm_year: i32,
        tm_mon: i32,
        tm_mday: i32,
        tm_hour: i32,
        tm_min: i32,
        tm_sec: i32,
        tm_wday: i32,
        tm_yday: i32,
        tm_isdst: i32,
    ) -> Self {
        Self {
            tm_year,
            tm_mon,
            tm_mday,
            tm_hour,
            tm_min,
            tm_sec,
            tm_wday,
            tm_yday,
            tm_isdst,
            tm_zone: None,
            tm_gmtoff: None,
        }
    }

    /// Create struct_time from Unix timestamp (seconds since epoch) in UTC.
    pub fn from_timestamp_utc(timestamp: i64) -> Self {
        let mut secs = timestamp;

        // Handle negative timestamps (before 1970)
        let is_negative = secs < 0;
        if is_negative {
            // Make positive for calculation, adjust later
            secs = -secs;
        }

        // Days since epoch
        let mut days = secs / (SECONDS_PER_DAY as i64);
        let mut remaining_secs = secs % (SECONDS_PER_DAY as i64);

        if is_negative {
            days = -days;
            if remaining_secs != 0 {
                days -= 1;
                remaining_secs = (SECONDS_PER_DAY as i64) - remaining_secs;
            }
        }

        // Time of day
        let tm_hour = (remaining_secs / 3600) as i32;
        remaining_secs %= 3600;
        let tm_min = (remaining_secs / 60) as i32;
        let tm_sec = (remaining_secs % 60) as i32;

        // Day of week (Thursday = 4 at epoch)
        let mut tm_wday = ((days + EPOCH_WEEKDAY as i64) % 7) as i32;
        if tm_wday < 0 {
            tm_wday += 7;
        }

        // Calculate year and day of year
        let (tm_year, tm_yday) = days_to_year_and_yday(days);

        // Calculate month and day of month
        let (tm_mon, tm_mday) = yday_to_month_and_day(tm_yday, is_leap_year(tm_year));

        Self {
            tm_year,
            tm_mon,
            tm_mday,
            tm_hour,
            tm_min,
            tm_sec,
            tm_wday,
            tm_yday,
            tm_isdst: 0, // UTC has no DST
            tm_zone: Some("UTC"),
            tm_gmtoff: Some(0),
        }
    }

    /// Create struct_time from Unix timestamp in local time.
    pub fn from_timestamp_local(timestamp: i64) -> Self {
        // Use libc's localtime_r for correct local time conversion
        #[cfg(not(target_os = "windows"))]
        {
            unsafe {
                let t = timestamp as libc::time_t;
                let mut tm = std::mem::zeroed::<libc::tm>();
                libc::localtime_r(&t, &mut tm);

                Self {
                    tm_year: tm.tm_year + 1900,
                    tm_mon: tm.tm_mon + 1,
                    tm_mday: tm.tm_mday,
                    tm_hour: tm.tm_hour,
                    tm_min: tm.tm_min,
                    tm_sec: tm.tm_sec,
                    tm_wday: tm.tm_wday,
                    tm_yday: tm.tm_yday + 1,
                    tm_isdst: tm.tm_isdst,
                    tm_zone: None, // Could extract from tm_zone
                    tm_gmtoff: Some(tm.tm_gmtoff as i32),
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            use std::mem::MaybeUninit;
            use windows_sys::Win32::System::Time::{GetTimeZoneInformation, TIME_ZONE_INFORMATION};

            // Get timezone offset from Windows
            let offset_seconds = unsafe {
                let mut tzi = MaybeUninit::<TIME_ZONE_INFORMATION>::uninit();
                GetTimeZoneInformation(tzi.as_mut_ptr());
                let tzi = tzi.assume_init();
                // Windows Bias is in minutes west of UTC, we need seconds
                -(tzi.Bias as i64 * 60)
            };

            // Apply offset to get local time, then convert
            let local_timestamp = timestamp + offset_seconds;
            let mut result = Self::from_timestamp_utc(local_timestamp);
            result.tm_gmtoff = Some(offset_seconds as i32);
            result.tm_isdst = -1; // Unknown on Windows
            result
        }
    }

    /// Convert struct_time to Unix timestamp.
    pub fn to_timestamp(&self) -> i64 {
        // Days from epoch to start of year
        let mut days = days_since_epoch(self.tm_year);

        // Add days for months
        let month_days = if is_leap_year(self.tm_year) {
            &DAYS_IN_MONTH_LEAP
        } else {
            &DAYS_IN_MONTH
        };

        for m in 0..(self.tm_mon - 1) as usize {
            if m < 12 {
                days += month_days[m] as i64;
            }
        }

        // Add day of month (1-indexed)
        days += (self.tm_mday - 1) as i64;

        // Convert to seconds and add time
        days * (SECONDS_PER_DAY as i64)
            + (self.tm_hour as i64) * 3600
            + (self.tm_min as i64) * 60
            + (self.tm_sec as i64)
    }

    /// Get field by index (for tuple-like access).
    #[inline]
    pub fn get(&self, index: usize) -> Option<i32> {
        match index {
            0 => Some(self.tm_year),
            1 => Some(self.tm_mon),
            2 => Some(self.tm_mday),
            3 => Some(self.tm_hour),
            4 => Some(self.tm_min),
            5 => Some(self.tm_sec),
            6 => Some(self.tm_wday),
            7 => Some(self.tm_yday),
            8 => Some(self.tm_isdst),
            _ => None,
        }
    }

    /// Get field by name.
    #[inline]
    pub fn get_by_name(&self, name: &str) -> Option<i32> {
        match name {
            "tm_year" => Some(self.tm_year),
            "tm_mon" => Some(self.tm_mon),
            "tm_mday" => Some(self.tm_mday),
            "tm_hour" => Some(self.tm_hour),
            "tm_min" => Some(self.tm_min),
            "tm_sec" => Some(self.tm_sec),
            "tm_wday" => Some(self.tm_wday),
            "tm_yday" => Some(self.tm_yday),
            "tm_isdst" => Some(self.tm_isdst),
            _ => None,
        }
    }

    /// Convert to a 9-element array of field values.
    #[inline]
    pub fn to_array(&self) -> [i32; 9] {
        [
            self.tm_year,
            self.tm_mon,
            self.tm_mday,
            self.tm_hour,
            self.tm_min,
            self.tm_sec,
            self.tm_wday,
            self.tm_yday,
            self.tm_isdst,
        ]
    }

    /// Create from a 9-tuple of integers.
    pub fn from_tuple(values: &[i64]) -> Option<Self> {
        if values.len() < 9 {
            return None;
        }

        // Validate ranges
        let tm_year = values[0] as i32;
        let tm_mon = values[1] as i32;
        let tm_mday = values[2] as i32;
        let tm_hour = values[3] as i32;
        let tm_min = values[4] as i32;
        let tm_sec = values[5] as i32;
        let tm_wday = values[6] as i32;
        let tm_yday = values[7] as i32;
        let tm_isdst = values[8] as i32;

        // Basic validation
        if !(1..=12).contains(&tm_mon)
            || !(1..=31).contains(&tm_mday)
            || !(0..=23).contains(&tm_hour)
            || !(0..=59).contains(&tm_min)
            || !(0..=61).contains(&tm_sec)
            || !(0..=6).contains(&tm_wday)
            || !(1..=366).contains(&tm_yday)
            || !(-1..=1).contains(&tm_isdst)
        {
            return None;
        }

        Some(Self::new(
            tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst,
        ))
    }

    /// Validate the struct_time values.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(1..=12).contains(&self.tm_mon) {
            return Err("month out of range [1, 12]");
        }

        let max_day = if is_leap_year(self.tm_year) {
            DAYS_IN_MONTH_LEAP[(self.tm_mon - 1) as usize]
        } else {
            DAYS_IN_MONTH[(self.tm_mon - 1) as usize]
        };

        if !(1..=max_day).contains(&self.tm_mday) {
            return Err("day out of range for month");
        }

        if !(0..=23).contains(&self.tm_hour) {
            return Err("hour out of range [0, 23]");
        }

        if !(0..=59).contains(&self.tm_min) {
            return Err("minute out of range [0, 59]");
        }

        if !(0..=61).contains(&self.tm_sec) {
            return Err("second out of range [0, 61]");
        }

        Ok(())
    }
}

impl Default for StructTime {
    fn default() -> Self {
        Self::new(EPOCH_YEAR, 1, 1, 0, 0, 0, EPOCH_WEEKDAY - 1, 1, 0)
    }
}

impl fmt::Display for StructTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "time.struct_time(tm_year={}, tm_mon={}, tm_mday={}, tm_hour={}, \
             tm_min={}, tm_sec={}, tm_wday={}, tm_yday={}, tm_isdst={})",
            self.tm_year,
            self.tm_mon,
            self.tm_mday,
            self.tm_hour,
            self.tm_min,
            self.tm_sec,
            self.tm_wday,
            self.tm_yday,
            self.tm_isdst
        )
    }
}

// =============================================================================
// Calendar Utilities
// =============================================================================

/// Check if a year is a leap year.
#[inline]
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Get the number of days in a year.
#[inline]
pub fn days_in_year(year: i32) -> i32 {
    if is_leap_year(year) { 366 } else { 365 }
}

/// Calculate days since Unix epoch for the start of a year.
fn days_since_epoch(year: i32) -> i64 {
    if year >= EPOCH_YEAR {
        let mut days = 0i64;
        for y in EPOCH_YEAR..year {
            days += days_in_year(y) as i64;
        }
        days
    } else {
        let mut days = 0i64;
        for y in year..EPOCH_YEAR {
            days -= days_in_year(y) as i64;
        }
        days
    }
}

/// Convert days since epoch to year and day of year.
fn days_to_year_and_yday(days: i64) -> (i32, i32) {
    let mut year = EPOCH_YEAR;
    let mut remaining = days;

    if remaining >= 0 {
        loop {
            let days_this_year = days_in_year(year) as i64;
            if remaining < days_this_year {
                break;
            }
            remaining -= days_this_year;
            year += 1;
        }
    } else {
        loop {
            year -= 1;
            remaining += days_in_year(year) as i64;
            if remaining >= 0 {
                break;
            }
        }
    }

    (year, (remaining + 1) as i32) // yday is 1-indexed
}

/// Convert day of year to month and day of month.
fn yday_to_month_and_day(yday: i32, leap: bool) -> (i32, i32) {
    let month_days = if leap {
        &DAYS_IN_MONTH_LEAP
    } else {
        &DAYS_IN_MONTH
    };

    let mut remaining = yday;
    for (i, &days) in month_days.iter().enumerate() {
        if remaining <= days {
            return ((i + 1) as i32, remaining);
        }
        remaining -= days;
    }

    // Should not reach here for valid yday
    (12, 31)
}

/// Calculate day of year from year, month, day.
pub fn day_of_year(year: i32, month: i32, day: i32) -> i32 {
    let month_days = if is_leap_year(year) {
        &DAYS_IN_MONTH_LEAP
    } else {
        &DAYS_IN_MONTH
    };

    let mut yday = day;
    for m in 0..(month - 1) as usize {
        if m < 12 {
            yday += month_days[m];
        }
    }
    yday
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod struct_time_tests {
    use super::*;

    #[test]
    fn test_leap_year() {
        assert!(!is_leap_year(1900)); // Divisible by 100 but not 400
        assert!(is_leap_year(2000)); // Divisible by 400
        assert!(is_leap_year(2004)); // Divisible by 4
        assert!(!is_leap_year(2001)); // Not divisible by 4
        assert!(is_leap_year(2024)); // Divisible by 4
    }

    #[test]
    fn test_days_in_year() {
        assert_eq!(days_in_year(2000), 366);
        assert_eq!(days_in_year(2001), 365);
        assert_eq!(days_in_year(2024), 366);
        assert_eq!(days_in_year(2023), 365);
    }

    #[test]
    fn test_epoch_conversion() {
        // Unix epoch: Jan 1, 1970, 00:00:00 UTC
        let st = StructTime::from_timestamp_utc(0);
        assert_eq!(st.tm_year, 1970);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 1);
        assert_eq!(st.tm_hour, 0);
        assert_eq!(st.tm_min, 0);
        assert_eq!(st.tm_sec, 0);
        assert_eq!(st.tm_wday, 4); // Thursday
        assert_eq!(st.tm_yday, 1);
    }

    #[test]
    fn test_known_timestamp() {
        // 2024-01-15 12:30:45 UTC
        // This is a Monday
        let timestamp = 1705321845i64;
        let st = StructTime::from_timestamp_utc(timestamp);

        assert_eq!(st.tm_year, 2024);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 15);
        assert_eq!(st.tm_hour, 12);
        assert_eq!(st.tm_min, 30);
        assert_eq!(st.tm_sec, 45);
        assert_eq!(st.tm_wday, 1); // Monday
        assert_eq!(st.tm_yday, 15);
    }

    #[test]
    fn test_roundtrip() {
        let original = 1705321845i64;
        let st = StructTime::from_timestamp_utc(original);
        let recovered = st.to_timestamp();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_negative_timestamp() {
        // Before 1970
        let timestamp = -86400i64; // Dec 31, 1969
        let st = StructTime::from_timestamp_utc(timestamp);
        assert_eq!(st.tm_year, 1969);
        assert_eq!(st.tm_mon, 12);
        assert_eq!(st.tm_mday, 31);
    }

    #[test]
    fn test_field_access() {
        let st = StructTime::new(2024, 6, 15, 10, 30, 45, 5, 167, 1);

        assert_eq!(st.get(0), Some(2024));
        assert_eq!(st.get(1), Some(6));
        assert_eq!(st.get(2), Some(15));
        assert_eq!(st.get(3), Some(10));
        assert_eq!(st.get(4), Some(30));
        assert_eq!(st.get(5), Some(45));
        assert_eq!(st.get(6), Some(5));
        assert_eq!(st.get(7), Some(167));
        assert_eq!(st.get(8), Some(1));
        assert_eq!(st.get(9), None);
    }

    #[test]
    fn test_named_access() {
        let st = StructTime::new(2024, 6, 15, 10, 30, 45, 5, 167, 1);

        assert_eq!(st.get_by_name("tm_year"), Some(2024));
        assert_eq!(st.get_by_name("tm_mon"), Some(6));
        assert_eq!(st.get_by_name("tm_mday"), Some(15));
        assert_eq!(st.get_by_name("tm_hour"), Some(10));
        assert_eq!(st.get_by_name("tm_min"), Some(30));
        assert_eq!(st.get_by_name("tm_sec"), Some(45));
        assert_eq!(st.get_by_name("tm_wday"), Some(5));
        assert_eq!(st.get_by_name("tm_yday"), Some(167));
        assert_eq!(st.get_by_name("tm_isdst"), Some(1));
        assert_eq!(st.get_by_name("invalid"), None);
    }

    #[test]
    fn test_validation() {
        let valid = StructTime::new(2024, 6, 15, 10, 30, 45, 5, 167, 1);
        assert!(valid.validate().is_ok());

        let invalid_month = StructTime::new(2024, 13, 15, 10, 30, 45, 5, 167, 1);
        assert!(invalid_month.validate().is_err());

        let invalid_day = StructTime::new(2024, 2, 30, 10, 30, 45, 5, 60, 1);
        assert!(invalid_day.validate().is_err());

        let invalid_hour = StructTime::new(2024, 6, 15, 25, 30, 45, 5, 167, 1);
        assert!(invalid_hour.validate().is_err());
    }

    #[test]
    fn test_from_tuple() {
        let values = vec![2024, 6, 15, 10, 30, 45, 5, 167, 1];
        let st = StructTime::from_tuple(&values).unwrap();

        assert_eq!(st.tm_year, 2024);
        assert_eq!(st.tm_mon, 6);
        assert_eq!(st.tm_mday, 15);
    }

    #[test]
    fn test_to_array() {
        let st = StructTime::new(2024, 6, 15, 10, 30, 45, 5, 167, 1);
        let arr = st.to_array();

        // Verify it has 9 elements with correct values
        assert_eq!(arr.len(), 9);
        assert_eq!(arr[0], 2024); // tm_year
        assert_eq!(arr[1], 6); // tm_mon
        assert_eq!(arr[2], 15); // tm_mday
        assert_eq!(arr[3], 10); // tm_hour
        assert_eq!(arr[4], 30); // tm_min
        assert_eq!(arr[5], 45); // tm_sec
        assert_eq!(arr[6], 5); // tm_wday
        assert_eq!(arr[7], 167); // tm_yday
        assert_eq!(arr[8], 1); // tm_isdst
    }

    #[test]
    fn test_display() {
        let st = StructTime::new(2024, 6, 15, 10, 30, 45, 5, 167, 1);
        let s = format!("{}", st);
        assert!(s.contains("tm_year=2024"));
        assert!(s.contains("tm_mon=6"));
    }

    #[test]
    fn test_end_of_year_transitions() {
        // Dec 31, 2023 23:59:59
        let timestamp = 1704067199i64;
        let st = StructTime::from_timestamp_utc(timestamp);
        assert_eq!(st.tm_year, 2023);
        assert_eq!(st.tm_mon, 12);
        assert_eq!(st.tm_mday, 31);
        assert_eq!(st.tm_hour, 23);
        assert_eq!(st.tm_min, 59);
        assert_eq!(st.tm_sec, 59);

        // Jan 1, 2024 00:00:00
        let timestamp = 1704067200i64;
        let st = StructTime::from_timestamp_utc(timestamp);
        assert_eq!(st.tm_year, 2024);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 1);
        assert_eq!(st.tm_hour, 0);
        assert_eq!(st.tm_min, 0);
        assert_eq!(st.tm_sec, 0);
    }

    #[test]
    fn test_leap_year_feb29() {
        // Feb 29, 2024 (leap year)
        let st = StructTime::new(2024, 2, 29, 12, 0, 0, 4, 60, 0);
        assert!(st.validate().is_ok());

        // Feb 29, 2023 (not leap year) - should fail validation
        let st = StructTime::new(2023, 2, 29, 12, 0, 0, 4, 60, 0);
        assert!(st.validate().is_err());
    }
}
