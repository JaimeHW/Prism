//! Comprehensive test suite for the time module.
//!
//! Tests cover:
//! - Clock functions (time, monotonic, perf_counter, process_time, thread_time)
//! - Sleep functionality
//! - struct_time creation and conversion
//! - Time formatting (strftime/strptime)
//! - Timezone handling
//! - Edge cases and error conditions

use super::*;
use std::time::Duration;

// =============================================================================
// Clock Function Tests
// =============================================================================

mod clock_tests {
    use super::*;

    #[test]
    fn test_time_returns_positive() {
        let t = time();
        assert!(t > 0.0, "time() should return positive value");
    }

    #[test]
    fn test_time_ns_returns_positive() {
        let t = time_ns();
        assert!(t > 0, "time_ns() should return positive value");
    }

    #[test]
    fn test_time_increases() {
        let t1 = time();
        std::thread::sleep(Duration::from_millis(10));
        let t2 = time();
        assert!(t2 > t1, "time() should increase");
    }

    #[test]
    fn test_time_ns_precision() {
        let t_ns = time_ns();
        let t_s = time();
        // Should agree within 1 second
        let diff = ((t_ns as f64 / 1_000_000_000.0) - t_s).abs();
        assert!(diff < 1.0, "time() and time_ns() should agree");
    }

    #[test]
    fn test_monotonic_never_decreases() {
        let mut prev = monotonic();
        for _ in 0..100 {
            let curr = monotonic();
            assert!(curr >= prev, "monotonic() should never decrease");
            prev = curr;
        }
    }

    #[test]
    fn test_monotonic_ns_never_decreases() {
        let mut prev = monotonic_ns();
        for _ in 0..100 {
            let curr = monotonic_ns();
            assert!(curr >= prev, "monotonic_ns() should never decrease");
            prev = curr;
        }
    }

    #[test]
    fn test_perf_counter_increases() {
        let t1 = perf_counter();
        std::thread::sleep(Duration::from_millis(10));
        let t2 = perf_counter();
        assert!(t2 > t1, "perf_counter() should increase");
    }

    #[test]
    fn test_perf_counter_ns_precision() {
        let t1 = perf_counter_ns();
        std::thread::sleep(Duration::from_millis(10));
        let t2 = perf_counter_ns();
        // Should have elapsed at least 10ms = 10_000_000 ns
        assert!(
            t2 - t1 >= 9_000_000,
            "perf_counter_ns() should measure at least 9ms"
        );
    }

    #[test]
    fn test_process_time_non_negative() {
        let t = process_time();
        assert!(t >= 0.0, "process_time() should be non-negative");
    }

    #[test]
    fn test_process_time_increases_with_work() {
        let t1 = process_time();
        // Do some CPU work
        let mut sum = 0u64;
        for i in 0..1_000_000 {
            sum = sum.wrapping_add(i);
        }
        let _ = sum; // Prevent optimization
        let t2 = process_time();
        assert!(t2 >= t1, "process_time() should not decrease");
    }

    #[test]
    fn test_thread_time_non_negative() {
        let t = thread_time();
        assert!(t >= 0.0, "thread_time() should be non-negative");
    }
}

// =============================================================================
// Sleep Tests
// =============================================================================

mod sleep_tests {
    use super::*;

    #[test]
    fn test_sleep_duration() {
        let start = perf_counter();
        sleep(0.05); // 50ms
        let elapsed = perf_counter() - start;
        // Allow 10ms tolerance
        assert!(
            elapsed >= 0.04,
            "sleep should wait at least 40ms, got {}s",
            elapsed
        );
        assert!(
            elapsed < 0.2,
            "sleep should not wait too long, got {}s",
            elapsed
        );
    }

    #[test]
    fn test_sleep_zero() {
        let start = perf_counter();
        sleep(0.0);
        let elapsed = perf_counter() - start;
        assert!(elapsed < 0.1, "sleep(0) should return quickly");
    }

    #[test]
    fn test_sleep_negative() {
        // Negative sleep should return immediately
        let start = perf_counter();
        sleep(-1.0);
        let elapsed = perf_counter() - start;
        assert!(elapsed < 0.1, "sleep(negative) should return quickly");
    }

    #[test]
    fn test_sleep_small_duration() {
        let start = perf_counter();
        sleep(0.001); // 1ms
        let elapsed = perf_counter() - start;
        assert!(elapsed >= 0.0005, "sleep(1ms) should wait");
    }
}

// =============================================================================
// StructTime Tests
// =============================================================================

mod struct_time_tests {
    use super::*;

    #[test]
    fn test_gmtime_epoch() {
        let st = gmtime(Some(0.0));
        assert_eq!(st.tm_year, 1970);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 1);
        assert_eq!(st.tm_hour, 0);
        assert_eq!(st.tm_min, 0);
        assert_eq!(st.tm_sec, 0);
    }

    #[test]
    fn test_gmtime_known_date() {
        // 2024-01-15 12:30:45 UTC = 1705321845
        let st = gmtime(Some(1705321845.0));
        assert_eq!(st.tm_year, 2024);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 15);
        assert_eq!(st.tm_hour, 12);
        assert_eq!(st.tm_min, 30);
        assert_eq!(st.tm_sec, 45);
    }

    #[test]
    fn test_gmtime_current() {
        let st = gmtime(None);
        assert!(st.tm_year >= 2024, "Year should be >= 2024");
        assert!((1..=12).contains(&st.tm_mon), "Month should be 1-12");
        assert!((1..=31).contains(&st.tm_mday), "Day should be 1-31");
    }

    #[test]
    fn test_localtime_current() {
        let st = localtime(None);
        assert!(st.tm_year >= 2024, "Year should be >= 2024");
        assert!((1..=12).contains(&st.tm_mon), "Month should be 1-12");
        assert!((1..=31).contains(&st.tm_mday), "Day should be 1-31");
    }

    #[test]
    fn test_mktime_roundtrip() {
        let original = 1705321845.0;
        let st = gmtime(Some(original));
        let recovered = mktime(&st);
        // Allow 1 hour difference for timezone
        assert!((recovered - original).abs() < 3700.0);
    }

    #[test]
    fn test_struct_time_tuple_access() {
        let st = StructTime::new(2024, 6, 15, 10, 30, 45, 5, 167, 1);
        assert_eq!(st.get(0), Some(2024));
        assert_eq!(st.get(1), Some(6));
        assert_eq!(st.get(2), Some(15));
        assert_eq!(st.get(9), None);
    }

    #[test]
    fn test_struct_time_named_access() {
        let st = StructTime::new(2024, 6, 15, 10, 30, 45, 5, 167, 1);
        assert_eq!(st.get_by_name("tm_year"), Some(2024));
        assert_eq!(st.get_by_name("tm_mon"), Some(6));
        assert_eq!(st.get_by_name("invalid"), None);
    }

    #[test]
    fn test_struct_time_validation() {
        let valid = StructTime::new(2024, 6, 15, 10, 30, 45, 5, 167, 1);
        assert!(valid.validate().is_ok());

        let invalid_month = StructTime::new(2024, 13, 15, 10, 30, 45, 5, 167, 1);
        assert!(invalid_month.validate().is_err());

        let invalid_hour = StructTime::new(2024, 6, 15, 25, 30, 45, 5, 167, 1);
        assert!(invalid_hour.validate().is_err());
    }

    #[test]
    fn test_leap_year_handling() {
        // Feb 29, 2024 (leap year)
        let st = StructTime::new(2024, 2, 29, 12, 0, 0, 0, 60, 0);
        assert!(st.validate().is_ok());

        // Feb 29, 2023 (not leap year)
        let st = StructTime::new(2023, 2, 29, 12, 0, 0, 0, 60, 0);
        assert!(st.validate().is_err());
    }
}

// =============================================================================
// Format Tests
// =============================================================================

mod format_tests {
    use super::*;

    #[test]
    fn test_strftime_date() {
        let st = StructTime::new(2024, 1, 15, 12, 30, 45, 1, 15, 0);
        assert_eq!(strftime_fn("%Y-%m-%d", Some(&st)).unwrap(), "2024-01-15");
    }

    #[test]
    fn test_strftime_time() {
        let st = StructTime::new(2024, 1, 15, 12, 30, 45, 1, 15, 0);
        assert_eq!(strftime_fn("%H:%M:%S", Some(&st)).unwrap(), "12:30:45");
    }

    #[test]
    fn test_strftime_combined() {
        let st = StructTime::new(2024, 1, 15, 12, 30, 45, 1, 15, 0);
        assert_eq!(
            strftime_fn("%Y-%m-%d %H:%M:%S", Some(&st)).unwrap(),
            "2024-01-15 12:30:45"
        );
    }

    #[test]
    fn test_strftime_weekday() {
        let st = StructTime::new(2024, 1, 15, 12, 30, 45, 0, 15, 0); // Monday
        assert_eq!(strftime_fn("%a", Some(&st)).unwrap(), "Mon");
        assert_eq!(strftime_fn("%A", Some(&st)).unwrap(), "Monday");
    }

    #[test]
    fn test_strftime_month() {
        let st = StructTime::new(2024, 3, 15, 12, 30, 45, 4, 75, 0);
        assert_eq!(strftime_fn("%b", Some(&st)).unwrap(), "Mar");
        assert_eq!(strftime_fn("%B", Some(&st)).unwrap(), "March");
    }

    #[test]
    fn test_strftime_12_hour() {
        let am = StructTime::new(2024, 1, 15, 9, 30, 0, 1, 15, 0);
        let pm = StructTime::new(2024, 1, 15, 21, 30, 0, 1, 15, 0);
        assert_eq!(strftime_fn("%I %p", Some(&am)).unwrap(), "09 AM");
        assert_eq!(strftime_fn("%I %p", Some(&pm)).unwrap(), "09 PM");
    }

    #[test]
    fn test_strftime_percent() {
        let st = StructTime::new(2024, 1, 15, 12, 30, 45, 1, 15, 0);
        assert_eq!(strftime_fn("100%%", Some(&st)).unwrap(), "100%");
    }

    #[test]
    fn test_strptime_date() {
        let st = strptime_fn("2024-01-15", "%Y-%m-%d").unwrap();
        assert_eq!(st.tm_year, 2024);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 15);
    }

    #[test]
    fn test_strptime_time() {
        let st = strptime_fn("12:30:45", "%H:%M:%S").unwrap();
        assert_eq!(st.tm_hour, 12);
        assert_eq!(st.tm_min, 30);
        assert_eq!(st.tm_sec, 45);
    }

    #[test]
    fn test_strptime_combined() {
        let st = strptime_fn("2024-01-15 12:30:45", "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(st.tm_year, 2024);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 15);
        assert_eq!(st.tm_hour, 12);
        assert_eq!(st.tm_min, 30);
        assert_eq!(st.tm_sec, 45);
    }

    #[test]
    fn test_strptime_month_name() {
        let st = strptime_fn("Jan 15, 2024", "%b %d, %Y").unwrap();
        assert_eq!(st.tm_year, 2024);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 15);
    }

    #[test]
    fn test_format_roundtrip() {
        let original = StructTime::new(2024, 6, 15, 14, 30, 45, 5, 167, 0);
        let formatted = strftime_fn("%Y-%m-%d %H:%M:%S", Some(&original)).unwrap();
        let parsed = strptime_fn(&formatted, "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(original.tm_year, parsed.tm_year);
        assert_eq!(original.tm_mon, parsed.tm_mon);
        assert_eq!(original.tm_mday, parsed.tm_mday);
        assert_eq!(original.tm_hour, parsed.tm_hour);
        assert_eq!(original.tm_min, parsed.tm_min);
        assert_eq!(original.tm_sec, parsed.tm_sec);
    }
}

// =============================================================================
// Asctime/Ctime Tests
// =============================================================================

mod asctime_tests {
    use super::*;

    #[test]
    fn test_asctime_format() {
        let st = StructTime::new(2024, 1, 15, 12, 30, 45, 0, 15, 0); // Monday
        let s = asctime(Some(&st)).unwrap();
        assert!(s.contains("Mon"), "Should contain weekday");
        assert!(s.contains("Jan"), "Should contain month");
        assert!(s.contains("15"), "Should contain day");
        assert!(s.contains("12:30:45"), "Should contain time");
        assert!(s.contains("2024"), "Should contain year");
    }

    #[test]
    fn test_ctime_format() {
        let s = ctime(Some(1705321845.0)).unwrap();
        assert!(
            s.contains("2024") || s.contains("Jan"),
            "Should contain date info"
        );
    }
}

// =============================================================================
// Clock Info Tests
// =============================================================================

mod clock_info_tests {
    use super::*;

    #[test]
    fn test_get_clock_info_time() {
        let info = get_clock_info("time").unwrap();
        assert!(!info.monotonic, "time clock should not be monotonic");
        assert!(info.adjustable, "time clock should be adjustable");
        assert!(info.resolution > 0.0, "resolution should be positive");
    }

    #[test]
    fn test_get_clock_info_monotonic() {
        let info = get_clock_info("monotonic").unwrap();
        assert!(info.monotonic, "monotonic clock should be monotonic");
        assert!(!info.adjustable, "monotonic clock should not be adjustable");
    }

    #[test]
    fn test_get_clock_info_perf_counter() {
        let info = get_clock_info("perf_counter").unwrap();
        assert!(info.monotonic, "perf_counter should be monotonic");
    }

    #[test]
    fn test_get_clock_info_invalid() {
        let result = get_clock_info("invalid_clock");
        assert!(result.is_err());
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_negative_timestamp() {
        let st = gmtime(Some(-86400.0)); // Dec 31, 1969
        assert_eq!(st.tm_year, 1969);
        assert_eq!(st.tm_mon, 12);
        assert_eq!(st.tm_mday, 31);
    }

    #[test]
    fn test_far_future_timestamp() {
        let st = gmtime(Some(4102444800.0)); // Jan 1, 2100
        assert_eq!(st.tm_year, 2100);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 1);
    }

    #[test]
    fn test_end_of_year() {
        let st = gmtime(Some(1704067199.0)); // Dec 31, 2023 23:59:59
        assert_eq!(st.tm_year, 2023);
        assert_eq!(st.tm_mon, 12);
        assert_eq!(st.tm_mday, 31);
        assert_eq!(st.tm_hour, 23);
        assert_eq!(st.tm_min, 59);
        assert_eq!(st.tm_sec, 59);
    }

    #[test]
    fn test_start_of_year() {
        let st = gmtime(Some(1704067200.0)); // Jan 1, 2024 00:00:00
        assert_eq!(st.tm_year, 2024);
        assert_eq!(st.tm_mon, 1);
        assert_eq!(st.tm_mday, 1);
        assert_eq!(st.tm_hour, 0);
        assert_eq!(st.tm_min, 0);
        assert_eq!(st.tm_sec, 0);
    }

    #[test]
    fn test_leap_second_handling() {
        // struct_time should accept seconds up to 61 for leap seconds
        let st = StructTime::new(2024, 6, 30, 23, 59, 60, 0, 182, 0);
        // This should be valid for struct_time (leap second)
        assert!(st.validate().is_ok());
    }

    #[test]
    fn test_strptime_error_invalid_format() {
        let result = strptime_fn("not-a-date", "%Y-%m-%d");
        assert!(result.is_err());
    }

    #[test]
    fn test_strptime_error_incomplete() {
        let result = strptime_fn("2024-01", "%Y-%m-%d");
        assert!(result.is_err());
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_workflow() {
        // Get current time
        let now = time();
        assert!(now > 0.0);

        // Convert to struct_time
        let st = gmtime(Some(now));
        assert!(st.tm_year >= 2024);

        // Format it
        let formatted = strftime_fn("%Y-%m-%d %H:%M:%S", Some(&st)).unwrap();
        assert!(!formatted.is_empty());

        // Parse it back
        let parsed = strptime_fn(&formatted, "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(st.tm_year, parsed.tm_year);
        assert_eq!(st.tm_mon, parsed.tm_mon);
        assert_eq!(st.tm_mday, parsed.tm_mday);

        // Convert back to timestamp
        let timestamp = mktime(&parsed);
        assert!(timestamp > 0.0);
    }

    #[test]
    fn test_perf_counter_for_benchmarking() {
        let start = perf_counter();

        // Do some work
        let mut sum = 0u64;
        for i in 0..100_000 {
            sum = sum.wrapping_add(i);
        }
        let _ = sum;

        let elapsed = perf_counter() - start;
        assert!(elapsed >= 0.0, "Elapsed time should be non-negative");
        assert!(elapsed < 10.0, "Should complete within 10 seconds");
    }

    #[test]
    fn test_monotonic_for_intervals() {
        let t1 = monotonic();
        std::thread::sleep(Duration::from_millis(50));
        let t2 = monotonic();
        let elapsed = t2 - t1;
        assert!(
            elapsed >= 0.045,
            "Should measure at least 45ms, got {}s",
            elapsed
        );
        assert!(elapsed < 0.5, "Should not take too long");
    }
}
