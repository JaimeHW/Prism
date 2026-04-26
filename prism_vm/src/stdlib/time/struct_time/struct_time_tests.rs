
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
