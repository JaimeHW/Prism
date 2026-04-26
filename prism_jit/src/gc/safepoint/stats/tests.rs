use super::*;

#[test]
fn test_stats_new() {
    let stats = SafepointStats::new();
    assert_eq!(stats.stw_requests(), 0);
    assert_eq!(stats.stw_achieved(), 0);
    assert_eq!(stats.polls_executed(), 0);
    assert_eq!(stats.traps_triggered(), 0);
}

#[test]
fn test_stats_record_request() {
    let stats = SafepointStats::new();
    stats.record_stw_request();
    stats.record_stw_request();
    assert_eq!(stats.stw_requests(), 2);
}

#[test]
fn test_stats_record_achieved() {
    let stats = SafepointStats::new();
    stats.record_stw_achieved(Duration::from_micros(100));
    stats.record_stw_achieved(Duration::from_micros(200));
    assert_eq!(stats.stw_achieved(), 2);
    assert_eq!(stats.avg_achieve_time(), Duration::from_micros(150));
}

#[test]
fn test_stats_record_duration() {
    let stats = SafepointStats::new();
    stats.record_stw_achieved(Duration::ZERO);
    stats.record_stw_duration(Duration::from_millis(10));
    stats.record_stw_achieved(Duration::ZERO);
    stats.record_stw_duration(Duration::from_millis(20));

    assert_eq!(stats.min_pause_time(), Duration::from_millis(10));
    assert_eq!(stats.max_pause_time(), Duration::from_millis(20));
}

#[test]
fn test_stats_record_polls() {
    let stats = SafepointStats::new();
    for _ in 0..100 {
        stats.record_poll();
    }
    stats.record_trap();

    assert_eq!(stats.polls_executed(), 100);
    assert_eq!(stats.traps_triggered(), 1);
    assert!((stats.trap_rate() - 0.01).abs() < 0.001);
}

#[test]
fn test_stats_reset() {
    let stats = SafepointStats::new();
    stats.record_stw_request();
    stats.record_poll();
    stats.reset();

    assert_eq!(stats.stw_requests(), 0);
    assert_eq!(stats.polls_executed(), 0);
}

#[test]
fn test_stats_clone() {
    let stats = SafepointStats::new();
    stats.record_stw_request();
    stats.record_stw_request();

    let cloned = stats.clone();
    assert_eq!(cloned.stw_requests(), 2);
}

#[test]
fn test_stats_display() {
    let stats = SafepointStats::new();
    stats.record_stw_request();

    let display = format!("{}", stats);
    assert!(display.contains("Safepoint Statistics"));
    assert!(display.contains("STW Requests"));
}
