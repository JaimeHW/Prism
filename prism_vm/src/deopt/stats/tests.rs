use super::*;

#[test]
fn test_deopt_site_key() {
    let key1 = DeoptSiteKey::new(1, 100);
    let key2 = DeoptSiteKey::new(1, 100);
    let key3 = DeoptSiteKey::new(1, 200);

    assert_eq!(key1, key2);
    assert_ne!(key1, key3);
}

#[test]
fn test_deopt_site_creation() {
    let key = DeoptSiteKey::new(1, 100);
    let site = DeoptSite::new(key);

    assert_eq!(site.total_count(), 0);
    assert!(!site.is_patched());
}

#[test]
fn test_deopt_site_record() {
    let key = DeoptSiteKey::new(1, 100);
    let site = DeoptSite::new(key);

    site.record(DeoptReason::TypeGuard, 12345);
    site.record(DeoptReason::TypeGuard, 12346);
    site.record(DeoptReason::Overflow, 12347);

    assert_eq!(site.total_count(), 3);
    assert_eq!(site.count_for_reason(DeoptReason::TypeGuard), 2);
    assert_eq!(site.count_for_reason(DeoptReason::Overflow), 1);
}

#[test]
fn test_deopt_site_dominant_reason() {
    let key = DeoptSiteKey::new(1, 100);
    let site = DeoptSite::new(key);

    site.record(DeoptReason::TypeGuard, 1);
    site.record(DeoptReason::TypeGuard, 2);
    site.record(DeoptReason::Overflow, 3);

    assert_eq!(site.dominant_reason(), Some(DeoptReason::TypeGuard));
}

#[test]
fn test_deopt_site_patched() {
    let key = DeoptSiteKey::new(1, 100);
    let site = DeoptSite::new(key);

    assert!(!site.is_patched());
    site.set_patched();
    assert!(site.is_patched());
}

#[test]
fn test_deopt_stats_creation() {
    let stats = DeoptStats::new();
    assert_eq!(stats.total_count(), 0);
    assert_eq!(stats.site_count(), 0);
}

#[test]
fn test_deopt_stats_record() {
    let stats = DeoptStats::new();

    stats.record(1, 100, DeoptReason::TypeGuard);
    stats.record(1, 100, DeoptReason::TypeGuard);
    stats.record(1, 200, DeoptReason::Overflow);

    assert_eq!(stats.total_count(), 3);
    assert_eq!(stats.site_count(), 2);
    assert_eq!(stats.count_for_reason(DeoptReason::TypeGuard), 2);
}

#[test]
fn test_deopt_stats_get_site() {
    let stats = DeoptStats::new();

    stats.record(1, 100, DeoptReason::TypeGuard);

    let snapshot = stats.get_site(1, 100);
    assert!(snapshot.is_some());
    assert_eq!(snapshot.unwrap().total_count, 1);

    let snapshot = stats.get_site(1, 999);
    assert!(snapshot.is_none());
}

#[test]
fn test_deopt_stats_hot_sites() {
    let stats = DeoptStats::new();

    for _ in 0..10 {
        stats.record(1, 100, DeoptReason::TypeGuard);
    }
    stats.record(1, 200, DeoptReason::Overflow);

    let hot = stats.hot_sites(5);
    assert_eq!(hot.len(), 1);
    assert_eq!(hot[0].bc_offset, 100);
}

#[test]
fn test_deopt_stats_reset() {
    let stats = DeoptStats::new();

    stats.record(1, 100, DeoptReason::TypeGuard);
    stats.reset();

    assert_eq!(stats.total_count(), 0);
    assert_eq!(stats.site_count(), 0);
}
