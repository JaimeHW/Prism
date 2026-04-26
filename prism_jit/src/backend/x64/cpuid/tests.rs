use super::*;

#[test]
fn test_baseline_features() {
    let baseline = CpuFeatures::baseline();

    // x86-64 baseline includes SSE2
    assert!(baseline.has_sse());
    assert!(baseline.has_sse2());
    assert!(baseline.has_cmov());

    // But not newer features
    assert!(!baseline.has_avx());
    assert!(!baseline.has_avx2());
}

#[test]
fn test_feature_detection() {
    let features = CpuFeatures::detect();

    // All x86-64 CPUs have these
    assert!(features.has_sse());
    assert!(features.has_sse2());
    assert!(features.has_cmov());

    // Cache line should be reasonable
    assert!(features.cache_line_size() >= 32);
    assert!(features.cache_line_size() <= 128);
}

#[test]
fn test_vendor_detection() {
    let features = CpuFeatures::detect();

    // Should detect one of the known vendors on most systems
    let vendor = features.vendor();
    assert!(vendor == CpuVendor::Intel || vendor == CpuVendor::Amd || vendor == CpuVendor::Unknown);
}

#[test]
fn test_cpu_level_ordering() {
    assert!(CpuLevel::Baseline < CpuLevel::Sse42);
    assert!(CpuLevel::Sse42 < CpuLevel::Avx);
    assert!(CpuLevel::Avx < CpuLevel::Avx2);
    assert!(CpuLevel::Avx2 < CpuLevel::Avx512);
}

#[test]
fn test_cpu_level_detection() {
    let level = CpuLevel::detect();

    // Should be at least baseline
    assert!(level >= CpuLevel::Baseline);
}

#[test]
fn test_cpu_level_names() {
    assert_eq!(CpuLevel::Baseline.name(), "x86-64");
    assert_eq!(CpuLevel::Sse42.name(), "x86-64-v2");
    assert_eq!(CpuLevel::Avx2.name(), "x86-64-v3");
    assert_eq!(CpuLevel::Avx512.name(), "x86-64-v4");
}

#[test]
fn test_feature_flags_display() {
    let flags = CpuFeatureFlags::SSE | CpuFeatureFlags::SSE2 | CpuFeatureFlags::AVX;

    // Should be debug-printable
    let debug_str = format!("{:?}", flags);
    assert!(debug_str.contains("SSE"));
    assert!(debug_str.contains("SSE2"));
    assert!(debug_str.contains("AVX"));
}

#[test]
fn test_is_vendor() {
    let features = CpuFeatures::detect();

    // At most one vendor should be true
    let intel = features.is_intel();
    let amd = features.is_amd();

    // Can't be both Intel and AMD
    assert!(!(intel && amd));
}

#[test]
fn test_feature_hierarchy() {
    let features = CpuFeatures::detect();

    // AVX2 implies AVX
    if features.has_avx2() {
        assert!(features.has_avx());
    }

    // AVX implies SSE4.2
    if features.has_avx() {
        assert!(features.has_sse42());
    }

    // SSE4.2 implies SSE4.1
    if features.has_sse42() {
        assert!(features.has_sse41());
    }

    // SSE4.1 implies SSSE3
    if features.has_sse41() {
        assert!(features.has_ssse3());
    }
}

#[test]
fn test_cpu_level_from_features() {
    // Test with baseline
    let baseline = CpuFeatures::baseline();
    assert_eq!(CpuLevel::from_features(&baseline), CpuLevel::Baseline);
}

#[test]
fn test_flags_accessor() {
    let features = CpuFeatures::detect();
    let flags = features.flags();

    // Should at least have SSE2
    assert!(flags.contains(CpuFeatureFlags::SSE2));
}

#[test]
fn test_cached_detection() {
    // Multiple calls should return the same instance
    let features1 = CpuFeatures::detect();
    let features2 = CpuFeatures::detect();

    // Should be the same cached instance (same pointer)
    assert!(std::ptr::eq(features1, features2));
}
