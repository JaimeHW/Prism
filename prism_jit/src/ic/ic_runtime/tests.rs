use super::*;
use prism_runtime::object::shape::PropertyFlags;

#[test]
fn test_ic_get_property_miss() {
    let mut ic = PropertyIc::new();

    let result = ic_get_property_miss(&mut ic, ShapeId(42), 5, PropertyFlags::default());

    assert_eq!(result.offset, 5);
    assert_eq!(ic.state(), IcState::Monomorphic);
}

#[test]
fn test_ic_set_property_miss() {
    let mut ic = PropertyIc::new();

    let result = ic_set_property_miss(&mut ic, ShapeId(10), 3, PropertyFlags::default());

    assert_eq!(result.offset, 3);
    assert_eq!(ic.state(), IcState::Monomorphic);
}

#[test]
fn test_ic_call_miss() {
    let mut ic = CallIc::new();

    ic_call_miss(&mut ic, 100, 0x1000, 2);

    assert_eq!(ic.state(), super::super::call_ic::CallIcState::Monomorphic);
}

#[test]
fn test_ic_method_call_miss() {
    let mut ic = CallIc::new();

    ic_method_call_miss(&mut ic, 100, 0x1000, 2, ShapeId(42));

    assert_eq!(ic.state(), super::super::call_ic::CallIcState::Monomorphic);
}

#[test]
fn test_ic_property_lookup_hit() {
    let mut ic = PropertyIc::new();
    ic.update(ShapeId(1), 5, PropertyFlags::default());

    let result = ic_property_lookup(&mut ic, ShapeId(1));
    match result {
        IcLookupResult::Hit(info) => assert_eq!(info.offset, 5),
        _ => panic!("Expected hit"),
    }
}

#[test]
fn test_ic_property_lookup_miss() {
    let mut ic = PropertyIc::new();
    ic.update(ShapeId(1), 5, PropertyFlags::default());

    let result = ic_property_lookup(&mut ic, ShapeId(2));
    assert!(matches!(result, IcLookupResult::Miss));
}

#[test]
fn test_ic_property_lookup_megamorphic() {
    let mut ic = PropertyIc::new();
    ic.force_state(IcState::Megamorphic);

    let result = ic_property_lookup(&mut ic, ShapeId(1));
    assert!(matches!(result, IcLookupResult::Megamorphic));
}

#[test]
fn test_ic_runtime_stats() {
    let stats = IcRuntimeStats {
        property_hits: 80,
        property_misses: 20,
        call_hits: 90,
        call_misses: 10,
        mega_hits: 50,
        mega_misses: 50,
        ..Default::default()
    };

    assert!((stats.property_hit_rate() - 0.8).abs() < 0.001);
    assert!((stats.call_hit_rate() - 0.9).abs() < 0.001);
}

#[test]
fn test_ic_runtime_stats_overall() {
    let stats = IcRuntimeStats {
        property_hits: 100,
        property_misses: 0,
        call_hits: 100,
        call_misses: 0,
        mega_hits: 0,
        mega_misses: 0,
        ..Default::default()
    };

    assert_eq!(stats.overall_hit_rate(), 1.0);
}

#[test]
fn test_ic_runtime_stats_empty() {
    let stats = IcRuntimeStats::default();
    assert_eq!(stats.overall_hit_rate(), 0.0);
    assert_eq!(stats.property_hit_rate(), 0.0);
    assert_eq!(stats.call_hit_rate(), 0.0);
}
