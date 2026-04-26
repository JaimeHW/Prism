use super::*;

#[test]
fn test_prog_point() {
    let p1 = ProgPoint::before(5);
    let p2 = ProgPoint::after(5);

    assert!(p1 < p2);
    assert!(p1.is_before());
    assert!(p2.is_after());
    assert_eq!(p1.inst_index(), 5);
    assert_eq!(p2.inst_index(), 5);
}

#[test]
fn test_live_range_overlap() {
    let r1 = LiveRange::new(ProgPoint::before(0), ProgPoint::before(10));
    let r2 = LiveRange::new(ProgPoint::before(5), ProgPoint::before(15));
    let r3 = LiveRange::new(ProgPoint::before(10), ProgPoint::before(20));

    assert!(r1.overlaps(&r2));
    assert!(!r1.overlaps(&r3)); // [0, 10) and [10, 20) don't overlap
}

#[test]
fn test_live_interval_add_range() {
    let mut interval = LiveInterval::new(VReg::new(0), RegClass::Int);

    interval.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(5)));
    interval.add_range(LiveRange::new(ProgPoint::before(10), ProgPoint::before(15)));
    interval.add_range(LiveRange::new(ProgPoint::before(5), ProgPoint::before(10))); // Fills gap

    // Should merge into one range [0, 15)
    assert_eq!(interval.ranges().len(), 1);
    assert_eq!(interval.start(), ProgPoint::before(0));
    assert_eq!(interval.end(), ProgPoint::before(15));
}

#[test]
fn test_live_interval_contains() {
    let mut interval = LiveInterval::new(VReg::new(0), RegClass::Int);
    interval.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(10)));
    interval.add_range(LiveRange::new(ProgPoint::before(20), ProgPoint::before(30)));

    assert!(interval.contains(ProgPoint::before(5)));
    assert!(interval.contains(ProgPoint::before(25)));
    assert!(!interval.contains(ProgPoint::before(15))); // In the hole
}

#[test]
fn test_live_interval_split() {
    let mut interval = LiveInterval::new(VReg::new(0), RegClass::Int);
    interval.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(20)));
    interval.add_use(UsePosition::def(ProgPoint::before(0)));
    interval.add_use(UsePosition::use_pos(ProgPoint::before(5)));
    interval.add_use(UsePosition::use_pos(ProgPoint::before(15)));

    let second = interval.split_at(ProgPoint::before(10), VReg::new(1));

    assert!(second.is_some());
    let second = second.unwrap();

    assert_eq!(interval.end(), ProgPoint::before(10));
    assert_eq!(second.start(), ProgPoint::before(10));
    assert_eq!(second.end(), ProgPoint::before(20));

    // Check uses were split correctly
    // Uses at positions < 10 stay in first half, uses at positions >= 10 go to second
    assert_eq!(interval.uses().len(), 2); // def at 0 and use at 5
    assert_eq!(second.uses().len(), 1); // use at 15
}
