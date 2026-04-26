    use super::*;
    use crate::regalloc::interval::{LiveRange, ProgPoint};

    fn make_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
        let mut interval = LiveInterval::new(VReg::new(vreg), RegClass::Int);
        interval.add_range(LiveRange::new(
            ProgPoint::before(start),
            ProgPoint::before(end),
        ));
        interval
    }

    #[test]
    fn test_simple_coloring() {
        let intervals = vec![
            make_interval(0, 0, 10),
            make_interval(1, 5, 15),
            make_interval(2, 20, 30),
        ];

        let config = AllocatorConfig::default();
        let allocator = GraphColoringAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        assert!(map.get(VReg::new(0)).is_register());
        assert!(map.get(VReg::new(1)).is_register());
        assert!(map.get(VReg::new(2)).is_register());
        assert_eq!(stats.num_spilled, 0);
    }

    #[test]
    fn test_different_colors() {
        // Overlapping intervals must get different colors
        let intervals = vec![make_interval(0, 0, 20), make_interval(1, 10, 30)];

        let config = AllocatorConfig::default();
        let allocator = GraphColoringAllocator::new(config);
        let (map, _) = allocator.allocate(intervals);

        let r0 = map.get(VReg::new(0)).reg();
        let r1 = map.get(VReg::new(1)).reg();

        assert!(r0.is_some());
        assert!(r1.is_some());
        assert_ne!(r0, r1);
    }

    #[test]
    fn test_spill_when_needed() {
        // Create more overlapping intervals than registers
        let mut intervals = Vec::new();
        for i in 0..16 {
            intervals.push(make_interval(i, 0, 100));
        }

        let config = AllocatorConfig::default();
        let allocator = GraphColoringAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // With 14 GPRs available, should spill at least 2
        assert!(stats.num_spilled >= 2);
        assert!(map.spill_slot_count() >= 2);
    }
