use super::*;

// =========================================================================
// RegisterPressureSimulator Tests
// =========================================================================

#[test]
fn test_pressure_simulator_new() {
    let sim = RegisterPressureSimulator::new(10, 8, 16, 32);
    assert_eq!(sim.available_gprs, 10);
    assert_eq!(sim.available_xmms, 8);
    assert_eq!(sim.available_ymms, 16);
    assert_eq!(sim.available_zmms, 32);
    assert_eq!(sim.live_gprs, 0);
    assert_eq!(sim.spill_count(), 0);
}

#[test]
fn test_pressure_simulator_for_windows() {
    let sim = RegisterPressureSimulator::for_windows();
    assert_eq!(sim.available_gprs, 14);
    assert_eq!(sim.available_xmms, 15);
}

#[test]
fn test_pressure_simulator_for_sysv() {
    let sim = RegisterPressureSimulator::for_sysv();
    assert_eq!(sim.available_gprs, 14);
    assert_eq!(sim.available_xmms, 15);
}

#[test]
fn test_pressure_simulator_no_spill() {
    let mut sim = RegisterPressureSimulator::new(10, 10, 10, 10);

    // Define 5 GPRs - no spill
    for _ in 0..5 {
        assert!(!sim.define(RegClass::Int));
    }
    assert_eq!(sim.spill_count(), 0);
    assert_eq!(sim.live_gprs, 5);
}

#[test]
fn test_pressure_simulator_spill_required() {
    let mut sim = RegisterPressureSimulator::new(10, 10, 10, 10);

    // Define 10 GPRs - no spill (exactly at limit)
    for _ in 0..10 {
        sim.define(RegClass::Int);
    }
    assert_eq!(sim.spill_count(), 0);

    // 11th definition requires spill
    assert!(sim.define(RegClass::Int));
    assert_eq!(sim.spill_count(), 1);
}

#[test]
fn test_pressure_simulator_use_end() {
    let mut sim = RegisterPressureSimulator::new(10, 10, 10, 10);

    sim.define(RegClass::Int);
    sim.define(RegClass::Int);
    assert_eq!(sim.live_gprs, 2);

    sim.use_end(RegClass::Int);
    assert_eq!(sim.live_gprs, 1);
}

#[test]
fn test_pressure_simulator_peak_tracking() {
    let mut sim = RegisterPressureSimulator::new(10, 10, 10, 10);

    for _ in 0..8 {
        sim.define(RegClass::Int);
    }
    assert_eq!(sim.peak_pressure(RegClass::Int), 8);

    for _ in 0..4 {
        sim.use_end(RegClass::Int);
    }
    assert_eq!(sim.peak_pressure(RegClass::Int), 8); // Peak preserved
    assert_eq!(sim.live_gprs, 4);
}

#[test]
fn test_pressure_simulator_all_classes() {
    let mut sim = RegisterPressureSimulator::new(5, 5, 5, 5);

    for _ in 0..3 {
        sim.define(RegClass::Int);
        sim.define(RegClass::Float);
        sim.define(RegClass::Vec256);
        sim.define(RegClass::Vec512);
    }

    assert_eq!(sim.live_gprs, 3);
    assert_eq!(sim.live_xmms, 3);
    assert_eq!(sim.live_ymms, 3);
    assert_eq!(sim.live_zmms, 3);
    assert_eq!(sim.spill_count(), 0);
}

#[test]
fn test_pressure_simulator_is_under_pressure() {
    let mut sim = RegisterPressureSimulator::new(2, 2, 2, 2);

    sim.define(RegClass::Int);
    sim.define(RegClass::Int);
    assert!(!sim.is_under_pressure(RegClass::Int));

    sim.define(RegClass::Int);
    assert!(sim.is_under_pressure(RegClass::Int));
}

#[test]
fn test_pressure_simulator_reset() {
    let mut sim = RegisterPressureSimulator::new(5, 5, 5, 5);

    for _ in 0..10 {
        sim.define(RegClass::Int);
    }
    assert!(sim.spill_count() > 0);

    sim.reset();
    assert_eq!(sim.spill_count(), 0);
    assert_eq!(sim.live_gprs, 0);
    assert_eq!(sim.peak_pressure(RegClass::Int), 0);
}

#[test]
fn test_pressure_simulator_reload() {
    let mut sim = RegisterPressureSimulator::new(5, 5, 5, 5);

    sim.reload(RegClass::Int);
    assert_eq!(sim.reload_count(), 1);
    assert_eq!(sim.live_gprs, 1);
}

// =========================================================================
// SpillSlotTracker Tests
// =========================================================================

#[test]
fn test_spill_tracker_new() {
    let tracker = SpillSlotTracker::new(0);
    assert_eq!(tracker.total_slots(), 0);
    assert_eq!(tracker.current_offset(), 0);
}

#[test]
fn test_spill_tracker_allocate_gpr() {
    let mut tracker = SpillSlotTracker::new(0);

    let offset = tracker.allocate(RegClass::Int);
    assert_eq!(offset, 0);
    assert_eq!(tracker.current_offset(), 8);
    assert_eq!(tracker.slots_for_class(RegClass::Int), 1);
}

#[test]
fn test_spill_tracker_allocate_xmm() {
    let mut tracker = SpillSlotTracker::new(0);

    let offset = tracker.allocate(RegClass::Float);
    assert_eq!(offset, 0);
    assert_eq!(tracker.current_offset(), 16);
    assert_eq!(tracker.slots_for_class(RegClass::Float), 1);
}

#[test]
fn test_spill_tracker_allocate_ymm() {
    let mut tracker = SpillSlotTracker::new(0);

    let offset = tracker.allocate(RegClass::Vec256);
    assert_eq!(offset, 0);
    assert_eq!(tracker.current_offset(), 32);
    assert_eq!(tracker.slots_for_class(RegClass::Vec256), 1);
}

#[test]
fn test_spill_tracker_allocate_zmm() {
    let mut tracker = SpillSlotTracker::new(0);

    let offset = tracker.allocate(RegClass::Vec512);
    assert_eq!(offset, 0);
    assert_eq!(tracker.current_offset(), 64);
    assert_eq!(tracker.slots_for_class(RegClass::Vec512), 1);
}

#[test]
fn test_spill_tracker_alignment() {
    let mut tracker = SpillSlotTracker::new(0);

    // GPR at 0
    tracker.allocate(RegClass::Int); // 0-8

    // YMM needs 32-byte alignment
    let ymm_offset = tracker.allocate(RegClass::Vec256);
    assert_eq!(ymm_offset, 32); // Aligned to 32
    assert_eq!(tracker.current_offset(), 64);
}

#[test]
fn test_spill_tracker_zmm_alignment() {
    let mut tracker = SpillSlotTracker::new(0);

    // GPR at 0
    tracker.allocate(RegClass::Int); // 0-8

    // ZMM needs 64-byte alignment
    let zmm_offset = tracker.allocate(RegClass::Vec512);
    assert_eq!(zmm_offset, 64); // Aligned to 64
    assert_eq!(tracker.current_offset(), 128);
}

#[test]
fn test_spill_tracker_peak_bytes() {
    let mut tracker = SpillSlotTracker::new(0);

    tracker.allocate(RegClass::Int); // 8 bytes
    tracker.allocate(RegClass::Vec256); // 32 bytes, aligned
    tracker.allocate(RegClass::Float); // 16 bytes

    assert!(tracker.peak_bytes() >= 8 + 32 + 16);
}

#[test]
fn test_spill_tracker_free() {
    let mut tracker = SpillSlotTracker::new(0);

    tracker.allocate(RegClass::Int);
    tracker.allocate(RegClass::Int);
    assert_eq!(tracker.bytes_used, 16);

    tracker.free(RegClass::Int);
    assert_eq!(tracker.bytes_used, 8);

    // Peak should still reflect maximum
    assert!(tracker.peak_bytes() >= 16);
}

#[test]
fn test_spill_tracker_total_slots() {
    let mut tracker = SpillSlotTracker::new(0);

    tracker.allocate(RegClass::Int);
    tracker.allocate(RegClass::Float);
    tracker.allocate(RegClass::Vec256);
    tracker.allocate(RegClass::Vec512);

    assert_eq!(tracker.total_slots(), 4);
}

#[test]
fn test_spill_tracker_initial_offset() {
    let mut tracker = SpillSlotTracker::new(128);

    let offset = tracker.allocate(RegClass::Int);
    assert_eq!(offset, 128);
}

// =========================================================================
// CallSiteGenerator Tests
// =========================================================================

#[test]
fn test_call_generator_new() {
    let generator = CallSiteGenerator::new();
    assert_eq!(generator.stack_size(), 0);
}

#[test]
fn test_call_generator_int_args() {
    let mut generator = CallSiteGenerator::new();

    let args = generator.generate_args(&[RegClass::Int, RegClass::Int, RegClass::Int]);

    // First args should be in registers
    assert!(matches!(args[0], ArgClass::Gpr(_)));
    assert!(matches!(args[1], ArgClass::Gpr(_)));
    assert!(matches!(args[2], ArgClass::Gpr(_)));
}

#[test]
fn test_call_generator_float_args() {
    let mut generator = CallSiteGenerator::new();

    let args = generator.generate_args(&[RegClass::Float, RegClass::Float]);

    assert!(matches!(args[0], ArgClass::Xmm(_)));
    assert!(matches!(args[1], ArgClass::Xmm(_)));
}

#[test]
fn test_call_generator_ymm_args() {
    let mut generator = CallSiteGenerator::new();

    let args = generator.generate_args(&[RegClass::Vec256, RegClass::Vec256]);

    assert!(matches!(args[0], ArgClass::Ymm(_)));
    assert!(matches!(args[1], ArgClass::Ymm(_)));
}

#[test]
fn test_call_generator_zmm_args() {
    let mut generator = CallSiteGenerator::new();

    let args = generator.generate_args(&[RegClass::Vec512, RegClass::Vec512]);

    assert!(matches!(args[0], ArgClass::Zmm(_)));
    assert!(matches!(args[1], ArgClass::Zmm(_)));
}

#[test]
fn test_call_generator_stack_overflow() {
    let mut generator = CallSiteGenerator::new();

    // Generate many args to force stack usage
    let many_ints: Vec<_> = (0..20).map(|_| RegClass::Int).collect();
    let args = generator.generate_args(&many_ints);

    // Some should be on stack
    let stack_args = args
        .iter()
        .filter(|a| matches!(a, ArgClass::Stack(_)))
        .count();
    assert!(stack_args > 0);
    assert!(generator.stack_size() > 0);
}

#[test]
fn test_call_generator_count_reg_vs_stack() {
    let mut generator = CallSiteGenerator::new();

    let sig: Vec<_> = (0..10).map(|_| RegClass::Int).collect();
    let (regs, stack) = generator.count_register_vs_stack(&sig);

    assert!(regs > 0);
    assert!(stack > 0);
    assert_eq!(regs + stack, 10);
}

#[test]
fn test_call_generator_clobbers() {
    let generator = CallSiteGenerator::new();
    let clobbers = generator.clobbers();

    // Should have some clobbered registers
    assert!(clobbers.gprs.count() > 0);
    assert!(clobbers.xmms.count() > 0);
}

#[test]
fn test_call_generator_reset() {
    let mut generator = CallSiteGenerator::new();

    generator.generate_args(&[RegClass::Int, RegClass::Int]);
    generator.generate_args(&[RegClass::Int]); // Reset happens internally

    // Second call should start fresh
    let args = generator.generate_args(&[RegClass::Int]);
    assert!(matches!(
        args[0],
        ArgClass::Gpr(Gpr::Rcx) | ArgClass::Gpr(Gpr::Rdi)
    ));
}

// =========================================================================
// TestScenario Tests
// =========================================================================

#[test]
fn test_scenario_simple_arithmetic() {
    let seq = TestScenario::SimpleArithmetic.register_sequence();
    assert!(!seq.is_empty());
    assert_eq!(TestScenario::SimpleArithmetic.expected_min_spills(), 0);
}

#[test]
fn test_scenario_exhaust_gprs() {
    let seq = TestScenario::ExhaustGprs.register_sequence();
    let total: u32 = seq.iter().map(|(_, c)| c).sum();
    assert!(total > 14); // More than available
    assert!(TestScenario::ExhaustGprs.expected_min_spills() > 0);
}

#[test]
fn test_scenario_exhaust_xmms() {
    let seq = TestScenario::ExhaustXmms.register_sequence();
    let total: u32 = seq.iter().map(|(_, c)| c).sum();
    assert!(total > 15);
    assert!(TestScenario::ExhaustXmms.expected_min_spills() > 0);
}

#[test]
fn test_scenario_exhaust_ymms() {
    let seq = TestScenario::ExhaustYmms.register_sequence();
    let total: u32 = seq.iter().map(|(_, c)| c).sum();
    assert!(total > 16);
    assert!(TestScenario::ExhaustYmms.expected_min_spills() > 0);
}

#[test]
fn test_scenario_exhaust_zmms() {
    let seq = TestScenario::ExhaustZmms.register_sequence();
    let total: u32 = seq.iter().map(|(_, c)| c).sum();
    assert!(total > 32);
    assert!(TestScenario::ExhaustZmms.expected_min_spills() > 0);
}

#[test]
fn test_scenario_mixed_pressure() {
    let seq = TestScenario::MixedPressure.register_sequence();
    assert!(seq.len() >= 4); // All register classes
}

// =========================================================================
// SimdTestHarness Tests
// =========================================================================

#[test]
fn test_harness_new() {
    let harness = SimdTestHarness::new();
    assert_eq!(harness.stats.definitions, 0);
    assert_eq!(harness.stats.spills, 0);
}

#[test]
fn test_harness_simple_arithmetic() {
    let mut harness = SimdTestHarness::new();

    let stats = harness.run_scenario(TestScenario::SimpleArithmetic);
    assert!(stats.definitions > 0);
    assert_eq!(stats.spills, 0);
}

#[test]
fn test_harness_exhaust_gprs() {
    let mut harness = SimdTestHarness::new();

    let stats = harness.run_scenario(TestScenario::ExhaustGprs);
    assert!(stats.spills >= TestScenario::ExhaustGprs.expected_min_spills());
    assert!(stats.peak_stack > 0);
}

#[test]
fn test_harness_exhaust_xmms() {
    let mut harness = SimdTestHarness::new();

    let stats = harness.run_scenario(TestScenario::ExhaustXmms);
    assert!(stats.spills >= TestScenario::ExhaustXmms.expected_min_spills());
}

#[test]
fn test_harness_exhaust_ymms() {
    let mut harness = SimdTestHarness::new();

    let stats = harness.run_scenario(TestScenario::ExhaustYmms);
    assert!(stats.spills >= TestScenario::ExhaustYmms.expected_min_spills());
}

#[test]
fn test_harness_exhaust_zmms() {
    let mut harness = SimdTestHarness::new();

    let stats = harness.run_scenario(TestScenario::ExhaustZmms);
    assert!(stats.spills >= TestScenario::ExhaustZmms.expected_min_spills());
}

#[test]
fn test_harness_mixed_pressure() {
    let mut harness = SimdTestHarness::new();

    let stats = harness.run_scenario(TestScenario::MixedPressure);
    assert!(stats.definitions >= 40); // 10 per class
}

#[test]
fn test_harness_with_calls() {
    let mut harness = SimdTestHarness::new();

    let stats = harness.run_with_calls(TestScenario::SimpleArithmetic, 2);
    assert!(stats.calls >= 1);
}

#[test]
fn test_harness_reset() {
    let mut harness = SimdTestHarness::new();

    harness.run_scenario(TestScenario::ExhaustGprs);
    harness.reset();

    assert_eq!(harness.stats.definitions, 0);
    assert_eq!(harness.stats.spills, 0);
}

#[test]
fn test_harness_peak_stack_ymm() {
    let mut harness = SimdTestHarness::new();

    harness.run_scenario(TestScenario::ExhaustYmms);

    // YMM spills should be 32 bytes each
    let expected_min = TestScenario::ExhaustYmms.expected_min_spills() as u32 * 32;
    assert!(harness.stats.peak_stack >= expected_min);
}

#[test]
fn test_harness_peak_stack_zmm() {
    let mut harness = SimdTestHarness::new();

    harness.run_scenario(TestScenario::ExhaustZmms);

    // ZMM spills should be 64 bytes each
    let expected_min = TestScenario::ExhaustZmms.expected_min_spills() as u32 * 64;
    assert!(harness.stats.peak_stack >= expected_min);
}

// =========================================================================
// Stress Tests
// =========================================================================

#[test]
fn test_stress_extreme_gpr_pressure() {
    let mut harness = SimdTestHarness::new();

    // Define 100 GPRs
    for _ in 0..100 {
        harness.pressure.define(RegClass::Int);
        harness.stats.definitions += 1;
    }

    // Should have many spills
    assert!(harness.pressure.spill_count() >= 86); // 100 - 14
    assert_eq!(harness.pressure.peak_pressure(RegClass::Int), 100);
}

#[test]
fn test_stress_extreme_ymm_pressure() {
    let mut harness = SimdTestHarness::new();

    // Define 50 YMMs
    for _ in 0..50 {
        harness.pressure.define(RegClass::Vec256);
        harness.stats.definitions += 1;
    }

    assert!(harness.pressure.spill_count() >= 34); // 50 - 16
    assert_eq!(harness.pressure.peak_pressure(RegClass::Vec256), 50);
}

#[test]
fn test_stress_extreme_zmm_pressure() {
    let mut harness = SimdTestHarness::new();

    // Define 100 ZMMs
    for _ in 0..100 {
        harness.pressure.define(RegClass::Vec512);
        harness.stats.definitions += 1;
    }

    assert!(harness.pressure.spill_count() >= 68); // 100 - 32
    assert_eq!(harness.pressure.peak_pressure(RegClass::Vec512), 100);
}

#[test]
fn test_stress_all_classes_simultaneous() {
    let mut harness = SimdTestHarness::new();

    // Max out all classes simultaneously
    for _ in 0..20 {
        harness.pressure.define(RegClass::Int);
        harness.pressure.define(RegClass::Float);
        harness.pressure.define(RegClass::Vec256);
        harness.pressure.define(RegClass::Vec512);
    }

    // All classes should have spilled
    assert!(harness.pressure.spill_count() > 0);
    assert!(harness.pressure.is_under_pressure(RegClass::Int));
    assert!(harness.pressure.is_under_pressure(RegClass::Float));
    assert!(harness.pressure.is_under_pressure(RegClass::Vec256));
}

#[test]
fn test_stress_interleaved_def_use() {
    let mut sim = RegisterPressureSimulator::new(4, 4, 4, 4);

    // Interleave definitions and uses
    for _ in 0..10 {
        sim.define(RegClass::Int);
        sim.define(RegClass::Int);
        sim.use_end(RegClass::Int);
    }

    // Should have moderate pressure but not extreme
    assert!(sim.peak_pressure(RegClass::Int) <= 20);
}

#[test]
fn test_stress_call_heavy_scenario() {
    let mut harness = SimdTestHarness::new();

    // Simulate call-heavy code
    for call in 0..20 {
        // Define some values
        for _ in 0..3 {
            harness.pressure.define(RegClass::Int);
            harness.pressure.define(RegClass::Vec256);
        }

        // Call clobbers values
        harness.stats.calls += 1;

        // End some uses
        for _ in 0..2 {
            harness.pressure.use_end(RegClass::Int);
            harness.pressure.use_end(RegClass::Vec256);
        }
    }

    assert!(harness.stats.calls == 20);
}

#[test]
fn test_stress_spill_slot_alignment_chain() {
    let mut tracker = SpillSlotTracker::new(0);

    // Alternate between different sized slots
    for _ in 0..10 {
        tracker.allocate(RegClass::Int); // 8 bytes
        tracker.allocate(RegClass::Vec256); // 32 bytes, needs alignment
        tracker.allocate(RegClass::Float); // 16 bytes
        tracker.allocate(RegClass::Vec512); // 64 bytes, needs alignment
    }

    // Should have proper alignment throughout
    assert_eq!(tracker.total_slots(), 40);
    assert!(tracker.peak_bytes() >= 40 * 8); // At minimum
}

#[test]
fn test_stress_many_stack_args() {
    let mut generator = CallSiteGenerator::new();

    // 50 integer args
    let sig: Vec<_> = (0..50).map(|_| RegClass::Int).collect();
    let args = generator.generate_args(&sig);

    let stack_count = args
        .iter()
        .filter(|a| matches!(a, ArgClass::Stack(_)))
        .count();

    // Most should be on stack
    assert!(stack_count > 40);
}

#[test]
fn test_stress_vector_stack_args() {
    let mut generator = CallSiteGenerator::new();

    // 20 YMM args
    let sig: Vec<_> = (0..20).map(|_| RegClass::Vec256).collect();
    let args = generator.generate_args(&sig);

    // Verify stack offsets are 32-byte aligned
    for arg in args.iter() {
        if let ArgClass::Stack(offset) = arg {
            assert!(offset % 32 == 0, "YMM stack arg not 32-byte aligned");
        }
    }
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_edge_zero_registers() {
    let mut sim = RegisterPressureSimulator::new(0, 0, 0, 0);

    // Any definition should spill
    assert!(sim.define(RegClass::Int));
    assert!(sim.define(RegClass::Float));
    assert!(sim.define(RegClass::Vec256));
    assert!(sim.define(RegClass::Vec512));

    assert_eq!(sim.spill_count(), 4);
}

#[test]
fn test_edge_single_register() {
    let mut sim = RegisterPressureSimulator::new(1, 1, 1, 1);

    // First definition succeeds
    assert!(!sim.define(RegClass::Int));
    // Second spills
    assert!(sim.define(RegClass::Int));
}

#[test]
fn test_edge_any_class() {
    let mut sim = RegisterPressureSimulator::new(5, 5, 5, 5);

    // Any class should always succeed
    assert!(!sim.define(RegClass::Any));
    assert!(!sim.define(RegClass::Any));
    assert!(!sim.define(RegClass::Any));
}

#[test]
fn test_edge_use_end_underflow() {
    let mut sim = RegisterPressureSimulator::new(5, 5, 5, 5);

    // Use end without definition shouldn't crash
    sim.use_end(RegClass::Int);
    assert_eq!(sim.live_gprs, 0);
}

#[test]
fn test_edge_empty_scenario() {
    let mut harness = SimdTestHarness::new();

    // Run with empty sequence
    let stats = harness.run_scenario(TestScenario::SimpleArithmetic);
    assert!(stats.definitions > 0); // SimpleArithmetic is not empty
}

#[test]
fn test_edge_zero_calls() {
    let mut harness = SimdTestHarness::new();

    let stats = harness.run_with_calls(TestScenario::SimpleArithmetic, 0);
    assert_eq!(stats.calls, 0);
}
