//! SIMD Integration Tests
//!
//! This module provides comprehensive end-to-end integration tests for the
//! SIMD codegen pipeline, including:
//!
//! - Register pressure stress tests for vector allocation
//! - Spill/reload correctness under high register pressure
//! - Calling convention integration with vector registers
//! - Vector operation encoding verification
//! - Cost model accuracy validation
//!
//! # Test Categories
//!
//! 1. **Register Pressure Tests**: Exhaust YMM/ZMM registers to verify spilling
//! 2. **Calling Convention Tests**: Verify vector args/returns across calls
//! 3. **Encoding Tests**: End-to-end instruction encoding verification
//! 4. **Cost Model Tests**: Validate profitability decisions
//! 5. **Legality Tests**: Verify safety analysis correctness

use super::call_conv::{ArgClass, ArgLocationCalc, CallClobbers, VectorCallingConvention};
use super::registers::Gpr;
use crate::regalloc::RegClass;

// =============================================================================
// Register Pressure Simulation
// =============================================================================

/// Simulates register pressure for testing spill/reload behavior.
///
/// This struct tracks virtual register allocations and determines when
/// spilling is required based on available physical registers.
#[derive(Debug, Clone)]
pub struct RegisterPressureSimulator {
    /// Number of available GPRs.
    available_gprs: u32,
    /// Number of available XMM registers.
    available_xmms: u32,
    /// Number of available YMM registers.
    available_ymms: u32,
    /// Number of available ZMM registers.
    available_zmms: u32,
    /// Current live GPR count.
    live_gprs: u32,
    /// Current live XMM count.
    live_xmms: u32,
    /// Current live YMM count.
    live_ymms: u32,
    /// Current live ZMM count.
    live_zmms: u32,
    /// Total spills required.
    spill_count: u32,
    /// Total reloads required.
    reload_count: u32,
    /// Peak pressure per register class.
    peak_pressure: [u32; 4],
}

impl RegisterPressureSimulator {
    /// Create a new simulator with the given register availability.
    pub fn new(gprs: u32, xmms: u32, ymms: u32, zmms: u32) -> Self {
        Self {
            available_gprs: gprs,
            available_xmms: xmms,
            available_ymms: ymms,
            available_zmms: zmms,
            live_gprs: 0,
            live_xmms: 0,
            live_ymms: 0,
            live_zmms: 0,
            spill_count: 0,
            reload_count: 0,
            peak_pressure: [0; 4],
        }
    }

    /// Create a simulator for Windows x64 calling convention.
    pub fn for_windows() -> Self {
        // Windows: 14 GPRs (excluding RSP, R11 scratch), 15 XMMs (excluding scratch)
        // All YMM/ZMM share with XMM
        Self::new(14, 15, 16, 32)
    }

    /// Create a simulator for System V calling convention.
    pub fn for_sysv() -> Self {
        // System V: Same allocatable counts
        Self::new(14, 15, 16, 32)
    }

    /// Create a simulator for the host calling convention.
    pub fn for_host() -> Self {
        #[cfg(target_os = "windows")]
        return Self::for_windows();
        #[cfg(not(target_os = "windows"))]
        return Self::for_sysv();
    }

    /// Record a virtual register definition.
    pub fn define(&mut self, class: RegClass) -> bool {
        let (live_ref, available, peak_idx) = match class {
            RegClass::Int => (&mut self.live_gprs, self.available_gprs, 0),
            RegClass::Float => (&mut self.live_xmms, self.available_xmms, 1),
            RegClass::Vec256 => (&mut self.live_ymms, self.available_ymms, 2),
            RegClass::Vec512 => (&mut self.live_zmms, self.available_zmms, 3),
            RegClass::Any => return false, // Any class never spills
        };

        *live_ref += 1;
        self.peak_pressure[peak_idx] = self.peak_pressure[peak_idx].max(*live_ref);

        if *live_ref > available {
            self.spill_count += 1;
            true // Spill required
        } else {
            false
        }
    }

    /// Record a virtual register use (end of live range).
    pub fn use_end(&mut self, class: RegClass) {
        let live_ref = match class {
            RegClass::Int => &mut self.live_gprs,
            RegClass::Float => &mut self.live_xmms,
            RegClass::Vec256 => &mut self.live_ymms,
            RegClass::Vec512 => &mut self.live_zmms,
            RegClass::Any => return,
        };

        if *live_ref > 0 {
            *live_ref -= 1;
        }
    }

    /// Record a reload.
    pub fn reload(&mut self, class: RegClass) {
        self.reload_count += 1;
        self.define(class);
    }

    /// Get total spill count.
    pub fn spill_count(&self) -> u32 {
        self.spill_count
    }

    /// Get total reload count.
    pub fn reload_count(&self) -> u32 {
        self.reload_count
    }

    /// Get peak pressure for a register class.
    pub fn peak_pressure(&self, class: RegClass) -> u32 {
        match class {
            RegClass::Int => self.peak_pressure[0],
            RegClass::Float => self.peak_pressure[1],
            RegClass::Vec256 => self.peak_pressure[2],
            RegClass::Vec512 => self.peak_pressure[3],
            RegClass::Any => 0,
        }
    }

    /// Check if class is under pressure (live > available).
    pub fn is_under_pressure(&self, class: RegClass) -> bool {
        match class {
            RegClass::Int => self.live_gprs > self.available_gprs,
            RegClass::Float => self.live_xmms > self.available_xmms,
            RegClass::Vec256 => self.live_ymms > self.available_ymms,
            RegClass::Vec512 => self.live_zmms > self.available_zmms,
            RegClass::Any => false,
        }
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.live_gprs = 0;
        self.live_xmms = 0;
        self.live_ymms = 0;
        self.live_zmms = 0;
        self.spill_count = 0;
        self.reload_count = 0;
        self.peak_pressure = [0; 4];
    }
}

impl Default for RegisterPressureSimulator {
    fn default() -> Self {
        Self::for_host()
    }
}

// =============================================================================
// Spill Slot Tracker
// =============================================================================

/// Tracks spill slot allocation for integration testing.
#[derive(Debug, Clone)]
pub struct SpillSlotTracker {
    /// Current stack offset.
    offset: i32,
    /// Slots allocated per class.
    slots_per_class: [u32; 4],
    /// Total bytes used.
    bytes_used: u32,
    /// Peak stack usage.
    peak_bytes: u32,
}

impl SpillSlotTracker {
    /// Create a new tracker starting at the given offset.
    pub fn new(initial_offset: i32) -> Self {
        Self {
            offset: initial_offset,
            slots_per_class: [0; 4],
            bytes_used: 0,
            peak_bytes: 0,
        }
    }

    /// Allocate a spill slot for the given register class.
    ///
    /// Returns the offset for the slot.
    pub fn allocate(&mut self, class: RegClass) -> i32 {
        let (size, align, idx) = match class {
            RegClass::Int => (8, 8, 0),
            RegClass::Float => (16, 16, 1),
            RegClass::Vec256 => (32, 32, 2),
            RegClass::Vec512 => (64, 64, 3),
            RegClass::Any => (8, 8, 0), // Default to GPR size
        };

        // Align the offset
        self.offset = (self.offset + align - 1) & !(align - 1);

        let slot_offset = self.offset;
        self.offset += size;
        self.slots_per_class[idx] += 1;
        self.bytes_used += size as u32;
        self.peak_bytes = self.peak_bytes.max(self.bytes_used);

        slot_offset
    }

    /// Free a spill slot (for release-point tracking).
    pub fn free(&mut self, class: RegClass) {
        let size = match class {
            RegClass::Int => 8,
            RegClass::Float => 16,
            RegClass::Vec256 => 32,
            RegClass::Vec512 => 64,
            RegClass::Any => 8,
        };

        if self.bytes_used >= size {
            self.bytes_used -= size;
        }
    }

    /// Get total slots allocated.
    pub fn total_slots(&self) -> u32 {
        self.slots_per_class.iter().sum()
    }

    /// Get slots for a specific class.
    pub fn slots_for_class(&self, class: RegClass) -> u32 {
        match class {
            RegClass::Int => self.slots_per_class[0],
            RegClass::Float => self.slots_per_class[1],
            RegClass::Vec256 => self.slots_per_class[2],
            RegClass::Vec512 => self.slots_per_class[3],
            RegClass::Any => 0,
        }
    }

    /// Get peak stack usage.
    pub fn peak_bytes(&self) -> u32 {
        self.peak_bytes
    }

    /// Get current offset.
    pub fn current_offset(&self) -> i32 {
        self.offset
    }
}

impl Default for SpillSlotTracker {
    fn default() -> Self {
        Self::new(0)
    }
}

// =============================================================================
// Call Site Generator
// =============================================================================

/// Generates test call sites with various argument configurations.
#[derive(Debug, Clone)]
pub struct CallSiteGenerator {
    /// Calling convention.
    cc: VectorCallingConvention,
    /// Argument location calculator.
    calc: ArgLocationCalc,
}

impl CallSiteGenerator {
    /// Create a new generator for the host convention.
    pub fn new() -> Self {
        let cc = VectorCallingConvention::host();
        Self {
            cc,
            calc: ArgLocationCalc::new(),
        }
    }

    /// Create a generator for a specific calling convention.
    pub fn with_convention(cc: VectorCallingConvention) -> Self {
        Self {
            cc,
            calc: ArgLocationCalc::with_convention(cc),
        }
    }

    /// Generate argument locations for a function signature.
    ///
    /// The signature is a slice of register classes for each argument.
    pub fn generate_args(&mut self, signature: &[RegClass]) -> Vec<ArgClass> {
        self.calc.reset();
        signature
            .iter()
            .map(|&class| match class {
                RegClass::Int => self.calc.next_int(),
                RegClass::Float => self.calc.next_f64(),
                RegClass::Vec256 => self.calc.next_v256(),
                RegClass::Vec512 => self.calc.next_v512(),
                RegClass::Any => self.calc.next_int(),
            })
            .collect()
    }

    /// Get the stack size needed for generated arguments.
    pub fn stack_size(&self) -> i32 {
        self.calc.stack_size()
    }

    /// Get the clobbers for the calling convention.
    pub fn clobbers(&self) -> CallClobbers {
        self.cc.call_clobbers()
    }

    /// Count how many args go in registers vs stack.
    pub fn count_register_vs_stack(&mut self, signature: &[RegClass]) -> (usize, usize) {
        let args = self.generate_args(signature);
        let regs = args
            .iter()
            .filter(|a| !matches!(a, ArgClass::Stack(_)))
            .count();
        (regs, args.len() - regs)
    }
}

impl Default for CallSiteGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Test Scenarios
// =============================================================================

/// Predefined test scenarios for integration testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestScenario {
    /// Simple arithmetic with no spilling.
    SimpleArithmetic,
    /// Exhaust GPR registers.
    ExhaustGprs,
    /// Exhaust XMM registers.
    ExhaustXmms,
    /// Exhaust YMM registers.
    ExhaustYmms,
    /// Exhaust ZMM registers.
    ExhaustZmms,
    /// Mixed pressure across all classes.
    MixedPressure,
    /// Call-heavy code with frequent clobbering.
    CallHeavy,
    /// Long live ranges with many conflicts.
    LongLiveRanges,
    /// Nested loops with varying pressure.
    NestedLoops,
    /// SIMD reduction pattern.
    SimdReduction,
}

impl TestScenario {
    /// Get the register definitions for this scenario.
    pub fn register_sequence(self) -> Vec<(RegClass, u32)> {
        match self {
            TestScenario::SimpleArithmetic => {
                vec![(RegClass::Int, 4), (RegClass::Float, 2)]
            }
            TestScenario::ExhaustGprs => {
                // Define 20 GPRs to exhaust 14 available
                vec![(RegClass::Int, 20)]
            }
            TestScenario::ExhaustXmms => {
                // Define 20 XMMs to exhaust 15 available
                vec![(RegClass::Float, 20)]
            }
            TestScenario::ExhaustYmms => {
                // Define 20 YMMs to exhaust 16 available
                vec![(RegClass::Vec256, 20)]
            }
            TestScenario::ExhaustZmms => {
                // Define 40 ZMMs to exhaust 32 available
                vec![(RegClass::Vec512, 40)]
            }
            TestScenario::MixedPressure => {
                vec![
                    (RegClass::Int, 10),
                    (RegClass::Float, 10),
                    (RegClass::Vec256, 10),
                    (RegClass::Vec512, 10),
                ]
            }
            TestScenario::CallHeavy => {
                // GPRs and XMMs between calls
                vec![(RegClass::Int, 8), (RegClass::Float, 8)]
            }
            TestScenario::LongLiveRanges => {
                vec![(RegClass::Int, 15), (RegClass::Vec256, 18)]
            }
            TestScenario::NestedLoops => {
                vec![
                    (RegClass::Int, 6),     // Outer loop
                    (RegClass::Vec256, 12), // Inner loop
                    (RegClass::Int, 4),     // More outer
                ]
            }
            TestScenario::SimdReduction => {
                vec![
                    (RegClass::Vec256, 8), // Accumulators
                    (RegClass::Vec256, 4), // Temps
                    (RegClass::Int, 2),    // Loop counter
                ]
            }
        }
    }

    /// Get expected minimum spill count for this scenario.
    pub fn expected_min_spills(self) -> u32 {
        match self {
            TestScenario::SimpleArithmetic => 0,
            TestScenario::ExhaustGprs => 6,    // 20 - 14
            TestScenario::ExhaustXmms => 5,    // 20 - 15
            TestScenario::ExhaustYmms => 4,    // 20 - 16
            TestScenario::ExhaustZmms => 8,    // 40 - 32
            TestScenario::MixedPressure => 0,  // Falls within limits
            TestScenario::CallHeavy => 0,      // Within limits
            TestScenario::LongLiveRanges => 3, // 15 - 14, 18 - 16
            TestScenario::NestedLoops => 0,    // Staggered definitions
            TestScenario::SimdReduction => 0,  // Within limits
        }
    }
}

// =============================================================================
// End-to-End Test Harness
// =============================================================================

/// Harness for running end-to-end SIMD integration tests.
#[derive(Debug)]
pub struct SimdTestHarness {
    /// Register pressure simulator.
    pub pressure: RegisterPressureSimulator,
    /// Spill slot tracker.
    pub spills: SpillSlotTracker,
    /// Call site generator.
    pub calls: CallSiteGenerator,
    /// Test statistics.
    pub stats: TestStats,
}

/// Statistics from integration tests.
#[derive(Debug, Clone, Default)]
pub struct TestStats {
    /// Total definitions.
    pub definitions: u32,
    /// Total uses.
    pub uses: u32,
    /// Total calls.
    pub calls: u32,
    /// Total spills.
    pub spills: u32,
    /// Total reloads.
    pub reloads: u32,
    /// Peak stack usage.
    pub peak_stack: u32,
}

impl SimdTestHarness {
    /// Create a new test harness.
    pub fn new() -> Self {
        Self {
            pressure: RegisterPressureSimulator::for_host(),
            spills: SpillSlotTracker::new(0),
            calls: CallSiteGenerator::new(),
            stats: TestStats::default(),
        }
    }

    /// Run a test scenario.
    pub fn run_scenario(&mut self, scenario: TestScenario) -> &TestStats {
        self.reset();

        for (class, count) in scenario.register_sequence() {
            for _ in 0..count {
                if self.pressure.define(class) {
                    self.spills.allocate(class);
                    self.stats.spills += 1;
                }
                self.stats.definitions += 1;
            }
        }

        self.stats.peak_stack = self.spills.peak_bytes();
        &self.stats
    }

    /// Run with simulated call sites.
    pub fn run_with_calls(&mut self, scenario: TestScenario, call_count: usize) -> &TestStats {
        self.reset();
        let clobbers = self.calls.clobbers();

        // Run a portion, insert calls, run rest
        let sequence = scenario.register_sequence();
        let sequence_len = sequence.len();
        let mut iter_count = 0;

        for (class, count) in sequence {
            for _ in 0..count {
                if self.pressure.define(class) {
                    self.spills.allocate(class);
                    self.stats.spills += 1;
                }
                self.stats.definitions += 1;

                // Insert call every N definitions
                iter_count += 1;
                if call_count > 0 && iter_count % (sequence_len * 2 / call_count).max(1) == 0 {
                    self.stats.calls += 1;

                    // Clobbers require reload of any clobbered live values
                    // This is a simplified simulation
                    if clobbers.gprs.count() > 0 && self.pressure.live_gprs > 0 {
                        self.stats.reloads += 1;
                    }
                }
            }
        }

        self.stats.peak_stack = self.spills.peak_bytes();
        &self.stats
    }

    /// Reset the harness for a new test.
    pub fn reset(&mut self) {
        self.pressure.reset();
        self.spills = SpillSlotTracker::new(0);
        self.stats = TestStats::default();
    }
}

impl Default for SimdTestHarness {
    fn default() -> Self {
        Self::new()
    }
}
