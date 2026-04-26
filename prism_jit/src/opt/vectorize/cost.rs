//! Vector Cost Model for Profitability Analysis
//!
//! This module provides target-aware cost estimation for vectorization decisions.
//! It models the execution cost of both scalar and vector operations to determine
//! whether vectorization is profitable.
//!
//! # Cost Factors
//!
//! The cost model considers:
//!
//! - **Operation Latency**: Cycles from input availability to output
//! - **Reciprocal Throughput**: Average cycles per operation (accounting for pipelining)
//! - **Memory Alignment**: Penalties for unaligned vector loads/stores
//! - **Shuffle Overhead**: Cost of lane permutations and cross-lane operations
//! - **Trip Count**: Whether loop overhead is amortized sufficiently
//!
//! # SIMD Levels
//!
//! Different x86-64 SIMD extensions have different capabilities and costs:
//!
//! - **SSE4.2**: 128-bit vectors, 2 lanes for 64-bit types
//! - **AVX/AVX2**: 256-bit vectors, 4 lanes for 64-bit types
//! - **AVX-512**: 512-bit vectors, 8 lanes for 64-bit types, masking support
//!
//! # Usage
//!
//! ```ignore
//! let model = VectorCostModel::new(SimdLevel::Avx2);
//! let analysis = model.analyze(graph, loop_info, 4, Some(1000));
//! if analysis.profitable {
//!     vectorize(graph, loop_info, 4);
//! }
//! ```

use crate::ir::operators::{ArithOp, CmpOp, VectorArithKind, VectorMemoryKind, VectorOp};
use crate::ir::types::ValueType;
use rustc_hash::FxHashMap;

// =============================================================================
// SIMD Level
// =============================================================================

/// Target SIMD capability level.
///
/// Represents the available SIMD instruction set on the target CPU.
/// Higher levels include all capabilities of lower levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdLevel {
    /// SSE4.2 (128-bit vectors)
    ///
    /// - 2×i64, 2×f64, 4×i32, 4×f32
    /// - No 256-bit operations
    Sse42,

    /// AVX (256-bit floating-point)
    ///
    /// - 4×f64, 8×f32 with 256-bit
    /// - Integer still 128-bit only
    Avx,

    /// AVX2 (full 256-bit support)
    ///
    /// - 4×i64, 4×f64, 8×i32, 8×f32
    /// - FMA instruction support
    /// - Gather instructions
    Avx2,

    /// AVX-512 (512-bit vectors)
    ///
    /// - 8×i64, 8×f64, 16×i32, 16×f32
    /// - Masking and predication
    /// - Scatter instructions
    Avx512,
}

impl SimdLevel {
    /// Get maximum vector width in bytes.
    pub const fn max_vector_bytes(self) -> usize {
        match self {
            SimdLevel::Sse42 => 16,
            SimdLevel::Avx | SimdLevel::Avx2 => 32,
            SimdLevel::Avx512 => 64,
        }
    }

    /// Get maximum vector width in bits.
    pub const fn max_vector_bits(self) -> usize {
        self.max_vector_bytes() * 8
    }

    /// Get maximum lanes for a given element type.
    pub const fn max_lanes(self, element: ValueType) -> usize {
        let elem_bits = match element {
            ValueType::Int64 | ValueType::Float64 => 64,
            // 32-bit types would use 4 bytes
            _ => 64, // Default to 64-bit for Python values
        };
        self.max_vector_bits() / elem_bits
    }

    /// Check if this level supports FMA (fused multiply-add).
    pub const fn has_fma(self) -> bool {
        matches!(self, SimdLevel::Avx2 | SimdLevel::Avx512)
    }

    /// Check if this level supports gather instructions.
    pub const fn has_gather(self) -> bool {
        matches!(self, SimdLevel::Avx2 | SimdLevel::Avx512)
    }

    /// Check if this level supports scatter instructions.
    pub const fn has_scatter(self) -> bool {
        matches!(self, SimdLevel::Avx512)
    }

    /// Check if this level supports masked operations.
    pub const fn has_masking(self) -> bool {
        matches!(self, SimdLevel::Avx512)
    }

    /// Get the VectorOp for this SIMD level with given element type.
    pub fn best_vector_op(self, element: ValueType) -> VectorOp {
        let lanes = self.max_lanes(element);
        VectorOp {
            element,
            lanes: lanes as u8,
        }
    }
}

impl Default for SimdLevel {
    fn default() -> Self {
        SimdLevel::Avx2
    }
}

// =============================================================================
// Operation Cost
// =============================================================================

/// Cost of a single operation.
///
/// Models both latency and throughput characteristics of an operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpCost {
    /// Latency in CPU cycles (input to output).
    ///
    /// Higher latency means the result takes longer to be available.
    pub latency: u8,

    /// Reciprocal throughput (cycles per operation).
    ///
    /// Lower throughput means more operations can execute per cycle.
    /// A value of 0.5 means 2 operations per cycle.
    pub throughput: f32,
}

impl OpCost {
    /// Create a new operation cost.
    pub const fn new(latency: u8, throughput: f32) -> Self {
        Self {
            latency,
            throughput,
        }
    }

    /// Create a free operation (for things that compile away).
    pub const fn free() -> Self {
        Self::new(0, 0.0)
    }

    /// Create a very cheap operation.
    pub const fn trivial() -> Self {
        Self::new(1, 0.33)
    }

    /// Create a typical ALU operation cost.
    pub const fn alu() -> Self {
        Self::new(1, 0.5)
    }

    /// Create a multiplication cost.
    pub const fn mul() -> Self {
        Self::new(3, 1.0)
    }

    /// Create a division cost (expensive).
    pub const fn div() -> Self {
        Self::new(14, 6.0)
    }

    /// Create a memory load cost.
    pub const fn load() -> Self {
        Self::new(5, 0.5)
    }

    /// Create a memory store cost.
    pub const fn store() -> Self {
        Self::new(4, 1.0)
    }

    /// Create a shuffle cost.
    pub const fn shuffle() -> Self {
        Self::new(1, 1.0)
    }

    /// Create a cross-lane operation cost (expensive on AVX).
    pub const fn cross_lane() -> Self {
        Self::new(3, 1.0)
    }

    /// Scale cost by number of operations.
    pub fn scale(self, factor: f32) -> Self {
        Self {
            latency: self.latency,
            throughput: self.throughput * factor,
        }
    }

    /// Add overhead/penalty to cost.
    pub fn with_penalty(self, throughput_penalty: f32) -> Self {
        Self {
            latency: self.latency,
            throughput: self.throughput + throughput_penalty,
        }
    }

    /// Combine costs (for dependent operations).
    pub fn chain(self, other: Self) -> Self {
        Self {
            latency: self.latency.saturating_add(other.latency),
            throughput: self.throughput + other.throughput,
        }
    }

    /// Convert to scalar cost for comparison.
    pub fn total_cost(&self) -> f32 {
        // Weight throughput more than latency for loop-intensive code
        self.throughput + self.latency as f32 * 0.2
    }
}

impl Default for OpCost {
    fn default() -> Self {
        Self::alu()
    }
}

// =============================================================================
// Vector Cost Model
// =============================================================================

/// Cost model for vectorization profitability analysis.
///
/// Provides accurate cost estimates for scalar and vector operations
/// to determine whether vectorization improves performance.
pub struct VectorCostModel {
    /// Target SIMD level.
    level: SimdLevel,

    /// Cached operation costs.
    arith_costs: FxHashMap<(VectorArithKind, usize), OpCost>,

    /// Memory alignment penalty factor.
    alignment_penalty: f32,

    /// Minimum profitability threshold.
    min_speedup: f32,
}

impl VectorCostModel {
    /// Create a cost model for a specific SIMD level.
    pub fn new(level: SimdLevel) -> Self {
        let mut model = Self {
            level,
            arith_costs: FxHashMap::default(),
            alignment_penalty: match level {
                SimdLevel::Sse42 => 0.5,
                SimdLevel::Avx | SimdLevel::Avx2 => 0.3,
                SimdLevel::Avx512 => 0.1,
            },
            min_speedup: 1.25, // Require 25% speedup
        };
        model.initialize_costs();
        model
    }

    /// Initialize the cost tables.
    fn initialize_costs(&mut self) {
        // Initialize for common widths: 2, 4, 8 lanes
        for &lanes in &[2usize, 4, 8] {
            // Arithmetic operations
            self.arith_costs
                .insert((VectorArithKind::Add, lanes), OpCost::alu());
            self.arith_costs
                .insert((VectorArithKind::Sub, lanes), OpCost::alu());
            self.arith_costs
                .insert((VectorArithKind::Mul, lanes), OpCost::mul());
            self.arith_costs.insert(
                (VectorArithKind::Div, lanes),
                if lanes <= 4 {
                    OpCost::div()
                } else {
                    OpCost::div().scale(1.5)
                },
            );
            self.arith_costs
                .insert((VectorArithKind::Min, lanes), OpCost::alu());
            self.arith_costs
                .insert((VectorArithKind::Max, lanes), OpCost::alu());
            self.arith_costs
                .insert((VectorArithKind::Abs, lanes), OpCost::trivial());
            self.arith_costs
                .insert((VectorArithKind::Neg, lanes), OpCost::trivial());
            self.arith_costs
                .insert((VectorArithKind::Sqrt, lanes), OpCost::new(12, 4.0));
        }
    }

    /// Get the target SIMD level.
    pub fn level(&self) -> SimdLevel {
        self.level
    }

    /// Get cost of a vector arithmetic operation.
    pub fn arith_cost(&self, kind: VectorArithKind, vop: VectorOp) -> OpCost {
        let lanes = vop.lanes as usize;

        // Check cache
        if let Some(&cost) = self.arith_costs.get(&(kind, lanes)) {
            return cost;
        }

        // Default costs by operation type
        match kind {
            VectorArithKind::Add | VectorArithKind::Sub => OpCost::alu(),
            VectorArithKind::Mul => OpCost::mul(),
            VectorArithKind::Div => OpCost::div(),
            VectorArithKind::Min | VectorArithKind::Max => OpCost::alu(),
            VectorArithKind::Abs | VectorArithKind::Neg => OpCost::trivial(),
            VectorArithKind::Sqrt => OpCost::new(12, 4.0),
        }
    }

    /// Get cost of vector FMA operation.
    pub fn fma_cost(&self, vop: VectorOp) -> OpCost {
        if self.level.has_fma() {
            // Native FMA instruction
            OpCost::new(4, 0.5)
        } else {
            // Emulated: mul + add
            let lanes = vop.lanes as usize;
            self.arith_cost(VectorArithKind::Mul, vop)
                .chain(self.arith_cost(
                    VectorArithKind::Add,
                    VectorOp {
                        lanes: lanes as u8,
                        element: vop.element,
                    },
                ))
        }
    }

    /// Get cost of vector memory operation.
    pub fn memory_cost(&self, kind: VectorMemoryKind, _vop: VectorOp, aligned: bool) -> OpCost {
        let base = match kind {
            VectorMemoryKind::LoadAligned => OpCost::load(),
            VectorMemoryKind::LoadUnaligned => {
                if aligned {
                    OpCost::load()
                } else {
                    OpCost::load().with_penalty(self.alignment_penalty)
                }
            }
            VectorMemoryKind::StoreAligned => OpCost::store(),
            VectorMemoryKind::StoreUnaligned => {
                if aligned {
                    OpCost::store()
                } else {
                    OpCost::store().with_penalty(self.alignment_penalty)
                }
            }
            VectorMemoryKind::Gather => {
                if self.level.has_gather() {
                    OpCost::new(8, 4.0) // Native gather
                } else {
                    OpCost::new(20, 10.0) // Emulated
                }
            }
            VectorMemoryKind::Scatter => {
                if self.level.has_scatter() {
                    OpCost::new(8, 4.0) // Native scatter
                } else {
                    OpCost::new(25, 12.0) // Emulated
                }
            }
        };

        base
    }

    /// Get cost of vector comparison.
    pub fn cmp_cost(&self, _cmp: CmpOp, _vop: VectorOp) -> OpCost {
        OpCost::alu()
    }

    /// Get cost of broadcast operation.
    pub fn broadcast_cost(&self, _vop: VectorOp) -> OpCost {
        OpCost::trivial()
    }

    /// Get cost of extract operation.
    pub fn extract_cost(&self, vop: VectorOp, lane: u8) -> OpCost {
        if lane == 0 {
            OpCost::trivial() // Extract from low lane is free
        } else if lane < vop.lanes / 2 {
            OpCost::shuffle() // Same 128-bit lane
        } else {
            OpCost::cross_lane() // Cross-lane extraction
        }
    }

    /// Get cost of insert operation.
    pub fn insert_cost(&self, _vop: VectorOp, _lane: u8) -> OpCost {
        OpCost::shuffle()
    }

    /// Get cost of shuffle operation.
    pub fn shuffle_cost(&self, vop: VectorOp, crosses_lanes: bool) -> OpCost {
        if crosses_lanes && vop.lanes > 2 {
            OpCost::cross_lane()
        } else {
            OpCost::shuffle()
        }
    }

    /// Get cost of horizontal add (reduction).
    pub fn hadd_cost(&self, vop: VectorOp) -> OpCost {
        // Horizontal operations require log2(lanes) shuffle+add pairs
        let steps = (vop.lanes as f32).log2().ceil() as u8;
        OpCost::new(
            steps * 2, // 2 cycles per step (shuffle + add)
            steps as f32 * 1.0,
        )
    }

    /// Get cost of blend operation.
    pub fn blend_cost(&self, _vop: VectorOp) -> OpCost {
        if self.level.has_masking() {
            OpCost::trivial() // AVX-512 blend is cheap with masks
        } else {
            OpCost::shuffle()
        }
    }

    /// Get cost of scalar operation (for comparison).
    pub fn scalar_arith_cost(&self, op: ArithOp) -> OpCost {
        match op {
            ArithOp::Add | ArithOp::Sub => OpCost::alu(),
            ArithOp::Mul => OpCost::mul(),
            ArithOp::TrueDiv => OpCost::div(),
            ArithOp::FloorDiv | ArithOp::Mod => OpCost::div(),
            ArithOp::Pow => OpCost::new(20, 10.0), // Very expensive
            ArithOp::LShift | ArithOp::RShift => OpCost::trivial(),
            ArithOp::Neg => OpCost::trivial(),
        }
    }

    /// Get cost of scalar memory operation.
    pub fn scalar_memory_cost(&self) -> OpCost {
        OpCost::new(4, 1.0)
    }

    /// Estimate vector cost for a loop iteration.
    ///
    /// This is the main API for determining profitability.
    pub fn estimate_vector_cost(
        &self,
        ops: &[(VectorArithKind, VectorOp)],
        loads: usize,
        stores: usize,
        aligned: bool,
    ) -> f32 {
        let mut total = 0.0f32;

        // Arithmetic costs
        for &(kind, vop) in ops {
            total += self.arith_cost(kind, vop).total_cost();
        }

        // Memory costs
        let load_cost = self.memory_cost(
            if aligned {
                VectorMemoryKind::LoadAligned
            } else {
                VectorMemoryKind::LoadUnaligned
            },
            VectorOp::V4I64,
            aligned,
        );
        let store_cost = self.memory_cost(
            if aligned {
                VectorMemoryKind::StoreAligned
            } else {
                VectorMemoryKind::StoreUnaligned
            },
            VectorOp::V4I64,
            aligned,
        );

        total += loads as f32 * load_cost.total_cost();
        total += stores as f32 * store_cost.total_cost();

        total
    }

    /// Estimate scalar cost for a loop iteration.
    pub fn estimate_scalar_cost(&self, ops: &[ArithOp], loads: usize, stores: usize) -> f32 {
        let mut total = 0.0f32;

        for &op in ops {
            total += self.scalar_arith_cost(op).total_cost();
        }

        total += loads as f32 * self.scalar_memory_cost().total_cost();
        total += stores as f32 * self.scalar_memory_cost().total_cost();

        total
    }

    /// Determine if vectorization is profitable.
    pub fn is_profitable(
        &self,
        scalar_cost: f32,
        vector_cost: f32,
        vector_width: usize,
        trip_count: Option<u64>,
    ) -> bool {
        // Vector cost per scalar iteration
        let vector_cost_per_scalar = vector_cost / vector_width as f32;

        // Speedup ratio
        let speedup = scalar_cost / vector_cost_per_scalar;

        // Check basic speedup threshold
        if speedup < self.min_speedup {
            return false;
        }

        // Check trip count is sufficient to amortize overhead
        if let Some(tc) = trip_count {
            // Need at least one full vector iteration
            if tc < vector_width as u64 {
                return false;
            }

            // Small trip counts need higher speedup to justify overhead
            if tc < vector_width as u64 * 4 {
                return speedup > 1.5;
            }
        }

        true
    }

    /// Determine optimal vector width for given element type.
    pub fn best_vector_width(&self, element: ValueType) -> usize {
        self.level.max_lanes(element)
    }
}

impl Default for VectorCostModel {
    fn default() -> Self {
        Self::new(SimdLevel::default())
    }
}

impl std::fmt::Debug for VectorCostModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorCostModel")
            .field("level", &self.level)
            .field("alignment_penalty", &self.alignment_penalty)
            .field("min_speedup", &self.min_speedup)
            .finish()
    }
}

// =============================================================================
// Cost Analysis Result
// =============================================================================

/// Result of cost analysis for a vectorizable region.
#[derive(Debug, Clone)]
pub struct CostAnalysis {
    /// Estimated scalar execution cost per iteration.
    pub scalar_cost: f32,

    /// Estimated vector execution cost per vector iteration.
    pub vector_cost: f32,

    /// Vector width used in analysis.
    pub vector_width: usize,

    /// Estimated speedup factor.
    pub speedup: f32,

    /// Whether vectorization is profitable.
    pub profitable: bool,

    /// Trip count used (if known).
    pub trip_count: Option<u64>,
}

impl CostAnalysis {
    /// Create a new cost analysis result.
    pub fn new(
        scalar_cost: f32,
        vector_cost: f32,
        vector_width: usize,
        trip_count: Option<u64>,
    ) -> Self {
        let vector_cost_per_iter = vector_cost / vector_width as f32;
        let speedup = if vector_cost_per_iter > 0.0 {
            scalar_cost / vector_cost_per_iter
        } else {
            f32::INFINITY
        };

        Self {
            scalar_cost,
            vector_cost,
            vector_width,
            speedup,
            profitable: speedup > 1.25,
            trip_count,
        }
    }

    /// Get the cost per scalar iteration when vectorized.
    pub fn vector_cost_per_iter(&self) -> f32 {
        self.vector_cost / self.vector_width as f32
    }

    /// Get the cost savings per scalar iteration.
    pub fn savings_per_iter(&self) -> f32 {
        self.scalar_cost - self.vector_cost_per_iter()
    }

    /// Get total savings for given trip count.
    pub fn total_savings(&self, trip_count: u64) -> f32 {
        self.savings_per_iter() * trip_count as f32
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
