//! Loop Trip Count Analysis.
//!
//! This module computes the number of iterations a loop will execute,
//! which is essential for bounds widening optimization. Knowing the trip
//! count allows us to compute the maximum value an induction variable
//! will reach and replace per-iteration bounds checks with a single
//! check in the loop preheader.
//!
//! # Trip Count Types
//!
//! - **Constant**: Exactly N iterations (e.g., `for i in range(0, 100)`)
//! - **Symbolic**: N iterations where N is a runtime value (e.g., `for i in range(0, len)`)
//! - **Unknown**: Cannot determine iteration count
//!
//! # Algorithm
//!
//! For a loop with induction variable `i`:
//! 1. Find the loop exit condition (comparison with bound)
//! 2. Extract the bound value (constant or symbolic)
//! 3. Compute: trip_count = (end - start) / step
//!
//! # Safety Considerations
//!
//! Trip count analysis must be conservative:
//! - If step could be zero, trip count is unknown
//! - If direction doesn't match comparison, trip count is unknown
//! - Integer overflow must be considered

use super::induction::{InductionDirection, InductionInit, InductionStep, InductionVariable};
use crate::ir::cfg::Loop;
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{CmpOp, ControlOp, Operator};

// =============================================================================
// Trip Count Types
// =============================================================================

/// The computed trip count for a loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TripCount {
    /// Loop executes exactly N times.
    Constant(u64),

    /// Loop executes at most N times (upper bound only known).
    AtMost(u64),

    /// Loop trip count depends on runtime value.
    Symbolic(SymbolicTripCount),

    /// Loop may execute zero times (has early exit).
    MaybeZero(Box<TripCount>),

    /// Cannot determine trip count.
    Unknown,
}

/// Symbolic trip count expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicTripCount {
    /// The node representing the upper bound.
    pub bound_node: NodeId,

    /// Offset from bound (e.g., for `< n` this is 0, for `<= n` this is 1).
    pub offset: i64,

    /// The initial value (constant or node).
    pub start: TripCountValue,

    /// The step value.
    pub step: u64,

    /// Whether the count is exact or an upper bound.
    pub exact: bool,
}

/// A value in trip count computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TripCountValue {
    /// Constant value.
    Constant(i64),

    /// Value from a node.
    Node(NodeId),
}

impl TripCount {
    /// Create a constant trip count.
    #[inline]
    pub const fn constant(n: u64) -> Self {
        TripCount::Constant(n)
    }

    /// Create an unknown trip count.
    #[inline]
    pub const fn unknown() -> Self {
        TripCount::Unknown
    }

    /// Check if trip count is known exactly.
    #[inline]
    pub fn is_exact(&self) -> bool {
        matches!(self, TripCount::Constant(_))
    }

    /// Check if trip count is unknown.
    #[inline]
    pub fn is_unknown(&self) -> bool {
        matches!(self, TripCount::Unknown)
    }

    /// Check if trip count is symbolic.
    #[inline]
    pub fn is_symbolic(&self) -> bool {
        matches!(self, TripCount::Symbolic(_))
    }

    /// Get constant trip count if known.
    #[inline]
    pub fn as_constant(&self) -> Option<u64> {
        match self {
            TripCount::Constant(n) => Some(*n),
            TripCount::AtMost(n) => Some(*n),
            _ => None,
        }
    }

    /// Get symbolic trip count if present.
    #[inline]
    pub fn as_symbolic(&self) -> Option<&SymbolicTripCount> {
        match self {
            TripCount::Symbolic(s) => Some(s),
            _ => None,
        }
    }

    /// Check if loop definitely executes at least once.
    #[inline]
    pub fn executes_at_least_once(&self) -> bool {
        match self {
            TripCount::Constant(n) => *n >= 1,
            TripCount::AtMost(_) => false, // Could be zero
            TripCount::MaybeZero(_) => false,
            TripCount::Symbolic(s) => s.exact && matches!(s.start, TripCountValue::Constant(0)),
            TripCount::Unknown => false,
        }
    }

    /// Compute maximum IV value at loop exit.
    ///
    /// For `for i in range(start, end)` with step 1:
    /// - max_value = end - 1 (last iteration value)
    ///
    /// This is critical for bounds widening.
    pub fn max_iv_value(&self, iv: &InductionVariable) -> Option<MaxIVValue> {
        let step = iv.constant_step()?;
        if step == 0 {
            return None;
        }

        match self {
            TripCount::Constant(n) => {
                if *n == 0 {
                    return None;
                }
                let init = iv.constant_init()?;
                // max = init + step * (n - 1)
                let max = init.checked_add(step.checked_mul(*n as i64 - 1)?)?;
                Some(MaxIVValue::Constant(max))
            }

            TripCount::Symbolic(sym) => {
                // For symbolic: max = bound + offset - step
                // e.g., for `i < n` with step 1: max = n - 1
                Some(MaxIVValue::Symbolic {
                    bound: sym.bound_node,
                    offset: sym.offset - step,
                })
            }

            TripCount::AtMost(n) => {
                // Upper bound only - be conservative
                let init = iv.constant_init()?;
                let max = init.checked_add(step.checked_mul(*n as i64 - 1)?)?;
                Some(MaxIVValue::AtMost(max))
            }

            TripCount::MaybeZero(inner) => inner.max_iv_value(iv),

            TripCount::Unknown => None,
        }
    }
}

/// Maximum induction variable value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaxIVValue {
    /// Exactly this constant value.
    Constant(i64),

    /// At most this constant value.
    AtMost(i64),

    /// Symbolic: bound_node + offset.
    Symbolic { bound: NodeId, offset: i64 },
}

impl MaxIVValue {
    /// Check if this max value is definitely less than a constant bound.
    #[inline]
    pub fn definitely_less_than(&self, bound: i64) -> bool {
        match self {
            MaxIVValue::Constant(max) => *max < bound,
            MaxIVValue::AtMost(max) => *max < bound,
            MaxIVValue::Symbolic { .. } => false, // Cannot prove statically
        }
    }

    /// Check if this max value is definitely less than or equal to a constant bound.
    #[inline]
    pub fn definitely_at_most(&self, bound: i64) -> bool {
        match self {
            MaxIVValue::Constant(max) => *max <= bound,
            MaxIVValue::AtMost(max) => *max <= bound,
            MaxIVValue::Symbolic { .. } => false,
        }
    }
}

impl SymbolicTripCount {
    /// Create a new symbolic trip count.
    #[inline]
    pub fn new(bound_node: NodeId, offset: i64, start: TripCountValue, step: u64) -> Self {
        Self {
            bound_node,
            offset,
            start,
            step,
            exact: true,
        }
    }

    /// Create an inexact (upper bound) symbolic trip count.
    #[inline]
    pub fn upper_bound(bound_node: NodeId, offset: i64, start: TripCountValue, step: u64) -> Self {
        Self {
            bound_node,
            offset,
            start,
            step,
            exact: false,
        }
    }
}

impl TripCountValue {
    /// Check if this is a constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, TripCountValue::Constant(_))
    }

    /// Get constant value if present.
    #[inline]
    pub fn as_constant(&self) -> Option<i64> {
        match self {
            TripCountValue::Constant(v) => Some(*v),
            TripCountValue::Node(_) => None,
        }
    }
}

// =============================================================================
// Trip Count Analyzer
// =============================================================================

/// Analyzes loops to compute trip counts.
#[derive(Debug)]
pub struct TripCountAnalyzer<'g> {
    graph: &'g Graph,
}

impl<'g> TripCountAnalyzer<'g> {
    /// Create a new trip count analyzer.
    #[inline]
    pub fn new(graph: &'g Graph) -> Self {
        Self { graph }
    }

    /// Compute trip count for a loop given its primary induction variable.
    pub fn compute(&self, loop_info: &Loop, iv: &InductionVariable) -> TripCount {
        // We need:
        // 1. The loop exit condition
        // 2. The bound being compared against
        // 3. The IV initial value and step

        // Find the loop exit condition
        let Some(exit_info) = self.find_exit_condition(loop_info, iv) else {
            return TripCount::Unknown;
        };

        // Compute trip count based on exit condition
        self.compute_from_exit(iv, &exit_info)
    }

    /// Find the loop exit condition involving the induction variable.
    fn find_exit_condition(&self, loop_info: &Loop, iv: &InductionVariable) -> Option<ExitInfo> {
        // Look for If nodes in the loop header that compare with IV
        let header = loop_info.header;

        // Find control nodes that could be loop exits
        for (node_id, node) in self.graph.iter() {
            // Look for If nodes
            let Operator::Control(ControlOp::If) = &node.op else {
                continue;
            };

            // Check if the condition involves the IV
            let condition = node.inputs.get(1)?;
            let cond_node = self.graph.get(condition)?;

            // Analyze the comparison
            match &cond_node.op {
                Operator::IntCmp(cmp_op) => {
                    if let Some(exit) =
                        self.analyze_exit_comparison(iv, condition, *cmp_op, &cond_node.inputs)
                    {
                        return Some(exit);
                    }
                }
                _ => continue,
            }
        }

        None
    }

    /// Analyze an exit comparison to extract bound information.
    fn analyze_exit_comparison(
        &self,
        iv: &InductionVariable,
        _condition: NodeId,
        cmp_op: CmpOp,
        inputs: &crate::ir::node::InputList,
    ) -> Option<ExitInfo> {
        if inputs.len() != 2 {
            return None;
        }

        let lhs = inputs.get(0)?;
        let rhs = inputs.get(1)?;

        // Check if LHS is the IV
        if lhs == iv.phi {
            return self.make_exit_info(cmp_op, rhs, false);
        }

        // Check if RHS is the IV (reversed comparison)
        if rhs == iv.phi {
            return self.make_exit_info(Self::reverse_cmp(cmp_op), lhs, false);
        }

        None
    }

    /// Create exit info from a comparison.
    fn make_exit_info(&self, cmp_op: CmpOp, bound: NodeId, exit_on_true: bool) -> Option<ExitInfo> {
        let bound_value = if let Some(n) = self.graph.get(bound) {
            match &n.op {
                Operator::ConstInt(v) => BoundInfo::Constant(*v),
                _ => BoundInfo::Node(bound),
            }
        } else {
            return None;
        };

        Some(ExitInfo {
            comparison: cmp_op,
            bound: bound_value,
            exit_on_true,
        })
    }

    /// Reverse a comparison operator.
    ///
    /// For reversing operand order: `a op b` becomes `b op' a`.
    #[inline]
    fn reverse_cmp(op: CmpOp) -> CmpOp {
        match op {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::Le => CmpOp::Ge,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::Ge => CmpOp::Le,
            CmpOp::Eq => CmpOp::Eq,
            CmpOp::Ne => CmpOp::Ne,
            // These don't make sense to reverse for IV comparisons
            CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => op,
        }
    }

    /// Compute trip count from exit condition.
    fn compute_from_exit(&self, iv: &InductionVariable, exit: &ExitInfo) -> TripCount {
        let step = match iv.step {
            InductionStep::Constant(s) if s != 0 => s,
            _ => return TripCount::Unknown,
        };

        let step_abs = step.unsigned_abs();

        match (&iv.init, &exit.bound, exit.comparison, iv.direction) {
            // Common case: for i in range(0, n): ...
            // init = 0, step = 1, exit when i >= n (i.e., continue while i < n)
            (
                InductionInit::Constant(start),
                BoundInfo::Constant(end),
                CmpOp::Lt,
                InductionDirection::Increasing,
            ) => {
                if *end <= *start {
                    TripCount::Constant(0)
                } else {
                    let count = (*end - *start) as u64 / step_abs;
                    TripCount::Constant(count)
                }
            }

            // i < n with increasing step, symbolic bound
            (
                InductionInit::Constant(start),
                BoundInfo::Node(bound),
                CmpOp::Lt,
                InductionDirection::Increasing,
            ) => {
                TripCount::Symbolic(SymbolicTripCount::new(
                    *bound,
                    0, // For < n, offset is 0 (last value is n-1)
                    TripCountValue::Constant(*start),
                    step_abs,
                ))
            }

            // i <= n with increasing step
            (
                InductionInit::Constant(start),
                BoundInfo::Constant(end),
                CmpOp::Le,
                InductionDirection::Increasing,
            ) => {
                if *end < *start {
                    TripCount::Constant(0)
                } else {
                    let count = (*end - *start + 1) as u64 / step_abs;
                    TripCount::Constant(count)
                }
            }

            // i > n with decreasing step (countdown)
            (
                InductionInit::Constant(start),
                BoundInfo::Constant(end),
                CmpOp::Gt,
                InductionDirection::Decreasing,
            ) => {
                if *start <= *end {
                    TripCount::Constant(0)
                } else {
                    let count = (*start - *end) as u64 / step_abs;
                    TripCount::Constant(count)
                }
            }

            // i >= n with decreasing step
            (
                InductionInit::Constant(start),
                BoundInfo::Constant(end),
                CmpOp::Ge,
                InductionDirection::Decreasing,
            ) => {
                if *start < *end {
                    TripCount::Constant(0)
                } else {
                    let count = (*start - *end + 1) as u64 / step_abs;
                    TripCount::Constant(count)
                }
            }

            // All other cases
            _ => TripCount::Unknown,
        }
    }
}

/// Information about a loop exit condition.
#[derive(Debug, Clone)]
struct ExitInfo {
    /// The comparison operation.
    comparison: CmpOp,

    /// The bound being compared against.
    bound: BoundInfo,

    /// Whether the loop exits when condition is true.
    exit_on_true: bool,
}

/// Information about a loop bound.
#[derive(Debug, Clone)]
enum BoundInfo {
    /// Constant bound value.
    Constant(i64),

    /// Bound from a node.
    Node(NodeId),
}

// =============================================================================
// Trip Count Cache
// =============================================================================

/// Cached trip count analysis for all loops.
#[derive(Debug, Default)]
pub struct TripCountCache {
    /// Trip counts indexed by loop index.
    counts: Vec<Option<TripCount>>,
}

impl TripCountCache {
    /// Create empty cache.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create cache with capacity.
    #[inline]
    pub fn with_capacity(num_loops: usize) -> Self {
        Self {
            counts: vec![None; num_loops],
        }
    }

    /// Store trip count for a loop.
    #[inline]
    pub fn set(&mut self, loop_idx: usize, count: TripCount) {
        while self.counts.len() <= loop_idx {
            self.counts.push(None);
        }
        self.counts[loop_idx] = Some(count);
    }

    /// Get trip count for a loop.
    #[inline]
    pub fn get(&self, loop_idx: usize) -> Option<&TripCount> {
        self.counts.get(loop_idx).and_then(|c| c.as_ref())
    }

    /// Check if loop has known constant trip count.
    #[inline]
    pub fn has_constant(&self, loop_idx: usize) -> bool {
        self.get(loop_idx)
            .map_or(false, |c| matches!(c, TripCount::Constant(_)))
    }

    /// Get constant trip count if known.
    #[inline]
    pub fn constant(&self, loop_idx: usize) -> Option<u64> {
        self.get(loop_idx).and_then(|c| c.as_constant())
    }

    /// Count loops with known trip counts.
    pub fn count_known(&self) -> usize {
        self.counts
            .iter()
            .filter(|c| c.as_ref().map_or(false, |t| !t.is_unknown()))
            .count()
    }

    /// Count loops with constant trip counts.
    pub fn count_constant(&self) -> usize {
        self.counts
            .iter()
            .filter(|c| c.as_ref().map_or(false, |t| t.is_exact()))
            .count()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TripCount Tests
    // =========================================================================

    #[test]
    fn test_trip_count_constant() {
        let tc = TripCount::constant(100);
        assert!(tc.is_exact());
        assert!(!tc.is_unknown());
        assert!(!tc.is_symbolic());
        assert_eq!(tc.as_constant(), Some(100));
    }

    #[test]
    fn test_trip_count_unknown() {
        let tc = TripCount::unknown();
        assert!(!tc.is_exact());
        assert!(tc.is_unknown());
        assert!(!tc.is_symbolic());
        assert_eq!(tc.as_constant(), None);
    }

    #[test]
    fn test_trip_count_at_most() {
        let tc = TripCount::AtMost(50);
        assert!(!tc.is_exact());
        assert!(!tc.is_unknown());
        assert_eq!(tc.as_constant(), Some(50));
    }

    #[test]
    fn test_trip_count_symbolic() {
        let sym = SymbolicTripCount::new(NodeId::new(5), 0, TripCountValue::Constant(0), 1);
        let tc = TripCount::Symbolic(sym);
        assert!(!tc.is_exact());
        assert!(tc.is_symbolic());
        assert!(tc.as_symbolic().is_some());
    }

    #[test]
    fn test_trip_count_executes_at_least_once() {
        assert!(TripCount::Constant(1).executes_at_least_once());
        assert!(TripCount::Constant(100).executes_at_least_once());
        assert!(!TripCount::Constant(0).executes_at_least_once());
        assert!(!TripCount::AtMost(100).executes_at_least_once());
        assert!(!TripCount::Unknown.executes_at_least_once());
    }

    #[test]
    fn test_trip_count_maybe_zero() {
        let inner = TripCount::Constant(10);
        let tc = TripCount::MaybeZero(Box::new(inner));
        assert!(!tc.executes_at_least_once());
    }

    // =========================================================================
    // MaxIVValue Tests
    // =========================================================================

    #[test]
    fn test_max_iv_constant() {
        let iv = make_canonical_iv();
        let tc = TripCount::Constant(100);
        let max = tc.max_iv_value(&iv);
        assert_eq!(max, Some(MaxIVValue::Constant(99)));
    }

    #[test]
    fn test_max_iv_with_step() {
        let iv = InductionVariable::new(
            NodeId::new(0),
            InductionInit::Constant(0),
            InductionStep::Constant(2),
            InductionDirection::Increasing,
            None,
        );
        let tc = TripCount::Constant(50);
        let max = tc.max_iv_value(&iv);
        // max = 0 + 2 * (50 - 1) = 98
        assert_eq!(max, Some(MaxIVValue::Constant(98)));
    }

    #[test]
    fn test_max_iv_zero_trip() {
        let iv = make_canonical_iv();
        let tc = TripCount::Constant(0);
        let max = tc.max_iv_value(&iv);
        assert_eq!(max, None);
    }

    #[test]
    fn test_max_iv_symbolic() {
        let iv = make_canonical_iv();
        let sym = SymbolicTripCount::new(NodeId::new(5), 0, TripCountValue::Constant(0), 1);
        let tc = TripCount::Symbolic(sym);
        let max = tc.max_iv_value(&iv);
        // For i < n with step 1: max = n - 1 (offset = 0 - 1 = -1)
        assert_eq!(
            max,
            Some(MaxIVValue::Symbolic {
                bound: NodeId::new(5),
                offset: -1
            })
        );
    }

    #[test]
    fn test_max_iv_definitely_less_than() {
        let max = MaxIVValue::Constant(99);
        assert!(max.definitely_less_than(100));
        assert!(!max.definitely_less_than(99));
        assert!(!max.definitely_less_than(50));
    }

    #[test]
    fn test_max_iv_definitely_at_most() {
        let max = MaxIVValue::Constant(99);
        assert!(max.definitely_at_most(100));
        assert!(max.definitely_at_most(99));
        assert!(!max.definitely_at_most(50));
    }

    #[test]
    fn test_max_iv_symbolic_cannot_prove() {
        let max = MaxIVValue::Symbolic {
            bound: NodeId::new(5),
            offset: -1,
        };
        assert!(!max.definitely_less_than(100));
        assert!(!max.definitely_at_most(100));
    }

    // =========================================================================
    // SymbolicTripCount Tests
    // =========================================================================

    #[test]
    fn test_symbolic_trip_count_new() {
        let sym = SymbolicTripCount::new(NodeId::new(10), 0, TripCountValue::Constant(0), 1);
        assert_eq!(sym.bound_node, NodeId::new(10));
        assert_eq!(sym.offset, 0);
        assert!(sym.exact);
    }

    #[test]
    fn test_symbolic_trip_count_upper_bound() {
        let sym =
            SymbolicTripCount::upper_bound(NodeId::new(10), 0, TripCountValue::Constant(0), 1);
        assert!(!sym.exact);
    }

    // =========================================================================
    // TripCountValue Tests
    // =========================================================================

    #[test]
    fn test_trip_count_value_constant() {
        let v = TripCountValue::Constant(42);
        assert!(v.is_constant());
        assert_eq!(v.as_constant(), Some(42));
    }

    #[test]
    fn test_trip_count_value_node() {
        let v = TripCountValue::Node(NodeId::new(5));
        assert!(!v.is_constant());
        assert_eq!(v.as_constant(), None);
    }

    // =========================================================================
    // TripCountCache Tests
    // =========================================================================

    #[test]
    fn test_cache_new() {
        let cache = TripCountCache::new();
        assert_eq!(cache.get(0), None);
        assert_eq!(cache.count_known(), 0);
    }

    #[test]
    fn test_cache_with_capacity() {
        let cache = TripCountCache::with_capacity(5);
        assert_eq!(cache.get(0), None);
        for i in 0..5 {
            assert_eq!(cache.get(i), None);
        }
    }

    #[test]
    fn test_cache_set_get() {
        let mut cache = TripCountCache::new();
        cache.set(0, TripCount::Constant(100));
        cache.set(2, TripCount::Unknown);

        assert_eq!(cache.get(0), Some(&TripCount::Constant(100)));
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), Some(&TripCount::Unknown));
    }

    #[test]
    fn test_cache_has_constant() {
        let mut cache = TripCountCache::new();
        cache.set(0, TripCount::Constant(100));
        cache.set(1, TripCount::Unknown);

        assert!(cache.has_constant(0));
        assert!(!cache.has_constant(1));
        assert!(!cache.has_constant(2));
    }

    #[test]
    fn test_cache_constant() {
        let mut cache = TripCountCache::new();
        cache.set(0, TripCount::Constant(100));
        cache.set(1, TripCount::AtMost(50));

        assert_eq!(cache.constant(0), Some(100));
        assert_eq!(cache.constant(1), Some(50));
        assert_eq!(cache.constant(2), None);
    }

    #[test]
    fn test_cache_count_known() {
        let mut cache = TripCountCache::new();
        cache.set(0, TripCount::Constant(100));
        cache.set(1, TripCount::Unknown);
        cache.set(2, TripCount::AtMost(50));

        assert_eq!(cache.count_known(), 2);
    }

    #[test]
    fn test_cache_count_constant() {
        let mut cache = TripCountCache::new();
        cache.set(0, TripCount::Constant(100));
        cache.set(1, TripCount::Unknown);
        cache.set(2, TripCount::AtMost(50));

        assert_eq!(cache.count_constant(), 1);
    }

    // =========================================================================
    // Helper Functions
    // =========================================================================

    fn make_canonical_iv() -> InductionVariable {
        InductionVariable::new(
            NodeId::new(0),
            InductionInit::Constant(0),
            InductionStep::Constant(1),
            InductionDirection::Increasing,
            None,
        )
    }

    use super::super::induction::InductionVariable;
}
