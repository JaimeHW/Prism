//! Partial Redundancy Elimination (PRE) optimization pass.
//!
//! PRE removes computations that are redundant along some execution paths
//! using the Lazy Code Motion algorithm (Knoop-Rüthing-Steffen).
//!
//! # Algorithm Overview
//!
//! 1. **Expression numbering**: Assign unique IDs to expressions
//! 2. **Anticipation analysis**: Which expressions will be computed downstream
//! 3. **Availability analysis**: Which expressions have been computed upstream
//! 4. **Placement computation**: Where to insert/delete computations
//! 5. **Code motion**: Perform the actual transformations
//!
//! # Benefits Over GVN
//!
//! - Hoists redundant computations out of branches
//! - Reduces total number of computations
//! - Handles partial redundancies (not just full)

mod anticipation;
mod availability;
mod motion;
mod placement;

pub use anticipation::AnticipationAnalysis;
pub use availability::AvailabilityAnalysis;
pub use motion::CodeMotionEngine;
pub use placement::PlacementAnalysis;

use rustc_hash::FxHashMap;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::Operator;
use crate::opt::OptimizationPass;

// =============================================================================
// Expression Numbering
// =============================================================================

/// Unique identifier for an expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

impl ExprId {
    /// Create a new expression ID.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID.
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// An expression in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Expression {
    /// The operator.
    pub op: Operator,
    /// Input expression IDs (for recursive numbering).
    pub inputs: Vec<ExprId>,
}

impl Expression {
    /// Create a new expression.
    pub fn new(op: Operator, inputs: Vec<ExprId>) -> Self {
        Self { op, inputs }
    }
}

/// Expression table for value numbering.
#[derive(Debug, Default)]
pub struct ExpressionTable {
    /// Expression -> ID mapping.
    expr_to_id: FxHashMap<Expression, ExprId>,
    /// ID -> Expression mapping.
    id_to_expr: Vec<Expression>,
    /// Node -> Expression ID mapping.
    node_to_expr: FxHashMap<NodeId, ExprId>,
    /// Expression ID -> canonical node(s).
    expr_to_nodes: FxHashMap<ExprId, Vec<NodeId>>,
}

impl ExpressionTable {
    /// Create a new expression table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build expression table from graph.
    pub fn build(graph: &Graph) -> Self {
        let mut table = Self::new();
        table.analyze(graph);
        table
    }

    /// Analyze the graph and number all expressions.
    fn analyze(&mut self, graph: &Graph) {
        // First pass: number leaf nodes (constants, parameters)
        for i in 0..graph.len() {
            let id = NodeId::new(i as u32);
            if let Some(node) = graph.get(id) {
                if Self::is_numberable(&node.op) && node.inputs.len() == 0 {
                    self.get_or_create_expr_id(graph, id);
                }
            }
        }

        // Second pass: number all other expressions
        for i in 0..graph.len() {
            let id = NodeId::new(i as u32);
            if let Some(node) = graph.get(id) {
                if Self::is_numberable(&node.op) {
                    self.get_or_create_expr_id(graph, id);
                }
            }
        }
    }

    /// Get or create expression ID for a node.
    fn get_or_create_expr_id(&mut self, graph: &Graph, node_id: NodeId) -> Option<ExprId> {
        // Already numbered?
        if let Some(&expr_id) = self.node_to_expr.get(&node_id) {
            return Some(expr_id);
        }

        let node = graph.get(node_id)?;

        // Get input expression IDs
        let mut input_exprs = Vec::new();
        for input in node.inputs.iter() {
            if let Some(&input_expr) = self.node_to_expr.get(&input) {
                input_exprs.push(input_expr);
            } else {
                // Input not numbered yet - try to number it
                if let Some(input_expr) = self.get_or_create_expr_id(graph, input) {
                    input_exprs.push(input_expr);
                } else {
                    // Can't number this expression
                    return None;
                }
            }
        }

        let expr = Expression::new(node.op.clone(), input_exprs);

        // Check if this expression already exists
        let expr_id = if let Some(&existing_id) = self.expr_to_id.get(&expr) {
            existing_id
        } else {
            // Create new ID
            let new_id = ExprId::new(self.id_to_expr.len() as u32);
            self.expr_to_id.insert(expr.clone(), new_id);
            self.id_to_expr.push(expr);
            new_id
        };

        // Map node to expression
        self.node_to_expr.insert(node_id, expr_id);

        // Map expression to node
        self.expr_to_nodes
            .entry(expr_id)
            .or_insert_with(Vec::new)
            .push(node_id);

        Some(expr_id)
    }

    /// Check if an operator should be numbered.
    fn is_numberable(op: &Operator) -> bool {
        match op {
            // Pure operations
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone
            | Operator::IntOp(_)
            | Operator::FloatOp(_)
            | Operator::IntCmp(_)
            | Operator::FloatCmp(_)
            | Operator::Bitwise(_)
            | Operator::LogicalNot => true,
            // Side-effecting or control ops are not numberable
            _ => false,
        }
    }

    /// Get the number of unique expressions.
    pub fn len(&self) -> usize {
        self.id_to_expr.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_expr.is_empty()
    }

    /// Get expression ID for a node.
    pub fn get_expr_id(&self, node: NodeId) -> Option<ExprId> {
        self.node_to_expr.get(&node).copied()
    }

    /// Get expression by ID.
    pub fn get_expr(&self, id: ExprId) -> Option<&Expression> {
        self.id_to_expr.get(id.0 as usize)
    }

    /// Get nodes for an expression.
    pub fn get_nodes(&self, id: ExprId) -> &[NodeId] {
        self.expr_to_nodes.get(&id).map_or(&[], |v| v.as_slice())
    }
}

// =============================================================================
// PRE Statistics
// =============================================================================

/// Statistics from PRE.
#[derive(Debug, Clone, Default)]
pub struct PreStats {
    /// Expressions eliminated.
    pub expressions_eliminated: usize,
    /// Expressions inserted.
    pub expressions_inserted: usize,
    /// Total expressions analyzed.
    pub expressions_analyzed: usize,
}

impl PreStats {
    /// Net code reduction (eliminated - inserted).
    pub fn net_reduction(&self) -> isize {
        self.expressions_eliminated as isize - self.expressions_inserted as isize
    }

    /// Merge statistics.
    pub fn merge(&mut self, other: &PreStats) {
        self.expressions_eliminated += other.expressions_eliminated;
        self.expressions_inserted += other.expressions_inserted;
        self.expressions_analyzed += other.expressions_analyzed;
    }
}

// =============================================================================
// PRE Configuration
// =============================================================================

/// Configuration for PRE.
#[derive(Debug, Clone)]
pub struct PreConfig {
    /// Maximum allowed code growth ratio (inserted/eliminated).
    pub max_code_growth: f32,
    /// Minimum frequency ratio for hoisting.
    pub min_frequency_ratio: f32,
    /// Maximum expression size to consider.
    pub max_expr_size: usize,
}

impl Default for PreConfig {
    fn default() -> Self {
        Self {
            max_code_growth: 1.5,
            min_frequency_ratio: 0.5,
            max_expr_size: 100,
        }
    }
}

// =============================================================================
// PRE Pass
// =============================================================================

/// Partial Redundancy Elimination pass.
#[derive(Debug)]
pub struct Pre {
    /// Configuration.
    config: PreConfig,
    /// Statistics from last run.
    stats: PreStats,
}

impl Pre {
    /// Create a new PRE pass.
    pub fn new() -> Self {
        Self {
            config: PreConfig::default(),
            stats: PreStats::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: PreConfig) -> Self {
        Self {
            config,
            stats: PreStats::default(),
        }
    }

    /// Get statistics from last run.
    pub fn stats(&self) -> &PreStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &PreConfig {
        &self.config
    }

    /// Run PRE analysis and optimization.
    fn run_pre(&mut self, graph: &mut Graph) -> bool {
        self.stats = PreStats::default();

        // Step 1: Build expression table
        let expr_table = ExpressionTable::build(graph);
        self.stats.expressions_analyzed = expr_table.len();

        if expr_table.is_empty() {
            return false;
        }

        // Step 2: Compute anticipation (backward dataflow)
        let anticipation = AnticipationAnalysis::compute(graph, &expr_table);

        // Step 3: Compute availability (forward dataflow)
        let availability = AvailabilityAnalysis::compute(graph, &expr_table);

        // Step 4: Compute optimal placement
        let placement = PlacementAnalysis::compute(&anticipation, &availability, &expr_table);

        // Step 5: Perform code motion
        let mut motion = CodeMotionEngine::new(graph, &placement);
        let changed = motion.apply();

        self.stats.expressions_inserted = motion.inserted();
        self.stats.expressions_eliminated = motion.eliminated();

        changed
    }
}

impl Default for Pre {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Pre {
    fn name(&self) -> &'static str {
        "pre"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_pre(graph)
    }
}
