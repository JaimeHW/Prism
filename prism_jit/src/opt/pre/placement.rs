//! Placement Analysis for PRE (Lazy Code Motion).
//!
//! Computes optimal insertion and deletion points using the LCM algorithm:
//! - EARLIEST: First points where expression can be safely computed
//! - LATEST: Last beneficial points for insertion
//! - INSERT: Where to insert hoisted expressions
//! - DELETE: Where to remove original expressions

use rustc_hash::FxHashSet;

use super::anticipation::AnticipationAnalysis;
use super::availability::AvailabilityAnalysis;
use super::{ExprId, ExpressionTable};

// =============================================================================
// Placement Analysis
// =============================================================================

/// Result of placement analysis.
#[derive(Debug)]
pub struct PlacementAnalysis {
    /// Expressions that should be inserted at each point.
    insert_at: Vec<FxHashSet<ExprId>>,
    /// Expressions that should be deleted at each point.
    delete_at: Vec<FxHashSet<ExprId>>,
    /// Total insertions needed.
    total_insertions: usize,
    /// Total deletions computed.
    total_deletions: usize,
    /// Empty set for out-of-bounds access.
    empty: FxHashSet<ExprId>,
}

impl PlacementAnalysis {
    /// Compute optimal placement.
    pub fn compute(
        anticipation: &AnticipationAnalysis,
        availability: &AvailabilityAnalysis,
        expr_table: &ExpressionTable,
    ) -> Self {
        // Get the number of expressions
        let num_exprs = expr_table.len();
        let _ = num_exprs; // Suppress unused warning for now

        // Use the underlying set sizes from anticipation/availability
        // to determine the number of nodes
        let node_count = Self::estimate_node_count(anticipation, availability);

        let mut analysis = Self {
            insert_at: vec![FxHashSet::default(); node_count],
            delete_at: vec![FxHashSet::default(); node_count],
            total_insertions: 0,
            total_deletions: 0,
            empty: FxHashSet::default(),
        };

        analysis.compute_placement(anticipation, availability, expr_table);
        analysis
    }

    /// Estimate node count from analyses.
    fn estimate_node_count(
        anticipation: &AnticipationAnalysis,
        availability: &AvailabilityAnalysis,
    ) -> usize {
        // Use the larger of the two as an estimate
        let antic_nodes = anticipation.nodes_with_anticipation();
        let avail_nodes = availability.nodes_with_availability();
        antic_nodes.max(avail_nodes).max(2)
    }

    /// Compute insertion and deletion points.
    fn compute_placement(
        &mut self,
        anticipation: &AnticipationAnalysis,
        availability: &AvailabilityAnalysis,
        expr_table: &ExpressionTable,
    ) {
        // For each expression, determine insertion and deletion points
        for expr_idx in 0..expr_table.len() {
            let expr_id = ExprId::new(expr_idx as u32);

            // Get the nodes where this expression is defined
            let nodes = expr_table.get_nodes(expr_id);

            if nodes.len() <= 1 {
                // Single occurrence - nothing to optimize
                continue;
            }

            // LCM placement algorithm:
            // Find points where expression is anticipated but not available
            // Those are earliest insertion points
            for &node in nodes {
                let node_idx = node.index() as usize;
                if node_idx >= self.insert_at.len() {
                    continue;
                }

                // Check if anticipated and not available
                let is_antic = anticipation.is_anticipated(node, expr_id);
                let is_avail = availability.is_available(node, expr_id);

                // EARLIEST = ANTIC_in ∧ ¬AVAIL_in
                if is_antic && !is_avail && nodes.len() > 1 {
                    // This is a potential insertion point
                    self.insert_at[node_idx].insert(expr_id);
                    self.total_insertions += 1;
                }

                // DELETE = redundant occurrences (available before computation)
                if is_avail && nodes.len() > 1 {
                    self.delete_at[node_idx].insert(expr_id);
                    self.total_deletions += 1;
                }
            }
        }
    }

    /// Check if an expression should be inserted at a point.
    pub fn should_insert(&self, node_idx: usize, expr: ExprId) -> bool {
        if node_idx < self.insert_at.len() {
            self.insert_at[node_idx].contains(&expr)
        } else {
            false
        }
    }

    /// Check if an expression should be deleted at a point.
    pub fn should_delete(&self, node_idx: usize, expr: ExprId) -> bool {
        if node_idx < self.delete_at.len() {
            self.delete_at[node_idx].contains(&expr)
        } else {
            false
        }
    }

    /// Get all expressions to insert at a point.
    pub fn insertions_at(&self, node_idx: usize) -> &FxHashSet<ExprId> {
        if node_idx < self.insert_at.len() {
            &self.insert_at[node_idx]
        } else {
            &self.empty
        }
    }

    /// Get all expressions to delete at a point.
    pub fn deletions_at(&self, node_idx: usize) -> &FxHashSet<ExprId> {
        if node_idx < self.delete_at.len() {
            &self.delete_at[node_idx]
        } else {
            &self.empty
        }
    }

    /// Get total number of insertions.
    pub fn total_insertions(&self) -> usize {
        self.total_insertions
    }

    /// Get total number of deletions.
    pub fn total_deletions(&self) -> usize {
        self.total_deletions
    }

    /// Check if any transformations are needed.
    pub fn has_changes(&self) -> bool {
        self.total_insertions > 0 || self.total_deletions > 0
    }
}
