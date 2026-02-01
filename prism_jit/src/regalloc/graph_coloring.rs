//! Chaitin-Briggs Graph Coloring Register Allocator
//!
//! An optimizing register allocator using graph coloring with:
//! - Aggressive coalescing for move elimination
//! - Iterated register coalescing (IRC)
//! - Biased coloring for better register assignment
//!
//! # Algorithm Overview
//!
//! The Chaitin-Briggs algorithm consists of the following phases:
//!
//! 1. **Build**: Construct interference graph
//! 2. **Simplify**: Push nodes with degree < K onto stack
//! 3. **Coalesce**: Merge non-interfering move-related nodes
//! 4. **Freeze**: Give up on coalescing low-degree move-related nodes
//! 5. **Spill**: Mark nodes for spilling if none can be simplified
//! 6. **Select**: Pop nodes from stack and assign colors
//!
//! # Complexity
//!
//! O(n² + n·e) in the worst case, where n = nodes and e = edges.
//! In practice, much faster due to sparse interference graphs.

use super::interference::InterferenceGraph;
use super::interval::LiveInterval;
use super::spill::SpillSlot;
use super::{Allocation, AllocationMap, AllocatorConfig, AllocatorStats, PReg, RegClass, VReg};
use crate::backend::x64::registers::{Gpr, GprSet, Xmm, XmmSet};
use std::collections::{HashMap, HashSet, VecDeque};

// =============================================================================
// Node Worklists
// =============================================================================

/// Classification of nodes for the allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeStatus {
    /// Not yet processed.
    Initial,
    /// Low-degree, not move-related.
    Simplify,
    /// Low-degree, move-related (might coalesce).
    Freeze,
    /// High-degree node.
    Spill,
    /// Pushed onto select stack.
    OnStack,
    /// Already colored.
    Colored,
    /// Coalesced into another node.
    Coalesced,
    /// Marked for actual spilling.
    Spilled,
}

// =============================================================================
// Graph Coloring Allocator
// =============================================================================

/// The Chaitin-Briggs graph coloring register allocator.
pub struct GraphColoringAllocator {
    /// Configuration.
    config: AllocatorConfig,
    /// Interference graph.
    igraph: InterferenceGraph,
    /// Live intervals for each vreg.
    intervals: HashMap<VReg, LiveInterval>,
    /// Status of each node.
    status: HashMap<VReg, NodeStatus>,
    /// Coalescing: maps coalesced nodes to their alias.
    alias: HashMap<VReg, VReg>,
    /// Select stack.
    select_stack: Vec<VReg>,
    /// Allocation results.
    allocations: AllocationMap,
    /// Statistics.
    stats: AllocatorStats,
    /// K for GPRs (number of colors).
    k_gpr: u32,
    /// K for XMMs.
    k_xmm: u32,
}

impl GraphColoringAllocator {
    /// Create a new allocator.
    pub fn new(config: AllocatorConfig) -> Self {
        let k_gpr = config.available_gprs.count();
        let k_xmm = config.available_xmms.count();

        GraphColoringAllocator {
            config,
            igraph: InterferenceGraph::new(k_gpr, k_xmm),
            intervals: HashMap::new(),
            status: HashMap::new(),
            alias: HashMap::new(),
            select_stack: Vec::new(),
            allocations: AllocationMap::new(),
            stats: AllocatorStats::default(),
            k_gpr,
            k_xmm,
        }
    }

    /// Run register allocation.
    pub fn allocate(mut self, intervals: Vec<LiveInterval>) -> (AllocationMap, AllocatorStats) {
        self.stats.num_vregs = intervals.len();

        // Store intervals
        for interval in &intervals {
            self.intervals.insert(interval.vreg, interval.clone());
            self.status.insert(interval.vreg, NodeStatus::Initial);
        }

        // Build interference graph
        self.igraph = InterferenceGraph::build(&intervals, self.k_gpr, self.k_xmm);

        // Main allocation loop
        loop {
            // Classify nodes into worklists
            self.classify_nodes();

            // Simplify - push low-degree non-move-related nodes
            if self.simplify() {
                continue;
            }

            // Coalesce - merge move-related nodes (if enabled)
            if self.config.enable_coalescing && self.coalesce() {
                continue;
            }

            // Freeze - give up on coalescing for a freeze node
            if self.freeze() {
                continue;
            }

            // Potential spill - select a node for spilling
            if self.select_spill() {
                continue;
            }

            // All nodes processed
            break;
        }

        // Select - pop from stack and assign colors
        self.select();

        (self.allocations, self.stats)
    }

    /// Get the representative (alias) of a possibly-coalesced node.
    fn get_alias(&self, vreg: VReg) -> VReg {
        if let Some(&alias) = self.alias.get(&vreg) {
            if self.status.get(&vreg) == Some(&NodeStatus::Coalesced) {
                return self.get_alias(alias);
            }
        }
        vreg
    }

    /// Get the K value (number of colors) for a vreg.
    fn k(&self, vreg: VReg) -> u32 {
        let is_float = self
            .intervals
            .get(&vreg)
            .map(|i| i.reg_class == RegClass::Float)
            .unwrap_or(false);

        if is_float { self.k_xmm } else { self.k_gpr }
    }

    /// Classify nodes into worklists based on degree and move-relatedness.
    fn classify_nodes(&mut self) {
        for (&vreg, status) in self.status.iter_mut() {
            if *status != NodeStatus::Initial {
                continue;
            }

            let k = self.k(vreg);
            let degree = self.igraph.degree(vreg);
            let move_related = self.is_move_related(vreg);

            *status = if degree < k {
                if move_related {
                    NodeStatus::Freeze
                } else {
                    NodeStatus::Simplify
                }
            } else {
                NodeStatus::Spill
            };
        }
    }

    /// Check if a vreg is involved in a move.
    fn is_move_related(&self, vreg: VReg) -> bool {
        self.igraph
            .move_edges()
            .iter()
            .any(|&(from, to)| self.get_alias(from) == vreg || self.get_alias(to) == vreg)
    }

    /// Simplify phase: push low-degree non-move-related nodes onto stack.
    fn simplify(&mut self) -> bool {
        let simplify_nodes: Vec<VReg> = self
            .status
            .iter()
            .filter(|(_, &s)| s == NodeStatus::Simplify)
            .map(|(&v, _)| v)
            .collect();

        if let Some(vreg) = simplify_nodes.first() {
            self.push_to_stack(*vreg);
            return true;
        }

        false
    }

    /// Push a node onto the select stack.
    fn push_to_stack(&mut self, vreg: VReg) {
        self.status.insert(vreg, NodeStatus::OnStack);
        self.select_stack.push(vreg);

        // Decrement degree of neighbors
        for neighbor in self.igraph.neighbors(vreg).collect::<Vec<_>>() {
            let neighbor = self.get_alias(neighbor);
            if self.status.get(&neighbor) == Some(&NodeStatus::OnStack) {
                continue;
            }

            // Update worklist classification if degree drops below K
            let k = self.k(neighbor);
            let degree = self.igraph.degree(neighbor);

            if degree == k {
                // Was high-degree, now might be low-degree
                let status = self.status.get_mut(&neighbor).unwrap();
                if *status == NodeStatus::Spill {
                    *status = if self.is_move_related(neighbor) {
                        NodeStatus::Freeze
                    } else {
                        NodeStatus::Simplify
                    };
                }
            }
        }
    }

    /// Coalesce phase: try to coalesce move-related nodes.
    fn coalesce(&mut self) -> bool {
        // Find a coalescable move edge
        for &(u, v) in self.igraph.move_edges() {
            let u = self.get_alias(u);
            let v = self.get_alias(v);

            if u == v {
                continue;
            }

            // Don't coalesce if both are on the stack or already colored
            let u_status = *self.status.get(&u).unwrap_or(&NodeStatus::Initial);
            let v_status = *self.status.get(&v).unwrap_or(&NodeStatus::Initial);

            if matches!(
                u_status,
                NodeStatus::OnStack
                    | NodeStatus::Colored
                    | NodeStatus::Coalesced
                    | NodeStatus::Spilled
            ) {
                continue;
            }
            if matches!(
                v_status,
                NodeStatus::OnStack
                    | NodeStatus::Colored
                    | NodeStatus::Coalesced
                    | NodeStatus::Spilled
            ) {
                continue;
            }

            // Check if they interfere
            if self.igraph.interferes(u, v) {
                continue;
            }

            // Use Briggs or George conservative coalescing heuristic
            if self.can_coalesce_briggs(u, v) || self.can_coalesce_george(u, v) {
                self.do_coalesce(u, v);
                self.stats.num_coalesced += 1;
                return true;
            }
        }

        false
    }

    /// Briggs heuristic: coalesce if combined node has < K high-degree neighbors.
    fn can_coalesce_briggs(&self, u: VReg, v: VReg) -> bool {
        let k = self.k(u);

        let mut high_degree_neighbors = HashSet::new();

        for neighbor in self.igraph.neighbors(u) {
            let neighbor = self.get_alias(neighbor);
            if self.igraph.degree(neighbor) >= k {
                high_degree_neighbors.insert(neighbor);
            }
        }

        for neighbor in self.igraph.neighbors(v) {
            let neighbor = self.get_alias(neighbor);
            if self.igraph.degree(neighbor) >= k {
                high_degree_neighbors.insert(neighbor);
            }
        }

        (high_degree_neighbors.len() as u32) < k
    }

    /// George heuristic: coalesce if every neighbor of u either:
    /// 1. Already interferes with v, or
    /// 2. Has degree < K
    fn can_coalesce_george(&self, u: VReg, v: VReg) -> bool {
        let k = self.k(u);

        for neighbor in self.igraph.neighbors(u) {
            let neighbor = self.get_alias(neighbor);
            if neighbor == v {
                continue;
            }

            // Must either interfere with v or have low degree
            if !self.igraph.interferes(neighbor, v) && self.igraph.degree(neighbor) >= k {
                return false;
            }
        }

        true
    }

    /// Perform coalescing: merge v into u.
    fn do_coalesce(&mut self, u: VReg, v: VReg) {
        self.alias.insert(v, u);
        self.status.insert(v, NodeStatus::Coalesced);

        // Combine interference edges
        self.igraph.coalesce(u, v);

        // If combined node is no longer move-related and low-degree, move to simplify
        let k = self.k(u);
        if self.igraph.degree(u) < k && !self.is_move_related(u) {
            self.status.insert(u, NodeStatus::Simplify);
        }
    }

    /// Freeze phase: give up on coalescing for the lowest-degree freeze node.
    fn freeze(&mut self) -> bool {
        // Find lowest-degree freeze node
        let mut best: Option<(VReg, u32)> = None;

        for (&vreg, &status) in &self.status {
            if status == NodeStatus::Freeze {
                let degree = self.igraph.degree(vreg);
                match best {
                    None => best = Some((vreg, degree)),
                    Some((_, best_degree)) if degree < best_degree => {
                        best = Some((vreg, degree));
                    }
                    _ => {}
                }
            }
        }

        if let Some((vreg, _)) = best {
            // Move from freeze to simplify
            self.status.insert(vreg, NodeStatus::Simplify);
            true
        } else {
            false
        }
    }

    /// Select spill phase: select a node for potential spilling.
    fn select_spill(&mut self) -> bool {
        // Find the best node to spill (lowest spill weight)
        let mut best: Option<(VReg, f32)> = None;

        for (&vreg, &status) in &self.status {
            if status == NodeStatus::Spill {
                let weight = self
                    .intervals
                    .get(&vreg)
                    .map(|i| i.spill_weight)
                    .unwrap_or(1.0);

                match best {
                    None => best = Some((vreg, weight)),
                    Some((_, best_weight)) if weight < best_weight => {
                        best = Some((vreg, weight));
                    }
                    _ => {}
                }
            }
        }

        if let Some((vreg, _)) = best {
            // Push to stack anyway - will be actual spill if coloring fails
            self.push_to_stack(vreg);
            true
        } else {
            false
        }
    }

    /// Select phase: pop nodes from stack and assign colors.
    fn select(&mut self) {
        while let Some(vreg) = self.select_stack.pop() {
            let is_float = self
                .intervals
                .get(&vreg)
                .map(|i| i.reg_class == RegClass::Float)
                .unwrap_or(false);

            // Collect colors used by neighbors
            let mut used_colors = HashSet::new();

            for neighbor in self.igraph.neighbors(vreg) {
                let neighbor = self.get_alias(neighbor);
                if let Allocation::Register(preg) = self.allocations.get(neighbor) {
                    used_colors.insert(preg);
                }
            }

            // Try to find an available color
            if let Some(preg) = self.find_color(is_float, &used_colors) {
                self.allocations.set(vreg, Allocation::Register(preg));
                self.status.insert(vreg, NodeStatus::Colored);
                self.stats.num_allocated += 1;
            } else {
                // Actual spill
                let slot = self.allocations.alloc_spill_slot();
                self.allocations.set(vreg, Allocation::Spill(slot));
                self.status.insert(vreg, NodeStatus::Spilled);
                self.stats.num_spilled += 1;
            }
        }

        // Handle coalesced nodes
        for (&vreg, &status) in &self.status.clone() {
            if status == NodeStatus::Coalesced {
                let alias = self.get_alias(vreg);
                let alloc = self.allocations.get(alias);
                self.allocations.set(vreg, alloc);
            }
        }
    }

    /// Find an available color (register).
    fn find_color(&self, is_float: bool, used: &HashSet<PReg>) -> Option<PReg> {
        if is_float {
            for xmm in self.config.available_xmms.iter() {
                let preg = PReg::Xmm(xmm);
                if !used.contains(&preg) {
                    return Some(preg);
                }
            }
        } else {
            for gpr in self.config.available_gprs.iter() {
                let preg = PReg::Gpr(gpr);
                if !used.contains(&preg) {
                    return Some(preg);
                }
            }
        }
        None
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
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
}
