//! Worklist for efficient instruction combining.
//!
//! The worklist uses a VecDeque for FIFO ordering with deduplication
//! to avoid processing the same instruction multiple times.

use rustc_hash::FxHashSet;
use std::collections::VecDeque;

use crate::ir::node::NodeId;

// =============================================================================
// Worklist
// =============================================================================

/// Worklist for instruction combining.
#[derive(Debug)]
pub struct Worklist {
    /// The queue of nodes to process.
    queue: VecDeque<NodeId>,
    /// Set of nodes currently in the queue (for deduplication).
    in_queue: FxHashSet<NodeId>,
    /// Total nodes ever added.
    total_added: usize,
    /// Total nodes processed (popped).
    total_processed: usize,
}

impl Worklist {
    /// Create a new empty worklist.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            in_queue: FxHashSet::default(),
            total_added: 0,
            total_processed: 0,
        }
    }

    /// Create a worklist with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(capacity),
            in_queue: FxHashSet::default(),
            total_added: 0,
            total_processed: 0,
        }
    }

    /// Push a node onto the worklist.
    /// Returns true if the node was added (not already present).
    pub fn push(&mut self, node: NodeId) -> bool {
        if self.in_queue.insert(node) {
            self.queue.push_back(node);
            self.total_added += 1;
            true
        } else {
            false
        }
    }

    /// Push multiple nodes onto the worklist.
    pub fn push_all(&mut self, nodes: impl IntoIterator<Item = NodeId>) {
        for node in nodes {
            self.push(node);
        }
    }

    /// Pop the next node from the worklist.
    pub fn pop(&mut self) -> Option<NodeId> {
        if let Some(node) = self.queue.pop_front() {
            self.in_queue.remove(&node);
            self.total_processed += 1;
            Some(node)
        } else {
            None
        }
    }

    /// Check if the worklist is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get the current size of the worklist.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check if a node is in the worklist.
    pub fn contains(&self, node: NodeId) -> bool {
        self.in_queue.contains(&node)
    }

    /// Get total nodes ever added.
    pub fn total_added(&self) -> usize {
        self.total_added
    }

    /// Get total nodes processed.
    pub fn total_processed(&self) -> usize {
        self.total_processed
    }

    /// Clear the worklist.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.in_queue.clear();
    }
}

impl Default for Worklist {
    fn default() -> Self {
        Self::new()
    }
}
