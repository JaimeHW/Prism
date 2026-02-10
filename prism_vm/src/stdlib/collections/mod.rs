//! Python `collections` module implementation.
//!
//! Provides high-performance specialized container datatypes with optimal
//! algorithmic complexity and minimal memory overhead.
//!
//! # Data Structures
//!
//! | Type | Description | Key Operations |
//! |------|-------------|----------------|
//! | `deque` | Double-ended queue | O(1) append/pop both ends |
//! | `Counter` | Hashable element counter | O(1) increment/lookup |
//! | `defaultdict` | Dict with default factory | O(1) missing key handling |
//! | `OrderedDict` | Insertion-ordered dict | O(1) ops with order |
//! | `namedtuple` | Named field tuples | Immutable, memory-efficient |
//!
//! # Performance Characteristics
//!
//! ## deque
//!
//! - Implemented as a ring buffer with dynamic growth for cache efficiency
//! - O(1) `append()`, `appendleft()`, `pop()`, `popleft()`
//! - O(1) amortized `extend()`, `extendleft()`
//! - O(n) indexed access (use list for random access)
//!
//! ## Counter
//!
//! - Built on HashMap with optimized update path
//! - O(1) element counting and retrieval
//! - O(n) `most_common()` operation
//!
//! # Thread Safety
//!
//! All container types are **not** thread-safe by design (matching Python).
//! Use external synchronization if concurrent access is needed.

pub mod counter;
pub mod defaultdict;
pub mod deque;
pub mod ordereddict;

#[cfg(test)]
mod tests;

use super::{Module, ModuleError, ModuleResult};
use std::sync::Arc;

// Re-export core types
pub use counter::Counter;
pub use deque::Deque;

// =============================================================================
// Collections Module
// =============================================================================

/// The collections module implementation.
pub struct CollectionsModule {
    attrs: Vec<Arc<str>>,
}

impl CollectionsModule {
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("deque"),
                Arc::from("Counter"),
                Arc::from("defaultdict"),
                Arc::from("OrderedDict"),
                Arc::from("namedtuple"),
                Arc::from("ChainMap"),
                Arc::from("UserDict"),
                Arc::from("UserList"),
                Arc::from("UserString"),
            ],
        }
    }
}

impl Default for CollectionsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CollectionsModule {
    fn name(&self) -> &str {
        "collections"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            // Container types - return type objects
            "deque" | "Counter" | "defaultdict" | "OrderedDict" | "namedtuple" | "ChainMap"
            | "UserDict" | "UserList" | "UserString" => {
                // TODO: Return actual type objects when type system is ready
                Err(ModuleError::AttributeError(format!(
                    "collections.{} is not yet available as a type object",
                    name
                )))
            }
            _ => Err(ModuleError::AttributeError(format!(
                "module 'collections' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Create a new collections module instance.
pub fn create_module() -> CollectionsModule {
    CollectionsModule::new()
}
