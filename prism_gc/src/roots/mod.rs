//! Root tracking and handle management.
//!
//! GC roots are the starting points for tracing:
//! - Stack frames (registers, locals)
//! - Global variables
//! - Handles held by Rust code

mod handles;

pub use handles::{GcHandle, HandleScope, RawHandle};

use crate::trace::Tracer;
use prism_core::Value;
use std::cell::RefCell;

/// Root set for garbage collection.
///
/// Collects all roots that need to be traced during GC.
pub struct RootSet {
    /// Explicit GC handles.
    handles: RefCell<Vec<RawHandle>>,
    /// Global roots (module globals, builtins).
    globals: RefCell<Vec<Value>>,
}

impl RootSet {
    /// Create a new empty root set.
    pub fn new() -> Self {
        Self {
            handles: RefCell::new(Vec::new()),
            globals: RefCell::new(Vec::new()),
        }
    }

    /// Register a handle as a root.
    pub fn register_handle(&self, handle: RawHandle) {
        self.handles.borrow_mut().push(handle);
    }

    /// Unregister a handle.
    pub fn unregister_handle(&self, handle: RawHandle) {
        self.handles.borrow_mut().retain(|h| h.ptr != handle.ptr);
    }

    /// Add a global root.
    pub fn add_global(&self, value: Value) {
        self.globals.borrow_mut().push(value);
    }

    /// Clear all globals (for reset/shutdown).
    pub fn clear_globals(&self) {
        self.globals.borrow_mut().clear();
    }

    /// Trace all roots.
    pub fn trace(&self, tracer: &mut dyn Tracer) {
        // Trace handles
        for handle in self.handles.borrow().iter() {
            if !handle.ptr.is_null() {
                tracer.trace_ptr(handle.ptr);
            }
        }

        // Trace globals
        for value in self.globals.borrow().iter() {
            tracer.trace_value(*value);
        }
    }

    /// Get number of registered handles.
    pub fn handle_count(&self) -> usize {
        self.handles.borrow().len()
    }

    /// Get number of global roots.
    pub fn global_count(&self) -> usize {
        self.globals.borrow().len()
    }
}

impl Default for RootSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can provide roots to the GC.
pub trait RootProvider {
    /// Trace all roots held by this provider.
    fn trace_roots(&self, tracer: &mut dyn Tracer);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::tracer::CountingTracer;

    #[test]
    fn test_root_set_creation() {
        let roots = RootSet::new();
        assert_eq!(roots.handle_count(), 0);
        assert_eq!(roots.global_count(), 0);
    }

    #[test]
    fn test_global_roots() {
        let roots = RootSet::new();

        roots.add_global(Value::int(42).unwrap());
        roots.add_global(Value::bool(true));

        assert_eq!(roots.global_count(), 2);

        let mut tracer = CountingTracer::new();
        roots.trace(&mut tracer);

        assert_eq!(tracer.value_count, 2);
    }

    #[test]
    fn test_clear_globals() {
        let roots = RootSet::new();

        roots.add_global(Value::int(1).unwrap());
        roots.add_global(Value::int(2).unwrap());

        roots.clear_globals();
        assert_eq!(roots.global_count(), 0);
    }
}
