//! VM-level Python finalizer scheduling.
//!
//! The current heap keeps nursery objects stable until exact moving collection is
//! available, so Python `__del__` cannot be tied to memory reclamation yet. This
//! registry tracks instances whose type has a finalizer and lets `gc.collect()`
//! run those finalizers once they are no longer reachable from Python roots.

use prism_core::Value;
use prism_gc::trace::{ObjectTracer, Tracer};
use prism_runtime::RuntimeObjectTracer;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Default)]
pub(crate) struct FinalizerRegistry {
    pending: FxHashMap<usize, Value>,
    draining: bool,
}

impl FinalizerRegistry {
    #[inline]
    pub(crate) fn register(&mut self, value: Value) {
        let Some(ptr) = value.as_object_ptr() else {
            return;
        };

        self.pending.entry(ptr as usize).or_insert(value);
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    #[inline]
    pub(crate) fn contains(&self, value: Value) -> bool {
        value
            .as_object_ptr()
            .is_some_and(|ptr| self.pending.contains_key(&(ptr as usize)))
    }

    #[inline]
    pub(crate) fn intersects(&self, reachable: &FxHashSet<usize>) -> bool {
        reachable.iter().any(|addr| self.pending.contains_key(addr))
    }

    #[inline]
    pub(crate) fn begin_drain(&mut self) -> bool {
        if self.draining {
            return false;
        }
        self.draining = true;
        true
    }

    #[inline]
    pub(crate) fn finish_drain(&mut self) {
        self.draining = false;
    }

    pub(crate) fn take_unreachable(&mut self, reachable: &FxHashSet<usize>) -> Vec<Value> {
        let mut unreachable = Vec::new();
        self.pending.retain(|addr, value| {
            if reachable.contains(addr) {
                true
            } else {
                unreachable.push(*value);
                false
            }
        });
        unreachable
    }
}

pub(crate) struct ReachabilityTracer {
    reachable: FxHashSet<usize>,
    worklist: Vec<*const ()>,
}

impl ReachabilityTracer {
    pub(crate) fn new() -> Self {
        Self {
            reachable: FxHashSet::default(),
            worklist: Vec::with_capacity(256),
        }
    }

    pub(crate) fn drain_object_graph(&mut self) {
        let tracer = RuntimeObjectTracer::new();
        while let Some(ptr) = self.worklist.pop() {
            unsafe {
                tracer.trace_object(ptr, self);
            }
        }
    }

    #[inline]
    pub(crate) fn reachable(&self) -> &FxHashSet<usize> {
        &self.reachable
    }
}

impl Tracer for ReachabilityTracer {
    #[inline]
    fn trace_value(&mut self, value: Value) {
        if let Some(ptr) = value.as_object_ptr() {
            self.trace_ptr(ptr);
        }
    }

    #[inline]
    fn trace_ptr(&mut self, ptr: *const ()) {
        if ptr.is_null() {
            return;
        }

        if self.reachable.insert(ptr as usize) {
            self.worklist.push(ptr);
        }
    }
}
