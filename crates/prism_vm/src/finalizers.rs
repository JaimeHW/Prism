//! VM-level Python finalizer scheduling.
//!
//! The current heap keeps nursery objects stable until exact moving collection is
//! available, so Python `__del__` cannot be tied to memory reclamation yet. This
//! registry tracks instances whose type has a finalizer and lets `gc.collect()`
//! run those finalizers once they are no longer reachable from Python roots.

use prism_core::Value;
use prism_gc::trace::{ObjectTracer, Tracer};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::{RuntimeObjectTracer, Trace};
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
                self.trace_object(ptr, &tracer);
            }
        }
    }

    #[inline]
    pub(crate) fn reachable(&self) -> &FxHashSet<usize> {
        &self.reachable
    }

    unsafe fn trace_object(&mut self, ptr: *const (), tracer: &RuntimeObjectTracer) {
        let header = unsafe { &*(ptr as *const ObjectHeader) };
        let type_id = header.type_id;

        if crate::stdlib::_weakref::is_reference_type_id(type_id)
            || crate::stdlib::weakref::weak_dict_kind_for_type_id(type_id).is_some()
        {
            let object = unsafe { &*(ptr as *const ShapedObject) };
            self.trace_shaped_object(type_id, object);
            return;
        }

        unsafe {
            tracer.trace_object(ptr, self);
        }
    }

    fn trace_shaped_object(
        &mut self,
        type_id: prism_runtime::object::type_obj::TypeId,
        object: &ShapedObject,
    ) {
        let is_weakref = crate::stdlib::_weakref::is_reference_type_id(type_id);
        let weakref_target = crate::stdlib::_weakref::reference_target_property();
        for (name, value) in object.iter_properties() {
            if is_weakref && name == weakref_target {
                continue;
            }
            self.trace_value(value);
        }

        if let Some(value) = object.instance_dict_value() {
            self.trace_value(value);
        }
        if let Some(dict) = object.dict_backing() {
            match crate::stdlib::weakref::weak_dict_kind_for_type_id(type_id) {
                Some(crate::stdlib::weakref::WeakDictKind::Key) => {
                    for (_, value) in dict.iter() {
                        self.trace_value(value);
                    }
                }
                Some(crate::stdlib::weakref::WeakDictKind::Value) => {
                    for (key, _) in dict.iter() {
                        self.trace_value(key);
                    }
                }
                None => dict.trace(self),
            }
        }
        if let Some(list) = object.list_backing() {
            list.trace(self);
        }
        if let Some(set) = object.set_backing() {
            set.trace(self);
        }
        if let Some(tuple) = object.tuple_backing() {
            tuple.trace(self);
        }
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
