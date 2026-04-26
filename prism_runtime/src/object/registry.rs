//! Type registry for mapping `TypeId` values to `TypeObject` metadata.
//!
//! The hot path uses a dense atomic table so common lookups stay lock-free. A
//! small overflow side table handles extremely large `TypeId` values without
//! penalizing the steady-state path.

use crate::object::type_obj::{TypeId, TypeObject};
use parking_lot::Mutex;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

const FAST_TABLE_CAPACITY: usize = 1 << 16;

fn null_type_ptr() -> *mut TypeObject {
    ptr::null_mut()
}

/// Global type registry.
///
/// Holds references to all registered type objects.
/// Built-in types are registered at startup; user types are added dynamically.
pub struct TypeRegistry {
    /// Dense lock-free lookup table for the common range of type IDs.
    fast_table: Box<[AtomicPtr<TypeObject>]>,
    /// Rare overflow storage for unusually large type IDs.
    overflow: Mutex<Vec<usize>>,
    /// Counter for generating new TypeIds.
    next_id: AtomicU32,
    /// Count of registered entries.
    registered: AtomicU32,
}

impl TypeRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        let mut fast_table = Vec::with_capacity(FAST_TABLE_CAPACITY);
        fast_table.resize_with(FAST_TABLE_CAPACITY, || AtomicPtr::new(null_type_ptr()));
        Self {
            fast_table: fast_table.into_boxed_slice(),
            overflow: Mutex::new(Vec::new()),
            next_id: AtomicU32::new(TypeId::FIRST_USER_TYPE),
            registered: AtomicU32::new(0),
        }
    }

    /// Allocate a new TypeId for a user-defined type.
    pub fn allocate_type_id(&self) -> TypeId {
        TypeId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Register a type object.
    ///
    /// # Safety
    /// The type object must have a 'static lifetime.
    pub fn register(&self, type_id: TypeId, type_obj: &'static TypeObject) {
        let index = type_id.raw() as usize;
        let new_ptr = type_obj as *const TypeObject as *mut TypeObject;

        if let Some(slot) = self.fast_table.get(index) {
            match slot.compare_exchange(
                null_type_ptr(),
                new_ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.registered.fetch_add(1, Ordering::Relaxed);
                }
                Err(existing) if existing == new_ptr => {}
                Err(existing) => {
                    let existing = unsafe { &*existing };
                    panic!(
                        "TypeId {:?} already registered to a different type object ({:?} vs {:?})",
                        type_id, existing.name, type_obj.name
                    );
                }
            }
            return;
        }

        let overflow_index = index - self.fast_table.len();
        let mut overflow = self.overflow.lock();
        if overflow.len() <= overflow_index {
            overflow.resize(overflow_index + 1, 0);
        }

        match overflow[overflow_index] {
            0 => {
                overflow[overflow_index] = new_ptr as usize;
                self.registered.fetch_add(1, Ordering::Relaxed);
            }
            ptr if ptr == new_ptr as usize => {}
            ptr => {
                let existing = unsafe { &*(ptr as *const TypeObject) };
                panic!(
                    "TypeId {:?} already registered to a different type object ({:?} vs {:?})",
                    type_id, existing.name, type_obj.name
                );
            }
        }
    }

    /// Look up a type by ID.
    #[inline]
    pub fn get(&self, type_id: TypeId) -> Option<&'static TypeObject> {
        let index = type_id.raw() as usize;
        if let Some(slot) = self.fast_table.get(index) {
            let ptr = slot.load(Ordering::Acquire);
            return if ptr.is_null() {
                None
            } else {
                Some(unsafe { &*ptr })
            };
        }

        let overflow_index = index - self.fast_table.len();
        let overflow = self.overflow.lock();
        let ptr = *overflow.get(overflow_index)?;
        if ptr == 0 {
            None
        } else {
            Some(unsafe { &*(ptr as *const TypeObject) })
        }
    }

    /// Check if a type is registered.
    #[inline]
    pub fn contains(&self, type_id: TypeId) -> bool {
        self.get(type_id).is_some()
    }

    /// Get the number of registered types.
    pub fn len(&self) -> usize {
        self.registered.load(Ordering::Relaxed) as usize
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global type registry singleton.
static GLOBAL_REGISTRY: OnceLock<TypeRegistry> = OnceLock::new();

/// Get the global type registry.
pub fn global_registry() -> &'static TypeRegistry {
    GLOBAL_REGISTRY.get_or_init(TypeRegistry::new)
}

#[cfg(test)]
mod tests;
