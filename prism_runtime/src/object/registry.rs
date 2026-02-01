//! Type registry for mapping TypeId to TypeObject.
//!
//! Provides O(1) lookup of type objects by TypeId.

use crate::object::type_obj::{TypeId, TypeObject};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Global type registry.
///
/// Holds references to all registered type objects.
/// Built-in types are registered at startup; user types are added dynamically.
pub struct TypeRegistry {
    /// Map from TypeId to TypeObject.
    types: RwLock<HashMap<TypeId, &'static TypeObject>>,
    /// Counter for generating new TypeIds.
    next_id: AtomicU32,
}

impl TypeRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            types: RwLock::new(HashMap::new()),
            next_id: AtomicU32::new(TypeId::FIRST_USER_TYPE),
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
        let mut types = self.types.write();
        types.insert(type_id, type_obj);
    }

    /// Look up a type by ID.
    #[inline]
    pub fn get(&self, type_id: TypeId) -> Option<&'static TypeObject> {
        let types = self.types.read();
        types.get(&type_id).copied()
    }

    /// Check if a type is registered.
    #[inline]
    pub fn contains(&self, type_id: TypeId) -> bool {
        let types = self.types.read();
        types.contains_key(&type_id)
    }

    /// Get the number of registered types.
    pub fn len(&self) -> usize {
        let types = self.types.read();
        types.len()
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

// =============================================================================
// Global Registry Access
// =============================================================================

use std::sync::OnceLock;

/// Global type registry singleton.
static GLOBAL_REGISTRY: OnceLock<TypeRegistry> = OnceLock::new();

/// Get the global type registry.
pub fn global_registry() -> &'static TypeRegistry {
    GLOBAL_REGISTRY.get_or_init(TypeRegistry::new)
}

/// Initialize the type registry with built-in types.
///
/// This should be called once at startup.
pub fn init_builtin_types() {
    let registry = global_registry();

    // Built-in types will be registered here
    // For now, we just ensure the registry is initialized
    let _ = registry;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = TypeRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_allocate_type_id() {
        let registry = TypeRegistry::new();
        let id1 = registry.allocate_type_id();
        let id2 = registry.allocate_type_id();
        assert_eq!(id1.raw(), 256);
        assert_eq!(id2.raw(), 257);
        assert!(!id1.is_builtin());
    }
}
