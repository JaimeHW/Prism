//! Shape system for hidden class optimization.
//!
//! Implements V8-style hidden classes (called Shapes) for O(1) property access.
//!
//! # Architecture
//!
//! Objects with the same sequence of property additions share a Shape. Each Shape
//! describes the layout of properties in the object, enabling inline caching.
//!
//! ## Shape Transitions
//!
//! When a property is added to an object, instead of doing expensive dictionary
//! lookups, we transition to a new Shape that includes the new property. Shapes
//! form a transition tree:
//!
//! ```text
//!     EmptyShape
//!         |
//!     +---+---+
//!     |       |
//!   "x"     "y"
//!     |       |
//!  Shape1  Shape2
//!     |
//!   "y"
//!     |
//!  Shape3 (has both x and y)
//! ```
//!
//! ## Property Descriptors
//!
//! Each Shape contains a descriptor for the property it adds, including:
//! - Property name (interned)
//! - Slot offset in the object's inline storage
//! - Property attributes (writable, enumerable, configurable)
//!
//! ## Inline Property Storage
//!
//! Objects store their first N properties directly in fixed slots (inline storage),
//! avoiding dictionary overhead for common cases. When inline storage is exhausted,
//! properties spill to a backing dictionary.

use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

// =============================================================================
// Property Attributes
// =============================================================================

bitflags::bitflags! {
    /// Property descriptor attributes.
    ///
    /// Follows ECMAScript/Python property semantics.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PropertyFlags: u8 {
        /// Property value can be changed.
        const WRITABLE = 1 << 0;
        /// Property appears in for-in loops.
        const ENUMERABLE = 1 << 1;
        /// Property can be deleted or have attributes changed.
        const CONFIGURABLE = 1 << 2;
        /// Property is a data property (vs accessor).
        const DATA = 1 << 3;
        /// Property has __get__.
        const HAS_GETTER = 1 << 4;
        /// Property has __set__.
        const HAS_SETTER = 1 << 5;
    }
}

impl Default for PropertyFlags {
    /// Default Python attribute: writable, enumerable, configurable, data.
    #[inline]
    fn default() -> Self {
        Self::WRITABLE | Self::ENUMERABLE | Self::CONFIGURABLE | Self::DATA
    }
}

impl PropertyFlags {
    /// Create read-only property flags.
    #[inline]
    pub const fn read_only() -> Self {
        Self::ENUMERABLE.union(Self::CONFIGURABLE).union(Self::DATA)
    }

    /// Create non-enumerable property flags (for special attributes).
    #[inline]
    pub const fn hidden() -> Self {
        Self::WRITABLE.union(Self::CONFIGURABLE).union(Self::DATA)
    }
}

// =============================================================================
// Property Descriptor
// =============================================================================

/// Describes a single property in a Shape.
///
/// Contains all information needed for O(1) property access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyDescriptor {
    /// Property name (interned for fast comparison).
    pub name: InternedString,
    /// Slot index in inline storage (0-based).
    pub slot_index: u16,
    /// Property attribute flags.
    pub flags: PropertyFlags,
}

impl PropertyDescriptor {
    /// Create a new data property descriptor.
    #[inline]
    pub fn new(name: InternedString, slot_index: u16, flags: PropertyFlags) -> Self {
        Self {
            name,
            slot_index,
            flags,
        }
    }

    /// Create a standard writable property.
    #[inline]
    pub fn writable(name: InternedString, slot_index: u16) -> Self {
        Self::new(name, slot_index, PropertyFlags::default())
    }

    /// Check if property is writable.
    #[inline]
    pub fn is_writable(&self) -> bool {
        self.flags.contains(PropertyFlags::WRITABLE)
    }

    /// Check if property is enumerable.
    #[inline]
    pub fn is_enumerable(&self) -> bool {
        self.flags.contains(PropertyFlags::ENUMERABLE)
    }

    /// Check if property is configurable.
    #[inline]
    pub fn is_configurable(&self) -> bool {
        self.flags.contains(PropertyFlags::CONFIGURABLE)
    }

    /// Check if property is a data property.
    #[inline]
    pub fn is_data(&self) -> bool {
        self.flags.contains(PropertyFlags::DATA)
    }
}

// =============================================================================
// Shape ID
// =============================================================================

/// Unique identifier for a Shape.
///
/// Used for fast comparison and IC keying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ShapeId(pub u32);

impl ShapeId {
    /// The empty shape ID (no properties).
    pub const EMPTY: Self = Self(0);

    /// Check if this is the empty shape.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Get raw value.
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

// =============================================================================
// Shape
// =============================================================================

/// Maximum number of inline property slots.
///
/// Objects with more properties spill to dictionary.
/// 8 slots = 64 bytes of inline storage (8 * 8-byte Values).
pub const MAX_INLINE_SLOTS: usize = 8;

/// A Shape describes the property layout of objects.
///
/// Objects with the same property sequence share a Shape, enabling:
/// - O(1) property access via cached slot offsets
/// - Efficient IC invalidation on shape transitions
///
/// Shapes are immutable once created and form a transition tree.
#[derive(Debug)]
pub struct Shape {
    /// Unique identifier for this shape.
    id: ShapeId,

    /// Parent shape (None for empty shape).
    parent: Option<Arc<Shape>>,

    /// Property added by this shape transition.
    /// None for the empty shape.
    property: Option<PropertyDescriptor>,

    /// Total number of properties in this shape chain.
    property_count: u16,

    /// Number of inline slots used.
    inline_count: u16,

    /// Transitions to child shapes (lazily populated).
    /// Key: property name, Value: child shape.
    /// Uses RwLock for thread-safe lazy initialization.
    transitions: RwLock<FxHashMap<InternedString, Arc<Shape>>>,
}

impl Shape {
    /// Create the empty shape (root of all shape trees).
    pub fn empty() -> Arc<Self> {
        Arc::new(Self {
            id: ShapeId::EMPTY,
            parent: None,
            property: None,
            property_count: 0,
            inline_count: 0,
            transitions: RwLock::new(FxHashMap::default()),
        })
    }

    /// Create a new shape by adding a property to the parent.
    fn with_property(
        parent: Arc<Shape>,
        name: InternedString,
        flags: PropertyFlags,
        id: ShapeId,
    ) -> Arc<Self> {
        let slot_index = parent.inline_count;
        let parent_property_count = parent.property_count;
        let inline_count = if (slot_index as usize) < MAX_INLINE_SLOTS {
            slot_index + 1
        } else {
            // Beyond inline storage - would spill to dictionary
            parent.inline_count
        };

        Arc::new(Self {
            id,
            parent: Some(parent),
            property: Some(PropertyDescriptor::new(name, slot_index, flags)),
            property_count: parent_property_count + 1,
            inline_count,
            transitions: RwLock::new(FxHashMap::default()),
        })
    }

    /// Get the shape ID.
    #[inline]
    pub fn id(&self) -> ShapeId {
        self.id
    }

    /// Get the parent shape.
    #[inline]
    pub fn parent(&self) -> Option<&Arc<Shape>> {
        self.parent.as_ref()
    }

    /// Get the property descriptor for this shape's added property.
    #[inline]
    pub fn property(&self) -> Option<&PropertyDescriptor> {
        self.property.as_ref()
    }

    /// Get total property count.
    #[inline]
    pub fn property_count(&self) -> u16 {
        self.property_count
    }

    /// Get number of inline slots used.
    #[inline]
    pub fn inline_count(&self) -> u16 {
        self.inline_count
    }

    /// Check if all properties fit in inline storage.
    #[inline]
    pub fn is_fully_inline(&self) -> bool {
        (self.property_count as usize) <= MAX_INLINE_SLOTS
    }

    /// Check if this is the empty shape.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// Lookup a property by name, traversing the shape chain.
    ///
    /// Returns the slot index if found.
    /// O(n) where n is property count, but short chains are fast.
    pub fn lookup(&self, name: &str) -> Option<u16> {
        let mut current = self;
        loop {
            if let Some(prop) = &current.property {
                if prop.name.as_str() == name {
                    return Some(prop.slot_index);
                }
            }
            match &current.parent {
                Some(parent) => current = parent.as_ref(),
                None => return None,
            }
        }
    }

    /// Lookup a property by interned name (faster comparison).
    #[inline]
    pub fn lookup_interned(&self, name: &InternedString) -> Option<u16> {
        let mut current = self;
        loop {
            if let Some(prop) = &current.property {
                // Interned string comparison is pointer equality
                if &prop.name == name {
                    return Some(prop.slot_index);
                }
            }
            match &current.parent {
                Some(parent) => current = parent.as_ref(),
                None => return None,
            }
        }
    }

    /// Get full property descriptor by name.
    pub fn get_descriptor(&self, name: &str) -> Option<&PropertyDescriptor> {
        let mut current = self;
        loop {
            if let Some(prop) = &current.property {
                if prop.name.as_str() == name {
                    return Some(prop);
                }
            }
            match &current.parent {
                Some(parent) => current = parent.as_ref(),
                None => return None,
            }
        }
    }

    /// Collect all property names in definition order.
    pub fn property_names(&self) -> Vec<InternedString> {
        let mut names = Vec::with_capacity(self.property_count as usize);
        self.collect_names(&mut names);
        names
    }

    /// Collect names by walking parent chain (reverses order).
    fn collect_names(&self, names: &mut Vec<InternedString>) {
        if let Some(parent) = &self.parent {
            parent.collect_names(names);
        }
        if let Some(prop) = &self.property {
            names.push(prop.name.clone());
        }
    }

    /// Collect all property descriptors in definition order.
    pub fn all_descriptors(&self) -> Vec<PropertyDescriptor> {
        let mut descriptors = Vec::with_capacity(self.property_count as usize);
        self.collect_descriptors(&mut descriptors);
        descriptors
    }

    fn collect_descriptors(&self, descriptors: &mut Vec<PropertyDescriptor>) {
        if let Some(parent) = &self.parent {
            parent.collect_descriptors(descriptors);
        }
        if let Some(prop) = &self.property {
            descriptors.push(prop.clone());
        }
    }

    /// Check if a transition exists for the given property.
    pub fn has_transition(&self, name: &InternedString) -> bool {
        self.transitions
            .read()
            .expect("Shape transitions lock poisoned")
            .contains_key(name)
    }

    /// Get an existing transition (if any).
    pub fn get_transition(&self, name: &InternedString) -> Option<Arc<Shape>> {
        self.transitions
            .read()
            .expect("Shape transitions lock poisoned")
            .get(name)
            .cloned()
    }

    /// Add a transition (internal use by ShapeRegistry).
    fn add_transition(&self, name: InternedString, shape: Arc<Shape>) {
        self.transitions
            .write()
            .expect("Shape transitions lock poisoned")
            .insert(name, shape);
    }
}

// =============================================================================
// Shape Registry
// =============================================================================

/// Global registry for Shape management.
///
/// Thread-safe, manages shape creation and transition caching.
/// Uses atomic counter for unique shape IDs.
pub struct ShapeRegistry {
    /// Counter for generating unique shape IDs.
    next_id: AtomicU32,

    /// The empty shape (shared root).
    empty_shape: Arc<Shape>,
}

impl ShapeRegistry {
    /// Create a new shape registry.
    pub fn new() -> Self {
        Self {
            // ID 0 is reserved for empty shape
            next_id: AtomicU32::new(1),
            empty_shape: Shape::empty(),
        }
    }

    /// Get the empty shape.
    #[inline]
    pub fn empty_shape(&self) -> Arc<Shape> {
        Arc::clone(&self.empty_shape)
    }

    /// Transition to a new shape by adding a property.
    ///
    /// If a transition already exists, returns the cached shape.
    /// Otherwise, creates a new shape and caches the transition.
    pub fn transition(
        &self,
        from: &Arc<Shape>,
        name: InternedString,
        flags: PropertyFlags,
    ) -> Arc<Shape> {
        // Fast path: check if transition already exists
        if let Some(existing) = from.get_transition(&name) {
            return existing;
        }

        // Slow path: create new shape
        let id = ShapeId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let new_shape = Shape::with_property(Arc::clone(from), name.clone(), flags, id);

        // Cache the transition
        from.add_transition(name, Arc::clone(&new_shape));

        new_shape
    }

    /// Transition with default property flags.
    #[inline]
    pub fn transition_default(&self, from: &Arc<Shape>, name: InternedString) -> Arc<Shape> {
        self.transition(from, name, PropertyFlags::default())
    }

    /// Get number of shapes created (including empty).
    pub fn shape_count(&self) -> u32 {
        self.next_id.load(Ordering::Relaxed)
    }

    /// Get registry statistics.
    pub fn stats(&self) -> ShapeStats {
        ShapeStats {
            total_shapes: self.shape_count(),
        }
    }
}

/// Statistics about shape registry usage.
#[derive(Debug, Clone, Copy)]
pub struct ShapeStats {
    /// Total number of shapes created.
    pub total_shapes: u32,
}

impl Default for ShapeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Global Registry Access
// =============================================================================

use std::sync::OnceLock;

/// Global shape registry instance.
static SHAPE_REGISTRY: OnceLock<ShapeRegistry> = OnceLock::new();

/// Get the global shape registry.
#[inline]
pub fn shape_registry() -> &'static ShapeRegistry {
    SHAPE_REGISTRY.get_or_init(ShapeRegistry::new)
}

/// Initialize the global shape registry (call at startup).
pub fn init_shape_registry() {
    let _ = shape_registry();
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn intern(s: &str) -> InternedString {
        prism_core::intern::intern(s)
    }

    // -------------------------------------------------------------------------
    // PropertyFlags Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_property_flags_default() {
        let flags = PropertyFlags::default();
        assert!(flags.contains(PropertyFlags::WRITABLE));
        assert!(flags.contains(PropertyFlags::ENUMERABLE));
        assert!(flags.contains(PropertyFlags::CONFIGURABLE));
        assert!(flags.contains(PropertyFlags::DATA));
    }

    #[test]
    fn test_property_flags_read_only() {
        let flags = PropertyFlags::read_only();
        assert!(!flags.contains(PropertyFlags::WRITABLE));
        assert!(flags.contains(PropertyFlags::ENUMERABLE));
        assert!(flags.contains(PropertyFlags::CONFIGURABLE));
    }

    #[test]
    fn test_property_flags_hidden() {
        let flags = PropertyFlags::hidden();
        assert!(flags.contains(PropertyFlags::WRITABLE));
        assert!(!flags.contains(PropertyFlags::ENUMERABLE));
        assert!(flags.contains(PropertyFlags::CONFIGURABLE));
    }

    #[test]
    fn test_property_flags_combinations() {
        let flags = PropertyFlags::WRITABLE | PropertyFlags::ENUMERABLE;
        assert!(flags.contains(PropertyFlags::WRITABLE));
        assert!(flags.contains(PropertyFlags::ENUMERABLE));
        assert!(!flags.contains(PropertyFlags::CONFIGURABLE));
    }

    // -------------------------------------------------------------------------
    // PropertyDescriptor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_property_descriptor_new() {
        let desc = PropertyDescriptor::new(intern("x"), 0, PropertyFlags::default());
        assert_eq!(desc.name.as_str(), "x");
        assert_eq!(desc.slot_index, 0);
        assert!(desc.is_writable());
        assert!(desc.is_enumerable());
        assert!(desc.is_configurable());
        assert!(desc.is_data());
    }

    #[test]
    fn test_property_descriptor_writable() {
        let desc = PropertyDescriptor::writable(intern("foo"), 3);
        assert_eq!(desc.name.as_str(), "foo");
        assert_eq!(desc.slot_index, 3);
        assert!(desc.is_writable());
    }

    #[test]
    fn test_property_descriptor_read_only() {
        let desc = PropertyDescriptor::new(intern("const"), 0, PropertyFlags::read_only());
        assert!(!desc.is_writable());
        assert!(desc.is_enumerable());
    }

    // -------------------------------------------------------------------------
    // ShapeId Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shape_id_empty() {
        assert!(ShapeId::EMPTY.is_empty());
        assert!(!ShapeId(1).is_empty());
    }

    #[test]
    fn test_shape_id_raw() {
        assert_eq!(ShapeId(42).raw(), 42);
    }

    #[test]
    fn test_shape_id_equality() {
        assert_eq!(ShapeId(1), ShapeId(1));
        assert_ne!(ShapeId(1), ShapeId(2));
    }

    // -------------------------------------------------------------------------
    // Shape Tests - Basic
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_shape() {
        let empty = Shape::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.id(), ShapeId::EMPTY);
        assert!(empty.parent().is_none());
        assert!(empty.property().is_none());
        assert_eq!(empty.property_count(), 0);
        assert_eq!(empty.inline_count(), 0);
        assert!(empty.is_fully_inline());
    }

    #[test]
    fn test_shape_single_property() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let shape_x = registry.transition_default(&empty, intern("x"));

        assert!(!shape_x.is_empty());
        assert!(shape_x.parent().is_some());
        assert_eq!(shape_x.property_count(), 1);
        assert_eq!(shape_x.inline_count(), 1);

        let prop = shape_x.property().unwrap();
        assert_eq!(prop.name.as_str(), "x");
        assert_eq!(prop.slot_index, 0);
    }

    #[test]
    fn test_shape_multiple_properties() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let shape_x = registry.transition_default(&empty, intern("x"));
        let shape_xy = registry.transition_default(&shape_x, intern("y"));
        let shape_xyz = registry.transition_default(&shape_xy, intern("z"));

        assert_eq!(shape_xyz.property_count(), 3);
        assert_eq!(shape_xyz.inline_count(), 3);

        let z_prop = shape_xyz.property().unwrap();
        assert_eq!(z_prop.name.as_str(), "z");
        assert_eq!(z_prop.slot_index, 2);
    }

    // -------------------------------------------------------------------------
    // Shape Tests - Lookup
    // -------------------------------------------------------------------------

    #[test]
    fn test_shape_lookup_found() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let shape = registry.transition_default(&empty, intern("x"));
        let shape = registry.transition_default(&shape, intern("y"));
        let shape = registry.transition_default(&shape, intern("z"));

        assert_eq!(shape.lookup("x"), Some(0));
        assert_eq!(shape.lookup("y"), Some(1));
        assert_eq!(shape.lookup("z"), Some(2));
    }

    #[test]
    fn test_shape_lookup_not_found() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let shape = registry.transition_default(&empty, intern("x"));

        assert_eq!(shape.lookup("y"), None);
        assert_eq!(shape.lookup("not_exists"), None);
    }

    #[test]
    fn test_shape_lookup_empty() {
        let empty = Shape::empty();
        assert_eq!(empty.lookup("anything"), None);
    }

    #[test]
    fn test_shape_lookup_interned() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let name = intern("property");
        let shape = registry.transition_default(&empty, name.clone());

        assert_eq!(shape.lookup_interned(&name), Some(0));
        assert_eq!(shape.lookup_interned(&intern("other")), None);
    }

    #[test]
    fn test_shape_get_descriptor() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let flags = PropertyFlags::hidden();
        let shape = registry.transition(&empty, intern("_private"), flags);

        let desc = shape.get_descriptor("_private").unwrap();
        assert_eq!(desc.name.as_str(), "_private");
        assert!(!desc.is_enumerable());
        assert!(desc.is_writable());
    }

    // -------------------------------------------------------------------------
    // Shape Tests - Property Collection
    // -------------------------------------------------------------------------

    #[test]
    fn test_property_names_order() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let shape = registry.transition_default(&empty, intern("first"));
        let shape = registry.transition_default(&shape, intern("second"));
        let shape = registry.transition_default(&shape, intern("third"));

        let names = shape.property_names();
        assert_eq!(names.len(), 3);
        assert_eq!(names[0].as_str(), "first");
        assert_eq!(names[1].as_str(), "second");
        assert_eq!(names[2].as_str(), "third");
    }

    #[test]
    fn test_all_descriptors() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let shape = registry.transition_default(&empty, intern("a"));
        let shape = registry.transition_default(&shape, intern("b"));

        let descriptors = shape.all_descriptors();
        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].name.as_str(), "a");
        assert_eq!(descriptors[0].slot_index, 0);
        assert_eq!(descriptors[1].name.as_str(), "b");
        assert_eq!(descriptors[1].slot_index, 1);
    }

    // -------------------------------------------------------------------------
    // Shape Tests - Transitions
    // -------------------------------------------------------------------------

    #[test]
    fn test_transition_caching() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let name = intern("x");

        let shape1 = registry.transition_default(&empty, name.clone());
        let shape2 = registry.transition_default(&empty, name.clone());

        // Should return the same cached shape
        assert!(Arc::ptr_eq(&shape1, &shape2));
    }

    #[test]
    fn test_transition_different_properties() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();

        let shape_x = registry.transition_default(&empty, intern("x"));
        let shape_y = registry.transition_default(&empty, intern("y"));

        // Different properties -> different shapes
        assert!(!Arc::ptr_eq(&shape_x, &shape_y));
        assert_ne!(shape_x.id(), shape_y.id());
    }

    #[test]
    fn test_transition_branching() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let shape_x = registry.transition_default(&empty, intern("x"));

        // Two different paths from shape_x
        let shape_xy = registry.transition_default(&shape_x, intern("y"));
        let shape_xz = registry.transition_default(&shape_x, intern("z"));

        assert_ne!(shape_xy.id(), shape_xz.id());
        assert_eq!(shape_xy.lookup("y"), Some(1));
        assert_eq!(shape_xz.lookup("z"), Some(1));
    }

    #[test]
    fn test_has_transition() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let name = intern("x");

        assert!(!empty.has_transition(&name));
        let _shape = registry.transition_default(&empty, name.clone());
        assert!(empty.has_transition(&name));
    }

    #[test]
    fn test_get_transition() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let name = intern("x");

        assert!(empty.get_transition(&name).is_none());
        let shape = registry.transition_default(&empty, name.clone());
        let cached = empty.get_transition(&name).unwrap();
        assert!(Arc::ptr_eq(&shape, &cached));
    }

    // -------------------------------------------------------------------------
    // Shape Tests - Inline Storage Limits
    // -------------------------------------------------------------------------

    #[test]
    fn test_inline_storage_grows() {
        let registry = ShapeRegistry::new();
        let mut shape = registry.empty_shape();

        for i in 0..MAX_INLINE_SLOTS {
            shape = registry.transition_default(&shape, intern(&format!("prop{}", i)));
            assert_eq!(shape.inline_count() as usize, i + 1);
            assert!(shape.is_fully_inline());
        }
    }

    #[test]
    fn test_inline_storage_limit() {
        let registry = ShapeRegistry::new();
        let mut shape = registry.empty_shape();

        // Fill inline storage
        for i in 0..MAX_INLINE_SLOTS {
            shape = registry.transition_default(&shape, intern(&format!("p{}", i)));
        }

        assert_eq!(shape.inline_count() as usize, MAX_INLINE_SLOTS);
        assert!(shape.is_fully_inline());

        // Add one more - should spill
        shape = registry.transition_default(&shape, intern("overflow"));
        assert_eq!(shape.property_count() as usize, MAX_INLINE_SLOTS + 1);
        // Inline count doesn't increase beyond max
        assert_eq!(shape.inline_count() as usize, MAX_INLINE_SLOTS);
        assert!(!shape.is_fully_inline());
    }

    // -------------------------------------------------------------------------
    // ShapeRegistry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_new() {
        let registry = ShapeRegistry::new();
        assert_eq!(registry.shape_count(), 1); // Empty shape counts
    }

    #[test]
    fn test_registry_unique_ids() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();

        let ids: Vec<_> = (0..100)
            .map(|i| {
                let shape = registry.transition_default(&empty, intern(&format!("p{}", i)));
                shape.id()
            })
            .collect();

        // All IDs should be unique
        let mut unique_ids = ids.clone();
        unique_ids.sort_by_key(|id| id.raw());
        unique_ids.dedup();
        assert_eq!(ids.len(), unique_ids.len());
    }

    #[test]
    fn test_registry_shape_count() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();

        let initial = registry.shape_count();
        let _s1 = registry.transition_default(&empty, intern("a"));
        assert_eq!(registry.shape_count(), initial + 1);

        let _s2 = registry.transition_default(&empty, intern("b"));
        assert_eq!(registry.shape_count(), initial + 2);

        // Cached transition doesn't increase count
        let _s1_again = registry.transition_default(&empty, intern("a"));
        assert_eq!(registry.shape_count(), initial + 2);
    }

    // -------------------------------------------------------------------------
    // Global Registry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_global_registry_access() {
        init_shape_registry();
        let registry = shape_registry();
        let empty = registry.empty_shape();
        assert!(empty.is_empty());
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shape_thread_safety() {
        use std::thread;

        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let empty_clone = Arc::clone(&empty);
                let name = intern(&format!("thread_prop_{}", i));
                thread::spawn(move || {
                    // Can't use registry across threads, but shapes should be thread-safe
                    empty_clone.lookup(&format!("prop{}", i))
                })
            })
            .collect();

        for handle in handles {
            let _ = handle.join().unwrap();
        }
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_property_name() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let shape = registry.transition_default(&empty, intern(""));

        assert_eq!(shape.lookup(""), Some(0));
        assert_eq!(shape.property().unwrap().name.as_str(), "");
    }

    #[test]
    fn test_unicode_property_names() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();
        let shape = registry.transition_default(&empty, intern("名前"));
        let shape = registry.transition_default(&shape, intern("привет"));
        let shape = registry.transition_default(&shape, intern("🚀"));

        assert_eq!(shape.lookup("名前"), Some(0));
        assert_eq!(shape.lookup("привет"), Some(1));
        assert_eq!(shape.lookup("🚀"), Some(2));
    }

    #[test]
    fn test_long_property_chain() {
        let registry = ShapeRegistry::new();
        let mut shape = registry.empty_shape();

        // Create a long chain
        for i in 0..50 {
            shape = registry.transition_default(&shape, intern(&format!("property_{}", i)));
        }

        // Should still be able to look up all properties
        for i in 0..50 {
            let name = format!("property_{}", i);
            assert!(shape.lookup(&name).is_some(), "Failed to find {}", name);
        }
    }

    #[test]
    fn test_property_flags_preserved() {
        let registry = ShapeRegistry::new();
        let empty = registry.empty_shape();

        let read_only = registry.transition(&empty, intern("ro"), PropertyFlags::read_only());
        let hidden = registry.transition(&empty, intern("hidden"), PropertyFlags::hidden());

        let ro_desc = read_only.get_descriptor("ro").unwrap();
        assert!(!ro_desc.is_writable());

        let hidden_desc = hidden.get_descriptor("hidden").unwrap();
        assert!(!hidden_desc.is_enumerable());
    }
}
