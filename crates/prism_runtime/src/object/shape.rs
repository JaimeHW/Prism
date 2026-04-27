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

use parking_lot::RwLock;
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TransitionKey {
    name: InternedString,
    flags: PropertyFlags,
}

impl TransitionKey {
    #[inline]
    fn new(name: InternedString, flags: PropertyFlags) -> Self {
        Self { name, flags }
    }
}

#[derive(Debug)]
struct ShapeLayout {
    descriptors: Box<[PropertyDescriptor]>,
    descriptor_indices: FxHashMap<InternedString, u16>,
}

impl ShapeLayout {
    fn empty() -> Arc<Self> {
        Arc::new(Self {
            descriptors: Box::new([]),
            descriptor_indices: FxHashMap::default(),
        })
    }

    fn extend(parent: &Arc<Self>, descriptor: PropertyDescriptor) -> Arc<Self> {
        let mut descriptors = Vec::with_capacity(parent.descriptors.len() + 1);
        descriptors.extend_from_slice(parent.descriptors.as_ref());
        let descriptor_index = descriptors.len() as u16;
        descriptors.push(descriptor.clone());

        let mut descriptor_indices = parent.descriptor_indices.clone();
        descriptor_indices.insert(descriptor.name.clone(), descriptor_index);

        Arc::new(Self {
            descriptors: descriptors.into_boxed_slice(),
            descriptor_indices,
        })
    }

    #[inline]
    fn lookup_interned(&self, name: &InternedString) -> Option<u16> {
        self.descriptor_indices
            .get(name)
            .map(|&index| self.descriptors[index as usize].slot_index)
    }

    #[inline]
    fn get_descriptor_interned(&self, name: &InternedString) -> Option<&PropertyDescriptor> {
        self.descriptor_indices
            .get(name)
            .map(|&index| &self.descriptors[index as usize])
    }
}

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

    /// Immutable descriptor table for O(1) baseline property lookup.
    layout: Arc<ShapeLayout>,

    /// Total number of properties in this shape chain.
    property_count: u16,

    /// Number of inline slots used.
    inline_count: u16,

    /// Transitions to child shapes (lazily populated).
    /// Key: property name + flags, Value: child shape.
    transitions: RwLock<FxHashMap<TransitionKey, Arc<Shape>>>,
}

impl Shape {
    /// Create the empty shape (root of all shape trees).
    pub fn empty() -> Arc<Self> {
        Arc::new(Self {
            id: ShapeId::EMPTY,
            parent: None,
            property: None,
            layout: ShapeLayout::empty(),
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
        let descriptor = PropertyDescriptor::new(name, slot_index, flags);
        let layout = ShapeLayout::extend(&parent.layout, descriptor.clone());

        Arc::new(Self {
            id,
            parent: Some(parent),
            property: Some(descriptor),
            layout,
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
    /// String-based lookups are kept off the hot path; interned lookups use the
    /// prebuilt descriptor index.
    pub fn lookup(&self, name: &str) -> Option<u16> {
        self.layout
            .descriptors
            .iter()
            .find(|descriptor| descriptor.name.as_str() == name)
            .map(|descriptor| descriptor.slot_index)
    }

    /// Lookup a property by interned name (faster comparison).
    #[inline]
    pub fn lookup_interned(&self, name: &InternedString) -> Option<u16> {
        self.layout.lookup_interned(name)
    }

    /// Get full property descriptor by name.
    pub fn get_descriptor(&self, name: &str) -> Option<&PropertyDescriptor> {
        self.layout
            .descriptors
            .iter()
            .find(|descriptor| descriptor.name.as_str() == name)
    }

    /// Get a full property descriptor by interned name.
    #[inline]
    pub fn get_descriptor_interned(&self, name: &InternedString) -> Option<&PropertyDescriptor> {
        self.layout.get_descriptor_interned(name)
    }

    /// Collect all property names in definition order.
    pub fn property_names(&self) -> Vec<InternedString> {
        self.layout
            .descriptors
            .iter()
            .map(|descriptor| descriptor.name.clone())
            .collect()
    }

    /// Collect all property descriptors in definition order.
    pub fn all_descriptors(&self) -> Vec<PropertyDescriptor> {
        self.layout.descriptors.to_vec()
    }

    /// Check if a transition exists for the given property.
    pub fn has_transition(&self, name: &InternedString) -> bool {
        self.transitions.read().keys().any(|key| &key.name == name)
    }

    /// Check if a transition exists for the exact property descriptor key.
    pub fn has_transition_with_flags(&self, name: &InternedString, flags: PropertyFlags) -> bool {
        self.transitions
            .read()
            .contains_key(&TransitionKey::new(name.clone(), flags))
    }

    /// Get the default-attribute transition for a property name (if any).
    pub fn get_transition(&self, name: &InternedString) -> Option<Arc<Shape>> {
        self.get_transition_with_flags(name, PropertyFlags::default())
    }

    /// Get an existing transition for the exact property descriptor key.
    pub fn get_transition_with_flags(
        &self,
        name: &InternedString,
        flags: PropertyFlags,
    ) -> Option<Arc<Shape>> {
        self.transitions
            .read()
            .get(&TransitionKey::new(name.clone(), flags))
            .cloned()
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
        let key = TransitionKey::new(name.clone(), flags);

        // Fast path: check if transition already exists
        if let Some(existing) = from.get_transition_with_flags(&name, flags) {
            return existing;
        }

        // Slow path: canonicalize under the transition-table write lock so
        // concurrent creators cannot materialize duplicate child shapes.
        let mut transitions = from.transitions.write();
        if let Some(existing) = transitions.get(&key) {
            return Arc::clone(existing);
        }

        let id = ShapeId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let new_shape = Shape::with_property(Arc::clone(from), name, flags, id);
        transitions.insert(key, Arc::clone(&new_shape));
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
