//! Python class object implementation.
//!
//! A `PyClassObject` represents a user-defined Python class. It contains:
//! - The class name
//! - Base classes
//! - Method Resolution Order (MRO)
//! - Class attributes (methods, class variables)
//! - Metaclass reference
//!
//! # Architecture
//!
//! ```text
//! PyClassObject
//! ├── ObjectHeader (16 bytes)
//! ├── name: InternedString (8 bytes)
//! ├── bases: SmallVec<ClassId; 2> (16 bytes inline)
//! ├── mro: SmallVec<ClassId; 8> (cached MRO)
//! ├── type_id: TypeId (4 bytes, unique for this class)
//! ├── flags: ClassFlags (4 bytes)
//! ├── dict: ClassDict (class attributes)
//! └── slots: TypeSlots (method dispatch table)
//! ```
//!
//! # Performance
//!
//! - MRO is cached after first computation using SmallVec
//! - Method lookup uses inline caching for O(1) access in steady state
//! - Shape-based hidden classes for instance attribute access
//! - Slots table enables direct dispatch for special methods

use crate::object::mro::{ClassId, Mro, MroError, compute_c3_mro};
use crate::object::shape::Shape;
use crate::object::type_obj::{TypeId, TypeSlots};
use crate::object::{ObjectHeader, PyObject};
use prism_core::Value;
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

// =============================================================================
// Global Type ID Counter
// =============================================================================

/// Global counter for allocating unique TypeIds to user-defined classes.
static NEXT_TYPE_ID: AtomicU32 = AtomicU32::new(TypeId::FIRST_USER_TYPE);

/// Allocate a new unique TypeId for a user-defined class.
fn allocate_type_id() -> TypeId {
    let id = NEXT_TYPE_ID.fetch_add(1, Ordering::Relaxed);
    TypeId::from_raw(id)
}

// =============================================================================
// Class Flags
// =============================================================================

bitflags::bitflags! {
    /// Flags describing class capabilities and state.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ClassFlags: u32 {
        /// Class has been fully initialized.
        const INITIALIZED = 1 << 0;
        /// Class is abstract (has unimplemented abstract methods).
        const ABSTRACT = 1 << 1;
        /// Class has custom `__new__`.
        const HAS_NEW = 1 << 2;
        /// Class has custom `__init__`.
        const HAS_INIT = 1 << 3;
        /// Class has `__slots__` (no instance `__dict__`).
        const HAS_SLOTS = 1 << 4;
        /// Class is a dataclass.
        const DATACLASS = 1 << 5;
        /// Class is final (cannot be subclassed).
        const FINAL = 1 << 6;
        /// Class defines `__hash__`.
        const HASHABLE = 1 << 7;
        /// Class defines `__eq__`.
        const HAS_EQ = 1 << 8;
        /// Class defines `__del__`.
        const HAS_FINALIZER = 1 << 9;
    }
}

impl Default for ClassFlags {
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// Instantiation Hints
// =============================================================================

/// JIT compilation hints for instance creation.
///
/// These hints help the JIT specialize allocation and initialization paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstantiationHint {
    /// Instance uses only inline slots (fast path).
    /// Class has `__slots__` with ≤4 slots.
    InlineSlots,

    /// Instance uses fixed slots from `__slots__`.
    /// Class has `__slots__` with >4 slots.
    FixedSlots,

    /// No custom `__init__` - skip init call.
    DefaultInit,

    /// Generic - full `__new__`/`__init__` protocol.
    Generic,
}

// =============================================================================
// Class Dictionary
// =============================================================================

/// Class attribute dictionary (methods, class variables).
///
/// Uses FxHashMap for fast lookup with string keys.
/// Values are boxed Python objects (methods, descriptors, etc.).
#[derive(Debug, Default)]
pub struct ClassDict {
    /// The underlying dictionary.
    attrs: RwLock<FxHashMap<InternedString, Value>>,
}

impl ClassDict {
    /// Create a new empty class dict.
    pub fn new() -> Self {
        Self {
            attrs: RwLock::new(FxHashMap::default()),
        }
    }

    /// Get an attribute.
    #[inline]
    pub fn get(&self, name: &InternedString) -> Option<Value> {
        self.attrs.read().unwrap().get(name).copied()
    }

    /// Set an attribute.
    #[inline]
    pub fn set(&self, name: InternedString, value: Value) {
        self.attrs.write().unwrap().insert(name, value);
    }

    /// Delete an attribute.
    #[inline]
    pub fn delete(&self, name: &InternedString) -> Option<Value> {
        self.attrs.write().unwrap().remove(name)
    }

    /// Check if attribute exists.
    #[inline]
    pub fn contains(&self, name: &InternedString) -> bool {
        self.attrs.read().unwrap().contains_key(name)
    }

    /// Get all attribute names.
    pub fn keys(&self) -> Vec<InternedString> {
        self.attrs.read().unwrap().keys().cloned().collect()
    }

    /// Number of attributes.
    pub fn len(&self) -> usize {
        self.attrs.read().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.attrs.read().unwrap().is_empty()
    }

    /// Iterate over all attributes.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(&InternedString, Value),
    {
        let attrs = self.attrs.read().unwrap();
        for (name, value) in attrs.iter() {
            f(name, *value);
        }
    }
}

impl Clone for ClassDict {
    fn clone(&self) -> Self {
        Self {
            attrs: RwLock::new(self.attrs.read().unwrap().clone()),
        }
    }
}

// =============================================================================
// Base Classes Storage
// =============================================================================

/// Stack-allocated storage for base classes.
/// Most classes have 1-2 base classes.
pub type Bases = SmallVec<[ClassId; 2]>;

// =============================================================================
// Method Slot Cache
// =============================================================================

/// Cached slot for fast method dispatch.
#[derive(Debug, Clone)]
pub struct MethodSlot {
    /// The method value.
    pub value: Value,
    /// ClassId where method was found.
    pub defining_class: ClassId,
    /// Index in MRO where method was found.
    pub mro_index: u16,
}

// =============================================================================
// Python Class Object
// =============================================================================

/// Python class object - represents a user-defined class.
///
/// # Memory Layout
///
/// The class object is heap-allocated and GC-tracked. Instances of this
/// class are stored separately and reference this class via TypeId.
///
/// # Thread Safety
///
/// The class dictionary uses RwLock for safe concurrent access.
/// MRO and bases are immutable after construction.
#[derive(Debug)]
pub struct PyClassObject {
    /// Object header.
    pub header: ObjectHeader,

    /// Class name.
    name: InternedString,

    /// Unique TypeId for this class.
    type_id: TypeId,

    /// Direct base classes (in declaration order).
    bases: Bases,

    /// Method Resolution Order (cached).
    mro: Mro,

    /// Class flags.
    flags: ClassFlags,

    /// Class attributes (methods, class variables).
    dict: ClassDict,

    /// Type slots for special method dispatch.
    slots: TypeSlots,

    /// Shape for instances of this class.
    instance_shape: Arc<Shape>,

    /// Optional __slots__ names.
    slot_names: Option<Vec<InternedString>>,
}

impl PyClassObject {
    /// Create a new class with the given name and bases.
    ///
    /// # Arguments
    ///
    /// * `name` - The class name
    /// * `bases` - Direct base classes (will be used with object if empty)
    /// * `mro_lookup` - Function to get MRO for parent classes
    ///
    /// # Returns
    ///
    /// The new class object, or an error if MRO computation fails.
    pub fn new<F>(name: InternedString, bases: &[ClassId], mro_lookup: F) -> Result<Self, MroError>
    where
        F: Fn(ClassId) -> Option<Mro>,
    {
        let type_id = allocate_type_id();
        let class_id = ClassId(type_id.raw());

        // Convert bases to SmallVec
        let bases_vec: Bases = bases.iter().copied().collect();

        // Compute MRO
        let mro = compute_c3_mro(class_id, bases, mro_lookup)?;

        Ok(Self {
            header: ObjectHeader::new(TypeId::TYPE),
            name,
            type_id,
            bases: bases_vec,
            mro,
            flags: ClassFlags::empty(),
            dict: ClassDict::new(),
            slots: TypeSlots::default(),
            instance_shape: Shape::empty(),
            slot_names: None,
        })
    }

    /// Create a simple class with no explicit bases (inherits from object).
    pub fn new_simple(name: InternedString) -> Self {
        let type_id = allocate_type_id();
        let class_id = ClassId(type_id.raw());

        // MRO for class with no bases: [class, object]
        let mut mro = Mro::new();
        mro.push(class_id);
        mro.push(ClassId::OBJECT);

        Self {
            header: ObjectHeader::new(TypeId::TYPE),
            name,
            type_id,
            bases: SmallVec::new(),
            mro,
            flags: ClassFlags::empty(),
            dict: ClassDict::new(),
            slots: TypeSlots::default(),
            instance_shape: Shape::empty(),
            slot_names: None,
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the class name.
    #[inline]
    pub fn name(&self) -> &InternedString {
        &self.name
    }

    /// Get the TypeId for this class.
    #[inline]
    pub fn class_type_id(&self) -> TypeId {
        self.type_id
    }

    /// Get the ClassId for this class.
    #[inline]
    pub fn class_id(&self) -> ClassId {
        ClassId(self.type_id.raw())
    }

    /// Get the base classes.
    #[inline]
    pub fn bases(&self) -> &[ClassId] {
        &self.bases
    }

    /// Get the MRO.
    #[inline]
    pub fn mro(&self) -> &[ClassId] {
        &self.mro
    }

    /// Get class flags.
    #[inline]
    pub fn flags(&self) -> ClassFlags {
        self.flags
    }

    /// Get type slots.
    #[inline]
    pub fn slots(&self) -> &TypeSlots {
        &self.slots
    }

    /// Get mutable type slots.
    #[inline]
    pub fn slots_mut(&mut self) -> &mut TypeSlots {
        &mut self.slots
    }

    /// Get the instance shape.
    #[inline]
    pub fn instance_shape(&self) -> &Arc<Shape> {
        &self.instance_shape
    }

    // =========================================================================
    // Attribute Access
    // =========================================================================

    /// Get a class attribute.
    #[inline]
    pub fn get_attr(&self, name: &InternedString) -> Option<Value> {
        self.dict.get(name)
    }

    /// Set a class attribute.
    #[inline]
    pub fn set_attr(&self, name: InternedString, value: Value) {
        self.dict.set(name, value);
    }

    /// Delete a class attribute.
    #[inline]
    pub fn del_attr(&self, name: &InternedString) -> Option<Value> {
        self.dict.delete(name)
    }

    /// Check if class has an attribute.
    #[inline]
    pub fn has_attr(&self, name: &InternedString) -> bool {
        self.dict.contains(name)
    }

    // =========================================================================
    // Method Lookup (MRO walk)
    // =========================================================================

    /// Look up a method by walking the MRO.
    ///
    /// This is the slow path - normal code should use inline caching.
    ///
    /// # Arguments
    ///
    /// * `name` - The method name to look up
    /// * `class_registry` - Function to get class objects by ClassId
    ///
    /// # Returns
    ///
    /// The method value and defining class, or None if not found.
    pub fn lookup_method<F>(&self, name: &InternedString, class_registry: F) -> Option<MethodSlot>
    where
        F: Fn(ClassId) -> Option<Arc<PyClassObject>>,
    {
        // Walk MRO from first (this class) to last (object)
        for (mro_index, &class_id) in self.mro.iter().enumerate() {
            if class_id == self.class_id() {
                // This class - check our dict
                if let Some(value) = self.dict.get(name) {
                    return Some(MethodSlot {
                        value,
                        defining_class: class_id,
                        mro_index: mro_index as u16,
                    });
                }
            } else {
                // Parent class - look up in registry
                if let Some(parent_class) = class_registry(class_id) {
                    if let Some(value) = parent_class.dict.get(name) {
                        return Some(MethodSlot {
                            value,
                            defining_class: class_id,
                            mro_index: mro_index as u16,
                        });
                    }
                }
            }
        }

        None
    }

    // =========================================================================
    // Flags
    // =========================================================================

    /// Mark class as initialized.
    pub fn mark_initialized(&mut self) {
        self.flags |= ClassFlags::INITIALIZED;
    }

    /// Check if class is initialized.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.flags.contains(ClassFlags::INITIALIZED)
    }

    /// Set __slots__ names.
    pub fn set_slots(&mut self, slot_names: Vec<InternedString>) {
        self.slot_names = Some(slot_names);
        self.flags |= ClassFlags::HAS_SLOTS;
    }

    /// Get __slots__ names.
    pub fn slot_names(&self) -> Option<&[InternedString]> {
        self.slot_names.as_deref()
    }

    /// Check if class has __slots__.
    #[inline]
    pub fn has_slots(&self) -> bool {
        self.flags.contains(ClassFlags::HAS_SLOTS)
    }

    /// Check if class is final (cannot be subclassed).
    #[inline]
    pub fn is_final(&self) -> bool {
        self.flags.contains(ClassFlags::FINAL)
    }

    /// Mark class as final.
    pub fn mark_final(&mut self) {
        self.flags |= ClassFlags::FINAL;
    }

    /// Set class flags (replaces all flags).
    pub fn set_flags(&mut self, flags: ClassFlags) {
        self.flags = flags;
    }

    /// Add flags to existing flags.
    pub fn add_flags(&mut self, flags: ClassFlags) {
        self.flags |= flags;
    }

    // =========================================================================
    // Instantiation Protocol
    // =========================================================================

    /// Check if this class has a custom `__new__` method.
    ///
    /// If not, the default object allocation is used.
    #[inline]
    pub fn has_custom_new(&self) -> bool {
        self.flags.contains(ClassFlags::HAS_NEW)
    }

    /// Check if this class has a custom `__init__` method.
    ///
    /// If not, no initialization is needed after allocation.
    #[inline]
    pub fn has_custom_init(&self) -> bool {
        self.flags.contains(ClassFlags::HAS_INIT)
    }

    /// Mark this class as having a custom `__new__`.
    pub fn mark_has_new(&mut self) {
        self.flags |= ClassFlags::HAS_NEW;
    }

    /// Mark this class as having a custom `__init__`.
    pub fn mark_has_init(&mut self) {
        self.flags |= ClassFlags::HAS_INIT;
    }

    /// Resolve `__new__` by walking the MRO.
    ///
    /// Returns the `__new__` method from the first class in MRO that defines it.
    /// If no class defines it, returns None (use default allocation).
    pub fn resolve_new<F>(&self, class_registry: F) -> Option<MethodSlot>
    where
        F: Fn(ClassId) -> Option<Arc<PyClassObject>>,
    {
        let new_name = prism_core::intern::intern("__new__");
        self.lookup_method(&new_name, class_registry)
    }

    /// Resolve `__init__` by walking the MRO.
    ///
    /// Returns the `__init__` method from the first class in MRO that defines it.
    /// If no class defines it, returns None (skip initialization).
    pub fn resolve_init<F>(&self, class_registry: F) -> Option<MethodSlot>
    where
        F: Fn(ClassId) -> Option<Arc<PyClassObject>>,
    {
        let init_name = prism_core::intern::intern("__init__");
        self.lookup_method(&init_name, class_registry)
    }

    /// Get specialization hint for JIT optimization.
    ///
    /// This helps the JIT specialize instance allocation paths.
    pub fn instantiation_hint(&self) -> InstantiationHint {
        if self.flags.contains(ClassFlags::HAS_SLOTS) {
            // Check if slots fit inline
            if let Some(names) = &self.slot_names {
                if names.len() <= 4 {
                    return InstantiationHint::InlineSlots;
                }
            }
            InstantiationHint::FixedSlots
        } else if !self.flags.contains(ClassFlags::HAS_INIT) {
            InstantiationHint::DefaultInit
        } else {
            InstantiationHint::Generic
        }
    }
}

impl PyObject for PyClassObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::intern;

    #[test]
    fn test_simple_class_creation() {
        let name = intern("MyClass");
        let class = PyClassObject::new_simple(name.clone());

        assert_eq!(class.name(), &name);
        assert!(class.bases().is_empty());
        assert_eq!(class.mro().len(), 2); // [MyClass, object]
        assert!(!class.is_initialized());
    }

    #[test]
    fn test_class_type_id_uniqueness() {
        let class1 = PyClassObject::new_simple(intern("Class1"));
        let class2 = PyClassObject::new_simple(intern("Class2"));
        let class3 = PyClassObject::new_simple(intern("Class3"));

        // Each class should have a unique TypeId
        assert_ne!(class1.class_type_id(), class2.class_type_id());
        assert_ne!(class2.class_type_id(), class3.class_type_id());
        assert_ne!(class1.class_type_id(), class3.class_type_id());
    }

    #[test]
    fn test_class_attributes() {
        let class = PyClassObject::new_simple(intern("Test"));
        let attr_name = intern("my_attr");

        // Initially no attribute
        assert!(!class.has_attr(&attr_name));
        assert!(class.get_attr(&attr_name).is_none());

        // Set attribute
        class.set_attr(attr_name.clone(), Value::int_unchecked(42));

        // Now should exist
        assert!(class.has_attr(&attr_name));
        assert_eq!(class.get_attr(&attr_name), Some(Value::int_unchecked(42)));

        // Delete attribute
        let deleted = class.del_attr(&attr_name);
        assert_eq!(deleted, Some(Value::int_unchecked(42)));
        assert!(!class.has_attr(&attr_name));
    }

    #[test]
    fn test_class_flags() {
        let mut class = PyClassObject::new_simple(intern("Test"));

        assert!(!class.is_initialized());
        class.mark_initialized();
        assert!(class.is_initialized());

        assert!(!class.has_slots());
        class.set_slots(vec![intern("x"), intern("y")]);
        assert!(class.has_slots());
        assert_eq!(class.slot_names().unwrap().len(), 2);
    }

    #[test]
    fn test_class_with_inheritance() {
        use std::collections::HashMap;

        // Create parent class
        let parent = PyClassObject::new_simple(intern("Parent"));
        let parent_id = parent.class_id();
        let parent = Arc::new(parent);

        // Create class registry
        let mut registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();
        registry.insert(parent_id, parent.clone());

        // Create child class
        let child_name = intern("Child");
        let child = PyClassObject::new(child_name.clone(), &[parent_id], |id| {
            registry.get(&id).map(|c| c.mro.clone())
        })
        .unwrap();

        // Child's MRO should include parent
        assert_eq!(child.mro().len(), 3); // [Child, Parent, object]
        assert!(child.bases().contains(&parent_id));
    }

    #[test]
    fn test_method_lookup_in_mro() {
        use std::collections::HashMap;
        use std::sync::Arc;

        // Create parent class with a method
        let mut parent = PyClassObject::new_simple(intern("Parent"));
        let method_name = intern("greet");
        parent.set_attr(method_name.clone(), Value::int_unchecked(100));
        let parent_id = parent.class_id();
        let parent = Arc::new(parent);

        // Registry
        let mut registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();
        registry.insert(parent_id, parent.clone());

        // Create child class
        let child = PyClassObject::new(intern("Child"), &[parent_id], |id| {
            registry.get(&id).map(|c| c.mro.clone())
        })
        .unwrap();

        // Add child to registry
        let child_id = child.class_id();
        let child = Arc::new(child);
        registry.insert(child_id, child.clone());

        // Look up method from child - should find in parent
        let slot = child.lookup_method(&method_name, |id| registry.get(&id).cloned());
        assert!(slot.is_some());
        let slot = slot.unwrap();
        assert_eq!(slot.defining_class, parent_id);
        assert_eq!(slot.mro_index, 1); // Second in MRO (after Child)
    }

    #[test]
    fn test_method_override() {
        use std::collections::HashMap;
        use std::sync::Arc;

        // Create parent class with a method
        let mut parent = PyClassObject::new_simple(intern("Parent"));
        let method_name = intern("greet");
        parent.set_attr(method_name.clone(), Value::int_unchecked(100));
        let parent_id = parent.class_id();
        let parent = Arc::new(parent);

        // Registry
        let mut registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();
        registry.insert(parent_id, parent.clone());

        // Create child class with overridden method
        let child = PyClassObject::new(intern("Child"), &[parent_id], |id| {
            registry.get(&id).map(|c| c.mro.clone())
        })
        .unwrap();

        // Override the method in child
        child.set_attr(method_name.clone(), Value::int_unchecked(200));

        let child_id = child.class_id();
        let child = Arc::new(child);
        registry.insert(child_id, child.clone());

        // Look up method - should find child's version
        let slot = child.lookup_method(&method_name, |id| registry.get(&id).cloned());
        assert!(slot.is_some());
        let slot = slot.unwrap();
        assert_eq!(slot.defining_class, child_id);
        assert_eq!(slot.mro_index, 0); // First in MRO (Child itself)
        assert_eq!(slot.value, Value::int_unchecked(200));
    }

    #[test]
    fn test_class_dict_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let class = Arc::new(PyClassObject::new_simple(intern("ThreadTest")));

        // Spawn multiple threads to read/write attributes
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let class = class.clone();
                thread::spawn(move || {
                    let attr = intern(&format!("attr_{}", i));
                    class.set_attr(attr.clone(), Value::int_unchecked(i as i64));
                    assert!(class.has_attr(&attr));
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All attributes should be set
        for i in 0..4 {
            let attr = intern(&format!("attr_{}", i));
            assert!(class.has_attr(&attr));
        }
    }

    #[test]
    fn test_mro_no_heap_allocation() {
        // For simple classes, MRO should not spill to heap
        let class = PyClassObject::new_simple(intern("Simple"));
        assert!(!class.mro.spilled());
    }

    // =========================================================================
    // Instantiation Protocol Tests
    // =========================================================================

    #[test]
    fn test_has_custom_new_default() {
        let class = PyClassObject::new_simple(intern("MyClass"));
        assert!(!class.has_custom_new());
    }

    #[test]
    fn test_has_custom_init_default() {
        let class = PyClassObject::new_simple(intern("MyClass"));
        assert!(!class.has_custom_init());
    }

    #[test]
    fn test_mark_has_new() {
        let mut class = PyClassObject::new_simple(intern("MyClass"));
        assert!(!class.has_custom_new());

        class.mark_has_new();
        assert!(class.has_custom_new());
    }

    #[test]
    fn test_mark_has_init() {
        let mut class = PyClassObject::new_simple(intern("MyClass"));
        assert!(!class.has_custom_init());

        class.mark_has_init();
        assert!(class.has_custom_init());
    }

    #[test]
    fn test_instantiation_hint_default() {
        let class = PyClassObject::new_simple(intern("MyClass"));
        // No slots, no init → DefaultInit
        assert_eq!(class.instantiation_hint(), InstantiationHint::DefaultInit);
    }

    #[test]
    fn test_instantiation_hint_with_init() {
        let mut class = PyClassObject::new_simple(intern("MyClass"));
        class.mark_has_init();
        // Has init → Generic
        assert_eq!(class.instantiation_hint(), InstantiationHint::Generic);
    }

    #[test]
    fn test_instantiation_hint_inline_slots() {
        let mut class = PyClassObject::new_simple(intern("MyClass"));
        // Set 4 or fewer slots → InlineSlots
        class.set_slots(vec![intern("x"), intern("y"), intern("z")]);
        assert_eq!(class.instantiation_hint(), InstantiationHint::InlineSlots);
    }

    #[test]
    fn test_instantiation_hint_fixed_slots() {
        let mut class = PyClassObject::new_simple(intern("MyClass"));
        // Set more than 4 slots → FixedSlots
        class.set_slots(vec![
            intern("a"),
            intern("b"),
            intern("c"),
            intern("d"),
            intern("e"),
        ]);
        assert_eq!(class.instantiation_hint(), InstantiationHint::FixedSlots);
    }

    #[test]
    fn test_resolve_new_not_found() {
        use std::collections::HashMap;
        use std::sync::Arc;

        let class = PyClassObject::new_simple(intern("MyClass"));
        let registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();

        // No __new__ defined
        let slot = class.resolve_new(|id| registry.get(&id).cloned());
        assert!(slot.is_none());
    }

    #[test]
    fn test_resolve_init_not_found() {
        use std::collections::HashMap;
        use std::sync::Arc;

        let class = PyClassObject::new_simple(intern("MyClass"));
        let registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();

        // No __init__ defined
        let slot = class.resolve_init(|id| registry.get(&id).cloned());
        assert!(slot.is_none());
    }

    #[test]
    fn test_resolve_new_found() {
        use std::collections::HashMap;
        use std::sync::Arc;

        let class = PyClassObject::new_simple(intern("MyClass"));
        let new_name = intern("__new__");
        class.set_attr(new_name.clone(), Value::int_unchecked(999));

        let registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();

        let slot = class.resolve_new(|id| registry.get(&id).cloned());
        assert!(slot.is_some());
        let slot = slot.unwrap();
        assert_eq!(slot.value, Value::int_unchecked(999));
        assert_eq!(slot.defining_class, class.class_id());
    }

    #[test]
    fn test_resolve_init_found() {
        use std::collections::HashMap;
        use std::sync::Arc;

        let class = PyClassObject::new_simple(intern("MyClass"));
        let init_name = intern("__init__");
        class.set_attr(init_name.clone(), Value::int_unchecked(888));

        let registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();

        let slot = class.resolve_init(|id| registry.get(&id).cloned());
        assert!(slot.is_some());
        let slot = slot.unwrap();
        assert_eq!(slot.value, Value::int_unchecked(888));
        assert_eq!(slot.defining_class, class.class_id());
    }

    #[test]
    fn test_resolve_init_inherited() {
        use std::collections::HashMap;
        use std::sync::Arc;

        // Parent with __init__
        let parent = PyClassObject::new_simple(intern("Parent"));
        let init_name = intern("__init__");
        parent.set_attr(init_name.clone(), Value::int_unchecked(777));
        let parent_id = parent.class_id();
        let parent = Arc::new(parent);

        let mut registry: HashMap<ClassId, Arc<PyClassObject>> = HashMap::new();
        registry.insert(parent_id, parent.clone());

        // Child without __init__
        let child = PyClassObject::new(intern("Child"), &[parent_id], |id| {
            registry.get(&id).map(|c| c.mro.clone())
        })
        .unwrap();
        let child_id = child.class_id();
        let child = Arc::new(child);
        registry.insert(child_id, child.clone());

        // Should find parent's __init__
        let slot = child.resolve_init(|id| registry.get(&id).cloned());
        assert!(slot.is_some());
        let slot = slot.unwrap();
        assert_eq!(slot.value, Value::int_unchecked(777));
        assert_eq!(slot.defining_class, parent_id);
        assert_eq!(slot.mro_index, 1); // Second in child's MRO
    }
}
