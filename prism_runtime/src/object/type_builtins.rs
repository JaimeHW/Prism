//! Python type builtin functions and O(1) subclass checking.
//!
//! This module provides:
//! - `type(obj)` - Get the type of an object (1-arg form)
//! - `type(name, bases, dict)` - Create a new class dynamically (3-arg form)
//! - `isinstance(obj, cls)` - Check if object is instance of class
//! - `issubclass(sub, cls)` - Check if class is subclass of another
//!
//! # Performance
//!
//! The key optimization is `SubclassBitmap` which enables **O(1)** subclass
//! checking instead of O(n) MRO traversal. Each class stores a bitmap where
//! bit positions correspond to type IDs of ancestor classes.
//!
//! Built-in types (0-63) use inline u64 storage. User-defined types use
//! overflow storage only when necessary.
//!
//! # Example
//!
//! ```python
//! class Animal: pass
//! class Dog(Animal): pass
//!
//! isinstance(Dog(), Animal)  # True - O(1) bitmap check
//! issubclass(Dog, Animal)    # True - O(1) bitmap check
//! ```

use crate::object::mro::ClassId;
use crate::object::type_obj::TypeId;
use prism_core::Value;

// =============================================================================
// Subclass Bitmap - O(1) Subclass Testing
// =============================================================================

/// Number of bits in inline storage (covers all built-in types).
const INLINE_BITS: usize = 64;

/// Number of u64 words in overflow storage.
/// Supports up to 64 * 16 = 1024 type IDs without reallocation.
const OVERFLOW_WORDS: usize = 16;

/// Compact bitmap for O(1) subclass testing.
///
/// Each class has a unique bit position corresponding to its TypeId.
/// A class's bitmap includes bits for itself and all ancestors.
///
/// # Memory Layout
///
/// ```text
/// SubclassBitmap (24 bytes typical)
/// ├── inline: u64 (8 bytes) - bits 0-63 (built-in types)
/// └── overflow: Option<Box<[u64; 16]>> (16 bytes)
/// ```
///
/// # Algorithm
///
/// To check if type A is subclass of type B:
/// 1. Get bit position = B.type_id
/// 2. Check if A.bitmap has that bit set
///
/// This is O(1) regardless of inheritance depth.
#[derive(Debug, Clone)]
pub struct SubclassBitmap {
    /// Inline storage for first 64 types (built-ins).
    /// Bit N is set if this type is a subclass of TypeId(N).
    inline: u64,

    /// Overflow storage for user-defined types (TypeId >= 64).
    /// Lazily allocated only when needed.
    overflow: Option<Box<[u64; OVERFLOW_WORDS]>>,
}

impl SubclassBitmap {
    /// Create a new empty bitmap.
    #[inline]
    pub const fn new() -> Self {
        Self {
            inline: 0,
            overflow: None,
        }
    }

    /// Create a bitmap with a single bit set (for leaf classes).
    #[inline]
    pub fn for_type(type_id: TypeId) -> Self {
        let mut bitmap = Self::new();
        bitmap.set_bit(type_id);
        bitmap
    }

    /// Create a bitmap by combining parent bitmaps + self.
    ///
    /// This is used when creating a new class to inherit all ancestor bits.
    pub fn from_parents<'a, I>(type_id: TypeId, parents: I) -> Self
    where
        I: Iterator<Item = &'a SubclassBitmap>,
    {
        let mut bitmap = Self::new();

        // Set our own bit
        bitmap.set_bit(type_id);

        // Merge all parent bitmaps
        for parent in parents {
            bitmap.merge(parent);
        }

        bitmap
    }

    /// Check if this type is a subclass of the given type.
    ///
    /// Returns true if the bit corresponding to `type_id` is set.
    /// This is the hot path - must be as fast as possible.
    #[inline]
    pub fn is_subclass_of(&self, type_id: TypeId) -> bool {
        let bit = type_id.raw() as usize;

        if bit < INLINE_BITS {
            // Fast path: check inline storage
            (self.inline & (1u64 << bit)) != 0
        } else {
            // Slow path: check overflow
            self.check_overflow_bit(bit)
        }
    }

    /// Check if this type is a subclass of any type in the given list.
    ///
    /// Used for `isinstance(obj, (A, B, C))` tuple form.
    #[inline]
    pub fn is_subclass_of_any(&self, type_ids: &[TypeId]) -> bool {
        for &type_id in type_ids {
            if self.is_subclass_of(type_id) {
                return true;
            }
        }
        false
    }

    /// Set the bit for a type ID.
    #[inline]
    pub fn set_bit(&mut self, type_id: TypeId) {
        let bit = type_id.raw() as usize;

        if bit < INLINE_BITS {
            self.inline |= 1u64 << bit;
        } else {
            self.set_overflow_bit(bit);
        }
    }

    /// Merge another bitmap into this one (union of bits).
    pub fn merge(&mut self, other: &SubclassBitmap) {
        // Merge inline bits
        self.inline |= other.inline;

        // Merge overflow bits if present
        if let Some(ref other_overflow) = other.overflow {
            let overflow = self
                .overflow
                .get_or_insert_with(|| Box::new([0u64; OVERFLOW_WORDS]));
            for (dst, src) in overflow.iter_mut().zip(other_overflow.iter()) {
                *dst |= *src;
            }
        }
    }

    /// Check bit in overflow storage.
    #[cold]
    fn check_overflow_bit(&self, bit: usize) -> bool {
        let overflow_bit = bit - INLINE_BITS;
        let word_idx = overflow_bit / 64;
        let bit_idx = overflow_bit % 64;

        if word_idx >= OVERFLOW_WORDS {
            return false;
        }

        match &self.overflow {
            Some(overflow) => (overflow[word_idx] & (1u64 << bit_idx)) != 0,
            None => false,
        }
    }

    /// Set bit in overflow storage.
    #[cold]
    fn set_overflow_bit(&mut self, bit: usize) {
        let overflow_bit = bit - INLINE_BITS;
        let word_idx = overflow_bit / 64;
        let bit_idx = overflow_bit % 64;

        if word_idx >= OVERFLOW_WORDS {
            // Type ID too large - would need dynamic growth
            // This is extremely rare (>1024 types in hierarchy)
            return;
        }

        let overflow = self
            .overflow
            .get_or_insert_with(|| Box::new([0u64; OVERFLOW_WORDS]));
        overflow[word_idx] |= 1u64 << bit_idx;
    }

    /// Get the number of bits set (for debugging).
    pub fn count_bits(&self) -> usize {
        let mut count = self.inline.count_ones() as usize;

        if let Some(ref overflow) = self.overflow {
            for word in overflow.iter() {
                count += word.count_ones() as usize;
            }
        }

        count
    }

    /// Check if bitmap is empty (no types).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inline == 0
            && self
                .overflow
                .as_ref()
                .map_or(true, |o| o.iter().all(|&w| w == 0))
    }
}

impl Default for SubclassBitmap {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Type Check Inline Cache
// =============================================================================

/// Maximum entries in polymorphic type check cache.
const TYPE_CHECK_IC_SIZE: usize = 4;

/// Polymorphic inline cache for type checks.
///
/// Caches recent (type_id, result) pairs for fast repeated checks.
/// Used by JIT-compiled isinstance/issubclass calls.
///
/// # Algorithm
///
/// 1. Check if type_id is in cache
/// 2. If hit, return cached result
/// 3. If miss, compute result and add to cache
///
/// Cache uses LRU replacement when full.
#[derive(Debug, Clone)]
pub struct TypeCheckIC {
    /// Cached entries: (type_id, is_subclass_result)
    entries: [(ClassId, bool); TYPE_CHECK_IC_SIZE],
    /// Number of valid entries (0-4)
    len: u8,
    /// Index for next replacement (circular)
    next_slot: u8,
}

impl TypeCheckIC {
    /// Create a new empty cache.
    #[inline]
    pub const fn new() -> Self {
        Self {
            entries: [(ClassId::NONE, false); TYPE_CHECK_IC_SIZE],
            len: 0,
            next_slot: 0,
        }
    }

    /// Look up a type in the cache.
    ///
    /// Returns Some(result) if cached, None if miss.
    #[inline]
    pub fn lookup(&self, type_id: ClassId) -> Option<bool> {
        // Unrolled loop for 4 entries
        let len = self.len as usize;
        if len > 0 && self.entries[0].0 == type_id {
            return Some(self.entries[0].1);
        }
        if len > 1 && self.entries[1].0 == type_id {
            return Some(self.entries[1].1);
        }
        if len > 2 && self.entries[2].0 == type_id {
            return Some(self.entries[2].1);
        }
        if len > 3 && self.entries[3].0 == type_id {
            return Some(self.entries[3].1);
        }
        None
    }

    /// Insert a new entry into the cache.
    ///
    /// Uses circular replacement when full.
    #[inline]
    pub fn insert(&mut self, type_id: ClassId, result: bool) {
        let slot = self.next_slot as usize;
        self.entries[slot] = (type_id, result);

        if self.len < TYPE_CHECK_IC_SIZE as u8 {
            self.len += 1;
        }

        self.next_slot = ((slot + 1) % TYPE_CHECK_IC_SIZE) as u8;
    }

    /// Clear the cache.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
        self.next_slot = 0;
    }

    /// Get number of cached entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Check if cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for TypeCheckIC {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Type Builtin Functions
// =============================================================================

/// Get the TypeId of a value.
///
/// This is the implementation of `type(obj)` (1-arg form).
///
/// # Performance
///
/// For NaN-boxed values, type extraction is O(1) via tag bits.
/// For heap objects, requires reading the object header.
#[inline]
pub fn type_of_value(value: Value) -> TypeId {
    // Use Value's type checking methods
    if value.is_int() {
        TypeId::INT
    } else if value.is_float() {
        TypeId::FLOAT
    } else if value.is_bool() {
        TypeId::BOOL
    } else if value.is_none() {
        TypeId::NONE
    } else if value.is_object() {
        // Heap object - read type from header
        // For now, return OBJECT as placeholder until header reading is implemented
        TypeId::OBJECT
    } else if value.is_string() {
        TypeId::STR
    } else {
        TypeId::OBJECT
    }
}

/// Check if a value is an instance of a class.
///
/// This is the implementation of `isinstance(obj, cls)`.
///
/// # Algorithm
///
/// 1. Get type of value
/// 2. Look up type's SubclassBitmap
/// 3. Check if target class bit is set
///
/// # Arguments
///
/// * `value` - The object to check
/// * `class_id` - The class to check against
/// * `get_bitmap` - Function to get SubclassBitmap for a ClassId
#[inline]
pub fn isinstance<F>(value: Value, class_id: ClassId, get_bitmap: F) -> bool
where
    F: Fn(ClassId) -> Option<SubclassBitmap>,
{
    // Get the type of the value
    let value_type = type_of_value(value);
    let value_class = ClassId(value_type.raw());

    // Get the bitmap for the value's type
    if let Some(bitmap) = get_bitmap(value_class) {
        // O(1) subclass check
        bitmap.is_subclass_of(TypeId::from_raw(class_id.0))
    } else {
        // Fallback: direct type comparison
        value_class == class_id
    }
}

/// Check if a value is an instance of any class in a tuple.
///
/// This is the implementation of `isinstance(obj, (cls1, cls2, ...))`.
#[inline]
pub fn isinstance_multi<F>(value: Value, class_ids: &[ClassId], get_bitmap: F) -> bool
where
    F: Fn(ClassId) -> Option<SubclassBitmap>,
{
    let value_type = type_of_value(value);
    let value_class = ClassId(value_type.raw());

    if let Some(bitmap) = get_bitmap(value_class) {
        for &class_id in class_ids {
            if bitmap.is_subclass_of(TypeId::from_raw(class_id.0)) {
                return true;
            }
        }
        false
    } else {
        // Fallback: direct type comparison
        class_ids.contains(&value_class)
    }
}

/// Check if one class is a subclass of another.
///
/// This is the implementation of `issubclass(sub, cls)`.
#[inline]
pub fn issubclass<F>(sub_class: ClassId, parent_class: ClassId, get_bitmap: F) -> bool
where
    F: Fn(ClassId) -> Option<SubclassBitmap>,
{
    if sub_class == parent_class {
        return true; // Every class is subclass of itself
    }

    if let Some(bitmap) = get_bitmap(sub_class) {
        bitmap.is_subclass_of(TypeId::from_raw(parent_class.0))
    } else {
        false
    }
}

/// Check if a class is a subclass of any class in a tuple.
///
/// This is the implementation of `issubclass(sub, (cls1, cls2, ...))`.
#[inline]
pub fn issubclass_multi<F>(sub_class: ClassId, parent_classes: &[ClassId], get_bitmap: F) -> bool
where
    F: Fn(ClassId) -> Option<SubclassBitmap>,
{
    if parent_classes.contains(&sub_class) {
        return true;
    }

    if let Some(bitmap) = get_bitmap(sub_class) {
        for &parent_class in parent_classes {
            if bitmap.is_subclass_of(TypeId::from_raw(parent_class.0)) {
                return true;
            }
        }
        false
    } else {
        false
    }
}

// =============================================================================
// Built-in Type Bitmaps
// =============================================================================

/// Pre-computed bitmap for the `object` type.
/// All types are subclasses of object.
pub static OBJECT_BITMAP: SubclassBitmap = SubclassBitmap {
    inline: 1u64 << TypeId::OBJECT.raw(),
    overflow: None,
};

/// Create bitmap for a built-in type that only inherits from object.
#[inline]
pub fn builtin_type_bitmap(type_id: TypeId) -> SubclassBitmap {
    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(type_id);
    bitmap.set_bit(TypeId::OBJECT); // All types inherit from object
    bitmap
}

// =============================================================================
// Type Creation (3-arg form)
// =============================================================================

use crate::object::class::{ClassDict, ClassFlags, PyClassObject};
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Error during dynamic class creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeCreationError {
    /// Base class is marked final and cannot be subclassed.
    FinalBase { class_name: String },

    /// MRO computation failed (C3 linearization conflict).
    MroError { message: String },

    /// Base class not found in registry.
    BaseNotFound { class_id: ClassId },

    /// Invalid class name (empty or contains invalid characters).
    InvalidName { name: String },

    /// Conflicting __slots__ declarations.
    SlotsConflict { message: String },

    /// Metaclass conflict.
    MetaclassConflict { message: String },
}

impl std::fmt::Display for TypeCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeCreationError::FinalBase { class_name } => {
                write!(f, "cannot subclass final class '{}'", class_name)
            }
            TypeCreationError::MroError { message } => {
                write!(f, "MRO error: {}", message)
            }
            TypeCreationError::BaseNotFound { class_id } => {
                write!(f, "base class not found: {:?}", class_id)
            }
            TypeCreationError::InvalidName { name } => {
                write!(f, "invalid class name: '{}'", name)
            }
            TypeCreationError::SlotsConflict { message } => {
                write!(f, "__slots__ conflict: {}", message)
            }
            TypeCreationError::MetaclassConflict { message } => {
                write!(f, "metaclass conflict: {}", message)
            }
        }
    }
}

impl std::error::Error for TypeCreationError {}

/// Registry trait for looking up class information.
pub trait ClassRegistry {
    /// Get class object by ID.
    fn get_class(&self, id: ClassId) -> Option<Arc<PyClassObject>>;

    /// Get subclass bitmap for a class.
    fn get_bitmap(&self, id: ClassId) -> Option<SubclassBitmap>;

    /// Register a new class.
    fn register_class(&self, class: Arc<PyClassObject>);
}

/// Result of type_new operation.
#[derive(Debug)]
pub struct TypeNewResult {
    /// The newly created class.
    pub class: Arc<PyClassObject>,

    /// Pre-computed subclass bitmap.
    pub bitmap: SubclassBitmap,

    /// Detected method flags.
    pub flags: ClassFlags,
}

/// Create a new class dynamically (3-arg form of `type()`).
///
/// This is the implementation of `type(name, bases, dict)` which creates
/// a new class at runtime.
///
/// # Arguments
///
/// * `name` - Class name
/// * `bases` - Base class IDs (empty = inherit from object)
/// * `namespace` - Class attributes (methods, class variables)
/// * `registry` - Registry for looking up base classes
///
/// # Algorithm
///
/// 1. Validate class name
/// 2. Check base classes exist and are not final
/// 3. Compute MRO via C3 linearization
/// 4. Compute SubclassBitmap from parent bitmaps
/// 5. Detect __new__, __init__, __slots__ in namespace
/// 6. Create and return PyClassObject
///
/// # Performance
///
/// * O(n) MRO computation (done once at class creation)
/// * O(n) bitmap merging (done once at class creation)
/// * Namespace scanning is O(k) where k = number of special methods
pub fn type_new<R>(
    name: InternedString,
    bases: &[ClassId],
    namespace: &ClassDict,
    registry: &R,
) -> Result<TypeNewResult, TypeCreationError>
where
    R: ClassRegistry,
{
    // 1. Validate class name
    let name_str = name.as_str();
    if name_str.is_empty() {
        return Err(TypeCreationError::InvalidName {
            name: name_str.to_string(),
        });
    }

    // 2. Validate base classes
    let mut parent_bitmaps: Vec<&SubclassBitmap> = Vec::with_capacity(bases.len());
    for &base_id in bases {
        let base = registry
            .get_class(base_id)
            .ok_or_else(|| TypeCreationError::BaseNotFound { class_id: base_id })?;

        // Check if base is final
        if base.is_final() {
            return Err(TypeCreationError::FinalBase {
                class_name: base.name().as_str().to_string(),
            });
        }
    }

    // 3. Create the class object
    let mut class = if bases.is_empty() {
        // No bases - simple class inheriting from object
        PyClassObject::new_simple(name.clone())
    } else {
        // Has bases - compute MRO
        PyClassObject::new(name.clone(), bases, |id| {
            registry
                .get_class(id)
                .map(|c| c.mro().iter().copied().collect())
        })
        .map_err(|e| TypeCreationError::MroError {
            message: e.to_string(),
        })?
    };

    // 4. Compute SubclassBitmap from parent bitmaps
    let class_type_id = TypeId::from_raw(class.class_id().0);
    let mut bitmap = SubclassBitmap::new();
    bitmap.set_bit(class_type_id);
    bitmap.set_bit(TypeId::OBJECT); // All classes inherit from object

    // Merge parent bitmaps
    for &base_id in bases {
        if let Some(parent_bitmap) = registry.get_bitmap(base_id) {
            bitmap.merge(&parent_bitmap);
        }
    }

    // 5. Detect special methods in namespace
    let mut flags = ClassFlags::INITIALIZED;

    // Check for __new__
    let new_name = prism_core::intern::intern("__new__");
    if namespace.contains(&new_name) {
        flags |= ClassFlags::HAS_NEW;
    }

    // Check for __init__
    let init_name = prism_core::intern::intern("__init__");
    if namespace.contains(&init_name) {
        flags |= ClassFlags::HAS_INIT;
    }

    // Check for __slots__
    let slots_name = prism_core::intern::intern("__slots__");
    if namespace.contains(&slots_name) {
        flags |= ClassFlags::HAS_SLOTS;
    }

    // Check for __hash__
    let hash_name = prism_core::intern::intern("__hash__");
    if namespace.contains(&hash_name) {
        flags |= ClassFlags::HASHABLE;
    }

    // Check for __eq__
    let eq_name = prism_core::intern::intern("__eq__");
    if namespace.contains(&eq_name) {
        flags |= ClassFlags::HAS_EQ;
    }

    // Check for __del__
    let del_name = prism_core::intern::intern("__del__");
    if namespace.contains(&del_name) {
        flags |= ClassFlags::HAS_FINALIZER;
    }

    // 6. Copy namespace to class dict
    namespace.for_each(|name, value| {
        class.set_attr(name.clone(), value);
    });

    // Set detected flags
    class.set_flags(flags);

    let class = Arc::new(class);

    Ok(TypeNewResult {
        class,
        bitmap,
        flags,
    })
}

/// Simple in-memory class registry for testing.
#[derive(Default)]
pub struct SimpleClassRegistry {
    classes: std::sync::RwLock<FxHashMap<ClassId, Arc<PyClassObject>>>,
    bitmaps: std::sync::RwLock<FxHashMap<ClassId, SubclassBitmap>>,
}

impl SimpleClassRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a class with its bitmap.
    pub fn register(&self, class: Arc<PyClassObject>, bitmap: SubclassBitmap) {
        let id = class.class_id();
        self.classes.write().unwrap().insert(id, class);
        self.bitmaps.write().unwrap().insert(id, bitmap);
    }
}

impl ClassRegistry for SimpleClassRegistry {
    fn get_class(&self, id: ClassId) -> Option<Arc<PyClassObject>> {
        self.classes.read().unwrap().get(&id).cloned()
    }

    fn get_bitmap(&self, id: ClassId) -> Option<SubclassBitmap> {
        self.bitmaps.read().unwrap().get(&id).cloned()
    }

    fn register_class(&self, class: Arc<PyClassObject>) {
        let id = class.class_id();
        self.classes.write().unwrap().insert(id, class);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SubclassBitmap Tests
    // =========================================================================

    #[test]
    fn test_bitmap_new_is_empty() {
        let bitmap = SubclassBitmap::new();
        assert!(bitmap.is_empty());
        assert_eq!(bitmap.count_bits(), 0);
    }

    #[test]
    fn test_bitmap_for_type() {
        let bitmap = SubclassBitmap::for_type(TypeId::INT);
        assert!(!bitmap.is_empty());
        assert!(bitmap.is_subclass_of(TypeId::INT));
        assert!(!bitmap.is_subclass_of(TypeId::FLOAT));
    }

    #[test]
    fn test_bitmap_set_bit_inline() {
        let mut bitmap = SubclassBitmap::new();

        bitmap.set_bit(TypeId::INT);
        bitmap.set_bit(TypeId::FLOAT);
        bitmap.set_bit(TypeId::STR);

        assert!(bitmap.is_subclass_of(TypeId::INT));
        assert!(bitmap.is_subclass_of(TypeId::FLOAT));
        assert!(bitmap.is_subclass_of(TypeId::STR));
        assert!(!bitmap.is_subclass_of(TypeId::LIST));
        assert_eq!(bitmap.count_bits(), 3);
    }

    #[test]
    fn test_bitmap_set_bit_overflow() {
        let mut bitmap = SubclassBitmap::new();

        // Type ID 256 is first user-defined type (in overflow region)
        let user_type = TypeId::from_raw(256);
        bitmap.set_bit(user_type);

        assert!(bitmap.is_subclass_of(user_type));
        assert!(bitmap.overflow.is_some());
    }

    #[test]
    fn test_bitmap_merge() {
        let mut parent1 = SubclassBitmap::new();
        parent1.set_bit(TypeId::INT);
        parent1.set_bit(TypeId::OBJECT);

        let mut parent2 = SubclassBitmap::new();
        parent2.set_bit(TypeId::STR);
        parent2.set_bit(TypeId::OBJECT);

        let mut child = SubclassBitmap::new();
        child.set_bit(TypeId::from_raw(300)); // Child's own bit
        child.merge(&parent1);
        child.merge(&parent2);

        // Child should have all parent bits
        assert!(child.is_subclass_of(TypeId::INT));
        assert!(child.is_subclass_of(TypeId::STR));
        assert!(child.is_subclass_of(TypeId::OBJECT));
        assert!(child.is_subclass_of(TypeId::from_raw(300)));
    }

    #[test]
    fn test_bitmap_from_parents() {
        let parent1 = builtin_type_bitmap(TypeId::INT);
        let parent2 = builtin_type_bitmap(TypeId::STR);

        let child_type = TypeId::from_raw(300);
        let child = SubclassBitmap::from_parents(child_type, [&parent1, &parent2].into_iter());

        assert!(child.is_subclass_of(child_type));
        assert!(child.is_subclass_of(TypeId::INT));
        assert!(child.is_subclass_of(TypeId::STR));
        assert!(child.is_subclass_of(TypeId::OBJECT));
    }

    #[test]
    fn test_bitmap_is_subclass_of_any() {
        let bitmap = builtin_type_bitmap(TypeId::INT);

        assert!(bitmap.is_subclass_of_any(&[TypeId::INT, TypeId::FLOAT]));
        assert!(bitmap.is_subclass_of_any(&[TypeId::OBJECT]));
        assert!(!bitmap.is_subclass_of_any(&[TypeId::STR, TypeId::LIST]));
        assert!(!bitmap.is_subclass_of_any(&[]));
    }

    #[test]
    fn test_bitmap_all_builtin_types() {
        // Test all built-in types fit in inline storage
        let builtins = [
            TypeId::NONE,
            TypeId::BOOL,
            TypeId::INT,
            TypeId::FLOAT,
            TypeId::STR,
            TypeId::BYTES,
            TypeId::BYTEARRAY,
            TypeId::LIST,
            TypeId::TUPLE,
            TypeId::DICT,
            TypeId::SET,
            TypeId::FROZENSET,
            TypeId::FUNCTION,
            TypeId::METHOD,
            TypeId::CLOSURE,
            TypeId::CODE,
            TypeId::MODULE,
            TypeId::TYPE,
            TypeId::OBJECT,
            TypeId::SLICE,
            TypeId::RANGE,
            TypeId::ITERATOR,
            TypeId::GENERATOR,
            TypeId::EXCEPTION,
            TypeId::BUILTIN_FUNCTION,
            TypeId::SUPER,
        ];

        let mut bitmap = SubclassBitmap::new();
        for &type_id in &builtins {
            bitmap.set_bit(type_id);
        }

        // Should all be in inline storage
        assert!(bitmap.overflow.is_none());

        for &type_id in &builtins {
            assert!(bitmap.is_subclass_of(type_id));
        }
    }

    #[test]
    fn test_bitmap_many_user_types() {
        let mut bitmap = SubclassBitmap::new();

        // Add 100 user-defined types
        for i in 256..356 {
            bitmap.set_bit(TypeId::from_raw(i));
        }

        assert!(bitmap.overflow.is_some());

        for i in 256..356 {
            assert!(bitmap.is_subclass_of(TypeId::from_raw(i)));
        }
    }

    #[test]
    fn test_bitmap_clone() {
        let mut original = SubclassBitmap::new();
        original.set_bit(TypeId::INT);
        original.set_bit(TypeId::from_raw(300));

        let cloned = original.clone();

        assert!(cloned.is_subclass_of(TypeId::INT));
        assert!(cloned.is_subclass_of(TypeId::from_raw(300)));
    }

    // =========================================================================
    // TypeCheckIC Tests
    // =========================================================================

    #[test]
    fn test_ic_new_is_empty() {
        let ic = TypeCheckIC::new();
        assert!(ic.is_empty());
        assert_eq!(ic.len(), 0);
    }

    #[test]
    fn test_ic_insert_and_lookup() {
        let mut ic = TypeCheckIC::new();

        ic.insert(ClassId(100), true);
        ic.insert(ClassId(101), false);

        assert_eq!(ic.lookup(ClassId(100)), Some(true));
        assert_eq!(ic.lookup(ClassId(101)), Some(false));
        assert_eq!(ic.lookup(ClassId(102)), None);
        assert_eq!(ic.len(), 2);
    }

    #[test]
    fn test_ic_full_replacement() {
        let mut ic = TypeCheckIC::new();

        // Fill cache
        ic.insert(ClassId(100), true);
        ic.insert(ClassId(101), true);
        ic.insert(ClassId(102), true);
        ic.insert(ClassId(103), true);

        assert_eq!(ic.len(), 4);

        // Insert fifth entry - should replace first
        ic.insert(ClassId(104), false);

        assert_eq!(ic.len(), 4);
        assert_eq!(ic.lookup(ClassId(104)), Some(false));
        // First entry was replaced
        assert_eq!(ic.lookup(ClassId(100)), None);
    }

    #[test]
    fn test_ic_clear() {
        let mut ic = TypeCheckIC::new();

        ic.insert(ClassId(100), true);
        ic.insert(ClassId(101), true);

        ic.clear();

        assert!(ic.is_empty());
        assert_eq!(ic.lookup(ClassId(100)), None);
    }

    #[test]
    fn test_ic_circular_replacement() {
        let mut ic = TypeCheckIC::new();

        // Fill and overflow multiple times
        for i in 0..12 {
            ic.insert(ClassId(i), i % 2 == 0);
        }

        // Only last 4 should be present
        assert_eq!(ic.len(), 4);
        assert!(ic.lookup(ClassId(8)).is_some());
        assert!(ic.lookup(ClassId(9)).is_some());
        assert!(ic.lookup(ClassId(10)).is_some());
        assert!(ic.lookup(ClassId(11)).is_some());
    }

    // =========================================================================
    // Type Builtin Function Tests
    // =========================================================================

    #[test]
    fn test_type_of_int() {
        let value = Value::int_unchecked(42);
        assert_eq!(type_of_value(value), TypeId::INT);
    }

    #[test]
    fn test_type_of_float() {
        let value = Value::from(3.14f64);
        assert_eq!(type_of_value(value), TypeId::FLOAT);
    }

    #[test]
    fn test_type_of_bool() {
        assert_eq!(type_of_value(Value::bool(true)), TypeId::BOOL);
        assert_eq!(type_of_value(Value::bool(false)), TypeId::BOOL);
    }

    #[test]
    fn test_type_of_none() {
        assert_eq!(type_of_value(Value::none()), TypeId::NONE);
    }

    #[test]
    fn test_isinstance_same_type() {
        let value = Value::int_unchecked(42);
        let int_class = ClassId(TypeId::INT.raw());

        let result = isinstance(value, int_class, |id| {
            if id == int_class {
                Some(builtin_type_bitmap(TypeId::INT))
            } else {
                None
            }
        });

        assert!(result);
    }

    #[test]
    fn test_isinstance_parent_type() {
        let value = Value::int_unchecked(42);
        let int_class = ClassId(TypeId::INT.raw());
        let object_class = ClassId(TypeId::OBJECT.raw());

        let result = isinstance(value, object_class, |id| {
            if id == int_class {
                Some(builtin_type_bitmap(TypeId::INT))
            } else {
                None
            }
        });

        assert!(result);
    }

    #[test]
    fn test_isinstance_unrelated_type() {
        let value = Value::int_unchecked(42);
        let int_class = ClassId(TypeId::INT.raw());
        let str_class = ClassId(TypeId::STR.raw());

        let result = isinstance(value, str_class, |id| {
            if id == int_class {
                Some(builtin_type_bitmap(TypeId::INT))
            } else {
                None
            }
        });

        assert!(!result);
    }

    #[test]
    fn test_isinstance_multi() {
        let value = Value::int_unchecked(42);
        let int_class = ClassId(TypeId::INT.raw());

        let classes = vec![
            ClassId(TypeId::STR.raw()),
            ClassId(TypeId::LIST.raw()),
            ClassId(TypeId::INT.raw()),
        ];

        let result = isinstance_multi(value, &classes, |id| {
            if id == int_class {
                Some(builtin_type_bitmap(TypeId::INT))
            } else {
                None
            }
        });

        assert!(result);
    }

    #[test]
    fn test_issubclass_same() {
        let int_class = ClassId(TypeId::INT.raw());

        let result = issubclass(int_class, int_class, |_| None);
        assert!(result);
    }

    #[test]
    fn test_issubclass_parent() {
        let int_class = ClassId(TypeId::INT.raw());
        let object_class = ClassId(TypeId::OBJECT.raw());

        let result = issubclass(int_class, object_class, |id| {
            if id == int_class {
                Some(builtin_type_bitmap(TypeId::INT))
            } else {
                None
            }
        });

        assert!(result);
    }

    #[test]
    fn test_issubclass_unrelated() {
        let int_class = ClassId(TypeId::INT.raw());
        let str_class = ClassId(TypeId::STR.raw());

        let result = issubclass(int_class, str_class, |id| {
            if id == int_class {
                Some(builtin_type_bitmap(TypeId::INT))
            } else {
                None
            }
        });

        assert!(!result);
    }

    #[test]
    fn test_issubclass_multi() {
        let int_class = ClassId(TypeId::INT.raw());
        let classes = vec![ClassId(TypeId::STR.raw()), ClassId(TypeId::OBJECT.raw())];

        let result = issubclass_multi(int_class, &classes, |id| {
            if id == int_class {
                Some(builtin_type_bitmap(TypeId::INT))
            } else {
                None
            }
        });

        assert!(result);
    }

    // =========================================================================
    // Builtin Type Bitmap Tests
    // =========================================================================

    #[test]
    fn test_builtin_type_bitmap() {
        let int_bitmap = builtin_type_bitmap(TypeId::INT);

        assert!(int_bitmap.is_subclass_of(TypeId::INT));
        assert!(int_bitmap.is_subclass_of(TypeId::OBJECT));
        assert!(!int_bitmap.is_subclass_of(TypeId::STR));
    }

    #[test]
    fn test_object_bitmap_static() {
        assert!(OBJECT_BITMAP.is_subclass_of(TypeId::OBJECT));
        assert!(!OBJECT_BITMAP.is_subclass_of(TypeId::INT));
    }

    // =========================================================================
    // Memory Layout Tests
    // =========================================================================

    #[test]
    fn test_bitmap_size() {
        // SubclassBitmap should be compact
        let size = std::mem::size_of::<SubclassBitmap>();
        assert!(
            size <= 32,
            "SubclassBitmap size ({} bytes) should be <= 32",
            size
        );
    }

    #[test]
    fn test_ic_size() {
        // TypeCheckIC should fit in a cache line
        let size = std::mem::size_of::<TypeCheckIC>();
        assert!(
            size <= 64,
            "TypeCheckIC size ({} bytes) should be <= 64",
            size
        );
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_bitmap_boundary_type_63() {
        // Type 63 is the last inline bit
        let mut bitmap = SubclassBitmap::new();
        bitmap.set_bit(TypeId::from_raw(63));

        assert!(bitmap.is_subclass_of(TypeId::from_raw(63)));
        assert!(bitmap.overflow.is_none());
    }

    #[test]
    fn test_bitmap_boundary_type_64() {
        // Type 64 is the first overflow bit
        let mut bitmap = SubclassBitmap::new();
        bitmap.set_bit(TypeId::from_raw(64));

        assert!(bitmap.is_subclass_of(TypeId::from_raw(64)));
        assert!(bitmap.overflow.is_some());
    }

    #[test]
    fn test_ic_lookup_none_class() {
        let mut ic = TypeCheckIC::new();
        ic.insert(ClassId::NONE, true);

        assert_eq!(ic.lookup(ClassId::NONE), Some(true));
    }

    #[test]
    fn test_isinstance_empty_tuple() {
        let value = Value::int_unchecked(42);

        let result = isinstance_multi(value, &[], |_| None);
        assert!(!result);
    }

    // =========================================================================
    // type_new Tests
    // =========================================================================

    use super::{ClassRegistry, SimpleClassRegistry, TypeCreationError, type_new};
    use crate::object::class::{ClassDict, PyClassObject};
    use prism_core::intern::intern;

    fn create_test_registry() -> SimpleClassRegistry {
        SimpleClassRegistry::new()
    }

    #[test]
    fn test_type_new_simple_class() {
        let registry = create_test_registry();
        let name = intern("MyClass");
        let namespace = ClassDict::new();

        let result = type_new(name, &[], &namespace, &registry);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.class.name().as_str(), "MyClass");
        assert!(result.class.bases().is_empty());
        assert!(
            result
                .flags
                .contains(crate::object::class::ClassFlags::INITIALIZED)
        );
    }

    #[test]
    fn test_type_new_with_init() {
        let registry = create_test_registry();
        let name = intern("InitClass");
        let namespace = ClassDict::new();

        // Add __init__ to namespace
        let init_name = intern("__init__");
        namespace.set(init_name, Value::int_unchecked(0));

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        assert!(
            result
                .flags
                .contains(crate::object::class::ClassFlags::HAS_INIT)
        );
    }

    #[test]
    fn test_type_new_with_new() {
        let registry = create_test_registry();
        let name = intern("NewClass");
        let namespace = ClassDict::new();

        // Add __new__ to namespace
        let new_name = intern("__new__");
        namespace.set(new_name, Value::int_unchecked(0));

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        assert!(
            result
                .flags
                .contains(crate::object::class::ClassFlags::HAS_NEW)
        );
    }

    #[test]
    fn test_type_new_with_slots() {
        let registry = create_test_registry();
        let name = intern("SlotsClass");
        let namespace = ClassDict::new();

        // Add __slots__ to namespace
        let slots_name = intern("__slots__");
        namespace.set(slots_name, Value::int_unchecked(0));

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        assert!(
            result
                .flags
                .contains(crate::object::class::ClassFlags::HAS_SLOTS)
        );
    }

    #[test]
    fn test_type_new_with_hash() {
        let registry = create_test_registry();
        let name = intern("HashClass");
        let namespace = ClassDict::new();

        // Add __hash__ to namespace
        let hash_name = intern("__hash__");
        namespace.set(hash_name, Value::int_unchecked(0));

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        assert!(
            result
                .flags
                .contains(crate::object::class::ClassFlags::HASHABLE)
        );
    }

    #[test]
    fn test_type_new_with_eq() {
        let registry = create_test_registry();
        let name = intern("EqClass");
        let namespace = ClassDict::new();

        // Add __eq__ to namespace
        let eq_name = intern("__eq__");
        namespace.set(eq_name, Value::int_unchecked(0));

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        assert!(
            result
                .flags
                .contains(crate::object::class::ClassFlags::HAS_EQ)
        );
    }

    #[test]
    fn test_type_new_with_del() {
        let registry = create_test_registry();
        let name = intern("DelClass");
        let namespace = ClassDict::new();

        // Add __del__ to namespace
        let del_name = intern("__del__");
        namespace.set(del_name, Value::int_unchecked(0));

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        assert!(
            result
                .flags
                .contains(crate::object::class::ClassFlags::HAS_FINALIZER)
        );
    }

    #[test]
    fn test_type_new_all_special_methods() {
        let registry = create_test_registry();
        let name = intern("AllSpecialClass");
        let namespace = ClassDict::new();

        // Add all special methods
        namespace.set(intern("__new__"), Value::int_unchecked(0));
        namespace.set(intern("__init__"), Value::int_unchecked(0));
        namespace.set(intern("__slots__"), Value::int_unchecked(0));
        namespace.set(intern("__hash__"), Value::int_unchecked(0));
        namespace.set(intern("__eq__"), Value::int_unchecked(0));
        namespace.set(intern("__del__"), Value::int_unchecked(0));

        let result = type_new(name, &[], &namespace, &registry).unwrap();

        use crate::object::class::ClassFlags;
        assert!(result.flags.contains(ClassFlags::HAS_NEW));
        assert!(result.flags.contains(ClassFlags::HAS_INIT));
        assert!(result.flags.contains(ClassFlags::HAS_SLOTS));
        assert!(result.flags.contains(ClassFlags::HASHABLE));
        assert!(result.flags.contains(ClassFlags::HAS_EQ));
        assert!(result.flags.contains(ClassFlags::HAS_FINALIZER));
    }

    #[test]
    fn test_type_new_empty_name_error() {
        let registry = create_test_registry();
        let name = intern("");
        let namespace = ClassDict::new();

        let result = type_new(name, &[], &namespace, &registry);
        assert!(result.is_err());

        match result.unwrap_err() {
            TypeCreationError::InvalidName { name } => {
                assert_eq!(name, "");
            }
            _ => panic!("Expected InvalidName error"),
        }
    }

    #[test]
    fn test_type_new_base_not_found() {
        let registry = create_test_registry();
        let name = intern("DerivedClass");
        let namespace = ClassDict::new();
        let fake_base = ClassId(99999);

        let result = type_new(name, &[fake_base], &namespace, &registry);
        assert!(result.is_err());

        match result.unwrap_err() {
            TypeCreationError::BaseNotFound { class_id } => {
                assert_eq!(class_id, fake_base);
            }
            _ => panic!("Expected BaseNotFound error"),
        }
    }

    #[test]
    fn test_type_new_final_base_error() {
        let registry = create_test_registry();

        // Create a final base class
        let mut base = PyClassObject::new_simple(intern("FinalBase"));
        base.mark_final();
        let base_id = base.class_id();
        let base = Arc::new(base);

        // Register it with a bitmap
        let bitmap = SubclassBitmap::new();
        registry.register(base, bitmap);

        // Try to inherit from final class
        let name = intern("DerivedFromFinal");
        let namespace = ClassDict::new();

        let result = type_new(name, &[base_id], &namespace, &registry);
        assert!(result.is_err());

        match result.unwrap_err() {
            TypeCreationError::FinalBase { class_name } => {
                assert_eq!(class_name, "FinalBase");
            }
            _ => panic!("Expected FinalBase error"),
        }
    }

    #[test]
    fn test_type_new_namespace_copied() {
        let registry = create_test_registry();
        let name = intern("AttrClass");
        let namespace = ClassDict::new();

        // Add some attributes
        let attr1 = intern("method1");
        let attr2 = intern("class_var");
        namespace.set(attr1.clone(), Value::int_unchecked(42));
        namespace.set(attr2.clone(), Value::int_unchecked(1));

        let result = type_new(name, &[], &namespace, &registry).unwrap();

        // Verify attributes were copied
        assert_eq!(
            result.class.get_attr(&attr1),
            Some(Value::int_unchecked(42))
        );
        assert_eq!(result.class.get_attr(&attr2), Some(Value::int_unchecked(1)));
    }

    #[test]
    fn test_type_new_bitmap_has_object() {
        let registry = create_test_registry();
        let name = intern("BitmapClass");
        let namespace = ClassDict::new();

        let result = type_new(name, &[], &namespace, &registry).unwrap();

        // All classes should be subclass of object
        assert!(result.bitmap.is_subclass_of(TypeId::OBJECT));
    }

    #[test]
    fn test_type_new_bitmap_has_self() {
        let registry = create_test_registry();
        let name = intern("SelfBitmapClass");
        let namespace = ClassDict::new();

        let result = type_new(name, &[], &namespace, &registry).unwrap();

        // Class should be in its own bitmap
        let self_type_id = TypeId::from_raw(result.class.class_id().0);
        assert!(result.bitmap.is_subclass_of(self_type_id));
    }

    #[test]
    fn test_type_new_with_inheritance() {
        let registry = create_test_registry();

        // Create parent class
        let parent = PyClassObject::new_simple(intern("Parent"));
        let parent_id = parent.class_id();
        let parent = Arc::new(parent);

        // Create parent bitmap
        let mut parent_bitmap = SubclassBitmap::new();
        parent_bitmap.set_bit(TypeId::from_raw(parent_id.0));
        parent_bitmap.set_bit(TypeId::OBJECT);
        registry.register(parent, parent_bitmap.clone());

        // Create child class
        let name = intern("Child");
        let namespace = ClassDict::new();

        let result = type_new(name, &[parent_id], &namespace, &registry).unwrap();

        // Child should have parent in bitmap
        assert!(result.bitmap.is_subclass_of(TypeId::from_raw(parent_id.0)));
        assert!(result.bitmap.is_subclass_of(TypeId::OBJECT));
    }

    #[test]
    fn test_type_creation_error_display() {
        let err = TypeCreationError::FinalBase {
            class_name: "MyFinal".to_string(),
        };
        assert_eq!(err.to_string(), "cannot subclass final class 'MyFinal'");

        let err = TypeCreationError::MroError {
            message: "conflict".to_string(),
        };
        assert_eq!(err.to_string(), "MRO error: conflict");

        let err = TypeCreationError::BaseNotFound {
            class_id: ClassId(123),
        };
        assert!(err.to_string().contains("123"));

        let err = TypeCreationError::InvalidName {
            name: "".to_string(),
        };
        assert_eq!(err.to_string(), "invalid class name: ''");
    }

    #[test]
    fn test_simple_registry_operations() {
        let registry = SimpleClassRegistry::new();

        // Create and register a class
        let class = Arc::new(PyClassObject::new_simple(intern("TestClass")));
        let class_id = class.class_id();
        let bitmap = SubclassBitmap::new();

        registry.register(class.clone(), bitmap.clone());

        // Test retrieval
        assert!(registry.get_class(class_id).is_some());
        assert!(registry.get_bitmap(class_id).is_some());

        // Test not found
        assert!(registry.get_class(ClassId(99999)).is_none());
        assert!(registry.get_bitmap(ClassId(99999)).is_none());
    }
}
