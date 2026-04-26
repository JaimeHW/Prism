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

use crate::allocation_context::alloc_value_in_current_heap_or_box;
use crate::object::mro::ClassId;
use crate::object::type_obj::TypeId;
use prism_core::Value;

// =============================================================================
// Subclass Bitmap - O(1) Subclass Testing
// =============================================================================

/// Number of bits in inline storage (covers all built-in types).
const INLINE_BITS: usize = 64;

/// Initial number of u64 words in overflow storage.
///
/// This covers type ids 64..1088 without reallocation while still allowing the
/// bitmap to grow for long-lived processes that create many heap types.
const INITIAL_OVERFLOW_WORDS: usize = 16;

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
/// └── overflow: Option<Box<[u64]>> (16 bytes)
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
    /// Lazily allocated and grown only when needed.
    overflow: Option<Box<[u64]>>,
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
            let overflow = self.ensure_overflow_words(other_overflow.len());
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

        match &self.overflow {
            Some(overflow) if word_idx < overflow.len() => {
                (overflow[word_idx] & (1u64 << bit_idx)) != 0
            }
            None => false,
            Some(_) => false,
        }
    }

    /// Set bit in overflow storage.
    #[cold]
    fn set_overflow_bit(&mut self, bit: usize) {
        let overflow_bit = bit - INLINE_BITS;
        let word_idx = overflow_bit / 64;
        let bit_idx = overflow_bit % 64;

        let overflow = self.ensure_overflow_words(word_idx + 1);
        overflow[word_idx] |= 1u64 << bit_idx;
    }

    #[cold]
    fn ensure_overflow_words(&mut self, min_words: usize) -> &mut [u64] {
        debug_assert!(min_words > 0);

        let needs_grow = self
            .overflow
            .as_ref()
            .map_or(true, |overflow| overflow.len() < min_words);
        if needs_grow {
            let new_len = min_words.next_power_of_two().max(INITIAL_OVERFLOW_WORDS);
            let mut new_overflow = vec![0u64; new_len].into_boxed_slice();
            if let Some(old_overflow) = self.overflow.take() {
                new_overflow[..old_overflow.len()].copy_from_slice(&old_overflow);
            }
            self.overflow = Some(new_overflow);
        }

        self.overflow.as_deref_mut().expect("overflow allocated")
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
        let ptr = value
            .as_object_ptr()
            .expect("Value::is_object() must imply an object pointer");
        let header = ptr as *const crate::object::ObjectHeader;
        // SAFETY: every heap object starts with ObjectHeader at offset 0.
        unsafe { (*header).type_id }
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
        bitmap.is_subclass_of(class_id_to_type_id(class_id))
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
            if bitmap.is_subclass_of(class_id_to_type_id(class_id)) {
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
        bitmap.is_subclass_of(class_id_to_type_id(parent_class))
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
            if bitmap.is_subclass_of(class_id_to_type_id(parent_class)) {
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
    if type_id == TypeId::BOOL {
        bitmap.set_bit(TypeId::INT);
    }
    bitmap
}

/// Convert a class identifier into the corresponding runtime type id.
///
/// Heap classes reuse their allocated type id directly. Builtin class ids
/// mirror their [`TypeId`] values, including [`ClassId::OBJECT`].
#[inline]
pub fn class_id_to_type_id(class_id: ClassId) -> TypeId {
    if class_id == ClassId::OBJECT {
        TypeId::OBJECT
    } else {
        TypeId::from_raw(class_id.0)
    }
}

/// Get the canonical MRO for a built-in type when used as a base class.
#[inline]
pub fn builtin_class_mro(type_id: TypeId) -> Vec<ClassId> {
    match type_id {
        TypeId::OBJECT => vec![ClassId::OBJECT],
        TypeId::BOOL => vec![
            ClassId(TypeId::BOOL.raw()),
            ClassId(TypeId::INT.raw()),
            ClassId::OBJECT,
        ],
        _ => vec![ClassId(type_id.raw()), ClassId::OBJECT],
    }
}

#[inline]
fn builtin_base_type_error(type_id: TypeId) -> Option<TypeCreationError> {
    match type_id {
        TypeId::BOOL => Some(TypeCreationError::UnacceptableBaseType {
            class_name: type_id.name().to_string(),
        }),
        _ => None,
    }
}

// =============================================================================
// Type Creation (3-arg form)
// =============================================================================

use crate::object::class::{ClassDict, ClassFlags, PyClassObject};
use crate::object::descriptor::{ClassMethodDescriptor, SlotDescriptor, StaticMethodDescriptor};
use crate::object::registry::global_registry;
use crate::object::type_obj::{TypeFlags, TypeObject};
use crate::types::list::ListObject;
use crate::types::string::value_as_string_ref;
use crate::types::tuple::TupleObject;
use arc_swap::ArcSwapOption;
use parking_lot::Mutex;
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

/// Error during dynamic class creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeCreationError {
    /// Base class is marked final and cannot be subclassed.
    FinalBase { class_name: String },

    /// Built-in type cannot be used as a base class.
    UnacceptableBaseType { class_name: String },

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
            TypeCreationError::UnacceptableBaseType { class_name } => {
                write!(f, "type '{}' is not an acceptable base type", class_name)
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
    type_new_with_metaclass(name, bases, namespace, Value::none(), registry)
}

/// Create a new class with an explicit metaclass selection.
pub fn type_new_with_metaclass<R>(
    name: InternedString,
    bases: &[ClassId],
    namespace: &ClassDict,
    metaclass: Value,
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
    for &base_id in bases {
        if base_id.0 < TypeId::FIRST_USER_TYPE {
            if let Some(err) = builtin_base_type_error(class_id_to_type_id(base_id)) {
                return Err(err);
            }
            continue;
        }

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
            if id.0 < TypeId::FIRST_USER_TYPE {
                Some(builtin_class_mro(class_id_to_type_id(id)).into())
            } else {
                registry
                    .get_class(id)
                    .map(|c| c.mro().iter().copied().collect())
            }
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
        if base_id.0 < TypeId::FIRST_USER_TYPE {
            bitmap.merge(&builtin_type_bitmap(class_id_to_type_id(base_id)));
        } else if let Some(parent_bitmap) = registry.get_bitmap(base_id) {
            bitmap.merge(&parent_bitmap);
        }
    }

    // 5. Detect special methods in namespace
    let mut flags = ClassFlags::INITIALIZED;

    if bases.iter().any(|&base_id| {
        if base_id.0 == TypeId::TYPE.raw() {
            return true;
        }
        if base_id.0 < TypeId::FIRST_USER_TYPE {
            return false;
        }
        registry
            .get_class(base_id)
            .map(|base| base.flags().contains(ClassFlags::METACLASS))
            .unwrap_or(false)
    }) {
        flags |= ClassFlags::METACLASS;
    }

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
    let slot_names = if let Some(slots_value) = namespace.get(&slots_name) {
        flags |= ClassFlags::HAS_SLOTS;
        Some(extract_slot_names(slots_value, namespace)?)
    } else {
        None
    };

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

    // 6. Copy namespace to class dict and record the resolved metaclass
    class.set_metaclass(metaclass);
    namespace.for_each(|name, value| {
        class.set_attr(name.clone(), normalize_class_namespace_value(name, value));
    });

    if let Some(slot_names) = slot_names {
        class.set_slots(slot_names.clone());
        for (index, slot_name) in slot_names.into_iter().enumerate() {
            let descriptor = SlotDescriptor::read_write(
                slot_name.clone(),
                index as u16,
                SlotDescriptor::compute_offset(index as u16),
            );
            class.set_attr(slot_name, alloc_value_in_current_heap_or_box(descriptor));
        }
    }

    // Set detected flags
    class.set_flags(flags);
    class.rebuild_method_layout(|id| registry.get_class(id));

    let class = Arc::new(class);

    Ok(TypeNewResult {
        class,
        bitmap,
        flags,
    })
}

fn extract_slot_names(
    slots_value: Value,
    namespace: &ClassDict,
) -> Result<Vec<InternedString>, TypeCreationError> {
    let mut names = Vec::new();
    collect_slot_names(slots_value, &mut names)?;

    for i in 0..names.len() {
        if names[i].as_str().is_empty() {
            return Err(TypeCreationError::SlotsConflict {
                message: "slot names must be non-empty strings".to_string(),
            });
        }

        if namespace.contains(&names[i]) {
            return Err(TypeCreationError::SlotsConflict {
                message: format!(
                    "'{}' in __slots__ conflicts with class variable",
                    names[i].as_str()
                ),
            });
        }

        if names[..i].iter().any(|existing| existing == &names[i]) {
            return Err(TypeCreationError::SlotsConflict {
                message: format!("duplicate slot name '{}'", names[i].as_str()),
            });
        }
    }

    Ok(names)
}

fn collect_slot_names(
    value: Value,
    out: &mut Vec<InternedString>,
) -> Result<(), TypeCreationError> {
    if let Some(name) = value_as_string_ref(value) {
        out.push(prism_core::intern::intern(name.as_str()));
        return Ok(());
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(invalid_slots_value());
    };
    let type_id = unsafe { (*(ptr as *const crate::object::ObjectHeader)).type_id };
    match type_id {
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            for item in tuple.iter().copied() {
                collect_single_slot_name(item, out)?;
            }
            Ok(())
        }
        TypeId::LIST => {
            let list = unsafe { &*(ptr as *const ListObject) };
            for item in list.iter().copied() {
                collect_single_slot_name(item, out)?;
            }
            Ok(())
        }
        _ => Err(invalid_slots_value()),
    }
}

fn collect_single_slot_name(
    value: Value,
    out: &mut Vec<InternedString>,
) -> Result<(), TypeCreationError> {
    let Some(name) = value_as_string_ref(value) else {
        return Err(TypeCreationError::SlotsConflict {
            message: "__slots__ items must be strings".to_string(),
        });
    };
    out.push(prism_core::intern::intern(name.as_str()));
    Ok(())
}

fn invalid_slots_value() -> TypeCreationError {
    TypeCreationError::SlotsConflict {
        message: "__slots__ must be a string or an iterable of strings".to_string(),
    }
}

#[inline]
fn normalize_class_namespace_value(name: &InternedString, value: Value) -> Value {
    let Some(kind) = implicit_descriptor_kind(name.as_str()) else {
        return value;
    };

    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };

    let header = ptr as *const crate::object::ObjectHeader;
    let type_id = unsafe { (*header).type_id };
    match type_id {
        TypeId::FUNCTION | TypeId::CLOSURE | TypeId::BUILTIN_FUNCTION => {
            wrap_implicit_descriptor(value, kind)
        }
        _ => value,
    }
}

#[derive(Clone, Copy)]
enum ImplicitDescriptorKind {
    StaticMethod,
    ClassMethod,
}

#[inline]
fn implicit_descriptor_kind(name: &str) -> Option<ImplicitDescriptorKind> {
    match name {
        "__new__" => Some(ImplicitDescriptorKind::StaticMethod),
        "__init_subclass__" | "__class_getitem__" => Some(ImplicitDescriptorKind::ClassMethod),
        _ => None,
    }
}

#[inline]
fn wrap_implicit_descriptor(value: Value, kind: ImplicitDescriptorKind) -> Value {
    match kind {
        ImplicitDescriptorKind::StaticMethod => {
            let descriptor = StaticMethodDescriptor::new(value);
            alloc_value_in_current_heap_or_box(descriptor)
        }
        ImplicitDescriptorKind::ClassMethod => {
            let descriptor = ClassMethodDescriptor::new(value);
            alloc_value_in_current_heap_or_box(descriptor)
        }
    }
}

const HEAP_TYPE_FAST_TABLE_CAPACITY: usize = 1 << 16;

#[derive(Debug)]
struct PublishedClassEntry {
    class: Arc<PyClassObject>,
    bitmap: SubclassBitmap,
}

#[derive(Debug)]
struct OverflowClassSlot {
    entry: ArcSwapOption<PublishedClassEntry>,
}

impl OverflowClassSlot {
    fn empty() -> Self {
        Self {
            entry: ArcSwapOption::from(None::<Arc<PublishedClassEntry>>),
        }
    }
}

/// Dense published heap-type registry for user-defined classes.
///
/// The common path is a direct indexed load by `TypeId`, avoiding the global
/// hash table and lock traffic that previously sat on every heap-type lookup.
pub struct PublishedClassRegistry {
    entries: Box<[ArcSwapOption<PublishedClassEntry>]>,
    overflow: Mutex<Vec<OverflowClassSlot>>,
    registered: AtomicU32,
}

impl PublishedClassRegistry {
    pub fn new() -> Self {
        let mut entries = Vec::with_capacity(HEAP_TYPE_FAST_TABLE_CAPACITY);
        entries.resize_with(HEAP_TYPE_FAST_TABLE_CAPACITY, || {
            ArcSwapOption::from(None::<Arc<PublishedClassEntry>>)
        });
        Self {
            entries: entries.into_boxed_slice(),
            overflow: Mutex::new(Vec::new()),
            registered: AtomicU32::new(0),
        }
    }

    fn load_entry(&self, id: ClassId) -> Option<Arc<PublishedClassEntry>> {
        let index = id.0 as usize;
        if let Some(slot) = self.entries.get(index) {
            return slot.load_full();
        }

        let overflow_index = index - self.entries.len();
        let overflow = self.overflow.lock();
        overflow.get(overflow_index)?.entry.load_full()
    }

    fn store_entry(&self, id: ClassId, entry: Arc<PublishedClassEntry>) {
        let index = id.0 as usize;
        if let Some(slot) = self.entries.get(index) {
            if slot.swap(Some(Arc::clone(&entry))).is_none() {
                self.registered.fetch_add(1, Ordering::Relaxed);
            }
            return;
        }

        let overflow_index = index - self.entries.len();
        let mut overflow = self.overflow.lock();
        if overflow.len() <= overflow_index {
            overflow.resize_with(overflow_index + 1, OverflowClassSlot::empty);
        }
        if overflow[overflow_index].entry.swap(Some(entry)).is_none() {
            self.registered.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Register or refresh a heap class publication.
    pub fn register(&self, class: Arc<PyClassObject>, bitmap: SubclassBitmap) {
        publish_heap_type_object(class.as_ref());
        class.rebuild_method_layout(global_class);
        let id = class.class_id();
        let entry = Arc::new(PublishedClassEntry { class, bitmap });
        self.store_entry(id, entry);
    }

    /// Remove a registered class publication.
    pub fn unregister(&self, id: ClassId) {
        let index = id.0 as usize;
        let removed = if let Some(slot) = self.entries.get(index) {
            slot.swap(None).is_some()
        } else {
            let overflow_index = index - self.entries.len();
            let overflow = self.overflow.lock();
            overflow
                .get(overflow_index)
                .and_then(|slot| slot.entry.swap(None))
                .is_some()
        };
        if removed {
            self.registered.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub fn get_class(&self, id: ClassId) -> Option<Arc<PyClassObject>> {
        self.load_entry(id).map(|entry| Arc::clone(&entry.class))
    }

    pub fn get_bitmap(&self, id: ClassId) -> Option<SubclassBitmap> {
        self.load_entry(id).map(|entry| entry.bitmap.clone())
    }

    pub fn class_version(&self, id: ClassId) -> Option<u64> {
        self.load_entry(id)
            .map(|entry| entry.class.method_layout_version())
    }

    pub fn refresh_layouts_for_hierarchy(&self, root_type_id: TypeId) {
        let mut affected = Vec::new();

        for slot in self.entries.iter() {
            if let Some(entry) = slot.load_full()
                && entry.bitmap.is_subclass_of(root_type_id)
            {
                affected.push(entry.class.clone());
            }
        }

        let overflow = self.overflow.lock();
        for slot in overflow.iter() {
            if let Some(entry) = slot.entry.load_full()
                && entry.bitmap.is_subclass_of(root_type_id)
            {
                affected.push(entry.class.clone());
            }
        }
        drop(overflow);

        for class in affected {
            class.rebuild_method_layout(global_class);
            class.bump_method_layout_version();
        }
    }

    pub fn direct_subclasses_of(&self, root_type_id: TypeId) -> Vec<Arc<PyClassObject>> {
        let mut subclasses = Vec::new();

        for slot in self.entries.iter() {
            if let Some(entry) = slot.load_full()
                && class_directly_inherits(entry.class.as_ref(), root_type_id)
            {
                subclasses.push(entry.class.clone());
            }
        }

        let overflow = self.overflow.lock();
        for slot in overflow.iter() {
            if let Some(entry) = slot.entry.load_full()
                && class_directly_inherits(entry.class.as_ref(), root_type_id)
            {
                subclasses.push(entry.class.clone());
            }
        }

        subclasses
    }

    pub fn len(&self) -> usize {
        self.registered.load(Ordering::Relaxed) as usize
    }
}

#[inline]
fn class_directly_inherits(class: &PyClassObject, root_type_id: TypeId) -> bool {
    if class.class_type_id() == root_type_id {
        return false;
    }

    if class.bases().is_empty() {
        return root_type_id == TypeId::OBJECT;
    }

    class
        .bases()
        .iter()
        .any(|&base| class_id_to_type_id(base) == root_type_id)
}

fn publish_heap_type_object(class: &PyClassObject) {
    let registry = global_registry();
    let type_id = class.class_type_id();
    if registry.contains(type_id) {
        return;
    }

    let base = class
        .bases()
        .iter()
        .copied()
        .find_map(|base_id| registry.get(class_id_to_type_id(base_id)));
    let type_object = Box::leak(Box::new(TypeObject::new(
        type_id,
        class.name().clone(),
        base,
        0,
        TypeFlags::HEAPTYPE,
    )));
    registry.register(type_id, type_object);
}

impl Default for PublishedClassRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassRegistry for PublishedClassRegistry {
    fn get_class(&self, id: ClassId) -> Option<Arc<PyClassObject>> {
        self.get_class(id)
    }

    fn get_bitmap(&self, id: ClassId) -> Option<SubclassBitmap> {
        self.get_bitmap(id)
    }

    fn register_class(&self, class: Arc<PyClassObject>) {
        let bitmap = self
            .get_bitmap(class.class_id())
            .unwrap_or_else(|| SubclassBitmap::for_type(class.class_type_id()));
        self.register(class, bitmap);
    }
}

/// Simple in-memory class registry for tests and isolated construction flows.
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

    /// Remove a registered class and its bitmap.
    pub fn unregister(&self, id: ClassId) {
        self.classes.write().unwrap().remove(&id);
        self.bitmaps.write().unwrap().remove(&id);
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

static GLOBAL_CLASS_REGISTRY: OnceLock<PublishedClassRegistry> = OnceLock::new();

/// Get the global registry for heap-defined Python classes.
#[inline]
pub fn global_class_registry() -> &'static PublishedClassRegistry {
    GLOBAL_CLASS_REGISTRY.get_or_init(PublishedClassRegistry::new)
}

/// Look up a user-defined class object by class id.
#[inline]
pub fn global_class(id: ClassId) -> Option<Arc<PyClassObject>> {
    if id.0 < TypeId::FIRST_USER_TYPE {
        None
    } else {
        global_class_registry().get_class(id)
    }
}

/// Look up subclass metadata for a class id, including built-in classes.
#[inline]
pub fn global_class_bitmap(id: ClassId) -> Option<SubclassBitmap> {
    if id.0 < TypeId::FIRST_USER_TYPE {
        Some(builtin_type_bitmap(class_id_to_type_id(id)))
    } else {
        global_class_registry().get_bitmap(id)
    }
}

/// Register a newly created heap class and its subclass bitmap globally.
#[inline]
pub fn register_global_class(class: Arc<PyClassObject>, bitmap: SubclassBitmap) {
    global_class_registry().register(class, bitmap);
}

/// Remove a previously registered heap class from the global registry.
#[inline]
pub fn unregister_global_class(id: ClassId) {
    global_class_registry().unregister(id);
}

/// Get the published method-layout version for a heap class.
#[inline]
pub fn global_class_version(id: ClassId) -> Option<u64> {
    (id.0 >= TypeId::FIRST_USER_TYPE)
        .then(|| global_class_registry().class_version(id))
        .flatten()
}

/// Return the currently published direct heap subclasses of a type.
#[inline]
pub fn global_direct_subclasses(type_id: TypeId) -> Vec<Arc<PyClassObject>> {
    global_class_registry().direct_subclasses_of(type_id)
}

/// Refresh published method layouts for a class and its registered subclasses.
#[inline]
pub fn refresh_global_class_layouts(type_id: TypeId) {
    if type_id.raw() >= TypeId::FIRST_USER_TYPE {
        global_class_registry().refresh_layouts_for_hierarchy(type_id);
    }
}
