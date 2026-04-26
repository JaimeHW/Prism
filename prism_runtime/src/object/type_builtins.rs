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
use crate::object::descriptor::{ClassMethodDescriptor, StaticMethodDescriptor};
use crate::object::registry::global_registry;
use crate::object::type_obj::{TypeFlags, TypeObject};
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

    // 6. Copy namespace to class dict and record the resolved metaclass
    class.set_metaclass(metaclass);
    namespace.for_each(|name, value| {
        class.set_attr(name.clone(), normalize_class_namespace_value(name, value));
    });

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

    pub fn len(&self) -> usize {
        self.registered.load(Ordering::Relaxed) as usize
    }
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

/// Refresh published method layouts for a class and its registered subclasses.
#[inline]
pub fn refresh_global_class_layouts(type_id: TypeId) {
    if type_id.raw() >= TypeId::FIRST_USER_TYPE {
        global_class_registry().refresh_layouts_for_hierarchy(type_id);
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
    fn test_bitmap_grows_for_high_heap_type_ids() {
        let mut bitmap = SubclassBitmap::new();
        let parent = TypeId::from_raw(1_500);
        let child = TypeId::from_raw(4_096);

        bitmap.set_bit(parent);
        bitmap.set_bit(child);

        assert!(bitmap.is_subclass_of(parent));
        assert!(bitmap.is_subclass_of(child));
        assert!(!bitmap.is_subclass_of(TypeId::from_raw(4_095)));
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
    fn test_bitmap_merge_preserves_high_heap_type_ids() {
        let high_parent_type = TypeId::from_raw(2_048);
        let high_child_type = TypeId::from_raw(4_096);

        let mut parent = SubclassBitmap::new();
        parent.set_bit(high_parent_type);
        parent.set_bit(TypeId::OBJECT);

        let mut child = SubclassBitmap::new();
        child.set_bit(high_child_type);
        child.merge(&parent);

        assert!(child.is_subclass_of(high_child_type));
        assert!(child.is_subclass_of(high_parent_type));
        assert!(child.is_subclass_of(TypeId::OBJECT));
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
    fn test_class_id_to_type_id_maps_object_sentinel() {
        assert_eq!(class_id_to_type_id(ClassId::OBJECT), TypeId::OBJECT);
        assert_eq!(
            class_id_to_type_id(ClassId(TypeId::NONE.raw())),
            TypeId::NONE
        );
        assert_eq!(class_id_to_type_id(ClassId(TypeId::INT.raw())), TypeId::INT);
        assert_eq!(
            class_id_to_type_id(ClassId(TypeId::FIRST_USER_TYPE)),
            TypeId::from_raw(TypeId::FIRST_USER_TYPE)
        );
    }

    #[test]
    fn test_builtin_none_mro_preserves_none_type() {
        assert_eq!(
            builtin_class_mro(TypeId::NONE),
            vec![ClassId(TypeId::NONE.raw()), ClassId::OBJECT]
        );
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
    fn test_isinstance_parent_type_accepts_object_sentinel() {
        let value = Value::int_unchecked(42);
        let int_class = ClassId(TypeId::INT.raw());

        let result = isinstance(value, ClassId::OBJECT, |id| {
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
    fn test_issubclass_parent_accepts_object_sentinel() {
        let int_class = ClassId(TypeId::INT.raw());

        let result = issubclass(int_class, ClassId::OBJECT, |id| {
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

    use super::{
        ClassRegistry, SimpleClassRegistry, TypeCreationError, type_new, type_new_with_metaclass,
    };
    use crate::object::class::{ClassDict, ClassFlags, PyClassObject};
    use crate::object::descriptor::{ClassMethodDescriptor, StaticMethodDescriptor};
    use crate::object::registry::global_registry;
    use crate::object::type_obj::TypeFlags;
    use crate::types::function::FunctionObject;
    use prism_code::CodeObject;
    use prism_core::intern::intern;
    use std::sync::Arc;

    fn create_test_registry() -> SimpleClassRegistry {
        SimpleClassRegistry::new()
    }

    fn bitmap_for_class(class: &PyClassObject) -> SubclassBitmap {
        let mut bitmap = SubclassBitmap::new();
        for &class_id in class.mro() {
            bitmap.set_bit(class_id_to_type_id(class_id));
        }
        bitmap
    }

    fn test_function_value(name: &str) -> Value {
        let code = Arc::new(CodeObject::new(name, "<test>"));
        let function = FunctionObject::new(code, Arc::from(name), None, None);
        Value::object_ptr(Box::into_raw(Box::new(function)) as *const ())
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
    fn test_type_new_records_explicit_metaclass() {
        let registry = create_test_registry();
        let name = intern("MetaBoundClass");
        let namespace = ClassDict::new();
        let explicit_metaclass = Value::int_unchecked(123);

        let result =
            type_new_with_metaclass(name, &[], &namespace, explicit_metaclass, &registry).unwrap();

        assert_eq!(result.class.metaclass(), explicit_metaclass);
    }

    #[test]
    fn test_type_new_marks_classes_derived_from_type_as_metaclasses() {
        let registry = create_test_registry();
        let name = intern("MetaClass");
        let namespace = ClassDict::new();

        let result = type_new(name, &[ClassId(TypeId::TYPE.raw())], &namespace, &registry).unwrap();

        assert!(result.class.flags().contains(ClassFlags::METACLASS));
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
    fn test_type_new_wraps_function_dunder_new_as_staticmethod() {
        let registry = create_test_registry();
        let name = intern("WrappedNewClass");
        let namespace = ClassDict::new();
        let function_value = test_function_value("__new__");
        namespace.set(intern("__new__"), function_value);

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        let stored = result
            .class
            .get_attr(&intern("__new__"))
            .expect("__new__ should be present on the class");
        let ptr = stored
            .as_object_ptr()
            .expect("normalized __new__ should be a descriptor object");
        let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };

        assert_eq!(
            unsafe { (*(ptr as *const crate::object::ObjectHeader)).type_id },
            TypeId::STATICMETHOD
        );
        assert_eq!(descriptor.function(), function_value);
    }

    #[test]
    fn test_type_new_implicit_descriptor_uses_bound_heap() {
        let heap = prism_gc::heap::GcHeap::with_defaults();
        let _binding = crate::allocation_context::RuntimeHeapBinding::register(&heap);
        let registry = create_test_registry();
        let name = intern("HeapWrappedNewClass");
        let namespace = ClassDict::new();
        namespace.set(intern("__new__"), test_function_value("__new__"));

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        let stored = result
            .class
            .get_attr(&intern("__new__"))
            .expect("__new__ should be present on the class");
        let ptr = stored
            .as_object_ptr()
            .expect("normalized __new__ should be a descriptor object");

        assert!(heap.contains(ptr));
    }

    #[test]
    fn test_type_new_wraps_function_dunder_init_subclass_as_classmethod() {
        let registry = create_test_registry();
        let name = intern("WrappedInitSubclassClass");
        let namespace = ClassDict::new();
        let function_value = test_function_value("__init_subclass__");
        namespace.set(intern("__init_subclass__"), function_value);

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        let stored = result
            .class
            .get_attr(&intern("__init_subclass__"))
            .expect("__init_subclass__ should be present on the class");
        let ptr = stored
            .as_object_ptr()
            .expect("normalized __init_subclass__ should be a descriptor object");
        let descriptor = unsafe { &*(ptr as *const ClassMethodDescriptor) };

        assert_eq!(
            unsafe { (*(ptr as *const crate::object::ObjectHeader)).type_id },
            TypeId::CLASSMETHOD
        );
        assert_eq!(descriptor.function(), function_value);
    }

    #[test]
    fn test_type_new_wraps_function_dunder_class_getitem_as_classmethod() {
        let registry = create_test_registry();
        let name = intern("WrappedClassGetitemClass");
        let namespace = ClassDict::new();
        let function_value = test_function_value("__class_getitem__");
        namespace.set(intern("__class_getitem__"), function_value);

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        let stored = result
            .class
            .get_attr(&intern("__class_getitem__"))
            .expect("__class_getitem__ should be present on the class");
        let ptr = stored
            .as_object_ptr()
            .expect("normalized __class_getitem__ should be a descriptor object");
        let descriptor = unsafe { &*(ptr as *const ClassMethodDescriptor) };

        assert_eq!(
            unsafe { (*(ptr as *const crate::object::ObjectHeader)).type_id },
            TypeId::CLASSMETHOD
        );
        assert_eq!(descriptor.function(), function_value);
    }

    #[test]
    fn test_type_new_preserves_explicit_staticmethod_dunder_new() {
        let registry = create_test_registry();
        let name = intern("ExplicitStaticNewClass");
        let namespace = ClassDict::new();
        let function_value = test_function_value("__new__");
        let descriptor = StaticMethodDescriptor::new(function_value);
        let descriptor_value = Value::object_ptr(Box::into_raw(Box::new(descriptor)) as *const ());
        namespace.set(intern("__new__"), descriptor_value);

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        let stored = result
            .class
            .get_attr(&intern("__new__"))
            .expect("__new__ should be present on the class");

        assert_eq!(stored, descriptor_value);
    }

    #[test]
    fn test_type_new_preserves_explicit_classmethod_dunder_init_subclass() {
        let registry = create_test_registry();
        let name = intern("ExplicitClassMethodInitSubclassClass");
        let namespace = ClassDict::new();
        let function_value = test_function_value("__init_subclass__");
        let descriptor = ClassMethodDescriptor::new(function_value);
        let descriptor_value = Value::object_ptr(Box::into_raw(Box::new(descriptor)) as *const ());
        namespace.set(intern("__init_subclass__"), descriptor_value);

        let result = type_new(name, &[], &namespace, &registry).unwrap();
        let stored = result
            .class
            .get_attr(&intern("__init_subclass__"))
            .expect("__init_subclass__ should be present on the class");

        assert_eq!(stored, descriptor_value);
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
    fn test_type_new_rejects_bool_as_base_type() {
        let registry = create_test_registry();
        let name = intern("DerivedFromBool");
        let namespace = ClassDict::new();

        let result = type_new(name, &[ClassId(TypeId::BOOL.raw())], &namespace, &registry);
        assert!(result.is_err());

        match result.unwrap_err() {
            TypeCreationError::UnacceptableBaseType { class_name } => {
                assert_eq!(class_name, "bool");
            }
            _ => panic!("Expected UnacceptableBaseType error"),
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

        let err = TypeCreationError::UnacceptableBaseType {
            class_name: "bool".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "type 'bool' is not an acceptable base type"
        );

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

    #[test]
    fn test_register_global_class_publishes_heap_type_metadata() {
        let class = Arc::new(PyClassObject::new_simple(intern("PublishedHeapType")));
        let class_id = class.class_id();
        let type_id = class.class_type_id();

        register_global_class(Arc::clone(&class), bitmap_for_class(class.as_ref()));

        let published = global_registry()
            .get(type_id)
            .expect("heap type should be published into the dense type registry");
        assert_eq!(published.type_id(), type_id);
        assert_eq!(published.name.as_str(), "PublishedHeapType");
        assert!(published.flags.contains(TypeFlags::HEAPTYPE));
        assert_eq!(
            global_class(class_id)
                .as_ref()
                .map(|published_class| published_class.class_type_id()),
            Some(type_id)
        );
        assert!(
            global_class_bitmap(class_id)
                .expect("heap class bitmap should be published")
                .is_subclass_of(type_id)
        );

        unregister_global_class(class_id);
    }

    #[test]
    fn test_registered_hierarchy_refreshes_published_layouts_and_versions() {
        let shared = intern("shared");

        let parent = Arc::new(PyClassObject::new_simple(intern("PublishedParent")));
        let parent_id = parent.class_id();
        register_global_class(Arc::clone(&parent), bitmap_for_class(parent.as_ref()));

        let child = Arc::new(
            PyClassObject::new(intern("PublishedChild"), &[parent_id], |id| {
                (id == parent_id).then(|| parent.mro().iter().copied().collect())
            })
            .expect("child class should build"),
        );
        let child_id = child.class_id();
        register_global_class(Arc::clone(&child), bitmap_for_class(child.as_ref()));

        let parent_version = global_class_version(parent_id).expect("parent version should exist");
        let child_version = global_class_version(child_id).expect("child version should exist");
        assert!(child.lookup_method_published(&shared).is_none());

        parent.set_attr(shared.clone(), Value::int_unchecked(10));

        let inherited = child
            .lookup_method_published(&shared)
            .expect("child layout should refresh inherited members after parent mutation");
        assert_eq!(inherited.value, Value::int_unchecked(10));
        assert_eq!(inherited.defining_class, parent_id);
        assert_eq!(inherited.mro_index, 1);
        assert!(
            global_class_version(parent_id).expect("parent version should update") > parent_version
        );
        let child_version_after_parent =
            global_class_version(child_id).expect("child version should update");
        assert!(child_version_after_parent > child_version);

        child.set_attr(shared.clone(), Value::int_unchecked(20));

        let overridden = child
            .lookup_method_published(&shared)
            .expect("child layout should prefer direct override");
        assert_eq!(overridden.value, Value::int_unchecked(20));
        assert_eq!(overridden.defining_class, child_id);
        assert_eq!(overridden.mro_index, 0);
        assert!(
            global_class_version(child_id).expect("child version should advance again")
                > child_version_after_parent
        );

        assert_eq!(
            child.del_attr(&shared),
            Some(Value::int_unchecked(20)),
            "deleting the override should fall back to the parent publication"
        );
        let fallback = child
            .lookup_method_published(&shared)
            .expect("published layout should fall back to the parent after delete");
        assert_eq!(fallback.value, Value::int_unchecked(10));
        assert_eq!(fallback.defining_class, parent_id);

        unregister_global_class(child_id);
        unregister_global_class(parent_id);
    }
}
