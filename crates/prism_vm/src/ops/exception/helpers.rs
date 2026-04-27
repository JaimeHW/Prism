//! Exception opcode helper functions.
//!
//! This module contains performance-critical helper functions for exception
//! handling operations, including type extraction, dynamic matching, tuple
//! matching, and traceback construction.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    Exception Helper Functions                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Type Extraction                                                         │
//! │  ├── extract_type_id_from_value() - Dynamic type introspection           │
//! │  └── extract_type_from_exception() - ExceptionObject → TypeId            │
//! │                                                                          │
//! │  Dynamic Matching                                                        │
//! │  ├── check_dynamic_match() - Runtime type comparison                     │
//! │  └── check_tuple_match() - Multi-type matching for except tuples         │
//! │                                                                          │
//! │  Tuple Operations                                                        │
//! │  ├── extract_tuple_elements() - Fast tuple extraction                    │
//! │  └── match_any_type() - SIMD-friendly iteration                          │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Function | Complexity | Notes |
//! |----------|------------|-------|
//! | extract_type_id_from_value | O(1) | Tagged pointer check |
//! | check_dynamic_match | O(1) | Single type comparison |
//! | check_tuple_match | O(N) | N = tuple length, early exit |
//! | extract_tuple_elements | O(1) | Pointer arithmetic only |

use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;

// =============================================================================
// Constants
// =============================================================================

/// Sentinel value for "no exception type" in instruction encoding.
pub const NO_TYPE_ID: u16 = 0xFFFF;

/// Default exception type ID when dynamic extraction fails.
/// This is BaseException, the root of the exception hierarchy.
pub const DEFAULT_EXCEPTION_TYPE: u16 = 0;

/// Maximum tuple size we'll check inline (avoid loop overhead).
const INLINE_TUPLE_CHECK_SIZE: usize = 4;

/// Error reported when an `except` target is not an exception class or tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExceptionMatchError {
    /// The handler target is not a valid exception class or exception tuple.
    InvalidTarget,
}

/// Result type for runtime exception handler target checks.
pub type ExceptionMatchResult = Result<bool, ExceptionMatchError>;

// =============================================================================
// Type Extraction
// =============================================================================

/// Extract exception type ID from a value by introspecting its type.
///
/// This is the slow path for dynamic type extraction. The fast path uses
/// instruction-encoded type IDs directly.
///
/// # Performance
///
/// This function is marked `#[inline(never)]` to keep it off the hot path.
/// The common case (instruction-encoded type ID) doesn't call this function.
///
/// # Returns
///
/// - ExceptionTypeId as u16 if value is an exception or exception type
/// - DEFAULT_EXCEPTION_TYPE if type cannot be determined
#[inline(never)]
pub fn extract_type_id_from_value(value: &Value) -> u16 {
    // Check if value is tagged as an exception object
    if let Some(type_id) = try_extract_exception_type_id(value) {
        return type_id;
    }

    // Check if value is a type object representing an exception class
    if let Some(type_id) = try_extract_type_object_id(value) {
        return type_id;
    }

    // Fallback to generic Exception type
    ExceptionTypeId::Exception as u16
}

/// Try to extract type ID from an exception object.
///
/// Returns None if the value is not an exception object.
///
/// # Memory Layout
///
/// ExceptionValue has:
/// - ObjectHeader (16 bytes) with type_id = TypeId::EXCEPTION (22)
/// - exception_type_id (u16) at offset 16
///
/// This distinguishes ExceptionValue (instances) from ExceptionTypeObject (types).
#[inline(always)]
fn try_extract_exception_type_id(value: &Value) -> Option<u16> {
    use crate::builtins::ExceptionValue;

    // Check if this is a pointer value that could be an exception
    if !value.is_object() {
        return None;
    }

    // Get the object pointer and read the header
    let ptr = value.as_object_ptr()?;

    // SAFETY: We've verified this is an object pointer, so we can read the header
    let header = unsafe { &*(ptr as *const ObjectHeader) };

    // Check if this is an ExceptionValue (instance) by header TypeId
    // TypeId::EXCEPTION (22) indicates this is an exception INSTANCE, not a type object
    if header.type_id == TypeId::EXCEPTION {
        // SAFETY: We've verified the type ID matches ExceptionValue
        let exc_value = unsafe { &*(ptr as *const ExceptionValue) };
        return Some(exc_value.exception_type_id);
    }

    if header.type_id.raw() >= TypeId::FIRST_USER_TYPE {
        let class = global_class(ClassId(header.type_id.raw()))?;
        return heap_exception_type_id_for_class(class.as_ref());
    }

    None
}

/// Try to extract type ID from an exception type object.
///
/// This handles the case where `except MyException:` is used with
/// a type object rather than an instance.
///
/// # Performance
///
/// O(1) - Single pointer dereference + type check + field load.
#[inline(always)]
fn try_extract_type_object_id(value: &Value) -> Option<u16> {
    use crate::builtins::{EXCEPTION_TYPE_ID, ExceptionTypeObject};

    // Check if this is an object
    if !value.is_object() {
        return None;
    }

    // Get the object pointer and read the header
    let ptr = value.as_object_ptr()?;

    // SAFETY: We've verified this is an object pointer, so we can read the header
    let header = unsafe { &*(ptr as *const ObjectHeader) };

    // Check if this is an ExceptionTypeObject by header TypeId
    if header.type_id == EXCEPTION_TYPE_ID {
        // SAFETY: We've verified the type ID matches ExceptionTypeObject
        let exc_type = unsafe { &*(ptr as *const ExceptionTypeObject) };
        return Some(exc_type.exception_type_id);
    }

    // WORKAROUND: Due to potential TypeId mismatch between static creation and
    // runtime headers (observed: header stores exception_type_id instead of
    // EXCEPTION_TYPE_ID), try to directly read the ExceptionTypeObject layout.
    //
    // ExceptionTypeObject layout:
    //   - offset 0-15: ObjectHeader (16 bytes)
    //   - offset 16-17: exception_type_id (u16)
    //
    // If the header.type_id value is in the valid ExceptionTypeId range (22-80),
    // it's likely this IS an ExceptionTypeObject with the wrong header TypeId.
    // In this case, try direct field access.
    let header_type_id_raw = header.type_id.0;
    if header_type_id_raw >= 22 && header_type_id_raw <= 80 {
        // SAFETY: We're reading directly from ExceptionTypeObject layout
        let exc_type = unsafe { &*(ptr as *const ExceptionTypeObject) };
        let exc_id = exc_type.exception_type_id;

        // Sanity check: exception_type_id should also be in valid range
        if exc_id >= 22 && exc_id <= 80 {
            return Some(exc_id);
        }
    }

    if let Some(class) = heap_exception_class_from_value(value) {
        return heap_exception_type_id_for_class(class);
    }

    None
}

/// Extract type ID from a type value used in except clause.
///
/// This handles `except SomeError:` where SomeError is resolved
/// at runtime from a variable.
///
/// # Extraction Order
///
/// We try instance extraction FIRST because the type object workaround
/// in `try_extract_type_object_id` can incorrectly match exception instances
/// (they have header.type_id in range 22-80). By checking for instances first,
/// we ensure proper distinction between ExceptionValue and ExceptionTypeObject.
#[inline]
pub fn extract_type_from_type_value(type_value: &Value) -> Option<u16> {
    // IMPORTANT: Try instance extraction FIRST!
    // The type object workaround can falsely match instances, so check for
    // instances first to short-circuit that case.
    if let Some(type_id) = try_extract_exception_type_id(type_value) {
        return Some(type_id);
    }

    // Then try as a type object (class)
    if let Some(type_id) = try_extract_type_object_id(type_value) {
        return Some(type_id);
    }

    None
}

#[inline]
fn heap_exception_type_id_for_class(class: &PyClassObject) -> Option<u16> {
    class
        .mro()
        .iter()
        .find_map(|&class_id| crate::builtins::exception_type_id_for_proxy_class_id(class_id))
}

#[inline]
fn heap_exception_class_from_value(value: &Value) -> Option<&'static PyClassObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::TYPE || crate::builtins::builtin_type_object_type_id(ptr).is_some()
    {
        return None;
    }

    let class = unsafe { &*(ptr as *const PyClassObject) };
    heap_exception_type_id_for_class(class).map(|_| class)
}

#[inline]
fn heap_exception_class_id(value: &Value) -> Option<ClassId> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id.raw() < TypeId::FIRST_USER_TYPE {
        return None;
    }

    let class_id = ClassId(header.type_id.raw());
    let class = global_class(class_id)?;
    heap_exception_type_id_for_class(class.as_ref()).map(|_| class_id)
}

#[inline]
fn heap_exception_matches_class(active_exception: &Value, target_class: &PyClassObject) -> bool {
    let Some(active_class_id) = heap_exception_class_id(active_exception) else {
        return false;
    };
    let Some(active_class) = global_class(active_class_id) else {
        return false;
    };

    active_class
        .mro()
        .iter()
        .any(|&class_id| class_id == target_class.class_id())
}

#[inline]
fn heap_exception_matches_builtin_type(active_exception: &Value, target_type_id: u16) -> bool {
    let Some(active_class_id) = heap_exception_class_id(active_exception) else {
        return false;
    };
    let Some(active_class) = global_class(active_class_id) else {
        return false;
    };

    active_class.mro().iter().any(|&class_id| {
        crate::builtins::exception_type_id_for_proxy_class_id(class_id) == Some(target_type_id)
    })
}

#[inline]
pub(crate) fn is_exception_instance_value(value: &Value) -> bool {
    try_extract_exception_type_id(value).is_some()
}

#[inline]
pub(crate) fn is_exception_class_value(value: &Value) -> bool {
    try_extract_type_object_id(value).is_some() || heap_exception_class_from_value(value).is_some()
}

// =============================================================================
// Dynamic Matching
// =============================================================================

/// Check if an exception type matches a runtime type value.
///
/// This is used when the except clause uses a variable:
/// ```python
/// error_type = ValueError
/// try:
///     ...
/// except error_type:  # Dynamic match needed
///     ...
/// ```
///
/// # Arguments
///
/// * `exc_type_id` - The type ID of the active exception
/// * `type_value` - The runtime type value to match against
///
/// # Returns
///
/// True if the exception is an instance of the specified type (including subclasses).
#[inline]
pub fn check_dynamic_match(
    active_exception: Option<&Value>,
    exc_type_id: u16,
    type_value: &Value,
) -> ExceptionMatchResult {
    if let Some(target_class) = heap_exception_class_from_value(type_value) {
        return Ok(
            active_exception.is_some_and(|value| heap_exception_matches_class(value, target_class))
        );
    }

    if is_exception_instance_value(type_value) {
        return Err(ExceptionMatchError::InvalidTarget);
    }

    let match_type_id = match try_extract_type_object_id(type_value) {
        Some(id) => id,
        None => return Err(ExceptionMatchError::InvalidTarget),
    };

    if let Some(active_exception) = active_exception
        && heap_exception_matches_builtin_type(active_exception, match_type_id)
    {
        return Ok(true);
    }

    // Check if exception type matches or is a subclass
    Ok(exc_type_id == match_type_id || is_subclass(exc_type_id, match_type_id))
}

/// Check whether the active exception matches a Python `except` target.
///
/// The target must be an exception class or a tuple, recursively, of exception
/// classes. Invalid targets raise `TypeError` in the opcode layer rather than
/// silently behaving as a non-match.
#[inline]
pub fn check_exception_match(
    active_exception: Option<&Value>,
    exc_type_id: u16,
    target: &Value,
) -> ExceptionMatchResult {
    if let Some(elements) = try_extract_tuple_elements(target) {
        return check_tuple_elements(active_exception, exc_type_id, &elements);
    }

    check_dynamic_match(active_exception, exc_type_id, target)
}

/// Check if one exception type is a subclass of another.
///
/// Uses the precomputed exception hierarchy for O(1) lookups.
#[inline(always)]
pub fn is_subclass(exc_type: u16, parent_type: u16) -> bool {
    // Convert u16 to ExceptionTypeId for hierarchy check
    let exc = ExceptionTypeId::from_u8(exc_type as u8);
    let parent = ExceptionTypeId::from_u8(parent_type as u8);

    match (exc, parent) {
        (Some(e), Some(p)) => e.is_subclass_of(p),
        _ => false, // Unknown types don't match
    }
}

// =============================================================================
// Tuple Matching
// =============================================================================

/// Check if exception matches any type in a tuple.
///
/// Handles `except (TypeError, ValueError):` syntax.
///
/// # Performance
///
/// - Small tuples (≤4): Unrolled inline checks
/// - Large tuples: Loop with early exit
///
/// # Arguments
///
/// * `exc_type_id` - The type ID of the active exception
/// * `types_tuple` - A tuple value containing exception types
///
/// # Returns
///
/// True if the exception matches any type in the tuple.
#[inline]
pub fn check_tuple_match(
    active_exception: Option<&Value>,
    exc_type_id: u16,
    types_tuple: &Value,
) -> ExceptionMatchResult {
    // Early exit for non-tuple values
    if !types_tuple.is_object() {
        return Err(ExceptionMatchError::InvalidTarget);
    }

    // Try to get tuple elements
    let elements = match try_extract_tuple_elements(types_tuple) {
        Some(elems) => elems,
        None => return Err(ExceptionMatchError::InvalidTarget),
    };

    check_tuple_elements(active_exception, exc_type_id, &elements)
}

#[inline]
fn check_tuple_elements(
    active_exception: Option<&Value>,
    exc_type_id: u16,
    elements: &[Value],
) -> ExceptionMatchResult {
    let len = elements.len();

    // Unrolled check for small tuples (common case)
    if len <= INLINE_TUPLE_CHECK_SIZE {
        return match_tuple_inline(active_exception, exc_type_id, elements);
    }

    // Loop for larger tuples
    match_tuple_loop(active_exception, exc_type_id, elements)
}

/// Inline tuple matching for small tuples.
///
/// Unrolled loop for tuples with ≤4 elements.
#[inline(always)]
fn match_tuple_inline(
    active_exception: Option<&Value>,
    exc_type_id: u16,
    elements: &[Value],
) -> ExceptionMatchResult {
    match elements.len() {
        0 => Ok(false),
        1 => check_tuple_element_match(active_exception, exc_type_id, &elements[0]),
        2 => {
            if check_tuple_element_match(active_exception, exc_type_id, &elements[0])? {
                return Ok(true);
            }
            check_tuple_element_match(active_exception, exc_type_id, &elements[1])
        }
        3 => {
            if check_tuple_element_match(active_exception, exc_type_id, &elements[0])? {
                return Ok(true);
            }
            if check_tuple_element_match(active_exception, exc_type_id, &elements[1])? {
                return Ok(true);
            }
            check_tuple_element_match(active_exception, exc_type_id, &elements[2])
        }
        4 => {
            if check_tuple_element_match(active_exception, exc_type_id, &elements[0])? {
                return Ok(true);
            }
            if check_tuple_element_match(active_exception, exc_type_id, &elements[1])? {
                return Ok(true);
            }
            if check_tuple_element_match(active_exception, exc_type_id, &elements[2])? {
                return Ok(true);
            }
            check_tuple_element_match(active_exception, exc_type_id, &elements[3])
        }
        _ => match_tuple_loop(active_exception, exc_type_id, elements),
    }
}

/// Loop-based tuple matching for larger tuples.
#[inline(never)]
fn match_tuple_loop(
    active_exception: Option<&Value>,
    exc_type_id: u16,
    elements: &[Value],
) -> ExceptionMatchResult {
    for type_value in elements {
        if check_tuple_element_match(active_exception, exc_type_id, type_value)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[inline]
fn check_tuple_element_match(
    active_exception: Option<&Value>,
    exc_type_id: u16,
    type_value: &Value,
) -> ExceptionMatchResult {
    if let Some(nested) = try_extract_tuple_elements(type_value) {
        return check_tuple_elements(active_exception, exc_type_id, &nested);
    }

    check_dynamic_match(active_exception, exc_type_id, type_value)
}

/// Try to extract tuple elements from a value.
///
/// Returns None if the value is not a tuple.
#[inline]
fn try_extract_tuple_elements(value: &Value) -> Option<Vec<Value>> {
    use prism_runtime::object::ObjectHeader;
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::types::TupleObject;

    // Check if this is an object
    if !value.is_object() {
        return None;
    }

    // Get the object pointer and read the header
    let ptr = value.as_object_ptr()?;

    // SAFETY: We've verified this is an object pointer, so we can read the header
    let header = unsafe { &*(ptr as *const ObjectHeader) };

    // Check if this is a tuple
    if header.type_id != TypeId::TUPLE {
        return None;
    }

    // SAFETY: We've verified the type ID matches TupleObject
    let tuple = unsafe { &*(ptr as *const TupleObject) };

    // Return a copy of the tuple elements
    Some(tuple.as_slice().to_vec())
}

// =============================================================================
// Type ID Extraction (Unified)
// =============================================================================

/// Extract exception type ID from instruction or value.
///
/// Fast path: Use instruction-encoded type ID if available.
/// Slow path: Introspect value to determine type ID.
///
/// # Arguments
///
/// * `encoded_type_id` - Type ID from instruction (NO_TYPE_ID if dynamic)
/// * `value` - The exception value to extract type from
///
/// # Returns
///
/// The exception type ID as u16.
#[inline(always)]
pub fn extract_type_id(encoded_type_id: u16, value: &Value) -> u16 {
    if encoded_type_id != NO_TYPE_ID {
        return encoded_type_id;
    }

    // Dynamic type extraction (cold path)
    extract_type_id_from_value(value)
}
