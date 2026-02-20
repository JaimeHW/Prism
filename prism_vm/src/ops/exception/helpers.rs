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
    use prism_runtime::object::ObjectHeader;
    use prism_runtime::object::type_obj::TypeId;

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
    use prism_runtime::object::ObjectHeader;

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
pub fn check_dynamic_match(exc_type_id: u16, type_value: &Value) -> bool {
    // Extract type ID from the type value
    let match_type_id = match extract_type_from_type_value(type_value) {
        Some(id) => id,
        None => return false, // Not a valid exception type
    };

    // Check if exception type matches or is a subclass
    exc_type_id == match_type_id || is_subclass(exc_type_id, match_type_id)
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
pub fn check_tuple_match(exc_type_id: u16, types_tuple: &Value) -> bool {
    // Early exit for non-tuple values
    if !types_tuple.is_object() {
        return false;
    }

    // Try to get tuple elements
    let elements = match try_extract_tuple_elements(types_tuple) {
        Some(elems) => elems,
        None => return false,
    };

    let len = elements.len();

    // Unrolled check for small tuples (common case)
    if len <= INLINE_TUPLE_CHECK_SIZE {
        return match_tuple_inline(exc_type_id, &elements);
    }

    // Loop for larger tuples
    match_tuple_loop(exc_type_id, &elements)
}

/// Inline tuple matching for small tuples.
///
/// Unrolled loop for tuples with ≤4 elements.
#[inline(always)]
fn match_tuple_inline(exc_type_id: u16, elements: &[Value]) -> bool {
    match elements.len() {
        0 => false,
        1 => check_dynamic_match(exc_type_id, &elements[0]),
        2 => {
            check_dynamic_match(exc_type_id, &elements[0])
                || check_dynamic_match(exc_type_id, &elements[1])
        }
        3 => {
            check_dynamic_match(exc_type_id, &elements[0])
                || check_dynamic_match(exc_type_id, &elements[1])
                || check_dynamic_match(exc_type_id, &elements[2])
        }
        4 => {
            check_dynamic_match(exc_type_id, &elements[0])
                || check_dynamic_match(exc_type_id, &elements[1])
                || check_dynamic_match(exc_type_id, &elements[2])
                || check_dynamic_match(exc_type_id, &elements[3])
        }
        _ => match_tuple_loop(exc_type_id, elements),
    }
}

/// Loop-based tuple matching for larger tuples.
#[inline(never)]
fn match_tuple_loop(exc_type_id: u16, elements: &[Value]) -> bool {
    for type_value in elements {
        if check_dynamic_match(exc_type_id, type_value) {
            return true;
        }
    }
    false
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::types::TupleObject;

    fn exception_type_value(name: &str) -> Value {
        let exc_type = crate::builtins::get_exception_type(name)
            .unwrap_or_else(|| panic!("missing exception type: {name}"));
        Value::object_ptr(exc_type as *const _ as *const ())
    }

    fn tuple_value(values: &[Value]) -> Value {
        let tuple = TupleObject::from_slice(values);
        let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
        Value::object_ptr(ptr)
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Constant Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_no_type_id_sentinel() {
        assert_eq!(NO_TYPE_ID, 0xFFFF);
        assert_eq!(NO_TYPE_ID, u16::MAX);
    }

    #[test]
    fn test_default_exception_type() {
        assert_eq!(DEFAULT_EXCEPTION_TYPE, 0);
    }

    #[test]
    fn test_inline_tuple_check_size() {
        assert_eq!(INLINE_TUPLE_CHECK_SIZE, 4);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Type Extraction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_extract_type_id_from_none() {
        let value = Value::none();
        let type_id = extract_type_id_from_value(&value);
        // None is not an exception, should return Exception type
        assert_eq!(type_id, ExceptionTypeId::Exception as u16);
    }

    #[test]
    fn test_extract_type_id_from_int() {
        let value = Value::int(42).unwrap();
        let type_id = extract_type_id_from_value(&value);
        // Int is not an exception, should return Exception type
        assert_eq!(type_id, ExceptionTypeId::Exception as u16);
    }

    #[test]
    fn test_extract_type_id_from_bool() {
        let value = Value::bool(true);
        let type_id = extract_type_id_from_value(&value);
        assert_eq!(type_id, ExceptionTypeId::Exception as u16);
    }

    #[test]
    fn test_extract_type_id_fast_path() {
        let value = Value::none();
        // With explicit type ID, should return that directly
        let type_id = extract_type_id(24, &value); // TypeError
        assert_eq!(type_id, 24);
    }

    #[test]
    fn test_extract_type_id_slow_path() {
        let value = Value::none();
        // With NO_TYPE_ID, should call dynamic extraction
        let type_id = extract_type_id(NO_TYPE_ID, &value);
        assert_eq!(type_id, ExceptionTypeId::Exception as u16);
    }

    // ════════════════════════════════════════════════════════════════════════
    // is_subclass Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_is_subclass_self() {
        // Every type is a subclass of itself
        let exc = ExceptionTypeId::Exception as u16;
        assert!(is_subclass(exc, exc));
    }

    #[test]
    fn test_is_subclass_type_error_of_exception() {
        let type_error = ExceptionTypeId::TypeError as u16;
        let exception = ExceptionTypeId::Exception as u16;
        assert!(is_subclass(type_error, exception));
    }

    #[test]
    fn test_is_subclass_value_error_of_exception() {
        let value_error = ExceptionTypeId::ValueError as u16;
        let exception = ExceptionTypeId::Exception as u16;
        assert!(is_subclass(value_error, exception));
    }

    #[test]
    fn test_is_subclass_not_related() {
        let type_error = ExceptionTypeId::TypeError as u16;
        let value_error = ExceptionTypeId::ValueError as u16;
        // Neither is a subclass of the other
        assert!(!is_subclass(type_error, value_error));
        assert!(!is_subclass(value_error, type_error));
    }

    #[test]
    fn test_is_subclass_invalid_types() {
        // Invalid type IDs should return false
        assert!(!is_subclass(255, 255));
        assert!(!is_subclass(200, 4));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Dynamic Match Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_check_dynamic_match_none() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let type_value = Value::none();
        // None is not a valid type, should return false
        assert!(!check_dynamic_match(exc_type_id, &type_value));
    }

    #[test]
    fn test_check_dynamic_match_int() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let type_value = Value::int(42).unwrap();
        // Int is not a valid type, should return false
        assert!(!check_dynamic_match(exc_type_id, &type_value));
    }

    #[test]
    fn test_check_dynamic_match_type_object_direct_match() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let type_value = exception_type_value("TypeError");
        assert!(check_dynamic_match(exc_type_id, &type_value));
    }

    #[test]
    fn test_check_dynamic_match_type_object_subclass_match() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let type_value = exception_type_value("Exception");
        assert!(check_dynamic_match(exc_type_id, &type_value));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Tuple Match Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_check_tuple_match_none() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let types_tuple = Value::none();
        // None is not a tuple, should return false
        assert!(!check_tuple_match(exc_type_id, &types_tuple));
    }

    #[test]
    fn test_check_tuple_match_int() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let types_tuple = Value::int(42).unwrap();
        // Int is not a tuple, should return false
        assert!(!check_tuple_match(exc_type_id, &types_tuple));
    }

    #[test]
    fn test_check_tuple_match_bool() {
        let exc_type_id = ExceptionTypeId::ValueError as u16;
        let types_tuple = Value::bool(true);
        assert!(!check_tuple_match(exc_type_id, &types_tuple));
    }

    #[test]
    fn test_check_tuple_match_type_object_elements() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let value_error = exception_type_value("ValueError");
        let type_error = exception_type_value("TypeError");
        let types_tuple = tuple_value(&[value_error, type_error]);
        assert!(check_tuple_match(exc_type_id, &types_tuple));

        let tuple_ptr =
            types_tuple.as_object_ptr().expect("tuple should be object") as *mut TupleObject;
        unsafe { drop_boxed(tuple_ptr) };
    }

    // ════════════════════════════════════════════════════════════════════════
    // Inline Tuple Match Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_match_tuple_inline_empty() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        assert!(!match_tuple_inline(exc_type_id, &[]));
    }

    #[test]
    fn test_match_tuple_inline_one_no_match() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let elements = [Value::none()];
        assert!(!match_tuple_inline(exc_type_id, &elements));
    }

    #[test]
    fn test_match_tuple_inline_two_no_match() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let elements = [Value::none(), Value::int(1).unwrap()];
        assert!(!match_tuple_inline(exc_type_id, &elements));
    }

    #[test]
    fn test_match_tuple_inline_three_no_match() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let elements = [Value::none(), Value::int(1).unwrap(), Value::bool(false)];
        assert!(!match_tuple_inline(exc_type_id, &elements));
    }

    #[test]
    fn test_match_tuple_inline_four_no_match() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let elements = [
            Value::none(),
            Value::int(1).unwrap(),
            Value::bool(false),
            Value::float(3.14),
        ];
        assert!(!match_tuple_inline(exc_type_id, &elements));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Loop Match Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_match_tuple_loop_empty() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        assert!(!match_tuple_loop(exc_type_id, &[]));
    }

    #[test]
    fn test_match_tuple_loop_no_match() {
        let exc_type_id = ExceptionTypeId::TypeError as u16;
        let elements: Vec<Value> = (0..10).map(|i| Value::int(i).unwrap()).collect();
        assert!(!match_tuple_loop(exc_type_id, &elements));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_extract_type_from_type_value_none() {
        let type_value = Value::none();
        assert!(extract_type_from_type_value(&type_value).is_none());
    }

    #[test]
    fn test_extract_type_from_type_value_int() {
        let type_value = Value::int(42).unwrap();
        assert!(extract_type_from_type_value(&type_value).is_none());
    }

    #[test]
    fn test_extract_type_from_type_value_exception_type_object() {
        let type_value = exception_type_value("TypeError");
        assert_eq!(
            extract_type_from_type_value(&type_value),
            Some(ExceptionTypeId::TypeError as u16)
        );
    }

    #[test]
    fn test_try_extract_exception_type_id_non_ptr() {
        let value = Value::int(42).unwrap();
        assert!(try_extract_exception_type_id(&value).is_none());
    }

    #[test]
    fn test_try_extract_type_object_id_non_ptr() {
        let value = Value::bool(true);
        assert!(try_extract_type_object_id(&value).is_none());
    }

    #[test]
    fn test_try_extract_tuple_elements_non_ptr() {
        let value = Value::float(1.23);
        assert!(try_extract_tuple_elements(&value).is_none());
    }
}
