//! Traceback extraction and construction for exception handling.
//!
//! This module provides functions for extracting traceback information from
//! exception objects and building traceback frames from VM state.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      Traceback Operations                                │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Extraction                                                              │
//! │  ├── extract_traceback() - Get traceback from exception value            │
//! │  ├── extract_traceback_from_object() - Direct ExceptionObject access     │
//! │  └── has_traceback() - Check if exception has traceback                  │
//! │                                                                          │
//! │  Construction                                                            │
//! │  ├── build_frame_info() - Create FrameInfo from VM state                 │
//! │  ├── capture_traceback() - Capture current call stack                    │
//! │  └── extend_traceback() - Add frames to existing traceback               │
//! │                                                                          │
//! │  Wrapper                                                                 │
//! │  ├── traceback_to_value() - Wrap TracebackObject in Value                │
//! │  └── value_to_traceback() - Unwrap Value to TracebackObject              │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! Traceback construction is lazy - we only capture the minimum information
//! at raise time, deferring line number resolution until the traceback is
//! actually printed (following CPython 3.11+ patterns).
//!
//! | Operation | Complexity | Notes |
//! |-----------|------------|-------|
//! | has_traceback | O(1) | Null check |
//! | extract_traceback | O(1) | Pointer copy |
//! | build_frame_info | O(1) | Field copies |
//! | capture_traceback | O(N) | N = stack depth |

use crate::VirtualMachine;
use crate::stdlib::exceptions::{FrameInfo, TracebackObject};
use prism_core::Value;
use std::sync::Arc;

// =============================================================================
// Traceback Extraction
// =============================================================================

/// Extract traceback from an exception value.
///
/// Returns None if:
/// - Value is not an exception
/// - Exception has no traceback attached
///
/// # Performance
///
/// This is O(1) - just pointer/tag checks and a clone of the Arc.
#[inline]
pub fn extract_traceback(value: &Value) -> Option<TracebackObject> {
    // Check if this is an object that could be an exception
    if !value.is_object() {
        return None;
    }

    // TODO: Implement proper exception object extraction
    // This requires reading the object header and type tag
    // For now, return None
    extract_traceback_from_object(value)
}

/// Extract traceback directly from an exception object.
///
/// Internal helper that assumes the value has already been validated.
#[inline(always)]
fn extract_traceback_from_object(_exc_value: &Value) -> Option<TracebackObject> {
    // TODO: Implement proper object introspection
    // 1. Read object header to confirm it's an ExceptionObject
    // 2. Read traceback field from the exception
    // 3. Clone the TracebackObject if present
    None
}

/// Check if an exception value has a traceback attached.
///
/// This is a quick check without extracting the full traceback.
#[inline(always)]
pub fn has_traceback(value: &Value) -> bool {
    if !value.is_object() {
        return false;
    }

    // TODO: Implement proper object introspection
    // For now, return false
    false
}

// =============================================================================
// Traceback Construction
// =============================================================================

/// Build a FrameInfo from the current VM state.
///
/// Captures the current function name, filename, and bytecode offset.
/// Line number resolution is deferred until the traceback is formatted.
///
/// # Arguments
///
/// * `vm` - The virtual machine
/// * `bytecode_offset` - Current instruction pointer offset
#[inline]
pub fn build_frame_info(vm: &VirtualMachine, bytecode_offset: u32) -> FrameInfo {
    let frame = vm.current_frame();

    // Get code object from frame (public field)
    let code = &frame.code;

    // Extract metadata from public fields
    let func_name: Arc<str> = code.name.clone();
    let filename: Arc<str> = code.filename.clone();
    let first_lineno = code.first_lineno;

    // Create frame info with deferred line resolution
    FrameInfo::with_offset(func_name, filename, bytecode_offset, first_lineno)
}

/// Build a FrameInfo with a known line number.
///
/// Used when the line number is already known (e.g., from a debug build).
#[inline]
pub fn build_frame_info_with_line(
    func_name: Arc<str>,
    filename: Arc<str>,
    line_number: u32,
) -> FrameInfo {
    FrameInfo::new(func_name, filename, line_number)
}

/// Capture the current call stack as a traceback.
///
/// Walks the frame stack and builds FrameInfo for each frame.
///
/// # Arguments
///
/// * `vm` - The virtual machine
/// * `max_frames` - Maximum number of frames to capture (0 = unlimited)
///
/// # Returns
///
/// A TracebackObject containing the captured frames.
#[inline(never)] // Cold path
pub fn capture_traceback(vm: &VirtualMachine, max_frames: usize) -> TracebackObject {
    let mut tb = TracebackObject::empty();

    // Get the current frame
    let frame = vm.current_frame();
    let code = &frame.code;

    // Build info for current frame using public fields
    let func_name: Arc<str> = code.name.clone();
    let filename: Arc<str> = code.filename.clone();
    let first_lineno = code.first_lineno;

    // Use instruction pointer as bytecode offset
    let bytecode_offset = frame.ip;

    let frame_info = FrameInfo::with_offset(func_name, filename, bytecode_offset, first_lineno);
    tb.push(frame_info);

    // TODO: Walk parent frames for full stack trace
    // For now, we just capture the current frame
    let _ = max_frames; // Will be used when we walk frames

    tb
}

/// Extend an existing traceback with the current frame.
///
/// Used when re-raising exceptions to add new frames to the traceback.
#[inline]
pub fn extend_traceback(tb: &mut TracebackObject, vm: &VirtualMachine, bytecode_offset: u32) {
    let frame_info = build_frame_info(vm, bytecode_offset);
    tb.push(frame_info);
}

// =============================================================================
// Value Wrapper Functions
// =============================================================================

/// Wrap a TracebackObject in a Value.
///
/// Creates a Value that contains the traceback for passing through
/// the VM register system.
///
/// # Note
///
/// This creates a heap allocation. Use sparingly on hot paths.
#[inline]
pub fn traceback_to_value(_tb: &TracebackObject) -> Value {
    // TODO: Implement proper Value wrapping for TracebackObject
    // This requires:
    // 1. Allocating TracebackObject on GC heap
    // 2. Creating tagged pointer Value
    // For now, return None as placeholder
    Value::none()
}

/// Unwrap a Value to get the TracebackObject.
///
/// Returns None if the value is not a traceback.
#[inline]
pub fn value_to_traceback(_value: &Value) -> Option<TracebackObject> {
    // TODO: Implement proper Value unwrapping
    // This requires:
    // 1. Checking type tag
    // 2. Reading TracebackObject from heap
    None
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // Extraction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_extract_traceback_from_none() {
        let value = Value::none();
        assert!(extract_traceback(&value).is_none());
    }

    #[test]
    fn test_extract_traceback_from_int() {
        let value = Value::int(42).unwrap();
        assert!(extract_traceback(&value).is_none());
    }

    #[test]
    fn test_extract_traceback_from_bool() {
        let value = Value::bool(true);
        assert!(extract_traceback(&value).is_none());
    }

    #[test]
    fn test_extract_traceback_from_float() {
        let value = Value::float(3.14);
        assert!(extract_traceback(&value).is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // has_traceback Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_has_traceback_none() {
        let value = Value::none();
        assert!(!has_traceback(&value));
    }

    #[test]
    fn test_has_traceback_int() {
        let value = Value::int(42).unwrap();
        assert!(!has_traceback(&value));
    }

    #[test]
    fn test_has_traceback_bool() {
        let value = Value::bool(false);
        assert!(!has_traceback(&value));
    }

    // ════════════════════════════════════════════════════════════════════════
    // FrameInfo Construction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_build_frame_info_with_line() {
        let func_name: Arc<str> = "test_func".into();
        let filename: Arc<str> = "test.py".into();
        let line_number = 42;

        let frame_info =
            build_frame_info_with_line(func_name.clone(), filename.clone(), line_number);

        assert_eq!(&*frame_info.func_name, "test_func");
        assert_eq!(&*frame_info.filename, "test.py");
        assert_eq!(frame_info.line_number(), line_number);
    }

    #[test]
    fn test_build_frame_info_with_line_zero() {
        let func_name: Arc<str> = "<module>".into();
        let filename: Arc<str> = "main.py".into();

        let frame_info = build_frame_info_with_line(func_name, filename, 0);

        assert_eq!(frame_info.line_number(), 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // TracebackObject Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_empty_traceback() {
        let tb = TracebackObject::empty();
        assert!(tb.is_empty());
        assert_eq!(tb.len(), 0);
    }

    #[test]
    fn test_single_frame_traceback() {
        let frame = FrameInfo::new("func".into(), "file.py".into(), 10);
        let tb = TracebackObject::single(frame);

        assert!(!tb.is_empty());
        assert_eq!(tb.len(), 1);
    }

    #[test]
    fn test_traceback_push() {
        let mut tb = TracebackObject::empty();

        let frame1 = FrameInfo::new("outer".into(), "file.py".into(), 10);
        let frame2 = FrameInfo::new("inner".into(), "file.py".into(), 20);

        tb.push(frame1);
        assert_eq!(tb.len(), 1);

        tb.push(frame2);
        assert_eq!(tb.len(), 2);
    }

    #[test]
    fn test_traceback_innermost() {
        let frame1 = FrameInfo::new("outer".into(), "file.py".into(), 10);
        let frame2 = FrameInfo::new("inner".into(), "file.py".into(), 20);

        let tb = TracebackObject::from_frames([frame1, frame2]);

        let innermost = tb.innermost().unwrap();
        assert_eq!(&*innermost.func_name, "inner");
    }

    #[test]
    fn test_traceback_outermost() {
        let frame1 = FrameInfo::new("outer".into(), "file.py".into(), 10);
        let frame2 = FrameInfo::new("inner".into(), "file.py".into(), 20);

        let tb = TracebackObject::from_frames([frame1, frame2]);

        let outermost = tb.outermost().unwrap();
        assert_eq!(&*outermost.func_name, "outer");
    }

    // ════════════════════════════════════════════════════════════════════════
    // Value Wrapper Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_traceback_to_value_placeholder() {
        let tb = TracebackObject::empty();
        let value = traceback_to_value(&tb);
        // Currently returns None
        assert!(value.is_none());
    }

    #[test]
    fn test_value_to_traceback_none() {
        let value = Value::none();
        assert!(value_to_traceback(&value).is_none());
    }

    #[test]
    fn test_value_to_traceback_int() {
        let value = Value::int(42).unwrap();
        assert!(value_to_traceback(&value).is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // FrameInfo Deferred Resolution Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_frame_info_with_offset() {
        let frame = FrameInfo::with_offset("func".into(), "file.py".into(), 100, 1);

        // Line number should not be immediately resolved
        assert!(frame.needs_resolution());
    }

    #[test]
    fn test_frame_info_set_line_number() {
        let mut frame = FrameInfo::with_offset("func".into(), "file.py".into(), 100, 1);

        frame.set_line_number(42);

        assert!(!frame.needs_resolution());
        assert_eq!(frame.line_number(), 42);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_traceback_clear() {
        let mut tb = TracebackObject::from_frames([
            FrameInfo::new("a".into(), "a.py".into(), 1),
            FrameInfo::new("b".into(), "b.py".into(), 2),
        ]);

        assert_eq!(tb.len(), 2);
        tb.clear();
        assert!(tb.is_empty());
    }

    #[test]
    fn test_traceback_extend() {
        let mut tb1 = TracebackObject::single(FrameInfo::new("a".into(), "a.py".into(), 1));
        let tb2 = TracebackObject::single(FrameInfo::new("b".into(), "b.py".into(), 2));

        tb1.extend(&tb2);
        assert_eq!(tb1.len(), 2);
    }

    #[test]
    fn test_frame_info_default_line_number() {
        let frame = FrameInfo::new("func".into(), "file.py".into(), 0);
        assert_eq!(frame.line_number(), 0);
    }

    #[test]
    fn test_traceback_iteration() {
        let frames = [
            FrameInfo::new("a".into(), "a.py".into(), 1),
            FrameInfo::new("b".into(), "b.py".into(), 2),
            FrameInfo::new("c".into(), "c.py".into(), 3),
        ];
        let tb = TracebackObject::from_frames(frames);

        let collected: Vec<_> = tb.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(&*collected[0].func_name, "a");
        assert_eq!(&*collected[1].func_name, "b");
        assert_eq!(&*collected[2].func_name, "c");
    }
}
