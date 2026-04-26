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

    let frame_info = build_frame_info_with_line(func_name.clone(), filename.clone(), line_number);

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
