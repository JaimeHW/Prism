use super::*;

// ════════════════════════════════════════════════════════════════════════
// FrameInfo Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_frame_info_new() {
    let frame = FrameInfo::new(Arc::from("test_func"), Arc::from("test.py"), 42);

    assert_eq!(&*frame.func_name, "test_func");
    assert_eq!(&*frame.filename, "test.py");
    assert_eq!(frame.line_number(), 42);
    assert!(!frame.needs_resolution());
}

#[test]
fn test_frame_info_with_offset() {
    let frame = FrameInfo::with_offset(Arc::from("test_func"), Arc::from("test.py"), 100, 10);

    assert_eq!(frame.bytecode_offset, 100);
    assert_eq!(frame.first_lineno, 10);
    assert!(frame.needs_resolution());
    assert_eq!(frame.line_number(), 0);
}

#[test]
fn test_frame_info_set_line_number() {
    let mut frame = FrameInfo::with_offset(Arc::from("test_func"), Arc::from("test.py"), 100, 10);

    assert!(frame.needs_resolution());
    frame.set_line_number(25);
    assert!(!frame.needs_resolution());
    assert_eq!(frame.line_number(), 25);
}

#[test]
fn test_frame_info_debug() {
    let frame = FrameInfo::new(Arc::from("func"), Arc::from("file.py"), 10);
    let debug = format!("{:?}", frame);

    assert!(debug.contains("FrameInfo"));
    assert!(debug.contains("func"));
    assert!(debug.contains("file.py"));
}

// ════════════════════════════════════════════════════════════════════════
// TracebackObject Creation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_traceback_empty() {
    let tb = TracebackObject::empty();
    assert!(tb.is_empty());
    assert_eq!(tb.len(), 0);
}

#[test]
fn test_traceback_single() {
    let frame = FrameInfo::new(Arc::from("main"), Arc::from("main.py"), 1);
    let tb = TracebackObject::single(frame);

    assert!(!tb.is_empty());
    assert_eq!(tb.len(), 1);
}

#[test]
fn test_traceback_from_frames() {
    let frames = vec![
        FrameInfo::new(Arc::from("outer"), Arc::from("a.py"), 10),
        FrameInfo::new(Arc::from("middle"), Arc::from("b.py"), 20),
        FrameInfo::new(Arc::from("inner"), Arc::from("c.py"), 30),
    ];

    let tb = TracebackObject::from_frames(frames);
    assert_eq!(tb.len(), 3);
}

// ════════════════════════════════════════════════════════════════════════
// TracebackObject Mutation Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_traceback_push() {
    let mut tb = TracebackObject::empty();

    tb.push(FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1));
    assert_eq!(tb.len(), 1);

    tb.push(FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2));
    assert_eq!(tb.len(), 2);
}

#[test]
fn test_traceback_extend() {
    let mut tb1 = TracebackObject::single(FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1));

    let tb2 = TracebackObject::from_frames(vec![
        FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2),
        FrameInfo::new(Arc::from("c"), Arc::from("c.py"), 3),
    ]);

    tb1.extend(&tb2);
    assert_eq!(tb1.len(), 3);
}

#[test]
fn test_traceback_clear() {
    let mut tb = TracebackObject::from_frames(vec![
        FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1),
        FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2),
    ]);

    assert!(!tb.is_empty());
    tb.clear();
    assert!(tb.is_empty());
}

// ════════════════════════════════════════════════════════════════════════
// TracebackObject Iteration Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_traceback_iter() {
    let tb = TracebackObject::from_frames(vec![
        FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1),
        FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2),
    ]);

    let names: Vec<_> = tb.iter().map(|f| f.func_name.as_ref()).collect();
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn test_traceback_iter_mut() {
    let mut tb = TracebackObject::from_frames(vec![
        FrameInfo::with_offset(Arc::from("a"), Arc::from("a.py"), 10, 1),
        FrameInfo::with_offset(Arc::from("b"), Arc::from("b.py"), 20, 2),
    ]);

    // Resolve line numbers
    for (i, frame) in tb.iter_mut().enumerate() {
        frame.set_line_number((i + 1) as u32 * 10);
    }

    assert!(tb.is_resolved());
}

#[test]
fn test_traceback_innermost() {
    let tb = TracebackObject::from_frames(vec![
        FrameInfo::new(Arc::from("outer"), Arc::from("a.py"), 1),
        FrameInfo::new(Arc::from("inner"), Arc::from("b.py"), 2),
    ]);

    let innermost = tb.innermost().unwrap();
    assert_eq!(&*innermost.func_name, "inner");
}

#[test]
fn test_traceback_outermost() {
    let tb = TracebackObject::from_frames(vec![
        FrameInfo::new(Arc::from("outer"), Arc::from("a.py"), 1),
        FrameInfo::new(Arc::from("inner"), Arc::from("b.py"), 2),
    ]);

    let outermost = tb.outermost().unwrap();
    assert_eq!(&*outermost.func_name, "outer");
}

#[test]
fn test_traceback_empty_innermost_outermost() {
    let tb = TracebackObject::empty();
    assert!(tb.innermost().is_none());
    assert!(tb.outermost().is_none());
}

// ════════════════════════════════════════════════════════════════════════
// TracebackObject Formatting Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_traceback_format() {
    let mut tb = TracebackObject::from_frames(vec![
        FrameInfo::new(Arc::from("main"), Arc::from("main.py"), 10),
        FrameInfo::new(Arc::from("helper"), Arc::from("utils.py"), 25),
    ]);

    let formatted = tb.format();
    assert!(formatted.contains("Traceback (most recent call last):"));
    assert!(formatted.contains("main.py"));
    assert!(formatted.contains("line 10"));
    assert!(formatted.contains("utils.py"));
    assert!(formatted.contains("line 25"));
}

#[test]
fn test_traceback_format_cached() {
    let mut tb =
        TracebackObject::single(FrameInfo::new(Arc::from("test"), Arc::from("test.py"), 1));

    let first = tb.format();
    let second = tb.format();

    // Should be the same Arc (cached)
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn test_traceback_format_cache_invalidated_on_push() {
    let mut tb = TracebackObject::single(FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1));

    let first = tb.format();
    tb.push(FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2));
    let second = tb.format();

    // Should NOT be the same Arc (cache invalidated)
    assert!(!Arc::ptr_eq(&first, &second));
}

#[test]
fn test_traceback_display() {
    let tb = TracebackObject::from_frames(vec![FrameInfo::new(
        Arc::from("test"),
        Arc::from("test.py"),
        42,
    )]);

    let display = format!("{}", tb);
    assert!(display.contains("Traceback"));
    assert!(display.contains("test.py"));
    assert!(display.contains("42"));
}

// ════════════════════════════════════════════════════════════════════════
// Resolution Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_traceback_is_resolved_true() {
    let tb = TracebackObject::from_frames(vec![
        FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1),
        FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2),
    ]);

    assert!(tb.is_resolved());
}

#[test]
fn test_traceback_is_resolved_false() {
    let tb = TracebackObject::from_frames(vec![
        FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1),
        FrameInfo::with_offset(Arc::from("b"), Arc::from("b.py"), 100, 2),
    ]);

    assert!(!tb.is_resolved());
}

// ════════════════════════════════════════════════════════════════════════
// Memory Layout Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_inline_frames_constant() {
    // Verify our inline size is 8
    assert_eq!(INLINE_FRAMES, 8);
}

// ════════════════════════════════════════════════════════════════════════
// Edge Case Tests
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_traceback_many_frames() {
    // Test with more frames than inline capacity
    let frames: Vec<_> = (0..20)
        .map(|i| {
            FrameInfo::new(
                Arc::from(format!("func_{}", i)),
                Arc::from(format!("file_{}.py", i)),
                i as u32,
            )
        })
        .collect();

    let tb = TracebackObject::from_frames(frames);
    assert_eq!(tb.len(), 20);
}

#[test]
fn test_traceback_deep_nesting() {
    // Simulate deep recursion
    let mut tb = TracebackObject::empty();

    for i in 0..100 {
        tb.push(FrameInfo::new(
            Arc::from(format!("recurse_{}", i)),
            Arc::from("recursive.py"),
            i as u32,
        ));
    }

    assert_eq!(tb.len(), 100);
    assert_eq!(&*tb.innermost().unwrap().func_name, "recurse_99");
}
