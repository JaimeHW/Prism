use super::*;

#[test]
fn test_span_new() {
    let span = Span::new(10, 20);
    assert_eq!(span.start, 10);
    assert_eq!(span.end, 20);
}

#[test]
fn test_span_empty() {
    let span = Span::empty(15);
    assert_eq!(span.start, 15);
    assert_eq!(span.end, 15);
    assert!(span.is_empty());
}

#[test]
fn test_span_single() {
    let span = Span::single(5);
    assert_eq!(span.start, 5);
    assert_eq!(span.end, 6);
    assert_eq!(span.len(), 1);
    assert!(!span.is_empty());
}

#[test]
fn test_span_dummy() {
    let span = Span::dummy();
    assert!(span.is_dummy());
    assert!(span.is_empty());
}

#[test]
fn test_span_len() {
    assert_eq!(Span::new(0, 10).len(), 10);
    assert_eq!(Span::new(5, 5).len(), 0);
    assert_eq!(Span::new(100, 200).len(), 100);
}

#[test]
fn test_span_is_empty() {
    assert!(Span::new(5, 5).is_empty());
    assert!(Span::new(10, 5).is_empty()); // Invalid span
    assert!(!Span::new(0, 1).is_empty());
}

#[test]
fn test_span_contains_offset() {
    let span = Span::new(10, 20);
    assert!(!span.contains(9));
    assert!(span.contains(10));
    assert!(span.contains(15));
    assert!(span.contains(19));
    assert!(!span.contains(20));
}

#[test]
fn test_span_contains_span() {
    let outer = Span::new(10, 50);
    let inner = Span::new(20, 30);
    let overlapping = Span::new(5, 25);
    let outside = Span::new(60, 70);

    assert!(outer.contains_span(inner));
    assert!(outer.contains_span(outer)); // Contains itself
    assert!(!outer.contains_span(overlapping));
    assert!(!outer.contains_span(outside));
}

#[test]
fn test_span_overlaps() {
    let span1 = Span::new(10, 20);
    let span2 = Span::new(15, 25);
    let span3 = Span::new(20, 30);
    let span4 = Span::new(0, 5);

    assert!(span1.overlaps(span2));
    assert!(span2.overlaps(span1));
    assert!(!span1.overlaps(span3)); // Adjacent, not overlapping
    assert!(!span1.overlaps(span4));
}

#[test]
fn test_span_merge() {
    let span1 = Span::new(10, 20);
    let span2 = Span::new(15, 30);
    let merged = span1.merge(span2);

    assert_eq!(merged.start, 10);
    assert_eq!(merged.end, 30);
}

#[test]
fn test_span_merge_non_overlapping() {
    let span1 = Span::new(0, 10);
    let span2 = Span::new(20, 30);
    let merged = span1.merge(span2);

    assert_eq!(merged.start, 0);
    assert_eq!(merged.end, 30);
}

#[test]
fn test_span_extend() {
    let mut span = Span::new(10, 20);
    span.extend(Span::new(5, 15));
    assert_eq!(span.start, 5);
    assert_eq!(span.end, 20);

    span.extend(Span::new(15, 30));
    assert_eq!(span.start, 5);
    assert_eq!(span.end, 30);
}

#[test]
fn test_span_shrink() {
    let span = Span::new(10, 30);
    let shrunk = span.shrink(5, 5);
    assert_eq!(shrunk.start, 15);
    assert_eq!(shrunk.end, 25);
}

#[test]
fn test_span_shrink_to_empty() {
    let span = Span::new(10, 20);
    let shrunk = span.shrink(10, 10);
    assert!(shrunk.is_empty());
}

#[test]
fn test_span_shrink_overflow() {
    let span = Span::new(10, 15);
    let shrunk = span.shrink(10, 10);
    assert!(shrunk.is_empty());
}

#[test]
fn test_span_as_range() {
    let span = Span::new(5, 15);
    let range = span.as_range();
    assert_eq!(range, 5..15);
}

#[test]
fn test_span_slice() {
    let source = "hello world";
    let span = Span::new(0, 5);
    assert_eq!(span.slice(source), "hello");

    let span2 = Span::new(6, 11);
    assert_eq!(span2.slice(source), "world");
}

#[test]
fn test_span_slice_out_of_bounds() {
    let source = "short";
    let span = Span::new(0, 100);
    assert_eq!(span.slice(source), "");
}

#[test]
fn test_span_line_col_first_line() {
    let source = "hello world";
    let span = Span::new(0, 5);
    assert_eq!(span.line_col(source), (1, 1));

    let span2 = Span::new(6, 11);
    assert_eq!(span2.line_col(source), (1, 7));
}

#[test]
fn test_span_line_col_multiline() {
    let source = "line1\nline2\nline3";

    // Start of line 1
    assert_eq!(Span::new(0, 1).line_col(source), (1, 1));

    // Start of line 2
    assert_eq!(Span::new(6, 7).line_col(source), (2, 1));

    // Middle of line 2
    assert_eq!(Span::new(8, 9).line_col(source), (2, 3));

    // Start of line 3
    assert_eq!(Span::new(12, 13).line_col(source), (3, 1));
}

#[test]
fn test_span_line_col_at_newline() {
    let source = "abc\ndef";
    let span = Span::new(3, 4); // At the newline
    assert_eq!(span.line_col(source), (1, 4));
}

#[test]
fn test_span_debug() {
    let span = Span::new(10, 20);
    assert_eq!(format!("{:?}", span), "10..20");
}

#[test]
fn test_span_display() {
    let span = Span::new(10, 20);
    assert_eq!(format!("{}", span), "[10, 20)");
}

#[test]
fn test_span_from_range_u32() {
    let span: Span = (5u32..15u32).into();
    assert_eq!(span.start, 5);
    assert_eq!(span.end, 15);
}

#[test]
fn test_span_from_range_usize() {
    let span: Span = (5usize..15usize).into();
    assert_eq!(span.start, 5);
    assert_eq!(span.end, 15);
}

#[test]
fn test_span_into_range() {
    let span = Span::new(5, 15);
    let range: Range<usize> = span.into();
    assert_eq!(range, 5..15);
}

// Spanned tests

#[test]
fn test_spanned_new() {
    let spanned = Spanned::new(42, Span::new(0, 2));
    assert_eq!(spanned.value, 42);
    assert_eq!(spanned.span, Span::new(0, 2));
}

#[test]
fn test_spanned_map() {
    let spanned = Spanned::new(21, Span::new(0, 2));
    let doubled = spanned.map(|x| x * 2);

    assert_eq!(doubled.value, 42);
    assert_eq!(doubled.span, Span::new(0, 2));
}

#[test]
fn test_spanned_as_ref() {
    let spanned = Spanned::new(String::from("hello"), Span::new(0, 5));
    let ref_spanned = spanned.as_ref();

    assert_eq!(ref_spanned.value, &String::from("hello"));
    assert_eq!(ref_spanned.span, Span::new(0, 5));
}

#[test]
fn test_spanned_debug() {
    let spanned = Spanned::new("test", Span::new(0, 4));
    let debug = format!("{:?}", spanned);
    assert!(debug.contains("test"));
    assert!(debug.contains("0..4"));
}

#[test]
fn test_spanned_display() {
    let spanned = Spanned::new(42, Span::new(0, 2));
    assert_eq!(format!("{}", spanned), "42");
}
