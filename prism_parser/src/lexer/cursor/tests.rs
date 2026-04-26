use super::*;

#[test]
fn test_cursor_new() {
    let cursor = Cursor::new("hello");
    assert_eq!(cursor.pos(), 0);
    assert!(!cursor.is_eof());
}

#[test]
fn test_cursor_empty() {
    let cursor = Cursor::new("");
    assert!(cursor.is_eof());
    assert_eq!(cursor.first(), EOF_CHAR);
}

#[test]
fn test_cursor_first() {
    let cursor = Cursor::new("abc");
    assert_eq!(cursor.first(), 'a');
}

#[test]
fn test_cursor_second() {
    let cursor = Cursor::new("abc");
    assert_eq!(cursor.second(), 'b');
}

#[test]
fn test_cursor_third() {
    let cursor = Cursor::new("abc");
    assert_eq!(cursor.third(), 'c');
}

#[test]
fn test_cursor_bump() {
    let mut cursor = Cursor::new("abc");
    assert_eq!(cursor.bump(), Some('a'));
    assert_eq!(cursor.pos(), 1);
    assert_eq!(cursor.bump(), Some('b'));
    assert_eq!(cursor.pos(), 2);
    assert_eq!(cursor.bump(), Some('c'));
    assert_eq!(cursor.pos(), 3);
    assert_eq!(cursor.bump(), None);
}

#[test]
fn test_cursor_unicode() {
    let mut cursor = Cursor::new("αβγ");
    assert_eq!(cursor.first(), 'α');
    assert_eq!(cursor.bump(), Some('α'));
    assert_eq!(cursor.pos(), 2); // Greek alpha is 2 bytes
    assert_eq!(cursor.first(), 'β');
}

#[test]
fn test_cursor_eat_while() {
    let mut cursor = Cursor::new("   abc");
    cursor.eat_while(|c| c == ' ');
    assert_eq!(cursor.first(), 'a');
    assert_eq!(cursor.pos(), 3);
}

#[test]
fn test_cursor_eat() {
    let mut cursor = Cursor::new("abc");
    assert!(cursor.eat('a'));
    assert!(!cursor.eat('a'));
    assert!(cursor.eat('b'));
}

#[test]
fn test_cursor_slice_from() {
    let mut cursor = Cursor::new("hello world");
    cursor.bump(); // h
    cursor.bump(); // e
    cursor.bump(); // l
    cursor.bump(); // l
    cursor.bump(); // o
    assert_eq!(cursor.slice_from(0), "hello");
}

#[test]
fn test_cursor_span_from() {
    let mut cursor = Cursor::new("hello");
    cursor.bump();
    cursor.bump();
    cursor.bump();
    let span = cursor.span_from(0);
    assert_eq!(span.start, 0);
    assert_eq!(span.end, 3);
}

#[test]
fn test_cursor_remaining() {
    let mut cursor = Cursor::new("hello world");
    cursor.eat_while(|c| c != ' ');
    assert_eq!(cursor.remaining(), " world");
}

#[test]
fn test_cursor_lookahead_at_end() {
    let mut cursor = Cursor::new("ab");
    cursor.bump();
    cursor.bump();
    assert_eq!(cursor.first(), EOF_CHAR);
    assert_eq!(cursor.second(), EOF_CHAR);
    assert_eq!(cursor.third(), EOF_CHAR);
}

#[test]
fn test_cursor_emoji() {
    let mut cursor = Cursor::new("🎉abc");
    assert_eq!(cursor.first(), '🎉');
    cursor.bump();
    assert_eq!(cursor.pos(), 4); // Emoji is 4 bytes
    assert_eq!(cursor.first(), 'a');
}
