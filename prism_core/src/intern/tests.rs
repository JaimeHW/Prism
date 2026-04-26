use super::*;

#[test]
fn test_intern_same_string_returns_same_handle() {
    let interner = StringInterner::new();
    let s1 = interner.intern("hello");
    let s2 = interner.intern("hello");

    assert!(Arc::ptr_eq(&s1.inner, &s2.inner));
    assert_eq!(s1, s2);
}

#[test]
fn test_intern_different_strings_returns_different_handles() {
    let interner = StringInterner::new();
    let s1 = interner.intern("hello");
    let s2 = interner.intern("world");

    assert!(!Arc::ptr_eq(&s1.inner, &s2.inner));
    assert_ne!(s1, s2);
}

#[test]
fn test_interned_string_as_str() {
    let interner = StringInterner::new();
    let s = interner.intern("test content");

    assert_eq!(s.as_str(), "test content");
}

#[test]
fn test_interned_string_len() {
    let interner = StringInterner::new();
    let s = interner.intern("hello");

    assert_eq!(s.len(), 5);
}

#[test]
fn test_interned_string_is_empty() {
    let interner = StringInterner::new();
    let empty = interner.intern("");
    let non_empty = interner.intern("x");

    assert!(empty.is_empty());
    assert!(!non_empty.is_empty());
}

#[test]
fn test_interner_get_existing() {
    let interner = StringInterner::new();
    interner.intern("existing");

    let result = interner.get("existing");
    assert!(result.is_some());
    assert_eq!(result.unwrap().as_str(), "existing");
}

#[test]
fn test_interner_get_non_existing() {
    let interner = StringInterner::new();
    let result = interner.get("non_existing");

    assert!(result.is_none());
}

#[test]
fn test_interner_contains() {
    let interner = StringInterner::new();
    interner.intern("present");

    assert!(interner.contains("present"));
    assert!(!interner.contains("absent"));
}

#[test]
fn test_interner_len() {
    let interner = StringInterner::new();
    assert_eq!(interner.len(), 0);

    interner.intern("one");
    assert_eq!(interner.len(), 1);

    interner.intern("two");
    assert_eq!(interner.len(), 2);

    interner.intern("one"); // Duplicate
    assert_eq!(interner.len(), 2);
}

#[test]
fn test_interner_is_empty() {
    let interner = StringInterner::new();
    assert!(interner.is_empty());

    interner.intern("something");
    assert!(!interner.is_empty());
}

#[test]
fn test_interner_clear() {
    let interner = StringInterner::new();
    let s1 = interner.intern("first");
    interner.intern("second");

    assert_eq!(interner.len(), 2);
    interner.clear();
    assert_eq!(interner.len(), 0);

    // Old handle still valid
    assert_eq!(s1.as_str(), "first");

    // New interned string is different handle
    let s1_new = interner.intern("first");
    assert_ne!(s1, s1_new); // Different Arc
}

#[test]
fn test_interner_with_capacity() {
    let interner = StringInterner::with_capacity(100);
    assert!(interner.is_empty());
}

#[test]
fn test_interned_string_hash() {
    use std::collections::HashMap;

    let interner = StringInterner::new();
    let s1 = interner.intern("key");
    let s2 = interner.intern("key");

    let mut map = HashMap::new();
    map.insert(s1.clone(), 42);

    assert_eq!(map.get(&s2), Some(&42));
}

#[test]
fn test_interned_string_debug() {
    let interner = StringInterner::new();
    let s = interner.intern("debug_test");

    let debug_str = format!("{:?}", s);
    assert!(debug_str.contains("debug_test"));
}

#[test]
fn test_interned_string_display() {
    let interner = StringInterner::new();
    let s = interner.intern("display_test");

    assert_eq!(format!("{}", s), "display_test");
}

#[test]
fn test_interned_string_as_ref() {
    let interner = StringInterner::new();
    let s = interner.intern("ref_test");

    let s_ref: &str = s.as_ref();
    assert_eq!(s_ref, "ref_test");
}

#[test]
fn test_interned_string_deref() {
    let interner = StringInterner::new();
    let s = interner.intern("deref_test");

    // Use deref to get &str
    let len = s.len(); // Uses Deref
    assert_eq!(len, 10);

    // String methods work
    assert!(s.starts_with("deref"));
}

#[test]
fn test_interned_string_eq_str() {
    let interner = StringInterner::new();
    let s = interner.intern("compare");

    assert!(s == "compare");
    assert!(s == *"compare");
    assert!(s != "different");
}

#[test]
fn test_interned_string_eq_string() {
    let interner = StringInterner::new();
    let s = interner.intern("compare");

    assert!(s == String::from("compare"));
    assert!(s != String::from("different"));
}

#[test]
fn test_intern_owned() {
    let interner = StringInterner::new();
    let s1 = interner.intern_owned(String::from("owned"));
    let s2 = interner.intern("owned");

    assert_eq!(s1, s2);
}

#[test]
fn test_intern_owned_deduplication() {
    let interner = StringInterner::new();
    interner.intern("existing");

    let s = interner.intern_owned(String::from("existing"));
    assert_eq!(interner.len(), 1);
    assert_eq!(s.as_str(), "existing");
}

#[test]
fn test_global_interner() {
    let s1 = intern("global_test");
    let s2 = intern("global_test");

    assert_eq!(s1, s2);
    assert!(Arc::ptr_eq(&s1.inner, &s2.inner));
}

#[test]
fn test_global_intern_owned() {
    let s1 = intern_owned(String::from("global_owned"));
    let s2 = intern("global_owned");

    assert_eq!(s1, s2);
}

#[test]
fn test_lookup_by_pointer_roundtrip() {
    let interner = StringInterner::new();
    let s = interner.intern("pointer_roundtrip");
    let ptr = s.as_str().as_ptr();

    let resolved = interner.get_by_ptr(ptr).expect("pointer should resolve");
    assert_eq!(resolved, s);
    assert_eq!(interner.len_by_ptr(ptr), Some("pointer_roundtrip".len()));
}

#[test]
fn test_lookup_by_pointer_unknown() {
    let interner = StringInterner::new();
    let bogus = "not_in_interner".as_ptr();
    assert!(interner.get_by_ptr(bogus).is_none());
    assert!(interner.len_by_ptr(bogus).is_none());
}

#[test]
fn test_global_lookup_by_pointer_roundtrip() {
    let s = intern("global_pointer_roundtrip");
    let ptr = s.as_str().as_ptr();

    let resolved = interned_by_ptr(ptr).expect("global pointer should resolve");
    assert_eq!(resolved, s);
    assert_eq!(
        interned_len_by_ptr(ptr),
        Some("global_pointer_roundtrip".len())
    );
}

#[test]
fn test_interner_debug() {
    let interner = StringInterner::new();
    interner.intern("a");
    interner.intern("b");

    let debug_str = format!("{:?}", interner);
    assert!(debug_str.contains("StringInterner"));
    assert!(debug_str.contains("count"));
}

#[test]
fn test_unicode_strings() {
    let interner = StringInterner::new();
    let s1 = interner.intern("こんにちは");
    let s2 = interner.intern("こんにちは");
    let s3 = interner.intern("世界");

    assert_eq!(s1, s2);
    assert_ne!(s1, s3);
    assert_eq!(s1.as_str(), "こんにちは");
}

#[test]
fn test_emoji_strings() {
    let interner = StringInterner::new();
    let s1 = interner.intern("🦀🐍");
    let s2 = interner.intern("🦀🐍");

    assert_eq!(s1, s2);
    assert_eq!(s1.as_str(), "🦀🐍");
}

#[test]
fn test_whitespace_significant() {
    let interner = StringInterner::new();
    let s1 = interner.intern("hello");
    let s2 = interner.intern("hello ");
    let s3 = interner.intern(" hello");

    assert_ne!(s1, s2);
    assert_ne!(s1, s3);
    assert_ne!(s2, s3);
}

#[test]
fn test_case_sensitive() {
    let interner = StringInterner::new();
    let s1 = interner.intern("Hello");
    let s2 = interner.intern("hello");
    let s3 = interner.intern("HELLO");

    assert_ne!(s1, s2);
    assert_ne!(s1, s3);
    assert_ne!(s2, s3);
}

#[test]
fn test_empty_string() {
    let interner = StringInterner::new();
    let s1 = interner.intern("");
    let s2 = interner.intern("");

    assert_eq!(s1, s2);
    assert!(s1.is_empty());
    assert_eq!(s1.len(), 0);
}

#[test]
fn test_long_string() {
    let interner = StringInterner::new();
    let long = "x".repeat(10000);
    let s1 = interner.intern(&long);
    let s2 = interner.intern(&long);

    assert_eq!(s1, s2);
    assert_eq!(s1.len(), 10000);
}

#[test]
fn test_special_characters() {
    let interner = StringInterner::new();
    let s1 = interner.intern("line1\nline2\ttab");
    let s2 = interner.intern("line1\nline2\ttab");

    assert_eq!(s1, s2);
    assert!(s1.contains('\n'));
    assert!(s1.contains('\t'));
}

#[test]
fn test_null_character() {
    let interner = StringInterner::new();
    let s1 = interner.intern("before\0after");
    let s2 = interner.intern("before\0after");

    assert_eq!(s1, s2);
    assert_eq!(s1.len(), 12);
}

#[test]
fn test_concurrent_interning() {
    use std::thread;

    let interner = Arc::new(StringInterner::new());
    let mut handles = vec![];

    for i in 0..10 {
        let interner = Arc::clone(&interner);
        handles.push(thread::spawn(move || {
            let s = format!("thread_{}", i);
            for _ in 0..100 {
                interner.intern(&s);
            }
            interner.intern(&s)
        }));
    }

    for handle in handles {
        let _ = handle.join().unwrap();
    }

    // Each thread should have created exactly one unique string
    assert_eq!(interner.len(), 10);
}

#[test]
fn test_concurrent_same_string() {
    use std::thread;

    let interner = Arc::new(StringInterner::new());
    let mut handles = vec![];

    for _ in 0..10 {
        let interner = Arc::clone(&interner);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                interner.intern("shared_string");
            }
            interner.intern("shared_string")
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should get the same handle
    for result in &results[1..] {
        assert_eq!(&results[0], result);
    }

    // Only one string should be interned
    assert_eq!(interner.len(), 1);
}
