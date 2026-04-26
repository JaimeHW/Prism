use super::*;
use crate::allocation_context::{
    RuntimeHeapBinding, current_heap_binding_depth, standalone_allocation_count,
};
use prism_gc::config::GcConfig;
use prism_gc::heap::GcHeap;

#[test]
fn test_empty_string() {
    let s = StringObject::empty();
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
    assert_eq!(s.as_str(), "");
    assert!(s.is_inline());
}

#[test]
fn test_short_string_inline() {
    let s = StringObject::new("hello");
    assert_eq!(s.as_str(), "hello");
    assert_eq!(s.len(), 5);
    assert!(s.is_inline());
}

#[test]
fn test_max_inline_string() {
    // Exactly 23 bytes should still be inline
    let data = "a".repeat(SSO_MAX_LEN);
    let s = StringObject::new(&data);
    assert!(s.is_inline());
    assert_eq!(s.len(), SSO_MAX_LEN);
}

#[test]
fn test_heap_string() {
    // 24+ bytes should go to heap
    let data = "a".repeat(SSO_MAX_LEN + 1);
    let s = StringObject::new(&data);
    assert!(!s.is_inline());
    assert_eq!(s.len(), SSO_MAX_LEN + 1);
}

#[test]
fn test_interned_string() {
    use prism_core::intern::intern;
    let interned = intern("identifier");
    let s = StringObject::from_interned(interned);
    assert!(s.is_interned());
    assert_eq!(s.as_str(), "identifier");
}

#[test]
fn test_interned_equality_fast_path() {
    use prism_core::intern::intern;
    let i1 = intern("test");
    let i2 = intern("test");
    let s1 = StringObject::from_interned(i1);
    let s2 = StringObject::from_interned(i2);
    assert_eq!(s1, s2); // Uses O(1) pointer comparison
}

#[test]
fn test_concat_inline() {
    let s1 = StringObject::new("hello");
    let s2 = StringObject::new(" world");
    let result = s1.concat(&s2);
    assert_eq!(result.as_str(), "hello world");
    assert!(result.is_inline()); // 11 bytes fits in SSO
}

#[test]
fn test_concat_heap() {
    let s1 = StringObject::new(&"a".repeat(20));
    let s2 = StringObject::new(&"b".repeat(20));
    let result = s1.concat(&s2);
    assert_eq!(result.len(), 40);
    assert!(!result.is_inline()); // >23 bytes goes to heap
}

#[test]
fn test_repeat() {
    let s = StringObject::new("ab");
    let r = s.repeat(3);
    assert_eq!(r.as_str(), "ababab");
}

#[test]
fn test_repeat_zero() {
    let s = StringObject::new("hello");
    let r = s.repeat(0);
    assert!(r.is_empty());
}

#[test]
fn test_concat_string_objects_preserves_heap_result_shape() {
    let left = Value::string(intern("alpha"));
    let right = Value::string(intern("beta"));
    let result = concat_string_objects(left, right).expect("concat should succeed");
    assert_eq!(result.as_str(), "alphabeta");
    assert!(result.is_inline());
}

#[test]
fn test_repeat_string_object_clones_existing_value_for_identity_repeat() {
    let original = StringObject::new("repeat me");
    let value = boxed_string_value(original.clone());
    let repeated = repeat_string_object(value, 1).expect("repeat should succeed");
    assert_eq!(repeated.as_str(), original.as_str());
}

#[test]
fn test_concat_string_values_uses_bound_vm_heap_when_available() {
    assert_eq!(current_heap_binding_depth(), 0);
    let heap = GcHeap::new(GcConfig::default());
    let _binding = RuntimeHeapBinding::register(&heap);

    let baseline = standalone_allocation_count();
    let result = concat_string_values(
        Value::string(intern("managed")),
        Value::string(intern(" heap")),
    )
    .expect("concat should allocate");

    assert_eq!(standalone_allocation_count(), baseline);
    let ptr = result
        .as_object_ptr()
        .expect("managed concat should produce heap-backed string");
    assert!(heap.contains(ptr));
    let string = unsafe { &*(ptr as *const StringObject) };
    assert_eq!(string.as_str(), "managed heap");
}

#[test]
fn test_get_char() {
    let s = StringObject::new("hello");
    assert_eq!(s.get_char(0), Some('h'));
    assert_eq!(s.get_char(4), Some('o'));
    assert_eq!(s.get_char(-1), Some('o'));
    assert_eq!(s.get_char(-5), Some('h'));
    assert_eq!(s.get_char(5), None);
}

#[test]
fn test_slice() {
    let s = StringObject::new("hello world");
    assert_eq!(s.slice(Some(0), Some(5)).as_str(), "hello");
    assert_eq!(s.slice(Some(6), None).as_str(), "world");
    assert_eq!(s.slice(None, Some(5)).as_str(), "hello");
}

#[test]
fn test_contains() {
    let s = StringObject::new("hello world");
    assert!(s.contains("world"));
    assert!(!s.contains("xyz"));
}

#[test]
fn test_starts_ends_with() {
    let s = StringObject::new("hello world");
    assert!(s.starts_with("hello"));
    assert!(s.ends_with("world"));
}

#[test]
fn test_case_conversion() {
    let s = StringObject::new("Hello World");
    assert_eq!(s.lower().as_str(), "hello world");
    assert_eq!(s.upper().as_str(), "HELLO WORLD");
}

#[test]
fn test_strip() {
    let s = StringObject::new("  hello  ");
    assert_eq!(s.strip().as_str(), "hello");
    assert_eq!(s.lstrip().as_str(), "hello  ");
    assert_eq!(s.rstrip().as_str(), "  hello");
}

#[test]
fn test_split() {
    let s = StringObject::new("a,b,c");
    let parts = s.split(",");
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].as_str(), "a");
    assert_eq!(parts[1].as_str(), "b");
    assert_eq!(parts[2].as_str(), "c");
}

#[test]
fn test_join() {
    let sep = StringObject::new(", ");
    let parts: Vec<_> = ["a", "b", "c"]
        .iter()
        .map(|s| StringObject::new(s))
        .collect();
    let result = sep.join(&parts);
    assert_eq!(result.as_str(), "a, b, c");
}

#[test]
fn test_replace() {
    let s = StringObject::new("hello world world");
    assert_eq!(s.replace("world", "rust", None).as_str(), "hello rust rust");
    assert_eq!(
        s.replace("world", "rust", Some(1)).as_str(),
        "hello rust world"
    );
}

#[test]
fn test_unicode() {
    let s = StringObject::new("こんにちは");
    assert_eq!(s.char_count(), 5);
    assert!(s.len() > 5); // UTF-8 bytes > codepoints
}

#[test]
fn test_emoji() {
    let s = StringObject::new("🦀🐍");
    assert_eq!(s.char_count(), 2);
    assert_eq!(s.get_char(0), Some('🦀'));
    assert_eq!(s.get_char(1), Some('🐍'));
}

#[test]
fn test_ordering() {
    let s1 = StringObject::new("apple");
    let s2 = StringObject::new("banana");
    assert!(s1 < s2);
}

// =========================================================================
// SIMD Integration Tests (Comprehensive)
// =========================================================================

mod simd_integration_tests {
    use super::*;

    // Helper: Generate a pattern string
    fn make_pattern_string(len: usize, ch: char) -> String {
        std::iter::repeat(ch).take(len).collect()
    }

    // Helper: Generate ASCII lowercase string
    fn make_lowercase_string(len: usize) -> String {
        (0..len)
            .map(|i| ((b'a' + (i % 26) as u8) as char))
            .collect()
    }

    // Helper: Generate ASCII uppercase string
    fn make_uppercase_string(len: usize) -> String {
        (0..len)
            .map(|i| ((b'A' + (i % 26) as u8) as char))
            .collect()
    }

    // -----------------------------------------------------------------
    // Equality Tests (SIMD str_eq integration)
    // -----------------------------------------------------------------

    #[test]
    fn test_simd_eq_empty() {
        let s1 = StringObject::new("");
        let s2 = StringObject::new("");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_simd_eq_small() {
        for len in 1..32 {
            let data = make_pattern_string(len, 'x');
            let s1 = StringObject::new(&data);
            let s2 = StringObject::new(&data);
            assert_eq!(s1, s2, "len={}", len);
        }
    }

    #[test]
    fn test_simd_eq_medium() {
        for len in [32, 64, 100, 128, 200] {
            let data = make_pattern_string(len, 'y');
            let s1 = StringObject::new(&data);
            let s2 = StringObject::new(&data);
            assert_eq!(s1, s2, "len={}", len);
        }
    }

    #[test]
    fn test_simd_eq_large() {
        for len in [256, 512, 1024, 4096, 10000] {
            let data = make_pattern_string(len, 'z');
            let s1 = StringObject::new(&data);
            let s2 = StringObject::new(&data);
            assert_eq!(s1, s2, "len={}", len);
        }
    }

    #[test]
    fn test_simd_neq_at_start() {
        let s1 = StringObject::new("hello world");
        let s2 = StringObject::new("xello world");
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_simd_neq_at_end() {
        let s1 = StringObject::new("hello world");
        let s2 = StringObject::new("hello worlx");
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_simd_neq_at_middle() {
        let data1: String = (0..100).map(|_| 'a').collect();
        let mut data2 = data1.clone();
        data2.replace_range(50..51, "x");
        let s1 = StringObject::new(&data1);
        let s2 = StringObject::new(&data2);
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_simd_eq_unicode() {
        let s1 = StringObject::new("héllo wörld 日本語");
        let s2 = StringObject::new("héllo wörld 日本語");
        assert_eq!(s1, s2);
    }

    // -----------------------------------------------------------------
    // Contains/Find Tests (SIMD search integration)
    // -----------------------------------------------------------------

    #[test]
    fn test_simd_contains_single_byte() {
        let s = StringObject::new("hello world");
        assert!(s.contains("o"));
        assert!(!s.contains("x"));
    }

    #[test]
    fn test_simd_contains_short_pattern() {
        let s = StringObject::new("the quick brown fox jumps over the lazy dog");
        assert!(s.contains("quick"));
        assert!(s.contains("fox"));
        assert!(s.contains("lazy"));
        assert!(!s.contains("cat"));
    }

    #[test]
    fn test_simd_contains_large_haystack() {
        let haystack = make_pattern_string(10000, 'x');
        let mut haystack_with_needle = haystack.clone();
        haystack_with_needle.push_str("needle");

        let s1 = StringObject::new(&haystack);
        let s2 = StringObject::new(&haystack_with_needle);

        assert!(!s1.contains("needle"));
        assert!(s2.contains("needle"));
    }

    #[test]
    fn test_simd_find_basic() {
        let s = StringObject::new("hello world");
        assert_eq!(s.find("world"), Some(6));
        assert_eq!(s.find("xyz"), None);
    }

    #[test]
    fn test_simd_find_at_start() {
        let s = StringObject::new("hello world");
        assert_eq!(s.find("hello"), Some(0));
    }

    #[test]
    fn test_simd_find_at_end() {
        let s = StringObject::new("hello world");
        assert_eq!(s.find("world"), Some(6));
    }

    #[test]
    fn test_simd_find_empty_pattern() {
        let s = StringObject::new("hello");
        assert_eq!(s.find(""), Some(0));
    }

    #[test]
    fn test_simd_find_overlapping() {
        let s = StringObject::new("aaaa");
        assert_eq!(s.find("aa"), Some(0));
    }

    // -----------------------------------------------------------------
    // Case Conversion Tests (SIMD case integration)
    // -----------------------------------------------------------------

    #[test]
    fn test_simd_lower_ascii() {
        let s = StringObject::new("HELLO WORLD");
        assert_eq!(s.lower().as_str(), "hello world");
    }

    #[test]
    fn test_simd_lower_mixed() {
        let s = StringObject::new("HeLLo WoRLd");
        assert_eq!(s.lower().as_str(), "hello world");
    }

    #[test]
    fn test_simd_lower_with_numbers() {
        let s = StringObject::new("HELLO123WORLD");
        assert_eq!(s.lower().as_str(), "hello123world");
    }

    #[test]
    fn test_simd_lower_large() {
        let upper = make_uppercase_string(10000);
        let expected_lower = make_lowercase_string(10000);
        let s = StringObject::new(&upper);
        assert_eq!(s.lower().as_str(), expected_lower);
    }

    #[test]
    fn test_simd_lower_unicode_fallback() {
        // Non-ASCII should fall back to std's Unicode-aware lowercase
        let s = StringObject::new("HÉLLO");
        // This goes through the Unicode fallback, which handles é properly
        let lower = s.lower();
        // Note: é is already lowercase in the input (it's the uppercase that differs)
        assert!(lower.as_str().starts_with("h"));
    }

    #[test]
    fn test_simd_upper_ascii() {
        let s = StringObject::new("hello world");
        assert_eq!(s.upper().as_str(), "HELLO WORLD");
    }

    #[test]
    fn test_simd_upper_mixed() {
        let s = StringObject::new("HeLLo WoRLd");
        assert_eq!(s.upper().as_str(), "HELLO WORLD");
    }

    #[test]
    fn test_simd_upper_large() {
        let lower = make_lowercase_string(10000);
        let expected_upper = make_uppercase_string(10000);
        let s = StringObject::new(&lower);
        assert_eq!(s.upper().as_str(), expected_upper);
    }

    // -----------------------------------------------------------------
    // Whitespace Trimming Tests (SIMD whitespace integration)
    // -----------------------------------------------------------------

    #[test]
    fn test_simd_strip_basic() {
        let s = StringObject::new("  hello world  ");
        assert_eq!(s.strip().as_str(), "hello world");
    }

    #[test]
    fn test_simd_strip_tabs_newlines() {
        let s = StringObject::new("\t\nhello\r\n\t");
        assert_eq!(s.strip().as_str(), "hello");
    }

    #[test]
    fn test_simd_strip_no_whitespace() {
        let s = StringObject::new("hello");
        assert_eq!(s.strip().as_str(), "hello");
    }

    #[test]
    fn test_simd_strip_all_whitespace() {
        let s = StringObject::new("   \t\n   ");
        assert_eq!(s.strip().as_str(), "");
    }

    #[test]
    fn test_simd_strip_large() {
        let mut data = String::new();
        for _ in 0..500 {
            data.push(' ');
        }
        data.push_str("hello");
        for _ in 0..500 {
            data.push(' ');
        }
        let s = StringObject::new(&data);
        assert_eq!(s.strip().as_str(), "hello");
    }

    #[test]
    fn test_simd_lstrip_basic() {
        let s = StringObject::new("  hello  ");
        assert_eq!(s.lstrip().as_str(), "hello  ");
    }

    #[test]
    fn test_simd_rstrip_basic() {
        let s = StringObject::new("  hello  ");
        assert_eq!(s.rstrip().as_str(), "  hello");
    }

    // -----------------------------------------------------------------
    // Character Count Tests (SIMD utf8_char_count integration)
    // -----------------------------------------------------------------

    #[test]
    fn test_simd_char_count_ascii() {
        let s = StringObject::new("hello world");
        assert_eq!(s.char_count(), 11);
    }

    #[test]
    fn test_simd_char_count_unicode() {
        let s = StringObject::new("héllo wörld");
        // Each non-ASCII takes 2 bytes but is 1 character
        assert_eq!(s.char_count(), 11);
    }

    #[test]
    fn test_simd_char_count_cjk() {
        let s = StringObject::new("日本語");
        assert_eq!(s.char_count(), 3);
    }

    #[test]
    fn test_simd_char_count_emoji() {
        let s = StringObject::new("🦀🐍🎉");
        assert_eq!(s.char_count(), 3);
    }

    #[test]
    fn test_simd_char_count_large_ascii() {
        let len = 10000;
        let data = make_pattern_string(len, 'a');
        let s = StringObject::new(&data);
        assert_eq!(s.char_count(), len);
    }

    #[test]
    fn test_simd_char_count_mixed() {
        // Mix of ASCII, Latin-1 extended, CJK, and emoji
        let s = StringObject::new("Hello héllo 日本語 🦀");
        // h-e-l-l-o- -h-é-l-l-o- -日-本-語- -🦀 = 17 chars
        assert_eq!(s.char_count(), 17);
    }

    // -----------------------------------------------------------------
    // Boundary and Edge Case Tests
    // -----------------------------------------------------------------

    #[test]
    fn test_simd_operations_at_simd_boundaries() {
        // Test at 16-byte (SSE), 32-byte (AVX2), 64-byte (AVX-512) boundaries
        for len in [15, 16, 17, 31, 32, 33, 63, 64, 65] {
            let data = make_pattern_string(len, 'x');
            let s = StringObject::new(&data);

            // Equality
            let s2 = StringObject::new(&data);
            assert_eq!(s, s2, "equality at len={}", len);

            // Contains
            assert!(s.contains("x"), "contains at len={}", len);

            // Strip (add whitespace around)
            let with_ws = format!("  {}  ", data);
            let s_ws = StringObject::new(&with_ws);
            assert_eq!(s_ws.strip().as_str(), data, "strip at len={}", len);
        }
    }

    #[test]
    fn test_simd_case_preserve_non_letter() {
        // Numbers, punctuation should be preserved
        let s = StringObject::new("HELLO123!@#WORLD");
        assert_eq!(s.lower().as_str(), "hello123!@#world");

        let s2 = StringObject::new("hello456$%^world");
        assert_eq!(s2.upper().as_str(), "HELLO456$%^WORLD");
    }
}
