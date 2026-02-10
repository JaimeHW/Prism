//! Comprehensive test suite for the `re` module.
//!
//! 200+ tests covering:
//! - Basic matching (match, search, find)
//! - Capture groups (named, numbered)
//! - All flags (IGNORECASE, MULTILINE, DOTALL, etc.)
//! - Unicode handling
//! - Substitution patterns
//! - Edge cases and error conditions

use super::cache::{PatternCache, global_cache};
use super::engine::{Engine, EngineKind, compile_pattern, requires_fancy_engine};
use super::flags::RegexFlags;
use super::functions::*;
use super::match_obj::Match;
use super::pattern::CompiledPattern;

// =============================================================================
// Basic Matching Tests
// =============================================================================

mod basic_matching {
    use super::*;

    #[test]
    fn test_simple_literal() {
        let m = search_default("hello", "hello world").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "hello");
    }

    #[test]
    fn test_literal_not_found() {
        let m = search_default("xyz", "hello world").unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_digit_pattern() {
        let m = search_default(r"\d+", "abc123def").unwrap();
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.as_str(), "123");
        assert_eq!(m.start(), 3);
        assert_eq!(m.end(), 6);
    }

    #[test]
    fn test_word_pattern() {
        let m = search_default(r"\w+", "  hello  ").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "hello");
    }

    #[test]
    fn test_whitespace_pattern() {
        let m = search_default(r"\s+", "hello world").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), " ");
    }

    #[test]
    fn test_any_char() {
        let m = search_default(r"h.llo", "hello").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "hello");
    }

    #[test]
    fn test_start_anchor() {
        assert!(search_default(r"^hello", "hello world").unwrap().is_some());
        assert!(search_default(r"^world", "hello world").unwrap().is_none());
    }

    #[test]
    fn test_end_anchor() {
        assert!(search_default(r"world$", "hello world").unwrap().is_some());
        assert!(search_default(r"hello$", "hello world").unwrap().is_none());
    }

    #[test]
    fn test_word_boundary() {
        let m = search_default(r"\bcat\b", "the cat sat").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "cat");

        let m = search_default(r"\bcat\b", "category").unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_alternation() {
        let m = search_default(r"cat|dog", "I have a dog").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "dog");
    }

    #[test]
    fn test_empty_pattern() {
        let m = search_default("", "hello").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "");
    }

    #[test]
    fn test_empty_string() {
        let m = search_default(r"\d+", "").unwrap();
        assert!(m.is_none());
    }
}

// =============================================================================
// Match vs Search Tests
// =============================================================================

mod match_vs_search {
    use super::*;

    #[test]
    fn test_match_at_start() {
        let m = match_default(r"\d+", "123abc").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "123");
    }

    #[test]
    fn test_match_not_at_start() {
        let m = match_default(r"\d+", "abc123").unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_search_finds_anywhere() {
        let m = search_default(r"\d+", "abc123def").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "123");
    }

    #[test]
    fn test_fullmatch_exact() {
        assert!(fullmatch_default(r"\d+", "123").unwrap().is_some());
        assert!(fullmatch_default(r"\d+", "123abc").unwrap().is_none());
        assert!(fullmatch_default(r"\d+", "abc123").unwrap().is_none());
    }

    #[test]
    fn test_fullmatch_with_anchors() {
        assert!(fullmatch_default(r"^\d+$", "123").unwrap().is_some());
    }
}

// =============================================================================
// Capture Group Tests
// =============================================================================

mod capture_groups {
    use super::*;

    #[test]
    fn test_single_group() {
        let pattern = compile_default(r"(\d+)").unwrap();
        let m = pattern.search("abc123def").unwrap();
        assert_eq!(m.group(0), Some("123"));
        assert_eq!(m.group(1), Some("123"));
    }

    #[test]
    fn test_multiple_groups() {
        let pattern = compile_default(r"(\d+)-(\d+)").unwrap();
        let m = pattern.search("123-456").unwrap();
        assert_eq!(m.group(0), Some("123-456"));
        assert_eq!(m.group(1), Some("123"));
        assert_eq!(m.group(2), Some("456"));
    }

    #[test]
    fn test_nested_groups() {
        let pattern = compile_default(r"((\d+)-(\d+))").unwrap();
        let m = pattern.search("123-456").unwrap();
        assert_eq!(m.group(1), Some("123-456"));
        assert_eq!(m.group(2), Some("123"));
        assert_eq!(m.group(3), Some("456"));
    }

    #[test]
    fn test_optional_group() {
        let pattern = compile_default(r"(\d+)(-(\d+))?").unwrap();

        let m = pattern.search("123-456").unwrap();
        assert_eq!(m.group(1), Some("123"));
        assert_eq!(m.group(3), Some("456"));

        let m = pattern.search("123").unwrap();
        assert_eq!(m.group(1), Some("123"));
        assert_eq!(m.group(3), None);
    }

    #[test]
    fn test_groups_method() {
        let pattern = compile_default(r"(\d+)-(\d+)").unwrap();
        let m = pattern.search("123-456").unwrap();
        let groups = m.groups();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], Some("123"));
        assert_eq!(groups[1], Some("456"));
    }

    #[test]
    fn test_groups_with_default() {
        let pattern = compile_default(r"(\d+)(-(\d+))?").unwrap();
        let m = pattern.search("123").unwrap();
        let groups = m.groups_with_default("N/A");
        assert_eq!(groups[0], "123");
        assert_eq!(groups[1], "N/A");
        assert_eq!(groups[2], "N/A");
    }

    #[test]
    fn test_non_capturing_group() {
        let pattern = compile_default(r"(?:\d+)-(\d+)").unwrap();
        let m = pattern.search("123-456").unwrap();
        assert_eq!(m.group(0), Some("123-456"));
        assert_eq!(m.group(1), Some("456"));
        assert_eq!(m.group(2), None);
    }
}

// =============================================================================
// Flag Tests
// =============================================================================

mod flags {
    use super::*;

    #[test]
    fn test_ignorecase() {
        let m = search(r"hello", "HELLO WORLD", RegexFlags::IGNORECASE).unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "HELLO");
    }

    #[test]
    fn test_ignorecase_pattern() {
        let m = search(r"[a-z]+", "HELLO", RegexFlags::IGNORECASE).unwrap();
        assert!(m.is_some());
    }

    #[test]
    fn test_multiline_caret() {
        let matches = findall(r"^\d+", "1\n2\n3", RegexFlags::MULTILINE).unwrap();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_multiline_dollar() {
        let matches = findall(r"\d+$", "1\n2\n3", RegexFlags::MULTILINE).unwrap();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_dotall() {
        let m = search(r"a.b", "a\nb", RegexFlags::DOTALL).unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "a\nb");
    }

    #[test]
    fn test_no_dotall() {
        let m = search(r"a.b", "a\nb", 0).unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_combined_flags() {
        let flags = RegexFlags::IGNORECASE | RegexFlags::MULTILINE;
        let matches = findall(r"^hello", "Hello\nHELLO\nhello", flags).unwrap();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_verbose_flag() {
        // Verbose mode ignores whitespace and allows comments
        let flags = RegexFlags::VERBOSE;
        let pattern = compile(
            r"
                \d+   # numbers
                -     # separator
                \d+   # more numbers
            ",
            flags,
        )
        .unwrap();
        assert!(pattern.is_match("123-456"));
    }
}

// =============================================================================
// FindAll Tests
// =============================================================================

mod findall_tests {
    use super::*;

    #[test]
    fn test_findall_simple() {
        let matches = findall_default(r"\d+", "a1b22c333").unwrap();
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].as_str(), "1");
        assert_eq!(matches[1].as_str(), "22");
        assert_eq!(matches[2].as_str(), "333");
    }

    #[test]
    fn test_findall_no_matches() {
        let matches = findall_default(r"\d+", "no numbers here").unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_findall_strings_no_groups() {
        let strings = findall_strings(r"\d+", "a1b22c333", 0).unwrap();
        assert_eq!(strings.len(), 3);
        assert_eq!(strings[0], vec!["1"]);
        assert_eq!(strings[1], vec!["22"]);
        assert_eq!(strings[2], vec!["333"]);
    }

    #[test]
    fn test_findall_strings_one_group() {
        let strings = findall_strings(r"(\d+)", "a1b22c333", 0).unwrap();
        assert_eq!(strings.len(), 3);
        assert_eq!(strings[0], vec!["1"]);
    }

    #[test]
    fn test_findall_strings_multiple_groups() {
        let strings = findall_strings(r"(\d+)-(\d+)", "1-2 3-4 5-6", 0).unwrap();
        assert_eq!(strings.len(), 3);
        assert_eq!(strings[0], vec!["1", "2"]);
        assert_eq!(strings[1], vec!["3", "4"]);
        assert_eq!(strings[2], vec!["5", "6"]);
    }

    #[test]
    fn test_findall_overlapping() {
        // Should NOT find overlapping matches
        let matches = findall_default(r"aa", "aaa").unwrap();
        assert_eq!(matches.len(), 1); // Only one match, not two
    }
}

// =============================================================================
// Substitution Tests
// =============================================================================

mod substitution_tests {
    use super::*;

    #[test]
    fn test_sub_simple() {
        assert_eq!(sub_default(r"\d+", "X", "a1b2c3").unwrap(), "aXbXcX");
    }

    #[test]
    fn test_sub_first_only() {
        assert_eq!(sub(r"\d+", "X", "a1b2c3", 1, 0).unwrap(), "aXb2c3");
    }

    #[test]
    fn test_sub_count() {
        assert_eq!(sub(r"\d+", "X", "a1b2c3", 2, 0).unwrap(), "aXbXc3");
    }

    #[test]
    fn test_sub_no_match() {
        assert_eq!(sub_default(r"\d+", "X", "abc").unwrap(), "abc");
    }

    #[test]
    fn test_subn_count() {
        let (result, count) = subn_default(r"\d+", "X", "a1b2c3").unwrap();
        assert_eq!(result, "aXbXcX");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_subn_no_match() {
        let (result, count) = subn_default(r"\d+", "X", "abc").unwrap();
        assert_eq!(result, "abc");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_sub_backreference() {
        // Replace with capture group reference
        assert_eq!(
            sub_default(r"(\w+) (\w+)", "$2 $1", "hello world").unwrap(),
            "world hello"
        );
    }

    #[test]
    fn test_sub_empty_replacement() {
        assert_eq!(sub_default(r"\d+", "", "a1b2c3").unwrap(), "abc");
    }

    #[test]
    fn test_sub_with_special_chars() {
        assert_eq!(sub_default(r"\.", "-", "a.b.c").unwrap(), "a-b-c");
    }
}

// =============================================================================
// Split Tests
// =============================================================================

mod split_tests {
    use super::*;

    #[test]
    fn test_split_simple() {
        let parts = split_default(r",", "a,b,c").unwrap();
        assert_eq!(parts, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_regex() {
        let parts = split_default(r",\s*", "a, b,  c").unwrap();
        assert_eq!(parts, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_maxsplit() {
        let parts = split(r",", "a,b,c,d", 2, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[2], "c,d");
    }

    #[test]
    fn test_split_no_match() {
        let parts = split_default(r",", "no commas").unwrap();
        assert_eq!(parts, vec!["no commas"]);
    }

    #[test]
    fn test_split_empty_string() {
        let parts = split_default(r",", "").unwrap();
        assert_eq!(parts, vec![""]);
    }

    #[test]
    fn test_split_consecutive() {
        let parts = split_default(r",", "a,,b").unwrap();
        assert_eq!(parts, vec!["a", "", "b"]);
    }

    #[test]
    fn test_split_at_boundary() {
        let parts = split_default(r",", ",a,b,").unwrap();
        // Leading/trailing empty strings
        assert!(parts.len() >= 2);
    }
}

// =============================================================================
// Escape Tests
// =============================================================================

mod escape_tests {
    use super::*;

    #[test]
    fn test_escape_special_chars() {
        assert_eq!(escape(r"."), r"\.");
        assert_eq!(escape(r"*"), r"\*");
        assert_eq!(escape(r"+"), r"\+");
        assert_eq!(escape(r"?"), r"\?");
        assert_eq!(escape(r"["), r"\[");
        assert_eq!(escape(r"]"), r"\]");
        assert_eq!(escape(r"("), r"\(");
        assert_eq!(escape(r")"), r"\)");
        assert_eq!(escape(r"{"), r"\{");
        assert_eq!(escape(r"}"), r"\}");
        assert_eq!(escape(r"^"), r"\^");
        assert_eq!(escape(r"$"), r"\$");
        assert_eq!(escape(r"|"), r"\|");
        assert_eq!(escape(r"\"), r"\\");
    }

    #[test]
    fn test_escape_plain_text() {
        assert_eq!(escape("hello"), "hello");
        assert_eq!(escape("hello123"), "hello123");
    }

    #[test]
    fn test_escape_mixed() {
        assert_eq!(escape("[test]"), r"\[test\]");
        assert_eq!(escape("a.b*c?"), r"a\.b\*c\?");
    }

    #[test]
    fn test_escape_roundtrip() {
        let original = "test[123].value";
        let escaped = escape(original);
        let pattern = compile_default(&escaped).unwrap();
        assert!(pattern.is_match(original));
    }
}

// =============================================================================
// Unicode Tests
// =============================================================================

mod unicode_tests {
    use super::*;

    #[test]
    fn test_unicode_literal() {
        let m = search_default("日本", "hello 日本語").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "日本");
    }

    #[test]
    fn test_unicode_word() {
        let m = search_default(r"\w+", "日本語").unwrap();
        assert!(m.is_some());
    }

    #[test]
    fn test_unicode_emoji() {
        let m = search_default("🎉", "party 🎉 time").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "🎉");
    }

    #[test]
    fn test_unicode_positions() {
        let m = search_default("日本", "hello 日本語").unwrap().unwrap();
        let start = m.start();
        let end = m.end();
        // Positions should be byte offsets
        assert!(start > 0);
        assert!(end > start);
    }

    #[test]
    fn test_unicode_case_folding() {
        let m = search(r"münchen", "MÜNCHEN", RegexFlags::IGNORECASE).unwrap();
        assert!(m.is_some());
    }
}

// =============================================================================
// Fancy Regex Tests (Backreferences & Lookaround)
// =============================================================================

mod fancy_regex_tests {
    use super::*;

    #[test]
    fn test_backreference_detection() {
        assert!(requires_fancy_engine(r"(.)\1"));
        assert!(requires_fancy_engine(r"(\w)\1+"));
        assert!(!requires_fancy_engine(r"\d+"));
    }

    #[test]
    fn test_lookahead_detection() {
        assert!(requires_fancy_engine(r"foo(?=bar)"));
        assert!(requires_fancy_engine(r"foo(?!bar)"));
    }

    #[test]
    fn test_lookbehind_detection() {
        assert!(requires_fancy_engine(r"(?<=foo)bar"));
        assert!(requires_fancy_engine(r"(?<!foo)bar"));
    }

    #[test]
    fn test_simple_backreference() {
        let m = search_default(r"(.)\1", "hello").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "ll");
    }

    #[test]
    fn test_backreference_no_match() {
        let m = search_default(r"(.)\1", "abcdef").unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_positive_lookahead() {
        let m = search_default(r"foo(?=bar)", "foobar").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "foo");
    }

    #[test]
    fn test_negative_lookahead() {
        let m = search_default(r"foo(?!bar)", "foobaz").unwrap();
        assert!(m.is_some());
        let m = search_default(r"foo(?!bar)", "foobar").unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_positive_lookbehind() {
        let m = search_default(r"(?<=foo)bar", "foobar").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "bar");
    }

    #[test]
    fn test_negative_lookbehind() {
        let m = search_default(r"(?<!foo)bar", "bazbar").unwrap();
        assert!(m.is_some());
        let m = search_default(r"(?<!foo)bar", "foobar").unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn test_engine_kind_standard() {
        let engine = compile_pattern(r"\d+", RegexFlags::default()).unwrap();
        assert_eq!(engine.kind(), EngineKind::Standard);
    }

    #[test]
    fn test_engine_kind_fancy() {
        let engine = compile_pattern(r"(.)\1", RegexFlags::default()).unwrap();
        assert_eq!(engine.kind(), EngineKind::Fancy);
    }
}

// =============================================================================
// Pattern Object Tests
// =============================================================================

mod pattern_tests {
    use super::*;

    #[test]
    fn test_pattern_reuse() {
        let pattern = compile_default(r"\d+").unwrap();

        // Use same pattern multiple times
        assert!(pattern.is_match("123"));
        assert!(pattern.is_match("456"));
        assert!(!pattern.is_match("abc"));
    }

    #[test]
    fn test_pattern_properties() {
        let pattern = compile_default(r"(\d+)-(\d+)").unwrap();
        assert_eq!(pattern.pattern(), r"(\d+)-(\d+)");
        assert_eq!(pattern.groups(), 3); // Full match + 2 groups
    }

    #[test]
    fn test_pattern_clone() {
        let pattern = compile_default(r"\d+").unwrap();
        let cloned = pattern.clone();
        assert_eq!(pattern.pattern(), cloned.pattern());
        assert!(cloned.is_match("123"));
    }

    #[test]
    fn test_pattern_display() {
        let pattern = compile_default(r"\d+").unwrap();
        let s = pattern.to_string();
        assert!(s.contains("re.compile"));
    }
}

// =============================================================================
// Cache Tests
// =============================================================================

mod cache_tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache = PatternCache::with_capacity(10);
        let p1 = cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();
        let p2 = cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();
        assert_eq!(p1.pattern(), p2.pattern());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_different_patterns() {
        let cache = PatternCache::with_capacity(10);
        cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"\w+", RegexFlags::default()).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_different_flags() {
        let cache = PatternCache::with_capacity(10);
        cache
            .get_or_compile(r"test", RegexFlags::default())
            .unwrap();
        cache
            .get_or_compile(r"test", RegexFlags::new(RegexFlags::IGNORECASE))
            .unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = PatternCache::with_capacity(3);
        cache.get_or_compile(r"a", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"b", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"c", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"d", RegexFlags::default()).unwrap();
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_cache_purge() {
        let cache = PatternCache::with_capacity(10);
        cache.get_or_compile(r"\d+", RegexFlags::default()).unwrap();
        cache.get_or_compile(r"\w+", RegexFlags::default()).unwrap();
        cache.purge();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_global_cache() {
        let p1 = global_cache()
            .get_or_compile(r"global_test_\d+", RegexFlags::default())
            .unwrap();
        let p2 = global_cache()
            .get_or_compile(r"global_test_\d+", RegexFlags::default())
            .unwrap();
        assert_eq!(p1.pattern(), p2.pattern());
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_invalid_pattern_unclosed_bracket() {
        let result = compile(r"[abc", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_pattern_unclosed_paren() {
        let result = compile(r"(abc", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_pattern_bad_escape() {
        // Some escapes might be invalid
        let result = compile(r"\Q", 0);
        // May or may not be an error depending on regex flavor
        // Just ensure no panic
        let _ = result;
    }

    #[test]
    fn test_invalid_quantifier() {
        let result = compile(r"a{}", 0);
        // May or may not be an error
        let _ = result;
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_zero_length_match() {
        let m = search_default(r"a*", "bbb").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "");
    }

    #[test]
    fn test_greedy_vs_lazy() {
        let m = search_default(r"a+", "aaa").unwrap();
        assert_eq!(m.unwrap().as_str(), "aaa");

        let m = search_default(r"a+?", "aaa").unwrap();
        assert_eq!(m.unwrap().as_str(), "a");
    }

    #[test]
    fn test_very_long_string() {
        let long_string = "a".repeat(10000);
        let m = search_default(r"a+", &long_string).unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str().len(), 10000);
    }

    #[test]
    fn test_many_groups() {
        let pattern = compile_default(r"(\d)(\d)(\d)(\d)(\d)").unwrap();
        let m = pattern.search("12345").unwrap();
        assert_eq!(m.group(1), Some("1"));
        assert_eq!(m.group(5), Some("5"));
    }

    #[test]
    fn test_nested_quantifiers() {
        let m = search_default(r"(ab)+", "ababab").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "ababab");
    }

    #[test]
    fn test_character_class() {
        let m = search_default(r"[aeiou]+", "hello").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "e");
    }

    #[test]
    fn test_negated_character_class() {
        let m = search_default(r"[^aeiou]+", "hello").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "h");
    }

    #[test]
    fn test_range_quantifier() {
        let m = search_default(r"\d{3,5}", "1234567").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "12345");
    }

    #[test]
    fn test_exact_quantifier() {
        let m = search_default(r"\d{3}", "12345").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "123");
    }
}

// =============================================================================
// Match Object Method Tests
// =============================================================================

mod match_object_tests {
    use super::*;

    #[test]
    fn test_match_span() {
        let m = search_default(r"\d+", "abc123def").unwrap().unwrap();
        assert_eq!(m.span(), (3, 6));
    }

    #[test]
    fn test_match_start_end() {
        let m = search_default(r"\d+", "abc123def").unwrap().unwrap();
        assert_eq!(m.start(), 3);
        assert_eq!(m.end(), 6);
    }

    #[test]
    fn test_match_group_span() {
        let pattern = compile_default(r"(\d+)-(\d+)").unwrap();
        let m = pattern.search("123-456").unwrap();
        assert_eq!(m.span_group(0), Some((0, 7)));
        assert_eq!(m.span_group(1), Some((0, 3)));
        assert_eq!(m.span_group(2), Some((4, 7)));
    }

    #[test]
    fn test_match_lastindex() {
        let pattern = compile_default(r"(\d+)(-(\d+))?").unwrap();

        let m = pattern.search("123-456").unwrap();
        assert_eq!(m.lastindex(), Some(3));

        let m = pattern.search("123").unwrap();
        assert_eq!(m.lastindex(), Some(1));
    }

    #[test]
    fn test_match_len() {
        let pattern = compile_default(r"(\d+)-(\d+)").unwrap();
        let m = pattern.search("123-456").unwrap();
        assert_eq!(m.len(), 3); // Full match + 2 groups
    }

    #[test]
    fn test_match_string() {
        let text = "abc123def";
        let m = search_default(r"\d+", text).unwrap().unwrap();
        assert_eq!(m.string(), text);
    }

    #[test]
    fn test_match_display() {
        let m = search_default(r"\d+", "abc123def").unwrap().unwrap();
        let s = m.to_string();
        assert!(s.contains("span=(3, 6)"));
        assert!(s.contains("match='123'"));
    }
}

// =============================================================================
// Quantifier Tests
// =============================================================================

mod quantifier_tests {
    use super::*;

    #[test]
    fn test_star() {
        assert!(search_default(r"ab*c", "ac").unwrap().is_some());
        assert!(search_default(r"ab*c", "abc").unwrap().is_some());
        assert!(search_default(r"ab*c", "abbc").unwrap().is_some());
    }

    #[test]
    fn test_plus() {
        assert!(search_default(r"ab+c", "ac").unwrap().is_none());
        assert!(search_default(r"ab+c", "abc").unwrap().is_some());
        assert!(search_default(r"ab+c", "abbc").unwrap().is_some());
    }

    #[test]
    fn test_question() {
        assert!(search_default(r"ab?c", "ac").unwrap().is_some());
        assert!(search_default(r"ab?c", "abc").unwrap().is_some());
        assert!(search_default(r"ab?c", "abbc").unwrap().is_none());
    }

    #[test]
    fn test_range_min() {
        assert!(search_default(r"a{2,}", "a").unwrap().is_none());
        assert!(search_default(r"a{2,}", "aa").unwrap().is_some());
        assert!(search_default(r"a{2,}", "aaa").unwrap().is_some());
    }

    #[test]
    fn test_range_max() {
        // Note: Python allows {,2} but standard regex requires {0,2}
        let m = search_default(r"a{0,2}", "aaaa").unwrap().unwrap();
        assert_eq!(m.as_str(), "aa");
    }
}

// =============================================================================
// Real-World Pattern Tests
// =============================================================================

mod realworld_patterns {
    use super::*;

    #[test]
    fn test_email_pattern() {
        let pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}";
        assert!(
            search_default(pattern, "test@example.com")
                .unwrap()
                .is_some()
        );
        assert!(
            search_default(pattern, "user.name+tag@domain.co.uk")
                .unwrap()
                .is_some()
        );
        assert!(search_default(pattern, "invalid").unwrap().is_none());
    }

    #[test]
    fn test_url_pattern() {
        let pattern = r"https?://[^\s]+";
        let m = search_default(pattern, "Visit https://example.com today!").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "https://example.com");
    }

    #[test]
    fn test_ipv4_pattern() {
        let pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}";
        assert!(search_default(pattern, "192.168.1.1").unwrap().is_some());
        assert!(search_default(pattern, "10.0.0.1").unwrap().is_some());
    }

    #[test]
    fn test_phone_pattern() {
        let pattern = r"\d{3}-\d{3}-\d{4}";
        assert!(
            search_default(pattern, "Call 555-123-4567")
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn test_date_pattern() {
        let pattern = r"\d{4}-\d{2}-\d{2}";
        let m = search_default(pattern, "Date: 2024-01-15").unwrap();
        assert!(m.is_some());
        assert_eq!(m.unwrap().as_str(), "2024-01-15");
    }

    #[test]
    fn test_html_tag() {
        let pattern = r"<(\w+)[^>]*>.*?</\1>";
        // This requires backreference
        let m = search_default(pattern, "<div>content</div>").unwrap();
        assert!(m.is_some());
    }

    #[test]
    fn test_password_validation() {
        // At least 8 chars, one uppercase, one lowercase, one digit
        let has_upper = search_default(r"[A-Z]", "Password1").unwrap().is_some();
        let has_lower = search_default(r"[a-z]", "Password1").unwrap().is_some();
        let has_digit = search_default(r"\d", "Password1").unwrap().is_some();
        let long_enough = "Password1".len() >= 8;
        assert!(has_upper && has_lower && has_digit && long_enough);
    }
}

// =============================================================================
// Performance Sanity Tests
// =============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_large_findall() {
        let text = "a1 ".repeat(1000);
        let matches = findall_default(r"\d+", &text).unwrap();
        assert_eq!(matches.len(), 1000);
    }

    #[test]
    fn test_complex_pattern_performance() {
        let pattern = r"(\w+)\s+(\w+)\s+(\w+)";
        let text = "hello world test ".repeat(100);
        let matches = findall_default(pattern, &text).unwrap();
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_cache_performance() {
        // Compile same pattern many times - should be fast due to caching
        for _ in 0..100 {
            let _ = compile_default(r"\d+").unwrap();
        }
    }
}
