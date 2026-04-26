use super::*;

#[test]
fn test_standard_engine_compile() {
    let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
    assert_eq!(engine.kind(), EngineKind::Standard);
}

#[test]
fn test_standard_engine_is_match() {
    let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
    assert!(engine.is_match("abc123def"));
    assert!(!engine.is_match("abcdef"));
}

#[test]
fn test_standard_engine_find() {
    let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
    let m = engine.find("abc123def").unwrap();
    assert_eq!(m.start(), 3);
    assert_eq!(m.end(), 6);
}

#[test]
fn test_standard_engine_find_all() {
    let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
    let matches = engine.find_all("a1b22c333");
    assert_eq!(matches.len(), 3);
}

#[test]
fn test_standard_engine_match_start() {
    let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
    assert!(engine.match_start("123abc").is_some());
    assert!(engine.match_start("abc123").is_none());
}

#[test]
fn test_standard_engine_replace() {
    let engine = StandardEngine::compile(r"\d+", RegexFlags::default()).unwrap();
    assert_eq!(engine.replace("a1b2c3", "X"), "aXb2c3");
    assert_eq!(engine.replace_all("a1b2c3", "X"), "aXbXcX");
}

#[test]
fn test_standard_engine_split() {
    let engine = StandardEngine::compile(r",", RegexFlags::default()).unwrap();
    let parts = engine.split("a,b,c");
    assert_eq!(parts, vec!["a", "b", "c"]);
}

#[test]
fn test_case_insensitive() {
    let flags = RegexFlags::new(RegexFlags::IGNORECASE);
    let engine = StandardEngine::compile(r"hello", flags).unwrap();
    assert!(engine.is_match("HELLO"));
    assert!(engine.is_match("Hello"));
}

#[test]
fn test_multiline() {
    let flags = RegexFlags::new(RegexFlags::MULTILINE);
    let engine = StandardEngine::compile(r"^line", flags).unwrap();
    let matches = engine.find_all("line1\nline2\nline3");
    assert_eq!(matches.len(), 3);
}

#[test]
fn test_dotall() {
    let flags = RegexFlags::new(RegexFlags::DOTALL);
    let engine = StandardEngine::compile(r"a.b", flags).unwrap();
    assert!(engine.is_match("a\nb"));
}

#[test]
fn test_requires_fancy_backreference() {
    assert!(requires_fancy_engine(r"(.)\1"));
    assert!(requires_fancy_engine(r#"(?P<quote>['"])(?P=quote)"#));
    assert!(!requires_fancy_engine(r"\111"));
    assert!(!requires_fancy_engine(r"[\041-\176]+:$"));
    assert!(!requires_fancy_engine(r"\\1"));
    assert!(!requires_fancy_engine(r"\d+"));
}

#[test]
fn test_python_named_backreference_uses_fancy_engine() {
    let engine = compile_pattern(
        r#"^(?P<name>\w+)=(?P<quote>["']?)(?P<value>.*)(?P=quote)$"#,
        RegexFlags::default(),
    )
    .expect("named Python backreference should compile");
    assert_eq!(engine.kind(), EngineKind::Fancy);
    let matched = engine.match_start("NAME=\"Prism\"").expect("match");
    assert_eq!(
        matched.group(matched.group_index("name").expect("name group")),
        Some("NAME")
    );
    assert_eq!(
        matched.group(matched.group_index("value").expect("value group")),
        Some("Prism")
    );
}

#[test]
fn test_requires_fancy_lookahead() {
    assert!(requires_fancy_engine(r"foo(?=bar)"));
    assert!(requires_fancy_engine(r"foo(?!bar)"));
}

#[test]
fn test_requires_fancy_lookbehind() {
    assert!(requires_fancy_engine(r"(?<=foo)bar"));
    assert!(requires_fancy_engine(r"(?<!foo)bar"));
}

#[test]
fn test_fancy_engine_compile() {
    let engine = FancyEngine::compile(r"(.)\1", RegexFlags::default()).unwrap();
    assert_eq!(engine.kind(), EngineKind::Fancy);
}

#[test]
fn test_fancy_engine_backreference() {
    let engine = FancyEngine::compile(r"(.)\1", RegexFlags::default()).unwrap();
    assert!(engine.is_match("aa"));
    assert!(engine.is_match("bb"));
    assert!(!engine.is_match("ab"));
}

#[test]
fn test_fancy_engine_lookahead() {
    let engine = FancyEngine::compile(r"foo(?=bar)", RegexFlags::default()).unwrap();
    let m = engine.find("foobar");
    assert!(m.is_some());
    let m = m.unwrap();
    assert_eq!(m.as_str(), "foo");
}

#[test]
fn test_compile_pattern_auto_select() {
    let simple = compile_pattern(r"\d+", RegexFlags::default()).unwrap();
    assert_eq!(simple.kind(), EngineKind::Standard);

    let fancy = compile_pattern(r"(.)\1", RegexFlags::default()).unwrap();
    assert_eq!(fancy.kind(), EngineKind::Fancy);
}

#[test]
fn test_fancy_engine_reports_named_group_metadata() {
    let engine = compile_pattern(r"(?P<word>foo)(?!bar)", RegexFlags::default()).unwrap();
    assert_eq!(engine.kind(), EngineKind::Fancy);
    let group_names = engine.group_names();
    assert_eq!(group_names.len(), 2);
    assert_eq!(group_names[1].as_deref(), Some("word"));
}

#[test]
fn test_normalize_python_pattern_converts_end_of_string_anchor() {
    assert_eq!(normalize_python_pattern(r"foo\Z").unwrap(), r"foo\z");
}

#[test]
fn test_normalize_python_pattern_preserves_escaped_anchor_literal() {
    assert_eq!(normalize_python_pattern(r"foo\\Z").unwrap(), r"foo\\Z");
}

#[test]
fn test_normalize_python_pattern_translates_ascii_scoped_flag_group() {
    assert_eq!(
        normalize_python_pattern(r"(?a:[_a-z][_a-z0-9]*)").unwrap(),
        r"(?-u:[_a-z][_a-z0-9]*)"
    );
}

#[test]
fn test_normalize_python_pattern_escapes_literal_braces() {
    assert_eq!(
        normalize_python_pattern(r"{(?P<braced>\w+)}").unwrap(),
        r"\{(?P<braced>\w+)\}"
    );
}

#[test]
fn test_normalize_python_pattern_translates_open_lower_quantifier_bound() {
    assert_eq!(normalize_python_pattern(r"a{,2}").unwrap(), r"a{0,2}");
}

#[test]
fn test_normalize_python_pattern_escapes_literal_open_bracket_inside_class() {
    assert_eq!(normalize_python_pattern(r"([*?[])").unwrap(), r"([*?\[])");
}

#[test]
fn test_normalize_python_pattern_translates_octal_escapes() {
    assert_eq!(
        normalize_python_pattern(r"[\041-\176]+:$").unwrap(),
        r"[\x{21}-\x{7e}]+:$"
    );
    assert_eq!(
        normalize_python_pattern(r"\0\01\018").unwrap(),
        r"\x{0}\x{1}\x{1}8"
    );
    assert_eq!(normalize_python_pattern(r"\111").unwrap(), r"\x{49}");
}

#[test]
fn test_normalize_python_pattern_rejects_out_of_range_octal_escapes() {
    let outside_class = normalize_python_pattern(r"\567").unwrap_err();
    assert_eq!(
        outside_class.message,
        r"octal escape value \567 outside of range 0-0o377"
    );
    assert_eq!(outside_class.position, Some(0));

    let inside_class = normalize_python_pattern(r"[\567]").unwrap_err();
    assert_eq!(
        inside_class.message,
        r"octal escape value \567 outside of range 0-0o377"
    );
    assert_eq!(inside_class.position, Some(1));
}

#[test]
fn test_prepare_pattern_for_backend_applies_ascii_and_verbose_flags() {
    let prepared = prepare_pattern_for_backend(
        r"\$(?:(?P<named>(?a:[_a-z][_a-z0-9]*)))",
        RegexFlags::new(RegexFlags::IGNORECASE | RegexFlags::VERBOSE),
    )
    .unwrap();
    assert!(prepared.starts_with("(?ix)"));
    assert!(prepared.contains(r"(?-u:[_a-z][_a-z0-9]*)"));
}

#[test]
fn test_prepare_pattern_for_backend_normalizes_atomic_groups() {
    let prepared = prepare_pattern_for_backend(r"(?s:(?>.*?a).*)\Z", RegexFlags::default())
        .expect("atomic groups should normalize for backend regex engines");
    assert_eq!(prepared, r"(?s:(?:.*?a).*)\z");
}

#[test]
fn test_standard_engine_compiles_string_template_identifier_pattern() {
    let pattern = r"
            \$(?:
              (?P<escaped>\$)  |
              (?P<named>(?a:[_a-z][_a-z0-9]*)) |
              {(?P<braced>(?a:[_a-z][_a-z0-9]*))} |
              (?P<invalid>)
            )
        ";
    let flags = RegexFlags::new(RegexFlags::IGNORECASE | RegexFlags::VERBOSE);
    let engine = StandardEngine::compile(pattern, flags).expect("template pattern should compile");
    assert!(engine.is_match("$name"));
}

#[test]
fn test_standard_engine_accepts_python_end_of_string_anchor() {
    let engine =
        StandardEngine::compile(r"foo\Z", RegexFlags::default()).expect(r"\Z should compile");
    assert!(engine.is_match("foo"));
    assert!(!engine.is_match("foo\n"));
}

#[test]
fn test_fancy_engine_accepts_python_end_of_string_anchor() {
    let engine = FancyEngine::compile(r"(?=foo)foo\Z", RegexFlags::default())
        .expect(r"\Z should normalize before fancy-regex parsing");
    assert!(engine.is_match("foo"));
    assert!(!engine.is_match("foo\n"));
}

#[test]
fn test_fancy_engine_compiles_cpython_textwrap_wordsep_anchor() {
    let whitespace = concat!("[", "\\\t", "\\\n", "\\\x0b", "\\\x0c", "\\\r", "\\ ", "]");
    let pattern = r#"
            ( # any whitespace
              %(ws)s+
            | # em-dash between words
              (?<=[\w!"'&.,?]) -{2,} (?=\w)
            | # word, possibly hyphenated
              [^%(ws_tail)s+? (?:
                # hyphenated word
                  -(?: (?<=[^\d\W]{2}-) | (?<=[^\d\W]-[^\d\W]-))
                  (?= [^\d\W] -? [^\d\W])
                | # end of word
                  (?=%(ws)s|\Z)
                | # em-dash
                  (?<=[\w!"'&.,?]) (?=-{2,}\w)
                )
            )"#
    .replace("%(ws)s", whitespace)
    .replace("%(ws_tail)s", &whitespace[1..]);

    let engine = FancyEngine::compile(&pattern, RegexFlags::new(RegexFlags::VERBOSE))
        .expect("CPython textwrap word separator pattern should compile");
    assert!(engine.is_match("word"));
}

#[test]
fn test_standard_engine_accepts_glob_magic_character_class_pattern() {
    let engine = StandardEngine::compile(r"([*?[])", RegexFlags::default())
        .expect("glob's magic-check pattern should compile");
    assert!(engine.is_match("["));
    assert!(engine.is_match("*"));
    assert!(engine.is_match("?"));
}

#[test]
fn test_standard_engine_accepts_cpython_email_header_character_range() {
    let engine = StandardEngine::compile(r"[\041-\176]+:$", RegexFlags::default())
        .expect("email.header's printable ASCII range should compile");
    assert!(engine.is_match("Subject:"));
    assert!(!engine.is_match("Subject"));
}
