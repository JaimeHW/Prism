use super::*;

// -------------------------------------------------------------------------
// Basic Mode Parsing
// -------------------------------------------------------------------------

#[test]
fn test_parse_read_mode() {
    let mode = FileMode::parse("r").unwrap();
    assert!(mode.read);
    assert!(!mode.write);
    assert!(!mode.binary);
    assert!(!mode.append);
    assert!(!mode.truncate);
}

#[test]
fn test_parse_write_mode() {
    let mode = FileMode::parse("w").unwrap();
    assert!(!mode.read);
    assert!(mode.write);
    assert!(!mode.binary);
    assert!(mode.truncate);
}

#[test]
fn test_parse_append_mode() {
    let mode = FileMode::parse("a").unwrap();
    assert!(!mode.read);
    assert!(mode.write);
    assert!(mode.append);
    assert!(!mode.truncate);
}

#[test]
fn test_parse_exclusive_mode() {
    let mode = FileMode::parse("x").unwrap();
    assert!(!mode.read);
    assert!(mode.write);
    assert!(mode.exclusive);
}

// -------------------------------------------------------------------------
// Binary Mode
// -------------------------------------------------------------------------

#[test]
fn test_parse_read_binary() {
    let mode = FileMode::parse("rb").unwrap();
    assert!(mode.read);
    assert!(mode.binary);
}

#[test]
fn test_parse_write_binary() {
    let mode = FileMode::parse("wb").unwrap();
    assert!(mode.write);
    assert!(mode.binary);
    assert!(mode.truncate);
}

#[test]
fn test_parse_append_binary() {
    let mode = FileMode::parse("ab").unwrap();
    assert!(mode.write);
    assert!(mode.append);
    assert!(mode.binary);
}

// -------------------------------------------------------------------------
// Update Mode (+)
// -------------------------------------------------------------------------

#[test]
fn test_parse_read_update() {
    let mode = FileMode::parse("r+").unwrap();
    assert!(mode.read);
    assert!(mode.write);
    assert!(!mode.truncate);
}

#[test]
fn test_parse_write_update() {
    let mode = FileMode::parse("w+").unwrap();
    assert!(mode.read);
    assert!(mode.write);
    assert!(mode.truncate);
}

#[test]
fn test_parse_append_update() {
    let mode = FileMode::parse("a+").unwrap();
    assert!(mode.read);
    assert!(mode.write);
    assert!(mode.append);
}

#[test]
fn test_parse_read_binary_update() {
    let mode = FileMode::parse("rb+").unwrap();
    assert!(mode.read);
    assert!(mode.write);
    assert!(mode.binary);
}

#[test]
fn test_parse_binary_read_update_order() {
    // Order shouldn't matter
    let mode1 = FileMode::parse("r+b").unwrap();
    let mode2 = FileMode::parse("rb+").unwrap();
    assert_eq!(mode1.read, mode2.read);
    assert_eq!(mode1.write, mode2.write);
    assert_eq!(mode1.binary, mode2.binary);
}

// -------------------------------------------------------------------------
// Text Mode Explicit
// -------------------------------------------------------------------------

#[test]
fn test_parse_text_mode_explicit() {
    let mode = FileMode::parse("rt").unwrap();
    assert!(mode.read);
    assert!(!mode.binary);
}

#[test]
fn test_parse_write_text_explicit() {
    let mode = FileMode::parse("wt").unwrap();
    assert!(mode.write);
    assert!(!mode.binary);
}

// -------------------------------------------------------------------------
// Error Cases
// -------------------------------------------------------------------------

#[test]
fn test_parse_empty_mode() {
    assert!(matches!(FileMode::parse(""), Err(ParseModeError::Empty)));
}

#[test]
fn test_parse_invalid_character() {
    assert!(matches!(
        FileMode::parse("rz"),
        Err(ParseModeError::InvalidCharacter('z'))
    ));
}

#[test]
fn test_parse_duplicate_read() {
    assert!(FileMode::parse("rr").is_err());
}

#[test]
fn test_parse_conflicting_modes() {
    assert!(FileMode::parse("rw").is_err());
    assert!(FileMode::parse("ra").is_err());
    assert!(FileMode::parse("wa").is_err());
}

#[test]
fn test_parse_duplicate_binary() {
    assert!(FileMode::parse("rbb").is_err());
}

#[test]
fn test_parse_conflicting_bt() {
    assert!(FileMode::parse("rbt").is_err());
    assert!(FileMode::parse("rtb").is_err());
}

#[test]
fn test_parse_duplicate_plus() {
    assert!(FileMode::parse("r++").is_err());
}

// -------------------------------------------------------------------------
// Display Round-Trip
// -------------------------------------------------------------------------

#[test]
fn test_display_read() {
    let mode = FileMode::parse("r").unwrap();
    assert_eq!(mode.to_string(), "r");
}

#[test]
fn test_display_write_binary() {
    let mode = FileMode::parse("wb").unwrap();
    assert_eq!(mode.to_string(), "wb");
}

#[test]
fn test_display_append() {
    let mode = FileMode::parse("a").unwrap();
    assert_eq!(mode.to_string(), "a");
}

// -------------------------------------------------------------------------
// Helper Methods
// -------------------------------------------------------------------------

#[test]
fn test_requires_existing() {
    assert!(FileMode::parse("r").unwrap().requires_existing());
    assert!(!FileMode::parse("w").unwrap().requires_existing());
    assert!(!FileMode::parse("r+").unwrap().requires_existing());
}

#[test]
fn test_creates_file() {
    assert!(!FileMode::parse("r").unwrap().creates_file());
    assert!(FileMode::parse("w").unwrap().creates_file());
    assert!(FileMode::parse("a").unwrap().creates_file());
    assert!(FileMode::parse("x").unwrap().creates_file());
}

// -------------------------------------------------------------------------
// Edge Cases and Stress Tests
// -------------------------------------------------------------------------

#[test]
fn test_all_valid_single_char_modes() {
    for mode in ["r", "w", "a", "x"] {
        assert!(FileMode::parse(mode).is_ok(), "Failed for mode: {}", mode);
    }
}

#[test]
fn test_all_valid_binary_modes() {
    for mode in ["rb", "wb", "ab", "xb"] {
        assert!(FileMode::parse(mode).is_ok(), "Failed for mode: {}", mode);
    }
}

#[test]
fn test_all_update_modes() {
    for mode in ["r+", "w+", "a+", "r+b", "w+b", "a+b", "rb+", "wb+", "ab+"] {
        assert!(FileMode::parse(mode).is_ok(), "Failed for mode: {}", mode);
    }
}
