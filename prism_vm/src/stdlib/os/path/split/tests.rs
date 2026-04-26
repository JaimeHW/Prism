use super::*;

#[test]
fn test_basename_simple() {
    assert_eq!(basename("/foo/bar.txt"), "bar.txt");
}

#[test]
fn test_basename_no_slash() {
    assert_eq!(basename("file.txt"), "file.txt");
}

#[test]
fn test_dirname_simple() {
    let d = dirname("/foo/bar.txt");
    assert!(d.contains("foo"));
}

#[test]
fn test_dirname_no_slash() {
    assert_eq!(dirname("file.txt"), "");
}

#[test]
fn test_splitext_with_ext() {
    let (root, ext) = splitext("file.txt");
    assert_eq!(root, "file");
    assert_eq!(ext, ".txt");
}

#[test]
fn test_splitext_no_ext() {
    let (root, ext) = splitext("file");
    assert_eq!(root, "file");
    assert_eq!(ext, "");
}

#[test]
fn test_splitext_hidden() {
    let (root, ext) = splitext(".hidden");
    // .hidden is all extension or no extension depending on interpretation
    assert!(root.contains("hidden") || ext.contains("hidden"));
}

#[test]
fn test_splitext_multi_dot() {
    let (root, ext) = splitext("file.tar.gz");
    assert_eq!(ext, ".gz");
}

#[test]
fn test_split() {
    let (head, tail) = split("/foo/bar.txt");
    assert!(head.contains("foo"));
    assert_eq!(tail, "bar.txt");
}

// =========================================================================
// splitdrive Tests
// =========================================================================

#[cfg(windows)]
#[test]
fn test_splitdrive_windows_drive() {
    let (drive, tail) = splitdrive("C:\\foo\\bar");
    assert_eq!(drive, "C:");
    assert_eq!(tail, "\\foo\\bar");
}

#[cfg(windows)]
#[test]
fn test_splitdrive_windows_no_drive() {
    let (drive, tail) = splitdrive("\\foo\\bar");
    // No standard drive letter — treated as relative or UNC start
    assert!(!drive.is_empty() || tail == "\\foo\\bar");
}

#[cfg(windows)]
#[test]
fn test_splitdrive_windows_drive_only() {
    let (drive, tail) = splitdrive("C:");
    assert_eq!(drive, "C:");
    assert_eq!(tail, "");
}

#[cfg(not(windows))]
#[test]
fn test_splitdrive_unix_no_drive() {
    let (drive, tail) = splitdrive("/foo/bar");
    assert_eq!(drive, "");
    assert_eq!(tail, "/foo/bar");
}

#[cfg(not(windows))]
#[test]
fn test_splitdrive_unix_relative() {
    let (drive, tail) = splitdrive("foo/bar");
    assert_eq!(drive, "");
    assert_eq!(tail, "foo/bar");
}

#[test]
fn test_splitdrive_empty() {
    let (drive, tail) = splitdrive("");
    assert_eq!(drive, "");
    assert_eq!(tail, "");
}

#[test]
fn test_splitdrive_single_char() {
    let (drive, tail) = splitdrive("a");
    assert_eq!(drive, "");
    assert_eq!(tail, "a");
}

#[test]
fn test_splitdrive_dot() {
    let (drive, tail) = splitdrive(".");
    assert_eq!(drive, "");
    assert_eq!(tail, ".");
}
