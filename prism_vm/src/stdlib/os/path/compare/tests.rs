use super::*;
use std::env;
use std::fs::File;
use std::io::Write;

// =========================================================================
// commonpath Tests
// =========================================================================

#[test]
fn test_commonpath_empty() {
    let paths: Vec<&str> = vec![];
    assert_eq!(commonpath(&paths), Err(CommonPathError::Empty));
}

#[test]
fn test_commonpath_single() {
    let result = commonpath(&["foo/bar"]).unwrap();
    // PathBuf normalizes separators on Windows
    assert!(result == "foo/bar" || result == "foo\\bar");
}

#[test]
fn test_commonpath_relative_same() {
    let result = commonpath(&["foo/bar", "foo/bar"]).unwrap();
    assert!(result == "foo/bar" || result == "foo\\bar");
}

#[test]
fn test_commonpath_relative_siblings() {
    let result = commonpath(&["foo/bar", "foo/baz"]).unwrap();
    assert_eq!(result, "foo");
}

#[test]
fn test_commonpath_relative_nested() {
    let result = commonpath(&["foo/bar/baz", "foo/bar/qux"]).unwrap();
    assert!(result == "foo/bar" || result == "foo\\bar");
}

#[test]
fn test_commonpath_no_common_relative() {
    let result = commonpath(&["alpha", "beta"]);
    assert_eq!(result, Err(CommonPathError::NoCommonPath));
}

#[cfg(not(windows))]
#[test]
fn test_commonpath_absolute_unix() {
    let result = commonpath(&["/usr/lib", "/usr/local"]).unwrap();
    assert_eq!(result, "/usr");
}

#[cfg(not(windows))]
#[test]
fn test_commonpath_absolute_root_only() {
    let result = commonpath(&["/usr", "/var"]).unwrap();
    assert_eq!(result, "/");
}

#[cfg(windows)]
#[test]
fn test_commonpath_absolute_windows() {
    let result = commonpath(&["C:\\Users\\foo", "C:\\Users\\bar"]).unwrap();
    assert!(result.contains("Users"), "result: {}", result);
}

#[test]
fn test_commonpath_mixed_error() {
    #[cfg(not(windows))]
    {
        let result = commonpath(&["/absolute", "relative"]);
        assert_eq!(result, Err(CommonPathError::MixedAbsoluteRelative));
    }
    #[cfg(windows)]
    {
        let result = commonpath(&["C:\\absolute", "relative"]);
        assert_eq!(result, Err(CommonPathError::MixedAbsoluteRelative));
    }
}

#[test]
fn test_commonpath_three_paths() {
    let result = commonpath(&["a/b/c", "a/b/d", "a/b/e"]).unwrap();
    assert!(result == "a/b" || result == "a\\b");
}

#[test]
fn test_commonpath_one_is_prefix_of_other() {
    let result = commonpath(&["a/b", "a/b/c"]).unwrap();
    assert!(result == "a/b" || result == "a\\b");
}

// =========================================================================
// commonprefix Tests
// =========================================================================

#[test]
fn test_commonprefix_empty() {
    let paths: Vec<&str> = vec![];
    assert_eq!(commonprefix(&paths), "");
}

#[test]
fn test_commonprefix_single() {
    assert_eq!(commonprefix(&["hello"]), "hello");
}

#[test]
fn test_commonprefix_identical() {
    assert_eq!(commonprefix(&["abc", "abc"]), "abc");
}

#[test]
fn test_commonprefix_partial() {
    assert_eq!(commonprefix(&["abcdef", "abcxyz"]), "abc");
}

#[test]
fn test_commonprefix_no_common() {
    assert_eq!(commonprefix(&["xyz", "abc"]), "");
}

#[test]
fn test_commonprefix_one_empty() {
    assert_eq!(commonprefix(&["abc", ""]), "");
}

#[test]
fn test_commonprefix_paths_split_mid_component() {
    // Note: commonprefix is a string operation, it may split mid-component
    assert_eq!(commonprefix(&["/usr/lib", "/usr/local"]), "/usr/l");
}

#[test]
fn test_commonprefix_three_strings() {
    assert_eq!(
        commonprefix(&["interspecies", "interstellar", "interstate"]),
        "inters"
    );
}

#[test]
fn test_commonprefix_all_same() {
    assert_eq!(commonprefix(&["abc", "abc", "abc"]), "abc");
}

#[test]
fn test_commonprefix_single_char() {
    assert_eq!(commonprefix(&["a", "ab"]), "a");
}

// =========================================================================
// relpath Tests
// =========================================================================

#[test]
fn test_relpath_same_dir() {
    #[cfg(not(windows))]
    let rel = relpath("/a/b", "/a/b");
    #[cfg(windows)]
    let rel = relpath("C:\\a\\b", "C:\\a\\b");
    assert_eq!(rel, ".");
}

#[test]
fn test_relpath_child() {
    #[cfg(not(windows))]
    {
        let rel = relpath("/a/b/c", "/a");
        assert_eq!(rel, "b/c");
    }
    #[cfg(windows)]
    {
        let rel = relpath("C:\\a\\b\\c", "C:\\a");
        assert!(rel.contains("b") && rel.contains("c"));
    }
}

#[test]
fn test_relpath_parent() {
    #[cfg(not(windows))]
    {
        let rel = relpath("/a", "/a/b/c");
        assert_eq!(rel, "../..");
    }
    #[cfg(windows)]
    {
        let rel = relpath("C:\\a", "C:\\a\\b\\c");
        assert!(rel.contains(".."));
    }
}

#[test]
fn test_relpath_sibling() {
    #[cfg(not(windows))]
    {
        let rel = relpath("/a/b", "/a/c");
        assert_eq!(rel, "../b");
    }
    #[cfg(windows)]
    {
        let rel = relpath("C:\\a\\b", "C:\\a\\c");
        assert!(rel.contains("..") && rel.contains("b"));
    }
}

#[test]
fn test_relpath_deep_divergence() {
    #[cfg(not(windows))]
    {
        let rel = relpath("/a/b/c/d", "/a/x/y/z");
        assert_eq!(rel, "../../../b/c/d");
    }
}

// =========================================================================
// samefile Tests
// =========================================================================

#[test]
fn test_samefile_same_path() {
    let exe = env::current_exe().unwrap();
    assert!(samefile(&exe, &exe));
}

#[test]
fn test_samefile_directory() {
    let temp = env::temp_dir();
    assert!(samefile(&temp, &temp));
}

#[test]
fn test_samefile_different_files() {
    let dir = env::temp_dir();
    let path1 = dir.join("_prism_samefile_1.txt");
    let path2 = dir.join("_prism_samefile_2.txt");

    File::create(&path1).unwrap().write_all(b"a").unwrap();
    File::create(&path2).unwrap().write_all(b"b").unwrap();

    assert!(!samefile(&path1, &path2));

    let _ = fs::remove_file(&path1);
    let _ = fs::remove_file(&path2);
}

#[test]
fn test_samefile_nonexistent_first() {
    let exe = env::current_exe().unwrap();
    assert!(!samefile("/nonexistent_12345", &exe));
}

#[test]
fn test_samefile_nonexistent_second() {
    let exe = env::current_exe().unwrap();
    assert!(!samefile(&exe, "/nonexistent_12345"));
}

#[test]
fn test_samefile_both_nonexistent() {
    assert!(!samefile("/nonexistent_12345", "/also_nonexistent_12345"));
}

#[test]
fn test_samefile_canonical_equivalence() {
    // "." and the actual CWD should be the same file
    let cwd = env::current_dir().unwrap();
    assert!(samefile(".", &cwd));
}

// =========================================================================
// CommonPathError Display Tests
// =========================================================================

#[test]
fn test_commonpath_error_display_empty() {
    let e = CommonPathError::Empty;
    assert!(e.to_string().contains("empty"));
}

#[test]
fn test_commonpath_error_display_mixed() {
    let e = CommonPathError::MixedAbsoluteRelative;
    assert!(e.to_string().contains("mix"));
}

#[test]
fn test_commonpath_error_display_no_common() {
    let e = CommonPathError::NoCommonPath;
    assert!(e.to_string().contains("no common"));
}
