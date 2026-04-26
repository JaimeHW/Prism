use super::*;
use std::fs;

// =========================================================================
// DottedName Tests
// =========================================================================

#[test]
fn test_dotted_name_simple() {
    let dn = DottedName::parse("math").unwrap();
    assert_eq!(dn.full_name(), "math");
    assert!(dn.is_simple());
    assert_eq!(dn.depth(), 1);
    assert_eq!(dn.top_level(), "math");
}

#[test]
fn test_dotted_name_two_parts() {
    let dn = DottedName::parse("os.path").unwrap();
    assert_eq!(dn.full_name(), "os.path");
    assert!(!dn.is_simple());
    assert_eq!(dn.depth(), 2);
    assert_eq!(dn.top_level(), "os");
    assert_eq!(dn.parts()[1].as_ref(), "path");
}

#[test]
fn test_dotted_name_three_parts() {
    let dn = DottedName::parse("a.b.c").unwrap();
    assert_eq!(dn.depth(), 3);
    assert_eq!(dn.parts()[0].as_ref(), "a");
    assert_eq!(dn.parts()[1].as_ref(), "b");
    assert_eq!(dn.parts()[2].as_ref(), "c");
}

#[test]
fn test_dotted_name_empty() {
    assert!(DottedName::parse("").is_none());
}

#[test]
fn test_dotted_name_leading_dot() {
    assert!(DottedName::parse(".os").is_none());
}

#[test]
fn test_dotted_name_trailing_dot() {
    assert!(DottedName::parse("os.").is_none());
}

#[test]
fn test_dotted_name_double_dot() {
    assert!(DottedName::parse("os..path").is_none());
}

#[test]
fn test_dotted_name_name_at_depth() {
    let dn = DottedName::parse("a.b.c.d").unwrap();
    assert_eq!(dn.name_at_depth(1), "a");
    assert_eq!(dn.name_at_depth(2), "a.b");
    assert_eq!(dn.name_at_depth(3), "a.b.c");
    assert_eq!(dn.name_at_depth(4), "a.b.c.d");
}

#[test]
fn test_dotted_name_name_at_depth_clamped() {
    let dn = DottedName::parse("a.b").unwrap();
    assert_eq!(dn.name_at_depth(100), "a.b");
}

#[test]
fn test_dotted_name_single_name_at_depth() {
    let dn = DottedName::parse("math").unwrap();
    assert_eq!(dn.name_at_depth(1), "math");
}

// =========================================================================
// Relative Import Tests
// =========================================================================

#[test]
fn test_relative_level_zero() {
    let result = resolve_relative_import("os", 0, "").unwrap();
    assert_eq!(result, "os");
}

#[test]
fn test_relative_level_one_no_name() {
    let result = resolve_relative_import("", 1, "foo.bar.baz").unwrap();
    assert_eq!(result, "foo.bar.baz");
}

#[test]
fn test_relative_level_one_with_name() {
    let result = resolve_relative_import("qux", 1, "foo.bar.baz").unwrap();
    assert_eq!(result, "foo.bar.baz.qux");
}

#[test]
fn test_relative_level_two_with_name() {
    let result = resolve_relative_import("qux", 2, "foo.bar.baz").unwrap();
    assert_eq!(result, "foo.bar.qux");
}

#[test]
fn test_relative_level_three_no_name() {
    let result = resolve_relative_import("", 3, "foo.bar.baz").unwrap();
    assert_eq!(result, "foo");
}

#[test]
fn test_relative_level_three_with_name() {
    let result = resolve_relative_import("x", 3, "foo.bar.baz").unwrap();
    assert_eq!(result, "foo.x");
}

#[test]
fn test_relative_exact_depth() {
    // level equal to package depth — goes to the very top
    let result = resolve_relative_import("new", 3, "a.b.c").unwrap();
    assert_eq!(result, "a.new");
}

#[test]
fn test_relative_beyond_top_level() {
    // level exceeds package depth — error
    let result = resolve_relative_import("x", 4, "a.b.c");
    assert!(result.is_err());
}

#[test]
fn test_relative_non_package() {
    let result = resolve_relative_import("x", 1, "");
    assert!(result.is_err());
}

#[test]
fn test_relative_single_level_package() {
    let result = resolve_relative_import("sub", 1, "pkg").unwrap();
    assert_eq!(result, "pkg.sub");
}

#[test]
fn test_relative_single_level_self() {
    let result = resolve_relative_import("", 1, "pkg").unwrap();
    assert_eq!(result, "pkg");
}

// =========================================================================
// Package Detection Tests
// =========================================================================

#[test]
fn test_is_package_no_dir() {
    assert!(!is_package(Path::new("/nonexistent/dir")));
}

#[test]
fn test_is_package_real() {
    let dir = std::env::temp_dir().join("prism_test_pkg_detect");
    let _ = fs::create_dir_all(&dir);
    let init = dir.join("__init__.py");
    let _ = fs::write(&init, "# init");

    assert!(is_package(&dir));

    // Cleanup
    let _ = fs::remove_file(init);
    let _ = fs::remove_dir(dir);
}

#[test]
fn test_is_package_no_init() {
    let dir = std::env::temp_dir().join("prism_test_pkg_no_init");
    let _ = fs::create_dir_all(&dir);

    assert!(!is_package(&dir));

    let _ = fs::remove_dir(dir);
}

#[test]
fn test_find_init_file() {
    let dir = std::env::temp_dir().join("prism_test_find_init");
    let _ = fs::create_dir_all(&dir);
    let init = dir.join("__init__.py");
    let _ = fs::write(&init, "# init");

    let found = find_init_file(&dir);
    assert!(found.is_some());
    assert_eq!(found.unwrap().file_name().unwrap(), "__init__.py");

    let _ = fs::remove_file(init);
    let _ = fs::remove_dir(dir);
}

#[test]
fn test_find_init_no_init() {
    let dir = std::env::temp_dir().join("prism_test_find_init_missing");
    let _ = fs::create_dir_all(&dir);

    assert!(find_init_file(&dir).is_none());

    let _ = fs::remove_dir(dir);
}

// =========================================================================
// Module Source Search Tests
// =========================================================================

#[test]
fn test_find_module_as_file() {
    let base = std::env::temp_dir().join("prism_test_find_module_file");
    let _ = fs::create_dir_all(&base);
    let mod_file = base.join("mymod.py");
    let _ = fs::write(&mod_file, "x = 1");

    let search = vec![Arc::from(base.to_str().unwrap())];
    let result = find_module_source("mymod", &search);
    assert!(result.is_some());
    let (path, is_pkg) = result.unwrap();
    assert!(!is_pkg);
    assert!(path.to_str().unwrap().ends_with("mymod.py"));

    let _ = fs::remove_file(mod_file);
    let _ = fs::remove_dir(base);
}

#[test]
fn test_find_module_as_package() {
    let base = std::env::temp_dir().join("prism_test_find_module_pkg");
    let pkg_dir = base.join("mypkg");
    let _ = fs::create_dir_all(&pkg_dir);
    let init = pkg_dir.join("__init__.py");
    let _ = fs::write(&init, "# pkg");

    let search = vec![Arc::from(base.to_str().unwrap())];
    let result = find_module_source("mypkg", &search);
    assert!(result.is_some());
    let (path, is_pkg) = result.unwrap();
    assert!(is_pkg);
    assert!(path.to_str().unwrap().contains("__init__.py"));

    let _ = fs::remove_file(init);
    let _ = fs::remove_dir_all(base);
}

#[test]
fn test_find_module_not_found() {
    let base = std::env::temp_dir().join("prism_test_find_module_notfound");
    let _ = fs::create_dir_all(&base);

    let search = vec![Arc::from(base.to_str().unwrap())];
    assert!(find_module_source("notexist", &search).is_none());

    let _ = fs::remove_dir(base);
}

// =========================================================================
// Utility Tests
// =========================================================================

#[test]
fn test_parent_package() {
    assert_eq!(parent_package("os.path"), Some("os"));
    assert_eq!(parent_package("a.b.c"), Some("a.b"));
    assert_eq!(parent_package("math"), None);
}

#[test]
fn test_leaf_name() {
    assert_eq!(leaf_name("os.path"), "path");
    assert_eq!(leaf_name("a.b.c"), "c");
    assert_eq!(leaf_name("math"), "math");
}

#[test]
fn test_is_known_submodule() {
    assert!(is_known_submodule("os.path"));
    assert!(!is_known_submodule("os"));
    assert!(!is_known_submodule("random_module"));
}

// =========================================================================
// Dotted source resolution tests
// =========================================================================

#[test]
fn test_find_dotted_module_source() {
    let base = std::env::temp_dir().join("prism_test_dotted_src");
    let pkg_dir = base.join("mypkg");
    let sub_dir = pkg_dir.join("sub");
    let _ = fs::create_dir_all(&sub_dir);
    let _ = fs::write(pkg_dir.join("__init__.py"), "");
    let _ = fs::write(sub_dir.join("__init__.py"), "");

    let search = vec![Arc::from(base.to_str().unwrap())];
    let dn = DottedName::parse("mypkg.sub").unwrap();
    let result = find_dotted_module_source(&dn, &search);
    assert!(result.is_some());
    let (path, is_pkg) = result.unwrap();
    assert!(is_pkg);
    assert!(path.to_str().unwrap().contains("sub"));

    let _ = fs::remove_dir_all(base);
}

#[test]
fn test_find_dotted_source_not_found() {
    let base = std::env::temp_dir().join("prism_test_dotted_notfound");
    let _ = fs::create_dir_all(&base);

    let search = vec![Arc::from(base.to_str().unwrap())];
    let dn = DottedName::parse("nonexist.sub").unwrap();
    assert!(find_dotted_module_source(&dn, &search).is_none());

    let _ = fs::remove_dir(base);
}
