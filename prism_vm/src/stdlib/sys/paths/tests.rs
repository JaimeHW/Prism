use super::*;
use prism_core::intern::interned_by_ptr;
use std::path::Path;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

// =========================================================================
// Construction Tests
// =========================================================================

#[test]
fn test_new_empty() {
    let paths = SysPaths::new();
    assert!(paths.is_empty());
    assert_eq!(paths.len(), 0);
}

#[test]
fn test_with_paths() {
    let paths = SysPaths::with_paths(vec![
        "/usr/lib/python".to_string(),
        "/home/user/lib".to_string(),
    ]);
    assert_eq!(paths.len(), 2);
}

#[test]
fn test_sys_prefixes_default_to_executable_directory() {
    let executable = Path::new(r"C:\Prism\bin\prism.exe");
    let prefixes = SysPrefixes::from_env(Some(executable));

    assert_eq!(prefixes.prefix(), r"C:\Prism\bin");
    assert_eq!(prefixes.exec_prefix(), r"C:\Prism\bin");
    assert_eq!(prefixes.base_prefix(), r"C:\Prism\bin");
    assert_eq!(prefixes.base_exec_prefix(), r"C:\Prism\bin");
}

#[test]
fn test_sys_prefixes_honor_pythonhome_single_root() {
    let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
    let executable = Path::new(r"C:\Prism\bin\prism.exe");
    unsafe {
        std::env::set_var("PYTHONHOME", r"C:\Python312");
    }
    let prefixes = SysPrefixes::from_env(Some(executable));
    unsafe {
        std::env::remove_var("PYTHONHOME");
    }

    assert_eq!(prefixes.prefix(), r"C:\Python312");
    assert_eq!(prefixes.exec_prefix(), r"C:\Python312");
    assert_eq!(prefixes.base_prefix(), r"C:\Python312");
    assert_eq!(prefixes.base_exec_prefix(), r"C:\Python312");
}

#[test]
fn test_sys_prefixes_honor_pythonhome_prefix_pair() {
    let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
    let executable = Path::new(r"C:\Prism\bin\prism.exe");
    let pair = if cfg!(windows) {
        r"C:\Python312;C:\Python312\plat"
    } else {
        "/opt/python:/opt/python/plat"
    };
    unsafe {
        std::env::set_var("PYTHONHOME", pair);
    }
    let prefixes = SysPrefixes::from_env(Some(executable));
    unsafe {
        std::env::remove_var("PYTHONHOME");
    }

    if cfg!(windows) {
        assert_eq!(prefixes.prefix(), r"C:\Python312");
        assert_eq!(prefixes.exec_prefix(), r"C:\Python312\plat");
    } else {
        assert_eq!(prefixes.prefix(), "/opt/python");
        assert_eq!(prefixes.exec_prefix(), "/opt/python/plat");
    }
    assert_eq!(prefixes.base_prefix(), prefixes.prefix());
    assert_eq!(prefixes.base_exec_prefix(), prefixes.exec_prefix());
}

// =========================================================================
// Append/Insert/Remove Tests
// =========================================================================

#[test]
fn test_append() {
    let mut paths = SysPaths::new();
    paths.append("/path/one");
    paths.append("/path/two");
    assert_eq!(paths.len(), 2);
    assert_eq!(paths.get(0).map(|s| s.as_ref()), Some("/path/one"));
    assert_eq!(paths.get(1).map(|s| s.as_ref()), Some("/path/two"));
}

#[test]
fn test_insert_beginning() {
    let mut paths = SysPaths::with_paths(vec!["/existing".to_string()]);
    paths.insert(0, "/first");
    assert_eq!(paths.get(0).map(|s| s.as_ref()), Some("/first"));
    assert_eq!(paths.get(1).map(|s| s.as_ref()), Some("/existing"));
}

#[test]
fn test_insert_middle() {
    let mut paths = SysPaths::with_paths(vec!["a".to_string(), "c".to_string()]);
    paths.insert(1, "b");
    assert_eq!(paths.len(), 3);
    assert_eq!(paths.get(1).map(|s| s.as_ref()), Some("b"));
}

#[test]
fn test_insert_end() {
    let mut paths = SysPaths::with_paths(vec!["a".to_string()]);
    paths.insert(1, "b");
    assert_eq!(paths.get(1).map(|s| s.as_ref()), Some("b"));
}

#[test]
fn test_insert_out_of_bounds() {
    let mut paths = SysPaths::new();
    paths.insert(100, "ignored");
    // Should not panic, just be a no-op
    assert!(paths.is_empty());
}

#[test]
fn test_remove() {
    let mut paths = SysPaths::with_paths(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    let removed = paths.remove(1);
    assert_eq!(removed.map(|s| s.to_string()), Some("b".to_string()));
    assert_eq!(paths.len(), 2);
}

#[test]
fn test_remove_first() {
    let mut paths = SysPaths::with_paths(vec!["first".to_string(), "second".to_string()]);
    paths.remove(0);
    assert_eq!(paths.get(0).map(|s| s.as_ref()), Some("second"));
}

#[test]
fn test_remove_out_of_bounds() {
    let mut paths = SysPaths::with_paths(vec!["a".to_string()]);
    let removed = paths.remove(10);
    assert!(removed.is_none());
}

#[test]
fn test_clear() {
    let mut paths = SysPaths::with_paths(vec!["a".to_string(), "b".to_string()]);
    paths.clear();
    assert!(paths.is_empty());
}

// =========================================================================
// Contains Tests
// =========================================================================

#[test]
fn test_contains_found() {
    let paths = SysPaths::with_paths(vec!["/path/a".to_string(), "/path/b".to_string()]);
    assert!(paths.contains("/path/a"));
    assert!(paths.contains("/path/b"));
}

#[test]
fn test_contains_not_found() {
    let paths = SysPaths::with_paths(vec!["/path/a".to_string()]);
    assert!(!paths.contains("/path/b"));
}

#[test]
fn test_contains_empty() {
    let paths = SysPaths::new();
    assert!(!paths.contains("anything"));
}

// =========================================================================
// Iteration Tests
// =========================================================================

#[test]
fn test_iter() {
    let paths = SysPaths::with_paths(vec!["x".to_string(), "y".to_string()]);
    let collected: Vec<&str> = paths.iter().map(|s| s.as_ref()).collect();
    assert_eq!(collected, vec!["x", "y"]);
}

#[test]
fn test_into_iter() {
    let paths = SysPaths::with_paths(vec!["a".to_string(), "b".to_string()]);
    let collected: Vec<&str> = (&paths).into_iter().map(|s| s.as_ref()).collect();
    assert_eq!(collected, vec!["a", "b"]);
}

// =========================================================================
// As Slice Tests
// =========================================================================

#[test]
fn test_as_slice() {
    let paths = SysPaths::with_paths(vec!["test".to_string()]);
    let slice = paths.as_slice();
    assert_eq!(slice.len(), 1);
}

// =========================================================================
// Path Separator Tests
// =========================================================================

#[test]
fn test_path_separator() {
    let sep = path_separator();
    #[cfg(target_os = "windows")]
    assert_eq!(sep, ';');
    #[cfg(not(target_os = "windows"))]
    assert_eq!(sep, ':');
}

// =========================================================================
// Unicode Path Tests
// =========================================================================

#[test]
fn test_unicode_paths() {
    let mut paths = SysPaths::new();
    paths.append("/home/用户/lib");
    paths.append("/data/データ");
    assert_eq!(paths.len(), 2);
    assert!(paths.contains("/home/用户/lib"));
}

// =========================================================================
// Empty String Path Tests
// =========================================================================

#[test]
fn test_empty_string_path() {
    let paths = SysPaths::with_paths(vec!["".to_string()]);
    assert_eq!(paths.len(), 1);
    assert!(paths.contains(""));
}

#[test]
fn test_to_value_roundtrip_paths() {
    let paths = SysPaths::with_paths(vec!["/a".to_string(), "/b".to_string()]);
    let value = paths.to_value();
    let ptr = value
        .as_object_ptr()
        .expect("sys.path should convert to list object");
    let list = unsafe { &*(ptr as *const ListObject) };
    assert_eq!(list.len(), 2);

    let first = list.get(0).expect("sys.path[0] should exist");
    let second = list.get(1).expect("sys.path[1] should exist");
    let first_ptr = first
        .as_string_object_ptr()
        .expect("sys.path[0] should be string") as *const u8;
    let second_ptr = second
        .as_string_object_ptr()
        .expect("sys.path[1] should be string") as *const u8;

    assert_eq!(
        interned_by_ptr(first_ptr)
            .expect("path[0] should resolve")
            .as_ref(),
        "/a"
    );
    assert_eq!(
        interned_by_ptr(second_ptr)
            .expect("path[1] should resolve")
            .as_ref(),
        "/b"
    );
}
