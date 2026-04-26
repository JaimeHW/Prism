use super::*;

#[test]
fn test_builtin_module_names_match_platform_surface() {
    assert!(is_builtin_module_name("_imp"));
    assert!(is_builtin_module_name("_io"));
    assert!(is_builtin_module_name("_functools"));
    assert!(is_builtin_module_name("_socket"));
    assert!(is_builtin_module_name("_ssl"));
    assert!(is_builtin_module_name("_sre"));
    assert!(is_builtin_module_name("array"));
    assert!(is_builtin_module_name("binascii"));
    assert!(is_builtin_module_name("builtins"));
    assert!(is_builtin_module_name("select"));
    assert!(is_builtin_module_name("sys"));

    if cfg!(windows) {
        assert!(is_builtin_module_name("_overlapped"));
        assert!(is_builtin_module_name("_winapi"));
        assert!(is_builtin_module_name("msvcrt"));
        assert!(is_builtin_module_name("winreg"));
        assert!(is_builtin_module_name("nt"));
    } else if cfg!(unix) {
        assert!(is_builtin_module_name("posix"));
        assert!(!is_builtin_module_name("_overlapped"));
        assert!(!is_builtin_module_name("_winapi"));
    }
}

#[test]
fn test_native_module_policy_covers_fallback_modules() {
    assert_eq!(
        native_module_policy("json"),
        Some(StdlibResolutionPolicy::PreferSourceWhenAvailable)
    );
    assert_eq!(
        native_module_policy("functools"),
        Some(StdlibResolutionPolicy::PreferSourceWhenAvailable)
    );
    assert_eq!(
        native_module_policy("collections"),
        Some(StdlibResolutionPolicy::PreferNative)
    );
    assert_eq!(
        native_module_policy("array"),
        Some(StdlibResolutionPolicy::PreferNative)
    );
    assert_eq!(
        native_module_policy("binascii"),
        Some(StdlibResolutionPolicy::PreferNative)
    );
    assert_eq!(
        native_module_policy("ctypes"),
        Some(StdlibResolutionPolicy::PreferNative)
    );
    assert_eq!(
        native_module_policy("_functools"),
        Some(StdlibResolutionPolicy::PreferNative)
    );
    assert_eq!(
        native_module_policy("math"),
        Some(StdlibResolutionPolicy::PreferNative)
    );
    assert_eq!(
        native_module_policy("select"),
        Some(StdlibResolutionPolicy::PreferNative)
    );
    assert_eq!(
        native_module_policy("_socket"),
        Some(StdlibResolutionPolicy::PreferNative)
    );
    assert_eq!(
        native_module_policy("_sre"),
        Some(StdlibResolutionPolicy::PreferNative)
    );
}

#[test]
fn test_platform_specific_native_modules_are_gated() {
    if cfg!(windows) {
        assert!(is_native_stdlib_module("_overlapped"));
        assert!(is_native_stdlib_module("_winapi"));
        assert!(is_native_stdlib_module("msvcrt"));
        assert!(is_native_stdlib_module("winreg"));
        assert!(is_native_stdlib_module("nt"));
    } else {
        assert!(!is_native_stdlib_module("_overlapped"));
        assert!(!is_native_stdlib_module("_winapi"));
        assert!(!is_native_stdlib_module("msvcrt"));
        assert!(!is_native_stdlib_module("winreg"));
        assert!(!is_native_stdlib_module("nt"));
    }
}
