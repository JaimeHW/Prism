use super::*;

#[test]
fn test_registry_creation() {
    let registry = StdlibRegistry::new();
    assert!(registry.contains("math"));
    assert!(registry.contains("errno"));
    assert!(registry.contains("gc"));
    assert!(registry.contains("ctypes"));
    assert!(registry.contains("builtins"));
    assert!(registry.contains("signal"));
    assert!(registry.contains("select"));
    assert!(registry.contains("_codecs"));
    assert!(registry.contains("_contextvars"));
    assert!(registry.contains("_functools"));
    assert!(registry.contains("_imp"));
    assert!(registry.contains("_random"));
    assert!(registry.contains("_sha2"));
    assert!(registry.contains("_socket"));
    assert!(registry.contains("_ssl"));
    assert!(registry.contains("_sre"));
    assert!(registry.contains("_tokenize"));
    assert!(registry.contains("_weakref"));
    assert!(registry.contains("_warnings"));
    assert!(registry.contains("weakref"));
    assert!(registry.contains("array"));
    if cfg!(windows) {
        assert!(registry.contains("_overlapped"));
        assert!(registry.contains("msvcrt"));
        assert!(registry.contains("nt"));
        assert!(registry.contains("winreg"));
    }
}

#[test]
fn test_registry_get_math() {
    let registry = StdlibRegistry::new();
    let math = registry.get("math");
    assert!(math.is_some());
    assert_eq!(math.unwrap().name(), "math");
}

#[test]
fn test_registry_unknown_module() {
    let registry = StdlibRegistry::new();
    assert!(!registry.contains("nonexistent"));
    assert!(registry.get("nonexistent").is_none());
}

#[test]
fn test_list_modules() {
    let registry = StdlibRegistry::new();
    let modules = registry.list_modules();
    assert!(modules.contains(&"math"));
    assert!(modules.contains(&"builtins"));
}

#[test]
fn test_registry_get_builtins() {
    let registry = StdlibRegistry::new();
    let builtins = registry
        .get("builtins")
        .expect("builtins module should be registered");

    assert_eq!(builtins.name(), "builtins");
    assert!(builtins.get_attr("open").is_ok());
}

#[test]
fn test_registry_marks_fallback_source_preferred_modules() {
    let registry = StdlibRegistry::new();

    assert!(registry.prefers_source_when_available("re"));
    assert!(!registry.prefers_source_when_available("collections"));
    assert!(registry.prefers_source_when_available("os"));
    assert!(!registry.prefers_source_when_available("sys"));
    assert!(!registry.prefers_source_when_available("math"));
    assert!(!registry.prefers_source_when_available("signal"));
    assert!(!registry.prefers_source_when_available("select"));
    assert!(!registry.prefers_source_when_available("_codecs"));
    assert!(!registry.prefers_source_when_available("_imp"));
    assert!(!registry.prefers_source_when_available("_functools"));
    assert!(!registry.prefers_source_when_available("_random"));
    assert!(!registry.prefers_source_when_available("_sha2"));
    assert!(!registry.prefers_source_when_available("_socket"));
    assert!(!registry.prefers_source_when_available("_sre"));
    assert!(!registry.prefers_source_when_available("_tokenize"));
    assert!(!registry.prefers_source_when_available("_weakref"));
    assert!(!registry.prefers_source_when_available("_warnings"));
    assert!(!registry.prefers_source_when_available("weakref"));
    if cfg!(windows) {
        assert!(!registry.prefers_source_when_available("_overlapped"));
    }
}

#[test]
fn test_builtin_module_name_registry_contains_importlib_bootstrap_modules() {
    assert!(is_builtin_module_name("_contextvars"));
    assert!(is_builtin_module_name("_functools"));
    assert!(is_builtin_module_name("_imp"));
    assert!(is_builtin_module_name("_io"));
    assert!(is_builtin_module_name("_random"));
    assert!(is_builtin_module_name("_sha2"));
    assert!(is_builtin_module_name("_socket"));
    assert!(is_builtin_module_name("_ssl"));
    assert!(is_builtin_module_name("_sre"));
    assert!(is_builtin_module_name("_thread"));
    assert!(is_builtin_module_name("_weakref"));
    assert!(is_builtin_module_name("_warnings"));
    assert!(is_builtin_module_name("array"));
    assert!(is_builtin_module_name("select"));
    if cfg!(windows) {
        assert!(is_builtin_module_name("_overlapped"));
        assert!(is_builtin_module_name("_winapi"));
        assert!(is_builtin_module_name("msvcrt"));
        assert!(is_builtin_module_name("winreg"));
    }
    assert!(!is_builtin_module_name("re"));
}
