//! Static metadata for Prism's native stdlib surface.
//!
//! This crate owns the stable inventory of native stdlib modules and builtin
//! importer metadata that higher-level layers use for planning and import
//! policy decisions.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

/// Import resolution policy for a native stdlib module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdlibResolutionPolicy {
    /// Always load Prism's native module implementation.
    PreferNative,
    /// Prefer a filesystem-backed Python source module when available.
    PreferSourceWhenAvailable,
}

const COMMON_BUILTIN_MODULE_NAMES: &[&str] = &[
    "_abc",
    "_ast",
    "_codecs",
    "_contextvars",
    "_functools",
    "_imp",
    "_io",
    "_overlapped",
    "_random",
    "_sha2",
    "_socket",
    "_ssl",
    "_sre",
    "_string",
    "_struct",
    "_thread",
    "_tokenize",
    "_winapi",
    "_warnings",
    "_weakref",
    "array",
    "atexit",
    "binascii",
    "builtins",
    "gc",
    "io",
    "itertools",
    "marshal",
    "math",
    "select",
    "signal",
    "sys",
    "time",
    "weakref",
    "winreg",
];

const WINDOWS_BUILTIN_MODULE_NAMES: &[&str] = &[
    "_abc",
    "_ast",
    "_codecs",
    "_contextvars",
    "_functools",
    "_imp",
    "_io",
    "_overlapped",
    "_random",
    "_sha2",
    "_socket",
    "_ssl",
    "_sre",
    "_string",
    "_struct",
    "_thread",
    "_tokenize",
    "_winapi",
    "_warnings",
    "_weakref",
    "array",
    "atexit",
    "binascii",
    "builtins",
    "gc",
    "io",
    "itertools",
    "marshal",
    "math",
    "msvcrt",
    "nt",
    "select",
    "signal",
    "sys",
    "time",
    "weakref",
    "winreg",
];

const POSIX_BUILTIN_MODULE_NAMES: &[&str] = &[
    "_abc",
    "_ast",
    "_codecs",
    "_contextvars",
    "_functools",
    "_imp",
    "_io",
    "_random",
    "_sha2",
    "_socket",
    "_ssl",
    "_sre",
    "_string",
    "_struct",
    "_thread",
    "_tokenize",
    "_warnings",
    "_weakref",
    "array",
    "atexit",
    "binascii",
    "builtins",
    "gc",
    "io",
    "itertools",
    "marshal",
    "math",
    "posix",
    "select",
    "signal",
    "sys",
    "time",
    "weakref",
];

/// Returns the builtin modules exposed through importlib's builtin importer.
pub fn builtin_module_names() -> &'static [&'static str] {
    if cfg!(windows) {
        WINDOWS_BUILTIN_MODULE_NAMES
    } else if cfg!(unix) {
        POSIX_BUILTIN_MODULE_NAMES
    } else {
        COMMON_BUILTIN_MODULE_NAMES
    }
}

/// Returns whether a module is exposed through importlib's builtin importer.
#[inline]
pub fn is_builtin_module_name(name: &str) -> bool {
    builtin_module_names().contains(&name)
}

/// Returns the native stdlib resolution policy for a module, if Prism ships one.
pub fn native_module_policy(name: &str) -> Option<StdlibResolutionPolicy> {
    match name {
        "builtins" | "_abc" | "_ast" | "_codecs" | "_contextvars" | "_functools" | "_imp"
        | "_random" | "_sha2" | "_socket" | "_ssl" | "_sre" | "_io" | "_string" | "_struct"
        | "_testcapi" | "_thread" | "_tokenize" | "_tracemalloc" | "_warnings" | "_weakref"
        | "array" | "atexit" | "math" | "errno" | "gc" | "sys" | "time" | "typing" | "signal"
        | "select" | "weakref" | "collections" | "ctypes" | "fnmatch" | "inspect" | "itertools"
        | "io" | "marshal" | "binascii" => Some(StdlibResolutionPolicy::PreferNative),
        "os" | "os.path" | "json" | "functools" | "re" => {
            Some(StdlibResolutionPolicy::PreferSourceWhenAvailable)
        }
        "_overlapped" | "_winapi" | "msvcrt" | "nt" | "winreg" if cfg!(windows) => {
            Some(StdlibResolutionPolicy::PreferNative)
        }
        _ => None,
    }
}

/// Returns whether Prism ships a native stdlib module implementation for `name`.
#[inline]
pub fn is_native_stdlib_module(name: &str) -> bool {
    native_module_policy(name).is_some()
}

/// Returns whether Prism should prefer a source-backed stdlib module for `name`.
#[inline]
pub fn prefers_source_when_available(name: &str) -> bool {
    native_module_policy(name) == Some(StdlibResolutionPolicy::PreferSourceWhenAvailable)
}

#[cfg(test)]
mod tests {
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
}
