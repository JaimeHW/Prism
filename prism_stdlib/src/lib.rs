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

/// Absolute path to Prism's source-backed Python stdlib modules in this checkout.
///
/// Native modules remain the preferred implementation for hot runtime paths.
/// This directory is for compatibility modules whose performance is irrelevant
/// to user code startup or steady-state execution, such as test infrastructure.
#[inline]
pub fn source_stdlib_path() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/python")
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
        "builtins"
        | "_abc"
        | "_ast"
        | "_codecs"
        | "_contextvars"
        | "_functools"
        | "_imp"
        | "_random"
        | "_sha2"
        | "_socket"
        | "_ssl"
        | "_sre"
        | "_io"
        | "_string"
        | "_struct"
        | "_testcapi"
        | "_thread"
        | "_tokenize"
        | "_tracemalloc"
        | "_warnings"
        | "_weakref"
        | "array"
        | "ast"
        | "atexit"
        | "math"
        | "errno"
        | "gc"
        | "sys"
        | "struct"
        | "time"
        | "types"
        | "typing"
        | "signal"
        | "select"
        | "shutil"
        | "weakref"
        | "collections"
        | "collections.abc"
        | "copy"
        | "copyreg"
        | "ctypes"
        | "dbm"
        | "fnmatch"
        | "http"
        | "http.cookies"
        | "inspect"
        | "importlib"
        | "importlib.util"
        | "itertools"
        | "io"
        | "marshal"
        | "binascii"
        | "keyword"
        | "locale"
        | "operator"
        | "pickle"
        | "pickletools"
        | "random"
        | "string"
        | "test.support"
        | "test.support.import_helper"
        | "test.support.os_helper"
        | "test.support.threading_helper"
        | "test.support.warnings_helper"
        | "textwrap"
        | "threading"
        | "traceback" => Some(StdlibResolutionPolicy::PreferNative),
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
