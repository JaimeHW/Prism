//! Python stdlib module implementations.
//!
//! This module provides implementations of Python's standard library modules
//! with maximum performance through direct hardware intrinsics and zero-allocation
//! algorithms.
//!
//! # Modules
//!
//! - `math` - Mathematical functions (sin, cos, sqrt, etc.)
//! - `os` - Operating system interface
//! - `sys` - System-specific parameters and functions
//! - `functools` - Higher-order functions and callable operations
//! - `itertools` - Iterator building blocks for efficient looping
//! - `io` - Core I/O primitives (StringIO, BytesIO, FileMode)

pub mod _abc;
pub mod _ast;
pub mod _codecs;
pub mod _contextvars;
pub mod _functools;
pub mod _imp;
pub mod _overlapped;
pub mod _random;
pub mod _sha2;
pub mod _socket;
pub mod _sre;
pub mod _ssl;
pub mod _string;
pub mod _struct;
pub mod _testcapi;
pub mod _thread;
pub mod _tokenize;
pub mod _tracemalloc;
pub mod _warnings;
pub mod _weakref;
pub mod _winapi;
pub mod array;
pub mod atexit;
pub mod binascii;
pub mod collections;
pub mod ctypes;
pub mod errno;
pub mod exceptions;
pub mod fnmatch;
pub mod functools;
pub mod gc;
pub mod generators;
pub mod inspect;
pub mod io;
pub mod itertools;
pub mod json;
pub mod marshal;
pub mod math;
pub mod msvcrt;
pub mod nt;
pub mod os;
pub mod python_builtins;
pub mod re;
pub(crate) mod secure_random;
pub mod select;
pub mod signal;
pub mod sys;
pub mod time;
pub mod typing;
pub mod weakref;
pub mod winreg;

use crate::builtins::BuiltinRegistry;
use prism_core::Value;
pub use prism_stdlib::StdlibResolutionPolicy;
use std::sync::Arc;

/// Result type for module attribute lookup.
pub type ModuleResult = Result<Value, ModuleError>;

/// Import resolution policy for native stdlib modules.
///
/// Prism ships a mix of true runtime modules (`sys`, `builtins`, `math`) and
/// native fallback implementations for modules that also exist as Python source
/// in a CPython stdlib tree (`re`, `collections`, `functools`, `os`, etc.).
///
/// When a CPython source tree is available on `sys.path`, the fallback modules
/// should defer to the source implementation for correctness. Native-preferred
/// modules remain authoritative regardless of filesystem contents.
/// Errors that can occur during module operations.
#[derive(Debug, Clone)]
pub enum ModuleError {
    /// Attribute not found in module.
    AttributeError(String),
    /// Invalid argument for function.
    ValueError(String),
    /// Type mismatch.
    TypeError(String),
    /// Domain error (e.g., sqrt of negative).
    MathDomainError(String),
    /// Range error (e.g., result too large).
    MathRangeError(String),
    /// OS error (e.g., file not found).
    OSError(String),
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::AttributeError(msg) => write!(f, "AttributeError: {}", msg),
            ModuleError::ValueError(msg) => write!(f, "ValueError: {}", msg),
            ModuleError::TypeError(msg) => write!(f, "TypeError: {}", msg),
            ModuleError::MathDomainError(msg) => write!(f, "math domain error: {}", msg),
            ModuleError::MathRangeError(msg) => write!(f, "math range error: {}", msg),
            ModuleError::OSError(msg) => write!(f, "OSError: {}", msg),
        }
    }
}

impl std::error::Error for ModuleError {}

/// Trait for Python module implementations.
pub trait Module {
    /// Get the module name.
    fn name(&self) -> &str;

    /// Get an attribute from the module.
    fn get_attr(&self, name: &str) -> ModuleResult;

    /// List all attribute names.
    fn dir(&self) -> Vec<Arc<str>> {
        Vec::new() // Default empty impl
    }
}

/// Returns the builtin modules exposed through importlib's builtin importer.
pub(crate) fn builtin_module_names() -> &'static [&'static str] {
    prism_stdlib::builtin_module_names()
}

/// Returns whether a module should be treated as a builtin module for
/// importlib bootstrap purposes.
pub(crate) fn is_builtin_module_name(name: &str) -> bool {
    prism_stdlib::is_builtin_module_name(name)
}

/// Registry of all stdlib modules.
pub struct StdlibRegistry {
    modules: std::collections::HashMap<Arc<str>, RegisteredModule>,
}

struct RegisteredModule {
    module: Box<dyn Module + Send + Sync>,
    policy: StdlibResolutionPolicy,
}

impl StdlibRegistry {
    /// Create a new registry with all stdlib modules.
    pub fn new() -> Self {
        Self::with_optional_sys_args_and_builtins(None, BuiltinRegistry::with_standard_builtins())
    }

    /// Create a registry with an explicit `sys.argv` payload.
    pub fn with_sys_args(args: Vec<String>) -> Self {
        Self::with_optional_sys_args_and_builtins(
            Some(args),
            BuiltinRegistry::with_standard_builtins(),
        )
    }

    /// Create a registry that projects a specific builtin registry.
    pub fn with_builtins(builtins: BuiltinRegistry) -> Self {
        Self::with_optional_sys_args_and_builtins(None, builtins)
    }

    /// Create a registry with explicit `sys.argv` and builtin registry state.
    pub fn with_sys_args_and_builtins(args: Vec<String>, builtins: BuiltinRegistry) -> Self {
        Self::with_optional_sys_args_and_builtins(Some(args), builtins)
    }

    fn with_optional_sys_args_and_builtins(
        sys_args: Option<Vec<String>>,
        builtins: BuiltinRegistry,
    ) -> Self {
        let mut modules: std::collections::HashMap<Arc<str>, RegisteredModule> =
            std::collections::HashMap::new();

        // Register builtins module first so import resolution can bootstrap
        // stdlib modules that rely on `from builtins import ...`.
        Self::insert_module(
            &mut modules,
            "builtins",
            StdlibResolutionPolicy::PreferNative,
            Box::new(python_builtins::BuiltinsModule::new(builtins)),
        );

        Self::insert_module(
            &mut modules,
            "_abc",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_abc::AbcModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_ast",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_ast::AstModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_codecs",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_codecs::CodecsModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_contextvars",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_contextvars::ContextVarsModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_functools",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_functools::FunctoolsNativeModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_imp",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_imp::ImpModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_random",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_random::RandomModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_sha2",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_sha2::Sha2Module::new()),
        );

        Self::insert_module(
            &mut modules,
            "_socket",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_socket::SocketModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_ssl",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_ssl::SslModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_sre",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_sre::SreModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_io",
            StdlibResolutionPolicy::PreferNative,
            Box::new(io::IoModule::with_name("_io")),
        );

        Self::insert_module(
            &mut modules,
            "_string",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_string::StringModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_struct",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_struct::StructModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_thread",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_thread::ThreadModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_testcapi",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_testcapi::TestCapiModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_tracemalloc",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_tracemalloc::TraceMallocModule::new()),
        );

        if cfg!(windows) {
            Self::insert_module(
                &mut modules,
                "_overlapped",
                StdlibResolutionPolicy::PreferNative,
                Box::new(_overlapped::OverlappedModule::new()),
            );

            Self::insert_module(
                &mut modules,
                "_winapi",
                StdlibResolutionPolicy::PreferNative,
                Box::new(_winapi::WinApiModule::new()),
            );

            Self::insert_module(
                &mut modules,
                "msvcrt",
                StdlibResolutionPolicy::PreferNative,
                Box::new(msvcrt::MsvcrtModule::new()),
            );

            Self::insert_module(
                &mut modules,
                "winreg",
                StdlibResolutionPolicy::PreferNative,
                Box::new(winreg::WinregModule::new()),
            );
        }

        Self::insert_module(
            &mut modules,
            "_tokenize",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_tokenize::TokenizeModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_weakref",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_weakref::WeakRefModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "_warnings",
            StdlibResolutionPolicy::PreferNative,
            Box::new(_warnings::WarningsModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "array",
            StdlibResolutionPolicy::PreferNative,
            Box::new(array::ArrayModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "binascii",
            StdlibResolutionPolicy::PreferNative,
            Box::new(binascii::BinasciiModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "ctypes",
            StdlibResolutionPolicy::PreferNative,
            Box::new(ctypes::CtypesModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "marshal",
            StdlibResolutionPolicy::PreferNative,
            Box::new(marshal::MarshalModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "math",
            StdlibResolutionPolicy::PreferNative,
            Box::new(math::MathModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "atexit",
            StdlibResolutionPolicy::PreferNative,
            Box::new(atexit::AtexitModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "errno",
            StdlibResolutionPolicy::PreferNative,
            Box::new(errno::ErrnoModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "gc",
            StdlibResolutionPolicy::PreferNative,
            Box::new(gc::GcModule::new()),
        );

        if cfg!(windows) {
            Self::insert_module(
                &mut modules,
                "nt",
                StdlibResolutionPolicy::PreferNative,
                Box::new(nt::NtModule::new()),
            );
        }

        // Register os module family as native fallbacks. When a CPython stdlib
        // tree is available, the Python source modules are more correct.
        Self::insert_module(
            &mut modules,
            "os",
            StdlibResolutionPolicy::PreferSourceWhenAvailable,
            Box::new(os::OsModule::new()),
        );
        Self::insert_module(
            &mut modules,
            "os.path",
            StdlibResolutionPolicy::PreferSourceWhenAvailable,
            Box::new(os::path::OsPathModule::new()),
        );

        // Register sys module
        let sys_module = match sys_args {
            Some(args) => sys::SysModule::with_args(args),
            None => sys::SysModule::new(),
        };
        Self::insert_module(
            &mut modules,
            "sys",
            StdlibResolutionPolicy::PreferNative,
            Box::new(sys_module),
        );

        // Register time module
        Self::insert_module(
            &mut modules,
            "time",
            StdlibResolutionPolicy::PreferNative,
            Box::new(time::TimeModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "typing",
            StdlibResolutionPolicy::PreferNative,
            Box::new(typing::TypingModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "signal",
            StdlibResolutionPolicy::PreferNative,
            Box::new(signal::SignalModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "weakref",
            StdlibResolutionPolicy::PreferNative,
            Box::new(weakref::WeakrefModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "select",
            StdlibResolutionPolicy::PreferNative,
            Box::new(select::SelectModule::new()),
        );

        // Register pure-Python stdlib fallbacks.
        Self::insert_module(
            &mut modules,
            "re",
            StdlibResolutionPolicy::PreferSourceWhenAvailable,
            Box::new(re::ReModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "json",
            StdlibResolutionPolicy::PreferSourceWhenAvailable,
            Box::new(json::JsonModule::new()),
        );

        // Prefer Prism's native collections module for now. The CPython
        // source implementation currently depends on runtime features such as
        // full f-string lowering and eval() compilation that are not yet
        // complete in Prism, while the native module already provides the
        // compatibility surface needed by core stdlib import chains.
        Self::insert_module(
            &mut modules,
            "collections",
            StdlibResolutionPolicy::PreferNative,
            Box::new(collections::CollectionsModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "functools",
            StdlibResolutionPolicy::PreferSourceWhenAvailable,
            Box::new(functools::FunctoolsModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "fnmatch",
            StdlibResolutionPolicy::PreferNative,
            Box::new(fnmatch::FnmatchModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "inspect",
            StdlibResolutionPolicy::PreferNative,
            Box::new(inspect::InspectModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "itertools",
            StdlibResolutionPolicy::PreferNative,
            Box::new(itertools::ItertoolsModule::new()),
        );

        Self::insert_module(
            &mut modules,
            "io",
            StdlibResolutionPolicy::PreferNative,
            Box::new(io::IoModule::new()),
        );

        Self { modules }
    }

    fn insert_module(
        modules: &mut std::collections::HashMap<Arc<str>, RegisteredModule>,
        name: &'static str,
        policy: StdlibResolutionPolicy,
        module: Box<dyn Module + Send + Sync>,
    ) {
        modules.insert(Arc::from(name), RegisteredModule { module, policy });
    }

    /// Get a module by name.
    pub fn get(&self, name: &str) -> Option<&(dyn Module + Send + Sync)> {
        self.modules.get(name).map(|entry| entry.module.as_ref())
    }

    /// Check if a module exists.
    pub fn contains(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// List all available module names.
    pub fn list_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|k| k.as_ref()).collect()
    }

    /// Returns true when a module should defer to filesystem source if a source
    /// module/package is present on the active search path.
    pub fn prefers_source_when_available(&self, name: &str) -> bool {
        self.modules
            .get(name)
            .is_some_and(|entry| entry.policy == StdlibResolutionPolicy::PreferSourceWhenAvailable)
    }
}

impl Default for StdlibRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
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
}
