//! `ImportResolver` - the core import machinery.
//!
//! Provides high-performance module resolution with multi-tier caching:
//! 1. `sys.modules` cache (first check)
//! 2. Stdlib registry (built-in modules)
//! 3. File system (future: .py/.pyc loading)

use super::{FrozenModuleSource, ModuleObject};
use crate::builtins::BuiltinRegistry;
use crate::stdlib::{Module, ModuleError, StdlibRegistry};
use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::value_as_list_ref;
use prism_runtime::types::string::value_as_string_ref;
use rustc_hash::FxHashMap;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::ThreadId;

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during import resolution.
#[derive(Debug, Clone)]
pub enum ImportError {
    /// Module not found in any search path.
    ModuleNotFound { module: Arc<str> },

    /// Attribute not found in module.
    ImportFromError { module: Arc<str>, name: Arc<str> },

    /// Circular import detected.
    CircularImport { module: Arc<str> },

    /// Error loading module source.
    LoadError { module: Arc<str>, message: Arc<str> },

    /// Error executing module code.
    ExecutionError { module: Arc<str>, message: Arc<str> },

    /// Error resolving the export set for `from module import *`.
    StarImportError { module: Arc<str>, message: Arc<str> },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::ModuleNotFound { module } => {
                write!(f, "ModuleNotFoundError: No module named '{}'", module)
            }
            ImportError::ImportFromError { module, name } => {
                write!(
                    f,
                    "ImportError: cannot import name '{}' from '{}'",
                    name, module
                )
            }
            ImportError::CircularImport { module } => {
                write!(f, "ImportError: circular import for '{}'", module)
            }
            ImportError::LoadError { module, message } => {
                write!(f, "ImportError: failed to load '{}': {}", module, message)
            }
            ImportError::ExecutionError { module, message } => {
                write!(
                    f,
                    "ImportError: failed to execute '{}': {}",
                    module, message
                )
            }
            ImportError::StarImportError { module, message } => {
                write!(
                    f,
                    "ImportError: cannot import * from '{}': {}",
                    module, message
                )
            }
        }
    }
}

impl std::error::Error for ImportError {}

/// Filesystem location of a Python source module or package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceModuleLocation {
    /// Path to the source file backing the module.
    pub path: PathBuf,
    /// Whether the path points at a package `__init__.py`.
    pub is_package: bool,
}

/// Resolved import action for a canonical module name.
#[derive(Debug, Clone)]
pub enum ImportLoadPlan {
    /// Module is already present in `sys.modules`.
    Cached(Arc<ModuleObject>),
    /// Execute an AOT/frozen source module.
    Frozen(Arc<FrozenModuleSource>),
    /// Compile and execute a filesystem source module.
    Source(SourceModuleLocation),
    /// Load Prism's native stdlib implementation.
    Native,
    /// No resolver tier can provide the module.
    Missing,
}

// =============================================================================
// ImportState - Concurrent Import Synchronization
// =============================================================================

/// State for a module currently being imported.
///
/// Tracks the loading thread ID to detect true circular imports (same thread
/// re-entering import) vs concurrent imports (different threads, which should
/// wait for the first one to complete).
struct ImportState {
    /// Thread ID of the thread performing the import.
    /// Used to detect true circular imports (same thread re-entering).
    loader_thread: ThreadId,

    /// Whether the import has completed.
    /// Protected by mutex and used with condvar for wait semantics.
    completed: Mutex<bool>,

    /// Condvar to signal when import completes.
    complete: Condvar,
}

impl ImportState {
    /// Create new import state for the current thread.
    fn new() -> Self {
        Self {
            loader_thread: std::thread::current().id(),
            completed: Mutex::new(false),
            complete: Condvar::new(),
        }
    }

    /// Check if this import is being performed by the current thread.
    /// If true, this is a true circular import.
    fn is_circular(&self) -> bool {
        self.loader_thread == std::thread::current().id()
    }

    /// Wait for the import to complete.
    /// Returns when the import has finished and the module is cached.
    fn wait(&self) {
        let mut completed = self.completed.lock().unwrap();
        while !*completed {
            // Wait for signal that import is complete
            // Use wait_while pattern to handle spurious wakeups
            completed = self.complete.wait(completed).unwrap();
        }
    }

    /// Signal all waiting threads that import is complete.
    fn signal_complete(&self) {
        let mut completed = self.completed.lock().unwrap();
        *completed = true;
        // Wake all waiting threads
        self.complete.notify_all();
    }
}

// =============================================================================
// ImportResolver
// =============================================================================

/// High-performance import resolver with multi-tier caching.
///
/// # Resolution Order
///
/// 1. **sys.modules cache** - O(1) lookup for already-imported modules
/// 2. **Stdlib registry** - Direct access to built-in modules (math, os, sys)
/// 3. **File system** - (Future) Search sys.path for .py/.pyc files
///
/// # Thread Safety
///
/// Uses `RwLock` for the sys.modules cache to allow concurrent reads
/// (the common case) while serializing writes. Cloned resolvers share import
/// state so native Python threads see one interpreter-wide `sys.modules`,
/// `sys.path`, frozen module table, and in-progress import lock map.
///
/// # Performance
///
/// - Cache keys are `InternedString` for O(1) hash/equality
/// - Built-in modules are pre-initialized (no parsing/compilation)
/// - `Arc<ModuleObject>` for zero-copy module sharing
#[derive(Clone)]
pub struct ImportResolver {
    /// sys.modules: canonical cache of all loaded modules.
    /// Uses InternedString keys for O(1) lookup.
    sys_modules: Arc<RwLock<FxHashMap<InternedString, Arc<ModuleObject>>>>,

    /// Python-visible `sys.modules` mapping shared with imported `sys`.
    sys_modules_value: Value,

    /// Stdlib registry for built-in modules (math, os, sys, etc.).
    stdlib: Arc<StdlibRegistry>,

    /// sys.path: search paths for source files.
    /// Future use for .py file loading.
    search_paths: Arc<RwLock<Vec<Arc<str>>>>,

    /// Frozen source modules installed by the AOT bootstrap path.
    frozen_modules: Arc<RwLock<FxHashMap<InternedString, Arc<FrozenModuleSource>>>>,

    /// Modules currently being imported (for circular import detection and wait semantics).
    /// Maps module name to ImportState which tracks the loading thread and provides
    /// a Condvar for other threads to wait on.
    loading: Arc<RwLock<FxHashMap<InternedString, Arc<ImportState>>>>,

    /// Fast pointer lookup for imported module objects.
    ///
    /// Keys are raw `ModuleObject` pointers cast to usize. This lets opcode handlers
    /// validate and resolve module pointers without unsafe casting.
    module_ptrs: Arc<RwLock<FxHashMap<usize, Arc<ModuleObject>>>>,
}

impl ImportResolver {
    /// Create a new import resolver with default configuration.
    pub fn new() -> Self {
        Self::with_stdlib_and_paths(StdlibRegistry::new(), Vec::new())
    }

    /// Create a new import resolver with explicit `sys.argv`.
    pub fn with_sys_args(args: Vec<String>) -> Self {
        Self::with_stdlib_and_paths(StdlibRegistry::with_sys_args(args), Vec::new())
    }

    /// Create a new import resolver sharing an explicit builtin registry.
    pub fn new_with_builtins(builtins: BuiltinRegistry) -> Self {
        Self::with_stdlib_and_paths(StdlibRegistry::with_builtins(builtins), Vec::new())
    }

    /// Create a new resolver with explicit `sys.argv` and shared builtin state.
    pub fn with_sys_args_and_builtins(args: Vec<String>, builtins: BuiltinRegistry) -> Self {
        Self::with_stdlib_and_paths(
            StdlibRegistry::with_sys_args_and_builtins(args, builtins),
            Vec::new(),
        )
    }

    /// Create a resolver with custom search paths.
    pub fn with_paths(paths: Vec<Arc<str>>) -> Self {
        Self::with_stdlib_and_paths(StdlibRegistry::new(), paths)
    }

    fn with_stdlib_and_paths(stdlib: StdlibRegistry, paths: Vec<Arc<str>>) -> Self {
        Self {
            sys_modules: Arc::new(RwLock::new(FxHashMap::default())),
            sys_modules_value: allocate_sys_modules_dict(),
            stdlib: Arc::new(stdlib),
            search_paths: Arc::new(RwLock::new(paths)),
            frozen_modules: Arc::new(RwLock::new(FxHashMap::default())),
            loading: Arc::new(RwLock::new(FxHashMap::default())),
            module_ptrs: Arc::new(RwLock::new(FxHashMap::default())),
        }
    }

    /// Import a module by name.
    ///
    /// # Resolution Order
    ///
    /// 1. Check sys.modules cache
    /// 2. Check stdlib registry
    /// 3. Search sys.path for .py files (future)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let resolver = ImportResolver::new();
    /// let math = resolver.import_module("math")?;
    /// ```
    pub fn import_module(&self, name: &str) -> Result<Arc<ModuleObject>, ImportError> {
        let key = intern(name);

        // 1. Check sys.modules cache (fast path)
        if let Some(module) = self.get_cached(name) {
            return Ok(module);
        }

        // 2. Atomically check if module is being imported OR insert our ImportState
        //    This prevents the TOCTOU race where multiple threads all think they're first
        let (should_wait, wait_state) = {
            let mut loading = self.loading.write().unwrap();

            if let Some(state) = loading.get(&key) {
                if state.is_circular() {
                    // Same thread re-entering - true circular import
                    return Err(ImportError::CircularImport {
                        module: Arc::from(name),
                    });
                }
                // Different thread - we need to wait
                (true, Some(Arc::clone(state)))
            } else {
                // No one is importing this module yet - we're first
                let import_state = Arc::new(ImportState::new());
                loading.insert(key.clone(), Arc::clone(&import_state));
                (false, Some(import_state))
            }
        };

        // 3. If another thread is importing, wait for it
        if should_wait {
            let state = wait_state.unwrap();
            state.wait();

            // After waiting, the module should be cached - retrieve it
            if let Some(module) = self.get_cached(name) {
                return Ok(module);
            } else {
                // Import failed in the other thread
                return Err(ImportError::ModuleNotFound {
                    module: Arc::from(name),
                });
            }
        }

        // 4. We are the loading thread - perform the actual import
        let import_state = wait_state.unwrap();

        // 5. Try stdlib registry
        let result = if let Some(stdlib_module) = self.stdlib.get(name) {
            // Create ModuleObject from stdlib module
            self.load_stdlib_module(name, stdlib_module)
        } else {
            // 6. Try file system (not yet implemented)
            Err(ImportError::ModuleNotFound {
                module: Arc::from(name),
            })
        };

        // 7. Cache successful imports BEFORE signaling waiters
        if let Ok(ref module) = result {
            self.cache_module(name, module);
        }

        // 8. Signal waiting threads that import is complete
        import_state.signal_complete();

        // 9. Remove from loading set
        self.loading.write().unwrap().remove(&key);

        result
    }

    /// Resolve the canonical load plan for `name`.
    ///
    /// The VM executes non-native plans because source and frozen modules need
    /// compiler and frame-stack access. Keeping the tier ordering here prevents
    /// source-first policy, native fallback, and cache behavior from diverging
    /// between import call sites.
    pub fn resolve_load_plan(&self, name: &str) -> ImportLoadPlan {
        if let Some(module) = self.get_cached(name) {
            return ImportLoadPlan::Cached(module);
        }

        if let Some(module) = self.get_frozen_module(name) {
            return ImportLoadPlan::Frozen(module);
        }

        let source_location = self.resolve_source_location(name);
        if self.stdlib.prefers_source_when_available(name)
            && let Some(location) = source_location.clone()
        {
            return ImportLoadPlan::Source(location);
        }

        if self.stdlib.contains(name) {
            return ImportLoadPlan::Native;
        }

        match source_location {
            Some(location) => ImportLoadPlan::Source(location),
            None => ImportLoadPlan::Missing,
        }
    }

    /// Import a module using a dotted name (e.g., `os.path`).
    ///
    /// This resolves each component of the dotted name in sequence:
    /// 1. Import the top-level module (`os`)
    /// 2. For each subsequent part, either:
    ///    a. Check sys.modules for the full dotted name
    ///    b. Get the attribute from the parent module
    ///    c. If the attribute is a submodule, cache it as a full dotted name
    ///
    /// Returns the final module in the chain.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let resolver = ImportResolver::new();
    /// let os_path = resolver.import_dotted("os.path")?;
    /// ```
    pub fn import_dotted(&self, name: &str) -> Result<Arc<ModuleObject>, ImportError> {
        use super::package::DottedName;

        // Fast path: check cache first
        if let Some(module) = self.get_cached(name) {
            return Ok(module);
        }

        // Parse the dotted name
        let dotted = DottedName::parse(name).ok_or_else(|| ImportError::ModuleNotFound {
            module: Arc::from(name),
        })?;

        // Simple name — delegate to import_module
        if dotted.is_simple() {
            return self.import_module(name);
        }

        // Import the top-level module first
        let mut current = self.import_module(dotted.top_level())?;

        // Resolve each subsequent component
        for depth in 2..=dotted.depth() {
            let full_name = dotted.name_at_depth(depth);
            let part = &dotted.parts()[depth - 1];

            // Check if this dotted name is already cached
            if let Some(module) = self.get_cached(&full_name) {
                current = module;
                continue;
            }

            // Try to import it as a stdlib submodule (e.g., "os.path")
            if let Some(stdlib_module) = self.stdlib.get(&full_name) {
                let module = self.load_stdlib_module(&full_name, stdlib_module)?;
                self.cache_module(&full_name, &module);
                // Also set as attribute on parent
                // (deferred — the parent module should already expose it)
                current.set_attr(part, Value::object_ptr(Arc::as_ptr(&module) as *const ()));
                current = module;
                continue;
            }

            // Try to get as attribute from parent module
            if let Some(value) = current.get_attr(part)
                && let Some(module_ptr) = value.as_object_ptr()
                && let Some(module) = self.module_from_ptr(module_ptr)
            {
                self.cache_module(&full_name, &module);
                current = module;
                continue;
            }

            return Err(ImportError::ModuleNotFound {
                module: Arc::from(full_name.as_str()),
            });
        }

        Ok(current)
    }

    /// Import a module using a relative import specification.
    ///
    /// # Parameters
    ///
    /// - `name`: The name to import (after the dots)
    /// - `level`: Number of leading dots (1 = current package, 2 = parent, etc.)
    /// - `package`: The `__package__` attribute of the importing module
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // from . import foo  (in package "mypackage")
    /// resolver.import_relative("foo", 1, "mypackage")?;
    ///
    /// // from ..bar import baz  (in package "a.b.c")
    /// resolver.import_relative("bar", 2, "a.b.c")?;
    /// ```
    pub fn import_relative(
        &self,
        name: &str,
        level: u32,
        package: &str,
    ) -> Result<Arc<ModuleObject>, ImportError> {
        let absolute_name = super::package::resolve_relative_import(name, level, package)?;
        self.import_dotted(&absolute_name)
    }

    /// Load a stdlib module into a ModuleObject.
    fn load_stdlib_module(
        &self,
        name: &str,
        stdlib: &(dyn Module + Send + Sync),
    ) -> Result<Arc<ModuleObject>, ImportError> {
        let module = ModuleObject::new(name);

        // Get all attributes from the stdlib module
        for attr_name in stdlib.dir() {
            match stdlib.get_attr(&attr_name) {
                Ok(value) => {
                    module.set_attr(&attr_name, value);
                }
                Err(ModuleError::AttributeError(_)) => {
                    // Skip attributes that error (shouldn't happen, but be defensive)
                    continue;
                }
                Err(e) => {
                    return Err(ImportError::LoadError {
                        module: Arc::from(name),
                        message: Arc::from(e.to_string()),
                    });
                }
            }
        }

        if name == "sys" {
            module.set_attr("modules", self.sys_modules_value);
        }

        Ok(Arc::new(module))
    }

    /// Import a specific attribute from a module.
    ///
    /// This is used for `from module import name`.
    pub fn import_from(
        &self,
        module: &Arc<ModuleObject>,
        name: &str,
    ) -> Result<Value, ImportError> {
        if let Some(value) = module.get_attr(name) {
            return Ok(value);
        }

        let submodule_name = format!("{}.{}", module.name(), name);
        if let Ok(submodule) = self.import_dotted(&submodule_name) {
            let value = Value::object_ptr(Arc::as_ptr(&submodule) as *const ());
            module.set_attr(name, value);
            return Ok(value);
        }

        Err(ImportError::ImportFromError {
            module: Arc::from(module.name()),
            name: Arc::from(name),
        })
    }

    /// Import all public names from a module.
    ///
    /// This is used for `from module import *`.
    ///
    /// Returns a list of (name, value) pairs to be inserted into the
    /// importing module's namespace.
    pub fn import_star(
        &self,
        module: &Arc<ModuleObject>,
    ) -> Result<Vec<(InternedString, Value)>, ImportError> {
        module
            .public_attrs()
            .map_err(|err| ImportError::StarImportError {
                module: Arc::from(module.name()),
                message: Arc::from(err.to_string()),
            })
    }

    #[inline]
    fn public_sys_modules(&self) -> &DictObject {
        let ptr = self
            .sys_modules_value
            .as_object_ptr()
            .expect("sys.modules should always be a dict object");
        unsafe { &*(ptr as *const DictObject) }
    }

    #[inline]
    fn public_sys_modules_mut(&self) -> &mut DictObject {
        let ptr = self
            .sys_modules_value
            .as_object_ptr()
            .expect("sys.modules should always be a dict object");
        unsafe { &mut *(ptr as *mut DictObject) }
    }

    fn sync_public_sys_modules_entry(&self, name: &str, module: &Arc<ModuleObject>) {
        self.public_sys_modules_mut().set(
            Value::string(intern(name)),
            Value::object_ptr(Arc::as_ptr(module) as *const ()),
        );
    }

    fn lookup_public_sys_modules(&self, name: &str) -> Option<Arc<ModuleObject>> {
        let value = self.public_sys_modules().get(Value::string(intern(name)))?;
        let module_ptr = value.as_object_ptr()?;
        let module = self.module_from_ptr(module_ptr)?;
        self.sys_modules
            .write()
            .unwrap()
            .insert(intern(name), Arc::clone(&module));
        Some(module)
    }

    fn cache_module(&self, name: &str, module: &Arc<ModuleObject>) {
        self.sys_modules
            .write()
            .unwrap()
            .insert(intern(name), Arc::clone(module));
        self.register_module_ptr(module);
        self.sync_public_sys_modules_entry(name, module);
    }

    /// Get a module from sys.modules cache.
    ///
    /// Returns `None` if the module hasn't been imported yet.
    pub fn get_cached(&self, name: &str) -> Option<Arc<ModuleObject>> {
        let key = intern(name);
        if let Some(module) = self.sys_modules.read().unwrap().get(&key) {
            return Some(Arc::clone(module));
        }

        self.lookup_public_sys_modules(name)
    }

    /// Insert a module directly into sys.modules.
    ///
    /// This is useful for injecting modules programmatically.
    pub fn insert_module(&self, name: &str, module: Arc<ModuleObject>) {
        self.cache_module(name, &module);
    }

    /// Remove a module from sys.modules.
    ///
    /// Returns the module if it was cached, `None` otherwise.
    pub fn remove_module(&self, name: &str) -> Option<Arc<ModuleObject>> {
        let key = intern(name);
        let removed = self.sys_modules.write().unwrap().remove(&key);
        self.public_sys_modules_mut()
            .remove(Value::string(intern(name)));
        if let Some(ref module) = removed {
            self.unregister_module_ptr(module);
        }
        removed
    }

    /// Resolve a raw object pointer to an imported module, if it is a known module.
    pub fn module_from_ptr(&self, ptr: *const ()) -> Option<Arc<ModuleObject>> {
        self.module_ptrs
            .read()
            .unwrap()
            .get(&(ptr as usize))
            .cloned()
    }

    /// Add a search path for source files.
    pub fn add_search_path(&self, path: Arc<str>) {
        self.search_paths.write().unwrap().push(path);
    }

    /// Install a frozen source module that can be executed without filesystem access.
    pub fn insert_frozen_module(&self, name: &str, module: FrozenModuleSource) {
        let key = intern(name);
        self.frozen_modules
            .write()
            .unwrap()
            .insert(key, Arc::new(module));
    }

    /// Resolve a frozen source module by canonical name.
    pub fn get_frozen_module(&self, name: &str) -> Option<Arc<FrozenModuleSource>> {
        let key = intern(name);
        self.frozen_modules.read().unwrap().get(&key).cloned()
    }

    /// Remove a frozen source module.
    pub fn remove_frozen_module(&self, name: &str) -> Option<Arc<FrozenModuleSource>> {
        let key = intern(name);
        self.frozen_modules.write().unwrap().remove(&key)
    }

    /// Check whether a frozen source module is installed.
    pub fn has_frozen_module(&self, name: &str) -> bool {
        let key = intern(name);
        self.frozen_modules.read().unwrap().contains_key(&key)
    }

    /// Get current search paths.
    pub fn search_paths(&self) -> Vec<Arc<str>> {
        let mut paths = self.public_sys_path_entries();
        for path in self.search_paths.read().unwrap().iter() {
            if !paths
                .iter()
                .any(|existing| existing.as_ref() == path.as_ref())
            {
                paths.push(Arc::clone(path));
            }
        }
        paths
    }

    fn public_sys_path_entries(&self) -> Vec<Arc<str>> {
        let sys_module = self
            .sys_modules
            .read()
            .unwrap()
            .get(&intern("sys"))
            .cloned();
        let Some(sys_module) = sys_module else {
            return Vec::new();
        };
        let Some(path_value) = sys_module.get_attr("path") else {
            return Vec::new();
        };
        let Some(path_list) = value_as_list_ref(path_value) else {
            return Vec::new();
        };

        path_list
            .iter()
            .filter_map(|value| value_as_string_ref(*value))
            .map(|path| Arc::<str>::from(path.as_str()))
            .collect()
    }

    /// Resolve a module name to a filesystem source location, if available.
    pub fn resolve_source_location(&self, name: &str) -> Option<SourceModuleLocation> {
        use super::package::{DottedName, find_dotted_module_source, find_module_source};

        let search_paths = self.search_paths();
        let (path, is_package) = if let Some(dotted) = DottedName::parse(name) {
            if dotted.is_simple() {
                find_module_source(name, &search_paths)
            } else {
                find_dotted_module_source(&dotted, &search_paths)
            }
        } else {
            None
        }?;

        Some(SourceModuleLocation { path, is_package })
    }

    /// Returns true when Prism has a native fallback for `name`, but the active
    /// search path also provides a source module/package that should be loaded
    /// first for better compatibility.
    pub fn should_load_from_source_first(&self, name: &str) -> bool {
        self.stdlib.prefers_source_when_available(name)
            && self.resolve_source_location(name).is_some()
    }

    /// Check whether a module name is provided by Prism's built-in stdlib.
    #[inline]
    pub fn is_stdlib_module(&self, name: &str) -> bool {
        self.stdlib.contains(name)
    }

    /// List all cached modules.
    pub fn cached_modules(&self) -> Vec<Arc<str>> {
        self.sys_modules
            .read()
            .unwrap()
            .keys()
            .map(|k| Arc::from(k.as_ref()))
            .collect()
    }

    /// Check if a module is available (cached or in stdlib).
    pub fn module_exists(&self, name: &str) -> bool {
        let key = intern(name);
        self.sys_modules.read().unwrap().contains_key(&key)
            || self.frozen_modules.read().unwrap().contains_key(&key)
            || self.stdlib.contains(name)
    }

    fn register_module_ptr(&self, module: &Arc<ModuleObject>) {
        self.module_ptrs
            .write()
            .unwrap()
            .insert(Arc::as_ptr(module) as usize, Arc::clone(module));
    }

    fn unregister_module_ptr(&self, module: &Arc<ModuleObject>) {
        self.module_ptrs
            .write()
            .unwrap()
            .remove(&(Arc::as_ptr(module) as usize));
    }
}

#[inline]
fn allocate_sys_modules_dict() -> Value {
    prism_runtime::allocation_context::alloc_value_in_current_heap_or_box(DictObject::new())
}

impl Default for ImportResolver {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
