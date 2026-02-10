//! `ModuleObject` - runtime representation of a Python module.
//!
//! Provides an efficient, cache-friendly representation of imported modules
//! with O(1) attribute access via interned string keys.

use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

// =============================================================================
// ModuleObject
// =============================================================================

/// A Python module object with attribute storage.
///
/// This is the runtime representation of an imported module. It stores:
/// - Module metadata (`__name__`, `__doc__`, `__file__`, etc.)
/// - Module attributes (functions, classes, constants)
///
/// # Performance
///
/// - Attribute lookup is O(1) via `FxHashMap` with interned string keys
/// - Thread-safe via `RwLock` for concurrent read access
/// - Attributes stored as `Value` for zero-copy access
#[derive(Debug)]
pub struct ModuleObject {
    /// Module name (e.g., "math", "os.path")
    name: Arc<str>,

    /// Module attributes (__name__, __doc__, functions, etc.)
    /// Uses RwLock for concurrent read access (common case)
    attrs: RwLock<FxHashMap<InternedString, Value>>,

    /// Module documentation string
    doc: Option<Arc<str>>,

    /// Module file path (None for built-in modules)
    file: Option<Arc<str>>,

    /// Package name (e.g., "os" for "os.path")
    package: Option<Arc<str>>,
}

impl ModuleObject {
    /// Create a new empty module with the given name.
    #[inline]
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        let name = name.into();
        let mut attrs = FxHashMap::default();

        // Set __name__ attribute
        let name_key = intern("__name__");
        attrs.insert(name_key, Value::string(intern(&name)));

        Self {
            name,
            attrs: RwLock::new(attrs),
            doc: None,
            file: None,
            package: None,
        }
    }

    /// Create a new module with optional metadata.
    pub fn with_metadata(
        name: impl Into<Arc<str>>,
        doc: Option<Arc<str>>,
        file: Option<Arc<str>>,
        package: Option<Arc<str>>,
    ) -> Self {
        let name = name.into();
        let mut attrs = FxHashMap::default();

        // Set __name__
        attrs.insert(intern("__name__"), Value::string(intern(&name)));

        // Set __doc__ if provided
        if let Some(ref doc_str) = doc {
            attrs.insert(intern("__doc__"), Value::string(intern(doc_str)));
        }

        // Set __file__ if provided
        if let Some(ref file_str) = file {
            attrs.insert(intern("__file__"), Value::string(intern(file_str)));
        }

        // Set __package__ if provided
        if let Some(ref pkg_str) = package {
            attrs.insert(intern("__package__"), Value::string(intern(pkg_str)));
        }

        Self {
            name,
            attrs: RwLock::new(attrs),
            doc,
            file,
            package,
        }
    }

    /// Get the module name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get an attribute from the module.
    ///
    /// Returns `None` if the attribute doesn't exist.
    #[inline]
    pub fn get_attr(&self, name: &str) -> Option<Value> {
        let key = intern(name);
        self.attrs.read().unwrap().get(&key).copied()
    }

    /// Set an attribute on the module.
    #[inline]
    pub fn set_attr(&self, name: &str, value: Value) {
        let key = intern(name);
        self.attrs.write().unwrap().insert(key, value);
    }

    /// Check if the module has an attribute.
    #[inline]
    pub fn has_attr(&self, name: &str) -> bool {
        let key = intern(name);
        self.attrs.read().unwrap().contains_key(&key)
    }

    /// Delete an attribute from the module.
    ///
    /// Returns `true` if the attribute existed and was removed.
    #[inline]
    pub fn del_attr(&self, name: &str) -> bool {
        let key = intern(name);
        self.attrs.write().unwrap().remove(&key).is_some()
    }

    /// Get all attribute names.
    ///
    /// This is used for `dir(module)` and `from module import *`.
    pub fn dir(&self) -> Vec<InternedString> {
        self.attrs.read().unwrap().keys().cloned().collect()
    }

    /// Get all public attribute names (for `import *`).
    ///
    /// If `__all__` is defined, returns those names.
    /// Otherwise, returns all names not starting with underscore.
    pub fn public_names(&self) -> Vec<InternedString> {
        let attrs = self.attrs.read().unwrap();

        // Check for __all__
        let all_key = intern("__all__");
        if let Some(all_val) = attrs.get(&all_key) {
            // TODO: Extract list of strings from __all__
            // For now, fall through to default behavior
            let _ = all_val;
        }

        // Return all public names (not starting with _)
        attrs
            .keys()
            .filter(|k| !k.as_ref().starts_with('_'))
            .cloned()
            .collect()
    }

    /// Get all attributes as (name, value) pairs.
    ///
    /// This is used for `import *` to inject names into the importing scope.
    pub fn all_attrs(&self) -> Vec<(InternedString, Value)> {
        self.attrs
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Get public attributes as (name, value) pairs.
    pub fn public_attrs(&self) -> Vec<(InternedString, Value)> {
        let attrs = self.attrs.read().unwrap();

        // Check for __all__
        let all_key = intern("__all__");
        if let Some(_all_val) = attrs.get(&all_key) {
            // TODO: Extract list of strings from __all__ and return those
            // For now, fall through to default behavior
        }

        // Return all public attrs
        attrs
            .iter()
            .filter(|(k, _)| !k.as_ref().starts_with('_'))
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Get the number of attributes.
    #[inline]
    pub fn len(&self) -> usize {
        self.attrs.read().unwrap().len()
    }

    /// Check if the module has no attributes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.attrs.read().unwrap().is_empty()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_new() {
        let module = ModuleObject::new("test_module");
        assert_eq!(module.name(), "test_module");
        assert!(module.has_attr("__name__"));
    }

    #[test]
    fn test_module_get_set_attr() {
        let module = ModuleObject::new("test");
        module.set_attr("foo", Value::int(42).unwrap());
        assert!(module.has_attr("foo"));
        let val = module.get_attr("foo").unwrap();
        assert_eq!(val.as_int(), Some(42));
    }

    #[test]
    fn test_module_del_attr() {
        let module = ModuleObject::new("test");
        module.set_attr("bar", Value::int(100).unwrap());
        assert!(module.has_attr("bar"));
        assert!(module.del_attr("bar"));
        assert!(!module.has_attr("bar"));
    }

    #[test]
    fn test_module_del_nonexistent() {
        let module = ModuleObject::new("test");
        assert!(!module.del_attr("nonexistent"));
    }

    #[test]
    fn test_module_dir() {
        let module = ModuleObject::new("test");
        module.set_attr("alpha", Value::int(1).unwrap());
        module.set_attr("beta", Value::int(2).unwrap());
        let names = module.dir();
        // Should have __name__, alpha, beta
        assert!(names.len() >= 3);
    }

    #[test]
    fn test_module_public_names() {
        let module = ModuleObject::new("test");
        module.set_attr("public", Value::int(1).unwrap());
        module.set_attr("_private", Value::int(2).unwrap());
        module.set_attr("__dunder__", Value::int(3).unwrap());

        let public = module.public_names();
        // Should contain "public" but not "_private" or "__dunder__"
        let public_strs: Vec<&str> = public.iter().map(|s| s.as_ref()).collect();
        assert!(public_strs.contains(&"public"));
        assert!(!public_strs.contains(&"_private"));
        assert!(!public_strs.contains(&"__dunder__"));
    }

    #[test]
    fn test_module_with_metadata() {
        let module = ModuleObject::with_metadata(
            "mymodule",
            Some(Arc::from("Module documentation")),
            Some(Arc::from("/path/to/module.py")),
            Some(Arc::from("mypackage")),
        );

        assert_eq!(module.name(), "mymodule");
        assert!(module.has_attr("__name__"));
        assert!(module.has_attr("__doc__"));
        assert!(module.has_attr("__file__"));
        assert!(module.has_attr("__package__"));
    }

    #[test]
    fn test_module_len_and_is_empty() {
        let module = ModuleObject::new("test");
        // Has at least __name__
        assert!(!module.is_empty());
        assert!(module.len() >= 1);
    }

    #[test]
    fn test_module_all_attrs() {
        let module = ModuleObject::new("test");
        module.set_attr("x", Value::int(10).unwrap());
        module.set_attr("y", Value::int(20).unwrap());

        let attrs = module.all_attrs();
        assert!(attrs.len() >= 3); // __name__, x, y
    }

    #[test]
    fn test_module_public_attrs() {
        let module = ModuleObject::new("test");
        module.set_attr("public_var", Value::int(1).unwrap());
        module.set_attr("_hidden", Value::int(2).unwrap());

        let public = module.public_attrs();
        let names: Vec<&str> = public.iter().map(|(k, _)| k.as_ref()).collect();
        assert!(names.contains(&"public_var"));
        assert!(!names.contains(&"_hidden"));
    }

    #[test]
    fn test_module_concurrent_access() {
        use std::thread;

        let module = Arc::new(ModuleObject::new("concurrent"));

        // Spawn multiple threads to read/write
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let m = Arc::clone(&module);
                thread::spawn(move || {
                    m.set_attr(&format!("attr_{}", i), Value::int(i).unwrap());
                    m.get_attr(&format!("attr_{}", i))
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All attributes should exist
        for i in 0..10 {
            assert!(module.has_attr(&format!("attr_{}", i)));
        }
    }
}
