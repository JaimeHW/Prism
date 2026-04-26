//! `ModuleObject` - runtime representation of a Python module.
//!
//! Provides an efficient, cache-friendly representation of imported modules
//! with O(1) attribute access via interned string keys.

use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use prism_runtime::allocation_context::alloc_value_in_current_heap_or_box;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

// =============================================================================
// Module Export Errors
// =============================================================================

/// Errors raised while resolving a module's public export surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleExportError {
    /// `__all__` is present but Prism cannot treat it as a sequence of names.
    InvalidAll { module: Arc<str>, message: Arc<str> },

    /// A member of `__all__` is not a string.
    NonStringAllItem { module: Arc<str>, index: usize },

    /// `__all__` names an attribute that is absent from the module namespace.
    MissingAllAttribute { module: Arc<str>, name: Arc<str> },
}

impl std::fmt::Display for ModuleExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAll { module, message } => {
                write!(f, "module '{}' has invalid __all__: {}", module, message)
            }
            Self::NonStringAllItem { module, index } => {
                write!(
                    f,
                    "item {} in module '{}'.__all__ must be str",
                    index, module
                )
            }
            Self::MissingAllAttribute { module, name } => {
                write!(f, "module '{}' has no attribute '{}'", module, name)
            }
        }
    }
}

impl std::error::Error for ModuleExportError {}

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
#[repr(C)]
#[derive(Debug)]
pub struct ModuleObject {
    /// Object header so modules participate correctly in generic object protocol paths.
    pub header: ObjectHeader,

    /// Module name (e.g., "math", "os.path")
    name: Arc<str>,

    /// Module attributes (__name__, __doc__, functions, etc.)
    /// Uses RwLock for concurrent read access (common case)
    attrs: RwLock<FxHashMap<InternedString, Value>>,

    /// Lazily materialized live module namespace exposed as `module.__dict__`.
    dict: RwLock<Option<Value>>,

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
        let attrs = base_module_attrs(&name, None);

        Self {
            header: ObjectHeader::new(TypeId::MODULE),
            name,
            attrs: RwLock::new(attrs),
            dict: RwLock::new(None),
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
        let mut attrs = base_module_attrs(&name, doc.as_ref());

        // Set __file__ if provided
        if let Some(ref file_str) = file {
            attrs.insert(intern("__file__"), Value::string(intern(file_str)));
        }

        // Set __package__ if provided
        if let Some(ref pkg_str) = package {
            attrs.insert(intern("__package__"), Value::string(intern(pkg_str)));
        }

        if let Some(path_value) =
            package_search_path_value(name.as_ref(), file.as_ref(), package.as_ref())
        {
            attrs.insert(intern("__path__"), path_value);
        }

        Self {
            header: ObjectHeader::new(TypeId::MODULE),
            name,
            attrs: RwLock::new(attrs),
            dict: RwLock::new(None),
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

    /// Get the backing module file path, if any.
    #[inline]
    pub fn file_path(&self) -> Option<&str> {
        self.file.as_deref()
    }

    /// Get the importing package name used for relative imports.
    #[inline]
    pub fn package_name(&self) -> Option<&str> {
        self.package.as_deref()
    }

    /// Get an attribute from the module.
    ///
    /// Returns `None` if the attribute doesn't exist.
    #[inline]
    pub fn get_attr(&self, name: &str) -> Option<Value> {
        if let Some(dict) = self.materialized_dict_ref() {
            return dict.get(Value::string(intern(name)));
        }

        let key = intern(name);
        self.attrs.read().unwrap().get(&key).copied()
    }

    /// Set an attribute on the module.
    #[inline]
    pub fn set_attr(&self, name: &str, value: Value) {
        let key = intern(name);
        let dict_slot = self.dict.read().unwrap();
        if let Some(dict_value) = *dict_slot
            && let Some(ptr) = dict_value.as_object_ptr()
        {
            let dict = unsafe { &mut *(ptr as *mut DictObject) };
            dict.set(Value::string(key.clone()), value);
        }
        self.attrs.write().unwrap().insert(key, value);
    }

    /// Check if the module has an attribute.
    #[inline]
    pub fn has_attr(&self, name: &str) -> bool {
        if let Some(dict) = self.materialized_dict_ref() {
            return dict.contains_key(Value::string(intern(name)));
        }

        let key = intern(name);
        self.attrs.read().unwrap().contains_key(&key)
    }

    /// Delete an attribute from the module.
    ///
    /// Returns `true` if the attribute existed and was removed.
    #[inline]
    pub fn del_attr(&self, name: &str) -> bool {
        let key = intern(name);
        let dict_slot = self.dict.read().unwrap();
        let removed_dict = if let Some(dict_value) = *dict_slot {
            if let Some(ptr) = dict_value.as_object_ptr() {
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                dict.remove(Value::string(key.clone())).is_some()
            } else {
                false
            }
        } else {
            false
        };
        let removed_attr = self.attrs.write().unwrap().remove(&key).is_some();
        removed_attr || removed_dict
    }

    /// Get all attribute names.
    ///
    /// This is used for `dir(module)` and `from module import *`.
    pub fn dir(&self) -> Vec<InternedString> {
        if let Some(dict) = self.materialized_dict_ref() {
            return module_dict_entries(dict)
                .into_iter()
                .map(|(name, _)| name)
                .collect();
        }

        self.attrs.read().unwrap().keys().cloned().collect()
    }

    /// Get all public attribute names (for `import *`).
    ///
    /// If `__all__` is defined, returns those names.
    /// Otherwise, returns all names not starting with underscore.
    pub fn public_names(&self) -> Result<Vec<InternedString>, ModuleExportError> {
        if let Some(dict) = self.materialized_dict_ref() {
            let entries = module_dict_entries(dict);
            return public_names_from_entries(&self.name, &entries);
        }

        let attrs = self.attrs.read().unwrap();

        if let Some(all_val) = attrs.get(&intern("__all__")) {
            return all_export_names(&self.name, *all_val);
        }

        Ok(attrs
            .keys()
            .filter(|k| !k.as_ref().starts_with('_'))
            .cloned()
            .collect())
    }

    /// Get all attributes as (name, value) pairs.
    ///
    /// This is used for `import *` to inject names into the importing scope.
    pub fn all_attrs(&self) -> Vec<(InternedString, Value)> {
        if let Some(dict) = self.materialized_dict_ref() {
            return module_dict_entries(dict);
        }

        self.attrs
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Get public attributes as (name, value) pairs.
    pub fn public_attrs(&self) -> Result<Vec<(InternedString, Value)>, ModuleExportError> {
        if let Some(dict) = self.materialized_dict_ref() {
            let entries = module_dict_entries(dict);
            return public_attrs_from_entries(&self.name, &entries);
        }

        let attrs = self.attrs.read().unwrap();

        if let Some(all_val) = attrs.get(&intern("__all__")) {
            let names = all_export_names(&self.name, *all_val)?;
            let mut exported = Vec::with_capacity(names.len());
            for name in names {
                let value = attrs.get(&name).copied().ok_or_else(|| {
                    ModuleExportError::MissingAllAttribute {
                        module: Arc::clone(&self.name),
                        name: Arc::from(name.as_ref()),
                    }
                })?;
                exported.push((name, value));
            }
            return Ok(exported);
        }

        Ok(attrs
            .iter()
            .filter(|(k, _)| !k.as_ref().starts_with('_'))
            .map(|(k, v)| (k.clone(), *v))
            .collect())
    }

    /// Get the number of attributes.
    #[inline]
    pub fn len(&self) -> usize {
        if let Some(dict) = self.materialized_dict_ref() {
            return module_dict_entries(dict).len();
        }

        self.attrs.read().unwrap().len()
    }

    /// Check if the module has no attributes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        if let Some(dict) = self.materialized_dict_ref() {
            return module_dict_entries(dict).is_empty();
        }

        self.attrs.read().unwrap().is_empty()
    }

    /// Return the live dictionary used as this module's global namespace.
    pub fn dict_value(&self) -> Value {
        if let Some(value) = *self.dict.read().unwrap() {
            return value;
        }

        let mut dict_slot = self.dict.write().unwrap();
        if let Some(value) = *dict_slot {
            return value;
        }

        let attrs = self.attrs.read().unwrap();
        let mut dict = DictObject::with_capacity(attrs.len());
        for (name, value) in attrs.iter() {
            dict.set(Value::string(name.clone()), *value);
        }
        let value = alloc_value_in_current_heap_or_box(dict);
        *dict_slot = Some(value);
        value
    }

    #[inline]
    fn materialized_dict_ref(&self) -> Option<&'static DictObject> {
        let value = *self.dict.read().unwrap();
        let ptr = value?.as_object_ptr()?;
        Some(unsafe { &*(ptr as *const DictObject) })
    }
}

fn module_dict_entries(dict: &DictObject) -> Vec<(InternedString, Value)> {
    dict.iter()
        .filter_map(|(key, value)| {
            let name = value_as_string_ref(key)?;
            Some((intern(name.as_str()), value))
        })
        .collect()
}

fn public_names_from_entries(
    module_name: &Arc<str>,
    entries: &[(InternedString, Value)],
) -> Result<Vec<InternedString>, ModuleExportError> {
    if let Some((_, all_val)) = entries.iter().find(|(name, _)| name.as_ref() == "__all__") {
        return all_export_names(module_name, *all_val);
    }

    Ok(entries
        .iter()
        .map(|(name, _)| name.clone())
        .filter(|name| !name.as_ref().starts_with('_'))
        .collect())
}

fn public_attrs_from_entries(
    module_name: &Arc<str>,
    entries: &[(InternedString, Value)],
) -> Result<Vec<(InternedString, Value)>, ModuleExportError> {
    if let Some((_, all_val)) = entries.iter().find(|(name, _)| name.as_ref() == "__all__") {
        let names = all_export_names(module_name, *all_val)?;
        let mut exported = Vec::with_capacity(names.len());
        for name in names {
            let value = entries
                .iter()
                .find(|(attr_name, _)| attr_name == &name)
                .map(|(_, value)| *value)
                .ok_or_else(|| ModuleExportError::MissingAllAttribute {
                    module: Arc::clone(module_name),
                    name: Arc::from(name.as_ref()),
                })?;
            exported.push((name, value));
        }
        return Ok(exported);
    }

    Ok(entries
        .iter()
        .filter(|(name, _)| !name.as_ref().starts_with('_'))
        .map(|(name, value)| (name.clone(), *value))
        .collect())
}

fn all_export_names(
    module_name: &Arc<str>,
    all_value: Value,
) -> Result<Vec<InternedString>, ModuleExportError> {
    let Some(ptr) = all_value.as_object_ptr() else {
        return Err(ModuleExportError::InvalidAll {
            module: Arc::clone(module_name),
            message: Arc::from("__all__ must be a tuple or list of strings"),
        });
    };

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    let items = match header.type_id {
        TypeId::TUPLE => unsafe { (&*(ptr as *const TupleObject)).as_slice() },
        TypeId::LIST => unsafe { (&*(ptr as *const ListObject)).as_slice() },
        _ => {
            return Err(ModuleExportError::InvalidAll {
                module: Arc::clone(module_name),
                message: Arc::from("__all__ must be a tuple or list of strings"),
            });
        }
    };

    let mut names = Vec::with_capacity(items.len());
    for (index, item) in items.iter().copied().enumerate() {
        let name =
            value_as_string_ref(item).ok_or_else(|| ModuleExportError::NonStringAllItem {
                module: Arc::clone(module_name),
                index,
            })?;
        names.push(intern(name.as_str()));
    }
    Ok(names)
}

fn base_module_attrs(name: &Arc<str>, doc: Option<&Arc<str>>) -> FxHashMap<InternedString, Value> {
    let mut attrs = FxHashMap::default();
    attrs.insert(intern("__name__"), Value::string(intern(name)));
    attrs.insert(
        intern("__doc__"),
        doc.map(|doc_str| Value::string(intern(doc_str)))
            .unwrap_or_else(Value::none),
    );
    attrs.insert(intern("__loader__"), Value::none());
    attrs.insert(intern("__package__"), Value::none());
    attrs.insert(intern("__spec__"), Value::none());
    attrs
}

fn package_search_path_value(
    name: &str,
    file: Option<&Arc<str>>,
    package: Option<&Arc<str>>,
) -> Option<Value> {
    let package = package?;
    if package.as_ref() != name {
        return None;
    }

    let file = file?;
    let package_dir = Path::new(file.as_ref()).parent()?.to_str()?;
    let entries = [Value::string(intern(package_dir))];
    Some(alloc_value_in_current_heap_or_box(ListObject::from_slice(
        &entries,
    )))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
