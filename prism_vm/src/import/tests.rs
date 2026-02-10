//! Integration tests for the import system.

#[cfg(test)]
mod integration_tests {
    use crate::import::{ImportResolver, ModuleObject};
    use std::sync::Arc;

    #[test]
    fn test_end_to_end_import_workflow() {
        let resolver = ImportResolver::new();

        // 1. Import math module
        let math = resolver.import_module("math").unwrap();
        assert_eq!(math.name(), "math");

        // 2. Import specific attribute
        let pi = resolver.import_from(&math, "pi").unwrap();
        assert!(pi.is_float());

        // 3. Import star
        let all_attrs = resolver.import_star(&math).unwrap();
        assert!(!all_attrs.is_empty());

        // 4. Verify caching
        let math2 = resolver.import_module("math").unwrap();
        assert!(Arc::ptr_eq(&math, &math2));
    }

    #[test]
    fn test_multiple_module_imports() {
        let resolver = ImportResolver::new();

        // Import all stdlib modules
        let math = resolver.import_module("math").unwrap();
        let os = resolver.import_module("os").unwrap();
        let sys = resolver.import_module("sys").unwrap();

        assert_eq!(math.name(), "math");
        assert_eq!(os.name(), "os");
        assert_eq!(sys.name(), "sys");

        // Verify all cached
        let cached = resolver.cached_modules();
        assert_eq!(cached.len(), 3);
    }

    #[test]
    fn test_custom_module_injection() {
        let resolver = ImportResolver::new();

        // Create a custom module
        let custom = Arc::new(ModuleObject::new("myapp.config"));
        custom.set_attr("DEBUG", prism_core::Value::bool(true));
        custom.set_attr("PORT", prism_core::Value::int(8080).unwrap());

        // Inject into resolver
        resolver.insert_module("myapp.config", Arc::clone(&custom));

        // Import it
        let imported = resolver.import_module("myapp.config").unwrap();
        assert!(Arc::ptr_eq(&imported, &custom));

        // Get attributes
        let debug = resolver.import_from(&imported, "DEBUG").unwrap();
        assert_eq!(debug.as_bool(), Some(true));

        let port = resolver.import_from(&imported, "PORT").unwrap();
        assert_eq!(port.as_int(), Some(8080));
    }

    #[test]
    fn test_module_reimport_after_remove() {
        let resolver = ImportResolver::new();

        // Import math
        let math1 = resolver.import_module("math").unwrap();
        let ptr1 = Arc::as_ptr(&math1);

        // Remove it
        resolver.remove_module("math");

        // Re-import (should create new module)
        let math2 = resolver.import_module("math").unwrap();
        let ptr2 = Arc::as_ptr(&math2);

        // Should be different module instances
        assert_ne!(ptr1, ptr2);
    }

    #[test]
    fn test_import_error_handling() {
        let resolver = ImportResolver::new();

        // Non-existent module
        let result = resolver.import_module("definitely_not_a_module");
        assert!(result.is_err());

        // Non-existent attribute
        let math = resolver.import_module("math").unwrap();
        let result = resolver.import_from(&math, "not_an_attribute");
        assert!(result.is_err());
    }

    #[test]
    fn test_import_preserves_module_attributes() {
        let resolver = ImportResolver::new();
        let math = resolver.import_module("math").unwrap();

        // Check that standard attributes are present
        assert!(math.has_attr("__name__"));

        // Check that math functions are present
        // (these come from the stdlib math module)
        assert!(math.has_attr("pi") || math.has_attr("e"));
    }
}
