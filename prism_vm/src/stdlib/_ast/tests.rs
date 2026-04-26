use super::*;
use crate::import::ImportResolver;
use prism_runtime::types::tuple::TupleObject;

#[test]
fn test_ast_module_exports_bootstrap_surface() {
    let module = AstModule::new();

    for name in EXPORTED_NAMES {
        assert!(
            module.get_attr(name).is_ok(),
            "_ast should expose bootstrap attribute {name}"
        );
    }
}

#[test]
fn test_ast_module_all_lists_public_exports() {
    let module = AstModule::new();
    let all_value = module
        .get_attr("__all__")
        .expect("_ast.__all__ should be present");
    let tuple_ptr = all_value
        .as_object_ptr()
        .expect("_ast.__all__ should be represented as a tuple object");
    let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
    assert_eq!(tuple.len(), EXPORTED_NAMES.len());
}

#[test]
fn test_import_stdlib_ast_bootstrap_module() {
    let resolver = ImportResolver::new();
    let module = resolver
        .import_module("_ast")
        .expect("_ast import should succeed");

    assert_eq!(module.name(), "_ast");
    assert!(module.get_attr("AST").is_some());
    assert!(module.get_attr("Constant").is_some());
    assert_eq!(
        module
            .get_attr("PyCF_ONLY_AST")
            .expect("PyCF_ONLY_AST should be present")
            .as_int()
            .expect("PyCF_ONLY_AST should be an integer"),
        PYCF_ONLY_AST
    );
}
