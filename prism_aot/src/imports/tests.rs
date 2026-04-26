use super::*;

fn collect(source: &str, package: &str) -> StaticImports {
    let module = prism_parser::parse(source).expect("parse should succeed");
    collect_static_imports(&module, package).expect("import collection should succeed")
}

#[test]
fn test_collects_top_level_imports() {
    let imports = collect("import math\nimport pkg.helper\n", "");
    assert_eq!(
        imports.required_modules,
        vec!["math".to_string(), "pkg.helper".to_string()]
    );
    assert!(imports.from_import_candidates.is_empty());
}

#[test]
fn test_collects_nested_scope_imports() {
    let imports = collect(
        "def outer():\n    if True:\n        import helper\nclass C:\n    import math\n",
        "",
    );
    assert_eq!(
        imports.required_modules,
        vec!["helper".to_string(), "math".to_string()]
    );
    assert!(imports.from_import_candidates.is_empty());
}

#[test]
fn test_collects_relative_from_imports() {
    let imports = collect(
        "from .helper import VALUE\nfrom . import submodule\n",
        "pkg",
    );
    assert_eq!(
        imports.required_modules,
        vec!["pkg".to_string(), "pkg.helper".to_string(),]
    );
    assert_eq!(
        imports.from_import_candidates,
        vec!["pkg.helper.VALUE".to_string(), "pkg.submodule".to_string(),]
    );
}

#[test]
fn test_collects_parent_relative_imports() {
    let imports = collect("from ..core import api\n", "pkg.subpkg");
    assert_eq!(imports.required_modules, vec!["pkg.core".to_string()]);
    assert_eq!(
        imports.from_import_candidates,
        vec!["pkg.core.api".to_string()]
    );
}

#[test]
fn test_star_imports_do_not_emit_candidates() {
    let imports = collect("from pkg import *\n", "");
    assert_eq!(imports.required_modules, vec!["pkg".to_string()]);
    assert!(imports.from_import_candidates.is_empty());
}
