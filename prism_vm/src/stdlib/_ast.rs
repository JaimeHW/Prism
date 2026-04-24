//! Native `_ast` bootstrap module.
//!
//! CPython's pure-Python `ast` module imports `_ast` very early, and higher
//! layers such as `inspect`, `dataclasses`, `pprint`, and `unittest` depend on
//! that import chain during interpreter startup. Prism provides a native
//! compatibility surface here so those bootstrap imports can succeed without
//! waiting on the full Python-visible AST object model.

use super::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::{SubclassBitmap, global_class, register_global_class};
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const PYCF_ONLY_AST: i64 = 0x0400;
const PYCF_TYPE_COMMENTS: i64 = 0x1000;

static AST_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| build_root_class("AST"));
static MOD_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| build_ast_subclass("mod"));
static EXPR_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| build_ast_subclass("expr"));
static EXPR_CONTEXT_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ast_subclass("expr_context"));
static OPERATOR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ast_subclass("operator"));
static CONSTANT_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ast_subclass_with_fields("Constant", &["value", "kind"]));
static TUPLE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ast_subclass_with_fields("Tuple", &["elts", "ctx"]));
static NAME_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_expr_subclass_with_fields("Name", &["id", "ctx"]));
static ATTRIBUTE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_expr_subclass_with_fields("Attribute", &["value", "attr", "ctx"]));
static LOAD_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_expr_context_subclass("Load"));
static ADD_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| build_operator_subclass("Add"));
static SUB_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| build_operator_subclass("Sub"));
static BITOR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_operator_subclass("BitOr"));
static MODULE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_mod_subclass_with_fields("Module", &["body", "type_ignores"]));
static EXPRESSION_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_mod_subclass_with_fields("Expression", &["body"]));
static ARG_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ast_subclass_with_fields("arg", &["arg", "annotation", "type_comment"]));
static ALL_VALUE: LazyLock<Value> = LazyLock::new(export_names_value);

const EXPORTED_NAMES: &[&str] = &[
    "AST",
    "mod",
    "expr",
    "expr_context",
    "operator",
    "Constant",
    "Tuple",
    "Name",
    "Attribute",
    "Load",
    "Add",
    "Sub",
    "BitOr",
    "Module",
    "Expression",
    "arg",
    "PyCF_ONLY_AST",
    "PyCF_TYPE_COMMENTS",
];

/// Native `_ast` module descriptor.
pub struct AstModule {
    attrs: Vec<Arc<str>>,
}

impl AstModule {
    /// Create a new `_ast` module descriptor.
    pub fn new() -> Self {
        let mut attrs = Vec::with_capacity(EXPORTED_NAMES.len() + 1);
        attrs.push(Arc::from("__all__"));
        attrs.extend(EXPORTED_NAMES.iter().copied().map(Arc::from));
        Self { attrs }
    }
}

impl Default for AstModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for AstModule {
    fn name(&self) -> &str {
        "_ast"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(*ALL_VALUE),
            "AST" => Ok(class_value(&AST_CLASS)),
            "mod" => Ok(class_value(&MOD_CLASS)),
            "expr" => Ok(class_value(&EXPR_CLASS)),
            "expr_context" => Ok(class_value(&EXPR_CONTEXT_CLASS)),
            "operator" => Ok(class_value(&OPERATOR_CLASS)),
            "Constant" => Ok(class_value(&CONSTANT_CLASS)),
            "Tuple" => Ok(class_value(&TUPLE_CLASS)),
            "Name" => Ok(class_value(&NAME_CLASS)),
            "Attribute" => Ok(class_value(&ATTRIBUTE_CLASS)),
            "Load" => Ok(class_value(&LOAD_CLASS)),
            "Add" => Ok(class_value(&ADD_CLASS)),
            "Sub" => Ok(class_value(&SUB_CLASS)),
            "BitOr" => Ok(class_value(&BITOR_CLASS)),
            "Module" => Ok(class_value(&MODULE_CLASS)),
            "Expression" => Ok(class_value(&EXPRESSION_CLASS)),
            "arg" => Ok(class_value(&ARG_CLASS)),
            "PyCF_ONLY_AST" => Ok(Value::int_unchecked(PYCF_ONLY_AST)),
            "PyCF_TYPE_COMMENTS" => Ok(Value::int_unchecked(PYCF_TYPE_COMMENTS)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_ast' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[inline]
fn class_value(class: &Arc<PyClassObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

fn build_root_class(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("_ast")));
    class.set_attr(intern("_fields"), empty_tuple_value());
    class.set_attr(intern("_attributes"), empty_tuple_value());
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);
    register_native_type(class)
}

fn build_ast_subclass(name: &str) -> Arc<PyClassObject> {
    build_ast_subclass_with_fields(name, &[])
}

fn build_ast_subclass_with_fields(name: &str, fields: &[&str]) -> Arc<PyClassObject> {
    build_subclass(name, &AST_CLASS, fields)
}

fn build_mod_subclass_with_fields(name: &str, fields: &[&str]) -> Arc<PyClassObject> {
    build_subclass(name, &MOD_CLASS, fields)
}

fn build_expr_subclass_with_fields(name: &str, fields: &[&str]) -> Arc<PyClassObject> {
    build_subclass(name, &EXPR_CLASS, fields)
}

fn build_expr_context_subclass(name: &str) -> Arc<PyClassObject> {
    build_subclass(name, &EXPR_CONTEXT_CLASS, &[])
}

fn build_operator_subclass(name: &str) -> Arc<PyClassObject> {
    build_subclass(name, &OPERATOR_CLASS, &[])
}

fn build_subclass(name: &str, base: &Arc<PyClassObject>, fields: &[&str]) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new(intern(name), &[base.class_id()], |id| {
        global_class(id).map(|class| class.mro().iter().copied().collect())
    })
    .unwrap_or_else(|err| panic!("failed to build _ast.{name}: {err}"));
    class.set_attr(intern("__module__"), Value::string(intern("_ast")));
    class.set_attr(intern("_fields"), tuple_of_names(fields));
    class.set_attr(intern("_attributes"), empty_tuple_value());
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);
    register_native_type(class)
}

fn register_native_type(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(type_id_for_class_id(class_id));
    }

    let class = Arc::new(class);
    register_global_class(class.clone(), bitmap);
    class
}

#[inline]
fn type_id_for_class_id(class_id: ClassId) -> prism_runtime::object::type_obj::TypeId {
    if class_id == ClassId::OBJECT {
        prism_runtime::object::type_obj::TypeId::OBJECT
    } else {
        prism_runtime::object::type_obj::TypeId::from_raw(class_id.0)
    }
}

fn export_names_value() -> Value {
    tuple_of_names(EXPORTED_NAMES)
}

fn tuple_of_names(names: &[&str]) -> Value {
    let values: Vec<Value> = names
        .iter()
        .map(|name| Value::string(intern(name)))
        .collect();
    leak_object_value(TupleObject::from_slice(&values))
}

fn empty_tuple_value() -> Value {
    leak_object_value(TupleObject::from_slice(&[]))
}

#[inline]
fn leak_object_value<T>(object: T) -> Value {
    let ptr = Box::leak(Box::new(object)) as *mut T as *const ();
    Value::object_ptr(ptr)
}

#[cfg(test)]
mod tests {
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
}
