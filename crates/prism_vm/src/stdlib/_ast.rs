//! Native `_ast` bootstrap module.
//!
//! CPython's pure-Python `ast` module imports `_ast` very early, and higher
//! layers such as `inspect`, `dataclasses`, `pprint`, and `unittest` depend on
//! that import chain during interpreter startup. Prism provides a native
//! compatibility surface here so those bootstrap imports can succeed without
//! waiting on the full Python-visible AST object model.

use super::{Module, ModuleError, ModuleResult};
use crate::alloc_managed_value;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, global_class, global_class_bitmap, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

/// Compiler flag that returns a parsed AST instead of bytecode.
pub const PYCF_ONLY_AST: i64 = 0x0400;
/// Compiler flag enabling type-comment parsing.
pub const PYCF_TYPE_COMMENTS: i64 = 0x1000;
/// Compiler flag allowing top-level await forms.
pub const PYCF_ALLOW_TOP_LEVEL_AWAIT: i64 = 0x2000;

const PRISM_AST_SOURCE_ATTR: &str = "__prism_source__";
const PRISM_AST_FILENAME_ATTR: &str = "__prism_filename__";
const PRISM_AST_MODE_ATTR: &str = "__prism_mode__";

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
static INTERACTIVE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_mod_subclass_with_fields("Interactive", &["body"]));
static ARG_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ast_subclass_with_fields("arg", &["arg", "annotation", "type_comment"]));
static ALL_VALUE: LazyLock<Value> = LazyLock::new(export_names_value);

/// Names exported by Prism's native `_ast` module.
pub const EXPORTED_NAMES: &[&str] = &[
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
    "Interactive",
    "arg",
    "PyCF_ONLY_AST",
    "PyCF_TYPE_COMMENTS",
    "PyCF_ALLOW_TOP_LEVEL_AWAIT",
];

/// Python parser mode represented by a root `_ast.mod` subclass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstParseMode {
    /// Module statement suite (`compile(..., "exec")`).
    Exec,
    /// Single expression (`compile(..., "eval")`).
    Eval,
    /// Interactive single-input mode (`compile(..., "single")`).
    Single,
}

impl AstParseMode {
    /// Parse a Python compile mode string.
    #[inline]
    pub fn from_str(mode: &str) -> Option<Self> {
        match mode {
            "exec" => Some(Self::Exec),
            "eval" => Some(Self::Eval),
            "single" => Some(Self::Single),
            _ => None,
        }
    }

    /// Return the canonical compile mode string.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exec => "exec",
            Self::Eval => "eval",
            Self::Single => "single",
        }
    }

    #[inline]
    fn root_class(self) -> &'static Arc<PyClassObject> {
        match self {
            Self::Exec => &MODULE_CLASS,
            Self::Eval => &EXPRESSION_CLASS,
            Self::Single => &INTERACTIVE_CLASS,
        }
    }
}

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
        exported_attr(name).ok_or_else(|| {
            ModuleError::AttributeError(format!("module '_ast' has no attribute '{}'", name))
        })
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

/// Return a public `_ast` module attribute by name.
pub fn exported_attr(name: &str) -> Option<Value> {
    match name {
        "__all__" => Some(*ALL_VALUE),
        "AST" => Some(class_value(&AST_CLASS)),
        "mod" => Some(class_value(&MOD_CLASS)),
        "expr" => Some(class_value(&EXPR_CLASS)),
        "expr_context" => Some(class_value(&EXPR_CONTEXT_CLASS)),
        "operator" => Some(class_value(&OPERATOR_CLASS)),
        "Constant" => Some(class_value(&CONSTANT_CLASS)),
        "Tuple" => Some(class_value(&TUPLE_CLASS)),
        "Name" => Some(class_value(&NAME_CLASS)),
        "Attribute" => Some(class_value(&ATTRIBUTE_CLASS)),
        "Load" => Some(class_value(&LOAD_CLASS)),
        "Add" => Some(class_value(&ADD_CLASS)),
        "Sub" => Some(class_value(&SUB_CLASS)),
        "BitOr" => Some(class_value(&BITOR_CLASS)),
        "Module" => Some(class_value(&MODULE_CLASS)),
        "Expression" => Some(class_value(&EXPRESSION_CLASS)),
        "Interactive" => Some(class_value(&INTERACTIVE_CLASS)),
        "arg" => Some(class_value(&ARG_CLASS)),
        "PyCF_ONLY_AST" => Some(Value::int_unchecked(PYCF_ONLY_AST)),
        "PyCF_TYPE_COMMENTS" => Some(Value::int_unchecked(PYCF_TYPE_COMMENTS)),
        "PyCF_ALLOW_TOP_LEVEL_AWAIT" => Some(Value::int_unchecked(PYCF_ALLOW_TOP_LEVEL_AWAIT)),
        _ => None,
    }
}

/// Build a Python-visible parsed AST root while preserving the source text.
///
/// The exposed node is a real `_ast.mod` subclass instance, so normal type
/// checks see `Module`, `Expression`, or `Interactive`. Prism also records the
/// validated source text so `compile(ast_obj, ...)` can re-enter the optimizing
/// compiler without stringly test-only paths.
pub fn parsed_ast_value(source: &str, filename: &str, mode: AstParseMode) -> Value {
    let class = mode.root_class();
    let registry = shape_registry();
    let mut object = ShapedObject::new(class.class_type_id(), Arc::clone(class.instance_shape()));

    object.set_property(
        intern(PRISM_AST_SOURCE_ATTR),
        alloc_managed_value(StringObject::new(source)),
        registry,
    );
    object.set_property(
        intern(PRISM_AST_FILENAME_ATTR),
        alloc_managed_value(StringObject::new(filename)),
        registry,
    );
    object.set_property(
        intern(PRISM_AST_MODE_ATTR),
        Value::string(intern(mode.as_str())),
        registry,
    );

    match mode {
        AstParseMode::Exec => {
            object.set_property(intern("body"), empty_list_value(), registry);
            object.set_property(intern("type_ignores"), empty_list_value(), registry);
        }
        AstParseMode::Eval => {
            object.set_property(intern("body"), Value::none(), registry);
        }
        AstParseMode::Single => {
            object.set_property(intern("body"), empty_list_value(), registry);
        }
    }

    alloc_managed_value(object)
}

/// Extract Prism's preserved source text from a parsed AST root.
pub fn parsed_ast_source(value: Value) -> Option<String> {
    let ptr = value.as_object_ptr()?;
    if !is_mod_subclass_value(value) {
        return None;
    }

    let object = unsafe { &*(ptr as *const ShapedObject) };
    let source = object.get_property(PRISM_AST_SOURCE_ATTR)?;
    value_as_string_ref(source).map(|source| source.as_str().to_string())
}

#[inline]
fn is_mod_subclass_value(value: Value) -> bool {
    let ptr = match value.as_object_ptr() {
        Some(ptr) => ptr,
        None => return false,
    };
    let type_id = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) }.type_id;
    let target = MOD_CLASS.class_type_id();
    type_id == target
        || global_class_bitmap(ClassId(type_id.raw()))
            .is_some_and(|bitmap| bitmap.is_subclass_of(target))
}

#[inline]
fn class_value(class: &Arc<PyClassObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

fn build_root_class(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    let empty_fields = empty_tuple_value();
    class.set_attr(intern("__module__"), Value::string(intern("_ast")));
    class.set_attr(intern("_fields"), empty_fields);
    class.set_attr(intern("__match_args__"), empty_fields);
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
    let fields_tuple = tuple_of_names(fields);
    class.set_attr(intern("__module__"), Value::string(intern("_ast")));
    class.set_attr(intern("_fields"), fields_tuple);
    class.set_attr(intern("__match_args__"), fields_tuple);
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
fn type_id_for_class_id(class_id: ClassId) -> TypeId {
    if class_id == ClassId::OBJECT {
        TypeId::OBJECT
    } else {
        TypeId::from_raw(class_id.0)
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

fn empty_list_value() -> Value {
    alloc_managed_value(ListObject::new())
}

#[inline]
fn leak_object_value<T>(object: T) -> Value {
    let ptr = Box::leak(Box::new(object)) as *mut T as *const ();
    Value::object_ptr(ptr)
}
