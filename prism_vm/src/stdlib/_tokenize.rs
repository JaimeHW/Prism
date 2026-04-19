//! Native `_tokenize` bootstrap module.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use std::sync::{Arc, LazyLock};

static TOKENIZER_ITER_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_tokenize.TokenizerIter"), builtin_tokenizer_iter)
});
static TOKENIZER_ITER_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(build_tokenizer_iter_class);

pub struct TokenizeModule {
    attrs: Vec<Arc<str>>,
}

impl TokenizeModule {
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("TokenizerIter")],
        }
    }
}

impl Default for TokenizeModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TokenizeModule {
    fn name(&self) -> &str {
        "_tokenize"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "TokenizerIter" => Ok(tokenizer_iter_class_value()),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_tokenize' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

fn tokenizer_iter_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&TOKENIZER_ITER_CLASS) as *const ())
}

fn build_tokenizer_iter_class() -> Arc<PyClassObject> {
    let class = Arc::new(PyClassObject::new_simple(intern("TokenizerIter")));
    class.set_attr(intern("__module__"), Value::string(intern("_tokenize")));
    class.set_attr(
        intern("__qualname__"),
        Value::string(intern("TokenizerIter")),
    );
    class.set_attr(
        intern("__call__"),
        Value::object_ptr(&*TOKENIZER_ITER_FUNCTION as *const BuiltinFunctionObject as *const ()),
    );

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);
    class
}

fn builtin_tokenizer_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    let _ = args;
    Err(BuiltinError::NotImplemented(
        "_tokenize.TokenizerIter is not implemented yet in Prism".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exposes_tokenizer_iter_placeholder() {
        let module = TokenizeModule::new();
        assert!(
            module
                .get_attr("TokenizerIter")
                .expect("TokenizerIter should exist")
                .as_object_ptr()
                .is_some()
        );
    }
}
