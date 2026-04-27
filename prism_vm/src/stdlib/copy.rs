//! Native subset of CPython's `copy` module.
//!
//! This provides the identity-preserving fast paths needed by early regression
//! tests. Rich object graph copying belongs in the full stdlib implementation;
//! immutable runtime objects can use these native identity paths directly.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::objects::extract_type_id;
use prism_core::Value;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::tuple::TupleObject;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

static COPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("copy.copy"), copy_value));
static DEEPCOPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("copy.deepcopy"), deepcopy_value));

/// Native `copy` module descriptor.
#[derive(Debug, Clone)]
pub struct CopyModule {
    attrs: Vec<Arc<str>>,
}

impl CopyModule {
    /// Create a new `copy` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("copy"), Arc::from("deepcopy")],
        }
    }
}

impl Default for CopyModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CopyModule {
    fn name(&self) -> &str {
        "copy"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "copy" => Ok(builtin_value(&COPY_FUNCTION)),
            "deepcopy" => Ok(builtin_value(&DEEPCOPY_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'copy' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn copy_value(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "copy() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(args[0])
}

fn deepcopy_value(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "deepcopy() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }
    let mut memo = HashMap::new();
    Ok(deepcopy_inner(args[0], &mut memo))
}

fn deepcopy_inner(value: Value, memo: &mut HashMap<usize, Value>) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };
    let key = ptr as usize;
    if let Some(copied) = memo.get(&key).copied() {
        return copied;
    }

    match extract_type_id(ptr) {
        prism_runtime::object::type_obj::TypeId::LIST => {
            let source = unsafe { &*(ptr as *const ListObject) };
            let copied = crate::alloc_managed_value(ListObject::with_capacity(source.len()));
            memo.insert(key, copied);
            let copied_ptr = copied
                .as_object_ptr()
                .expect("new list values are object pointers");
            let target = unsafe { &mut *(copied_ptr as *mut ListObject) };
            for item in source.as_slice().iter().copied() {
                target.push(deepcopy_inner(item, memo));
            }
            copied
        }
        prism_runtime::object::type_obj::TypeId::TUPLE => {
            let source = unsafe { &*(ptr as *const TupleObject) };
            let items = source
                .as_slice()
                .iter()
                .copied()
                .map(|item| deepcopy_inner(item, memo))
                .collect::<Vec<_>>();
            let copied = crate::alloc_managed_value(TupleObject::from_vec(items));
            memo.insert(key, copied);
            copied
        }
        prism_runtime::object::type_obj::TypeId::DICT => {
            let source = unsafe { &*(ptr as *const DictObject) };
            let copied = crate::alloc_managed_value(DictObject::with_capacity(source.len()));
            memo.insert(key, copied);
            let copied_ptr = copied
                .as_object_ptr()
                .expect("new dict values are object pointers");
            let target = unsafe { &mut *(copied_ptr as *mut DictObject) };
            for (key, item) in source.iter() {
                target.set(deepcopy_inner(key, memo), deepcopy_inner(item, memo));
            }
            copied
        }
        prism_runtime::object::type_obj::TypeId::SLICE => {
            let source = unsafe { &*(ptr as *const SliceObject) };
            let copied = crate::alloc_managed_value(SliceObject::new(
                deepcopy_inner(source.start_value(), memo),
                deepcopy_inner(source.stop_value(), memo),
                deepcopy_inner(source.step_value(), memo),
            ));
            memo.insert(key, copied);
            copied
        }
        _ => value,
    }
}
