//! Native process-spawn primitive for Prism's Python `subprocess` facade.
//!
//! The public `subprocess` API is Python code, matching CPython's layering, but
//! process creation and pipe collection belong in Rust so argument validation,
//! environment handling, and I/O capture stay fast and portable.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::{BytesObject, value_as_bytes_ref};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{Arc, LazyLock};

static RUN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_prism_subprocess.run"), run));

#[derive(Debug, Clone)]
pub struct PrismSubprocessModule {
    attrs: Vec<Arc<str>>,
}

impl PrismSubprocessModule {
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("run")],
        }
    }
}

impl Default for PrismSubprocessModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for PrismSubprocessModule {
    fn name(&self) -> &str {
        "_prism_subprocess"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "run" => Ok(builtin_value(&RUN_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_prism_subprocess' has no attribute '{}'",
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

fn run(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 5 {
        return Err(BuiltinError::TypeError(format!(
            "run() takes exactly 5 arguments ({} given)",
            args.len()
        )));
    }

    let argv = command_args(args[0])?;
    if argv.is_empty() {
        return Err(BuiltinError::ValueError(
            "run() command must contain at least one argument".to_string(),
        ));
    }

    let input = optional_bytes(args[1], "input")?;
    let env = optional_env(args[2])?;
    let cwd = optional_string(args[3], "cwd")?;
    let merge_stderr = bool_arg(args[4], "merge_stderr")?;

    let mut command = Command::new(&argv[0]);
    command.args(&argv[1..]);
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    if input.is_some() {
        command.stdin(Stdio::piped());
    }
    if let Some(env) = env {
        command.env_clear();
        command.envs(env);
    }
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }

    let output = if let Some(input) = input {
        let mut child = command.spawn().map_err(|err| {
            BuiltinError::OSError(format!("run() failed to spawn process: {err}"))
        })?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(&input).map_err(|err| {
                BuiltinError::OSError(format!("run() failed to write stdin: {err}"))
            })?;
        }
        child.wait_with_output().map_err(|err| {
            BuiltinError::OSError(format!("run() failed to wait for process: {err}"))
        })?
    } else {
        command.output().map_err(|err| {
            BuiltinError::OSError(format!("run() failed to execute process: {err}"))
        })?
    };

    let status = output.status.code().unwrap_or(-1);
    let mut stdout = output.stdout;
    let stderr = if merge_stderr {
        stdout.extend_from_slice(&output.stderr);
        Vec::new()
    } else {
        output.stderr
    };

    Ok(crate::alloc_managed_value(TupleObject::from_vec(vec![
        Value::int(status as i64).expect("process return code fits in tagged int"),
        crate::alloc_managed_value(BytesObject::from_vec(stdout)),
        crate::alloc_managed_value(BytesObject::from_vec(stderr)),
    ])))
}

fn command_args(value: Value) -> Result<Vec<String>, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "run() args must be a sequence of str, not {}",
            value.type_name()
        ))
    })?;

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::LIST => {
            let list = unsafe { &*(ptr as *const ListObject) };
            list.as_slice()
                .iter()
                .copied()
                .enumerate()
                .map(|(index, value)| string_item(value, index))
                .collect()
        }
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple
                .as_slice()
                .iter()
                .copied()
                .enumerate()
                .map(|(index, value)| string_item(value, index))
                .collect()
        }
        _ => Err(BuiltinError::TypeError(format!(
            "run() args must be a sequence of str, not {}",
            value.type_name()
        ))),
    }
}

fn string_item(value: Value, index: usize) -> Result<String, BuiltinError> {
    value_to_string(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "run() argument {index} must be str, not {}",
            value.type_name()
        ))
    })
}

fn optional_bytes(value: Value, name: &str) -> Result<Option<Vec<u8>>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }
    value_as_bytes_ref(value)
        .map(|bytes| bytes.as_bytes().to_vec())
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "run() {name} must be bytes or None, not {}",
                value.type_name()
            ))
        })
        .map(Some)
}

fn optional_string(value: Value, name: &str) -> Result<Option<String>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }
    value_to_string(value)
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "run() {name} must be str or None, not {}",
                value.type_name()
            ))
        })
        .map(Some)
}

fn optional_env(value: Value) -> Result<Option<Vec<(String, String)>>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }

    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "run() env must be dict or None, not {}",
            value.type_name()
        ))
    })?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::DICT {
        return Err(BuiltinError::TypeError(format!(
            "run() env must be dict or None, not {}",
            value.type_name()
        )));
    }

    let dict = unsafe { &*(ptr as *const DictObject) };
    let mut env = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key = value_to_string(key)
            .ok_or_else(|| BuiltinError::TypeError("run() env keys must be str".to_string()))?;
        let value = value_to_string(value)
            .ok_or_else(|| BuiltinError::TypeError("run() env values must be str".to_string()))?;
        if key.is_empty() || key.contains('=') || key.contains('\0') {
            return Err(BuiltinError::ValueError(format!(
                "run() illegal environment variable name: {key:?}"
            )));
        }
        if value.contains('\0') {
            return Err(BuiltinError::ValueError(format!(
                "run() embedded null character in environment value for {key:?}"
            )));
        }
        env.push((key, value));
    }
    Ok(Some(env))
}

fn bool_arg(value: Value, name: &str) -> Result<bool, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(flag);
    }
    if let Some(integer) = value.as_int() {
        return Ok(integer != 0);
    }
    Err(BuiltinError::TypeError(format!(
        "run() {name} must be bool, not {}",
        value.type_name()
    )))
}

fn value_to_string(value: Value) -> Option<String> {
    value_as_string_ref(value).map(|string| string.as_str().to_string())
}
