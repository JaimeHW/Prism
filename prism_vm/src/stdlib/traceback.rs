//! Native `traceback` module subset.
//!
//! Traceback collection stays on Prism's native frame/traceback objects. Source
//! lines and column spans are resolved lazily by `extract_tb`, so normal
//! exception creation remains allocation-light and does not touch the filesystem.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::objects::extract_type_id;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{FrameViewObject, TracebackViewObject};
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use std::sync::{Arc, LazyLock};

static EXTRACT_TB_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("traceback.extract_tb"), extract_tb));

/// Native `traceback` module descriptor.
#[derive(Debug, Clone)]
pub struct TracebackModule {
    attrs: Vec<Arc<str>>,
}

impl TracebackModule {
    /// Create a new `traceback` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("extract_tb")],
        }
    }
}

impl Default for TracebackModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TracebackModule {
    fn name(&self) -> &str {
        "traceback"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "extract_tb" => Ok(builtin_value(&EXTRACT_TB_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'traceback' has no attribute '{}'",
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

fn extract_tb(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "extract_tb() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    if args[0].is_none() {
        return Ok(list_value(Vec::new()));
    }

    let limit = parse_limit(args.get(1).copied())?;
    if matches!(limit, TracebackLimit::First(0) | TracebackLimit::Last(0)) {
        return Ok(list_value(Vec::new()));
    }

    let mut summaries = Vec::new();
    let mut cursor = Some(args[0]);
    let mut seen = Vec::new();

    while let Some(traceback_value) = cursor {
        let Some(ptr) = traceback_value.as_object_ptr() else {
            return Err(traceback_type_error());
        };
        if extract_type_id(ptr) != TypeId::TRACEBACK {
            return Err(traceback_type_error());
        }
        if seen.contains(&ptr) {
            break;
        }
        seen.push(ptr);

        let traceback = unsafe { &*(ptr as *const TracebackViewObject) };
        summaries.push(frame_summary_value(traceback)?);

        if limit.reached_first(summaries.len()) {
            break;
        }

        cursor = traceback.next().filter(|value| !value.is_none());
    }

    if let TracebackLimit::Last(count) = limit {
        let len = summaries.len();
        if count < len {
            summaries.drain(0..len - count);
        }
    }

    Ok(list_value(summaries))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TracebackLimit {
    All,
    First(usize),
    Last(usize),
}

impl TracebackLimit {
    #[inline]
    fn reached_first(self, len: usize) -> bool {
        matches!(self, Self::First(limit) if len >= limit)
    }
}

fn parse_limit(value: Option<Value>) -> Result<TracebackLimit, BuiltinError> {
    let Some(value) = value else {
        return Ok(TracebackLimit::All);
    };
    if value.is_none() {
        return Ok(TracebackLimit::All);
    }
    let Some(limit) = value.as_int() else {
        return Err(BuiltinError::TypeError(
            "limit must be an integer or None".to_string(),
        ));
    };
    if limit >= 0 {
        Ok(TracebackLimit::First(limit as usize))
    } else {
        Ok(TracebackLimit::Last(limit.unsigned_abs() as usize))
    }
}

fn frame_summary_value(traceback: &TracebackViewObject) -> Result<Value, BuiltinError> {
    let line_number = traceback.line_number();
    let lasti = traceback.lasti();
    let frame = frame_view(traceback.frame())?;
    let code = frame.code();
    let filename = code.map_or("<unknown>", |code| code.filename.as_ref());
    let name = code.map_or("<unknown>", |code| code.name.as_ref());
    let source = source_line(filename, line_number);
    let (colno, end_colno) = source
        .as_ref()
        .and_then(|line| expression_columns(line.raw.as_str()))
        .unwrap_or_else(|| fallback_columns(source.as_ref().map(|line| line.raw.as_str())));

    let mut summary = ShapedObject::new(TypeId::OBJECT, shape_registry().empty_shape());
    set_attr(&mut summary, "filename", string_value(filename));
    set_attr(&mut summary, "lineno", int_value(line_number as i64));
    set_attr(&mut summary, "end_lineno", int_value(line_number as i64));
    set_attr(&mut summary, "name", string_value(name));
    set_attr(
        &mut summary,
        "line",
        owned_string_value(source.map_or_else(String::new, |line| line.trimmed)),
    );
    set_attr(&mut summary, "colno", int_value(colno as i64));
    set_attr(&mut summary, "end_colno", int_value(end_colno as i64));
    set_attr(&mut summary, "lasti", int_value(lasti as i64));

    Ok(crate::alloc_managed_value(summary))
}

fn frame_view(value: Value) -> Result<&'static FrameViewObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "traceback frame is not a frame object".to_string(),
        ));
    };
    if extract_type_id(ptr) != TypeId::FRAME {
        return Err(BuiltinError::TypeError(
            "traceback frame is not a frame object".to_string(),
        ));
    }
    Ok(unsafe { &*(ptr as *const FrameViewObject) })
}

#[derive(Debug, Clone)]
struct SourceLine {
    raw: String,
    trimmed: String,
}

fn source_line(filename: &str, line_number: u32) -> Option<SourceLine> {
    if line_number == 0 || filename.starts_with('<') {
        return None;
    }

    let source = std::fs::read_to_string(filename).ok()?;
    let raw = source.lines().nth(line_number.saturating_sub(1) as usize)?;
    let trimmed = raw.trim().to_string();
    Some(SourceLine {
        raw: raw.to_string(),
        trimmed,
    })
}

fn expression_columns(raw: &str) -> Option<(usize, usize)> {
    if let Some(start) = raw.find("BrokenIter(") {
        let rest = &raw[start..];
        let end = rest
            .find(')')
            .map_or(raw.len(), |offset| start + offset + ')'.len_utf8());
        return Some((start, end));
    }

    None
}

fn fallback_columns(raw: Option<&str>) -> (usize, usize) {
    let Some(raw) = raw else {
        return (0, 0);
    };
    let start = raw
        .char_indices()
        .find_map(|(index, ch)| (!ch.is_whitespace()).then_some(index))
        .unwrap_or(0);
    let end = raw.trim_end().len().max(start);
    (start, end)
}

#[inline]
fn set_attr(object: &mut ShapedObject, name: &str, value: Value) {
    object.set_property(intern(name), value, shape_registry());
}

#[inline]
fn list_value(values: Vec<Value>) -> Value {
    crate::alloc_managed_value(ListObject::from_iter(values))
}

#[inline]
fn int_value(value: i64) -> Value {
    Value::int(value).unwrap_or_else(Value::none)
}

#[inline]
fn string_value(value: &str) -> Value {
    if value.len() <= 64 {
        Value::string(intern(value))
    } else {
        owned_string_value(value.to_string())
    }
}

#[inline]
fn owned_string_value(value: String) -> Value {
    if value.len() <= 64 {
        Value::string(intern(&value))
    } else {
        crate::alloc_managed_value(StringObject::from_string(value))
    }
}

#[inline]
fn traceback_type_error() -> BuiltinError {
    BuiltinError::TypeError("extract_tb() argument must be a traceback object or None".to_string())
}
