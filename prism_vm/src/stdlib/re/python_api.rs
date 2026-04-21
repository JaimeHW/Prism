use super::engine::{RegexError, RegexErrorKind, prepare_pattern_for_backend};
use super::flags::RegexFlags;
use super::functions;
use super::match_obj::Match;
use super::pattern::CompiledPattern;
use crate::VirtualMachine;
use crate::builtins::BuiltinError;
use crate::error::RuntimeError;
use prism_core::Value;
use prism_core::intern::InternedString;
use prism_gc::trace::{Trace, Tracer};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::{ObjectHeader, PyObject};
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use regex::bytes::{Regex as BytesRegex, RegexBuilder as BytesRegexBuilder};
use std::ops::Range;
use std::sync::Arc;

#[derive(Debug, Clone)]
enum RegexPatternKind {
    Text(CompiledPattern),
    Bytes(CompiledBytesPattern),
}

#[repr(C)]
#[derive(Debug)]
pub struct RegexPatternObject {
    header: ObjectHeader,
    pattern: RegexPatternKind,
}

impl Clone for RegexPatternObject {
    fn clone(&self) -> Self {
        Self::new(self.pattern.clone())
    }
}

impl RegexPatternObject {
    fn new(pattern: RegexPatternKind) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::REGEX_PATTERN),
            pattern,
        }
    }

    fn flags(&self) -> u32 {
        match &self.pattern {
            RegexPatternKind::Text(pattern) => pattern.flags().bits(),
            RegexPatternKind::Bytes(pattern) => pattern.flags().bits(),
        }
    }

    fn groups(&self) -> usize {
        match &self.pattern {
            RegexPatternKind::Text(pattern) => pattern.groups().saturating_sub(1),
            RegexPatternKind::Bytes(pattern) => pattern.groups(),
        }
    }
}

impl PyObject for RegexPatternObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

unsafe impl Trace for RegexPatternObject {
    fn trace(&self, _tracer: &mut dyn Tracer) {}

    fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
            + match &self.pattern {
                RegexPatternKind::Text(pattern) => pattern.pattern().len(),
                RegexPatternKind::Bytes(pattern) => pattern.pattern().len(),
            }
    }
}

#[derive(Debug, Clone)]
enum RegexMatchKind {
    Text(Match),
    Bytes(BytesMatch),
}

#[repr(C)]
#[derive(Debug)]
pub struct RegexMatchObject {
    header: ObjectHeader,
    match_value: RegexMatchKind,
}

impl Clone for RegexMatchObject {
    fn clone(&self) -> Self {
        Self::new(self.match_value.clone())
    }
}

impl RegexMatchObject {
    fn new(match_value: RegexMatchKind) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::REGEX_MATCH),
            match_value,
        }
    }
}

impl PyObject for RegexMatchObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

unsafe impl Trace for RegexMatchObject {
    fn trace(&self, _tracer: &mut dyn Tracer) {}

    fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
            + match &self.match_value {
                RegexMatchKind::Text(match_value) => {
                    match_value.string().len()
                        + match_value.len() * std::mem::size_of::<Option<Range<usize>>>()
                }
                RegexMatchKind::Bytes(match_value) => {
                    match_value.string().len()
                        + match_value.len() * std::mem::size_of::<Option<Range<usize>>>()
                }
            }
    }
}

#[derive(Debug, Clone)]
struct CompiledBytesPattern {
    regex: BytesRegex,
    pattern: Arc<[u8]>,
    flags: RegexFlags,
}

impl CompiledBytesPattern {
    fn compile(pattern: &[u8], flags: RegexFlags) -> Result<Self, RegexError> {
        if flags.contains(RegexFlags::UNICODE) {
            return Err(RegexError {
                kind: RegexErrorKind::Unsupported,
                message: "cannot use UNICODE flag with a bytes pattern".to_string(),
                pattern: Some(String::from_utf8_lossy(pattern).into_owned()),
                position: None,
            });
        }
        if flags.contains(RegexFlags::LOCALE) {
            return Err(RegexError {
                kind: RegexErrorKind::Unsupported,
                message: "LOCALE flag is not yet supported for bytes patterns".to_string(),
                pattern: Some(String::from_utf8_lossy(pattern).into_owned()),
                position: None,
            });
        }

        let pattern_text = std::str::from_utf8(pattern).map_err(|_| RegexError {
            kind: RegexErrorKind::Syntax,
            message: "bytes patterns must be valid UTF-8".to_string(),
            pattern: Some(String::from_utf8_lossy(pattern).into_owned()),
            position: None,
        })?;
        let pattern_text = prepare_pattern_for_backend(pattern_text, flags)?;

        let mut builder = BytesRegexBuilder::new(&pattern_text);
        builder.unicode(false);

        let regex = builder.build().map_err(|err| RegexError {
            kind: RegexErrorKind::Syntax,
            message: err.to_string(),
            pattern: Some(String::from_utf8_lossy(pattern).into_owned()),
            position: None,
        })?;

        Ok(Self {
            regex,
            pattern: Arc::from(pattern.to_vec()),
            flags,
        })
    }

    fn pattern(&self) -> &[u8] {
        &self.pattern
    }

    fn flags(&self) -> RegexFlags {
        self.flags
    }

    fn groups(&self) -> usize {
        self.regex.captures_len().saturating_sub(1)
    }

    fn match_(&self, text: &[u8]) -> Option<BytesMatch> {
        self.regex.captures(text).and_then(|captures| {
            let matched = captures.get(0)?;
            if matched.start() == 0 {
                Some(BytesMatch::from_captures(&captures, text))
            } else {
                None
            }
        })
    }

    fn search(&self, text: &[u8]) -> Option<BytesMatch> {
        self.regex
            .captures(text)
            .map(|captures| BytesMatch::from_captures(&captures, text))
    }

    fn fullmatch(&self, text: &[u8]) -> Option<BytesMatch> {
        let matched = self.match_(text)?;
        (matched.full_end() == text.len()).then_some(matched)
    }
}

#[derive(Debug, Clone)]
struct BytesMatch {
    string: Arc<[u8]>,
    full_span: Range<usize>,
    groups: Vec<Option<Range<usize>>>,
}

impl BytesMatch {
    fn from_captures(captures: &regex::bytes::Captures<'_>, text: &[u8]) -> Self {
        let full = captures.get(0).expect("regex capture 0 should exist");
        let groups = captures
            .iter()
            .map(|group| group.map(|group| group.start()..group.end()))
            .collect();
        Self {
            string: Arc::from(text.to_vec()),
            full_span: full.start()..full.end(),
            groups,
        }
    }

    fn len(&self) -> usize {
        self.groups.len()
    }

    fn string(&self) -> &[u8] {
        &self.string
    }

    fn full_end(&self) -> usize {
        self.full_span.end
    }

    fn group(&self, index: usize) -> Result<Option<&[u8]>, BuiltinError> {
        let Some(group) = self.groups.get(index) else {
            return Err(BuiltinError::IndexError("no such group".to_string()));
        };
        Ok(group
            .as_ref()
            .map(|span| &self.string[span.start..span.end]))
    }

    fn start(&self, index: usize) -> Result<i64, BuiltinError> {
        let Some(group) = self.groups.get(index) else {
            return Err(BuiltinError::IndexError("no such group".to_string()));
        };
        Ok(group.as_ref().map_or(-1, |span| span.start as i64))
    }

    fn end(&self, index: usize) -> Result<i64, BuiltinError> {
        let Some(group) = self.groups.get(index) else {
            return Err(BuiltinError::IndexError("no such group".to_string()));
        };
        Ok(group.as_ref().map_or(-1, |span| span.end as i64))
    }
}

pub fn builtin_compile(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "compile() takes from 1 to 2 arguments ({} given)",
            args.len()
        )));
    }

    let flags = parse_flags(args.get(1).copied())?;
    let pattern = compile_pattern_object(args[0], flags)?;
    alloc_tenured_value(vm, pattern, "regex pattern")
}

pub fn builtin_match(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_pattern_entrypoint(vm, "match", args, match_subject)
}

pub fn builtin_search(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_pattern_entrypoint(vm, "search", args, search_subject)
}

pub fn builtin_fullmatch(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_pattern_entrypoint(vm, "fullmatch", args, fullmatch_subject)
}

pub fn builtin_escape(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "escape() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    match parse_text_or_bytes(args[0], "pattern")? {
        SubjectValue::Text(text) => alloc_value(
            vm,
            StringObject::from_string(functions::escape(&text)),
            "escaped regex string",
        ),
        SubjectValue::Bytes(bytes) => alloc_value(
            vm,
            BytesObject::from_vec(escape_bytes(&bytes)),
            "escaped regex bytes",
        ),
    }
}

pub fn builtin_purge(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "purge() takes no arguments ({} given)",
            args.len()
        )));
    }

    functions::purge();
    Ok(Value::none())
}

pub fn builtin_pattern_match(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    builtin_bound_pattern_entrypoint(vm, "match", args, match_subject)
}

pub fn builtin_pattern_search(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    builtin_bound_pattern_entrypoint(vm, "search", args, search_subject)
}

pub fn builtin_pattern_fullmatch(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    builtin_bound_pattern_entrypoint(vm, "fullmatch", args, fullmatch_subject)
}

pub fn builtin_match_group(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "Match.group() takes from 1 to 2 arguments ({} given)",
            args.len()
        )));
    }

    let group_index = parse_group_index(args.get(1).copied())?;
    let match_value = expect_match_ref(args[0], "group")?;
    match &match_value.match_value {
        RegexMatchKind::Text(match_value) => {
            value_from_optional_str(vm, match_value.group(group_index))
        }
        RegexMatchKind::Bytes(match_value) => {
            value_from_optional_bytes(vm, match_value.group(group_index)?)
        }
    }
}

pub fn builtin_match_groups(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "Match.groups() takes from 1 to 2 arguments ({} given)",
            args.len()
        )));
    }

    let match_value = expect_match_ref(args[0], "groups")?;
    let default = args.get(1).copied().unwrap_or_else(Value::none);
    let elements = match &match_value.match_value {
        RegexMatchKind::Text(match_value) => {
            let groups = match_value.groups();
            let mut values = Vec::with_capacity(groups.len());
            for group in groups {
                values.push(value_from_optional_str_or_default(vm, group, default)?);
            }
            values
        }
        RegexMatchKind::Bytes(match_value) => {
            let mut values = Vec::with_capacity(match_value.len().saturating_sub(1));
            for group_index in 1..match_value.len() {
                values.push(value_from_optional_bytes_or_default(
                    vm,
                    match_value.group(group_index)?,
                    default,
                )?);
            }
            values
        }
    };

    alloc_value(vm, TupleObject::from_vec(elements), "regex groups tuple")
}

pub fn builtin_match_groupdict(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "Match.groupdict() takes from 1 to 2 arguments ({} given)",
            args.len()
        )));
    }

    let match_value = expect_match_ref(args[0], "groupdict")?;
    let default = args.get(1).copied().unwrap_or_else(Value::none);
    let mut dict = DictObject::new();

    if let RegexMatchKind::Text(match_value) = &match_value.match_value {
        for (name, value) in match_value.groupdict() {
            let key = Value::string(prism_core::intern::intern(name.as_ref()));
            let value = value_from_optional_str_or_default(vm, value, default)?;
            dict.set(key, value);
        }
    }

    alloc_value(vm, dict, "regex groupdict")
}

pub fn builtin_match_start(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_match_position(
        "start",
        args,
        |text_match, index| {
            let Some(group) = text_match.group(index) else {
                return Ok(-1);
            };
            let _ = group;
            Ok(text_match
                .start_group(index)
                .map_or(-1, |value| value as i64))
        },
        |bytes_match, index| bytes_match.start(index),
    )
}

pub fn builtin_match_end(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_match_position(
        "end",
        args,
        |text_match, index| {
            let Some(group) = text_match.group(index) else {
                return Ok(-1);
            };
            let _ = group;
            Ok(text_match.end_group(index).map_or(-1, |value| value as i64))
        },
        |bytes_match, index| bytes_match.end(index),
    )
}

pub fn builtin_match_span(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "Match.span() takes from 1 to 2 arguments ({} given)",
            args.len()
        )));
    }

    let group_index = parse_group_index(args.get(1).copied())?;
    let match_value = expect_match_ref(args[0], "span")?;
    let span = match &match_value.match_value {
        RegexMatchKind::Text(match_value) => match match_value.group(group_index) {
            Some(_) => match_value
                .span_group(group_index)
                .map(|(start, end)| (start as i64, end as i64))
                .unwrap_or((-1, -1)),
            None => (-1, -1),
        },
        RegexMatchKind::Bytes(match_value) => (
            match_value.start(group_index)?,
            match_value.end(group_index)?,
        ),
    };

    alloc_value(
        vm,
        TupleObject::from_slice(&[
            Value::int(span.0).expect("span start should fit in Value::int"),
            Value::int(span.1).expect("span end should fit in Value::int"),
        ]),
        "regex span tuple",
    )
}

pub fn pattern_attr_value(
    vm: &mut VirtualMachine,
    value: Value,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let Some(pattern) = pattern_ref(value) else {
        return Ok(None);
    };

    match name.as_str() {
        "pattern" => match &pattern.pattern {
            RegexPatternKind::Text(pattern) => alloc_runtime_value(
                vm,
                StringObject::from_string(pattern.pattern().to_string()),
                "regex pattern string",
            )
            .map(Some),
            RegexPatternKind::Bytes(pattern) => alloc_runtime_value(
                vm,
                BytesObject::from_slice(pattern.pattern()),
                "regex pattern bytes",
            )
            .map(Some),
        },
        "flags" => Ok(Some(
            Value::int(pattern.flags() as i64).expect("regex flags should fit in Value::int"),
        )),
        "groups" => Ok(Some(
            Value::int(pattern.groups() as i64)
                .expect("regex group count should fit in Value::int"),
        )),
        _ => Ok(None),
    }
}

pub fn match_attr_value(
    vm: &mut VirtualMachine,
    value: Value,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let Some(match_value) = match_ref(value) else {
        return Ok(None);
    };

    match name.as_str() {
        "string" => match &match_value.match_value {
            RegexMatchKind::Text(match_value) => alloc_runtime_value(
                vm,
                StringObject::from_string(match_value.string().to_string()),
                "regex match source string",
            )
            .map(Some),
            RegexMatchKind::Bytes(match_value) => alloc_runtime_value(
                vm,
                BytesObject::from_slice(match_value.string()),
                "regex match source bytes",
            )
            .map(Some),
        },
        _ => Ok(None),
    }
}

fn builtin_pattern_entrypoint(
    vm: &mut VirtualMachine,
    fn_name: &'static str,
    args: &[Value],
    executor: impl Fn(&RegexPatternObject, SubjectValue) -> Result<Option<RegexMatchKind>, BuiltinError>,
) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes from 2 to 3 arguments ({} given)",
            args.len()
        )));
    }

    let flags = parse_flags(args.get(2).copied())?;
    let pattern = compile_pattern_object(args[0], flags)?;
    let subject = parse_subject_for_pattern(args[1], &pattern, fn_name)?;
    match_result_to_value(vm, executor(&pattern, subject)?)
}

fn builtin_bound_pattern_entrypoint(
    vm: &mut VirtualMachine,
    method_name: &'static str,
    args: &[Value],
    executor: impl Fn(&RegexPatternObject, SubjectValue) -> Result<Option<RegexMatchKind>, BuiltinError>,
) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "Pattern.{method_name}() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1),
        )));
    }

    let pattern = expect_pattern_ref(args[0], method_name)?;
    let subject = parse_subject_for_pattern(args[1], pattern, method_name)?;
    match_result_to_value(vm, executor(pattern, subject)?)
}

fn match_subject(
    pattern: &RegexPatternObject,
    subject: SubjectValue,
) -> Result<Option<RegexMatchKind>, BuiltinError> {
    match (&pattern.pattern, subject) {
        (RegexPatternKind::Text(pattern), SubjectValue::Text(text)) => {
            Ok(pattern.match_(&text).map(RegexMatchKind::Text))
        }
        (RegexPatternKind::Bytes(pattern), SubjectValue::Bytes(bytes)) => {
            Ok(pattern.match_(&bytes).map(RegexMatchKind::Bytes))
        }
        _ => Err(BuiltinError::TypeError(
            "pattern and search text must be the same type".to_string(),
        )),
    }
}

fn search_subject(
    pattern: &RegexPatternObject,
    subject: SubjectValue,
) -> Result<Option<RegexMatchKind>, BuiltinError> {
    match (&pattern.pattern, subject) {
        (RegexPatternKind::Text(pattern), SubjectValue::Text(text)) => {
            Ok(pattern.search(&text).map(RegexMatchKind::Text))
        }
        (RegexPatternKind::Bytes(pattern), SubjectValue::Bytes(bytes)) => {
            Ok(pattern.search(&bytes).map(RegexMatchKind::Bytes))
        }
        _ => Err(BuiltinError::TypeError(
            "pattern and search text must be the same type".to_string(),
        )),
    }
}

fn fullmatch_subject(
    pattern: &RegexPatternObject,
    subject: SubjectValue,
) -> Result<Option<RegexMatchKind>, BuiltinError> {
    match (&pattern.pattern, subject) {
        (RegexPatternKind::Text(pattern), SubjectValue::Text(text)) => {
            Ok(pattern.fullmatch(&text).map(RegexMatchKind::Text))
        }
        (RegexPatternKind::Bytes(pattern), SubjectValue::Bytes(bytes)) => {
            Ok(pattern.fullmatch(&bytes).map(RegexMatchKind::Bytes))
        }
        _ => Err(BuiltinError::TypeError(
            "pattern and search text must be the same type".to_string(),
        )),
    }
}

fn compile_pattern_object(
    pattern: Value,
    flags: RegexFlags,
) -> Result<RegexPatternObject, BuiltinError> {
    if let Some(existing) = pattern_ref(pattern) {
        if flags.bits() != 0 {
            return Err(BuiltinError::ValueError(
                "cannot process flags argument with a compiled pattern".to_string(),
            ));
        }
        return Ok(existing.clone());
    }

    match parse_text_or_bytes(pattern, "pattern")? {
        SubjectValue::Text(pattern) => CompiledPattern::compile(&pattern, flags)
            .map(|pattern| RegexPatternObject::new(RegexPatternKind::Text(pattern)))
            .map_err(regex_error_to_builtin_error),
        SubjectValue::Bytes(pattern) => CompiledBytesPattern::compile(&pattern, flags)
            .map(|pattern| RegexPatternObject::new(RegexPatternKind::Bytes(pattern)))
            .map_err(regex_error_to_builtin_error),
    }
}

enum SubjectValue {
    Text(String),
    Bytes(Vec<u8>),
}

fn parse_text_or_bytes(
    value: Value,
    param_name: &'static str,
) -> Result<SubjectValue, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError(format!("{param_name} must be str or bytes")))?;
        let interned = prism_core::intern::interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError(format!("{param_name} must be str or bytes")))?;
        return Ok(SubjectValue::Text(interned.as_str().to_string()));
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{param_name} must be str or bytes"
        )));
    };
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::STR => Ok(SubjectValue::Text(
            unsafe { &*(ptr as *const StringObject) }
                .as_str()
                .to_string(),
        )),
        TypeId::BYTES | TypeId::BYTEARRAY => Ok(SubjectValue::Bytes(
            unsafe { &*(ptr as *const BytesObject) }.as_bytes().to_vec(),
        )),
        _ => Err(BuiltinError::TypeError(format!(
            "{param_name} must be str or bytes"
        ))),
    }
}

fn parse_subject_for_pattern(
    value: Value,
    pattern: &RegexPatternObject,
    param_name: &'static str,
) -> Result<SubjectValue, BuiltinError> {
    let subject = parse_text_or_bytes(value, param_name)?;
    match (&pattern.pattern, &subject) {
        (RegexPatternKind::Text(_), SubjectValue::Text(_))
        | (RegexPatternKind::Bytes(_), SubjectValue::Bytes(_)) => Ok(subject),
        (RegexPatternKind::Text(_), SubjectValue::Bytes(_)) => Err(BuiltinError::TypeError(
            format!("{param_name} must be str, not bytes"),
        )),
        (RegexPatternKind::Bytes(_), SubjectValue::Text(_)) => Err(BuiltinError::TypeError(
            format!("{param_name} must be bytes, not str"),
        )),
    }
}

fn parse_flags(value: Option<Value>) -> Result<RegexFlags, BuiltinError> {
    let Some(value) = value else {
        return Ok(RegexFlags::default());
    };
    if value.is_none() {
        return Ok(RegexFlags::default());
    }

    let raw = value
        .as_int()
        .or_else(|| value.as_bool().map(|flag| if flag { 1 } else { 0 }))
        .ok_or_else(|| BuiltinError::TypeError("flags must be an integer".to_string()))?;
    if raw < 0 {
        return Err(BuiltinError::ValueError(
            "flags must be a non-negative integer".to_string(),
        ));
    }

    Ok(RegexFlags::new(raw as u32))
}

fn parse_group_index(value: Option<Value>) -> Result<usize, BuiltinError> {
    let Some(value) = value else {
        return Ok(0);
    };
    let raw = value
        .as_int()
        .or_else(|| value.as_bool().map(|flag| if flag { 1 } else { 0 }))
        .ok_or_else(|| BuiltinError::TypeError("group index must be an integer".to_string()))?;
    if raw < 0 {
        return Err(BuiltinError::IndexError("no such group".to_string()));
    }
    Ok(raw as usize)
}

fn regex_error_to_builtin_error(err: RegexError) -> BuiltinError {
    match err.kind {
        RegexErrorKind::Unsupported => BuiltinError::NotImplemented(err.message),
        _ => BuiltinError::ValueError(err.message),
    }
}

fn value_from_optional_str(
    vm: &mut VirtualMachine,
    value: Option<&str>,
) -> Result<Value, BuiltinError> {
    match value {
        Some(value) => alloc_value(
            vm,
            StringObject::from_string(value.to_string()),
            "regex group string",
        ),
        None => Ok(Value::none()),
    }
}

fn value_from_optional_bytes(
    vm: &mut VirtualMachine,
    value: Option<&[u8]>,
) -> Result<Value, BuiltinError> {
    match value {
        Some(value) => alloc_value(vm, BytesObject::from_slice(value), "regex group bytes"),
        None => Ok(Value::none()),
    }
}

fn value_from_optional_str_or_default(
    vm: &mut VirtualMachine,
    value: Option<&str>,
    default: Value,
) -> Result<Value, BuiltinError> {
    match value {
        Some(value) => value_from_optional_str(vm, Some(value)),
        None => Ok(default),
    }
}

fn value_from_optional_bytes_or_default(
    vm: &mut VirtualMachine,
    value: Option<&[u8]>,
    default: Value,
) -> Result<Value, BuiltinError> {
    match value {
        Some(value) => value_from_optional_bytes(vm, Some(value)),
        None => Ok(default),
    }
}

fn match_result_to_value(
    vm: &mut VirtualMachine,
    match_value: Option<RegexMatchKind>,
) -> Result<Value, BuiltinError> {
    match match_value {
        Some(match_value) => alloc_value(vm, RegexMatchObject::new(match_value), "regex match"),
        None => Ok(Value::none()),
    }
}

fn alloc_value<T: Trace>(
    vm: &mut VirtualMachine,
    value: T,
    context: &'static str,
) -> Result<Value, BuiltinError> {
    vm.allocator()
        .alloc_value(value)
        .ok_or_else(|| BuiltinError::TypeError(format!("out of memory allocating {context}")))
}

fn alloc_tenured_value<T: Trace>(
    vm: &mut VirtualMachine,
    value: T,
    context: &'static str,
) -> Result<Value, BuiltinError> {
    vm.allocator()
        .alloc_tenured(value)
        .map(|ptr| Value::object_ptr(ptr as *const ()))
        .ok_or_else(|| BuiltinError::TypeError(format!("out of memory allocating {context}")))
}

fn alloc_runtime_value<T: Trace>(
    vm: &mut VirtualMachine,
    value: T,
    context: &'static str,
) -> Result<Value, RuntimeError> {
    vm.allocator()
        .alloc_value(value)
        .ok_or_else(|| RuntimeError::internal(format!("out of memory allocating {context}")))
}

fn escape_bytes(pattern: &[u8]) -> Vec<u8> {
    let mut escaped = Vec::with_capacity(pattern.len() * 2);
    for &byte in pattern {
        match byte {
            b'\\' | b'.' | b'+' | b'*' | b'?' | b'[' | b']' | b'{' | b'}' | b'(' | b')' | b'^'
            | b'$' | b'|' | b'#' | b'&' | b'-' | b'~' => {
                escaped.push(b'\\');
                escaped.push(byte);
            }
            _ => escaped.push(byte),
        }
    }
    escaped
}

fn expect_pattern_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static RegexPatternObject, BuiltinError> {
    pattern_ref(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor 'Pattern.{method_name}' requires a 'Pattern' object"
        ))
    })
}

fn expect_match_ref(
    value: Value,
    method_name: &'static str,
) -> Result<&'static RegexMatchObject, BuiltinError> {
    match_ref(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor 'Match.{method_name}' requires a 'Match' object"
        ))
    })
}

fn builtin_match_position(
    method_name: &'static str,
    args: &[Value],
    text: impl Fn(&Match, usize) -> Result<i64, BuiltinError>,
    bytes: impl Fn(&BytesMatch, usize) -> Result<i64, BuiltinError>,
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "Match.{method_name}() takes from 1 to 2 arguments ({} given)",
            args.len()
        )));
    }

    let group_index = parse_group_index(args.get(1).copied())?;
    let match_value = expect_match_ref(args[0], method_name)?;
    let position = match &match_value.match_value {
        RegexMatchKind::Text(match_value) => text(match_value, group_index)?,
        RegexMatchKind::Bytes(match_value) => bytes(match_value, group_index)?,
    };
    Ok(Value::int(position).expect("match positions should fit in Value::int"))
}

fn pattern_ref(value: Value) -> Option<&'static RegexPatternObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::REGEX_PATTERN)
        .then(|| unsafe { &*(ptr as *const RegexPatternObject) })
}

fn match_ref(value: Value) -> Option<&'static RegexMatchObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::REGEX_MATCH).then(|| unsafe { &*(ptr as *const RegexMatchObject) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VirtualMachine;
    use prism_core::intern::intern;

    fn string_value(value: Value) -> String {
        if value.is_string() {
            let ptr = value
                .as_string_object_ptr()
                .expect("tagged string should provide a pointer");
            return prism_core::intern::interned_by_ptr(ptr as *const u8)
                .expect("tagged string pointer should resolve")
                .as_str()
                .to_string();
        }

        let ptr = value.as_object_ptr().expect("expected string object");
        unsafe { &*(ptr as *const StringObject) }
            .as_str()
            .to_string()
    }

    fn bytes_value(value: Value) -> Vec<u8> {
        let ptr = value.as_object_ptr().expect("expected bytes object");
        unsafe { &*(ptr as *const BytesObject) }.as_bytes().to_vec()
    }

    fn tuple_values(value: Value) -> Vec<Value> {
        let ptr = value.as_object_ptr().expect("expected tuple object");
        unsafe { &*(ptr as *const TupleObject) }.as_slice().to_vec()
    }

    fn dict_entries(value: Value) -> Vec<(Value, Value)> {
        let ptr = value.as_object_ptr().expect("expected dict object");
        unsafe { &*(ptr as *const DictObject) }.iter().collect()
    }

    #[test]
    fn test_compile_string_pattern_and_match_group() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("compile should succeed");
        let matched =
            builtin_pattern_match(&mut vm, &[pattern, Value::string(intern("abc123def"))])
                .expect("pattern.match should succeed");
        assert!(matched.is_none(), "match() should anchor at the start");

        let searched =
            builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc123def"))])
                .expect("pattern.search should succeed");
        let group = builtin_match_group(&mut vm, &[searched]).expect("group() should succeed");
        assert_eq!(string_value(group), "123");
    }

    #[test]
    fn test_compile_bytes_pattern_and_match_group() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(br"\w+"))) as *const (),
            )],
        )
        .expect("bytes compile should succeed");
        let searched = builtin_pattern_search(
            &mut vm,
            &[
                pattern,
                Value::object_ptr(
                    Box::into_raw(Box::new(BytesObject::from_slice(b"abc123"))) as *const ()
                ),
            ],
        )
        .expect("bytes search should succeed");
        let group = builtin_match_group(&mut vm, &[searched]).expect("group() should succeed");
        assert_eq!(bytes_value(group), b"abc123");
    }

    #[test]
    fn test_escape_accepts_bytes() {
        let mut vm = VirtualMachine::new();
        let escaped = builtin_escape(
            &mut vm,
            &[Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(b"a+b"))) as *const (),
            )],
        )
        .expect("escape should succeed");
        assert_eq!(bytes_value(escaped), br"a\+b");
    }

    #[test]
    fn test_match_groups_returns_capture_tuple() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"(\d+)-(\w+)"))])
            .expect("compile should succeed");
        let matched = builtin_pattern_search(
            &mut vm,
            &[pattern, Value::string(intern("prefix 123-word suffix"))],
        )
        .expect("search should succeed");

        let groups = builtin_match_groups(&mut vm, &[matched]).expect("groups() should succeed");
        let values = tuple_values(groups);
        assert_eq!(values.len(), 2);
        assert_eq!(string_value(values[0]), "123");
        assert_eq!(string_value(values[1]), "word");
    }

    #[test]
    fn test_match_groups_uses_default_for_unmatched_captures() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"(a)?(b)"))])
            .expect("compile should succeed");
        let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("b"))])
            .expect("search should succeed");

        let groups = builtin_match_groups(&mut vm, &[matched, Value::string(intern("<missing>"))])
            .expect("groups() should succeed");
        let values = tuple_values(groups);
        assert_eq!(values.len(), 2);
        assert_eq!(string_value(values[0]), "<missing>");
        assert_eq!(string_value(values[1]), "b");
    }

    #[test]
    fn test_match_groupdict_returns_named_groups() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::string(intern(r"(?P<lhs>\w+)=(?P<rhs>\w+)"))],
        )
        .expect("compile should succeed");
        let matched =
            builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("alpha=beta"))])
                .expect("search should succeed");

        let groupdict =
            builtin_match_groupdict(&mut vm, &[matched]).expect("groupdict() should succeed");
        let entries = dict_entries(groupdict);
        assert_eq!(entries.len(), 2);

        let lhs = entries
            .iter()
            .find(|(key, _)| {
                prism_runtime::types::string::value_as_string_ref(*key)
                    .is_some_and(|key| key.as_str() == "lhs")
            })
            .map(|(_, value)| *value)
            .expect("lhs entry should exist");
        let rhs = entries
            .iter()
            .find(|(key, _)| {
                prism_runtime::types::string::value_as_string_ref(*key)
                    .is_some_and(|key| key.as_str() == "rhs")
            })
            .map(|(_, value)| *value)
            .expect("rhs entry should exist");

        assert_eq!(string_value(lhs), "alpha");
        assert_eq!(string_value(rhs), "beta");
    }
}
