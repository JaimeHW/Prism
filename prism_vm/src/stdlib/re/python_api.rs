use super::engine::{
    RegexError, RegexErrorKind, prepare_pattern_for_backend, requires_fancy_engine,
};
use super::flags::RegexFlags;
use super::functions;
use super::match_obj::Match;
use super::pattern::CompiledPattern;
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, iterator_to_value, runtime_error_to_builtin_error};
use crate::error::RuntimeError;
use crate::ops::calls::{invoke_callable_value, value_supports_call_protocol};
use prism_core::Value;
use prism_core::intern::InternedString;
use prism_gc::trace::{Trace, Tracer};
use prism_runtime::allocation_context::alloc_value_in_current_heap_or_box;
use prism_runtime::gc_dispatch::{DispatchEntry, register_external_dispatch};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::{ObjectHeader, PyObject};
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::iter::IteratorObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use regex::bytes::{Regex as BytesRegex, RegexBuilder as BytesRegexBuilder};
use rustc_hash::FxHashMap;
use std::ops::Range;
use std::sync::{Arc, Once};

static REGEX_GC_DISPATCH_ONCE: Once = Once::new();

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
        ensure_regex_gc_dispatch_registered();
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

    fn groupindex(&self) -> Vec<Option<String>> {
        match &self.pattern {
            RegexPatternKind::Text(pattern) => pattern.groupindex(),
            RegexPatternKind::Bytes(pattern) => pattern.group_names(),
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
        ensure_regex_gc_dispatch_registered();
        Self {
            header: ObjectHeader::new(TypeId::REGEX_MATCH),
            match_value,
        }
    }
}

pub(crate) fn ensure_regex_gc_dispatch_registered() {
    REGEX_GC_DISPATCH_ONCE.call_once(|| {
        register_external_dispatch(
            TypeId::REGEX_PATTERN,
            DispatchEntry {
                trace: trace_regex_pattern,
                size: size_regex_pattern,
                finalize: finalize_regex_pattern,
            },
        );
        register_external_dispatch(
            TypeId::REGEX_MATCH,
            DispatchEntry {
                trace: trace_regex_match,
                size: size_regex_match,
                finalize: finalize_regex_match,
            },
        );
    });
}

unsafe fn trace_regex_pattern(ptr: *const (), tracer: &mut dyn Tracer) {
    let object = unsafe { &*(ptr as *const RegexPatternObject) };
    object.trace(tracer);
}

unsafe fn size_regex_pattern(ptr: *const ()) -> usize {
    let object = unsafe { &*(ptr as *const RegexPatternObject) };
    object.size_of()
}

unsafe fn finalize_regex_pattern(ptr: *mut ()) {
    unsafe { std::ptr::drop_in_place(ptr as *mut RegexPatternObject) };
}

unsafe fn trace_regex_match(ptr: *const (), tracer: &mut dyn Tracer) {
    let object = unsafe { &*(ptr as *const RegexMatchObject) };
    object.trace(tracer);
}

unsafe fn size_regex_match(ptr: *const ()) -> usize {
    let object = unsafe { &*(ptr as *const RegexMatchObject) };
    object.size_of()
}

unsafe fn finalize_regex_match(ptr: *mut ()) {
    unsafe { std::ptr::drop_in_place(ptr as *mut RegexMatchObject) };
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
enum BytesRegexEngine {
    Standard(BytesRegex),
    Fancy(fancy_regex::Regex),
}

#[derive(Debug, Clone)]
struct CompiledBytesPattern {
    engine: BytesRegexEngine,
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
        let requires_fancy = requires_fancy_engine(pattern_text);
        let prepared_pattern = prepare_pattern_for_backend(pattern_text, flags)?;
        let engine = if requires_fancy {
            let regex = fancy_regex::Regex::new(&prepared_pattern).map_err(|err| RegexError {
                kind: RegexErrorKind::Syntax,
                message: err.to_string(),
                pattern: Some(String::from_utf8_lossy(pattern).into_owned()),
                position: None,
            })?;
            BytesRegexEngine::Fancy(regex)
        } else {
            let mut builder = BytesRegexBuilder::new(&prepared_pattern);
            builder.unicode(false);

            let regex = builder.build().map_err(|err| RegexError {
                kind: RegexErrorKind::Syntax,
                message: err.to_string(),
                pattern: Some(String::from_utf8_lossy(pattern).into_owned()),
                position: None,
            })?;
            BytesRegexEngine::Standard(regex)
        };

        Ok(Self {
            engine,
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
        match &self.engine {
            BytesRegexEngine::Standard(regex) => regex.captures_len().saturating_sub(1),
            BytesRegexEngine::Fancy(regex) => regex.captures_len().saturating_sub(1),
        }
    }

    fn group_names(&self) -> Vec<Option<String>> {
        match &self.engine {
            BytesRegexEngine::Standard(regex) => regex
                .capture_names()
                .map(|name| name.map(|name| name.to_string()))
                .collect(),
            BytesRegexEngine::Fancy(regex) => regex
                .capture_names()
                .map(|name| name.map(|name| name.to_string()))
                .collect(),
        }
    }

    fn match_(&self, text: &[u8]) -> Option<BytesMatch> {
        match &self.engine {
            BytesRegexEngine::Standard(regex) => regex.captures(text).and_then(|captures| {
                let matched = captures.get(0)?;
                if matched.start() == 0 {
                    Some(BytesMatch::from_captures_with_regex(&captures, text, regex))
                } else {
                    None
                }
            }),
            BytesRegexEngine::Fancy(regex) => {
                let text_str = std::str::from_utf8(text).ok()?;
                regex
                    .captures(text_str)
                    .ok()
                    .flatten()
                    .and_then(|captures| {
                        let matched = captures.get(0)?;
                        if matched.start() == 0 {
                            Some(BytesMatch::from_fancy_captures_with_regex(
                                &captures, text, regex,
                            ))
                        } else {
                            None
                        }
                    })
            }
        }
    }

    fn match_range(&self, text: &[u8], pos: usize, endpos: Option<usize>) -> Option<BytesMatch> {
        let (slice, offset) = bounded_bytes_range(text, pos, endpos)?;
        self.match_(slice).map(|m| m.with_offset(text, offset))
    }

    fn search(&self, text: &[u8]) -> Option<BytesMatch> {
        match &self.engine {
            BytesRegexEngine::Standard(regex) => regex
                .captures(text)
                .map(|captures| BytesMatch::from_captures_with_regex(&captures, text, regex)),
            BytesRegexEngine::Fancy(regex) => {
                let text_str = std::str::from_utf8(text).ok()?;
                regex.captures(text_str).ok().flatten().map(|captures| {
                    BytesMatch::from_fancy_captures_with_regex(&captures, text, regex)
                })
            }
        }
    }

    fn search_range(&self, text: &[u8], pos: usize, endpos: Option<usize>) -> Option<BytesMatch> {
        let (slice, offset) = bounded_bytes_range(text, pos, endpos)?;
        self.search(slice).map(|m| m.with_offset(text, offset))
    }

    fn fullmatch(&self, text: &[u8]) -> Option<BytesMatch> {
        let matched = self.match_(text)?;
        (matched.full_end() == text.len()).then_some(matched)
    }

    fn fullmatch_range(
        &self,
        text: &[u8],
        pos: usize,
        endpos: Option<usize>,
    ) -> Option<BytesMatch> {
        let (slice, offset) = bounded_bytes_range(text, pos, endpos)?;
        let matched = self.match_(slice)?;
        (matched.full_end() == slice.len()).then(|| matched.with_offset(text, offset))
    }

    fn findall(&self, text: &[u8]) -> Vec<BytesMatch> {
        match &self.engine {
            BytesRegexEngine::Standard(regex) => regex
                .captures_iter(text)
                .map(|captures| BytesMatch::from_captures_with_regex(&captures, text, regex))
                .collect(),
            BytesRegexEngine::Fancy(regex) => {
                let Ok(text_str) = std::str::from_utf8(text) else {
                    return Vec::new();
                };
                regex
                    .captures_iter(text_str)
                    .filter_map(|captures| captures.ok())
                    .map(|captures| {
                        BytesMatch::from_fancy_captures_with_regex(&captures, text, regex)
                    })
                    .collect()
            }
        }
    }

    fn findall_range(&self, text: &[u8], pos: usize, endpos: Option<usize>) -> Vec<BytesMatch> {
        let Some((slice, offset)) = bounded_bytes_range(text, pos, endpos) else {
            return Vec::new();
        };
        self.findall(slice)
            .into_iter()
            .map(|m| m.with_offset(text, offset))
            .collect()
    }

    fn sub_n(&self, repl: &[u8], text: &[u8], count: usize) -> Vec<u8> {
        match &self.engine {
            BytesRegexEngine::Standard(regex) => {
                if count == 0 {
                    regex.replace_all(text, repl).into_owned()
                } else if count == 1 {
                    regex.replace(text, repl).into_owned()
                } else {
                    let mut result = text.to_vec();
                    let mut replaced = 0;
                    while replaced < count {
                        let new = regex.replace(&result, repl).into_owned();
                        if new == result {
                            break;
                        }
                        result = new;
                        replaced += 1;
                    }
                    result
                }
            }
            BytesRegexEngine::Fancy(regex) => {
                let (Ok(text_str), Ok(repl_str)) =
                    (std::str::from_utf8(text), std::str::from_utf8(repl))
                else {
                    return text.to_vec();
                };
                if count == 0 {
                    regex
                        .replace_all(text_str, repl_str)
                        .into_owned()
                        .into_bytes()
                } else if count == 1 {
                    regex.replace(text_str, repl_str).into_owned().into_bytes()
                } else {
                    let mut result = text_str.to_string();
                    let mut replaced = 0;
                    while replaced < count {
                        let new = regex.replace(&result, repl_str).into_owned();
                        if new == result {
                            break;
                        }
                        result = new;
                        replaced += 1;
                    }
                    result.into_bytes()
                }
            }
        }
    }

    fn subn(&self, repl: &[u8], text: &[u8], count: usize) -> (Vec<u8>, usize) {
        let replacements = match &self.engine {
            BytesRegexEngine::Standard(regex) => {
                let total = regex.find_iter(text).count();
                if count == 0 { total } else { total.min(count) }
            }
            BytesRegexEngine::Fancy(regex) => {
                let Ok(text_str) = std::str::from_utf8(text) else {
                    return (text.to_vec(), 0);
                };
                let total = regex.find_iter(text_str).filter_map(|m| m.ok()).count();
                if count == 0 { total } else { total.min(count) }
            }
        };
        (self.sub_n(repl, text, count), replacements)
    }

    fn split_n(&self, text: &[u8], maxsplit: usize) -> Vec<Vec<u8>> {
        match &self.engine {
            BytesRegexEngine::Standard(regex) => {
                if maxsplit == 0 {
                    regex.split(text).map(|part| part.to_vec()).collect()
                } else {
                    regex
                        .splitn(text, maxsplit + 1)
                        .map(|part| part.to_vec())
                        .collect()
                }
            }
            BytesRegexEngine::Fancy(regex) => {
                let Ok(text_str) = std::str::from_utf8(text) else {
                    return vec![text.to_vec()];
                };
                split_fancy_bytes(regex, text_str, maxsplit)
            }
        }
    }
}

fn split_fancy_bytes(regex: &fancy_regex::Regex, text: &str, maxsplit: usize) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    let mut last_end = 0usize;
    let mut split_count = 0usize;

    for matched in regex.find_iter(text).filter_map(|matched| matched.ok()) {
        if maxsplit != 0 && split_count >= maxsplit {
            break;
        }
        result.push(text.as_bytes()[last_end..matched.start()].to_vec());
        last_end = matched.end();
        split_count += 1;
    }

    result.push(text.as_bytes()[last_end..].to_vec());
    result
}

#[derive(Debug, Clone)]
struct BytesMatch {
    string: Arc<[u8]>,
    full_span: Range<usize>,
    groups: Vec<Option<Range<usize>>>,
    named_groups: FxHashMap<Arc<str>, usize>,
}

impl BytesMatch {
    fn from_captures_with_regex(
        captures: &regex::bytes::Captures<'_>,
        text: &[u8],
        regex: &BytesRegex,
    ) -> Self {
        let full = captures.get(0).expect("regex capture 0 should exist");
        let groups = captures
            .iter()
            .map(|group| group.map(|group| group.start()..group.end()))
            .collect();
        let mut named_groups = FxHashMap::default();
        for (index, name) in regex.capture_names().enumerate() {
            if let Some(name) = name {
                named_groups.insert(Arc::from(name), index);
            }
        }
        Self {
            string: Arc::from(text.to_vec()),
            full_span: full.start()..full.end(),
            groups,
            named_groups,
        }
    }

    fn from_fancy_captures_with_regex(
        captures: &fancy_regex::Captures<'_>,
        text: &[u8],
        regex: &fancy_regex::Regex,
    ) -> Self {
        let full = captures.get(0).expect("regex capture 0 should exist");
        let groups = captures
            .iter()
            .map(|group| group.map(|group| group.start()..group.end()))
            .collect();
        let mut named_groups = FxHashMap::default();
        for (index, name) in regex.capture_names().enumerate() {
            if let Some(name) = name {
                named_groups.insert(Arc::from(name), index);
            }
        }
        Self {
            string: Arc::from(text.to_vec()),
            full_span: full.start()..full.end(),
            groups,
            named_groups,
        }
    }

    fn len(&self) -> usize {
        self.groups.len()
    }

    fn string(&self) -> &[u8] {
        &self.string
    }

    fn as_bytes(&self) -> &[u8] {
        &self.string[self.full_span.clone()]
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

    fn group_index(&self, name: &str) -> Option<usize> {
        self.named_groups.get(name).copied()
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

    fn lastindex(&self) -> Option<usize> {
        self.groups
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, group)| (index > 0 && group.is_some()).then_some(index))
    }

    fn lastgroup(&self) -> Option<&str> {
        let last_index = self.lastindex()?;
        self.named_groups
            .iter()
            .find_map(|(name, &index)| (index == last_index).then_some(name.as_ref()))
    }

    fn with_offset(mut self, text: &[u8], offset: usize) -> Self {
        self.string = Arc::from(text.to_vec());
        self.full_span = (self.full_span.start + offset)..(self.full_span.end + offset);
        for group in &mut self.groups {
            if let Some(span) = group {
                *span = (span.start + offset)..(span.end + offset);
            }
        }
        self
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

pub fn builtin_findall(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "findall() takes from 2 to 3 arguments ({} given)",
            args.len()
        )));
    }

    let flags = parse_flags(args.get(2).copied())?;
    let pattern = compile_pattern_object(args[0], flags)?;
    let subject = parse_subject_for_pattern(args[1], &pattern, "findall")?;
    findall_result_to_value(vm, &pattern, subject, SearchBounds::default())
}

pub fn builtin_finditer(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "finditer() takes from 2 to 3 arguments ({} given)",
            args.len()
        )));
    }

    let flags = parse_flags(args.get(2).copied())?;
    let pattern = compile_pattern_object(args[0], flags)?;
    let subject = parse_subject_for_pattern(args[1], &pattern, "finditer")?;
    finditer_result_to_value(vm, &pattern, subject, SearchBounds::default())
}

pub fn builtin_sub(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 3 || args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "sub() takes from 3 to 5 arguments ({} given)",
            args.len()
        )));
    }

    let flags = parse_flags(args.get(4).copied())?;
    let count = parse_substitution_limit(args.get(3).copied(), "count")?;
    let pattern = compile_pattern_object(args[0], flags)?;
    let replacement = parse_replacement_for_pattern(args[1], &pattern)?;
    let subject = parse_subject_for_pattern(args[2], &pattern, "string")?;
    sub_result_to_value(vm, &pattern, replacement, subject, count)
}

pub fn builtin_subn(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 3 || args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "subn() takes from 3 to 5 arguments ({} given)",
            args.len()
        )));
    }

    let flags = parse_flags(args.get(4).copied())?;
    let count = parse_substitution_limit(args.get(3).copied(), "count")?;
    let pattern = compile_pattern_object(args[0], flags)?;
    let replacement = parse_replacement_for_pattern(args[1], &pattern)?;
    let subject = parse_subject_for_pattern(args[2], &pattern, "string")?;
    subn_result_to_value(vm, &pattern, replacement, subject, count)
}

pub fn builtin_split(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "split() takes from 2 to 4 arguments ({} given)",
            args.len()
        )));
    }

    let maxsplit = parse_substitution_limit(args.get(2).copied(), "maxsplit")?;
    let flags = parse_flags(args.get(3).copied())?;
    let pattern = compile_pattern_object(args[0], flags)?;
    let subject = parse_subject_for_pattern(args[1], &pattern, "string")?;
    split_result_to_value(vm, &pattern, subject, maxsplit)
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
    builtin_bound_pattern_entrypoint(vm, "match", args, match_subject_with_bounds)
}

pub fn builtin_pattern_search(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    builtin_bound_pattern_entrypoint(vm, "search", args, search_subject_with_bounds)
}

pub fn builtin_pattern_fullmatch(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    builtin_bound_pattern_entrypoint(vm, "fullmatch", args, fullmatch_subject_with_bounds)
}

pub fn builtin_pattern_findall(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let (pattern, subject, bounds) = parse_bound_pattern_args(args, "findall")?;
    findall_result_to_value(vm, pattern, subject, bounds)
}

pub fn builtin_pattern_finditer(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let (pattern, subject, bounds) = parse_bound_pattern_args(args, "finditer")?;
    finditer_result_to_value(vm, pattern, subject, bounds)
}

pub fn builtin_pattern_sub(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let (pattern, replacement, subject, count) = parse_bound_pattern_sub_args(args, "sub")?;
    sub_result_to_value(vm, pattern, replacement, subject, count)
}

pub fn builtin_pattern_subn(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let (pattern, replacement, subject, count) = parse_bound_pattern_sub_args(args, "subn")?;
    subn_result_to_value(vm, pattern, replacement, subject, count)
}

pub fn builtin_pattern_split(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let (pattern, subject, maxsplit) = parse_bound_pattern_split_args(args)?;
    split_result_to_value(vm, pattern, subject, maxsplit)
}

pub fn builtin_match_group(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "descriptor 'Match.group' requires a 'Match' object".to_string(),
        ));
    }

    let match_value = expect_match_ref(args[0], "group")?;
    let selectors = parse_group_selectors(&args[1..])?;

    if selectors.len() <= 1 {
        let selector = selectors
            .first()
            .cloned()
            .unwrap_or(GroupSelector::Index(0));
        return match_group_value(vm, match_value, &selector);
    }

    let mut items = Vec::with_capacity(selectors.len());
    for selector in &selectors {
        items.push(match_group_value(vm, match_value, selector)?);
    }
    alloc_value(vm, TupleObject::from_vec(items), "regex group tuple")
}

pub fn builtin_match_getitem(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "descriptor 'Match.__getitem__' requires a 'Match' object".to_string(),
        ));
    }
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "Match.__getitem__() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let match_value = expect_match_ref(args[0], "__getitem__")?;
    let selector = parse_group_selector(Some(args[1]))?;
    match_group_value(vm, match_value, &selector)
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

    match &match_value.match_value {
        RegexMatchKind::Text(match_value) => {
            for (name, value) in match_value.groupdict() {
                let key = Value::string(prism_core::intern::intern(name.as_ref()));
                let value = value_from_optional_str_or_default(vm, value, default)?;
                dict.set(key, value);
            }
        }
        RegexMatchKind::Bytes(match_value) => {
            for (name, index) in &match_value.named_groups {
                let key = Value::string(prism_core::intern::intern(name.as_ref()));
                let value =
                    value_from_optional_bytes_or_default(vm, match_value.group(*index)?, default)?;
                dict.set(key, value);
            }
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

    let match_value = expect_match_ref(args[0], "span")?;
    let selector = parse_group_selector(args.get(1).copied())?;
    let span = match &match_value.match_value {
        RegexMatchKind::Text(match_value) => {
            let group_index = resolve_text_group_index(match_value, &selector)?;
            match match_value.group(group_index) {
                Some(_) => match_value
                    .span_group(group_index)
                    .map(|(start, end)| (start as i64, end as i64))
                    .unwrap_or((-1, -1)),
                None => (-1, -1),
            }
        }
        RegexMatchKind::Bytes(match_value) => {
            let group_index = resolve_bytes_group_index(match_value, &selector)?;
            (
                match_value.start(group_index)?,
                match_value.end(group_index)?,
            )
        }
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
        "groupindex" => {
            let mut dict = DictObject::new();
            for (index, name) in pattern.groupindex().into_iter().enumerate() {
                if let Some(name) = name {
                    dict.set(
                        Value::string(prism_core::intern::intern(&name)),
                        Value::int(index as i64)
                            .expect("regex named group index should fit in Value::int"),
                    );
                }
            }
            alloc_runtime_value(vm, dict, "regex groupindex mapping").map(Some)
        }
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
        "lastindex" => Ok(Some(match_lastindex_value(match_value))),
        "lastgroup" => match &match_value.match_value {
            RegexMatchKind::Text(match_value) => {
                match_lastgroup_value(vm, match_value.lastgroup()).map(Some)
            }
            RegexMatchKind::Bytes(match_value) => {
                match_lastgroup_value(vm, match_value.lastgroup()).map(Some)
            }
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
    executor: impl Fn(
        &RegexPatternObject,
        SubjectValue,
        SearchBounds,
    ) -> Result<Option<RegexMatchKind>, BuiltinError>,
) -> Result<Value, BuiltinError> {
    let (pattern, subject, bounds) = parse_bound_pattern_args(args, method_name)?;
    match_result_to_value(vm, executor(pattern, subject, bounds)?)
}

fn match_subject(
    pattern: &RegexPatternObject,
    subject: SubjectValue,
) -> Result<Option<RegexMatchKind>, BuiltinError> {
    match_subject_with_bounds(pattern, subject, SearchBounds::default())
}

fn match_subject_with_bounds(
    pattern: &RegexPatternObject,
    subject: SubjectValue,
    bounds: SearchBounds,
) -> Result<Option<RegexMatchKind>, BuiltinError> {
    match (&pattern.pattern, subject) {
        (RegexPatternKind::Text(pattern), SubjectValue::Text(text)) => Ok(pattern
            .match_range(&text, bounds.pos, bounds.endpos)
            .map(RegexMatchKind::Text)),
        (RegexPatternKind::Bytes(pattern), SubjectValue::Bytes(bytes)) => Ok(pattern
            .match_range(&bytes, bounds.pos, bounds.endpos)
            .map(RegexMatchKind::Bytes)),
        _ => Err(BuiltinError::TypeError(
            "pattern and search text must be the same type".to_string(),
        )),
    }
}

fn search_subject(
    pattern: &RegexPatternObject,
    subject: SubjectValue,
) -> Result<Option<RegexMatchKind>, BuiltinError> {
    search_subject_with_bounds(pattern, subject, SearchBounds::default())
}

fn search_subject_with_bounds(
    pattern: &RegexPatternObject,
    subject: SubjectValue,
    bounds: SearchBounds,
) -> Result<Option<RegexMatchKind>, BuiltinError> {
    match (&pattern.pattern, subject) {
        (RegexPatternKind::Text(pattern), SubjectValue::Text(text)) => Ok(pattern
            .search_range(&text, bounds.pos, bounds.endpos)
            .map(RegexMatchKind::Text)),
        (RegexPatternKind::Bytes(pattern), SubjectValue::Bytes(bytes)) => Ok(pattern
            .search_range(&bytes, bounds.pos, bounds.endpos)
            .map(RegexMatchKind::Bytes)),
        _ => Err(BuiltinError::TypeError(
            "pattern and search text must be the same type".to_string(),
        )),
    }
}

fn fullmatch_subject(
    pattern: &RegexPatternObject,
    subject: SubjectValue,
) -> Result<Option<RegexMatchKind>, BuiltinError> {
    fullmatch_subject_with_bounds(pattern, subject, SearchBounds::default())
}

fn fullmatch_subject_with_bounds(
    pattern: &RegexPatternObject,
    subject: SubjectValue,
    bounds: SearchBounds,
) -> Result<Option<RegexMatchKind>, BuiltinError> {
    match (&pattern.pattern, subject) {
        (RegexPatternKind::Text(pattern), SubjectValue::Text(text)) => Ok(pattern
            .fullmatch_range(&text, bounds.pos, bounds.endpos)
            .map(RegexMatchKind::Text)),
        (RegexPatternKind::Bytes(pattern), SubjectValue::Bytes(bytes)) => Ok(pattern
            .fullmatch_range(&bytes, bounds.pos, bounds.endpos)
            .map(RegexMatchKind::Bytes)),
        _ => Err(BuiltinError::TypeError(
            "pattern and search text must be the same type".to_string(),
        )),
    }
}

fn findall_result_to_value(
    vm: &mut VirtualMachine,
    pattern: &RegexPatternObject,
    subject: SubjectValue,
    bounds: SearchBounds,
) -> Result<Value, BuiltinError> {
    let items = match (&pattern.pattern, subject) {
        (RegexPatternKind::Text(pattern), SubjectValue::Text(text)) => {
            let matches = pattern.findall_range(&text, bounds.pos, bounds.endpos);
            let group_count = pattern.groups().saturating_sub(1);
            let mut items = Vec::with_capacity(matches.len());
            for match_value in &matches {
                items.push(text_findall_item_to_value(vm, match_value, group_count)?);
            }
            items
        }
        (RegexPatternKind::Bytes(pattern), SubjectValue::Bytes(bytes)) => {
            let matches = pattern.findall_range(&bytes, bounds.pos, bounds.endpos);
            let group_count = pattern.groups();
            let mut items = Vec::with_capacity(matches.len());
            for match_value in &matches {
                items.push(bytes_findall_item_to_value(vm, match_value, group_count)?);
            }
            items
        }
        _ => {
            return Err(BuiltinError::TypeError(
                "pattern and search text must be the same type".to_string(),
            ));
        }
    };

    alloc_value(vm, ListObject::from_iter(items), "regex findall list")
}

fn finditer_result_to_value(
    vm: &mut VirtualMachine,
    pattern: &RegexPatternObject,
    subject: SubjectValue,
    bounds: SearchBounds,
) -> Result<Value, BuiltinError> {
    let mut items = match (&pattern.pattern, subject) {
        (RegexPatternKind::Text(pattern), SubjectValue::Text(text)) => {
            let matches = pattern.findall_range(&text, bounds.pos, bounds.endpos);
            let mut items = Vec::with_capacity(matches.len());
            for match_value in matches {
                items.push(alloc_value(
                    vm,
                    RegexMatchObject::new(RegexMatchKind::Text(match_value)),
                    "regex match",
                )?);
            }
            items
        }
        (RegexPatternKind::Bytes(pattern), SubjectValue::Bytes(bytes)) => {
            let matches = pattern.findall_range(&bytes, bounds.pos, bounds.endpos);
            let mut items = Vec::with_capacity(matches.len());
            for match_value in matches {
                items.push(alloc_value(
                    vm,
                    RegexMatchObject::new(RegexMatchKind::Bytes(match_value)),
                    "regex match",
                )?);
            }
            items
        }
        _ => {
            return Err(BuiltinError::TypeError(
                "pattern and search text must be the same type".to_string(),
            ));
        }
    };

    Ok(iterator_to_value(IteratorObject::from_values(
        std::mem::take(&mut items),
    )))
}

fn text_findall_item_to_value(
    vm: &mut VirtualMachine,
    match_value: &Match,
    group_count: usize,
) -> Result<Value, BuiltinError> {
    match group_count {
        0 => value_from_optional_str(vm, Some(match_value.as_str())),
        1 => value_from_optional_str(vm, Some(match_value.group(1).unwrap_or(""))),
        _ => {
            let mut items = Vec::with_capacity(group_count);
            for group_index in 1..=group_count {
                items.push(value_from_optional_str(
                    vm,
                    Some(match_value.group(group_index).unwrap_or("")),
                )?);
            }
            alloc_value(vm, TupleObject::from_vec(items), "regex findall tuple")
        }
    }
}

fn bytes_findall_item_to_value(
    vm: &mut VirtualMachine,
    match_value: &BytesMatch,
    group_count: usize,
) -> Result<Value, BuiltinError> {
    match group_count {
        0 => value_from_optional_bytes(vm, Some(match_value.as_bytes())),
        1 => value_from_optional_bytes(vm, Some(match_value.group(1)?.unwrap_or(b""))),
        _ => {
            let mut items = Vec::with_capacity(group_count);
            for group_index in 1..=group_count {
                items.push(value_from_optional_bytes(
                    vm,
                    Some(match_value.group(group_index)?.unwrap_or(b"")),
                )?);
            }
            alloc_value(vm, TupleObject::from_vec(items), "regex findall tuple")
        }
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

#[derive(Clone)]
enum GroupSelector {
    Index(usize),
    Name(String),
}

#[derive(Clone, Copy, Debug, Default)]
struct SearchBounds {
    pos: usize,
    endpos: Option<usize>,
}

enum SubjectValue {
    Text(String),
    Bytes(Vec<u8>),
}

enum ReplacementValue {
    Text(String),
    Bytes(Vec<u8>),
    Callable(Value),
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

fn parse_group_selector(value: Option<Value>) -> Result<GroupSelector, BuiltinError> {
    let Some(value) = value else {
        return Ok(GroupSelector::Index(0));
    };
    if let Some(name) = try_parse_python_string(value) {
        return Ok(GroupSelector::Name(name));
    }
    let raw = value
        .as_int()
        .or_else(|| value.as_bool().map(|flag| if flag { 1 } else { 0 }))
        .ok_or_else(|| {
            BuiltinError::TypeError("group index must be an integer or string".to_string())
        })?;
    if raw < 0 {
        return Err(BuiltinError::IndexError("no such group".to_string()));
    }
    Ok(GroupSelector::Index(raw as usize))
}

fn parse_group_selectors(values: &[Value]) -> Result<Vec<GroupSelector>, BuiltinError> {
    values
        .iter()
        .copied()
        .map(|value| parse_group_selector(Some(value)))
        .collect()
}

fn parse_search_bounds(
    pos: Option<Value>,
    endpos: Option<Value>,
) -> Result<SearchBounds, BuiltinError> {
    Ok(SearchBounds {
        pos: pos
            .map(|value| parse_search_index(value, "pos"))
            .transpose()?
            .unwrap_or(0),
        endpos: endpos
            .map(|value| parse_search_index(value, "endpos"))
            .transpose()?,
    })
}

fn parse_search_index(value: Value, param_name: &'static str) -> Result<usize, BuiltinError> {
    let raw = value
        .as_int()
        .or_else(|| value.as_bool().map(|flag| if flag { 1 } else { 0 }))
        .ok_or_else(|| BuiltinError::TypeError(format!("{param_name} must be an integer")))?;
    Ok(raw.max(0) as usize)
}

fn parse_substitution_limit(
    value: Option<Value>,
    param_name: &'static str,
) -> Result<usize, BuiltinError> {
    value
        .map(|value| parse_search_index(value, param_name))
        .transpose()
        .map(|value| value.unwrap_or(0))
}

fn parse_replacement_for_pattern(
    value: Value,
    pattern: &RegexPatternObject,
) -> Result<ReplacementValue, BuiltinError> {
    if value_supports_call_protocol(value) {
        return Ok(ReplacementValue::Callable(value));
    }

    match (&pattern.pattern, parse_text_or_bytes(value, "repl")?) {
        (RegexPatternKind::Text(_), SubjectValue::Text(text)) => {
            Ok(ReplacementValue::Text(translate_text_replacement(&text)?))
        }
        (RegexPatternKind::Bytes(_), SubjectValue::Bytes(bytes)) => Ok(ReplacementValue::Bytes(
            translate_bytes_replacement(&bytes)?,
        )),
        (RegexPatternKind::Text(_), SubjectValue::Bytes(_)) => Err(BuiltinError::TypeError(
            "sequence item 0: expected str instance, bytes found".to_string(),
        )),
        (RegexPatternKind::Bytes(_), SubjectValue::Text(_)) => Err(BuiltinError::TypeError(
            "sequence item 0: expected a bytes-like object, str found".to_string(),
        )),
    }
}

fn translate_text_replacement(replacement: &str) -> Result<String, BuiltinError> {
    String::from_utf8(translate_replacement_bytes(replacement.as_bytes())?)
        .map_err(|_| BuiltinError::ValueError("replacement is not valid UTF-8".to_string()))
}

fn translate_bytes_replacement(replacement: &[u8]) -> Result<Vec<u8>, BuiltinError> {
    translate_replacement_bytes(replacement)
}

fn translate_replacement_bytes(replacement: &[u8]) -> Result<Vec<u8>, BuiltinError> {
    let mut translated = Vec::with_capacity(replacement.len());
    let mut index = 0usize;

    while index < replacement.len() {
        match replacement[index] {
            b'$' => {
                translated.extend_from_slice(b"$$");
                index += 1;
            }
            b'\\' => {
                index += 1;
                if index >= replacement.len() {
                    return Err(BuiltinError::ValueError(
                        "bad escape (end of pattern)".to_string(),
                    ));
                }

                match replacement[index] {
                    b'1'..=b'9' => {
                        let start = index;
                        index += 1;
                        if index < replacement.len() && replacement[index].is_ascii_digit() {
                            index += 1;
                        }
                        translated.push(b'$');
                        translated.extend_from_slice(&replacement[start..index]);
                    }
                    b'g' if index + 1 < replacement.len() && replacement[index + 1] == b'<' => {
                        let name_start = index + 2;
                        let Some(relative_end) = replacement[name_start..]
                            .iter()
                            .position(|byte| *byte == b'>')
                        else {
                            return Err(BuiltinError::ValueError(
                                "missing > in replacement group name".to_string(),
                            ));
                        };
                        let name_end = name_start + relative_end;
                        if name_end == name_start {
                            return Err(BuiltinError::ValueError("missing group name".to_string()));
                        }
                        let name = &replacement[name_start..name_end];
                        if name.iter().all(u8::is_ascii_digit) {
                            translated.push(b'$');
                            translated.extend_from_slice(name);
                        } else {
                            translated.extend_from_slice(b"${");
                            translated.extend_from_slice(name);
                            translated.push(b'}');
                        }
                        index = name_end + 1;
                    }
                    b'\\' => {
                        translated.push(b'\\');
                        index += 1;
                    }
                    b'a' => {
                        translated.push(0x07);
                        index += 1;
                    }
                    b'b' => {
                        translated.push(0x08);
                        index += 1;
                    }
                    b'f' => {
                        translated.push(0x0c);
                        index += 1;
                    }
                    b'n' => {
                        translated.push(b'\n');
                        index += 1;
                    }
                    b'r' => {
                        translated.push(b'\r');
                        index += 1;
                    }
                    b't' => {
                        translated.push(b'\t');
                        index += 1;
                    }
                    b'v' => {
                        translated.push(0x0b);
                        index += 1;
                    }
                    escaped if escaped.is_ascii_alphabetic() => {
                        return Err(BuiltinError::ValueError(format!(
                            "bad escape \\{}",
                            escaped as char
                        )));
                    }
                    escaped => {
                        translated.push(b'\\');
                        translated.push(escaped);
                        index += 1;
                    }
                }
            }
            byte => {
                translated.push(byte);
                index += 1;
            }
        }
    }

    Ok(translated)
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

fn match_group_value(
    vm: &mut VirtualMachine,
    match_value: &RegexMatchObject,
    selector: &GroupSelector,
) -> Result<Value, BuiltinError> {
    match &match_value.match_value {
        RegexMatchKind::Text(match_value) => {
            let group_index = resolve_text_group_index(match_value, selector)?;
            value_from_optional_str(vm, match_value.group(group_index))
        }
        RegexMatchKind::Bytes(match_value) => {
            let group_index = resolve_bytes_group_index(match_value, selector)?;
            value_from_optional_bytes(vm, match_value.group(group_index)?)
        }
    }
}

fn match_lastindex_value(match_value: &RegexMatchObject) -> Value {
    let lastindex = match &match_value.match_value {
        RegexMatchKind::Text(match_value) => match_value.lastindex(),
        RegexMatchKind::Bytes(match_value) => match_value.lastindex(),
    };
    match lastindex {
        Some(index) => {
            Value::int(index as i64).expect("regex group index should fit in Value::int")
        }
        None => Value::none(),
    }
}

fn match_lastgroup_value(
    vm: &mut VirtualMachine,
    lastgroup: Option<&str>,
) -> Result<Value, RuntimeError> {
    match lastgroup {
        Some(name) => alloc_runtime_value(
            vm,
            StringObject::from_string(name.to_string()),
            "regex lastgroup string",
        ),
        None => Ok(Value::none()),
    }
}

fn resolve_text_group_index(
    match_value: &Match,
    selector: &GroupSelector,
) -> Result<usize, BuiltinError> {
    match selector {
        GroupSelector::Index(index) => {
            if *index < match_value.len() {
                Ok(*index)
            } else {
                Err(BuiltinError::IndexError("no such group".to_string()))
            }
        }
        GroupSelector::Name(name) => match_value
            .group_index(name)
            .ok_or_else(|| BuiltinError::IndexError("no such group".to_string())),
    }
}

fn resolve_bytes_group_index(
    match_value: &BytesMatch,
    selector: &GroupSelector,
) -> Result<usize, BuiltinError> {
    match selector {
        GroupSelector::Index(index) => {
            if *index < match_value.len() {
                Ok(*index)
            } else {
                Err(BuiltinError::IndexError("no such group".to_string()))
            }
        }
        GroupSelector::Name(name) => match_value
            .group_index(name)
            .ok_or_else(|| BuiltinError::IndexError("no such group".to_string())),
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

fn sub_result_to_value(
    vm: &mut VirtualMachine,
    pattern: &RegexPatternObject,
    replacement: ReplacementValue,
    subject: SubjectValue,
    count: usize,
) -> Result<Value, BuiltinError> {
    match (&pattern.pattern, replacement, subject) {
        (
            RegexPatternKind::Text(pattern),
            ReplacementValue::Text(repl),
            SubjectValue::Text(text),
        ) => alloc_value(
            vm,
            StringObject::from_string(pattern.sub_n(&repl, &text, count)),
            "regex substitution string",
        ),
        (
            RegexPatternKind::Bytes(pattern),
            ReplacementValue::Bytes(repl),
            SubjectValue::Bytes(bytes),
        ) => alloc_value(
            vm,
            BytesObject::from_slice(&pattern.sub_n(&repl, &bytes, count)),
            "regex substitution bytes",
        ),
        (
            RegexPatternKind::Text(pattern),
            ReplacementValue::Callable(callable),
            SubjectValue::Text(text),
        ) => {
            let (result, _) = substitute_text_callable(vm, pattern, callable, &text, count)?;
            alloc_value(
                vm,
                StringObject::from_string(result),
                "regex substitution string",
            )
        }
        (
            RegexPatternKind::Bytes(pattern),
            ReplacementValue::Callable(callable),
            SubjectValue::Bytes(bytes),
        ) => {
            let (result, _) = substitute_bytes_callable(vm, pattern, callable, &bytes, count)?;
            alloc_value(
                vm,
                BytesObject::from_slice(&result),
                "regex substitution bytes",
            )
        }
        _ => Err(BuiltinError::TypeError(
            "pattern and replacement must be the same type as the search text".to_string(),
        )),
    }
}

fn subn_result_to_value(
    vm: &mut VirtualMachine,
    pattern: &RegexPatternObject,
    replacement: ReplacementValue,
    subject: SubjectValue,
    count: usize,
) -> Result<Value, BuiltinError> {
    let (result, replacements) = match (&pattern.pattern, replacement, subject) {
        (
            RegexPatternKind::Text(pattern),
            ReplacementValue::Text(repl),
            SubjectValue::Text(text),
        ) => {
            let (result, replacements) = pattern.subn(&repl, &text, count);
            (
                alloc_value(
                    vm,
                    StringObject::from_string(result),
                    "regex substitution string",
                )?,
                replacements,
            )
        }
        (
            RegexPatternKind::Bytes(pattern),
            ReplacementValue::Bytes(repl),
            SubjectValue::Bytes(bytes),
        ) => {
            let (result, replacements) = pattern.subn(&repl, &bytes, count);
            (
                alloc_value(
                    vm,
                    BytesObject::from_slice(&result),
                    "regex substitution bytes",
                )?,
                replacements,
            )
        }
        (
            RegexPatternKind::Text(pattern),
            ReplacementValue::Callable(callable),
            SubjectValue::Text(text),
        ) => {
            let (result, replacements) =
                substitute_text_callable(vm, pattern, callable, &text, count)?;
            (
                alloc_value(
                    vm,
                    StringObject::from_string(result),
                    "regex substitution string",
                )?,
                replacements,
            )
        }
        (
            RegexPatternKind::Bytes(pattern),
            ReplacementValue::Callable(callable),
            SubjectValue::Bytes(bytes),
        ) => {
            let (result, replacements) =
                substitute_bytes_callable(vm, pattern, callable, &bytes, count)?;
            (
                alloc_value(
                    vm,
                    BytesObject::from_slice(&result),
                    "regex substitution bytes",
                )?,
                replacements,
            )
        }
        _ => {
            return Err(BuiltinError::TypeError(
                "pattern and replacement must be the same type as the search text".to_string(),
            ));
        }
    };

    alloc_value(
        vm,
        TupleObject::from_vec(vec![
            result,
            Value::int(replacements as i64).expect("substitution count should fit"),
        ]),
        "regex substitution tuple",
    )
}

fn substitute_text_callable(
    vm: &mut VirtualMachine,
    pattern: &CompiledPattern,
    callable: Value,
    text: &str,
    count: usize,
) -> Result<(String, usize), BuiltinError> {
    let limit = if count == 0 { usize::MAX } else { count };
    let matches = pattern.findall(text);
    let mut result = String::with_capacity(text.len());
    let mut last_end = 0usize;
    let mut replacements = 0usize;

    for match_value in matches.into_iter().take(limit) {
        let start = match_value.start();
        let end = match_value.end();
        result.push_str(&text[last_end..start]);

        let match_object = alloc_value(
            vm,
            RegexMatchObject::new(RegexMatchKind::Text(match_value)),
            "regex match",
        )?;
        let replacement = invoke_callable_value(vm, callable, &[match_object])
            .map_err(runtime_error_to_builtin_error)?;
        match parse_text_or_bytes(replacement, "repl")? {
            SubjectValue::Text(text) => result.push_str(&text),
            SubjectValue::Bytes(_) => {
                return Err(BuiltinError::TypeError(
                    "sequence item 0: expected str instance, bytes found".to_string(),
                ));
            }
        }

        last_end = end;
        replacements += 1;
    }

    result.push_str(&text[last_end..]);
    Ok((result, replacements))
}

fn substitute_bytes_callable(
    vm: &mut VirtualMachine,
    pattern: &CompiledBytesPattern,
    callable: Value,
    bytes: &[u8],
    count: usize,
) -> Result<(Vec<u8>, usize), BuiltinError> {
    let limit = if count == 0 { usize::MAX } else { count };
    let matches = pattern.findall(bytes);
    let mut result = Vec::with_capacity(bytes.len());
    let mut last_end = 0usize;
    let mut replacements = 0usize;

    for match_value in matches.into_iter().take(limit) {
        let start = usize::try_from(match_value.start(0)?)
            .expect("matched byte start should be non-negative");
        let end =
            usize::try_from(match_value.end(0)?).expect("matched byte end should be non-negative");
        result.extend_from_slice(&bytes[last_end..start]);

        let match_object = alloc_value(
            vm,
            RegexMatchObject::new(RegexMatchKind::Bytes(match_value)),
            "regex match",
        )?;
        let replacement = invoke_callable_value(vm, callable, &[match_object])
            .map_err(runtime_error_to_builtin_error)?;
        match parse_text_or_bytes(replacement, "repl")? {
            SubjectValue::Bytes(bytes) => result.extend_from_slice(&bytes),
            SubjectValue::Text(_) => {
                return Err(BuiltinError::TypeError(
                    "sequence item 0: expected a bytes-like object, str found".to_string(),
                ));
            }
        }

        last_end = end;
        replacements += 1;
    }

    result.extend_from_slice(&bytes[last_end..]);
    Ok((result, replacements))
}

fn split_result_to_value(
    vm: &mut VirtualMachine,
    pattern: &RegexPatternObject,
    subject: SubjectValue,
    maxsplit: usize,
) -> Result<Value, BuiltinError> {
    let items = match (&pattern.pattern, subject) {
        (RegexPatternKind::Text(pattern), SubjectValue::Text(text)) => {
            let mut items = Vec::new();
            for part in pattern.split_n(&text, maxsplit) {
                items.push(alloc_value(
                    vm,
                    StringObject::from_string(part),
                    "regex split part",
                )?);
            }
            items
        }
        (RegexPatternKind::Bytes(pattern), SubjectValue::Bytes(bytes)) => {
            let mut items = Vec::new();
            for part in pattern.split_n(&bytes, maxsplit) {
                items.push(alloc_value(
                    vm,
                    BytesObject::from_slice(&part),
                    "regex split part",
                )?);
            }
            items
        }
        _ => {
            return Err(BuiltinError::TypeError(
                "pattern and search text must be the same type".to_string(),
            ));
        }
    };

    alloc_value(vm, ListObject::from_iter(items), "regex split list")
}

fn alloc_value<T: Trace + 'static>(
    _vm: &mut VirtualMachine,
    value: T,
    _context: &'static str,
) -> Result<Value, BuiltinError> {
    Ok(alloc_value_in_current_heap_or_box(value))
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

fn alloc_runtime_value<T: Trace + 'static>(
    _vm: &mut VirtualMachine,
    value: T,
    _context: &'static str,
) -> Result<Value, RuntimeError> {
    Ok(alloc_value_in_current_heap_or_box(value))
}

fn try_parse_python_string(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        return prism_core::intern::interned_by_ptr(ptr as *const u8)
            .map(|interned| interned.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::STR {
        return None;
    }

    Some(
        unsafe { &*(ptr as *const StringObject) }
            .as_str()
            .to_string(),
    )
}

fn parse_bound_pattern_args(
    args: &[Value],
    method_name: &'static str,
) -> Result<(&'static RegexPatternObject, SubjectValue, SearchBounds), BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "Pattern.{method_name}() takes from 1 to 3 arguments ({} given)",
            args.len().saturating_sub(1),
        )));
    }

    let pattern = expect_pattern_ref(args[0], method_name)?;
    let subject = parse_subject_for_pattern(args[1], pattern, method_name)?;
    let bounds = parse_search_bounds(args.get(2).copied(), args.get(3).copied())?;
    Ok((pattern, subject, bounds))
}

fn parse_bound_pattern_sub_args(
    args: &[Value],
    method_name: &'static str,
) -> Result<
    (
        &'static RegexPatternObject,
        ReplacementValue,
        SubjectValue,
        usize,
    ),
    BuiltinError,
> {
    if args.len() < 3 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "Pattern.{method_name}() takes from 2 to 3 arguments ({} given)",
            args.len().saturating_sub(1),
        )));
    }

    let pattern = expect_pattern_ref(args[0], method_name)?;
    let replacement = parse_replacement_for_pattern(args[1], pattern)?;
    let subject = parse_subject_for_pattern(args[2], pattern, "string")?;
    let count = parse_substitution_limit(args.get(3).copied(), "count")?;
    Ok((pattern, replacement, subject, count))
}

fn parse_bound_pattern_split_args(
    args: &[Value],
) -> Result<(&'static RegexPatternObject, SubjectValue, usize), BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "Pattern.split() takes from 1 to 2 arguments ({} given)",
            args.len().saturating_sub(1),
        )));
    }

    let pattern = expect_pattern_ref(args[0], "split")?;
    let subject = parse_subject_for_pattern(args[1], pattern, "string")?;
    let maxsplit = parse_substitution_limit(args.get(2).copied(), "maxsplit")?;
    Ok((pattern, subject, maxsplit))
}

fn bounded_bytes_range(text: &[u8], pos: usize, endpos: Option<usize>) -> Option<(&[u8], usize)> {
    let text_len = text.len();
    let pos = pos.min(text_len);
    let end = endpos.unwrap_or(text_len).min(text_len);
    if pos > end {
        return None;
    }
    Some((&text[pos..end], pos))
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

    let match_value = expect_match_ref(args[0], method_name)?;
    let selector = parse_group_selector(args.get(1).copied())?;
    let position = match &match_value.match_value {
        RegexMatchKind::Text(match_value) => text(
            match_value,
            resolve_text_group_index(match_value, &selector)?,
        )?,
        RegexMatchKind::Bytes(match_value) => bytes(
            match_value,
            resolve_bytes_group_index(match_value, &selector)?,
        )?,
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
    use crate::builtins::get_iterator_mut;
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

    fn bytes_object_value(bytes: &[u8]) -> Value {
        Value::object_ptr(Box::into_raw(Box::new(BytesObject::from_slice(bytes))) as *const ())
    }

    fn tuple_values(value: Value) -> Vec<Value> {
        let ptr = value.as_object_ptr().expect("expected tuple object");
        unsafe { &*(ptr as *const TupleObject) }.as_slice().to_vec()
    }

    fn list_values(value: Value) -> Vec<Value> {
        let ptr = value.as_object_ptr().expect("expected list object");
        unsafe { &*(ptr as *const ListObject) }.as_slice().to_vec()
    }

    fn dict_entries(value: Value) -> Vec<(Value, Value)> {
        let ptr = value.as_object_ptr().expect("expected dict object");
        unsafe { &*(ptr as *const DictObject) }.iter().collect()
    }

    fn exhaust_nursery(vm: &VirtualMachine) {
        while vm.allocator().alloc(DictObject::new()).is_some() {}
    }

    #[test]
    fn test_pattern_search_allocates_match_after_full_nursery() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("compile should succeed");

        exhaust_nursery(&vm);

        let searched =
            builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc123def"))])
                .expect("search should allocate a match after nursery exhaustion");
        assert!(!searched.is_none());

        let group = builtin_match_group(&mut vm, &[searched]).expect("group should allocate");
        assert_eq!(string_value(group), "123");
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
    fn test_compile_bytes_pattern_uses_fancy_engine_for_lookahead() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[bytes_object_value(br"\n(?![ \t])|\r(?![ \t\n])")],
        )
        .expect("bytes lookahead pattern should compile");

        let legal = builtin_pattern_search(&mut vm, &[pattern, bytes_object_value(b"ok\n value")])
            .expect("bytes search should run");
        assert!(legal.is_none(), "continuation whitespace should not match");

        let pattern = builtin_compile(
            &mut vm,
            &[bytes_object_value(br"\n(?![ \t])|\r(?![ \t\n])")],
        )
        .expect("bytes lookahead pattern should compile");
        let illegal =
            builtin_pattern_search(&mut vm, &[pattern, bytes_object_value(b"bad\nHeader")])
                .expect("bytes search should run");
        let group = builtin_match_group(&mut vm, &[illegal]).expect("group() should succeed");
        assert_eq!(bytes_value(group), b"\n");
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

    #[test]
    fn test_match_group_accepts_named_and_multiple_selectors() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::string(intern(r"(?P<lhs>\w+)=(?P<rhs>\w+)"))],
        )
        .expect("compile should succeed");
        let matched =
            builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("alpha=beta"))])
                .expect("search should succeed");

        let lhs = builtin_match_group(&mut vm, &[matched, Value::string(intern("lhs"))])
            .expect("named group lookup should succeed");
        assert_eq!(string_value(lhs), "alpha");

        let values = builtin_match_group(
            &mut vm,
            &[
                matched,
                Value::string(intern("lhs")),
                Value::int(2).expect("group index should fit"),
            ],
        )
        .expect("mixed group lookup should succeed");
        let values = tuple_values(values);
        assert_eq!(values.len(), 2);
        assert_eq!(string_value(values[0]), "alpha");
        assert_eq!(string_value(values[1]), "beta");
    }

    #[test]
    fn test_match_positions_accept_named_groups() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::string(intern(r"(?P<indent>\s+)(?P<name>\w+)"))],
        )
        .expect("compile should succeed");
        let matched =
            builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("   prism"))])
                .expect("search should succeed");

        let start = builtin_match_start(&[matched, Value::string(intern("name"))])
            .expect("named start should succeed");
        let end = builtin_match_end(&[matched, Value::string(intern("name"))])
            .expect("named end should succeed");
        let span = builtin_match_span(&mut vm, &[matched, Value::string(intern("name"))])
            .expect("named span should succeed");

        assert_eq!(start.as_int(), Some(3));
        assert_eq!(end.as_int(), Some(8));
        let span = tuple_values(span);
        assert_eq!(span[0].as_int(), Some(3));
        assert_eq!(span[1].as_int(), Some(8));
    }

    #[test]
    fn test_pattern_groupindex_exposes_named_capture_mapping() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::string(intern(r"(?P<first>a)(?P<second>b)"))],
        )
        .expect("compile should succeed");

        let groupindex = pattern_attr_value(&mut vm, pattern, &intern("groupindex"))
            .expect("groupindex lookup should succeed")
            .expect("groupindex attribute should exist");
        let entries = dict_entries(groupindex);
        assert_eq!(entries.len(), 2);

        let first = entries
            .iter()
            .find(|(key, _)| string_value(*key) == "first")
            .map(|(_, value)| *value)
            .expect("first group should exist");
        let second = entries
            .iter()
            .find(|(key, _)| string_value(*key) == "second")
            .map(|(_, value)| *value)
            .expect("second group should exist");

        assert_eq!(first.as_int(), Some(1));
        assert_eq!(second.as_int(), Some(2));
    }

    #[test]
    fn test_match_lastindex_and_lastgroup_attributes_follow_cpython() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::string(intern(r"(?P<word>\w+)(?:-(\d+))?"))],
        )
        .expect("compile should succeed");

        let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc"))])
            .expect("search should succeed");

        let lastindex = match_attr_value(&mut vm, matched, &intern("lastindex"))
            .expect("lastindex lookup should succeed")
            .expect("lastindex attribute should exist");
        let lastgroup = match_attr_value(&mut vm, matched, &intern("lastgroup"))
            .expect("lastgroup lookup should succeed")
            .expect("lastgroup attribute should exist");

        assert_eq!(lastindex.as_int(), Some(1));
        assert_eq!(string_value(lastgroup), "word");

        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\w+"))])
            .expect("compile should succeed");
        let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc"))])
            .expect("search should succeed");

        let lastindex = match_attr_value(&mut vm, matched, &intern("lastindex"))
            .expect("lastindex lookup should succeed")
            .expect("lastindex attribute should exist");
        let lastgroup = match_attr_value(&mut vm, matched, &intern("lastgroup"))
            .expect("lastgroup lookup should succeed")
            .expect("lastgroup attribute should exist");

        assert!(lastindex.is_none());
        assert!(lastgroup.is_none());
    }

    #[test]
    fn test_match_getitem_indexes_capture_groups_like_group() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::string(intern(r"(?P<word>\w+)-(?P<num>\d+)"))],
        )
        .expect("compile should succeed");
        let matched = builtin_pattern_search(&mut vm, &[pattern, Value::string(intern("abc-123"))])
            .expect("search should succeed");

        let whole = builtin_match_getitem(
            &mut vm,
            &[
                matched,
                Value::int(0).expect("index should fit in Value::int"),
            ],
        )
        .expect("getitem should return whole match");
        let first = builtin_match_getitem(
            &mut vm,
            &[
                matched,
                Value::int(1).expect("index should fit in Value::int"),
            ],
        )
        .expect("getitem should return numeric group");
        let named = builtin_match_getitem(&mut vm, &[matched, Value::string(intern("num"))])
            .expect("getitem should return named group");

        assert_eq!(string_value(whole), "abc-123");
        assert_eq!(string_value(first), "abc");
        assert_eq!(string_value(named), "123");

        let pattern = builtin_compile(
            &mut vm,
            &[Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(
                    br"(?P<word>\w+)-(?P<num>\d+)",
                ))) as *const (),
            )],
        )
        .expect("bytes compile should succeed");
        let matched = builtin_pattern_search(
            &mut vm,
            &[
                pattern,
                Value::object_ptr(
                    Box::into_raw(Box::new(BytesObject::from_slice(b"abc-123"))) as *const ()
                ),
            ],
        )
        .expect("bytes search should succeed");

        let whole = builtin_match_getitem(
            &mut vm,
            &[
                matched,
                Value::int(0).expect("index should fit in Value::int"),
            ],
        )
        .expect("bytes getitem should return whole match");
        let named = builtin_match_getitem(&mut vm, &[matched, Value::string(intern("num"))])
            .expect("bytes getitem should return named group");

        assert_eq!(bytes_value(whole), b"abc-123");
        assert_eq!(bytes_value(named), b"123");
    }

    #[test]
    fn test_pattern_match_accepts_pos_and_rebases_span() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("compile should succeed");

        let matched = builtin_pattern_match(
            &mut vm,
            &[
                pattern,
                Value::string(intern("abc123def")),
                Value::int(3).expect("pos should fit in Value::int"),
            ],
        )
        .expect("pattern.match should accept pos");

        let group = builtin_match_group(&mut vm, &[matched]).expect("group() should succeed");
        assert_eq!(string_value(group), "123");
        assert_eq!(
            builtin_match_start(&[matched])
                .expect("start() should succeed")
                .as_int(),
            Some(3)
        );
        assert_eq!(
            builtin_match_end(&[matched])
                .expect("end() should succeed")
                .as_int(),
            Some(6)
        );
    }

    #[test]
    fn test_pattern_fullmatch_accepts_pos_and_endpos() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("compile should succeed");

        let matched = builtin_pattern_fullmatch(
            &mut vm,
            &[
                pattern,
                Value::string(intern("abc123def")),
                Value::int(3).expect("pos should fit in Value::int"),
                Value::int(6).expect("endpos should fit in Value::int"),
            ],
        )
        .expect("pattern.fullmatch should accept bounds");

        let group = builtin_match_group(&mut vm, &[matched]).expect("group() should succeed");
        assert_eq!(string_value(group), "123");
        let span = builtin_match_span(&mut vm, &[matched]).expect("span() should succeed");
        let span = tuple_values(span);
        assert_eq!(span[0].as_int(), Some(3));
        assert_eq!(span[1].as_int(), Some(6));
    }

    #[test]
    fn test_bytes_match_named_groups_support_group_and_groupdict() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(
                    br"(?P<lhs>\w+)=(?P<rhs>\w+)",
                ))) as *const (),
            )],
        )
        .expect("bytes compile should succeed");
        let matched = builtin_pattern_search(
            &mut vm,
            &[
                pattern,
                Value::object_ptr(
                    Box::into_raw(Box::new(BytesObject::from_slice(b"left=right"))) as *const (),
                ),
            ],
        )
        .expect("bytes search should succeed");

        let lhs = builtin_match_group(&mut vm, &[matched, Value::string(intern("lhs"))])
            .expect("bytes named group lookup should succeed");
        assert_eq!(bytes_value(lhs), b"left");

        let groupdict =
            builtin_match_groupdict(&mut vm, &[matched]).expect("bytes groupdict should succeed");
        let entries = dict_entries(groupdict);
        assert_eq!(entries.len(), 2);

        let lhs = entries
            .iter()
            .find(|(key, _)| string_value(*key) == "lhs")
            .map(|(_, value)| *value)
            .expect("lhs entry should exist");
        let rhs = entries
            .iter()
            .find(|(key, _)| string_value(*key) == "rhs")
            .map(|(_, value)| *value)
            .expect("rhs entry should exist");
        assert_eq!(bytes_value(lhs), b"left");
        assert_eq!(bytes_value(rhs), b"right");

        let lastindex = match_attr_value(&mut vm, matched, &intern("lastindex"))
            .expect("lastindex lookup should succeed")
            .expect("lastindex attribute should exist");
        let lastgroup = match_attr_value(&mut vm, matched, &intern("lastgroup"))
            .expect("lastgroup lookup should succeed")
            .expect("lastgroup attribute should exist");

        assert_eq!(lastindex.as_int(), Some(2));
        assert_eq!(string_value(lastgroup), "rhs");
    }

    #[test]
    fn test_bytes_pattern_search_accepts_pos_and_rebases_span() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(br"\d+"))) as *const (),
            )],
        )
        .expect("bytes compile should succeed");

        let matched = builtin_pattern_search(
            &mut vm,
            &[
                pattern,
                Value::object_ptr(
                    Box::into_raw(Box::new(BytesObject::from_slice(b"abc123def456"))) as *const (),
                ),
                Value::int(6).expect("pos should fit in Value::int"),
            ],
        )
        .expect("bytes search should accept pos");

        let group = builtin_match_group(&mut vm, &[matched]).expect("group() should succeed");
        assert_eq!(bytes_value(group), b"456");
        assert_eq!(
            builtin_match_start(&[matched])
                .expect("start() should succeed")
                .as_int(),
            Some(9)
        );
        assert_eq!(
            builtin_match_end(&[matched])
                .expect("end() should succeed")
                .as_int(),
            Some(12)
        );
    }

    #[test]
    fn test_pattern_findall_returns_full_matches_without_groups() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("compile should succeed");

        let matches = builtin_pattern_findall(&mut vm, &[pattern, Value::string(intern("a1b22"))])
            .expect("findall should succeed");
        let values = list_values(matches);
        assert_eq!(values.len(), 2);
        assert_eq!(string_value(values[0]), "1");
        assert_eq!(string_value(values[1]), "22");
    }

    #[test]
    fn test_pattern_findall_returns_capture_values_for_single_group() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"([a-z]+)?=(\d+)"))])
            .expect("compile should succeed");
        let matches =
            builtin_pattern_findall(&mut vm, &[pattern, Value::string(intern("foo=1 =2 bar=3"))])
                .expect("findall should succeed");

        let values = list_values(matches);
        assert_eq!(values.len(), 3);

        let first = tuple_values(values[0]);
        let second = tuple_values(values[1]);
        let third = tuple_values(values[2]);
        assert_eq!(
            first.into_iter().map(string_value).collect::<Vec<_>>(),
            vec!["foo".to_string(), "1".to_string()]
        );
        assert_eq!(
            second.into_iter().map(string_value).collect::<Vec<_>>(),
            vec!["".to_string(), "2".to_string()]
        );
        assert_eq!(
            third.into_iter().map(string_value).collect::<Vec<_>>(),
            vec!["bar".to_string(), "3".to_string()]
        );
    }

    #[test]
    fn test_pattern_findall_returns_plain_strings_for_one_group() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"(\d+)"))])
            .expect("compile should succeed");

        let matches =
            builtin_pattern_findall(&mut vm, &[pattern, Value::string(intern("x7 y88 z999"))])
                .expect("findall should succeed");
        let values = list_values(matches);
        assert_eq!(values.len(), 3);
        assert_eq!(string_value(values[0]), "7");
        assert_eq!(string_value(values[1]), "88");
        assert_eq!(string_value(values[2]), "999");
    }

    #[test]
    fn test_pattern_findall_returns_bytes_for_bytes_patterns() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(
            &mut vm,
            &[Value::object_ptr(
                Box::into_raw(Box::new(BytesObject::from_slice(br"(\w+)-(\d+)"))) as *const (),
            )],
        )
        .expect("bytes compile should succeed");

        let matches = builtin_pattern_findall(
            &mut vm,
            &[
                pattern,
                Value::object_ptr(
                    Box::into_raw(Box::new(BytesObject::from_slice(b"abc-1 def-22"))) as *const (),
                ),
            ],
        )
        .expect("bytes findall should succeed");
        let values = list_values(matches);
        assert_eq!(values.len(), 2);

        let first = tuple_values(values[0]);
        let second = tuple_values(values[1]);
        assert_eq!(bytes_value(first[0]), b"abc");
        assert_eq!(bytes_value(first[1]), b"1");
        assert_eq!(bytes_value(second[0]), b"def");
        assert_eq!(bytes_value(second[1]), b"22");
    }

    #[test]
    fn test_pattern_findall_accepts_pos_and_endpos() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("compile should succeed");

        let matches = builtin_pattern_findall(
            &mut vm,
            &[
                pattern,
                Value::string(intern("a1 b22 c333")),
                Value::int(3).expect("pos should fit in Value::int"),
                Value::int(7).expect("endpos should fit in Value::int"),
            ],
        )
        .expect("findall should accept bounds");
        let values = list_values(matches);
        assert_eq!(values.len(), 1);
        assert_eq!(string_value(values[0]), "22");
    }

    #[test]
    fn test_module_findall_entrypoint_honors_flags() {
        let mut vm = VirtualMachine::new();
        let matches = builtin_findall(
            &mut vm,
            &[
                Value::string(intern(r"^hello")),
                Value::string(intern("Hello\nhello")),
                Value::int(RegexFlags::IGNORECASE as i64).unwrap(),
            ],
        )
        .expect("module findall should succeed");
        let values = list_values(matches);
        assert_eq!(values.len(), 1);
        assert_eq!(string_value(values[0]), "Hello");
    }

    #[test]
    fn test_pattern_finditer_returns_iterator_of_match_objects() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("compile should succeed");

        let iterator =
            builtin_pattern_finditer(&mut vm, &[pattern, Value::string(intern("a1 b22 c333"))])
                .expect("finditer should succeed");
        let iter = get_iterator_mut(&iterator).expect("finditer should return an iterator");

        let first = builtin_match_group(&mut vm, &[iter.next().expect("first match should exist")])
            .expect("group should work");
        let second =
            builtin_match_group(&mut vm, &[iter.next().expect("second match should exist")])
                .expect("group should work");
        let third = builtin_match_group(&mut vm, &[iter.next().expect("third match should exist")])
            .expect("group should work");
        assert_eq!(string_value(first), "1");
        assert_eq!(string_value(second), "22");
        assert_eq!(string_value(third), "333");
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_pattern_finditer_accepts_pos_and_rebases_spans() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("compile should succeed");

        let iterator = builtin_pattern_finditer(
            &mut vm,
            &[
                pattern,
                Value::string(intern("a1 b22 c333")),
                Value::int(3).expect("pos should fit in Value::int"),
            ],
        )
        .expect("finditer should accept pos");
        let iter = get_iterator_mut(&iterator).expect("finditer should return an iterator");

        let first = iter.next().expect("first match should exist");
        let second = iter.next().expect("second match should exist");
        assert!(iter.next().is_none());

        let first_group = builtin_match_group(&mut vm, &[first]).expect("group() should work");
        let second_group = builtin_match_group(&mut vm, &[second]).expect("group() should work");
        assert_eq!(string_value(first_group), "22");
        assert_eq!(string_value(second_group), "333");
        assert_eq!(
            builtin_match_start(&[first])
                .expect("start() should succeed")
                .as_int(),
            Some(4)
        );
        assert_eq!(
            builtin_match_start(&[second])
                .expect("start() should succeed")
                .as_int(),
            Some(8)
        );
    }

    #[test]
    fn test_module_finditer_entrypoint_returns_iterable_matches() {
        let mut vm = VirtualMachine::new();
        let iterator = builtin_finditer(
            &mut vm,
            &[
                Value::string(intern(r"[a-z]+")),
                Value::string(intern("ab  cd")),
            ],
        )
        .expect("module finditer should succeed");
        let iter = get_iterator_mut(&iterator).expect("finditer should return an iterator");

        let first = builtin_match_group(&mut vm, &[iter.next().expect("first match should exist")])
            .expect("group should work");
        let second =
            builtin_match_group(&mut vm, &[iter.next().expect("second match should exist")])
                .expect("group should work");
        assert_eq!(string_value(first), "ab");
        assert_eq!(string_value(second), "cd");
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_module_sub_entrypoint_replaces_matches() {
        let mut vm = VirtualMachine::new();
        let result = builtin_sub(
            &mut vm,
            &[
                Value::string(intern(r"\d+")),
                Value::string(intern("X")),
                Value::string(intern("a1b22c333")),
            ],
        )
        .expect("module sub should succeed");
        assert_eq!(string_value(result), "aXbXcX");
    }

    #[test]
    fn test_module_sub_expands_python_numeric_backreferences() {
        let mut vm = VirtualMachine::new();
        let result = builtin_sub(
            &mut vm,
            &[
                Value::string(intern(r#"\\([\\\$"'`])"#)),
                Value::string(intern(r"\1")),
                Value::string(intern(r#"\$\`\\\'\""#)),
            ],
        )
        .expect("module sub should expand Python replacement backrefs");
        assert_eq!(string_value(result), "$`\\'\"");
    }

    #[test]
    fn test_module_sub_escapes_literal_dollars_for_rust_regex_backend() {
        let mut vm = VirtualMachine::new();
        let result = builtin_sub(
            &mut vm,
            &[
                Value::string(intern(r"(a)(b)")),
                Value::string(intern(r"\2-\1-$")),
                Value::string(intern("ab")),
            ],
        )
        .expect("module sub should preserve literal dollars");
        assert_eq!(string_value(result), "b-a-$");
    }

    #[test]
    fn test_module_subn_entrypoint_returns_result_and_count() {
        let mut vm = VirtualMachine::new();
        let result = builtin_subn(
            &mut vm,
            &[
                Value::string(intern(r"\d+")),
                Value::string(intern("X")),
                Value::string(intern("a1b22c333")),
                Value::int(2).expect("count should fit"),
            ],
        )
        .expect("module subn should succeed");
        let values = tuple_values(result);
        assert_eq!(values.len(), 2);
        assert_eq!(string_value(values[0]), "aXbXc333");
        assert_eq!(values[1].as_int(), Some(2));
    }

    #[test]
    fn test_module_split_entrypoint_returns_parts() {
        let mut vm = VirtualMachine::new();
        let result = builtin_split(
            &mut vm,
            &[
                Value::string(intern(r",\s*")),
                Value::string(intern("a, b,  c")),
            ],
        )
        .expect("module split should succeed");
        let values = list_values(result);
        assert_eq!(values.len(), 3);
        assert_eq!(string_value(values[0]), "a");
        assert_eq!(string_value(values[1]), "b");
        assert_eq!(string_value(values[2]), "c");
    }

    #[test]
    fn test_pattern_sub_entrypoint_returns_replaced_text() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("pattern compile should succeed");
        let result = builtin_pattern_sub(
            &mut vm,
            &[
                pattern,
                Value::string(intern("X")),
                Value::string(intern("a1b22c333")),
            ],
        )
        .expect("pattern sub should succeed");
        assert_eq!(string_value(result), "aXbXcX");
    }

    #[test]
    fn test_pattern_subn_entrypoint_returns_result_and_count() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r"\d+"))])
            .expect("pattern compile should succeed");
        let result = builtin_pattern_subn(
            &mut vm,
            &[
                pattern,
                Value::string(intern("X")),
                Value::string(intern("a1b22c333")),
                Value::int(2).expect("count should fit"),
            ],
        )
        .expect("pattern subn should succeed");
        let values = tuple_values(result);
        assert_eq!(values.len(), 2);
        assert_eq!(string_value(values[0]), "aXbXc333");
        assert_eq!(values[1].as_int(), Some(2));
    }

    #[test]
    fn test_pattern_split_entrypoint_returns_parts() {
        let mut vm = VirtualMachine::new();
        let pattern = builtin_compile(&mut vm, &[Value::string(intern(r",\s*"))])
            .expect("pattern compile should succeed");
        let result = builtin_pattern_split(&mut vm, &[pattern, Value::string(intern("a, b,  c"))])
            .expect("pattern split should succeed");
        let values = list_values(result);
        assert_eq!(values.len(), 3);
        assert_eq!(string_value(values[0]), "a");
        assert_eq!(string_value(values[1]), "b");
        assert_eq!(string_value(values[2]), "c");
    }

    #[test]
    fn test_module_sub_entrypoint_supports_bytes_patterns() {
        let mut vm = VirtualMachine::new();
        let result = builtin_sub(
            &mut vm,
            &[
                Value::object_ptr(
                    Box::into_raw(Box::new(BytesObject::from_slice(br"\d+"))) as *const ()
                ),
                Value::object_ptr(
                    Box::into_raw(Box::new(BytesObject::from_slice(b"X"))) as *const ()
                ),
                Value::object_ptr(
                    Box::into_raw(Box::new(BytesObject::from_slice(b"a1b22c333"))) as *const (),
                ),
            ],
        )
        .expect("module bytes sub should succeed");
        assert_eq!(bytes_value(result), b"aXbXcX");
    }
}
