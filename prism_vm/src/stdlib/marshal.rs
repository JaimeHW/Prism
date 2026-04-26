//! Native `marshal` module bootstrap surface.
//!
//! Prism uses this native module to cover the scalar marshal semantics that
//! CPython's regression suite relies on during early compatibility work. The
//! codec is intentionally structured so more marshal tags can be added without
//! rewriting the public module surface.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use prism_code::{
    CodeFlags, CodeObject, Constant, ExceptionEntry, Instruction, KwNamesTuple, LineTableEntry,
    Opcode,
};
use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::CodeObjectView;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::{bigint_to_value, is_int_value, value_to_bigint};
use prism_runtime::types::memoryview::value_as_memoryview_ref;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, LazyLock};

const MARSHAL_VERSION: i64 = 4;

const TYPE_NONE: u8 = b'N';
const TYPE_FALSE: u8 = b'F';
const TYPE_TRUE: u8 = b'T';
const TYPE_INT: u8 = b'i';
const TYPE_LONG: u8 = b'l';
const TYPE_STRING: u8 = b's';
const TYPE_UNICODE: u8 = b'u';
const TYPE_FLOAT: u8 = b'g';
const TYPE_TUPLE: u8 = b'(';
const TYPE_CODE: u8 = b'c';
const TYPE_CODE_REF: u8 = b'r';
const TYPE_KWNAMES: u8 = b'K';

const PRISM_CODE_MARKER: &[u8; 8] = b"PRSMCOD1";

static DUMPS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("marshal.dumps"), marshal_dumps));
static LOADS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("marshal.loads"), marshal_loads));

/// Native `marshal` module descriptor.
#[derive(Debug, Clone)]
pub struct MarshalModule {
    attrs: Vec<Arc<str>>,
}

impl MarshalModule {
    /// Create a new `marshal` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("dumps"), Arc::from("loads"), Arc::from("version")],
        }
    }
}

impl Default for MarshalModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for MarshalModule {
    fn name(&self) -> &str {
        "marshal"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "dumps" => Ok(builtin_value(&DUMPS_FUNCTION)),
            "loads" => Ok(builtin_value(&LOADS_FUNCTION)),
            "version" => Ok(Value::int(MARSHAL_VERSION).expect("marshal version should fit")),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'marshal' has no attribute '{}'",
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

#[inline]
fn bytes_value(value: &[u8]) -> Value {
    crate::alloc_managed_value(BytesObject::from_slice(value))
}

fn marshal_dumps(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "dumps() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    if args.len() == 2 {
        value_to_version(args[1])?;
    }

    let mut buffer = Vec::new();
    encode_marshaled_value(args[0], &mut buffer)?;
    Ok(bytes_value(&buffer))
}

fn marshal_loads(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "loads() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let bytes = value_to_bytes_like(args[0], "loads() argument")?;
    decode_marshaled_value(&bytes)
}

fn value_to_version(value: Value) -> Result<i32, BuiltinError> {
    let bigint = value_to_bigint(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "marshal version must be int, not {}",
            value.type_name()
        ))
    })?;

    bigint.to_i32().ok_or_else(|| {
        BuiltinError::OverflowError("Python int too large to convert to C int".to_string())
    })
}

fn value_to_bytes_like(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        )));
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec()),
        TypeId::MEMORYVIEW => {
            let view = value_as_memoryview_ref(value)
                .ok_or_else(|| BuiltinError::TypeError("invalid memoryview object".to_string()))?;
            if view.released() {
                return Err(BuiltinError::ValueError(
                    "operation forbidden on released memoryview object".to_string(),
                ));
            }
            Ok(view.to_vec())
        }
        _ => Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        ))),
    }
}

fn encode_marshaled_value(value: Value, out: &mut Vec<u8>) -> Result<(), BuiltinError> {
    if value.is_none() {
        out.push(TYPE_NONE);
        return Ok(());
    }

    if let Some(boolean) = value.as_bool() {
        out.push(if boolean { TYPE_TRUE } else { TYPE_FALSE });
        return Ok(());
    }

    if is_int_value(value) {
        let bigint = value_to_bigint(value)
            .ok_or_else(|| BuiltinError::ValueError("unmarshallable object".to_string()))?;
        encode_int(&bigint, out);
        return Ok(());
    }

    if let Some(float) = value.as_float() {
        out.push(TYPE_FLOAT);
        out.extend_from_slice(&float.to_le_bytes());
        return Ok(());
    }

    if let Some(string) = value_as_string_ref(value) {
        let text = string.as_str();
        out.push(TYPE_UNICODE);
        write_len(text.len(), out)?;
        out.extend_from_slice(text.as_bytes());
        return Ok(());
    }

    if let Some(bytes) = value.as_object_ptr().and_then(byte_sequence_ref_from_ptr) {
        out.push(TYPE_STRING);
        write_len(bytes.len(), out)?;
        out.extend_from_slice(bytes.as_bytes());
        return Ok(());
    }

    if let Some(ptr) = value.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::CODE => {
                let code = unsafe { &*(ptr as *const CodeObjectView) };
                out.push(TYPE_CODE);
                encode_code_object(code.code(), out)?;
                return Ok(());
            }
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                out.push(TYPE_TUPLE);
                write_len(tuple.len(), out)?;
                for item in tuple.iter() {
                    encode_marshaled_value(*item, out)?;
                }
                return Ok(());
            }
            _ => {}
        }
    }

    Err(BuiltinError::ValueError(
        "unmarshallable object".to_string(),
    ))
}

fn byte_sequence_ref_from_ptr(ptr: *const ()) -> Option<&'static BytesObject> {
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => Some(unsafe { &*(ptr as *const BytesObject) }),
        _ => None,
    }
}

struct CodeEncodeContext {
    nested_code_indices: FxHashMap<usize, usize>,
    kwnames_indices: FxHashSet<usize>,
}

fn encode_code_object(code: &CodeObject, out: &mut Vec<u8>) -> Result<(), BuiltinError> {
    out.extend_from_slice(PRISM_CODE_MARKER);
    write_str(&code.name, out)?;
    write_str(&code.qualname, out)?;
    write_str(&code.filename, out)?;
    write_u32(code.first_lineno, out);

    write_u16(code.arg_count, out);
    write_u16(code.posonlyarg_count, out);
    write_u16(code.kwonlyarg_count, out);
    write_u16(code.register_count, out);
    write_u32(code.flags.bits(), out);

    write_str_array(&code.locals, out)?;
    write_str_array(&code.names, out)?;
    write_str_array(&code.freevars, out)?;
    write_str_array(&code.cellvars, out)?;

    write_len(code.instructions.len(), out)?;
    for instruction in code.instructions.iter() {
        write_u32(instruction.raw(), out);
    }

    write_len(code.line_table.len(), out)?;
    for entry in code.line_table.iter() {
        write_u32(entry.start_pc, out);
        write_u32(entry.end_pc, out);
        write_u32(entry.line, out);
    }

    write_len(code.exception_table.len(), out)?;
    for entry in code.exception_table.iter() {
        write_u32(entry.start_pc, out);
        write_u32(entry.end_pc, out);
        write_u32(entry.handler_pc, out);
        write_u32(entry.finally_pc, out);
        write_u16(entry.depth, out);
        write_u16(entry.exception_type_idx, out);
    }

    write_len(code.nested_code_objects.len(), out)?;
    for nested in code.nested_code_objects.iter() {
        encode_code_object(nested, out)?;
    }

    let context = CodeEncodeContext {
        nested_code_indices: nested_code_indices(code),
        kwnames_indices: keyword_name_constant_indices(code),
    };
    write_len(code.constants.len(), out)?;
    for (index, constant) in code.constants.iter().enumerate() {
        encode_code_constant(index, constant, &context, out)?;
    }

    Ok(())
}

fn nested_code_indices(code: &CodeObject) -> FxHashMap<usize, usize> {
    code.nested_code_objects
        .iter()
        .enumerate()
        .map(|(index, nested)| (Arc::as_ptr(nested) as usize, index))
        .collect()
}

fn keyword_name_constant_indices(code: &CodeObject) -> FxHashSet<usize> {
    let mut indices = FxHashSet::default();
    for instruction in code.instructions.iter() {
        if Opcode::from_u8(instruction.opcode()) == Some(Opcode::CallKwEx) {
            let index = u16::from(instruction.src1().0) | (u16::from(instruction.src2().0) << 8);
            indices.insert(usize::from(index));
        }
    }
    indices
}

fn encode_code_constant(
    index: usize,
    constant: &Constant,
    context: &CodeEncodeContext,
    out: &mut Vec<u8>,
) -> Result<(), BuiltinError> {
    match constant {
        Constant::BigInt(value) => {
            encode_int(value, out);
            Ok(())
        }
        Constant::Value(value) => encode_code_constant_value(index, *value, context, out),
    }
}

fn encode_code_constant_value(
    index: usize,
    value: Value,
    context: &CodeEncodeContext,
    out: &mut Vec<u8>,
) -> Result<(), BuiltinError> {
    if let Some(ptr) = value.as_object_ptr() {
        let ptr_key = ptr as usize;
        if let Some(&nested_index) = context.nested_code_indices.get(&ptr_key) {
            out.push(TYPE_CODE_REF);
            write_len(nested_index, out)?;
            return Ok(());
        }

        if context.kwnames_indices.contains(&index) {
            let kwnames = unsafe { &*(ptr as *const KwNamesTuple) };
            out.push(TYPE_KWNAMES);
            write_len(kwnames.len(), out)?;
            for name in kwnames.iter() {
                write_str(name, out)?;
            }
            return Ok(());
        }
    }

    encode_marshaled_value(value, out)
}

fn write_len(len: usize, out: &mut Vec<u8>) -> Result<(), BuiltinError> {
    let len = i32::try_from(len)
        .map_err(|_| BuiltinError::ValueError("unmarshallable object".to_string()))?;
    out.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn write_u16(value: u16, out: &mut Vec<u8>) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(value: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_str(value: &str, out: &mut Vec<u8>) -> Result<(), BuiltinError> {
    write_len(value.len(), out)?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_str_array(values: &[Arc<str>], out: &mut Vec<u8>) -> Result<(), BuiltinError> {
    write_len(values.len(), out)?;
    for value in values {
        write_str(value, out)?;
    }
    Ok(())
}

fn encode_int(value: &BigInt, out: &mut Vec<u8>) {
    if let Some(n) = value.to_i32() {
        out.push(TYPE_INT);
        out.extend_from_slice(&n.to_le_bytes());
        return;
    }

    out.push(TYPE_LONG);
    encode_long_digits(value, out);
}

fn encode_long_digits(value: &BigInt, out: &mut Vec<u8>) {
    const BASE_BITS: u32 = 15;
    const BASE: i32 = 1 << BASE_BITS;
    const BASE_MASK: i64 = (1 << BASE_BITS) - 1;

    let (sign, bytes) = value.to_bytes_le();
    let mut magnitude = BigInt::from_bytes_le(Sign::Plus, &bytes);
    let mut digits = Vec::new();
    while magnitude != BigInt::from(0_u8) {
        let digit = (&magnitude & BigInt::from(BASE_MASK))
            .to_i32()
            .expect("masked marshal digit should fit in i32");
        digits.push(u16::try_from(digit).expect("marshal digit should fit in u16"));
        magnitude >>= BASE_BITS;
    }

    let count = if sign == Sign::Minus {
        -(digits.len() as i32)
    } else {
        digits.len() as i32
    };
    out.extend_from_slice(&count.to_le_bytes());
    for digit in digits {
        let digit = i32::from(digit % u16::try_from(BASE).expect("base should fit u16"));
        out.extend_from_slice(&(digit as u16).to_le_bytes());
    }
}

fn decode_marshaled_value(bytes: &[u8]) -> Result<Value, BuiltinError> {
    let mut reader = MarshalReader::new(bytes);
    decode_value(&mut reader)
}

struct MarshalReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> MarshalReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_u8(&mut self, eof_message: &'static str) -> Result<u8, BuiltinError> {
        let byte = self
            .bytes
            .get(self.offset)
            .copied()
            .ok_or_else(|| BuiltinError::ValueError(eof_message.to_string()))?;
        self.offset += 1;
        Ok(byte)
    }

    fn read_exact(
        &mut self,
        len: usize,
        eof_message: &'static str,
    ) -> Result<&'a [u8], BuiltinError> {
        let end = self.offset.checked_add(len).ok_or_else(|| {
            BuiltinError::ValueError("bad marshal data (size out of range)".to_string())
        })?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| BuiltinError::ValueError(eof_message.to_string()))?;
        self.offset = end;
        Ok(slice)
    }

    fn read_i32(&mut self) -> Result<i32, BuiltinError> {
        let raw = self.read_exact(4, "EOF read where not expected")?;
        Ok(i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn read_u16(&mut self) -> Result<u16, BuiltinError> {
        let raw = self.read_exact(2, "EOF read where not expected")?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, BuiltinError> {
        let raw = self.read_exact(4, "EOF read where not expected")?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn read_f64(&mut self) -> Result<f64, BuiltinError> {
        let raw = self.read_exact(8, "EOF read where not expected")?;
        Ok(f64::from_le_bytes([
            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
        ]))
    }

    fn read_len(&mut self) -> Result<usize, BuiltinError> {
        let len = self.read_i32()?;
        if len < 0 {
            return Err(BuiltinError::ValueError(
                "bad marshal data (bytes object size out of range)".to_string(),
            ));
        }
        Ok(usize::try_from(len).expect("non-negative i32 should fit usize"))
    }

    fn read_payload_bytes(&mut self) -> Result<&'a [u8], BuiltinError> {
        let len = self.read_len()?;
        self.read_exact(len, "marshal data too short")
    }

    fn read_string(&mut self) -> Result<Arc<str>, BuiltinError> {
        let bytes = self.read_payload_bytes()?;
        let text = std::str::from_utf8(bytes).map_err(|_| {
            BuiltinError::ValueError("bad marshal data (invalid utf-8)".to_string())
        })?;
        Ok(Arc::from(text))
    }

    fn read_string_array(&mut self) -> Result<Box<[Arc<str>]>, BuiltinError> {
        let len = self.read_len()?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_string()?);
        }
        Ok(values.into_boxed_slice())
    }
}

fn decode_value(reader: &mut MarshalReader<'_>) -> Result<Value, BuiltinError> {
    let tag = reader.read_u8("EOF read where object expected")?;
    match tag {
        TYPE_NONE => Ok(Value::none()),
        TYPE_FALSE => Ok(Value::bool(false)),
        TYPE_TRUE => Ok(Value::bool(true)),
        TYPE_INT => {
            let n = reader.read_i32()?;
            Value::int(i64::from(n)).ok_or_else(|| {
                BuiltinError::ValueError("bad marshal data (integer overflow)".to_string())
            })
        }
        TYPE_LONG => Ok(bigint_to_value(read_long_bigint(reader)?)),
        TYPE_FLOAT => Ok(Value::float(reader.read_f64()?)),
        TYPE_STRING => Ok(bytes_value(reader.read_payload_bytes()?)),
        TYPE_UNICODE => {
            let bytes = reader.read_payload_bytes()?;
            let text = std::str::from_utf8(bytes).map_err(|_| {
                BuiltinError::ValueError("bad marshal data (invalid utf-8)".to_string())
            })?;
            Ok(Value::string(prism_core::intern::intern(text)))
        }
        TYPE_TUPLE => {
            let len = reader.read_len()?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(decode_value(reader)?);
            }
            Ok(crate::alloc_managed_value(TupleObject::from_slice(&values)))
        }
        TYPE_CODE => {
            let code = decode_code_object(reader)?;
            Ok(crate::alloc_managed_value(CodeObjectView::new(code)))
        }
        _ => Err(BuiltinError::ValueError(
            "bad marshal data (unknown type code)".to_string(),
        )),
    }
}

fn read_long_bigint(reader: &mut MarshalReader<'_>) -> Result<BigInt, BuiltinError> {
    let digit_count = reader.read_i32()?;
    let negative = digit_count < 0;
    let digit_count = digit_count.unsigned_abs() as usize;
    let raw_digits = reader.read_exact(digit_count * 2, "marshal data too short")?;

    let mut value = BigInt::from(0_u8);
    for (index, chunk) in raw_digits.chunks_exact(2).enumerate() {
        let digit = u16::from_le_bytes([chunk[0], chunk[1]]);
        if digit >= (1 << 15) {
            return Err(BuiltinError::ValueError(
                "bad marshal data (digit out of range in long)".to_string(),
            ));
        }
        let term = BigInt::from(digit) << (index * 15);
        value += term;
    }

    if negative {
        value = -value;
    }

    Ok(value)
}

fn decode_code_object(reader: &mut MarshalReader<'_>) -> Result<Arc<CodeObject>, BuiltinError> {
    let marker = reader.read_exact(PRISM_CODE_MARKER.len(), "marshal data too short")?;
    if marker != PRISM_CODE_MARKER {
        return Err(BuiltinError::ValueError(
            "bad marshal data (invalid Prism code object)".to_string(),
        ));
    }

    let name = reader.read_string()?;
    let qualname = reader.read_string()?;
    let filename = reader.read_string()?;
    let first_lineno = reader.read_u32()?;

    let arg_count = reader.read_u16()?;
    let posonlyarg_count = reader.read_u16()?;
    let kwonlyarg_count = reader.read_u16()?;
    let register_count = reader.read_u16()?;
    let flags_bits = reader.read_u32()?;
    let flags = CodeFlags::from_bits(flags_bits).ok_or_else(|| {
        BuiltinError::ValueError("bad marshal data (invalid code flags)".to_string())
    })?;

    let locals = reader.read_string_array()?;
    let names = reader.read_string_array()?;
    let freevars = reader.read_string_array()?;
    let cellvars = reader.read_string_array()?;

    let instruction_count = reader.read_len()?;
    let mut instructions = Vec::with_capacity(instruction_count);
    for _ in 0..instruction_count {
        instructions.push(Instruction::from_raw(reader.read_u32()?));
    }

    let line_count = reader.read_len()?;
    let mut line_table = Vec::with_capacity(line_count);
    for _ in 0..line_count {
        line_table.push(LineTableEntry {
            start_pc: reader.read_u32()?,
            end_pc: reader.read_u32()?,
            line: reader.read_u32()?,
        });
    }

    let exception_count = reader.read_len()?;
    let mut exception_table = Vec::with_capacity(exception_count);
    for _ in 0..exception_count {
        exception_table.push(ExceptionEntry {
            start_pc: reader.read_u32()?,
            end_pc: reader.read_u32()?,
            handler_pc: reader.read_u32()?,
            finally_pc: reader.read_u32()?,
            depth: reader.read_u16()?,
            exception_type_idx: reader.read_u16()?,
        });
    }

    let nested_count = reader.read_len()?;
    let mut nested_code_objects = Vec::with_capacity(nested_count);
    for _ in 0..nested_count {
        nested_code_objects.push(decode_code_object(reader)?);
    }

    let constant_count = reader.read_len()?;
    let mut constants = Vec::with_capacity(constant_count);
    for _ in 0..constant_count {
        constants.push(decode_code_constant(reader, &nested_code_objects)?);
    }

    Ok(Arc::new(CodeObject {
        name,
        qualname,
        filename,
        first_lineno,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        locals,
        names,
        freevars,
        cellvars,
        arg_count,
        posonlyarg_count,
        kwonlyarg_count,
        register_count,
        flags,
        line_table: line_table.into_boxed_slice(),
        exception_table: exception_table.into_boxed_slice(),
        nested_code_objects: nested_code_objects.into_boxed_slice(),
    }))
}

fn decode_code_constant(
    reader: &mut MarshalReader<'_>,
    nested_code_objects: &[Arc<CodeObject>],
) -> Result<Constant, BuiltinError> {
    let tag = reader.read_u8("EOF read where object expected")?;
    match tag {
        TYPE_CODE_REF => {
            let index = reader.read_len()?;
            let code = nested_code_objects.get(index).ok_or_else(|| {
                BuiltinError::ValueError("bad marshal data (invalid code reference)".to_string())
            })?;
            Ok(Constant::Value(Value::object_ptr(
                Arc::into_raw(Arc::clone(code)) as *const (),
            )))
        }
        TYPE_KWNAMES => {
            let len = reader.read_len()?;
            let mut names = Vec::with_capacity(len);
            for _ in 0..len {
                names.push(reader.read_string()?);
            }
            Ok(Constant::Value(crate::alloc_managed_value(
                KwNamesTuple::new(names),
            )))
        }
        _ => {
            reader.offset -= 1;
            Ok(Constant::Value(decode_value(reader)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::intern;

    fn bytes_to_vec(value: Value) -> Vec<u8> {
        let ptr = value.as_object_ptr().expect("bytes should be heap-backed");
        let bytes = unsafe { &*(ptr as *const BytesObject) };
        bytes.to_vec()
    }

    fn code_view(value: Value) -> &'static CodeObjectView {
        let ptr = value
            .as_object_ptr()
            .expect("code object should be heap-backed");
        assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::CODE);
        unsafe { &*(ptr as *const CodeObjectView) }
    }

    #[test]
    fn test_marshal_module_exposes_version_and_callables() {
        let module = MarshalModule::new();

        assert_eq!(
            module.get_attr("version").unwrap().as_int(),
            Some(MARSHAL_VERSION)
        );
        assert!(module.get_attr("dumps").is_ok());
        assert!(module.get_attr("loads").is_ok());
        assert_eq!(
            module.dir(),
            vec![Arc::from("dumps"), Arc::from("loads"), Arc::from("version")]
        );
    }

    #[test]
    fn test_marshal_dumps_and_loads_bool_values() {
        let true_bytes =
            marshal_dumps(&[Value::bool(true)]).expect("marshal.dumps(True) should succeed");
        let false_bytes =
            marshal_dumps(&[Value::bool(false)]).expect("marshal.dumps(False) should succeed");

        assert_eq!(bytes_to_vec(true_bytes), vec![TYPE_TRUE]);
        assert_eq!(bytes_to_vec(false_bytes), vec![TYPE_FALSE]);
        assert_eq!(marshal_loads(&[true_bytes]).unwrap().as_bool(), Some(true));
        assert_eq!(
            marshal_loads(&[false_bytes]).unwrap().as_bool(),
            Some(false)
        );
    }

    #[test]
    fn test_marshal_round_trips_small_and_large_ints() {
        let small =
            marshal_dumps(&[Value::int(123_456).unwrap()]).expect("small ints should marshal");
        assert_eq!(marshal_loads(&[small]).unwrap().as_int(), Some(123_456));

        let big = bigint_to_value(BigInt::from(1_u8) << 80_u32);
        let round_tripped =
            marshal_loads(&[marshal_dumps(&[big]).expect("big ints should marshal")])
                .expect("big ints should unmarshal");
        assert_eq!(value_to_bigint(round_tripped), value_to_bigint(big));
    }

    #[test]
    fn test_marshal_round_trips_string_and_bytes() {
        let text = Value::string(intern("prism"));
        let text_round_trip =
            marshal_loads(&[marshal_dumps(&[text]).expect("strings should marshal")])
                .expect("strings should unmarshal");
        assert_eq!(
            value_as_string_ref(text_round_trip)
                .expect("string round-trip should return a string")
                .as_str(),
            "prism"
        );

        let bytes = bytes_value(b"abc");
        let bytes_round_trip =
            marshal_loads(&[marshal_dumps(&[bytes]).expect("bytes should marshal")])
                .expect("bytes should unmarshal");
        assert_eq!(bytes_to_vec(bytes_round_trip), b"abc");
    }

    #[test]
    fn test_marshal_round_trips_float_and_tuple_values() {
        let tuple = Value::object_ptr(Box::into_raw(Box::new(TupleObject::from_slice(&[
            Value::float(1.25),
            Value::string(intern("ok")),
        ]))) as *const ());

        let round_tripped =
            marshal_loads(&[marshal_dumps(&[tuple]).expect("tuple should marshal")])
                .expect("tuple should unmarshal");
        let tuple_ptr = round_tripped
            .as_object_ptr()
            .expect("tuple should be heap-backed");
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };

        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.get(0).and_then(|value| value.as_float()), Some(1.25));
        assert_eq!(
            tuple
                .get(1)
                .and_then(|value| value_as_string_ref(value))
                .map(|value| value.as_str().to_string()),
            Some("ok".to_string())
        );
    }

    #[test]
    fn test_marshal_round_trips_prism_code_objects() {
        let nested = Arc::new(CodeObject {
            name: Arc::from("child"),
            qualname: Arc::from("parent.child"),
            filename: Arc::from("pkg.py"),
            first_lineno: 7,
            instructions: Box::new([Instruction::op(Opcode::ReturnNone)]),
            constants: Box::new([Constant::Value(Value::none())]),
            locals: Box::new([Arc::from("x")]),
            names: Box::new([Arc::from("global_name")]),
            freevars: Box::new([]),
            cellvars: Box::new([]),
            arg_count: 1,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            register_count: 2,
            flags: CodeFlags::NESTED,
            line_table: Box::new([LineTableEntry {
                start_pc: 0,
                end_pc: 1,
                line: 7,
            }]),
            exception_table: Box::new([]),
            nested_code_objects: Box::new([]),
        });
        let nested_const = Value::object_ptr(Arc::into_raw(Arc::clone(&nested)) as *const ());
        let code = Arc::new(CodeObject {
            name: Arc::from("<module>"),
            qualname: Arc::from("<module>"),
            filename: Arc::from("pkg.py"),
            first_lineno: 1,
            instructions: Box::new([
                Instruction::op_di(Opcode::MakeFunction, prism_code::Register(0), 1),
                Instruction::op(Opcode::ReturnNone),
            ]),
            constants: Box::new([
                Constant::Value(Value::string(intern("module-constant"))),
                Constant::Value(nested_const),
            ]),
            locals: Box::new([]),
            names: Box::new([Arc::from("__name__")]),
            freevars: Box::new([]),
            cellvars: Box::new([]),
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            register_count: 1,
            flags: CodeFlags::MODULE,
            line_table: Box::new([LineTableEntry {
                start_pc: 0,
                end_pc: 2,
                line: 1,
            }]),
            exception_table: Box::new([ExceptionEntry {
                start_pc: 0,
                end_pc: 1,
                handler_pc: 1,
                finally_pc: u32::MAX,
                depth: 0,
                exception_type_idx: u16::MAX,
            }]),
            nested_code_objects: Box::new([nested]),
        });
        let code_value =
            Value::object_ptr(Box::into_raw(Box::new(CodeObjectView::new(code))) as *const ());

        let round_tripped =
            marshal_loads(&[marshal_dumps(&[code_value]).expect("code should marshal")])
                .expect("code should unmarshal");
        let loaded = code_view(round_tripped).code();

        assert_eq!(loaded.name.as_ref(), "<module>");
        assert_eq!(loaded.filename.as_ref(), "pkg.py");
        assert_eq!(loaded.instructions.len(), 2);
        assert_eq!(loaded.constants.len(), 2);
        assert_eq!(loaded.nested_code_objects.len(), 1);
        assert_eq!(loaded.nested_code_objects[0].name.as_ref(), "child");
        assert_eq!(loaded.exception_table[0].handler_pc, 1);

        let nested_ptr = match loaded.constants[1] {
            Constant::Value(value) => value.as_object_ptr().expect("nested code ref"),
            Constant::BigInt(_) => panic!("nested code constant should be a value"),
        };
        assert_eq!(
            nested_ptr,
            Arc::as_ptr(&loaded.nested_code_objects[0]) as *const ()
        );
    }

    #[test]
    fn test_marshal_round_trips_keyword_name_constants_in_code_objects() {
        let kwnames = Box::into_raw(Box::new(KwNamesTuple::new(vec![
            Arc::from("encoding"),
            Arc::from("errors"),
        ])));
        let code = Arc::new(CodeObject {
            name: Arc::from("call_site"),
            qualname: Arc::from("call_site"),
            filename: Arc::from("pkg.py"),
            first_lineno: 1,
            instructions: Box::new([
                Instruction::new(Opcode::CallKw, 0, 1, 0),
                Instruction::new(Opcode::CallKwEx, 2, 0, 0),
            ]),
            constants: vec![Constant::Value(Value::object_ptr(kwnames as *const ()))]
                .into_boxed_slice(),
            locals: Box::new([]),
            names: Box::new([]),
            freevars: Box::new([]),
            cellvars: Box::new([]),
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            register_count: 3,
            flags: CodeFlags::NONE,
            line_table: Box::new([]),
            exception_table: Box::new([]),
            nested_code_objects: Box::new([]),
        });
        let code_value =
            Value::object_ptr(Box::into_raw(Box::new(CodeObjectView::new(code))) as *const ());

        let round_tripped =
            marshal_loads(&[marshal_dumps(&[code_value]).expect("code should marshal")])
                .expect("code should unmarshal");
        let loaded = code_view(round_tripped).code();
        let kwnames_ptr = match loaded.constants[0] {
            Constant::Value(value) => value.as_object_ptr().expect("keyword names ref"),
            Constant::BigInt(_) => panic!("keyword names constant should be a value"),
        };
        let kwnames = unsafe { &*(kwnames_ptr as *const KwNamesTuple) };

        assert_eq!(kwnames.len(), 2);
        assert_eq!(kwnames.get(0).map(AsRef::as_ref), Some("encoding"));
        assert_eq!(kwnames.get(1).map(AsRef::as_ref), Some("errors"));
    }

    #[test]
    fn test_marshal_loads_ignores_trailing_bytes() {
        let result = decode_marshaled_value(&[TYPE_TRUE, 0x99, 0x98])
            .expect("trailing bytes should be ignored");
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_marshal_rejects_unsupported_values() {
        let err = marshal_dumps(&[bytes_value(b"abc"), Value::none(), Value::none()])
            .expect_err("extra marshal.dumps args should fail");
        assert!(
            err.to_string()
                .contains("takes from 1 to 2 positional arguments")
        );

        let object = crate::builtins::builtin_object(&[]).expect("object() should succeed");
        let err = marshal_dumps(&[object]).expect_err("unsupported values should fail");
        assert!(err.to_string().contains("unmarshallable object"));
    }
}
