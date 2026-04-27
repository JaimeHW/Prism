//! Native `pickle` bootstrap subset.
//!
//! The full pickle protocol is intentionally broad. This native module starts
//! with the protocol-stable scalar opcodes that CPython's bool regression tests
//! require, keeping the implementation table-driven and easy to extend without
//! penalizing ordinary runtime startup.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class,
    builtin_type_object_for_type_id, builtin_type_object_type_id, get_iterator_mut,
    runtime_error_to_builtin_error,
};
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::{
    dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id, get_attribute_value,
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::GenericAliasObject;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
use prism_runtime::types::iter::is_native_iterator_type_id;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::range::RangeObject;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

const HIGHEST_PROTOCOL: i64 = 5;
const DEFAULT_PROTOCOL: i64 = 4;
const TRUE_PROTO0: &[u8] = b"I01\n.";
const FALSE_PROTO0: &[u8] = b"I00\n.";
const PROTO: u8 = 0x80;
const NEWTRUE: u8 = 0x88;
const NEWFALSE: u8 = 0x89;
const STOP: u8 = b'.';
const PRISM_PICKLE_MAGIC: &[u8] = b"\x80PRISM-PICKLE\x01";

const STD_MARK: u8 = b'(';
const STD_EMPTY_TUPLE: u8 = b')';
const STD_NONE: u8 = b'N';
const STD_INT: u8 = b'I';
const STD_BININT: u8 = b'J';
const STD_BININT1: u8 = b'K';
const STD_BININT2: u8 = b'M';
const STD_LONG: u8 = b'L';
const STD_TUPLE: u8 = b't';
const STD_GLOBAL: u8 = b'c';
const STD_REDUCE: u8 = b'R';
const STD_BUILD: u8 = b'b';
const STD_GET: u8 = b'g';
const STD_BINGET: u8 = b'h';
const STD_LONG_BINGET: u8 = b'j';
const STD_PUT: u8 = b'p';
const STD_BINPUT: u8 = b'q';
const STD_LONG_BINPUT: u8 = b'r';
const STD_TUPLE1: u8 = 0x85;
const STD_TUPLE2: u8 = 0x86;
const STD_TUPLE3: u8 = 0x87;
const STD_LONG1: u8 = 0x8a;
const STD_LONG4: u8 = 0x8b;
const STD_SHORT_BINUNICODE: u8 = 0x8c;
const STD_BINUNICODE: u8 = b'X';
const STD_BINUNICODE8: u8 = 0x8d;
const STD_STACK_GLOBAL: u8 = 0x93;
const STD_MEMOIZE: u8 = 0x94;
const STD_FRAME: u8 = 0x95;

const TAG_REF: u8 = 0;
const TAG_NONE: u8 = 1;
const TAG_FALSE: u8 = 2;
const TAG_TRUE: u8 = 3;
const TAG_INT: u8 = 4;
const TAG_FLOAT: u8 = 5;
const TAG_STRING: u8 = 6;
const TAG_BYTES: u8 = 7;
const TAG_BYTEARRAY: u8 = 8;
const TAG_LIST: u8 = 9;
const TAG_TUPLE: u8 = 10;
const TAG_DICT: u8 = 11;
const TAG_BUILTIN_FUNCTION: u8 = 12;
const TAG_TYPE: u8 = 13;
const TAG_USER_OBJECT: u8 = 14;
const TAG_REDUCE: u8 = 15;
const TAG_GENERIC_ALIAS: u8 = 16;
const TAG_SLICE: u8 = 17;
const TAG_RANGE: u8 = 18;
const TAG_DICT_BACKED_USER_OBJECT: u8 = 19;

const TYPE_KIND_BUILTIN: u8 = 0;
const TYPE_KIND_USER: u8 = 1;

static DUMPS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm_kw(Arc::from("pickle.dumps"), pickle_dumps));
static LOADS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("pickle.loads"), pickle_loads));

/// Native `pickle` module descriptor.
#[derive(Debug, Clone)]
pub struct PickleModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl PickleModule {
    /// Create a native `pickle` module.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("DEFAULT_PROTOCOL"),
                Arc::from("HIGHEST_PROTOCOL"),
                Arc::from("__all__"),
                Arc::from("dumps"),
                Arc::from("loads"),
            ],
            all: string_list_value(&["dumps", "loads"]),
        }
    }
}

impl Default for PickleModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for PickleModule {
    fn name(&self) -> &str {
        "pickle"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "DEFAULT_PROTOCOL" => Ok(Value::int(DEFAULT_PROTOCOL).expect("protocol fits")),
            "HIGHEST_PROTOCOL" => Ok(Value::int(HIGHEST_PROTOCOL).expect("protocol fits")),
            "__all__" => Ok(self.all),
            "dumps" => Ok(builtin_value(&DUMPS_FUNCTION)),
            "loads" => Ok(builtin_value(&LOADS_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'pickle' has no attribute '{}'",
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
fn bytes_value(bytes: &[u8]) -> Value {
    crate::alloc_managed_value(BytesObject::from_slice(bytes))
}

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}

struct PickleWriter<'vm> {
    vm: &'vm mut VirtualMachine,
    out: Vec<u8>,
    memo: HashMap<usize, u32>,
}

impl<'vm> PickleWriter<'vm> {
    fn new(vm: &'vm mut VirtualMachine) -> Self {
        let mut out = Vec::with_capacity(PRISM_PICKLE_MAGIC.len() + 64);
        out.extend_from_slice(PRISM_PICKLE_MAGIC);
        Self {
            vm,
            out,
            memo: HashMap::new(),
        }
    }

    fn finish(self) -> Vec<u8> {
        self.out
    }

    fn write_value(&mut self, value: Value) -> Result<(), BuiltinError> {
        if value.is_none() {
            self.write_tag(TAG_NONE);
            return Ok(());
        }
        if let Some(flag) = value.as_bool() {
            self.write_tag(if flag { TAG_TRUE } else { TAG_FALSE });
            return Ok(());
        }
        if let Some(integer) = value_to_bigint(value) {
            self.write_tag(TAG_INT);
            self.write_bytes(&integer.to_signed_bytes_le())?;
            return Ok(());
        }
        if let Some(float) = value.as_float() {
            self.write_tag(TAG_FLOAT);
            self.write_u64(float.to_bits());
            return Ok(());
        }
        if let Some(string) = value_as_string_ref(value) {
            self.write_tag(TAG_STRING);
            self.write_str(string.as_str())?;
            return Ok(());
        }

        let Some(ptr) = value.as_object_ptr() else {
            return Err(cannot_pickle(value));
        };
        self.write_object(value, ptr, extract_type_id(ptr))
    }

    fn write_object(
        &mut self,
        value: Value,
        ptr: *const (),
        type_id: TypeId,
    ) -> Result<(), BuiltinError> {
        match type_id {
            TypeId::BYTES | TypeId::BYTEARRAY => {
                let bytes = unsafe { &*(ptr as *const BytesObject) };
                self.write_tag(if bytes.is_bytearray() {
                    TAG_BYTEARRAY
                } else {
                    TAG_BYTES
                });
                self.write_bytes(&bytes.to_vec())
            }
            TypeId::LIST => self.write_list(ptr),
            TypeId::TUPLE => self.write_tuple(ptr),
            TypeId::DICT => self.write_dict(ptr),
            TypeId::BUILTIN_FUNCTION => self.write_builtin_function(ptr),
            TypeId::TYPE => self.write_type(ptr),
            TypeId::ITERATOR | TypeId::ENUMERATE => self.write_reduce(value, ptr),
            TypeId::GENERIC_ALIAS => self.write_generic_alias(ptr),
            TypeId::SLICE => self.write_slice(ptr),
            TypeId::RANGE => self.write_range(ptr),
            type_id
                if type_id.raw() >= TypeId::FIRST_USER_TYPE
                    && is_native_iterator_type_id(type_id) =>
            {
                self.write_reduce(value, ptr)
            }
            type_id
                if type_id.raw() >= TypeId::FIRST_USER_TYPE
                    && dict_storage_ref_from_ptr(ptr).is_some() =>
            {
                self.write_dict_backed_user_object(ptr, type_id)
            }
            type_id if type_id.raw() >= TypeId::FIRST_USER_TYPE => {
                self.write_user_object(ptr, type_id)
            }
            _ => Err(cannot_pickle(value)),
        }
    }

    fn write_list(&mut self, ptr: *const ()) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };
        let list = unsafe { &*(ptr as *const ListObject) };
        let values = list.iter().copied().collect::<Vec<_>>();

        self.write_tag(TAG_LIST);
        self.write_u32(id);
        self.write_len(values.len(), "list length")?;
        for value in values {
            self.write_value(value)?;
        }
        Ok(())
    }

    fn write_tuple(&mut self, ptr: *const ()) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        let values = tuple.as_slice().to_vec();

        self.write_tag(TAG_TUPLE);
        self.write_u32(id);
        self.write_len(values.len(), "tuple length")?;
        for value in values {
            self.write_value(value)?;
        }
        Ok(())
    }

    fn write_dict(&mut self, ptr: *const ()) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };
        let dict = unsafe { &*(ptr as *const DictObject) };
        let entries = dict.iter().collect::<Vec<_>>();

        self.write_tag(TAG_DICT);
        self.write_u32(id);
        self.write_len(entries.len(), "dict length")?;
        for (key, value) in entries {
            self.write_value(key)?;
            self.write_value(value)?;
        }
        Ok(())
    }

    fn write_builtin_function(&mut self, ptr: *const ()) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };
        let function = unsafe { &*(ptr as *const BuiltinFunctionObject) };
        if function.bound_self().is_some() {
            return Err(BuiltinError::TypeError(format!(
                "cannot pickle bound builtin function '{}'",
                function.name()
            )));
        }

        self.write_tag(TAG_BUILTIN_FUNCTION);
        self.write_u32(id);
        self.write_str(function.name())
    }

    fn write_type(&mut self, ptr: *const ()) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };

        self.write_tag(TAG_TYPE);
        self.write_u32(id);
        if let Some(type_id) = builtin_type_object_type_id(ptr) {
            self.write_u8(TYPE_KIND_BUILTIN);
            self.write_u32(type_id.raw());
            return Ok(());
        }

        let class = unsafe { &*(ptr as *const PyClassObject) };
        self.write_u8(TYPE_KIND_USER);
        self.write_u32(class.class_type_id().raw());
        let (module, name) = class_export_name(class);
        self.write_str(&module)?;
        self.write_str(&name)
    }

    fn write_user_object(&mut self, ptr: *const (), type_id: TypeId) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };
        let class = global_class(ClassId(type_id.raw())).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "cannot pickle instance of unpublished type '{}'",
                type_id.name()
            ))
        })?;
        let (module, name) = class_export_name(class.as_ref());
        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        let attrs = shaped.iter_properties().collect::<Vec<_>>();

        self.write_tag(TAG_USER_OBJECT);
        self.write_u32(id);
        self.write_u32(type_id.raw());
        self.write_str(&module)?;
        self.write_str(&name)?;
        self.write_len(attrs.len(), "attribute count")?;
        for (name, value) in attrs {
            self.write_str(name.as_str())?;
            self.write_value(value)?;
        }
        Ok(())
    }

    fn write_dict_backed_user_object(
        &mut self,
        ptr: *const (),
        type_id: TypeId,
    ) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };
        let class = global_class(ClassId(type_id.raw())).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "cannot pickle instance of unpublished type '{}'",
                type_id.name()
            ))
        })?;
        let (module, name) = class_export_name(class.as_ref());
        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        let attrs = shaped.iter_properties().collect::<Vec<_>>();
        let dict = dict_storage_ref_from_ptr(ptr).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "cannot pickle dict-backed instance of '{}'",
                type_id.name()
            ))
        })?;
        let entries = dict.iter().collect::<Vec<_>>();

        self.write_tag(TAG_DICT_BACKED_USER_OBJECT);
        self.write_u32(id);
        self.write_u32(type_id.raw());
        self.write_str(&module)?;
        self.write_str(&name)?;
        self.write_len(attrs.len(), "attribute count")?;
        for (name, value) in attrs {
            self.write_str(name.as_str())?;
            self.write_value(value)?;
        }
        self.write_len(entries.len(), "dictionary entry count")?;
        for (key, value) in entries {
            self.write_value(key)?;
            self.write_value(value)?;
        }
        Ok(())
    }

    fn write_reduce(&mut self, value: Value, ptr: *const ()) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };

        let reduce = get_attribute_value(self.vm, value, &intern("__reduce__"))
            .map_err(runtime_error_to_builtin_error)?;
        let reduction =
            invoke_callable_value(self.vm, reduce, &[]).map_err(runtime_error_to_builtin_error)?;

        self.write_tag(TAG_REDUCE);
        self.write_u32(id);
        self.write_value(reduction)
    }

    fn write_generic_alias(&mut self, ptr: *const ()) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };
        let alias = unsafe { &*(ptr as *const GenericAliasObject) };

        self.write_tag(TAG_GENERIC_ALIAS);
        self.write_u32(id);
        self.write_value(alias.origin())?;
        self.write_len(alias.args().len(), "generic alias argument count")?;
        for arg in alias.args() {
            self.write_value(*arg)?;
        }
        self.write_u8(u8::from(alias.is_starred()));
        Ok(())
    }

    fn write_slice(&mut self, ptr: *const ()) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };
        let slice = unsafe { &*(ptr as *const SliceObject) };

        self.write_tag(TAG_SLICE);
        self.write_u32(id);
        self.write_value(slice.start_value())?;
        self.write_value(slice.stop_value())?;
        self.write_value(slice.step_value())
    }

    fn write_range(&mut self, ptr: *const ()) -> Result<(), BuiltinError> {
        let Some(id) = self.begin_memo(ptr)? else {
            return Ok(());
        };
        let range = unsafe { &*(ptr as *const RangeObject) };

        self.write_tag(TAG_RANGE);
        self.write_u32(id);
        self.write_value(bigint_to_value(range.start_bigint()))?;
        self.write_value(bigint_to_value(range.stop_bigint()))?;
        self.write_value(bigint_to_value(range.step_bigint()))
    }

    fn begin_memo(&mut self, ptr: *const ()) -> Result<Option<u32>, BuiltinError> {
        let key = ptr as usize;
        if let Some(id) = self.memo.get(&key).copied() {
            self.write_tag(TAG_REF);
            self.write_u32(id);
            return Ok(None);
        }

        let id = checked_u32(self.memo.len(), "pickle memo")?;
        self.memo.insert(key, id);
        Ok(Some(id))
    }

    #[inline]
    fn write_tag(&mut self, tag: u8) {
        self.out.push(tag);
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.out.push(value);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.out.extend_from_slice(&value.to_le_bytes());
    }

    fn write_len(&mut self, len: usize, context: &'static str) -> Result<(), BuiltinError> {
        self.write_u32(checked_u32(len, context)?);
        Ok(())
    }

    fn write_str(&mut self, value: &str) -> Result<(), BuiltinError> {
        self.write_bytes(value.as_bytes())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), BuiltinError> {
        self.write_len(bytes.len(), "byte string length")?;
        self.out.extend_from_slice(bytes);
        Ok(())
    }
}

struct PickleReader<'bytes> {
    bytes: &'bytes [u8],
    pos: usize,
    memo: Vec<Option<Value>>,
}

impl<'bytes> PickleReader<'bytes> {
    fn new(bytes: &'bytes [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            memo: Vec::new(),
        }
    }

    fn expect_end(&self) -> Result<(), BuiltinError> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(invalid_pickle("trailing data after pickle payload"))
        }
    }

    fn read_value(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        match self.read_u8()? {
            TAG_REF => self.read_ref(),
            TAG_NONE => Ok(Value::none()),
            TAG_FALSE => Ok(Value::bool(false)),
            TAG_TRUE => Ok(Value::bool(true)),
            TAG_INT => {
                let bytes = self.read_bytes()?;
                Ok(bigint_to_value(num_bigint::BigInt::from_signed_bytes_le(
                    bytes,
                )))
            }
            TAG_FLOAT => Ok(Value::float(f64::from_bits(self.read_u64()?))),
            TAG_STRING => {
                let string = self.read_string()?;
                Ok(Value::string(intern(&string)))
            }
            TAG_BYTES => {
                let bytes = self.read_bytes()?;
                Ok(crate::alloc_managed_value(BytesObject::from_slice(bytes)))
            }
            TAG_BYTEARRAY => {
                let bytes = self.read_bytes()?;
                Ok(crate::alloc_managed_value(
                    BytesObject::bytearray_from_slice(bytes),
                ))
            }
            TAG_LIST => self.read_list(vm),
            TAG_TUPLE => self.read_tuple(vm),
            TAG_DICT => self.read_dict(vm),
            TAG_BUILTIN_FUNCTION => self.read_builtin_function(vm),
            TAG_TYPE => self.read_type(vm),
            TAG_USER_OBJECT => self.read_user_object(vm),
            TAG_DICT_BACKED_USER_OBJECT => self.read_dict_backed_user_object(vm),
            TAG_REDUCE => self.read_reduce(vm),
            TAG_GENERIC_ALIAS => self.read_generic_alias(vm),
            TAG_SLICE => self.read_slice(vm),
            TAG_RANGE => self.read_range(vm),
            _ => Err(invalid_pickle("unknown value tag")),
        }
    }

    fn read_ref(&mut self) -> Result<Value, BuiltinError> {
        let id = self.read_u32()? as usize;
        self.memo
            .get(id)
            .and_then(|slot| *slot)
            .ok_or_else(|| invalid_pickle("invalid memo reference"))
    }

    fn read_list(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let len = self.read_len()?;
        let value = crate::alloc_managed_value(ListObject::new());
        self.insert_memo(id, value)?;

        for _ in 0..len {
            let item = self.read_value(vm)?;
            let ptr = value
                .as_object_ptr()
                .expect("new list values are heap objects");
            unsafe { &mut *(ptr as *mut ListObject) }.push(item);
        }
        Ok(value)
    }

    fn read_tuple(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let len = self.read_len()?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_value(vm)?);
        }

        let value = crate::alloc_managed_value(TupleObject::from_vec(values));
        self.insert_memo(id, value)?;
        Ok(value)
    }

    fn read_dict(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let len = self.read_len()?;
        let value = crate::alloc_managed_value(DictObject::with_capacity(len));
        self.insert_memo(id, value)?;

        for _ in 0..len {
            let key = self.read_value(vm)?;
            let item = self.read_value(vm)?;
            let ptr = value
                .as_object_ptr()
                .expect("new dict values are heap objects");
            unsafe { &mut *(ptr as *mut DictObject) }.set(key, item);
        }
        Ok(value)
    }

    fn read_builtin_function(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let name = self.read_string()?;
        let value = resolve_builtin(vm, &name)
            .ok_or_else(|| invalid_pickle(format!("unknown builtin function '{}'", name)))?;
        self.insert_memo(id, value)?;
        Ok(value)
    }

    fn read_type(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let value = match self.read_u8()? {
            TYPE_KIND_BUILTIN => {
                let raw = self.read_u32()?;
                builtin_type_object_for_type_id(TypeId::from_raw(raw))
            }
            TYPE_KIND_USER => {
                let raw = self.read_u32()?;
                let module = self.read_string()?;
                let name = self.read_string()?;
                let class = resolve_user_class(vm, raw, &module, &name)?;
                Value::object_ptr(Arc::as_ptr(&class) as *const ())
            }
            _ => return Err(invalid_pickle("unknown type kind")),
        };
        self.insert_memo(id, value)?;
        Ok(value)
    }

    fn read_user_object(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let class_id = self.read_u32()?;
        let module = self.read_string()?;
        let name = self.read_string()?;
        let attr_count = self.read_len()?;
        let class = resolve_user_class(vm, class_id, &module, &name)?;
        let object = allocate_heap_instance_for_class(class.as_ref());
        let value = crate::alloc_managed_value(object);
        self.insert_memo(id, value)?;

        for _ in 0..attr_count {
            let name = self.read_string()?;
            let attr_value = self.read_value(vm)?;
            let ptr = value
                .as_object_ptr()
                .expect("new heap instances are object values");
            let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
            shaped.set_property(intern(&name), attr_value, shape_registry());
        }
        Ok(value)
    }

    fn read_dict_backed_user_object(
        &mut self,
        vm: &mut VirtualMachine,
    ) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let class_id = self.read_u32()?;
        let module = self.read_string()?;
        let name = self.read_string()?;
        let attr_count = self.read_len()?;
        let class = resolve_user_class(vm, class_id, &module, &name)?;
        let object = allocate_heap_instance_for_class(class.as_ref());
        let value = crate::alloc_managed_value(object);
        self.insert_memo(id, value)?;

        for _ in 0..attr_count {
            let name = self.read_string()?;
            let attr_value = self.read_value(vm)?;
            let ptr = value
                .as_object_ptr()
                .expect("new heap instances are object values");
            let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
            shaped.set_property(intern(&name), attr_value, shape_registry());
        }

        let entry_count = self.read_len()?;
        for _ in 0..entry_count {
            let key = self.read_value(vm)?;
            let item = self.read_value(vm)?;
            let ptr = value
                .as_object_ptr()
                .expect("new heap instances are object values");
            let dict = dict_storage_mut_from_ptr(ptr)
                .ok_or_else(|| invalid_pickle("class is not dict-backed"))?;
            dict.set(key, item);
        }

        Ok(value)
    }

    fn read_reduce(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let reduction = self.read_value(vm)?;
        let value = reconstruct_reduce(vm, reduction)?;
        self.insert_memo(id, value)?;
        Ok(value)
    }

    fn read_generic_alias(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let origin = self.read_value(vm)?;
        let len = self.read_len()?;
        let mut args = Vec::with_capacity(len);
        for _ in 0..len {
            args.push(self.read_value(vm)?);
        }
        let starred = self.read_u8()? != 0;
        let value =
            crate::alloc_managed_value(GenericAliasObject::new_with_starred(origin, args, starred));
        self.insert_memo(id, value)?;
        Ok(value)
    }

    fn read_slice(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let start = self.read_value(vm)?;
        let stop = self.read_value(vm)?;
        let step = self.read_value(vm)?;
        let value = crate::alloc_managed_value(SliceObject::new(start, stop, step));
        self.insert_memo(id, value)?;
        Ok(value)
    }

    fn read_range(&mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        let id = self.read_u32()?;
        let start = self.read_range_component(vm, "start")?;
        let stop = self.read_range_component(vm, "stop")?;
        let step = self.read_range_component(vm, "step")?;
        if step == num_bigint::BigInt::from(0) {
            return Err(invalid_pickle("range step cannot be zero"));
        }

        let value = crate::alloc_managed_value(RangeObject::from_bigints(start, stop, step));
        self.insert_memo(id, value)?;
        Ok(value)
    }

    fn read_range_component(
        &mut self,
        vm: &mut VirtualMachine,
        name: &'static str,
    ) -> Result<num_bigint::BigInt, BuiltinError> {
        let value = self.read_value(vm)?;
        value_to_bigint(value).ok_or_else(|| invalid_pickle(format!("range {name} is not an int")))
    }

    fn insert_memo(&mut self, id: u32, value: Value) -> Result<(), BuiltinError> {
        let id = id as usize;
        if id >= self.memo.len() {
            self.memo.resize(id + 1, None);
        }
        if self.memo[id].is_some() {
            return Err(invalid_pickle("duplicate memo id"));
        }
        self.memo[id] = Some(value);
        Ok(())
    }

    fn read_len(&mut self) -> Result<usize, BuiltinError> {
        Ok(self.read_u32()? as usize)
    }

    fn read_string(&mut self) -> Result<String, BuiltinError> {
        let bytes = self.read_bytes()?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| invalid_pickle("invalid UTF-8 string payload"))
    }

    fn read_bytes(&mut self) -> Result<&'bytes [u8], BuiltinError> {
        let len = self.read_len()?;
        self.read_exact(len)
    }

    fn read_u8(&mut self) -> Result<u8, BuiltinError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, BuiltinError> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("read_exact returned 4 bytes"),
        ))
    }

    fn read_u64(&mut self) -> Result<u64, BuiltinError> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("read_exact returned 8 bytes"),
        ))
    }

    fn read_exact(&mut self, len: usize) -> Result<&'bytes [u8], BuiltinError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| invalid_pickle("pickle payload offset overflow"))?;
        if end > self.bytes.len() {
            return Err(invalid_pickle("truncated pickle payload"));
        }
        let bytes = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }
}

#[derive(Clone, Copy)]
enum StdPickleCallable {
    Iter,
    Range,
}

#[derive(Clone)]
enum StdPickleItem {
    Mark,
    Value(Value),
    String(String),
    Callable(StdPickleCallable),
}

struct StdPickleReader<'bytes> {
    bytes: &'bytes [u8],
    pos: usize,
    stack: Vec<StdPickleItem>,
    memo: Vec<Option<StdPickleItem>>,
}

impl<'bytes> StdPickleReader<'bytes> {
    fn new(bytes: &'bytes [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            stack: Vec::with_capacity(16),
            memo: Vec::new(),
        }
    }

    fn read(mut self, vm: &mut VirtualMachine) -> Result<Value, BuiltinError> {
        loop {
            let opcode = self.read_u8()?;
            match opcode {
                PROTO => self.read_protocol()?,
                STOP => return self.finish(),
                STD_FRAME => self.read_frame()?,
                STD_MARK => self.stack.push(StdPickleItem::Mark),
                STD_NONE => self.push_value(Value::none()),
                NEWFALSE => self.push_value(Value::bool(false)),
                NEWTRUE => self.push_value(Value::bool(true)),
                STD_INT => {
                    let integer = self.read_decimal_line("pickle integer")?;
                    self.push_value(bigint_to_value(integer));
                }
                STD_LONG => {
                    let integer = self.read_long_line()?;
                    self.push_value(bigint_to_value(integer));
                }
                STD_BININT => {
                    let bytes = self.read_exact(4)?;
                    let integer =
                        i32::from_le_bytes(bytes.try_into().expect("read_exact returned 4 bytes"));
                    self.push_value(bigint_to_value(BigInt::from(integer)));
                }
                STD_BININT1 => {
                    let integer = self.read_u8()?;
                    self.push_value(bigint_to_value(BigInt::from(integer)));
                }
                STD_BININT2 => {
                    let bytes = self.read_exact(2)?;
                    let integer =
                        u16::from_le_bytes(bytes.try_into().expect("read_exact returned 2 bytes"));
                    self.push_value(bigint_to_value(BigInt::from(integer)));
                }
                STD_LONG1 => {
                    let len = self.read_u8()? as usize;
                    let integer = BigInt::from_signed_bytes_le(self.read_exact(len)?);
                    self.push_value(bigint_to_value(integer));
                }
                STD_LONG4 => {
                    let len = self.read_u32_as_usize("LONG4 length")?;
                    let integer = BigInt::from_signed_bytes_le(self.read_exact(len)?);
                    self.push_value(bigint_to_value(integer));
                }
                STD_GLOBAL => {
                    let module = self.read_utf8_line("global module")?;
                    let name = self.read_utf8_line("global name")?;
                    let item = resolve_std_global(vm, &module, &name)?;
                    self.stack.push(item);
                }
                STD_SHORT_BINUNICODE => {
                    let len = self.read_u8()? as usize;
                    let string = self.read_utf8_exact(len, "SHORT_BINUNICODE payload")?;
                    self.stack.push(StdPickleItem::String(string));
                }
                STD_BINUNICODE => {
                    let len = self.read_u32_as_usize("BINUNICODE length")?;
                    let string = self.read_utf8_exact(len, "BINUNICODE payload")?;
                    self.stack.push(StdPickleItem::String(string));
                }
                STD_BINUNICODE8 => {
                    let len = self.read_u64_as_usize("BINUNICODE8 length")?;
                    let string = self.read_utf8_exact(len, "BINUNICODE8 payload")?;
                    self.stack.push(StdPickleItem::String(string));
                }
                STD_STACK_GLOBAL => {
                    let name = self.pop_string("STACK_GLOBAL name")?;
                    let module = self.pop_string("STACK_GLOBAL module")?;
                    let item = resolve_std_global(vm, &module, &name)?;
                    self.stack.push(item);
                }
                STD_EMPTY_TUPLE => self.push_value(tuple_value(Vec::new())),
                STD_TUPLE => self.push_mark_tuple()?,
                STD_TUPLE1 => self.push_fixed_tuple(1)?,
                STD_TUPLE2 => self.push_fixed_tuple(2)?,
                STD_TUPLE3 => self.push_fixed_tuple(3)?,
                STD_REDUCE => {
                    let args = self.pop_value("REDUCE arguments")?;
                    let callable = self.pop_item("REDUCE callable")?;
                    let value = call_std_reduce(vm, callable, args)?;
                    self.push_value(value);
                }
                STD_BUILD => {
                    let state = self.pop_value("BUILD state")?;
                    let object = self.peek_value("BUILD target")?;
                    restore_reduce_state(vm, object, state)?;
                }
                STD_BINPUT => {
                    let index = self.read_u8()? as usize;
                    self.memoize_top(index)?;
                }
                STD_LONG_BINPUT => {
                    let index = self.read_u32_as_usize("LONG_BINPUT index")?;
                    self.memoize_top(index)?;
                }
                STD_PUT => {
                    let index = self.read_memo_line("PUT index")?;
                    self.memoize_top(index)?;
                }
                STD_MEMOIZE => {
                    let index = self.memo.len();
                    self.memoize_top(index)?;
                }
                STD_BINGET => {
                    let index = self.read_u8()? as usize;
                    self.push_memo(index)?;
                }
                STD_LONG_BINGET => {
                    let index = self.read_u32_as_usize("LONG_BINGET index")?;
                    self.push_memo(index)?;
                }
                STD_GET => {
                    let index = self.read_memo_line("GET index")?;
                    self.push_memo(index)?;
                }
                _ => {
                    return Err(invalid_pickle(format!(
                        "unsupported pickle opcode 0x{opcode:02x}"
                    )));
                }
            }
        }
    }

    fn read_protocol(&mut self) -> Result<(), BuiltinError> {
        let protocol = self.read_u8()?;
        if i64::from(protocol) > HIGHEST_PROTOCOL {
            return Err(invalid_pickle(format!(
                "unsupported pickle protocol {}",
                protocol
            )));
        }
        Ok(())
    }

    fn read_frame(&mut self) -> Result<(), BuiltinError> {
        let len = self.read_u64_as_usize("FRAME length")?;
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| invalid_pickle("pickle frame offset overflow"))?;
        if end > self.bytes.len() {
            return Err(invalid_pickle("truncated pickle frame"));
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Value, BuiltinError> {
        let value = self.pop_value("STOP value")?;
        if !self.stack.is_empty() {
            return Err(invalid_pickle("extra values left on pickle stack"));
        }
        Ok(value)
    }

    #[inline]
    fn push_value(&mut self, value: Value) {
        self.stack.push(StdPickleItem::Value(value));
    }

    fn pop_item(&mut self, context: &'static str) -> Result<StdPickleItem, BuiltinError> {
        self.stack
            .pop()
            .ok_or_else(|| invalid_pickle(format!("missing {context}")))
    }

    fn pop_value(&mut self, context: &'static str) -> Result<Value, BuiltinError> {
        let item = self.pop_item(context)?;
        item.into_value(context)
    }

    fn peek_value(&self, context: &'static str) -> Result<Value, BuiltinError> {
        self.stack
            .last()
            .ok_or_else(|| invalid_pickle(format!("missing {context}")))?
            .clone()
            .into_value(context)
    }

    fn pop_string(&mut self, context: &'static str) -> Result<String, BuiltinError> {
        match self.pop_item(context)? {
            StdPickleItem::String(value) => Ok(value),
            _ => Err(invalid_pickle(format!("{context} is not a string"))),
        }
    }

    fn push_mark_tuple(&mut self) -> Result<(), BuiltinError> {
        let mark = self
            .stack
            .iter()
            .rposition(|item| matches!(item, StdPickleItem::Mark))
            .ok_or_else(|| invalid_pickle("tuple opcode without mark"))?;
        let tail = self.stack.split_off(mark + 1);
        self.stack.pop();
        let values = tail
            .into_iter()
            .map(|item| item.into_value("tuple item"))
            .collect::<Result<Vec<_>, _>>()?;
        self.push_value(tuple_value(values));
        Ok(())
    }

    fn push_fixed_tuple(&mut self, len: usize) -> Result<(), BuiltinError> {
        if self.stack.len() < len {
            return Err(invalid_pickle("tuple opcode underflow"));
        }
        let start = self.stack.len() - len;
        let tail = self.stack.split_off(start);
        let values = tail
            .into_iter()
            .map(|item| item.into_value("tuple item"))
            .collect::<Result<Vec<_>, _>>()?;
        self.push_value(tuple_value(values));
        Ok(())
    }

    fn memoize_top(&mut self, index: usize) -> Result<(), BuiltinError> {
        let item = self
            .stack
            .last()
            .ok_or_else(|| invalid_pickle("memo opcode without stack value"))?
            .clone();
        if matches!(item, StdPickleItem::Mark) {
            return Err(invalid_pickle("cannot memoize pickle mark"));
        }
        if index >= self.memo.len() {
            self.memo.resize(index + 1, None);
        }
        self.memo[index] = Some(item);
        Ok(())
    }

    fn push_memo(&mut self, index: usize) -> Result<(), BuiltinError> {
        let item = self
            .memo
            .get(index)
            .and_then(|slot| slot.clone())
            .ok_or_else(|| invalid_pickle("invalid pickle memo reference"))?;
        self.stack.push(item);
        Ok(())
    }

    fn read_decimal_line(&mut self, context: &'static str) -> Result<BigInt, BuiltinError> {
        let text = self.read_utf8_line(context)?;
        parse_decimal_bigint(&text).ok_or_else(|| invalid_pickle(format!("invalid {context}")))
    }

    fn read_long_line(&mut self) -> Result<BigInt, BuiltinError> {
        let text = self.read_utf8_line("pickle long")?;
        let text = text
            .strip_suffix('L')
            .or_else(|| text.strip_suffix('l'))
            .unwrap_or(&text);
        parse_decimal_bigint(text).ok_or_else(|| invalid_pickle("invalid pickle long"))
    }

    fn read_memo_line(&mut self, context: &'static str) -> Result<usize, BuiltinError> {
        let text = self.read_utf8_line(context)?;
        text.parse::<usize>()
            .map_err(|_| invalid_pickle(format!("invalid {context}")))
    }

    fn read_utf8_line(&mut self, context: &'static str) -> Result<String, BuiltinError> {
        let line = self.read_line()?;
        std::str::from_utf8(line)
            .map(str::to_owned)
            .map_err(|_| invalid_pickle(format!("invalid UTF-8 {context}")))
    }

    fn read_utf8_exact(
        &mut self,
        len: usize,
        context: &'static str,
    ) -> Result<String, BuiltinError> {
        let bytes = self.read_exact(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| invalid_pickle(format!("invalid UTF-8 {context}")))
    }

    fn read_line(&mut self) -> Result<&'bytes [u8], BuiltinError> {
        let start = self.pos;
        while self.pos < self.bytes.len() {
            if self.bytes[self.pos] == b'\n' {
                let line = &self.bytes[start..self.pos];
                self.pos += 1;
                return Ok(line);
            }
            self.pos += 1;
        }
        Err(invalid_pickle("unterminated pickle line"))
    }

    fn read_u8(&mut self) -> Result<u8, BuiltinError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, BuiltinError> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("read_exact returned 4 bytes"),
        ))
    }

    fn read_u64(&mut self) -> Result<u64, BuiltinError> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("read_exact returned 8 bytes"),
        ))
    }

    fn read_u32_as_usize(&mut self, context: &'static str) -> Result<usize, BuiltinError> {
        usize::try_from(self.read_u32()?)
            .map_err(|_| invalid_pickle(format!("{context} does not fit in memory")))
    }

    fn read_u64_as_usize(&mut self, context: &'static str) -> Result<usize, BuiltinError> {
        usize::try_from(self.read_u64()?)
            .map_err(|_| invalid_pickle(format!("{context} does not fit in memory")))
    }

    fn read_exact(&mut self, len: usize) -> Result<&'bytes [u8], BuiltinError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| invalid_pickle("pickle payload offset overflow"))?;
        if end > self.bytes.len() {
            return Err(invalid_pickle("truncated pickle payload"));
        }
        let bytes = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }
}

impl StdPickleItem {
    fn into_value(self, context: &'static str) -> Result<Value, BuiltinError> {
        match self {
            Self::Value(value) => Ok(value),
            Self::String(value) => Ok(Value::string(intern(&value))),
            Self::Mark => Err(invalid_pickle(format!("{context} is a mark"))),
            Self::Callable(_) => Err(invalid_pickle(format!("{context} is not a value"))),
        }
    }
}

fn resolve_std_global(
    vm: &VirtualMachine,
    module: &str,
    name: &str,
) -> Result<StdPickleItem, BuiltinError> {
    if matches!(module, "builtins" | "__builtin__") {
        return match name {
            "iter" => Ok(StdPickleItem::Callable(StdPickleCallable::Iter)),
            "range" | "xrange" => Ok(StdPickleItem::Callable(StdPickleCallable::Range)),
            _ => resolve_builtin(vm, name)
                .map(StdPickleItem::Value)
                .ok_or_else(|| {
                    invalid_pickle(format!("unsupported pickle global {module}.{name}"))
                }),
        };
    }

    Err(invalid_pickle(format!(
        "unsupported pickle global {module}.{name}"
    )))
}

fn call_std_reduce(
    vm: &mut VirtualMachine,
    callable: StdPickleItem,
    args_value: Value,
) -> Result<Value, BuiltinError> {
    let args_tuple = value_as_tuple_ref(args_value)
        .ok_or_else(|| invalid_pickle("REDUCE arguments are not a tuple"))?;
    let args = args_tuple.as_slice();

    match callable {
        StdPickleItem::Callable(StdPickleCallable::Range) => reduce_std_range(args),
        StdPickleItem::Callable(StdPickleCallable::Iter) => {
            let iter = vm
                .builtin_value("iter")
                .ok_or_else(|| invalid_pickle("cannot resolve builtins.iter"))?;
            invoke_callable_value(vm, iter, args).map_err(runtime_error_to_builtin_error)
        }
        StdPickleItem::Value(value) => {
            invoke_callable_value(vm, value, args).map_err(runtime_error_to_builtin_error)
        }
        _ => Err(invalid_pickle("REDUCE callable is not callable")),
    }
}

fn reduce_std_range(args: &[Value]) -> Result<Value, BuiltinError> {
    let (start, stop, step) = match args {
        [stop] => (
            BigInt::from(0),
            std_pickle_index(*stop, "range stop")?,
            BigInt::from(1),
        ),
        [start, stop] => (
            std_pickle_index(*start, "range start")?,
            std_pickle_index(*stop, "range stop")?,
            BigInt::from(1),
        ),
        [start, stop, step] => (
            std_pickle_index(*start, "range start")?,
            std_pickle_index(*stop, "range stop")?,
            std_pickle_index(*step, "range step")?,
        ),
        _ => {
            return Err(invalid_pickle(
                "range reducer expects one to three arguments",
            ));
        }
    };
    if step == BigInt::from(0) {
        return Err(invalid_pickle("range step cannot be zero"));
    }

    Ok(crate::alloc_managed_value(RangeObject::from_bigints(
        start, stop, step,
    )))
}

fn std_pickle_index(value: Value, context: &'static str) -> Result<BigInt, BuiltinError> {
    value_to_bigint(value).ok_or_else(|| invalid_pickle(format!("{context} is not an integer")))
}

fn parse_decimal_bigint(text: &str) -> Option<BigInt> {
    let text = text.trim();
    let text = text.strip_prefix('+').unwrap_or(text);
    if text.is_empty() {
        return None;
    }
    BigInt::parse_bytes(text.as_bytes(), 10)
}

#[inline]
fn tuple_value(values: Vec<Value>) -> Value {
    crate::alloc_managed_value(TupleObject::from_vec(values))
}

fn pickle_dumps(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "dumps() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut protocol = args.get(1).copied();
    for (name, value) in keywords {
        match *name {
            "protocol" => assign_keyword(&mut protocol, *value, "protocol")?,
            "fix_imports" | "buffer_callback" => {}
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "dumps() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    let protocol = protocol
        .filter(|value| !value.is_none())
        .map(protocol_number)
        .transpose()?
        .unwrap_or(DEFAULT_PROTOCOL);
    let protocol = normalize_protocol(protocol)?;

    if let Some(boolean) = args[0].as_bool() {
        return Ok(bytes_value(&encode_bool(boolean, protocol)));
    }

    let mut writer = PickleWriter::new(vm);
    writer.write_value(args[0])?;
    let encoded = writer.finish();
    Ok(bytes_value(&encoded))
}

fn pickle_loads(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "loads() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let bytes = bytes_from_value(args[0], "loads() argument")?;
    if let Some(value) = decode_bool_pickle(&bytes) {
        return Ok(value);
    }

    let Some(payload) = bytes.strip_prefix(PRISM_PICKLE_MAGIC) else {
        return StdPickleReader::new(&bytes).read(vm);
    };

    let mut reader = PickleReader::new(payload);
    let value = reader.read_value(vm)?;
    reader.expect_end()?;
    Ok(value)
}

fn reconstruct_reduce(vm: &mut VirtualMachine, reduction: Value) -> Result<Value, BuiltinError> {
    let tuple = value_as_tuple_ref(reduction)
        .ok_or_else(|| invalid_pickle("reducer payload is not a tuple"))?;
    if !(2..=3).contains(&tuple.len()) {
        return Err(invalid_pickle("reducer tuple must have two or three items"));
    }

    let callable = tuple
        .get(0)
        .ok_or_else(|| invalid_pickle("missing reducer callable"))?;
    let args_value = tuple
        .get(1)
        .ok_or_else(|| invalid_pickle("missing reducer argument tuple"))?;
    let args_tuple = value_as_tuple_ref(args_value)
        .ok_or_else(|| invalid_pickle("reducer arguments are not a tuple"))?;
    let result = invoke_callable_value(vm, callable, args_tuple.as_slice())
        .map_err(runtime_error_to_builtin_error)?;

    if let Some(state) = tuple.get(2) {
        restore_reduce_state(vm, result, state)?;
    }

    Ok(result)
}

fn restore_reduce_state(
    vm: &mut VirtualMachine,
    object: Value,
    state: Value,
) -> Result<(), BuiltinError> {
    if let Some(iter) = get_iterator_mut(&object) {
        let state = value_to_bigint(state)
            .ok_or_else(|| invalid_pickle("iterator reducer state is not an integer"))?;
        iter.set_state_bigint(&state);
        return Ok(());
    }

    let setstate = match get_attribute_value(vm, object, &intern("__setstate__")) {
        Ok(value) => value,
        Err(err) if err.is_attribute_error() => {
            return Err(invalid_pickle("reducer state target has no __setstate__"));
        }
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };
    invoke_callable_value(vm, setstate, &[state])
        .map(|_| ())
        .map_err(runtime_error_to_builtin_error)
}

fn resolve_builtin(vm: &VirtualMachine, name: &str) -> Option<Value> {
    vm.builtin_value(name).or_else(|| {
        name.rsplit_once('.')
            .and_then(|(_, short_name)| vm.builtin_value(short_name))
    })
}

fn resolve_user_class(
    vm: &mut VirtualMachine,
    class_id: u32,
    module: &str,
    name: &str,
) -> Result<Arc<PyClassObject>, BuiltinError> {
    if let Some(class) = global_class(ClassId(class_id)) {
        return Ok(class);
    }

    if module == "__main__" {
        let lookup_name = Arc::<str>::from(name);
        if let Some(value) = vm.module_scope_value(&lookup_name)
            && let Some(class) = class_from_type_value(value)
        {
            return Ok(class);
        }
    }

    if !module.is_empty()
        && let Ok(module_object) = vm.import_module_named(module)
        && let Some(value) = module_object.get_attr(name)
        && let Some(class) = class_from_type_value(value)
    {
        return Ok(class);
    }

    Err(invalid_pickle(format!(
        "cannot resolve class '{}.{}'",
        module, name
    )))
}

fn class_from_type_value(value: Value) -> Option<Arc<PyClassObject>> {
    let ptr = value.as_object_ptr()?;
    if extract_type_id(ptr) != TypeId::TYPE || builtin_type_object_type_id(ptr).is_some() {
        return None;
    }
    let class = unsafe { &*(ptr as *const PyClassObject) };
    global_class(class.class_id())
}

fn class_export_name(class: &PyClassObject) -> (String, String) {
    let module = class
        .get_attr(&intern("__module__"))
        .and_then(|value| value_as_string_ref(value))
        .map(|value| value.as_str().to_owned())
        .unwrap_or_else(|| "__main__".to_string());
    (module, class.name().as_str().to_owned())
}

fn checked_u32(value: usize, context: &'static str) -> Result<u32, BuiltinError> {
    u32::try_from(value).map_err(|_| {
        BuiltinError::OverflowError(format!("{context} is too large to encode in Prism pickle"))
    })
}

fn cannot_pickle(value: Value) -> BuiltinError {
    BuiltinError::TypeError(format!("cannot pickle '{}' objects yet", value.type_name()))
}

fn invalid_pickle(message: impl Into<String>) -> BuiltinError {
    BuiltinError::ValueError(message.into())
}

fn assign_keyword(
    slot: &mut Option<Value>,
    value: Value,
    name: &'static str,
) -> Result<(), BuiltinError> {
    if slot.is_some() {
        return Err(BuiltinError::TypeError(format!(
            "dumps() got multiple values for argument '{}'",
            name
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn protocol_number(value: Value) -> Result<i64, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(i64::from(flag));
    }
    if let Some(integer) = value.as_int() {
        return Ok(integer);
    }
    value_to_bigint(value)
        .and_then(|integer| integer.to_i64())
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "pickle protocol must be int, not {}",
                value.type_name()
            ))
        })
}

fn normalize_protocol(protocol: i64) -> Result<i64, BuiltinError> {
    if protocol == -1 {
        return Ok(HIGHEST_PROTOCOL);
    }
    if (0..=HIGHEST_PROTOCOL).contains(&protocol) {
        Ok(protocol)
    } else {
        Err(BuiltinError::ValueError(format!(
            "pickle protocol must be <= {HIGHEST_PROTOCOL}"
        )))
    }
}

fn encode_bool(value: bool, protocol: i64) -> Vec<u8> {
    if protocol < 2 {
        return if value {
            TRUE_PROTO0.to_vec()
        } else {
            FALSE_PROTO0.to_vec()
        };
    }

    vec![
        PROTO,
        u8::try_from(protocol).expect("protocol is normalized to a byte"),
        if value { NEWTRUE } else { NEWFALSE },
        STOP,
    ]
}

fn bytes_from_value(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        )));
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec()),
        _ => Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        ))),
    }
}

fn decode_bool_pickle(bytes: &[u8]) -> Option<Value> {
    match bytes {
        TRUE_PROTO0 => Some(Value::bool(true)),
        FALSE_PROTO0 => Some(Value::bool(false)),
        [PROTO, protocol, NEWTRUE, STOP] if i64::from(*protocol) <= HIGHEST_PROTOCOL => {
            Some(Value::bool(true))
        }
        [PROTO, protocol, NEWFALSE, STOP] if i64::from(*protocol) <= HIGHEST_PROTOCOL => {
            Some(Value::bool(false))
        }
        _ => None,
    }
}
