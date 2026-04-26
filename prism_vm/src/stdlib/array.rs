//! Native `array` module.
//!
//! CPython ships `array` as a C extension. Prism models it as a native heap
//! type backed by compact bytes plus a validated typecode, which gives stdlib
//! users the import-time and buffer-adjacent behavior they expect while keeping
//! room for a full buffer-protocol implementation.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, global_class_bitmap, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::value_to_i64;
use prism_runtime::types::iter::IteratorObject;
use prism_runtime::types::list::{ListObject, value_as_list_ref};
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const MODULE_DOC: &str = "Efficient arrays of uniformly typed values.";
const TYPECODES: &str = "bBuhHiIlLqQfd";
const ARRAY_TYPECODE_ATTR: &str = "__prism_array_typecode__";
const ARRAY_BYTES_ATTR: &str = "__prism_array_bytes__";

#[cfg(windows)]
const WCHAR_ITEMSIZE: usize = 2;
#[cfg(not(windows))]
const WCHAR_ITEMSIZE: usize = 4;

static ARRAY_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_array_class);

static ARRAY_INIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("array.array.__init__"), array_init));
static ARRAY_LEN_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("array.array.__len__"), array_len));
static ARRAY_ITER_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("array.array.__iter__"), array_iter));
static ARRAY_GETITEM_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("array.array.__getitem__"), array_getitem)
});
static ARRAY_APPEND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("array.array.append"), array_append));
static ARRAY_FROMBYTES_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("array.array.frombytes"), array_frombytes)
});
static ARRAY_TOBYTES_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("array.array.tobytes"), array_tobytes));
static ARRAY_TOLIST_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("array.array.tolist"), array_tolist));
static ARRAY_RECONSTRUCTOR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("array._array_reconstructor"), array_reconstructor)
});

/// Native `array` module descriptor.
#[derive(Debug, Clone)]
pub struct ArrayModule {
    attrs: Vec<Arc<str>>,
}

impl ArrayModule {
    /// Create a new `array` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__doc__"),
                Arc::from("_array_reconstructor"),
                Arc::from("array"),
                Arc::from("typecodes"),
            ],
        }
    }
}

impl Default for ArrayModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ArrayModule {
    fn name(&self) -> &str {
        "array"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "array" => Ok(array_class_value()),
            "typecodes" => Ok(Value::string(intern(TYPECODES))),
            "_array_reconstructor" => Ok(builtin_value(&ARRAY_RECONSTRUCTOR_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'array' has no attribute '{}'",
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
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

#[inline]
fn array_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&ARRAY_CLASS) as *const ())
}

fn build_array_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("array"));
    class.set_attr(intern("__module__"), Value::string(intern("array")));
    class.set_attr(intern("__qualname__"), Value::string(intern("array")));
    class.set_attr(
        intern("__doc__"),
        Value::string(intern("array(typecode[, initializer])")),
    );
    class.set_attr(intern("__init__"), builtin_value(&ARRAY_INIT_METHOD));
    class.set_attr(intern("__len__"), builtin_value(&ARRAY_LEN_METHOD));
    class.set_attr(intern("__iter__"), builtin_value(&ARRAY_ITER_METHOD));
    class.set_attr(intern("__getitem__"), builtin_value(&ARRAY_GETITEM_METHOD));
    class.set_attr(intern("append"), builtin_value(&ARRAY_APPEND_METHOD));
    class.set_attr(intern("frombytes"), builtin_value(&ARRAY_FROMBYTES_METHOD));
    class.set_attr(intern("tobytes"), builtin_value(&ARRAY_TOBYTES_METHOD));
    class.set_attr(intern("tolist"), builtin_value(&ARRAY_TOLIST_METHOD));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_INIT | ClassFlags::NATIVE_HEAPTYPE);

    let class = Arc::new(class);
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);
    class
}

fn array_init(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "array() takes 1 or 2 positional arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let typecode = typecode_arg(args[1])?;
    let spec = TypeSpec::from_typecode(typecode)?;
    let bytes = match args.get(2).copied() {
        Some(initializer) => encode_initializer(spec, initializer)?,
        None => Vec::new(),
    };

    let object = array_object_mut(args[0])?;
    set_array_state(object, spec, bytes);
    Ok(Value::none())
}

fn array_len(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "__len__", 1)?;
    let spec = array_spec(args[0])?;
    let len = array_bytes(args[0])?.len() / spec.itemsize;
    Value::int(
        i64::try_from(len).map_err(|_| {
            BuiltinError::OverflowError("array length does not fit in int".to_string())
        })?,
    )
    .ok_or_else(|| BuiltinError::OverflowError("array length does not fit in int".to_string()))
}

fn array_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "__iter__", 1)?;
    Ok(crate::builtins::iterator_to_value(
        IteratorObject::from_values(array_values(args[0])?),
    ))
}

fn array_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "__getitem__", 2)?;

    if let Some(slice) = slice_arg(args[1]) {
        return array_slice(args[0], slice);
    }

    let values = array_values(args[0])?;
    let index = value_to_i64(args[1])
        .ok_or_else(|| BuiltinError::TypeError("array indices must be integers".to_string()))?;
    let len = i64::try_from(values.len())
        .map_err(|_| BuiltinError::OverflowError("array length overflow".to_string()))?;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Err(BuiltinError::IndexError(
            "array index out of range".to_string(),
        ));
    }
    Ok(values[normalized as usize])
}

fn array_slice(value: Value, slice: &SliceObject) -> Result<Value, BuiltinError> {
    let spec = array_spec(value)?;
    let bytes = array_bytes(value)?;
    let len = bytes.len() / spec.itemsize;
    let indices = slice.indices(len);
    let capacity = indices
        .length
        .checked_mul(spec.itemsize)
        .ok_or_else(|| BuiltinError::OverflowError("array slice is too large".to_string()))?;
    let mut sliced = Vec::with_capacity(capacity);

    if indices.step == 1 {
        let start = indices
            .start
            .checked_mul(spec.itemsize)
            .ok_or_else(|| BuiltinError::OverflowError("array slice is too large".to_string()))?;
        let end = start
            .checked_add(capacity)
            .ok_or_else(|| BuiltinError::OverflowError("array slice is too large".to_string()))?;
        sliced.extend_from_slice(&bytes[start..end]);
    } else {
        for index in indices.iter() {
            let start = index.checked_mul(spec.itemsize).ok_or_else(|| {
                BuiltinError::OverflowError("array slice is too large".to_string())
            })?;
            let end = start.checked_add(spec.itemsize).ok_or_else(|| {
                BuiltinError::OverflowError("array slice is too large".to_string())
            })?;
            sliced.extend_from_slice(&bytes[start..end]);
        }
    }

    Ok(new_array_value(spec, sliced))
}

fn array_append(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "append", 2)?;
    let spec = array_spec(args[0])?;
    let mut bytes = array_bytes(args[0])?;
    encode_element(spec, args[1], &mut bytes)?;
    set_array_bytes(array_object_mut(args[0])?, bytes);
    Ok(Value::none())
}

fn array_frombytes(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "frombytes", 2)?;
    let spec = array_spec(args[0])?;
    let incoming = bytes_arg(args[1], "frombytes() argument must be bytes-like")?;
    if incoming.len() % spec.itemsize != 0 {
        return Err(BuiltinError::ValueError(
            "bytes length not a multiple of item size".to_string(),
        ));
    }
    let mut bytes = array_bytes(args[0])?;
    bytes.extend_from_slice(&incoming);
    set_array_bytes(array_object_mut(args[0])?, bytes);
    Ok(Value::none())
}

fn array_tobytes(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "tobytes", 1)?;
    Ok(leak_object_value(BytesObject::from_vec(array_bytes(
        args[0],
    )?)))
}

fn array_tolist(args: &[Value]) -> Result<Value, BuiltinError> {
    parse_exact_arity(args, "tolist", 1)?;
    Ok(leak_object_value(ListObject::from_iter(array_values(
        args[0],
    )?)))
}

fn array_reconstructor(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "_array_reconstructor() takes exactly 4 arguments ({} given)",
            args.len()
        )));
    }
    Err(BuiltinError::NotImplemented(
        "_array_reconstructor() is not available yet".to_string(),
    ))
}

fn parse_exact_arity(args: &[Value], fn_name: &str, expected: usize) -> Result<(), BuiltinError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes exactly {expected} argument{} ({} given)",
            if expected == 1 { "" } else { "s" },
            args.len()
        )))
    }
}

fn typecode_arg(value: Value) -> Result<char, BuiltinError> {
    let text = value_as_string_ref(value)
        .map(|text| text.as_str().to_string())
        .ok_or_else(|| {
            BuiltinError::TypeError("array() argument 1 must be a unicode character".to_string())
        })?;
    let mut chars = text.chars();
    let Some(typecode) = chars.next() else {
        return Err(BuiltinError::TypeError(
            "array() argument 1 must be a unicode character".to_string(),
        ));
    };
    if chars.next().is_some() {
        return Err(BuiltinError::TypeError(
            "array() argument 1 must be a unicode character".to_string(),
        ));
    }
    Ok(typecode)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TypeSpec {
    typecode: char,
    itemsize: usize,
    kind: ElementKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ElementKind {
    Signed,
    Unsigned,
    Float,
    Unicode,
}

impl TypeSpec {
    fn from_typecode(typecode: char) -> Result<Self, BuiltinError> {
        let (itemsize, kind) = match typecode {
            'b' => (1, ElementKind::Signed),
            'B' => (1, ElementKind::Unsigned),
            'u' => (WCHAR_ITEMSIZE, ElementKind::Unicode),
            'h' => (2, ElementKind::Signed),
            'H' => (2, ElementKind::Unsigned),
            'i' | 'l' => (4, ElementKind::Signed),
            'I' | 'L' => (4, ElementKind::Unsigned),
            'q' => (8, ElementKind::Signed),
            'Q' => (8, ElementKind::Unsigned),
            'f' => (4, ElementKind::Float),
            'd' => (8, ElementKind::Float),
            _ => {
                return Err(BuiltinError::ValueError(format!(
                    "bad typecode (must be one of {TYPECODES})"
                )));
            }
        };
        Ok(Self {
            typecode,
            itemsize,
            kind,
        })
    }
}

fn encode_initializer(spec: TypeSpec, initializer: Value) -> Result<Vec<u8>, BuiltinError> {
    if initializer.is_none() {
        return Ok(Vec::new());
    }
    if let Ok(bytes) = array_bytes(initializer) {
        return Ok(bytes);
    }
    if let Ok(bytes) = bytes_arg(initializer, "array initializer must be iterable") {
        if matches!(spec.typecode, 'b' | 'B') {
            return Ok(bytes);
        }
        let mut encoded = Vec::with_capacity(bytes.len() * spec.itemsize);
        for byte in bytes {
            encode_integer_element(spec, i64::from(byte), &mut encoded)?;
        }
        return Ok(encoded);
    }

    let mut encoded = Vec::new();
    if let Some(list) = value_as_list_ref(initializer) {
        for &value in list.as_slice() {
            encode_element(spec, value, &mut encoded)?;
        }
        return Ok(encoded);
    }

    let Some(ptr) = initializer.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "array initializer must be iterable".to_string(),
        ));
    };
    if crate::ops::objects::extract_type_id(ptr) == TypeId::TUPLE {
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        for &value in tuple.as_slice() {
            encode_element(spec, value, &mut encoded)?;
        }
        return Ok(encoded);
    }

    Err(BuiltinError::TypeError(
        "array initializer must be iterable".to_string(),
    ))
}

fn encode_element(spec: TypeSpec, value: Value, out: &mut Vec<u8>) -> Result<(), BuiltinError> {
    match spec.kind {
        ElementKind::Signed | ElementKind::Unsigned | ElementKind::Unicode => {
            let integer = value_to_i64(value).ok_or_else(|| {
                BuiltinError::TypeError("array item must be an integer".to_string())
            })?;
            encode_integer_element(spec, integer, out)
        }
        ElementKind::Float => {
            let float = value
                .as_float()
                .or_else(|| value_to_i64(value).map(|i| i as f64))
                .ok_or_else(|| {
                    BuiltinError::TypeError("array item must be a number".to_string())
                })?;
            match spec.itemsize {
                4 => out.extend_from_slice(&(float as f32).to_ne_bytes()),
                8 => out.extend_from_slice(&float.to_ne_bytes()),
                _ => unreachable!("float itemsize is validated by TypeSpec"),
            }
            Ok(())
        }
    }
}

fn encode_integer_element(
    spec: TypeSpec,
    integer: i64,
    out: &mut Vec<u8>,
) -> Result<(), BuiltinError> {
    match (spec.kind, spec.itemsize) {
        (ElementKind::Signed, 1) => out.extend_from_slice(
            &i8::try_from(integer)
                .map_err(|_| overflow_error(spec))?
                .to_ne_bytes(),
        ),
        (ElementKind::Unsigned, 1) => {
            out.push(u8::try_from(integer).map_err(|_| overflow_error(spec))?)
        }
        (ElementKind::Signed, 2) => out.extend_from_slice(
            &i16::try_from(integer)
                .map_err(|_| overflow_error(spec))?
                .to_ne_bytes(),
        ),
        (ElementKind::Unsigned, 2) | (ElementKind::Unicode, 2) => out.extend_from_slice(
            &u16::try_from(integer)
                .map_err(|_| overflow_error(spec))?
                .to_ne_bytes(),
        ),
        (ElementKind::Signed, 4) => out.extend_from_slice(
            &i32::try_from(integer)
                .map_err(|_| overflow_error(spec))?
                .to_ne_bytes(),
        ),
        (ElementKind::Unsigned, 4) | (ElementKind::Unicode, 4) => out.extend_from_slice(
            &u32::try_from(integer)
                .map_err(|_| overflow_error(spec))?
                .to_ne_bytes(),
        ),
        (ElementKind::Signed, 8) => out.extend_from_slice(&integer.to_ne_bytes()),
        (ElementKind::Unsigned, 8) => out.extend_from_slice(
            &u64::try_from(integer)
                .map_err(|_| overflow_error(spec))?
                .to_ne_bytes(),
        ),
        _ => unreachable!("integer itemsize is validated by TypeSpec"),
    }
    Ok(())
}

fn decode_values(spec: TypeSpec, bytes: &[u8]) -> Result<Vec<Value>, BuiltinError> {
    let mut values = Vec::with_capacity(bytes.len() / spec.itemsize);
    for chunk in bytes.chunks_exact(spec.itemsize) {
        let value = match (spec.kind, spec.itemsize) {
            (ElementKind::Signed, 1) => i64::from(i8::from_ne_bytes([chunk[0]])),
            (ElementKind::Unsigned, 1) => i64::from(chunk[0]),
            (ElementKind::Signed, 2) => i64::from(i16::from_ne_bytes([chunk[0], chunk[1]])),
            (ElementKind::Unsigned, 2) | (ElementKind::Unicode, 2) => {
                i64::from(u16::from_ne_bytes([chunk[0], chunk[1]]))
            }
            (ElementKind::Signed, 4) => {
                i64::from(i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }
            (ElementKind::Unsigned, 4) | (ElementKind::Unicode, 4) => {
                i64::from(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }
            (ElementKind::Signed, 8) => i64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            (ElementKind::Unsigned, 8) => {
                let unsigned = u64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                i64::try_from(unsigned).map_err(|_| {
                    BuiltinError::OverflowError("array item does not fit in Prism int".to_string())
                })?
            }
            (ElementKind::Float, 4) => {
                values.push(Value::float(f32::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ]) as f64));
                continue;
            }
            (ElementKind::Float, 8) => {
                values.push(Value::float(f64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])));
                continue;
            }
            _ => unreachable!("itemsize is validated by TypeSpec"),
        };
        values.push(
            Value::int(value)
                .ok_or_else(|| BuiltinError::OverflowError("array item overflow".to_string()))?,
        );
    }
    Ok(values)
}

fn overflow_error(spec: TypeSpec) -> BuiltinError {
    BuiltinError::OverflowError(format!(
        "signed/unsigned overflow for typecode '{}'",
        spec.typecode
    ))
}

fn array_object_mut(value: Value) -> Result<&'static mut ShapedObject, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("array methods require an instance".to_string()))?;
    if !is_array_type_id(crate::ops::objects::extract_type_id(ptr)) {
        return Err(BuiltinError::TypeError(
            "array methods require an array instance".to_string(),
        ));
    }
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn array_object_ref(value: Value) -> Result<&'static ShapedObject, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("array methods require an instance".to_string()))?;
    if !is_array_type_id(crate::ops::objects::extract_type_id(ptr)) {
        return Err(BuiltinError::TypeError(
            "array methods require an array instance".to_string(),
        ));
    }
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

#[inline]
fn is_array_type_id(type_id: TypeId) -> bool {
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return false;
    }

    let array_type = ARRAY_CLASS.class_type_id();
    type_id == array_type
        || global_class_bitmap(ClassId(type_id.raw()))
            .is_some_and(|bitmap| bitmap.is_subclass_of(array_type))
}

fn set_array_state(object: &mut ShapedObject, spec: TypeSpec, bytes: Vec<u8>) {
    let registry = shape_registry();
    object.set_property(
        intern(ARRAY_TYPECODE_ATTR),
        Value::string(intern(&spec.typecode.to_string())),
        registry,
    );
    object.set_property(
        intern("typecode"),
        Value::string(intern(&spec.typecode.to_string())),
        registry,
    );
    object.set_property(
        intern("itemsize"),
        Value::int(spec.itemsize as i64).expect("array item size should fit"),
        registry,
    );
    set_array_bytes(object, bytes);
}

fn new_array_value(spec: TypeSpec, bytes: Vec<u8>) -> Value {
    let mut instance = crate::builtins::allocate_heap_instance_for_class(&ARRAY_CLASS);
    set_array_state(&mut instance, spec, bytes);
    leak_object_value(instance)
}

fn set_array_bytes(object: &mut ShapedObject, bytes: Vec<u8>) {
    object.set_property(
        intern(ARRAY_BYTES_ATTR),
        leak_object_value(BytesObject::bytearray_from_slice(&bytes)),
        shape_registry(),
    );
}

fn array_spec(value: Value) -> Result<TypeSpec, BuiltinError> {
    let typecode_value = array_object_ref(value)?
        .get_property(ARRAY_TYPECODE_ATTR)
        .ok_or_else(|| BuiltinError::TypeError("uninitialized array object".to_string()))?;
    TypeSpec::from_typecode(typecode_arg(typecode_value)?)
}

fn array_bytes(value: Value) -> Result<Vec<u8>, BuiltinError> {
    let bytes_value = array_object_ref(value)?
        .get_property(ARRAY_BYTES_ATTR)
        .ok_or_else(|| BuiltinError::TypeError("uninitialized array object".to_string()))?;
    bytes_arg(bytes_value, "invalid array storage")
}

pub(crate) fn value_as_array_bytes(value: Value) -> Result<Option<Vec<u8>>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };
    if !is_array_type_id(crate::ops::objects::extract_type_id(ptr)) {
        return Ok(None);
    }
    array_bytes(value).map(Some)
}

fn array_values(value: Value) -> Result<Vec<Value>, BuiltinError> {
    let spec = array_spec(value)?;
    let bytes = array_bytes(value)?;
    decode_values(spec, &bytes)
}

fn slice_arg(value: Value) -> Option<&'static SliceObject> {
    let ptr = value.as_object_ptr()?;
    (crate::ops::objects::extract_type_id(ptr) == TypeId::SLICE)
        .then(|| unsafe { &*(ptr as *const SliceObject) })
}

fn bytes_arg(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(context.to_string()))?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec()),
        _ => Err(BuiltinError::TypeError(context.to_string())),
    }
}

#[cfg(test)]
mod tests;
