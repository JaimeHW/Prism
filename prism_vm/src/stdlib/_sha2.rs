//! Native `_sha2` compatibility layer.
//!
//! CPython's pure-Python `random.py` and `hashlib.py` import constructors from
//! `_sha2`. Prism only needs a compact, correct SHA-2 surface here: module
//! constructors plus hash objects that support the standard `update`,
//! `digest`, `hexdigest`, and `copy` APIs.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, global_class, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::string::StringObject;
use sha2::{Digest, Sha224, Sha256, Sha384, Sha512};
use std::sync::{Arc, LazyLock};

const BUFFER_ATTR: &str = "__prism_sha2_buffer__";
const MODULE_DOC: &str = "Native compatibility implementation of the _sha2 module.";
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sha2Kind {
    Sha224,
    Sha256,
    Sha384,
    Sha512,
}

impl Sha2Kind {
    const ALL: [Self; 4] = [Self::Sha224, Self::Sha256, Self::Sha384, Self::Sha512];

    #[inline]
    fn constructor_name(self) -> &'static str {
        match self {
            Self::Sha224 => "sha224",
            Self::Sha256 => "sha256",
            Self::Sha384 => "sha384",
            Self::Sha512 => "sha512",
        }
    }

    #[inline]
    fn class_name(self) -> &'static str {
        match self {
            Self::Sha224 => "SHA224Type",
            Self::Sha256 => "SHA256Type",
            Self::Sha384 => "SHA384Type",
            Self::Sha512 => "SHA512Type",
        }
    }

    #[inline]
    fn digest_size(self) -> usize {
        match self {
            Self::Sha224 => 28,
            Self::Sha256 => 32,
            Self::Sha384 => 48,
            Self::Sha512 => 64,
        }
    }

    #[inline]
    fn block_size(self) -> usize {
        match self {
            Self::Sha224 | Self::Sha256 => 64,
            Self::Sha384 | Self::Sha512 => 128,
        }
    }

    #[inline]
    fn class(self) -> &'static Arc<PyClassObject> {
        match self {
            Self::Sha224 => &SHA224_CLASS,
            Self::Sha256 => &SHA256_CLASS,
            Self::Sha384 => &SHA384_CLASS,
            Self::Sha512 => &SHA512_CLASS,
        }
    }
}

static SHA224_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_hash_class(Sha2Kind::Sha224));
static SHA256_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_hash_class(Sha2Kind::Sha256));
static SHA384_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_hash_class(Sha2Kind::Sha384));
static SHA512_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_hash_class(Sha2Kind::Sha512));

static HASH_UPDATE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_sha2.hash.update"), hash_update));
static HASH_DIGEST_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_sha2.hash.digest"), hash_digest));
static HASH_HEXDIGEST_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_sha2.hash.hexdigest"), hash_hexdigest));
static HASH_COPY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_sha2.hash.copy"), hash_copy));

static SHA224_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("_sha2.sha224"), sha224_constructor));
static SHA256_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("_sha2.sha256"), sha256_constructor));
static SHA384_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("_sha2.sha384"), sha384_constructor));
static SHA512_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("_sha2.sha512"), sha512_constructor));

/// Native `_sha2` module descriptor.
#[derive(Debug, Clone)]
pub struct Sha2Module {
    attrs: Vec<Arc<str>>,
}

impl Sha2Module {
    /// Create a new `_sha2` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__doc__"),
                Arc::from("SHA224Type"),
                Arc::from("SHA256Type"),
                Arc::from("SHA384Type"),
                Arc::from("SHA512Type"),
                Arc::from("sha224"),
                Arc::from("sha256"),
                Arc::from("sha384"),
                Arc::from("sha512"),
            ],
        }
    }
}

impl Default for Sha2Module {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sha2Module {
    fn name(&self) -> &str {
        "_sha2"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "SHA224Type" => Ok(class_value(Sha2Kind::Sha224)),
            "SHA256Type" => Ok(class_value(Sha2Kind::Sha256)),
            "SHA384Type" => Ok(class_value(Sha2Kind::Sha384)),
            "SHA512Type" => Ok(class_value(Sha2Kind::Sha512)),
            "sha224" => Ok(builtin_value(&SHA224_FUNCTION)),
            "sha256" => Ok(builtin_value(&SHA256_FUNCTION)),
            "sha384" => Ok(builtin_value(&SHA384_FUNCTION)),
            "sha512" => Ok(builtin_value(&SHA512_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_sha2' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

fn sha224_constructor(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    hash_constructor(Sha2Kind::Sha224, positional, keywords)
}

fn sha256_constructor(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    hash_constructor(Sha2Kind::Sha256, positional, keywords)
}

fn sha384_constructor(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    hash_constructor(Sha2Kind::Sha384, positional, keywords)
}

fn sha512_constructor(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    hash_constructor(Sha2Kind::Sha512, positional, keywords)
}

fn hash_constructor(
    kind: Sha2Kind,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if positional.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes at most 1 positional argument ({} given)",
            kind.constructor_name(),
            positional.len()
        )));
    }

    let mut data = positional.first().copied();
    let mut saw_data_keyword = false;
    let mut saw_usedforsecurity = false;

    for &(name, value) in keywords {
        match name {
            "string" | "data" => {
                if data.is_some() || saw_data_keyword {
                    return Err(BuiltinError::TypeError(format!(
                        "{}() got multiple values for argument 'string'",
                        kind.constructor_name()
                    )));
                }
                data = Some(value);
                saw_data_keyword = true;
            }
            "usedforsecurity" => {
                if saw_usedforsecurity {
                    return Err(BuiltinError::TypeError(format!(
                        "{}() got multiple values for argument 'usedforsecurity'",
                        kind.constructor_name()
                    )));
                }
                saw_usedforsecurity = true;
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "{}() got an unexpected keyword argument '{}'",
                    kind.constructor_name(),
                    other
                )));
            }
        }
    }

    let instance = new_hash_instance(kind);
    if let Some(value) = data {
        set_buffer(
            instance,
            bytes_argument(value, kind.constructor_name())?.to_vec(),
        )?;
    }

    Ok(instance)
}

fn hash_update(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "update() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let mut next = buffer_bytes(args[0])?.to_vec();
    next.extend_from_slice(bytes_argument(args[1], "update")?);
    set_buffer(args[0], next)?;
    Ok(Value::none())
}

fn hash_digest(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "digest() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let digest = compute_digest(hash_kind(args[0])?, buffer_bytes(args[0])?);
    Ok(to_object_value(BytesObject::from_vec(digest)))
}

fn hash_hexdigest(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "hexdigest() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let digest = compute_digest(hash_kind(args[0])?, buffer_bytes(args[0])?);
    let mut text = String::with_capacity(digest.len() * 2);
    for byte in digest {
        text.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        text.push(HEX_DIGITS[(byte & 0x0F) as usize] as char);
    }
    Ok(to_object_value(StringObject::from_string(text)))
}

fn hash_copy(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "copy() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let copied = new_hash_instance_for_receiver(args[0])?;
    set_buffer(copied, buffer_bytes(args[0])?.to_vec())?;
    Ok(copied)
}

fn build_hash_class(kind: Sha2Kind) -> Arc<PyClassObject> {
    let class = Arc::new(PyClassObject::new_simple(intern(kind.class_name())));

    class.set_attr(intern("__module__"), Value::string(intern("_sha2")));
    class.set_attr(
        intern("__qualname__"),
        Value::string(intern(kind.class_name())),
    );
    class.set_attr(intern("__doc__"), Value::string(intern(MODULE_DOC)));
    class.set_attr(
        intern("name"),
        Value::string(intern(kind.constructor_name())),
    );
    class.set_attr(
        intern("digest_size"),
        Value::int(kind.digest_size() as i64).expect("digest size should fit in Value::int"),
    );
    class.set_attr(
        intern("block_size"),
        Value::int(kind.block_size() as i64).expect("block size should fit in Value::int"),
    );
    class.set_attr(intern("update"), builtin_value(&HASH_UPDATE_METHOD));
    class.set_attr(intern("digest"), builtin_value(&HASH_DIGEST_METHOD));
    class.set_attr(intern("hexdigest"), builtin_value(&HASH_HEXDIGEST_METHOD));
    class.set_attr(intern("copy"), builtin_value(&HASH_COPY_METHOD));

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);

    class
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn class_value(kind: Sha2Kind) -> Value {
    Value::object_ptr(Arc::as_ptr(kind.class()) as *const ())
}

#[inline]
fn to_object_value<T: prism_runtime::Trace>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn new_hash_instance(kind: Sha2Kind) -> Value {
    new_hash_instance_from_class(kind.class())
}

fn new_hash_instance_for_receiver(receiver: Value) -> Result<Value, BuiltinError> {
    let ptr = receiver.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_sha2 hash methods require a SHA-2 instance".to_string())
    })?;
    let type_id = crate::ops::objects::extract_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Err(BuiltinError::TypeError(
            "_sha2 hash methods require a SHA-2 instance".to_string(),
        ));
    }

    let class = global_class(ClassId(type_id.raw())).ok_or_else(|| {
        BuiltinError::TypeError("_sha2 hash methods require a SHA-2 instance".to_string())
    })?;
    Ok(new_hash_instance_from_class(class.as_ref()))
}

fn new_hash_instance_from_class(class: &PyClassObject) -> Value {
    let instance = allocate_heap_instance_for_class(class);
    let value = crate::alloc_managed_value(instance);
    set_buffer(value, Vec::new()).expect("hash instance initialization should succeed");
    value
}

fn hash_kind(value: Value) -> Result<Sha2Kind, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_sha2 hash methods require a SHA-2 instance".to_string())
    })?;
    let type_id = crate::ops::objects::extract_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Err(BuiltinError::TypeError(
            "_sha2 hash methods require a SHA-2 instance".to_string(),
        ));
    }

    let class = global_class(ClassId(type_id.raw())).ok_or_else(|| {
        BuiltinError::TypeError("_sha2 hash methods require a SHA-2 instance".to_string())
    })?;

    for kind in Sha2Kind::ALL {
        if class.mro().contains(&kind.class().class_id()) {
            return Ok(kind);
        }
    }

    Err(BuiltinError::TypeError(
        "_sha2 hash methods require a SHA-2 instance".to_string(),
    ))
}

fn buffer_bytes(receiver: Value) -> Result<&'static [u8], BuiltinError> {
    let attr = intern(BUFFER_ATTR);
    let object = shaped_object_ref(receiver)?;
    let Some(value) = object.get_property_interned(&attr) else {
        return Ok(&[]);
    };
    bytes_argument(value, "digest")
}

fn set_buffer(receiver: Value, bytes: Vec<u8>) -> Result<(), BuiltinError> {
    shaped_object_mut(receiver)?.set_property(
        intern(BUFFER_ATTR),
        to_object_value(BytesObject::from_vec(bytes)),
        shape_registry(),
    );
    Ok(())
}

fn shaped_object_ref(value: Value) -> Result<&'static ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_sha2 hash methods require a SHA-2 instance".to_string())
    })?;
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn shaped_object_mut(value: Value) -> Result<&'static mut ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_sha2 hash methods require a SHA-2 instance".to_string())
    })?;
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn bytes_argument(value: Value, context: &str) -> Result<&'static [u8], BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{}() argument must be a bytes-like object, not '{}'",
            context,
            value.type_name()
        ))
    })?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            Ok(unsafe { &*(ptr as *const BytesObject) }.as_bytes())
        }
        _ => Err(BuiltinError::TypeError(format!(
            "{}() argument must be a bytes-like object, not '{}'",
            context,
            value.type_name()
        ))),
    }
}

fn compute_digest(kind: Sha2Kind, data: &[u8]) -> Vec<u8> {
    match kind {
        Sha2Kind::Sha224 => Sha224::digest(data).to_vec(),
        Sha2Kind::Sha256 => Sha256::digest(data).to_vec(),
        Sha2Kind::Sha384 => Sha384::digest(data).to_vec(),
        Sha2Kind::Sha512 => Sha512::digest(data).to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::types::string::value_as_string_ref;

    fn bytes_value(bytes: &[u8]) -> Value {
        to_object_value(BytesObject::from_slice(bytes))
    }

    fn string_value(value: Value) -> String {
        value_as_string_ref(value)
            .expect("value should be a string")
            .as_str()
            .to_string()
    }

    #[test]
    fn test_module_exposes_sha512_surface() {
        let module = Sha2Module::new();
        assert!(module.get_attr("sha512").is_ok());

        let class_value = module
            .get_attr("SHA512Type")
            .expect("SHA512Type should exist");
        let class_ptr = class_value
            .as_object_ptr()
            .expect("SHA512Type should be a class object");
        let class = unsafe { &*(class_ptr as *const PyClassObject) };
        assert_eq!(class.name().as_str(), "SHA512Type");
    }

    #[test]
    fn test_sha512_constructor_hashes_initial_bytes() {
        let value = hash_constructor(Sha2Kind::Sha512, &[bytes_value(b"prism")], &[])
            .expect("sha512 should construct");
        let digest = hash_digest(&[value]).expect("digest should succeed");
        let bytes = bytes_argument(digest, "digest").expect("digest should return bytes");

        assert_eq!(bytes, Sha512::digest(b"prism").as_slice());
    }

    #[test]
    fn test_update_and_copy_are_independent() {
        let value = hash_constructor(Sha2Kind::Sha256, &[], &[]).expect("sha256 should construct");
        hash_update(&[value, bytes_value(b"abc")]).expect("update should succeed");

        let copied = hash_copy(&[value]).expect("copy should succeed");
        hash_update(&[value, bytes_value(b"def")]).expect("second update should succeed");

        let original_hex =
            string_value(hash_hexdigest(&[value]).expect("hexdigest should succeed"));
        let copied_hex = string_value(hash_hexdigest(&[copied]).expect("hexdigest should succeed"));

        assert_eq!(
            original_hex,
            "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
        );
        assert_eq!(
            copied_hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_constructor_accepts_usedforsecurity_keyword() {
        let value = sha512_constructor(&[], &[("usedforsecurity", Value::bool(false))])
            .expect("usedforsecurity keyword should be accepted");
        let digest = hash_hexdigest(&[value]).expect("hexdigest should succeed");
        assert_eq!(
            string_value(digest),
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
             47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
        );
    }
}
