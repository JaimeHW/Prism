//! Native `_ssl` bootstrap surface.
//!
//! CPython's `ssl.py` is a pure-Python wrapper around the `_ssl` extension
//! module. Prism does not yet perform TLS handshakes natively, but the standard
//! library needs the extension module's import-time shape: SSL exception
//! classes, enum constants, OpenSSL metadata, random byte helpers, ASN.1 object
//! lookup, and heap types that `ssl.SSLContext` can subclass.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, exception_proxy_class};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassDict, ClassFlags, PyClassObject};
use prism_runtime::object::descriptor::PropertyDescriptor;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, global_class_bitmap, global_class_registry,
    register_global_class, type_new,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const MODULE_DOC: &str = "Native bootstrap implementation of the _ssl module.";
const OPENSSL_VERSION: &str = "OpenSSL 3.0.0 (Prism compatibility layer)";
const DEFAULT_CIPHERS: &str = "DEFAULT:@SECLEVEL=2";

const PROTOCOL_TLS: i64 = 2;
const PROTOCOL_TLS_CLIENT: i64 = 16;
const PROTOCOL_TLS_SERVER: i64 = 17;
const CERT_NONE: i64 = 0;
const CERT_OPTIONAL: i64 = 1;
const CERT_REQUIRED: i64 = 2;
const VERIFY_DEFAULT: i64 = 0;
const OP_ALL: i64 = 0x8000_0050;

const SSL_CONTEXT_PROTOCOL_ATTR: &str = "__ssl_protocol__";
const SSL_CONTEXT_OPTIONS_ATTR: &str = "__ssl_options__";
const SSL_CONTEXT_VERIFY_FLAGS_ATTR: &str = "__ssl_verify_flags__";
const SSL_CONTEXT_VERIFY_MODE_ATTR: &str = "__ssl_verify_mode__";
const SSL_CONTEXT_MSG_CALLBACK_ATTR: &str = "__ssl_msg_callback__";

const INTEGER_CONSTANTS: &[(&str, i64)] = &[
    ("OPENSSL_VERSION_NUMBER", 0x3000_0000),
    ("ALERT_DESCRIPTION_ACCESS_DENIED", 49),
    ("ALERT_DESCRIPTION_BAD_CERTIFICATE", 42),
    ("ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE", 114),
    ("ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE", 113),
    ("ALERT_DESCRIPTION_BAD_RECORD_MAC", 20),
    ("ALERT_DESCRIPTION_CERTIFICATE_EXPIRED", 45),
    ("ALERT_DESCRIPTION_CERTIFICATE_REVOKED", 44),
    ("ALERT_DESCRIPTION_CERTIFICATE_UNKNOWN", 46),
    ("ALERT_DESCRIPTION_CERTIFICATE_UNOBTAINABLE", 111),
    ("ALERT_DESCRIPTION_CLOSE_NOTIFY", 0),
    ("ALERT_DESCRIPTION_DECODE_ERROR", 50),
    ("ALERT_DESCRIPTION_DECOMPRESSION_FAILURE", 30),
    ("ALERT_DESCRIPTION_DECRYPT_ERROR", 51),
    ("ALERT_DESCRIPTION_HANDSHAKE_FAILURE", 40),
    ("ALERT_DESCRIPTION_ILLEGAL_PARAMETER", 47),
    ("ALERT_DESCRIPTION_INSUFFICIENT_SECURITY", 71),
    ("ALERT_DESCRIPTION_INTERNAL_ERROR", 80),
    ("ALERT_DESCRIPTION_NO_RENEGOTIATION", 100),
    ("ALERT_DESCRIPTION_PROTOCOL_VERSION", 70),
    ("ALERT_DESCRIPTION_RECORD_OVERFLOW", 22),
    ("ALERT_DESCRIPTION_UNEXPECTED_MESSAGE", 10),
    ("ALERT_DESCRIPTION_UNKNOWN_CA", 48),
    ("ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY", 115),
    ("ALERT_DESCRIPTION_UNRECOGNIZED_NAME", 112),
    ("ALERT_DESCRIPTION_UNSUPPORTED_CERTIFICATE", 43),
    ("ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION", 110),
    ("ALERT_DESCRIPTION_USER_CANCELLED", 90),
    ("CERT_NONE", CERT_NONE),
    ("CERT_OPTIONAL", CERT_OPTIONAL),
    ("CERT_REQUIRED", CERT_REQUIRED),
    ("HOSTFLAG_NEVER_CHECK_SUBJECT", 0x20),
    ("OP_ALL", OP_ALL),
    ("OP_CIPHER_SERVER_PREFERENCE", 0x0040_0000),
    ("OP_ENABLE_MIDDLEBOX_COMPAT", 0x0010_0000),
    ("OP_NO_COMPRESSION", 0x0002_0000),
    ("OP_NO_SSLv2", 0x0100_0000),
    ("OP_NO_SSLv3", 0x0200_0000),
    ("OP_NO_TLSv1", 0x0400_0000),
    ("OP_NO_TLSv1_1", 0x1000_0000),
    ("OP_NO_TLSv1_2", 0x0800_0000),
    ("OP_NO_TLSv1_3", 0x2000_0000),
    ("OP_SINGLE_DH_USE", 0x0010_0000),
    ("OP_SINGLE_ECDH_USE", 0x0008_0000),
    ("PROTO_MAXIMUM_SUPPORTED", -1),
    ("PROTO_MINIMUM_SUPPORTED", -2),
    ("PROTO_SSLv3", 0x0300),
    ("PROTO_TLSv1", 0x0301),
    ("PROTO_TLSv1_1", 0x0302),
    ("PROTO_TLSv1_2", 0x0303),
    ("PROTO_TLSv1_3", 0x0304),
    ("PROTOCOL_SSLv23", PROTOCOL_TLS),
    ("PROTOCOL_TLS", PROTOCOL_TLS),
    ("PROTOCOL_TLS_CLIENT", PROTOCOL_TLS_CLIENT),
    ("PROTOCOL_TLS_SERVER", PROTOCOL_TLS_SERVER),
    ("PROTOCOL_TLSv1", 3),
    ("PROTOCOL_TLSv1_1", 4),
    ("PROTOCOL_TLSv1_2", 5),
    ("SSL_ERROR_EOF", 8),
    ("SSL_ERROR_INVALID_ERROR_CODE", 10),
    ("SSL_ERROR_SSL", 1),
    ("SSL_ERROR_SYSCALL", 5),
    ("SSL_ERROR_WANT_CONNECT", 7),
    ("SSL_ERROR_WANT_READ", 2),
    ("SSL_ERROR_WANT_WRITE", 3),
    ("SSL_ERROR_WANT_X509_LOOKUP", 4),
    ("SSL_ERROR_ZERO_RETURN", 6),
    ("VERIFY_ALLOW_PROXY_CERTS", 0x40),
    ("VERIFY_CRL_CHECK_CHAIN", 0x0c),
    ("VERIFY_CRL_CHECK_LEAF", 0x04),
    ("VERIFY_DEFAULT", VERIFY_DEFAULT),
    ("VERIFY_X509_PARTIAL_CHAIN", 0x80000),
    ("VERIFY_X509_STRICT", 0x20),
    ("VERIFY_X509_TRUSTED_FIRST", 0x8000),
];

const BOOLEAN_CONSTANTS: &[(&str, bool)] = &[
    ("HAS_ALPN", true),
    ("HAS_ECDH", true),
    ("HAS_NEVER_CHECK_COMMON_NAME", true),
    ("HAS_NPN", false),
    ("HAS_SNI", true),
    ("HAS_SSLv2", false),
    ("HAS_SSLv3", false),
    ("HAS_TLSv1", true),
    ("HAS_TLSv1_1", true),
    ("HAS_TLSv1_2", true),
    ("HAS_TLSv1_3", true),
];

const EXPORTED_NAMES: &[&str] = &[
    "__doc__",
    "_DEFAULT_CIPHERS",
    "_OPENSSL_API_VERSION",
    "_SSLContext",
    "MemoryBIO",
    "OPENSSL_VERSION",
    "OPENSSL_VERSION_INFO",
    "RAND_add",
    "RAND_bytes",
    "RAND_status",
    "SSLCertVerificationError",
    "SSLEOFError",
    "SSLError",
    "SSLSession",
    "SSLSyscallError",
    "SSLWantReadError",
    "SSLWantWriteError",
    "SSLZeroReturnError",
    "enum_certificates",
    "enum_crls",
    "get_default_verify_paths",
    "nid2obj",
    "txt2obj",
];

static SSL_CONTEXT_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ssl_context_class("_SSLContext"));
static MEMORY_BIO_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_simple_native_class("MemoryBIO"));
static SSL_SESSION_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_simple_native_class("SSLSession"));

static SSL_ERROR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_exception_class("SSLError", ExceptionTypeId::OSError));
static SSL_ZERO_RETURN_ERROR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ssl_error_subclass("SSLZeroReturnError"));
static SSL_WANT_READ_ERROR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ssl_error_subclass("SSLWantReadError"));
static SSL_WANT_WRITE_ERROR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ssl_error_subclass("SSLWantWriteError"));
static SSL_SYSCALL_ERROR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ssl_error_subclass("SSLSyscallError"));
static SSL_EOF_ERROR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ssl_error_subclass("SSLEOFError"));
static SSL_CERT_VERIFICATION_ERROR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_ssl_error_subclass("SSLCertVerificationError"));

static SSL_CONTEXT_NEW_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_ssl._SSLContext.__new__"), ssl_context_new)
});
static SIMPLE_CLASS_NEW_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_ssl.native.__new__"), simple_class_new)
});
static SSL_CONTEXT_NOOP_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_ssl._SSLContext.noop"), ssl_context_noop)
});
static SSL_CONTEXT_GET_CA_CERTS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_ssl._SSLContext.get_ca_certs"),
        ssl_context_empty_list,
    )
});
static SSL_CONTEXT_STATS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_ssl._SSLContext.cert_store_stats"),
        ssl_context_stats,
    )
});
static SSL_CONTEXT_GET_PROTOCOL_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_ssl._SSLContext.protocol.fget"),
        ssl_context_get_protocol,
    )
});
static SSL_CONTEXT_GET_OPTIONS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_ssl._SSLContext.options.fget"),
        ssl_context_get_options,
    )
});
static SSL_CONTEXT_SET_OPTIONS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_ssl._SSLContext.options.fset"),
        ssl_context_set_options,
    )
});
static SSL_CONTEXT_GET_VERIFY_FLAGS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("_ssl._SSLContext.verify_flags.fget"),
            ssl_context_get_verify_flags,
        )
    });
static SSL_CONTEXT_SET_VERIFY_FLAGS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("_ssl._SSLContext.verify_flags.fset"),
            ssl_context_set_verify_flags,
        )
    });
static SSL_CONTEXT_GET_VERIFY_MODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("_ssl._SSLContext.verify_mode.fget"),
            ssl_context_get_verify_mode,
        )
    });
static SSL_CONTEXT_SET_VERIFY_MODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("_ssl._SSLContext.verify_mode.fset"),
            ssl_context_set_verify_mode,
        )
    });
static SSL_CONTEXT_GET_MSG_CALLBACK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("_ssl._SSLContext._msg_callback.fget"),
            ssl_context_get_msg_callback,
        )
    });
static SSL_CONTEXT_SET_MSG_CALLBACK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("_ssl._SSLContext._msg_callback.fset"),
            ssl_context_set_msg_callback,
        )
    });

static GET_DEFAULT_VERIFY_PATHS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_ssl.get_default_verify_paths"),
        get_default_verify_paths,
    )
});
static NID2OBJ_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_ssl.nid2obj"), nid2obj));
static RAND_ADD_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_ssl.RAND_add"), rand_add));
static RAND_BYTES_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_ssl.RAND_bytes"), rand_bytes));
static RAND_STATUS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_ssl.RAND_status"), rand_status));
static TXT2OBJ_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("_ssl.txt2obj"), txt2obj));
static ENUM_CERTIFICATES_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_ssl.enum_certificates"), enum_certificates)
});
static ENUM_CRLS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_ssl.enum_crls"), enum_crls));

static PROTOCOL_PROPERTY: LazyLock<PropertyDescriptor> = LazyLock::new(|| {
    PropertyDescriptor::new_getter(builtin_value(&SSL_CONTEXT_GET_PROTOCOL_FUNCTION))
});
static OPTIONS_PROPERTY: LazyLock<PropertyDescriptor> = LazyLock::new(|| {
    PropertyDescriptor::new_full(
        Some(builtin_value(&SSL_CONTEXT_GET_OPTIONS_FUNCTION)),
        Some(builtin_value(&SSL_CONTEXT_SET_OPTIONS_FUNCTION)),
        None,
        None,
    )
});
static VERIFY_FLAGS_PROPERTY: LazyLock<PropertyDescriptor> = LazyLock::new(|| {
    PropertyDescriptor::new_full(
        Some(builtin_value(&SSL_CONTEXT_GET_VERIFY_FLAGS_FUNCTION)),
        Some(builtin_value(&SSL_CONTEXT_SET_VERIFY_FLAGS_FUNCTION)),
        None,
        None,
    )
});
static VERIFY_MODE_PROPERTY: LazyLock<PropertyDescriptor> = LazyLock::new(|| {
    PropertyDescriptor::new_full(
        Some(builtin_value(&SSL_CONTEXT_GET_VERIFY_MODE_FUNCTION)),
        Some(builtin_value(&SSL_CONTEXT_SET_VERIFY_MODE_FUNCTION)),
        None,
        None,
    )
});
static MSG_CALLBACK_PROPERTY: LazyLock<PropertyDescriptor> = LazyLock::new(|| {
    PropertyDescriptor::new_full(
        Some(builtin_value(&SSL_CONTEXT_GET_MSG_CALLBACK_FUNCTION)),
        Some(builtin_value(&SSL_CONTEXT_SET_MSG_CALLBACK_FUNCTION)),
        None,
        None,
    )
});

/// Native `_ssl` module descriptor.
#[derive(Debug, Clone)]
pub struct SslModule {
    attrs: Vec<Arc<str>>,
}

impl SslModule {
    /// Create a new `_ssl` module descriptor.
    pub fn new() -> Self {
        let mut attrs = Vec::with_capacity(
            EXPORTED_NAMES.len() + INTEGER_CONSTANTS.len() + BOOLEAN_CONSTANTS.len(),
        );
        attrs.extend(EXPORTED_NAMES.iter().copied().map(Arc::from));
        attrs.extend(INTEGER_CONSTANTS.iter().map(|(name, _)| Arc::from(*name)));
        attrs.extend(BOOLEAN_CONSTANTS.iter().map(|(name, _)| Arc::from(*name)));
        attrs.sort_unstable();
        attrs.dedup();
        Self { attrs }
    }
}

impl Default for SslModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SslModule {
    fn name(&self) -> &str {
        "_ssl"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        if let Some((_, value)) = INTEGER_CONSTANTS
            .iter()
            .find(|(constant_name, _)| *constant_name == name)
        {
            return Ok(Value::int(*value).expect("_ssl integer constant should fit Value::int"));
        }
        if let Some((_, value)) = BOOLEAN_CONSTANTS
            .iter()
            .find(|(constant_name, _)| *constant_name == name)
        {
            return Ok(Value::bool(*value));
        }

        match name {
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "_DEFAULT_CIPHERS" => Ok(Value::string(intern(DEFAULT_CIPHERS))),
            "_OPENSSL_API_VERSION" => Ok(tuple_i64(&[3, 0, 0, 0, 0])),
            "_SSLContext" => Ok(class_value(&SSL_CONTEXT_CLASS)),
            "MemoryBIO" => Ok(class_value(&MEMORY_BIO_CLASS)),
            "OPENSSL_VERSION" => Ok(Value::string(intern(OPENSSL_VERSION))),
            "OPENSSL_VERSION_INFO" => Ok(tuple_i64(&[3, 0, 0, 0, 0])),
            "RAND_add" => Ok(builtin_value(&RAND_ADD_FUNCTION)),
            "RAND_bytes" => Ok(builtin_value(&RAND_BYTES_FUNCTION)),
            "RAND_status" => Ok(builtin_value(&RAND_STATUS_FUNCTION)),
            "SSLCertVerificationError" => Ok(class_value(&SSL_CERT_VERIFICATION_ERROR_CLASS)),
            "SSLEOFError" => Ok(class_value(&SSL_EOF_ERROR_CLASS)),
            "SSLError" => Ok(class_value(&SSL_ERROR_CLASS)),
            "SSLSession" => Ok(class_value(&SSL_SESSION_CLASS)),
            "SSLSyscallError" => Ok(class_value(&SSL_SYSCALL_ERROR_CLASS)),
            "SSLWantReadError" => Ok(class_value(&SSL_WANT_READ_ERROR_CLASS)),
            "SSLWantWriteError" => Ok(class_value(&SSL_WANT_WRITE_ERROR_CLASS)),
            "SSLZeroReturnError" => Ok(class_value(&SSL_ZERO_RETURN_ERROR_CLASS)),
            "enum_certificates" => Ok(builtin_value(&ENUM_CERTIFICATES_FUNCTION)),
            "enum_crls" => Ok(builtin_value(&ENUM_CRLS_FUNCTION)),
            "get_default_verify_paths" => Ok(builtin_value(&GET_DEFAULT_VERIFY_PATHS_FUNCTION)),
            "nid2obj" => Ok(builtin_value(&NID2OBJ_FUNCTION)),
            "txt2obj" => Ok(builtin_value(&TXT2OBJ_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_ssl' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

fn build_ssl_context_class(name: &'static str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("_ssl")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    class.set_attr(intern("__new__"), builtin_value(&SSL_CONTEXT_NEW_FUNCTION));
    class.set_attr(intern("protocol"), property_value(&PROTOCOL_PROPERTY));
    class.set_attr(intern("options"), property_value(&OPTIONS_PROPERTY));
    class.set_attr(
        intern("verify_flags"),
        property_value(&VERIFY_FLAGS_PROPERTY),
    );
    class.set_attr(intern("verify_mode"), property_value(&VERIFY_MODE_PROPERTY));
    class.set_attr(
        intern("_msg_callback"),
        property_value(&MSG_CALLBACK_PROPERTY),
    );
    class.set_attr(
        intern("cert_store_stats"),
        builtin_value(&SSL_CONTEXT_STATS_FUNCTION),
    );
    class.set_attr(
        intern("get_ca_certs"),
        builtin_value(&SSL_CONTEXT_GET_CA_CERTS_FUNCTION),
    );
    for name in [
        "_set_alpn_protocols",
        "_set_npn_protocols",
        "_wrap_bio",
        "_wrap_socket",
        "load_cert_chain",
        "load_verify_locations",
        "set_ciphers",
        "set_default_verify_paths",
    ] {
        class.set_attr(intern(name), builtin_value(&SSL_CONTEXT_NOOP_FUNCTION));
    }
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_NEW | ClassFlags::NATIVE_HEAPTYPE);
    register_native_type(class)
}

fn build_simple_native_class(name: &'static str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("_ssl")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    class.set_attr(intern("__new__"), builtin_value(&SIMPLE_CLASS_NEW_FUNCTION));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_NEW | ClassFlags::NATIVE_HEAPTYPE);
    register_native_type(class)
}

fn build_ssl_error_subclass(name: &'static str) -> Arc<PyClassObject> {
    let base = SSL_ERROR_CLASS.class_id();
    build_heap_class(name, &[base])
}

fn build_exception_class(name: &'static str, base: ExceptionTypeId) -> Arc<PyClassObject> {
    let base = exception_proxy_class(base).class_id();
    build_heap_class(name, &[base])
}

fn build_heap_class(name: &'static str, bases: &[ClassId]) -> Arc<PyClassObject> {
    let namespace = ClassDict::new();
    namespace.set(intern("__module__"), Value::string(intern("_ssl")));
    namespace.set(intern("__qualname__"), Value::string(intern(name)));

    let result = type_new(intern(name), bases, &namespace, global_class_registry())
        .expect("_ssl native heap classes should construct");
    let mut class =
        Arc::try_unwrap(result.class).expect("freshly created _ssl class should be unique");
    class.add_flags(ClassFlags::NATIVE_HEAPTYPE);
    let class = Arc::new(class);
    register_global_class(Arc::clone(&class), result.bitmap);
    class
}

fn register_native_type(class: PyClassObject) -> Arc<PyClassObject> {
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    let class = Arc::new(class);
    register_global_class(Arc::clone(&class), bitmap);
    class
}

fn ssl_context_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "_SSLContext.__new__() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }

    let class = ssl_context_class_from_value(args[0])?;
    let protocol = args
        .get(1)
        .copied()
        .unwrap_or_else(|| Value::int(PROTOCOL_TLS).expect("protocol constant should fit"));
    let mut object = ShapedObject::new(class.class_type_id(), Arc::clone(class.instance_shape()));
    let registry = shape_registry();
    object.set_property(intern(SSL_CONTEXT_PROTOCOL_ATTR), protocol, registry);
    object.set_property(
        intern(SSL_CONTEXT_OPTIONS_ATTR),
        Value::int(OP_ALL).expect("options should fit"),
        registry,
    );
    object.set_property(
        intern(SSL_CONTEXT_VERIFY_FLAGS_ATTR),
        Value::int(VERIFY_DEFAULT).expect("verify flags should fit"),
        registry,
    );
    object.set_property(
        intern(SSL_CONTEXT_VERIFY_MODE_ATTR),
        Value::int(CERT_NONE).expect("verify mode should fit"),
        registry,
    );
    object.set_property(intern("check_hostname"), Value::bool(false), registry);
    object.set_property(
        intern("_host_flags"),
        Value::int(0).expect("host flags should fit"),
        registry,
    );
    object.set_property(
        intern(SSL_CONTEXT_MSG_CALLBACK_ATTR),
        Value::none(),
        registry,
    );
    Ok(leak_object_value(object))
}

fn simple_class_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "__new__() missing required class argument".to_string(),
        ));
    }
    let class = class_from_value(args[0], "__new__")?;
    Ok(leak_object_value(ShapedObject::new(
        class.class_type_id(),
        Arc::clone(class.instance_shape()),
    )))
}

fn ssl_context_noop(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "_SSLContext method requires a receiver".to_string(),
        ));
    }
    let _ = expect_ssl_context_instance(args[0])?;
    Ok(Value::none())
}

fn ssl_context_empty_list(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "_SSLContext method requires a receiver".to_string(),
        ));
    }
    let _ = expect_ssl_context_instance(args[0])?;
    Ok(list_value(Vec::new()))
}

fn ssl_context_stats(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "cert_store_stats() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let _ = expect_ssl_context_instance(args[0])?;
    let mut dict = DictObject::new();
    dict.set(Value::string(intern("x509")), Value::int(0).unwrap());
    dict.set(Value::string(intern("crl")), Value::int(0).unwrap());
    dict.set(Value::string(intern("x509_ca")), Value::int(0).unwrap());
    Ok(leak_object_value(dict))
}

fn ssl_context_get_protocol(args: &[Value]) -> Result<Value, BuiltinError> {
    ssl_context_get_slot(
        args,
        SSL_CONTEXT_PROTOCOL_ATTR,
        Value::int(PROTOCOL_TLS).expect("protocol should fit"),
    )
}

fn ssl_context_get_options(args: &[Value]) -> Result<Value, BuiltinError> {
    ssl_context_get_slot(
        args,
        SSL_CONTEXT_OPTIONS_ATTR,
        Value::int(OP_ALL).expect("options should fit"),
    )
}

fn ssl_context_set_options(args: &[Value]) -> Result<Value, BuiltinError> {
    ssl_context_set_slot(args, SSL_CONTEXT_OPTIONS_ATTR)
}

fn ssl_context_get_verify_flags(args: &[Value]) -> Result<Value, BuiltinError> {
    ssl_context_get_slot(
        args,
        SSL_CONTEXT_VERIFY_FLAGS_ATTR,
        Value::int(VERIFY_DEFAULT).expect("verify flags should fit"),
    )
}

fn ssl_context_set_verify_flags(args: &[Value]) -> Result<Value, BuiltinError> {
    ssl_context_set_slot(args, SSL_CONTEXT_VERIFY_FLAGS_ATTR)
}

fn ssl_context_get_verify_mode(args: &[Value]) -> Result<Value, BuiltinError> {
    ssl_context_get_slot(
        args,
        SSL_CONTEXT_VERIFY_MODE_ATTR,
        Value::int(CERT_NONE).expect("verify mode should fit"),
    )
}

fn ssl_context_set_verify_mode(args: &[Value]) -> Result<Value, BuiltinError> {
    ssl_context_set_slot(args, SSL_CONTEXT_VERIFY_MODE_ATTR)
}

fn ssl_context_get_msg_callback(args: &[Value]) -> Result<Value, BuiltinError> {
    ssl_context_get_slot(args, SSL_CONTEXT_MSG_CALLBACK_ATTR, Value::none())
}

fn ssl_context_set_msg_callback(args: &[Value]) -> Result<Value, BuiltinError> {
    ssl_context_set_slot(args, SSL_CONTEXT_MSG_CALLBACK_ATTR)
}

fn ssl_context_get_slot(
    args: &[Value],
    property: &'static str,
    default: Value,
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "descriptor getter expected 1 argument, got {}",
            args.len()
        )));
    }
    let object = expect_ssl_context_instance(args[0])?;
    Ok(object.get_property(property).unwrap_or(default))
}

fn ssl_context_set_slot(args: &[Value], property: &'static str) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "descriptor setter expected 2 arguments, got {}",
            args.len()
        )));
    }
    let object = expect_ssl_context_instance(args[0])?;
    object.set_property(intern(property), args[1], shape_registry());
    Ok(Value::none())
}

fn get_default_verify_paths(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "get_default_verify_paths")?;
    Ok(tuple_value(vec![
        Value::string(intern("SSL_CERT_FILE")),
        Value::string(intern("")),
        Value::string(intern("SSL_CERT_DIR")),
        Value::string(intern("")),
    ]))
}

fn txt2obj(positional: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    if positional.is_empty() || positional.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "txt2obj() takes 1 or 2 positional arguments ({} given)",
            positional.len()
        )));
    }
    let mut text = Some(positional[0]);
    let mut name = positional.get(1).copied();
    for &(keyword, value) in keywords {
        match keyword {
            "txt" | "oid" => {
                if text.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "txt2obj() got multiple values for argument 'txt'".to_string(),
                    ));
                }
            }
            "name" => {
                if name.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "txt2obj() got multiple values for argument 'name'".to_string(),
                    ));
                }
            }
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "txt2obj() got an unexpected keyword argument '{}'",
                    keyword
                )));
            }
        }
    }

    let lookup = string_arg(text.expect("txt should be present"), "txt")?;
    let by_name = name
        .and_then(|value| {
            value
                .as_bool()
                .or_else(|| value.as_int().map(|int| int != 0))
        })
        .unwrap_or(false);
    Ok(asn1_tuple(asn1_object(&lookup, by_name)))
}

fn nid2obj(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "nid2obj() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let Some(nid) = args[0].as_int() else {
        return Err(BuiltinError::TypeError(
            "nid2obj() argument must be an integer".to_string(),
        ));
    };
    Ok(asn1_tuple(asn1_object_from_nid(nid)))
}

fn rand_status(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args(args, "RAND_status")?;
    Ok(Value::bool(true))
}

fn rand_add(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "RAND_add() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    Ok(Value::none())
}

fn rand_bytes(args: &[Value]) -> Result<Value, BuiltinError> {
    crate::stdlib::secure_random::urandom_value_from_args(args, "RAND_bytes")
}

fn enum_certificates(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "enum_certificates() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(list_value(Vec::new()))
}

fn enum_crls(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "enum_crls() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(list_value(Vec::new()))
}

fn asn1_object(text: &str, by_name: bool) -> (i64, &'static str, &'static str, String) {
    match text {
        "1.3.6.1.5.5.7.3.1" | "serverAuth" if by_name || text.starts_with("1.") => (
            129,
            "serverAuth",
            "TLS Web Server Authentication",
            "1.3.6.1.5.5.7.3.1".to_string(),
        ),
        "1.3.6.1.5.5.7.3.2" | "clientAuth" if by_name || text.starts_with("1.") => (
            130,
            "clientAuth",
            "TLS Web Client Authentication",
            "1.3.6.1.5.5.7.3.2".to_string(),
        ),
        _ => (0, "undefined", "undefined", text.to_string()),
    }
}

fn asn1_object_from_nid(nid: i64) -> (i64, &'static str, &'static str, String) {
    match nid {
        129 => (
            129,
            "serverAuth",
            "TLS Web Server Authentication",
            "1.3.6.1.5.5.7.3.1".to_string(),
        ),
        130 => (
            130,
            "clientAuth",
            "TLS Web Client Authentication",
            "1.3.6.1.5.5.7.3.2".to_string(),
        ),
        _ => (nid, "undefined", "undefined", nid.to_string()),
    }
}

fn asn1_tuple(object: (i64, &'static str, &'static str, String)) -> Value {
    let (nid, shortname, longname, oid) = object;
    tuple_value(vec![
        Value::int(nid).expect("ASN.1 nid should fit"),
        Value::string(intern(shortname)),
        Value::string(intern(longname)),
        Value::string(intern(&oid)),
    ])
}

fn ssl_context_class_from_value(value: Value) -> Result<&'static PyClassObject, BuiltinError> {
    let class = class_from_value(value, "_SSLContext.__new__")?;
    let required_type_id = SSL_CONTEXT_CLASS.class_type_id();
    if class.class_type_id() != required_type_id
        && !global_class_bitmap(class.class_id())
            .is_some_and(|bitmap| bitmap.is_subclass_of(required_type_id))
    {
        return Err(BuiltinError::TypeError(
            "_SSLContext.__new__() argument 1 must be a subtype of _SSLContext".to_string(),
        ));
    }
    Ok(class)
}

fn class_from_value(
    value: Value,
    context: &'static str,
) -> Result<&'static PyClassObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{}() argument 1 must be a class",
            context
        )));
    };
    if crate::ops::objects::extract_type_id(ptr) != TypeId::TYPE
        || crate::builtins::builtin_type_object_type_id(ptr).is_some()
    {
        return Err(BuiltinError::TypeError(format!(
            "{}() argument 1 must be a class",
            context
        )));
    }
    Ok(unsafe { &*(ptr as *const PyClassObject) })
}

fn expect_ssl_context_instance(value: Value) -> Result<&'static mut ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "_SSLContext method requires an _SSLContext instance".to_string(),
        ));
    };
    let type_id = crate::ops::objects::extract_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Err(BuiltinError::TypeError(
            "_SSLContext method requires an _SSLContext instance".to_string(),
        ));
    }
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn expect_no_args(args: &[Value], name: &'static str) -> Result<(), BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes no arguments ({} given)",
            name,
            args.len()
        )));
    }
    Ok(())
}

fn string_arg(value: Value, name: &'static str) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|text| text.as_str().to_string())
        .ok_or_else(|| BuiltinError::TypeError(format!("{} must be str", name)))
}

fn tuple_i64(values: &[i64]) -> Value {
    tuple_value(
        values
            .iter()
            .map(|value| Value::int(*value).expect("tuple integer should fit"))
            .collect(),
    )
}

fn tuple_value(items: Vec<Value>) -> Value {
    leak_object_value(TupleObject::from_vec(items))
}

fn list_value(items: Vec<Value>) -> Value {
    leak_object_value(ListObject::from_iter(items))
}

fn leak_object_value<T: prism_runtime::Trace>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn class_value(class: &'static Arc<PyClassObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

#[inline]
fn property_value(property: &'static LazyLock<PropertyDescriptor>) -> Value {
    Value::object_ptr(&**property as *const PropertyDescriptor as *const ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::types::bytes::BytesObject;

    fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
        let ptr = value.as_object_ptr().expect("expected builtin function");
        unsafe { &*(ptr as *const BuiltinFunctionObject) }
    }

    #[test]
    fn test_ssl_module_exposes_import_time_surface() {
        let module = SslModule::new();
        assert!(module.get_attr("SSLError").is_ok());
        assert!(module.get_attr("SSLWantReadError").is_ok());
        assert!(module.get_attr("_SSLContext").is_ok());
        assert_eq!(module.get_attr("HAS_SNI").unwrap().as_bool(), Some(true));
        assert_eq!(
            module.get_attr("PROTOCOL_TLS_CLIENT").unwrap().as_int(),
            Some(PROTOCOL_TLS_CLIENT)
        );
    }

    #[test]
    fn test_ssl_txt2obj_supports_purpose_oids() {
        let txt2obj = builtin_from_value(SslModule::new().get_attr("txt2obj").unwrap());
        let result = txt2obj
            .call_with_keywords(
                &[Value::string(intern("1.3.6.1.5.5.7.3.1"))],
                &[("name", Value::bool(false))],
            )
            .expect("txt2obj should resolve server auth");
        let tuple = unsafe { &*(result.as_object_ptr().unwrap() as *const TupleObject) };
        assert_eq!(tuple.len(), 4);
        assert_eq!(tuple.get(0).and_then(|value| value.as_int()), Some(129));
    }

    #[test]
    fn test_ssl_rand_bytes_returns_requested_size() {
        let rand_bytes = builtin_from_value(SslModule::new().get_attr("RAND_bytes").unwrap());
        let result = rand_bytes
            .call(&[Value::int(8).unwrap()])
            .expect("RAND_bytes should succeed");
        let bytes = unsafe { &*(result.as_object_ptr().unwrap() as *const BytesObject) };
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn test_ssl_context_new_allocates_shaped_context_instances() {
        let class_value = SslModule::new().get_attr("_SSLContext").unwrap();
        let value = ssl_context_new(&[class_value, Value::int(PROTOCOL_TLS_CLIENT).unwrap()])
            .expect("_SSLContext.__new__ should allocate");
        let object = expect_ssl_context_instance(value).expect("context instance");
        assert_eq!(
            object
                .get_property(SSL_CONTEXT_PROTOCOL_ATTR)
                .and_then(|value| value.as_int()),
            Some(PROTOCOL_TLS_CLIENT)
        );
    }
}
