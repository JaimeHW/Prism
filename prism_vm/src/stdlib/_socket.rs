//! Native `_socket` bootstrap surface.
//!
//! CPython's `socket.py` is a pure-Python wrapper around the `_socket`
//! extension module. Prism exposes the same import-time shape here so the
//! standard library can define its higher-level `socket.socket` subclass and
//! export list before full OS socket handles are implemented.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

const MODULE_DOC: &str = "Native bootstrap implementation of the _socket module.";

#[cfg(windows)]
const AF_INET6_VALUE: i64 = 23;
#[cfg(not(windows))]
const AF_INET6_VALUE: i64 = 10;

#[cfg(windows)]
const SOL_SOCKET_VALUE: i64 = 0xffff;
#[cfg(not(windows))]
const SOL_SOCKET_VALUE: i64 = 1;

const EXPORTED_CONSTANTS: &[(&str, i64)] = &[
    ("AF_UNSPEC", 0),
    ("AF_UNIX", 1),
    ("AF_INET", 2),
    ("SOCK_STREAM", 1),
    ("SOCK_DGRAM", 2),
    ("SOCK_RAW", 3),
    ("SOCK_RDM", 4),
    ("SOCK_SEQPACKET", 5),
    ("IPPROTO_IP", 0),
    ("IPPROTO_TCP", 6),
    ("IPPROTO_UDP", 17),
    ("IPPROTO_IPV6", 41),
    ("TCP_NODELAY", 1),
    ("SO_REUSEADDR", 4),
    ("SO_KEEPALIVE", 8),
    ("SO_BROADCAST", 32),
    ("SO_LINGER", 128),
    ("SO_ERROR", 0x1007),
    ("SO_TYPE", 0x1008),
    ("SO_REUSEPORT", 15),
    ("IPV6_V6ONLY", 27),
    ("AI_PASSIVE", 1),
    ("AI_CANONNAME", 2),
    ("AI_NUMERICHOST", 4),
    ("AI_NUMERICSERV", 8),
    ("AI_ALL", 256),
    ("AI_ADDRCONFIG", 1024),
    ("AI_V4MAPPED", 2048),
    ("EAI_AGAIN", -3),
    ("EAI_FAIL", -4),
    ("EAI_FAMILY", -6),
    ("EAI_NONAME", -2),
    ("EAI_SERVICE", -8),
    ("SHUT_RD", 0),
    ("SHUT_WR", 1),
    ("SHUT_RDWR", 2),
    ("MSG_OOB", 1),
    ("MSG_PEEK", 2),
    ("MSG_DONTROUTE", 4),
    ("MSG_WAITALL", 8),
];

const EXPORTED_NAMES: &[&str] = &[
    "AF_INET",
    "AF_INET6",
    "AF_UNIX",
    "AF_UNSPEC",
    "AI_ADDRCONFIG",
    "AI_ALL",
    "AI_CANONNAME",
    "AI_NUMERICHOST",
    "AI_NUMERICSERV",
    "AI_PASSIVE",
    "AI_V4MAPPED",
    "CMSG_LEN",
    "CMSG_SPACE",
    "EAI_AGAIN",
    "EAI_FAIL",
    "EAI_FAMILY",
    "EAI_NONAME",
    "EAI_SERVICE",
    "IPPROTO_IP",
    "IPPROTO_IPV6",
    "IPPROTO_TCP",
    "IPPROTO_UDP",
    "IPV6_V6ONLY",
    "MSG_DONTROUTE",
    "MSG_OOB",
    "MSG_PEEK",
    "MSG_WAITALL",
    "SHUT_RD",
    "SHUT_RDWR",
    "SHUT_WR",
    "SOCK_DGRAM",
    "SOCK_RAW",
    "SOCK_RDM",
    "SOCK_SEQPACKET",
    "SOCK_STREAM",
    "SOL_SOCKET",
    "SO_BROADCAST",
    "SO_ERROR",
    "SO_KEEPALIVE",
    "SO_LINGER",
    "SO_REUSEADDR",
    "SO_REUSEPORT",
    "SO_TYPE",
    "SocketType",
    "TCP_NODELAY",
    "error",
    "gaierror",
    "getaddrinfo",
    "getdefaulttimeout",
    "gethostbyaddr",
    "gethostbyname",
    "gethostbyname_ex",
    "gethostname",
    "getnameinfo",
    "getprotobyname",
    "getservbyname",
    "getservbyport",
    "has_ipv6",
    "herror",
    "htonl",
    "htons",
    "inet_aton",
    "inet_ntoa",
    "inet_ntop",
    "inet_pton",
    "ntohl",
    "ntohs",
    "setdefaulttimeout",
    "socket",
    "timeout",
];

static SOCKET_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_socket_class);
static DEFAULT_TIMEOUT: Mutex<Option<f64>> = Mutex::new(None);
static NEXT_SOCKET_FD: AtomicI64 = AtomicI64::new(10_000);

static SOCKET_INIT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.__init__"), socket_init));
static SOCKET_CLOSE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.close"), socket_close));
static SOCKET_DETACH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.detach"), socket_detach));
static SOCKET_FILENO_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.fileno"), socket_fileno));
static SOCKET_GETTIMEOUT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.gettimeout"), socket_gettimeout)
});
static SOCKET_SETTIMEOUT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.settimeout"), socket_settimeout)
});
static SOCKET_SETBLOCKING_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.setblocking"), socket_setblocking)
});
static SOCKET_BIND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.bind"), socket_noop));
static SOCKET_LISTEN_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.listen"), socket_noop));
static SOCKET_CONNECT_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.connect"), socket_noop));
static SOCKET_CONNECT_EX_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.connect_ex"), socket_connect_ex)
});
static SOCKET_SHUTDOWN_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.shutdown"), socket_noop));
static SOCKET_SETSOCKOPT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.setsockopt"), socket_noop)
});
static SOCKET_GETSOCKOPT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.getsockopt"), socket_getsockopt)
});

static CMSG_LEN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.CMSG_LEN"), cmsg_len));
static CMSG_SPACE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.CMSG_SPACE"), cmsg_space));
static GETADDRINFO_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.getaddrinfo"), socket_getaddrinfo)
});
static GETDEFAULTTIMEOUT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_socket.getdefaulttimeout"),
        socket_getdefaulttimeout,
    )
});
static GETHOSTBYADDR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.gethostbyaddr"), socket_gethostbyaddr)
});
static GETHOSTBYNAME_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.gethostbyname"), socket_gethostbyname)
});
static GETHOSTBYNAME_EX_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_socket.gethostbyname_ex"),
        socket_gethostbyname_ex,
    )
});
static GETHOSTNAME_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.gethostname"), socket_gethostname)
});
static GETNAMEINFO_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.getnameinfo"), socket_getnameinfo)
});
static GETPROTOBYNAME_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.getprotobyname"), socket_getprotobyname)
});
static GETSERVBYNAME_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.getservbyname"), socket_getservbyname)
});
static GETSERVBYPORT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.getservbyport"), socket_getservbyport)
});
static HTONL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.htonl"), socket_htonl));
static HTONS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.htons"), socket_htons));
static INET_ATON_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.inet_aton"), socket_inet_aton));
static INET_NTOA_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.inet_ntoa"), socket_inet_ntoa));
static INET_NTOP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.inet_ntop"), socket_inet_ntop));
static INET_PTON_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.inet_pton"), socket_inet_pton));
static NTOHL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.ntohl"), socket_ntohl));
static NTOHS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.ntohs"), socket_ntohs));
static SETDEFAULTTIMEOUT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_socket.setdefaulttimeout"),
        socket_setdefaulttimeout,
    )
});

/// Native `_socket` module descriptor.
pub struct SocketModule {
    attrs: Vec<Arc<str>>,
    all_value: Value,
    values: FxHashMap<Arc<str>, Value>,
}

impl SocketModule {
    /// Create a new `_socket` module descriptor.
    pub fn new() -> Self {
        let mut attrs = Vec::with_capacity(EXPORTED_NAMES.len() + 2);
        attrs.push(Arc::from("__all__"));
        attrs.push(Arc::from("__doc__"));
        attrs.extend(EXPORTED_NAMES.iter().copied().map(Arc::from));
        attrs.sort_unstable();
        attrs.dedup();

        let mut values = FxHashMap::default();
        for &(name, value) in EXPORTED_CONSTANTS {
            values.insert(
                Arc::from(name),
                Value::int(value).expect("_socket constant should fit in tagged int"),
            );
        }
        values.insert(Arc::from("AF_INET6"), Value::int(AF_INET6_VALUE).unwrap());
        values.insert(
            Arc::from("SOL_SOCKET"),
            Value::int(SOL_SOCKET_VALUE).unwrap(),
        );
        values.insert(Arc::from("has_ipv6"), Value::bool(true));

        Self {
            attrs,
            all_value: export_names_value(),
            values,
        }
    }
}

impl Default for SocketModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SocketModule {
    fn name(&self) -> &str {
        "_socket"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all_value),
            "__doc__" => Ok(Value::string(intern(MODULE_DOC))),
            "socket" | "SocketType" => Ok(socket_class_value()),
            "error" | "gaierror" | "herror" => Ok(os_error_type_value()),
            "timeout" => Ok(timeout_error_type_value()),
            "CMSG_LEN" => Ok(builtin_value(&CMSG_LEN_FUNCTION)),
            "CMSG_SPACE" => Ok(builtin_value(&CMSG_SPACE_FUNCTION)),
            "getaddrinfo" => Ok(builtin_value(&GETADDRINFO_FUNCTION)),
            "getdefaulttimeout" => Ok(builtin_value(&GETDEFAULTTIMEOUT_FUNCTION)),
            "gethostbyaddr" => Ok(builtin_value(&GETHOSTBYADDR_FUNCTION)),
            "gethostbyname" => Ok(builtin_value(&GETHOSTBYNAME_FUNCTION)),
            "gethostbyname_ex" => Ok(builtin_value(&GETHOSTBYNAME_EX_FUNCTION)),
            "gethostname" => Ok(builtin_value(&GETHOSTNAME_FUNCTION)),
            "getnameinfo" => Ok(builtin_value(&GETNAMEINFO_FUNCTION)),
            "getprotobyname" => Ok(builtin_value(&GETPROTOBYNAME_FUNCTION)),
            "getservbyname" => Ok(builtin_value(&GETSERVBYNAME_FUNCTION)),
            "getservbyport" => Ok(builtin_value(&GETSERVBYPORT_FUNCTION)),
            "htonl" => Ok(builtin_value(&HTONL_FUNCTION)),
            "htons" => Ok(builtin_value(&HTONS_FUNCTION)),
            "inet_aton" => Ok(builtin_value(&INET_ATON_FUNCTION)),
            "inet_ntoa" => Ok(builtin_value(&INET_NTOA_FUNCTION)),
            "inet_ntop" => Ok(builtin_value(&INET_NTOP_FUNCTION)),
            "inet_pton" => Ok(builtin_value(&INET_PTON_FUNCTION)),
            "ntohl" => Ok(builtin_value(&NTOHL_FUNCTION)),
            "ntohs" => Ok(builtin_value(&NTOHS_FUNCTION)),
            "setdefaulttimeout" => Ok(builtin_value(&SETDEFAULTTIMEOUT_FUNCTION)),
            _ => self.values.get(name).copied().ok_or_else(|| {
                ModuleError::AttributeError(format!("module '_socket' has no attribute '{}'", name))
            }),
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
fn leak_object_value<T>(object: T) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(object)) as *const ())
}

fn export_names_value() -> Value {
    leak_object_value(TupleObject::from_vec(
        EXPORTED_NAMES
            .iter()
            .copied()
            .map(|name| Value::string(intern(name)))
            .collect(),
    ))
}

fn os_error_type_value() -> Value {
    Value::object_ptr((&*crate::builtins::OS_ERROR) as *const _ as *const ())
}

fn timeout_error_type_value() -> Value {
    Value::object_ptr((&*crate::builtins::TIMEOUT_ERROR) as *const _ as *const ())
}

fn socket_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&SOCKET_CLASS) as *const ())
}

fn build_socket_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("socket"));
    class.set_attr(intern("__module__"), Value::string(intern("_socket")));
    class.set_attr(intern("__qualname__"), Value::string(intern("socket")));
    class.set_attr(intern("__doc__"), Value::string(intern("socket object")));
    class.set_attr(intern("__init__"), builtin_value(&SOCKET_INIT_METHOD));
    class.set_attr(intern("bind"), builtin_value(&SOCKET_BIND_METHOD));
    class.set_attr(intern("close"), builtin_value(&SOCKET_CLOSE_METHOD));
    class.set_attr(intern("connect"), builtin_value(&SOCKET_CONNECT_METHOD));
    class.set_attr(
        intern("connect_ex"),
        builtin_value(&SOCKET_CONNECT_EX_METHOD),
    );
    class.set_attr(intern("detach"), builtin_value(&SOCKET_DETACH_METHOD));
    class.set_attr(intern("fileno"), builtin_value(&SOCKET_FILENO_METHOD));
    class.set_attr(
        intern("getsockopt"),
        builtin_value(&SOCKET_GETSOCKOPT_METHOD),
    );
    class.set_attr(
        intern("gettimeout"),
        builtin_value(&SOCKET_GETTIMEOUT_METHOD),
    );
    class.set_attr(intern("listen"), builtin_value(&SOCKET_LISTEN_METHOD));
    class.set_attr(
        intern("setblocking"),
        builtin_value(&SOCKET_SETBLOCKING_METHOD),
    );
    class.set_attr(
        intern("setsockopt"),
        builtin_value(&SOCKET_SETSOCKOPT_METHOD),
    );
    class.set_attr(
        intern("settimeout"),
        builtin_value(&SOCKET_SETTIMEOUT_METHOD),
    );
    class.set_attr(intern("shutdown"), builtin_value(&SOCKET_SHUTDOWN_METHOD));
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_INIT | ClassFlags::NATIVE_HEAPTYPE);

    let class = Arc::new(class);
    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);
    class
}

fn socket_init(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "socket.__init__() takes at most 4 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let family = int_arg(
        args.get(1)
            .copied()
            .unwrap_or_else(|| Value::int(-1).unwrap()),
        "family",
    )?;
    let kind = int_arg(
        args.get(2)
            .copied()
            .unwrap_or_else(|| Value::int(-1).unwrap()),
        "type",
    )?;
    let proto = int_arg(
        args.get(3)
            .copied()
            .unwrap_or_else(|| Value::int(-1).unwrap()),
        "proto",
    )?;
    let fileno = match args.get(4).copied() {
        Some(value) if !value.is_none() => int_arg(value, "fileno")?,
        _ => NEXT_SOCKET_FD.fetch_add(1, Ordering::Relaxed),
    };

    let object = shaped_socket_mut(args[0])?;
    let registry = shape_registry();
    object.set_property(intern("family"), Value::int(family).unwrap(), registry);
    object.set_property(intern("type"), Value::int(kind).unwrap(), registry);
    object.set_property(intern("proto"), Value::int(proto).unwrap(), registry);
    object.set_property(
        intern("__prism_socket_fd__"),
        Value::int(fileno).unwrap(),
        registry,
    );
    object.set_property(
        intern("__prism_socket_closed__"),
        Value::bool(false),
        registry,
    );
    object.set_property(
        intern("__prism_socket_timeout__"),
        timeout_option_value(*DEFAULT_TIMEOUT.lock().unwrap()),
        registry,
    );
    Ok(Value::none())
}

fn socket_close(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "close", 1)?;
    let object = shaped_socket_mut(args[0])?;
    let registry = shape_registry();
    object.set_property(
        intern("__prism_socket_closed__"),
        Value::bool(true),
        registry,
    );
    object.set_property(
        intern("__prism_socket_fd__"),
        Value::int(-1).expect("sentinel fd should fit"),
        registry,
    );
    Ok(Value::none())
}

fn socket_detach(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "detach", 1)?;
    let object = shaped_socket_mut(args[0])?;
    let fd = socket_fd(object);
    let registry = shape_registry();
    object.set_property(
        intern("__prism_socket_closed__"),
        Value::bool(true),
        registry,
    );
    object.set_property(
        intern("__prism_socket_fd__"),
        Value::int(-1).unwrap(),
        registry,
    );
    Ok(Value::int(fd).unwrap_or_else(Value::none))
}

fn socket_fileno(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "fileno", 1)?;
    Ok(Value::int(socket_fd(shaped_socket_ref(args[0])?)).unwrap_or_else(Value::none))
}

fn socket_gettimeout(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "gettimeout", 1)?;
    Ok(shaped_socket_ref(args[0])?
        .get_property("__prism_socket_timeout__")
        .unwrap_or_else(Value::none))
}

fn socket_settimeout(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "settimeout() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let timeout = timeout_value(args[1])?;
    shaped_socket_mut(args[0])?.set_property(
        intern("__prism_socket_timeout__"),
        timeout_option_value(timeout),
        shape_registry(),
    );
    Ok(Value::none())
}

fn socket_setblocking(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "setblocking() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let timeout = if args[1].is_falsey() { Some(0.0) } else { None };
    shaped_socket_mut(args[0])?.set_property(
        intern("__prism_socket_timeout__"),
        timeout_option_value(timeout),
        shape_registry(),
    );
    Ok(Value::none())
}

fn socket_noop(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "socket method", 1)?;
    let _ = shaped_socket_ref(args[0])?;
    Ok(Value::none())
}

fn socket_connect_ex(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "connect_ex", 1)?;
    let _ = shaped_socket_ref(args[0])?;
    Ok(Value::int(0).unwrap())
}

fn socket_getsockopt(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "getsockopt", 1)?;
    let _ = shaped_socket_ref(args[0])?;
    Ok(Value::int(0).unwrap())
}

fn socket_getdefaulttimeout(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "getdefaulttimeout() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(timeout_option_value(*DEFAULT_TIMEOUT.lock().unwrap()))
}

fn socket_setdefaulttimeout(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "setdefaulttimeout() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    *DEFAULT_TIMEOUT.lock().unwrap() = timeout_value(args[0])?;
    Ok(Value::none())
}

fn socket_gethostname(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "gethostname() takes no arguments ({} given)",
            args.len()
        )));
    }
    let hostname = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "localhost".to_string());
    Ok(Value::string(intern(&hostname)))
}

fn socket_gethostbyname(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "gethostbyname() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let host = string_arg(args[0], "host")?;
    Ok(Value::string(intern(resolve_host_to_ipv4(&host))))
}

fn socket_gethostbyname_ex(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "gethostbyname_ex() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let host = string_arg(args[0], "host")?;
    let address = Value::string(intern(resolve_host_to_ipv4(&host)));
    Ok(tuple_value(vec![
        Value::string(intern(&host)),
        list_value(Vec::new()),
        list_value(vec![address]),
    ]))
}

fn socket_gethostbyaddr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "gethostbyaddr() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let host = string_arg(args[0], "host")?;
    Ok(tuple_value(vec![
        Value::string(intern(&host)),
        list_value(Vec::new()),
        list_value(vec![Value::string(intern(&host))]),
    ]))
}

fn socket_getaddrinfo(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=6).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "getaddrinfo() takes 2 to 6 arguments ({} given)",
            args.len()
        )));
    }
    let host = if args[0].is_none() {
        "127.0.0.1".to_string()
    } else {
        string_arg(args[0], "host")?
    };
    let port = port_number(args[1])?;
    let family = optional_i64_arg(args.get(2).copied(), 0, "family")?;
    let kind = optional_i64_arg(args.get(3).copied(), 0, "type")?;
    let proto = optional_i64_arg(args.get(4).copied(), 0, "proto")?;
    let family = if family == 0 { 2 } else { family };
    let kind = if kind == 0 { 1 } else { kind };
    let sockaddr = if family == AF_INET6_VALUE {
        tuple_value(vec![
            Value::string(intern(resolve_host_to_ipv6(&host))),
            Value::int(port).unwrap(),
            Value::int(0).unwrap(),
            Value::int(0).unwrap(),
        ])
    } else {
        tuple_value(vec![
            Value::string(intern(resolve_host_to_ipv4(&host))),
            Value::int(port).unwrap(),
        ])
    };
    let entry = tuple_value(vec![
        Value::int(family).unwrap(),
        Value::int(kind).unwrap(),
        Value::int(proto).unwrap(),
        Value::string(intern("")),
        sockaddr,
    ]);
    Ok(list_value(vec![entry]))
}

fn socket_getnameinfo(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "getnameinfo() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }
    Ok(tuple_value(vec![
        Value::string(intern("localhost")),
        Value::string(intern("0")),
    ]))
}

fn socket_getprotobyname(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "getprotobyname() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let name = string_arg(args[0], "protocol")?.to_ascii_lowercase();
    let proto = match name.as_str() {
        "ip" => 0,
        "tcp" => 6,
        "udp" => 17,
        _ => {
            return Err(BuiltinError::OSError(format!("protocol not found: {name}")));
        }
    };
    Ok(Value::int(proto).unwrap())
}

fn socket_getservbyname(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "getservbyname() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }
    let service = string_arg(args[0], "service")?.to_ascii_lowercase();
    let port = service_port(&service)
        .ok_or_else(|| BuiltinError::OSError(format!("service/proto not found: {service}")))?;
    Ok(Value::int(port).unwrap())
}

fn socket_getservbyport(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "getservbyport() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }
    let port = int_arg(args[0], "port")?;
    let service = match port {
        80 => "http",
        443 => "https",
        53 => "domain",
        25 => "smtp",
        _ => "unknown",
    };
    Ok(Value::string(intern(service)))
}

fn socket_htonl(args: &[Value]) -> Result<Value, BuiltinError> {
    convert_u32(args, "htonl", u32::to_be)
}

fn socket_ntohl(args: &[Value]) -> Result<Value, BuiltinError> {
    convert_u32(args, "ntohl", u32::from_be)
}

fn socket_htons(args: &[Value]) -> Result<Value, BuiltinError> {
    convert_u16(args, "htons", u16::to_be)
}

fn socket_ntohs(args: &[Value]) -> Result<Value, BuiltinError> {
    convert_u16(args, "ntohs", u16::from_be)
}

fn socket_inet_aton(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "inet_aton() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let addr = string_arg(args[0], "address")?
        .parse::<Ipv4Addr>()
        .map_err(|_| {
            BuiltinError::OSError("illegal IP address string passed to inet_aton".into())
        })?;
    Ok(bytes_value(&addr.octets()))
}

fn socket_inet_ntoa(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "inet_ntoa() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let bytes = bytes_arg(args[0], "packed IP wrong length for inet_ntoa")?;
    if bytes.len() != 4 {
        return Err(BuiltinError::OSError(
            "packed IP wrong length for inet_ntoa".to_string(),
        ));
    }
    let octets = <[u8; 4]>::try_from(bytes.as_slice())
        .expect("inet_ntoa length check guarantees four octets");
    Ok(Value::string(intern(&Ipv4Addr::from(octets).to_string())))
}

fn socket_inet_pton(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "inet_pton() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    let family = int_arg(args[0], "family")?;
    let addr = string_arg(args[1], "address")?;
    if family == 2 {
        let parsed = addr.parse::<Ipv4Addr>().map_err(|_| {
            BuiltinError::OSError("illegal IP address string passed to inet_pton".into())
        })?;
        return Ok(bytes_value(&parsed.octets()));
    }
    if family == AF_INET6_VALUE {
        let parsed = addr.parse::<Ipv6Addr>().map_err(|_| {
            BuiltinError::OSError("illegal IP address string passed to inet_pton".into())
        })?;
        return Ok(bytes_value(&parsed.octets()));
    }
    Err(BuiltinError::OSError(
        "address family not supported by protocol".to_string(),
    ))
}

fn socket_inet_ntop(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "inet_ntop() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    let family = int_arg(args[0], "family")?;
    let bytes = bytes_arg(args[1], "packed IP wrong length for inet_ntop")?;
    if family == 2 && bytes.len() == 4 {
        return Ok(Value::string(intern(
            &Ipv4Addr::from(<[u8; 4]>::try_from(bytes.as_slice()).unwrap()).to_string(),
        )));
    }
    if family == AF_INET6_VALUE && bytes.len() == 16 {
        return Ok(Value::string(intern(
            &Ipv6Addr::from(<[u8; 16]>::try_from(bytes.as_slice()).unwrap()).to_string(),
        )));
    }
    Err(BuiltinError::OSError(
        "packed IP wrong length for inet_ntop".to_string(),
    ))
}

fn cmsg_len(args: &[Value]) -> Result<Value, BuiltinError> {
    cmsg_size(args, "CMSG_LEN", false)
}

fn cmsg_space(args: &[Value]) -> Result<Value, BuiltinError> {
    cmsg_size(args, "CMSG_SPACE", true)
}

fn cmsg_size(args: &[Value], name: &str, align_payload: bool) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{name}() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let payload = usize::try_from(int_arg(args[0], "length")?)
        .map_err(|_| BuiltinError::OverflowError("CMSG length is too large".to_string()))?;
    let header = 16_usize;
    let total = if align_payload {
        header + align_usize(payload, std::mem::size_of::<usize>())
    } else {
        header + payload
    };
    Ok(Value::int(total as i64).expect("CMSG size should fit"))
}

#[inline]
fn align_usize(value: usize, align: usize) -> usize {
    let mask = align - 1;
    (value + mask) & !mask
}

fn exact_min_args(args: &[Value], fn_name: &str, min: usize) -> Result<(), BuiltinError> {
    if args.len() >= min {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{fn_name}() missing required receiver"
        )))
    }
}

fn shaped_socket_ref(value: Value) -> Result<&'static ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_socket.socket methods require an instance".to_string())
    })?;
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn shaped_socket_mut(value: Value) -> Result<&'static mut ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_socket.socket methods require an instance".to_string())
    })?;
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn socket_fd(object: &ShapedObject) -> i64 {
    object
        .get_property("__prism_socket_fd__")
        .and_then(|value| value.as_int())
        .unwrap_or(-1)
}

fn timeout_value(value: Value) -> Result<Option<f64>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }
    let timeout = if let Some(float) = value.as_float() {
        float
    } else if let Some(integer) = value.as_int() {
        integer as f64
    } else {
        return Err(BuiltinError::TypeError(
            "timeout must be a non-negative number or None".to_string(),
        ));
    };
    if timeout < 0.0 {
        return Err(BuiltinError::ValueError(
            "Timeout value out of range".to_string(),
        ));
    }
    Ok(Some(timeout))
}

fn timeout_option_value(timeout: Option<f64>) -> Value {
    timeout.map(Value::float).unwrap_or_else(Value::none)
}

fn int_arg(value: Value, name: &str) -> Result<i64, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(if flag { 1 } else { 0 });
    }
    value
        .as_int()
        .ok_or_else(|| BuiltinError::TypeError(format!("{name} must be an integer")))
}

fn optional_i64_arg(value: Option<Value>, default: i64, name: &str) -> Result<i64, BuiltinError> {
    value.map_or(Ok(default), |value| int_arg(value, name))
}

fn string_arg(value: Value, name: &str) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|text| text.as_str().to_string())
        .ok_or_else(|| BuiltinError::TypeError(format!("{name} must be str")))
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

fn bytes_value(bytes: &[u8]) -> Value {
    leak_object_value(BytesObject::from_slice(bytes))
}

fn tuple_value(items: Vec<Value>) -> Value {
    leak_object_value(TupleObject::from_vec(items))
}

fn list_value(items: Vec<Value>) -> Value {
    leak_object_value(ListObject::from_iter(items))
}

fn port_number(value: Value) -> Result<i64, BuiltinError> {
    if value.is_none() {
        return Ok(0);
    }
    if let Some(port) = value.as_int() {
        return Ok(port);
    }
    let service = string_arg(value, "port")?;
    Ok(service_port(&service).unwrap_or(0))
}

fn service_port(service: &str) -> Option<i64> {
    Some(match service {
        "http" => 80,
        "https" => 443,
        "domain" | "dns" => 53,
        "smtp" => 25,
        "imap" => 143,
        "pop3" => 110,
        _ => return None,
    })
}

fn resolve_host_to_ipv4(host: &str) -> &str {
    if host == "localhost" || host.is_empty() {
        "127.0.0.1"
    } else if host.parse::<Ipv4Addr>().is_ok() {
        host
    } else {
        "127.0.0.1"
    }
}

fn resolve_host_to_ipv6(host: &str) -> &str {
    if host == "localhost" || host.is_empty() {
        "::1"
    } else if host.parse::<Ipv6Addr>().is_ok() {
        host
    } else {
        "::1"
    }
}

fn convert_u16(
    args: &[Value],
    fn_name: &str,
    convert: fn(u16) -> u16,
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let value = u16::try_from(int_arg(args[0], "integer")?)
        .map_err(|_| BuiltinError::OverflowError("unsigned short is out of range".to_string()))?;
    Ok(Value::int(i64::from(convert(value))).unwrap())
}

fn convert_u32(
    args: &[Value],
    fn_name: &str,
    convert: fn(u32) -> u32,
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let value = u32::try_from(int_arg(args[0], "integer")?)
        .map_err(|_| BuiltinError::OverflowError("unsigned int is out of range".to_string()))?;
    Ok(Value::int(i64::from(convert(value))).unwrap())
}

trait ValueFalsey {
    fn is_falsey(self) -> bool;
}

impl ValueFalsey for Value {
    fn is_falsey(self) -> bool {
        self.is_none() || self.as_bool() == Some(false) || self.as_int() == Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::allocate_heap_instance_for_class;

    fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
        let ptr = value
            .as_object_ptr()
            .expect("expected builtin function object");
        unsafe { &*(ptr as *const BuiltinFunctionObject) }
    }

    fn new_socket_instance() -> Value {
        let instance = allocate_heap_instance_for_class(&SOCKET_CLASS);
        Value::object_ptr(Box::into_raw(Box::new(instance)) as *const ())
    }

    #[test]
    fn test_socket_module_exposes_socket_py_bootstrap_surface() {
        let module = SocketModule::new();

        for name in [
            "__all__",
            "AF_INET",
            "AF_INET6",
            "SOCK_STREAM",
            "SOL_SOCKET",
            "socket",
            "SocketType",
            "error",
            "timeout",
            "getaddrinfo",
            "inet_aton",
        ] {
            assert!(module.get_attr(name).is_ok(), "{name} should be exported");
        }
    }

    #[test]
    fn test_socket_class_initializes_metadata_properties() {
        let instance = new_socket_instance();
        socket_init(&[
            instance,
            Value::int(2).unwrap(),
            Value::int(1).unwrap(),
            Value::int(6).unwrap(),
            Value::none(),
        ])
        .expect("socket.__init__ should accept metadata arguments");

        let object = shaped_socket_ref(instance).expect("socket instance");
        assert_eq!(
            object.get_property("family").and_then(|v| v.as_int()),
            Some(2)
        );
        assert_eq!(
            object.get_property("type").and_then(|v| v.as_int()),
            Some(1)
        );
        assert_eq!(
            object.get_property("proto").and_then(|v| v.as_int()),
            Some(6)
        );
        assert!(socket_fd(object) >= 10_000);
    }

    #[test]
    fn test_socket_timeout_helpers_roundtrip() {
        let setdefault =
            builtin_from_value(SocketModule::new().get_attr("setdefaulttimeout").unwrap());
        let getdefault =
            builtin_from_value(SocketModule::new().get_attr("getdefaulttimeout").unwrap());

        setdefault.call(&[Value::float(2.5)]).expect("set timeout");
        assert_eq!(
            getdefault.call(&[]).expect("get timeout").as_float(),
            Some(2.5)
        );

        let instance = new_socket_instance();
        socket_init(&[instance]).expect("socket init");
        assert_eq!(
            socket_gettimeout(&[instance])
                .expect("instance timeout")
                .as_float(),
            Some(2.5)
        );

        setdefault.call(&[Value::none()]).expect("reset timeout");
    }

    #[test]
    fn test_socket_address_conversion_helpers() {
        let packed = socket_inet_aton(&[Value::string(intern("127.0.0.1"))]).expect("inet_aton");
        let text = socket_inet_ntoa(&[packed]).expect("inet_ntoa");
        assert_eq!(
            value_as_string_ref(text).expect("string").as_str(),
            "127.0.0.1"
        );

        let net = socket_htons(&[Value::int(0x1234).unwrap()]).expect("htons");
        let host = socket_ntohs(&[net]).expect("ntohs");
        assert_eq!(host.as_int(), Some(0x1234));
    }

    #[test]
    fn test_socket_getaddrinfo_returns_socket_py_shape() {
        let result = socket_getaddrinfo(&[
            Value::string(intern("localhost")),
            Value::int(80).unwrap(),
            Value::int(2).unwrap(),
            Value::int(1).unwrap(),
            Value::int(6).unwrap(),
        ])
        .expect("getaddrinfo");
        let ptr = result.as_object_ptr().expect("result should be a list");
        let list = unsafe { &*(ptr as *const ListObject) };
        assert_eq!(list.len(), 1);
    }
}
