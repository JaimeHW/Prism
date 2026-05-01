//! Native `_socket` bootstrap surface.
//!
//! CPython's `socket.py` is a pure-Python wrapper around the `_socket`
//! extension module. Prism exposes the same import-time shape here so the
//! standard library can define its higher-level `socket.socket` subclass and
//! export list before full OS socket handles are implemented.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject, runtime_error_to_builtin_error};
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::descriptor::PropertyDescriptor;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::value_to_bigint;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use rustc_hash::FxHashMap;
use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

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

const AF_INET_VALUE: i64 = 2;
const SOCK_STREAM_VALUE: i64 = 1;

const EXPORTED_NAMES: &[&str] = &[
    "AF_INET",
    "AF_INET6",
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
static SOCKET_STATES: LazyLock<Mutex<FxHashMap<i64, SocketState>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

static SOCKET_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.socket.__init__"), socket_init)
});
static SOCKET_FAMILY_GET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.family.fget"), socket_family_get)
});
static SOCKET_TYPE_GET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.type.fget"), socket_type_get)
});
static SOCKET_PROTO_GET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.proto.fget"), socket_proto_get)
});
static SOCKET_FAMILY_PROPERTY: LazyLock<PropertyDescriptor> =
    LazyLock::new(|| PropertyDescriptor::new_getter(builtin_value(&SOCKET_FAMILY_GET_METHOD)));
static SOCKET_TYPE_PROPERTY: LazyLock<PropertyDescriptor> =
    LazyLock::new(|| PropertyDescriptor::new_getter(builtin_value(&SOCKET_TYPE_GET_METHOD)));
static SOCKET_PROTO_PROPERTY: LazyLock<PropertyDescriptor> =
    LazyLock::new(|| PropertyDescriptor::new_getter(builtin_value(&SOCKET_PROTO_GET_METHOD)));
static SOCKET_CLOSE_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.close"), socket_close));
static SOCKET_DETACH_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.detach"), socket_detach));
static SOCKET_FILENO_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.fileno"), socket_fileno));
static SOCKET_GETBLOCKING_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.getblocking"), socket_getblocking)
});
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
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_socket.socket.bind"), socket_bind));
static SOCKET_LISTEN_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.socket.listen"), socket_listen)
});
static SOCKET_CONNECT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.socket.connect"), socket_connect)
});
static SOCKET_CONNECT_EX_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.socket.connect_ex"), socket_connect_ex)
});
static SOCKET_ACCEPT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket._accept"), socket_accept)
});
static SOCKET_GETSOCKNAME_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.getsockname"), socket_getsockname)
});
static SOCKET_GETPEERNAME_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.getpeername"), socket_getpeername)
});
static SOCKET_DUP_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.dup"), socket_dup));
static SOCKET_RECV_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_socket.socket.recv"), socket_recv));
static SOCKET_SEND_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.socket.send"), socket_send));
static SOCKET_SENDALL_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.sendall"), socket_sendall)
});
static SOCKET_SHUTDOWN_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.socket.shutdown"), socket_shutdown)
});
static SOCKET_SETSOCKOPT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.socket.setsockopt"), socket_setsockopt)
});
static SOCKET_GETSOCKOPT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.socket.getsockopt"), socket_getsockopt)
});

static CMSG_LEN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_socket.CMSG_LEN"), cmsg_len));
static CMSG_SPACE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_socket.CMSG_SPACE"), cmsg_space));
static GETADDRINFO_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.getaddrinfo"), socket_getaddrinfo)
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
    BuiltinFunctionObject::new_vm(Arc::from("_socket.getnameinfo"), socket_getnameinfo)
});
static GETPROTOBYNAME_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.getprotobyname"), socket_getprotobyname)
});
static GETSERVBYNAME_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_socket.getservbyname"), socket_getservbyname)
});
static GETSERVBYPORT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.getservbyport"), socket_getservbyport)
});
static HTONL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_socket.htonl"), socket_htonl));
static HTONS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_socket.htons"), socket_htons));
static INET_ATON_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.inet_aton"), socket_inet_aton));
static INET_NTOA_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_socket.inet_ntoa"), socket_inet_ntoa));
static INET_NTOP_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.inet_ntop"), socket_inet_ntop)
});
static INET_PTON_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("_socket.inet_pton"), socket_inet_pton)
});
static NTOHL_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_socket.ntohl"), socket_ntohl));
static NTOHS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_socket.ntohs"), socket_ntohs));
static SETDEFAULTTIMEOUT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_socket.setdefaulttimeout"),
        socket_setdefaulttimeout,
    )
});

#[derive(Debug)]
struct SocketState {
    family: i64,
    kind: i64,
    proto: i64,
    timeout: Option<f64>,
    handle: SocketHandle,
    bound_addr: Option<SocketAddr>,
    options: FxHashMap<(i64, i64), i64>,
}

#[derive(Debug)]
enum SocketHandle {
    Empty,
    Listener(TcpListener),
    Stream(TcpStream),
}

pub(crate) fn write_signal_wakeup_byte(fd: i64, byte: u8) -> io::Result<()> {
    let mut states = SOCKET_STATES
        .lock()
        .expect("socket state registry lock poisoned");
    let state = states.get_mut(&fd).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "signal wakeup file descriptor is not an open Prism socket",
        )
    })?;
    let SocketHandle::Stream(stream) = &mut state.handle else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "signal wakeup file descriptor is not a stream socket",
        ));
    };

    match stream.write(&[byte]) {
        Ok(1) => Ok(()),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "failed to write signal wakeup byte",
        )),
        Err(error) => Err(error),
    }
}

impl SocketState {
    fn new(family: i64, kind: i64, proto: i64, timeout: Option<f64>) -> Self {
        Self {
            family: normalize_family(family),
            kind: normalize_socket_type(kind),
            proto: normalize_proto(proto),
            timeout,
            handle: SocketHandle::Empty,
            bound_addr: None,
            options: FxHashMap::default(),
        }
    }

    fn local_addr(&self) -> Option<SocketAddr> {
        match &self.handle {
            SocketHandle::Listener(listener) => listener.local_addr().ok(),
            SocketHandle::Stream(stream) => stream.local_addr().ok(),
            SocketHandle::Empty => self.bound_addr,
        }
    }

    fn peer_addr(&self) -> Option<SocketAddr> {
        match &self.handle {
            SocketHandle::Stream(stream) => stream.peer_addr().ok(),
            SocketHandle::Listener(_) | SocketHandle::Empty => None,
        }
    }

    fn set_nonblocking(&self, nonblocking: bool) -> Result<(), BuiltinError> {
        match &self.handle {
            SocketHandle::Listener(listener) => listener.set_nonblocking(nonblocking),
            SocketHandle::Stream(stream) => stream.set_nonblocking(nonblocking),
            SocketHandle::Empty => Ok(()),
        }
        .map_err(socket_io_error)
    }

    fn apply_timeout(&self) -> Result<(), BuiltinError> {
        let read_write_timeout =
            timeout_duration(self.timeout).filter(|duration| !duration.is_zero());
        match &self.handle {
            SocketHandle::Stream(stream) => {
                stream
                    .set_read_timeout(read_write_timeout)
                    .map_err(socket_io_error)?;
                stream
                    .set_write_timeout(read_write_timeout)
                    .map_err(socket_io_error)?;
                stream
                    .set_nonblocking(self.timeout == Some(0.0))
                    .map_err(socket_io_error)
            }
            SocketHandle::Listener(listener) => listener
                .set_nonblocking(self.timeout == Some(0.0))
                .map_err(socket_io_error),
            SocketHandle::Empty => Ok(()),
        }
    }

    fn clone_state(&self) -> Result<Self, BuiltinError> {
        let handle = match &self.handle {
            SocketHandle::Empty => SocketHandle::Empty,
            SocketHandle::Listener(listener) => {
                SocketHandle::Listener(listener.try_clone().map_err(socket_io_error)?)
            }
            SocketHandle::Stream(stream) => {
                SocketHandle::Stream(stream.try_clone().map_err(socket_io_error)?)
            }
        };
        let clone = Self {
            family: self.family,
            kind: self.kind,
            proto: self.proto,
            timeout: self.timeout,
            handle,
            bound_addr: self.bound_addr,
            options: self.options.clone(),
        };
        clone.apply_timeout()?;
        Ok(clone)
    }
}

#[derive(Clone, Copy)]
struct SocketSnapshot {
    fd: i64,
    family: i64,
    kind: i64,
    proto: i64,
    timeout: Option<f64>,
}

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
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
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

#[inline]
fn property_value(property: &'static LazyLock<PropertyDescriptor>) -> Value {
    Value::object_ptr(&**property as *const PropertyDescriptor as *const ())
}

fn build_socket_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("socket"));
    class.set_attr(intern("__module__"), Value::string(intern("_socket")));
    class.set_attr(intern("__qualname__"), Value::string(intern("socket")));
    class.set_attr(intern("__doc__"), Value::string(intern("socket object")));
    class.set_attr(intern("__init__"), builtin_value(&SOCKET_INIT_METHOD));
    class.set_attr(intern("family"), property_value(&SOCKET_FAMILY_PROPERTY));
    class.set_attr(intern("type"), property_value(&SOCKET_TYPE_PROPERTY));
    class.set_attr(intern("proto"), property_value(&SOCKET_PROTO_PROPERTY));
    class.set_attr(intern("_accept"), builtin_value(&SOCKET_ACCEPT_METHOD));
    class.set_attr(intern("bind"), builtin_value(&SOCKET_BIND_METHOD));
    class.set_attr(intern("close"), builtin_value(&SOCKET_CLOSE_METHOD));
    class.set_attr(intern("connect"), builtin_value(&SOCKET_CONNECT_METHOD));
    class.set_attr(
        intern("connect_ex"),
        builtin_value(&SOCKET_CONNECT_EX_METHOD),
    );
    class.set_attr(intern("detach"), builtin_value(&SOCKET_DETACH_METHOD));
    class.set_attr(intern("dup"), builtin_value(&SOCKET_DUP_METHOD));
    class.set_attr(intern("fileno"), builtin_value(&SOCKET_FILENO_METHOD));
    class.set_attr(
        intern("getblocking"),
        builtin_value(&SOCKET_GETBLOCKING_METHOD),
    );
    class.set_attr(
        intern("getpeername"),
        builtin_value(&SOCKET_GETPEERNAME_METHOD),
    );
    class.set_attr(
        intern("getsockname"),
        builtin_value(&SOCKET_GETSOCKNAME_METHOD),
    );
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
    class.set_attr(intern("recv"), builtin_value(&SOCKET_RECV_METHOD));
    class.set_attr(intern("send"), builtin_value(&SOCKET_SEND_METHOD));
    class.set_attr(intern("sendall"), builtin_value(&SOCKET_SENDALL_METHOD));
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

fn socket_init(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "socket.__init__() takes at most 4 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let family = int_arg_vm(
        vm,
        args.get(1)
            .copied()
            .unwrap_or_else(|| Value::int(-1).unwrap()),
        "family",
    )?;
    let kind = int_arg_vm(
        vm,
        args.get(2)
            .copied()
            .unwrap_or_else(|| Value::int(-1).unwrap()),
        "type",
    )?;
    let proto = int_arg_vm(
        vm,
        args.get(3)
            .copied()
            .unwrap_or_else(|| Value::int(-1).unwrap()),
        "proto",
    )?;
    let fileno = match args.get(4).copied() {
        Some(value) if !value.is_none() => int_arg_vm(vm, value, "fileno")?,
        _ => NEXT_SOCKET_FD.fetch_add(1, Ordering::Relaxed),
    };

    let snapshot = initialize_socket_state(fileno, family, kind, proto);
    write_socket_snapshot(shaped_socket_mut(args[0])?, snapshot);
    Ok(Value::none())
}

fn socket_family_get(args: &[Value]) -> Result<Value, BuiltinError> {
    socket_member_get(args, "family")
}

fn socket_type_get(args: &[Value]) -> Result<Value, BuiltinError> {
    socket_member_get(args, "type")
}

fn socket_proto_get(args: &[Value]) -> Result<Value, BuiltinError> {
    socket_member_get(args, "proto")
}

fn socket_close(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "close", 1)?;
    let object = shaped_socket_mut(args[0])?;
    let fd = socket_fd(object);
    if fd >= 0 {
        SOCKET_STATES.lock().unwrap().remove(&fd);
    }
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

fn socket_getblocking(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "getblocking", 1)?;
    let timeout = socket_timeout(shaped_socket_ref(args[0])?);
    Ok(Value::bool(timeout != Some(0.0)))
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
    let object = shaped_socket_mut(args[0])?;
    if let Some(fd) = open_socket_fd(object) {
        let mut states = SOCKET_STATES.lock().unwrap();
        if let Some(state) = states.get_mut(&fd) {
            state.timeout = timeout;
            state.apply_timeout()?;
        }
    }
    object.set_property(
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
    let object = shaped_socket_mut(args[0])?;
    if let Some(fd) = open_socket_fd(object) {
        let mut states = SOCKET_STATES.lock().unwrap();
        if let Some(state) = states.get_mut(&fd) {
            state.timeout = timeout;
            state.apply_timeout()?;
        }
    }
    object.set_property(
        intern("__prism_socket_timeout__"),
        timeout_option_value(timeout),
        shape_registry(),
    );
    Ok(Value::none())
}

fn socket_connect_ex(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    match socket_connect(vm, args) {
        Ok(_) => Ok(Value::int(0).unwrap()),
        Err(_) => Ok(Value::int(1).unwrap()),
    }
}

fn socket_getsockopt(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 3 {
        return Err(BuiltinError::TypeError(
            "getsockopt() missing required argument".to_string(),
        ));
    }
    let object = shaped_socket_ref(args[0])?;
    let level = int_arg_vm(vm, args[1], "level")?;
    let optname = int_arg_vm(vm, args[2], "optname")?;
    let Some(fd) = open_socket_fd(object) else {
        return Err(BuiltinError::OSError("Bad file descriptor".to_string()));
    };
    let states = SOCKET_STATES.lock().unwrap();
    let Some(state) = states.get(&fd) else {
        return Err(BuiltinError::OSError("Bad file descriptor".to_string()));
    };
    if level == 6
        && optname == 1
        && let SocketHandle::Stream(stream) = &state.handle
        && let Ok(enabled) = stream.nodelay()
    {
        return Ok(Value::int(i64::from(enabled)).unwrap());
    }
    let value = state.options.get(&(level, optname)).copied().unwrap_or(0);
    Ok(Value::int(value).unwrap())
}

fn socket_bind(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "bind() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let addr = sockaddr_arg(vm, args[1], socket_family(object))?;
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get_mut(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;

    if state.kind == SOCK_STREAM_VALUE {
        let listener = TcpListener::bind(addr).map_err(socket_io_error)?;
        listener
            .set_nonblocking(state.timeout == Some(0.0))
            .map_err(socket_io_error)?;
        state.bound_addr = listener.local_addr().ok();
        state.handle = SocketHandle::Listener(listener);
    } else {
        state.bound_addr = Some(addr);
    }
    Ok(Value::none())
}

fn socket_listen(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "listen() takes at most one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    if let Some(backlog) = args.get(1).copied() {
        let _ = int_arg_vm(vm, backlog, "backlog")?;
    }
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get_mut(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    if matches!(state.handle, SocketHandle::Empty) && state.kind == SOCK_STREAM_VALUE {
        let addr = default_bind_addr(state.family);
        let listener = TcpListener::bind(addr).map_err(socket_io_error)?;
        listener
            .set_nonblocking(state.timeout == Some(0.0))
            .map_err(socket_io_error)?;
        state.bound_addr = listener.local_addr().ok();
        state.handle = SocketHandle::Listener(listener);
    }
    Ok(Value::none())
}

fn socket_connect(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "connect() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let addr = sockaddr_arg(vm, args[1], socket_family(object))?;
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get_mut(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    if state.kind != SOCK_STREAM_VALUE {
        state.bound_addr = Some(addr);
        return Ok(Value::none());
    }

    let stream = connect_tcp_stream(addr, state.timeout)?;
    apply_stream_options(&stream, state)?;
    state.bound_addr = stream.local_addr().ok();
    state.handle = SocketHandle::Stream(stream);
    Ok(Value::none())
}

fn socket_accept(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "_accept", 1)?;
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get_mut(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    let SocketHandle::Listener(listener) = &state.handle else {
        return Err(BuiltinError::OSError("socket is not listening".to_string()));
    };
    let family = state.family;
    let kind = state.kind;
    let proto = state.proto;
    let timeout = state.timeout;

    let (stream, peer) = listener.accept().map_err(socket_io_error)?;
    stream
        .set_nonblocking(timeout == Some(0.0))
        .map_err(socket_io_error)?;
    let accepted_fd = NEXT_SOCKET_FD.fetch_add(1, Ordering::Relaxed);
    let mut accepted = SocketState::new(family, kind, proto, timeout);
    accepted.bound_addr = stream.local_addr().ok();
    accepted.handle = SocketHandle::Stream(stream);
    states.insert(accepted_fd, accepted);
    Ok(tuple_value(vec![
        Value::int(accepted_fd).unwrap(),
        sockaddr_value(peer, family),
    ]))
}

fn socket_getsockname(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "getsockname", 1)?;
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    let addr = state
        .local_addr()
        .unwrap_or_else(|| default_socket_addr(state.family));
    Ok(sockaddr_value(addr, state.family))
}

fn socket_getpeername(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "getpeername", 1)?;
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    let addr = state
        .peer_addr()
        .ok_or_else(|| BuiltinError::OSError("socket is not connected".to_string()))?;
    Ok(sockaddr_value(addr, state.family))
}

fn socket_dup(args: &[Value]) -> Result<Value, BuiltinError> {
    exact_min_args(args, "dup", 1)?;
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let new_fd = NEXT_SOCKET_FD.fetch_add(1, Ordering::Relaxed);
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    let cloned = state.clone_state()?;
    states.insert(new_fd, cloned);
    Ok(Value::int(new_fd).unwrap())
}

fn socket_recv(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "recv() takes 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    if let Some(flags) = args.get(2).copied() {
        let _ = int_arg_vm(vm, flags, "flags")?;
    }
    let size = usize::try_from(int_arg_vm(vm, args[1], "bufsize")?)
        .map_err(|_| BuiltinError::ValueError("negative buffersize in recv".to_string()))?;
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get_mut(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    let SocketHandle::Stream(stream) = &mut state.handle else {
        return Err(BuiltinError::OSError("socket is not connected".to_string()));
    };
    let mut buffer = vec![0_u8; size];
    let read = stream.read(&mut buffer).map_err(socket_io_error)?;
    buffer.truncate(read);
    Ok(bytes_value(&buffer))
}

fn socket_send(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "send() takes 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let bytes = bytes_arg(args[1], "a bytes-like object is required")?;
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get_mut(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    let SocketHandle::Stream(stream) = &mut state.handle else {
        return Err(BuiltinError::OSError("socket is not connected".to_string()));
    };
    let written = stream.write(&bytes).map_err(socket_io_error)?;
    Ok(Value::int(written as i64).unwrap())
}

fn socket_sendall(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "sendall() takes 1 or 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let bytes = bytes_arg(args[1], "a bytes-like object is required")?;
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get_mut(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    let SocketHandle::Stream(stream) = &mut state.handle else {
        return Err(BuiltinError::OSError("socket is not connected".to_string()));
    };
    stream.write_all(&bytes).map_err(socket_io_error)?;
    Ok(Value::none())
}

fn socket_shutdown(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "shutdown() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let how = match args[1].as_int().unwrap_or(2) {
        0 => Shutdown::Read,
        1 => Shutdown::Write,
        _ => Shutdown::Both,
    };
    let object = shaped_socket_ref(args[0])?;
    let fd = require_open_socket_fd(object)?;
    let states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    if let SocketHandle::Stream(stream) = &state.handle {
        stream.shutdown(how).map_err(socket_io_error)?;
    }
    Ok(Value::none())
}

fn socket_setsockopt(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 4 {
        return Err(BuiltinError::TypeError(
            "setsockopt() missing required argument".to_string(),
        ));
    }
    let object = shaped_socket_ref(args[0])?;
    let level = int_arg_vm(vm, args[1], "level")?;
    let optname = int_arg_vm(vm, args[2], "optname")?;
    let optvalue = if let Some(integer) = int_value_to_i64(args[3])? {
        integer
    } else {
        1
    };
    let fd = require_open_socket_fd(object)?;
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .get_mut(&fd)
        .ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))?;
    state.options.insert((level, optname), optvalue);
    if level == 6
        && optname == 1
        && let SocketHandle::Stream(stream) = &state.handle
    {
        stream.set_nodelay(optvalue != 0).map_err(socket_io_error)?;
    }
    Ok(Value::none())
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

fn socket_getaddrinfo(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
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
    let port = port_number(vm, args[1])?;
    let family = optional_i64_arg_vm(vm, args.get(2).copied(), 0, "family")?;
    let kind = optional_i64_arg_vm(vm, args.get(3).copied(), 0, "type")?;
    let proto = optional_i64_arg_vm(vm, args.get(4).copied(), 0, "proto")?;
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

fn socket_getnameinfo(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "getnameinfo() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }
    if let Some(flags) = args.get(1).copied() {
        let _ = int_arg_vm(vm, flags, "flags")?;
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

fn socket_getservbyport(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "getservbyport() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }
    let port = int_arg_vm(vm, args[0], "port")?;
    let service = match port {
        80 => "http",
        443 => "https",
        53 => "domain",
        25 => "smtp",
        _ => "unknown",
    };
    Ok(Value::string(intern(service)))
}

fn socket_htonl(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    convert_u32(vm, args, "htonl", u32::to_be)
}

fn socket_ntohl(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    convert_u32(vm, args, "ntohl", u32::from_be)
}

fn socket_htons(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    convert_u16(vm, args, "htons", u16::to_be)
}

fn socket_ntohs(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    convert_u16(vm, args, "ntohs", u16::from_be)
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

fn socket_inet_pton(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "inet_pton() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    let family = int_arg_vm(vm, args[0], "family")?;
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

fn socket_inet_ntop(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "inet_ntop() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    let family = int_arg_vm(vm, args[0], "family")?;
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

fn cmsg_len(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    cmsg_size(vm, args, "CMSG_LEN", false)
}

fn cmsg_space(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    cmsg_size(vm, args, "CMSG_SPACE", true)
}

fn cmsg_size(
    vm: &mut VirtualMachine,
    args: &[Value],
    name: &str,
    align_payload: bool,
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{name}() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let payload = usize::try_from(int_arg_vm(vm, args[0], "length")?)
        .map_err(|_| BuiltinError::OverflowError("CMSG length is too large".to_string()))?;
    let header = 16_usize;
    let total = if align_payload {
        header + align_usize(payload, std::mem::size_of::<usize>())
    } else {
        header + payload
    };
    Ok(Value::int(total as i64).expect("CMSG size should fit"))
}

fn initialize_socket_state(fileno: i64, family: i64, kind: i64, proto: i64) -> SocketSnapshot {
    let default_timeout = *DEFAULT_TIMEOUT.lock().unwrap();
    let mut states = SOCKET_STATES.lock().unwrap();
    let state = states
        .entry(fileno)
        .or_insert_with(|| SocketState::new(family, kind, proto, default_timeout));
    if family != -1 {
        state.family = normalize_family(family);
    }
    if kind != -1 {
        state.kind = normalize_socket_type(kind);
    }
    if proto != -1 {
        state.proto = normalize_proto(proto);
    }
    if state.timeout.is_none() {
        state.timeout = default_timeout;
    }
    SocketSnapshot {
        fd: fileno,
        family: state.family,
        kind: state.kind,
        proto: state.proto,
        timeout: state.timeout,
    }
}

fn write_socket_snapshot(object: &mut ShapedObject, snapshot: SocketSnapshot) {
    let registry = shape_registry();
    object.set_property(
        intern("family"),
        Value::int(snapshot.family).unwrap(),
        registry,
    );
    object.set_property(intern("type"), Value::int(snapshot.kind).unwrap(), registry);
    object.set_property(
        intern("proto"),
        Value::int(snapshot.proto).unwrap(),
        registry,
    );
    object.set_property(
        intern("__prism_socket_fd__"),
        Value::int(snapshot.fd).unwrap(),
        registry,
    );
    object.set_property(
        intern("__prism_socket_closed__"),
        Value::bool(false),
        registry,
    );
    object.set_property(
        intern("__prism_socket_timeout__"),
        timeout_option_value(snapshot.timeout),
        registry,
    );
}

fn socket_member_get(args: &[Value], name: &str) -> Result<Value, BuiltinError> {
    exact_min_args(args, name, 1)?;
    let object = shaped_socket_ref(args[0])?;
    if let Some(fd) = open_socket_fd(object)
        && let Some(state) = SOCKET_STATES.lock().unwrap().get(&fd)
    {
        let value = match name {
            "family" => state.family,
            "type" => state.kind,
            "proto" => state.proto,
            _ => unreachable!("socket member getter is registered for known names"),
        };
        return Ok(Value::int(value).unwrap());
    }
    object
        .get_property(name)
        .ok_or_else(|| BuiltinError::AttributeError(format!("socket has no attribute '{name}'")))
}

fn socket_family(object: &ShapedObject) -> i64 {
    object
        .get_property("family")
        .and_then(|value| value.as_int())
        .unwrap_or(AF_INET_VALUE)
}

fn socket_timeout(object: &ShapedObject) -> Option<f64> {
    object
        .get_property("__prism_socket_timeout__")
        .and_then(|value| {
            if value.is_none() {
                Some(None)
            } else if let Some(float) = value.as_float() {
                Some(Some(float))
            } else {
                value.as_int().map(|integer| Some(integer as f64))
            }
        })
        .unwrap_or_else(|| *DEFAULT_TIMEOUT.lock().unwrap())
}

fn open_socket_fd(object: &ShapedObject) -> Option<i64> {
    let fd = socket_fd(object);
    (fd >= 0
        && object
            .get_property("__prism_socket_closed__")
            .and_then(|value| value.as_bool())
            != Some(true))
    .then_some(fd)
}

fn require_open_socket_fd(object: &ShapedObject) -> Result<i64, BuiltinError> {
    open_socket_fd(object).ok_or_else(|| BuiltinError::OSError("Bad file descriptor".to_string()))
}

fn normalize_family(family: i64) -> i64 {
    match family {
        -1 | 0 => AF_INET_VALUE,
        value => value,
    }
}

fn normalize_socket_type(kind: i64) -> i64 {
    match kind {
        -1 | 0 => SOCK_STREAM_VALUE,
        value => value,
    }
}

fn normalize_proto(proto: i64) -> i64 {
    match proto {
        -1 => 0,
        value => value,
    }
}

fn timeout_duration(timeout: Option<f64>) -> Option<Duration> {
    let timeout = timeout?;
    if timeout <= 0.0 {
        return Some(Duration::ZERO);
    }
    Some(Duration::from_secs_f64(timeout))
}

fn socket_io_error(err: std::io::Error) -> BuiltinError {
    BuiltinError::OSError(err.to_string())
}

fn connect_tcp_stream(addr: SocketAddr, timeout: Option<f64>) -> Result<TcpStream, BuiltinError> {
    match timeout_duration(timeout) {
        Some(duration) if !duration.is_zero() => {
            TcpStream::connect_timeout(&addr, duration).map_err(socket_io_error)
        }
        _ => TcpStream::connect(addr).map_err(socket_io_error),
    }
}

fn apply_stream_options(stream: &TcpStream, state: &SocketState) -> Result<(), BuiltinError> {
    if let Some(value) = state.options.get(&(6, 1)) {
        stream.set_nodelay(*value != 0).map_err(socket_io_error)?;
    }
    stream
        .set_nonblocking(state.timeout == Some(0.0))
        .map_err(socket_io_error)?;
    let duration = timeout_duration(state.timeout).filter(|duration| !duration.is_zero());
    stream.set_read_timeout(duration).map_err(socket_io_error)?;
    stream.set_write_timeout(duration).map_err(socket_io_error)
}

fn default_bind_addr(family: i64) -> SocketAddr {
    if family == AF_INET6_VALUE {
        SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 0)
    } else {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)
    }
}

fn default_socket_addr(family: i64) -> SocketAddr {
    if family == AF_INET6_VALUE {
        SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0)
    } else {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0)
    }
}

fn sockaddr_arg(
    vm: &mut VirtualMachine,
    value: Value,
    family: i64,
) -> Result<SocketAddr, BuiltinError> {
    let tuple = value_as_tuple_ref(value)
        .ok_or_else(|| BuiltinError::TypeError("socket address must be a tuple".to_string()))?;
    if tuple.len() < 2 {
        return Err(BuiltinError::TypeError(
            "socket address must contain host and port".to_string(),
        ));
    }
    let host_value = tuple.get(0).expect("length checked above");
    let port_value = tuple.get(1).expect("length checked above");
    let host = string_arg(host_value, "host")?;
    let port = u16::try_from(port_number(vm, port_value)?)
        .map_err(|_| BuiltinError::OverflowError("port must be 0-65535".to_string()))?;
    let ip = if family == AF_INET6_VALUE {
        IpAddr::V6(
            resolve_host_to_ipv6(&host)
                .parse::<Ipv6Addr>()
                .map_err(|_| {
                    BuiltinError::OSError("address family not supported by protocol".to_string())
                })?,
        )
    } else {
        IpAddr::V4(
            resolve_host_to_ipv4(&host)
                .parse::<Ipv4Addr>()
                .map_err(|_| {
                    BuiltinError::OSError("address family not supported by protocol".to_string())
                })?,
        )
    };
    Ok(SocketAddr::new(ip, port))
}

fn sockaddr_value(addr: SocketAddr, family: i64) -> Value {
    if family == AF_INET6_VALUE || addr.is_ipv6() {
        tuple_value(vec![
            Value::string(intern(&addr.ip().to_string())),
            Value::int(i64::from(addr.port())).unwrap(),
            Value::int(0).unwrap(),
            Value::int(0).unwrap(),
        ])
    } else {
        tuple_value(vec![
            Value::string(intern(&addr.ip().to_string())),
            Value::int(i64::from(addr.port())).unwrap(),
        ])
    }
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

fn int_arg_vm(vm: &mut VirtualMachine, value: Value, name: &str) -> Result<i64, BuiltinError> {
    if let Some(integer) = int_value_to_i64(value)? {
        return Ok(integer);
    }

    let target = match resolve_special_method(value, "__index__") {
        Ok(target) => target,
        Err(err) if err.is_attribute_error() => {
            return Err(BuiltinError::TypeError(format!(
                "{name} must be an integer"
            )));
        }
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };

    let indexed = invoke_bound_method_no_args(vm, target)?;
    match int_value_to_i64(indexed)? {
        Some(integer) => Ok(integer),
        None => Err(BuiltinError::TypeError(format!(
            "__index__ returned non-int (type {})",
            indexed.type_name()
        ))),
    }
}

fn optional_i64_arg_vm(
    vm: &mut VirtualMachine,
    value: Option<Value>,
    default: i64,
    name: &str,
) -> Result<i64, BuiltinError> {
    value.map_or(Ok(default), |value| int_arg_vm(vm, value, name))
}

fn int_value_to_i64(value: Value) -> Result<Option<i64>, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(Some(i64::from(flag)));
    }

    let Some(integer) = value_to_bigint(value) else {
        return Ok(None);
    };

    integer
        .to_i64()
        .map(Some)
        .ok_or_else(|| BuiltinError::OverflowError("Python int too large to convert".to_string()))
}

#[inline]
fn invoke_bound_method_no_args(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
) -> Result<Value, BuiltinError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
    .map_err(runtime_error_to_builtin_error)
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

fn port_number(vm: &mut VirtualMachine, value: Value) -> Result<i64, BuiltinError> {
    if value.is_none() {
        return Ok(0);
    }
    if let Some(port) = int_value_to_i64(value)? {
        return Ok(port);
    }
    if let Some(service) = value_as_string_ref(value) {
        return Ok(service_port(service.as_str()).unwrap_or(0));
    }
    int_arg_vm(vm, value, "port")
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
    vm: &mut VirtualMachine,
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
    let value = u16::try_from(int_arg_vm(vm, args[0], "integer")?)
        .map_err(|_| BuiltinError::OverflowError("unsigned short is out of range".to_string()))?;
    Ok(Value::int(i64::from(convert(value))).unwrap())
}

fn convert_u32(
    vm: &mut VirtualMachine,
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
    let value = u32::try_from(int_arg_vm(vm, args[0], "integer")?)
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
