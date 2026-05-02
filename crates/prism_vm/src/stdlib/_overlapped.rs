//! Native Windows `_overlapped` bootstrap surface.
//!
//! CPython's `asyncio.windows_events` imports `_overlapped` for IOCP and
//! overlapped I/O primitives. Prism does not yet provide a real Windows IOCP
//! backend, but the stdlib import path needs the same module shape and a
//! coherent inert Overlapped object so higher-level code can import and inspect
//! the Windows event-loop implementation.

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
use prism_runtime::types::tuple::TupleObject;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

const INVALID_HANDLE_VALUE: i64 = -1;
const ERROR_NETNAME_DELETED: i64 = 64;
const ERROR_PIPE_BUSY: i64 = 231;
const ERROR_OPERATION_ABORTED: i64 = 995;
const ERROR_IO_PENDING: i64 = 997;
const ERROR_PORT_UNREACHABLE: i64 = 1234;
const SO_UPDATE_ACCEPT_CONTEXT: i64 = 0x700B;
const SO_UPDATE_CONNECT_CONTEXT: i64 = 0x7010;

const EXPORTED_CONSTANTS: &[(&str, i64)] = &[
    ("ERROR_IO_PENDING", ERROR_IO_PENDING),
    ("ERROR_NETNAME_DELETED", ERROR_NETNAME_DELETED),
    ("ERROR_OPERATION_ABORTED", ERROR_OPERATION_ABORTED),
    ("ERROR_PIPE_BUSY", ERROR_PIPE_BUSY),
    ("ERROR_PORT_UNREACHABLE", ERROR_PORT_UNREACHABLE),
    ("INVALID_HANDLE_VALUE", INVALID_HANDLE_VALUE),
    ("SO_UPDATE_ACCEPT_CONTEXT", SO_UPDATE_ACCEPT_CONTEXT),
    ("SO_UPDATE_CONNECT_CONTEXT", SO_UPDATE_CONNECT_CONTEXT),
];

static NEXT_FAKE_HANDLE: AtomicI64 = AtomicI64::new(0x1000);
static NEXT_OVERLAPPED_ADDRESS: AtomicI64 = AtomicI64::new(0x1_0000_0000);
static IOCP_REGISTRY: LazyLock<Mutex<IocpRegistry>> =
    LazyLock::new(|| Mutex::new(IocpRegistry::default()));

#[derive(Clone, Copy)]
struct IocpRegistration {
    port: i64,
    key: i64,
}

#[derive(Clone, Copy)]
struct CompletionStatus {
    error: i64,
    transferred: i64,
    key: i64,
    address: i64,
}

#[derive(Default)]
struct IocpRegistry {
    ports: HashMap<i64, VecDeque<CompletionStatus>>,
    registrations: HashMap<i64, IocpRegistration>,
    owned_handles: HashSet<i64>,
}

impl IocpRegistry {
    fn allocate_handle(&mut self) -> i64 {
        let handle = NEXT_FAKE_HANDLE.fetch_add(1, Ordering::Relaxed);
        self.owned_handles.insert(handle);
        handle
    }

    fn ensure_port(&mut self, port: i64) {
        self.ports.entry(port).or_default();
    }

    fn register_handle(&mut self, handle: i64, port: i64, key: i64) {
        if handle == 0 || handle == INVALID_HANDLE_VALUE {
            return;
        }
        self.ensure_port(port);
        self.registrations
            .insert(handle, IocpRegistration { port, key });
    }

    fn queue_for_handle(&mut self, handle: i64, transferred: i64, address: i64) {
        let Some(registration) = self.registrations.get(&handle).copied() else {
            return;
        };
        self.push(
            registration.port,
            CompletionStatus {
                error: 0,
                transferred,
                key: registration.key,
                address,
            },
        );
    }

    fn push(&mut self, port: i64, status: CompletionStatus) {
        self.ports.entry(port).or_default().push_back(status);
    }

    fn pop(&mut self, port: i64) -> Option<CompletionStatus> {
        self.ports.get_mut(&port)?.pop_front()
    }

    fn close_owned_handle(&mut self, handle: i64) -> bool {
        if !self.owned_handles.remove(&handle) {
            return false;
        }
        self.ports.remove(&handle);
        self.registrations
            .retain(|_, registration| registration.port != handle);
        true
    }
}

static OVERLAPPED_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_overlapped_class("Overlapped"));

static OVERLAPPED_NEW_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_overlapped.Overlapped.__new__"), overlapped_new)
});
static OVERLAPPED_CANCEL_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_overlapped.Overlapped.cancel"),
        overlapped_complete,
    )
});
static OVERLAPPED_GETRESULT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_overlapped.Overlapped.getresult"),
        overlapped_getresult,
    )
});
static OVERLAPPED_CONNECT_NAMED_PIPE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("_overlapped.Overlapped.ConnectNamedPipe"),
            overlapped_bool_operation,
        )
    });
static OVERLAPPED_OPERATION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_overlapped.Overlapped.operation"),
        overlapped_io_operation,
    )
});

static CREATE_EVENT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_overlapped.CreateEvent"), create_event)
});
static CREATE_IO_COMPLETION_PORT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_overlapped.CreateIoCompletionPort"),
        create_io_completion_port,
    )
});
static GET_QUEUED_COMPLETION_STATUS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new(
            Arc::from("_overlapped.GetQueuedCompletionStatus"),
            get_queued_completion_status,
        )
    });
static REGISTER_WAIT_WITH_QUEUE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_overlapped.RegisterWaitWithQueue"),
        register_wait_with_queue,
    )
});
static CONNECT_PIPE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_overlapped.ConnectPipe"), fake_handle_function)
});
static NOOP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_overlapped.noop"), noop_function));

/// Native `_overlapped` module descriptor.
#[derive(Debug, Clone)]
pub struct OverlappedModule {
    attrs: Vec<Arc<str>>,
}

pub(crate) fn close_overlapped_handle(handle: i64) -> bool {
    IOCP_REGISTRY
        .lock()
        .expect("_overlapped IOCP registry mutex poisoned")
        .close_owned_handle(handle)
}

impl OverlappedModule {
    /// Create a new `_overlapped` module descriptor.
    pub fn new() -> Self {
        let mut attrs: Vec<Arc<str>> = EXPORTED_CONSTANTS
            .iter()
            .map(|(name, _)| Arc::from(*name))
            .collect();
        attrs.extend(
            [
                "BindLocal",
                "ConnectPipe",
                "CreateEvent",
                "CreateIoCompletionPort",
                "GetQueuedCompletionStatus",
                "Overlapped",
                "RegisterWaitWithQueue",
                "UnregisterWait",
                "UnregisterWaitEx",
                "WSAConnect",
            ]
            .into_iter()
            .map(Arc::from),
        );
        attrs.sort_unstable();
        Self { attrs }
    }
}

impl Default for OverlappedModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for OverlappedModule {
    fn name(&self) -> &str {
        "_overlapped"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "Overlapped" => Ok(class_value(&OVERLAPPED_CLASS)),
            "BindLocal" | "UnregisterWait" | "UnregisterWaitEx" | "WSAConnect" => {
                Ok(builtin_value(&NOOP_FUNCTION))
            }
            "ConnectPipe" => Ok(builtin_value(&CONNECT_PIPE_FUNCTION)),
            "CreateEvent" => Ok(builtin_value(&CREATE_EVENT_FUNCTION)),
            "CreateIoCompletionPort" => Ok(builtin_value(&CREATE_IO_COMPLETION_PORT_FUNCTION)),
            "GetQueuedCompletionStatus" => {
                Ok(builtin_value(&GET_QUEUED_COMPLETION_STATUS_FUNCTION))
            }
            "RegisterWaitWithQueue" => Ok(builtin_value(&REGISTER_WAIT_WITH_QUEUE_FUNCTION)),
            _ => EXPORTED_CONSTANTS
                .iter()
                .find(|(constant, _)| *constant == name)
                .map(|(_, value)| Value::int(*value).expect("_overlapped constant should fit"))
                .ok_or_else(|| {
                    ModuleError::AttributeError(format!(
                        "module '_overlapped' has no attribute '{}'",
                        name
                    ))
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
fn class_value(class: &'static Arc<PyClassObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

#[inline]
fn leak_object_value<T: prism_runtime::Trace + 'static>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

fn build_overlapped_class(name: &str) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("_overlapped")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    class.set_attr(intern("__new__"), builtin_value(&OVERLAPPED_NEW_FUNCTION));
    class.set_attr(intern("cancel"), builtin_value(&OVERLAPPED_CANCEL_FUNCTION));
    class.set_attr(
        intern("getresult"),
        builtin_value(&OVERLAPPED_GETRESULT_FUNCTION),
    );
    class.set_attr(
        intern("ConnectNamedPipe"),
        builtin_value(&OVERLAPPED_CONNECT_NAMED_PIPE_FUNCTION),
    );
    for name in [
        "AcceptEx",
        "ConnectEx",
        "ReadFile",
        "ReadFileInto",
        "TransmitFile",
        "WSARecv",
        "WSARecvFrom",
        "WSARecvFromInto",
        "WSARecvInto",
        "WSASend",
        "WSASendTo",
        "WriteFile",
    ] {
        class.set_attr(intern(name), builtin_value(&OVERLAPPED_OPERATION_FUNCTION));
    }
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_NEW | ClassFlags::NATIVE_HEAPTYPE);
    register_native_type(class)
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

fn overlapped_new(args: &[Value]) -> Result<Value, BuiltinError> {
    let class = class_from_value(args.first().copied(), "Overlapped.__new__")?;
    let mut object = ShapedObject::new(class.class_type_id(), Arc::clone(class.instance_shape()));
    let registry = shape_registry();
    let address = NEXT_OVERLAPPED_ADDRESS.fetch_add(0x20, Ordering::Relaxed);
    object.set_property(intern("address"), int_value(address)?, registry);
    object.set_property(intern("pending"), Value::bool(false), registry);
    object.set_property(
        intern("__overlapped_result__"),
        Value::int(0).unwrap(),
        registry,
    );
    Ok(leak_object_value(object))
}

fn overlapped_complete(args: &[Value]) -> Result<Value, BuiltinError> {
    complete_overlapped(args.first().copied());
    Ok(Value::none())
}

fn overlapped_bool_operation(args: &[Value]) -> Result<Value, BuiltinError> {
    if let Some(address) = complete_overlapped(args.first().copied())
        && let Some(handle) = args.get(1).and_then(|value| value.as_int())
    {
        queue_completion_for_handle(handle, 0, address);
    }
    Ok(Value::bool(false))
}

fn overlapped_io_operation(args: &[Value]) -> Result<Value, BuiltinError> {
    if let Some(address) = complete_overlapped(args.first().copied())
        && let Some(handle) = args.get(1).and_then(|value| value.as_int())
    {
        queue_completion_for_handle(handle, 0, address);
    }
    Ok(Value::none())
}

fn overlapped_getresult(args: &[Value]) -> Result<Value, BuiltinError> {
    let Some(object) = args.first().and_then(|value| overlapped_object_ref(*value)) else {
        return Ok(Value::int(0).unwrap());
    };
    Ok(object
        .get_property("__overlapped_result__")
        .unwrap_or_else(|| Value::int(0).unwrap()))
}

fn create_event(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "CreateEvent() takes at most 4 arguments ({} given)",
            args.len()
        )));
    }
    fake_handle()
}

fn create_io_completion_port(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "CreateIoCompletionPort() takes exactly 4 arguments ({} given)",
            args.len()
        )));
    }
    let file_handle = integer_arg(args[0], "CreateIoCompletionPort", "file_handle")?;
    let existing_port = integer_arg(args[1], "CreateIoCompletionPort", "existing_port")?;
    let completion_key = integer_arg(args[2], "CreateIoCompletionPort", "completion_key")?;
    let port = if existing_port != 0 {
        existing_port
    } else {
        IOCP_REGISTRY
            .lock()
            .expect("_overlapped IOCP registry mutex poisoned")
            .allocate_handle()
    };

    IOCP_REGISTRY
        .lock()
        .expect("_overlapped IOCP registry mutex poisoned")
        .register_handle(file_handle, port, completion_key);
    int_value(port)
}

fn get_queued_completion_status(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "GetQueuedCompletionStatus() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    let port = integer_arg(args[0], "GetQueuedCompletionStatus", "completion_port")?;
    let Some(status) = IOCP_REGISTRY
        .lock()
        .expect("_overlapped IOCP registry mutex poisoned")
        .pop(port)
    else {
        return Ok(Value::none());
    };
    Ok(completion_status_value(status)?)
}

fn fake_handle_function(_args: &[Value]) -> Result<Value, BuiltinError> {
    fake_handle()
}

fn register_wait_with_queue(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "RegisterWaitWithQueue() takes exactly 4 arguments ({} given)",
            args.len()
        )));
    }

    let completion_port = integer_arg(args[1], "RegisterWaitWithQueue", "completion_port")?;
    let address = integer_arg(args[2], "RegisterWaitWithQueue", "overlapped_address")?;
    let mut registry = IOCP_REGISTRY
        .lock()
        .expect("_overlapped IOCP registry mutex poisoned");
    let wait_handle = registry.allocate_handle();
    registry.push(
        completion_port,
        CompletionStatus {
            error: 0,
            transferred: 0,
            key: 0,
            address,
        },
    );
    int_value(wait_handle)
}

fn noop_function(_args: &[Value]) -> Result<Value, BuiltinError> {
    Ok(Value::none())
}

fn fake_handle() -> Result<Value, BuiltinError> {
    let handle = IOCP_REGISTRY
        .lock()
        .expect("_overlapped IOCP registry mutex poisoned")
        .allocate_handle();
    int_value(handle)
}

fn int_value(value: i64) -> Result<Value, BuiltinError> {
    Value::int(value).ok_or_else(|| BuiltinError::OverflowError("integer overflow".to_string()))
}

fn integer_arg(value: Value, fn_name: &str, arg_name: &str) -> Result<i64, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(i64::from(flag));
    }
    value.as_int().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{fn_name}() {arg_name} must be an integer, not {}",
            value.type_name()
        ))
    })
}

fn completion_status_value(status: CompletionStatus) -> Result<Value, BuiltinError> {
    Ok(tuple_value(vec![
        int_value(status.error)?,
        int_value(status.transferred)?,
        int_value(status.key)?,
        int_value(status.address)?,
    ]))
}

fn tuple_value(items: Vec<Value>) -> Value {
    leak_object_value(TupleObject::from_vec(items))
}

fn complete_overlapped(value: Option<Value>) -> Option<i64> {
    let object = value.and_then(overlapped_object_mut)?;
    object.set_property(intern("pending"), Value::bool(false), shape_registry());
    object
        .get_property("address")
        .and_then(|value| value.as_int())
}

fn queue_completion_for_handle(handle: i64, transferred: i64, address: i64) {
    IOCP_REGISTRY
        .lock()
        .expect("_overlapped IOCP registry mutex poisoned")
        .queue_for_handle(handle, transferred, address);
}

fn class_from_value(
    value: Option<Value>,
    context: &'static str,
) -> Result<&'static PyClassObject, BuiltinError> {
    let Some(value) = value else {
        return Err(BuiltinError::TypeError(format!(
            "{context} missing required type argument"
        )));
    };
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} first argument must be a type"
        )));
    };
    Ok(unsafe { &*(ptr as *const PyClassObject) })
}

fn overlapped_object_ref(value: Value) -> Option<&'static ShapedObject> {
    let ptr = value.as_object_ptr()?;
    Some(unsafe { &*(ptr as *const ShapedObject) })
}

fn overlapped_object_mut(value: Value) -> Option<&'static mut ShapedObject> {
    let ptr = value.as_object_ptr()?;
    Some(unsafe { &mut *(ptr as *mut ShapedObject) })
}
