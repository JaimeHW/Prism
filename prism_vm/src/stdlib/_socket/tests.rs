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
    let setdefault = builtin_from_value(SocketModule::new().get_attr("setdefaulttimeout").unwrap());
    let getdefault = builtin_from_value(SocketModule::new().get_attr("getdefaulttimeout").unwrap());

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
