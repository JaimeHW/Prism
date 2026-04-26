use super::*;

#[test]
fn test_overlapped_module_exports_windows_asyncio_surface() {
    let module = OverlappedModule::new();
    for name in [
        "Overlapped",
        "CreateEvent",
        "CreateIoCompletionPort",
        "GetQueuedCompletionStatus",
        "RegisterWaitWithQueue",
        "ERROR_IO_PENDING",
        "INVALID_HANDLE_VALUE",
        "SO_UPDATE_ACCEPT_CONTEXT",
        "SO_UPDATE_CONNECT_CONTEXT",
    ] {
        assert!(module.get_attr(name).is_ok(), "{name} should be exported");
    }
}

#[test]
fn test_overlapped_new_exposes_pending_and_address() {
    let value = overlapped_new(&[class_value(&OVERLAPPED_CLASS), Value::int(0).unwrap()])
        .expect("Overlapped construction should succeed");
    let object = overlapped_object_ref(value).expect("Overlapped should be shaped object");

    assert_eq!(
        object
            .get_property("pending")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert!(
        object
            .get_property("address")
            .and_then(|value| value.as_int())
            .is_some()
    );
}

#[test]
fn test_completion_port_poll_returns_none_when_empty() {
    let port = create_io_completion_port(&[
        Value::int(INVALID_HANDLE_VALUE).unwrap(),
        Value::int(0).unwrap(),
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
    ])
    .expect("completion port should be created");
    assert!(port.as_int().is_some());

    let status = get_queued_completion_status(&[port, Value::int(0).unwrap()])
        .expect("empty completion port poll should succeed");
    assert!(status.is_none());
}
