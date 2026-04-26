use super::*;
use crate::VirtualMachine;
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::{get_attribute_value, set_attribute_value};

#[test]
fn test_ctypes_exposes_pythonapi_thread_symbols() {
    let module = CtypesModule::new();
    let pythonapi = module
        .get_attr("pythonapi")
        .expect("pythonapi should be exposed");
    let mut vm = VirtualMachine::new();

    for name in [
        "PyThreadState_SetAsyncExc",
        "PyGILState_Ensure",
        "PyGILState_Release",
    ] {
        let symbol = get_attribute_value(&mut vm, pythonapi, &intern(name))
            .expect("pythonapi symbol should resolve");
        assert!(symbol.as_object_ptr().is_some());
    }
}

#[test]
fn test_pythonapi_symbols_allow_ctypes_metadata_assignment() {
    let module = CtypesModule::new();
    let pythonapi = module.get_attr("pythonapi").unwrap();
    let mut vm = VirtualMachine::new();
    let symbol =
        get_attribute_value(&mut vm, pythonapi, &intern("PyThreadState_SetAsyncExc")).unwrap();

    set_attribute_value(&mut vm, symbol, &intern("argtypes"), Value::none())
        .expect("ctypes function metadata should be writable");
}

#[test]
fn test_py_thread_state_set_async_exc_reports_unknown_thread() {
    let module = CtypesModule::new();
    let pythonapi = module.get_attr("pythonapi").unwrap();
    let mut vm = VirtualMachine::new();
    let symbol =
        get_attribute_value(&mut vm, pythonapi, &intern("PyThreadState_SetAsyncExc")).unwrap();

    let result = invoke_callable_value(
        &mut vm,
        symbol,
        &[
            Value::int(1_000_000_000).unwrap(),
            Value::string(intern("AsyncExc")),
        ],
    )
    .expect("unknown thread id should be a successful no-op");

    assert_eq!(result.as_int(), Some(0));
}
