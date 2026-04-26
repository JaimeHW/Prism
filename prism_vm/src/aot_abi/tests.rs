use super::*;
use crate::import::FrozenModuleSource;
use prism_compiler::{Compiler, OptimizationLevel};
use prism_parser::parse;

fn compile_module(source: &str, filename: &str) -> Arc<prism_code::CodeObject> {
    let parsed = parse(source).expect("source should parse");
    Arc::new(
        Compiler::compile_module_with_optimization(&parsed, filename, OptimizationLevel::Basic)
            .expect("source should compile"),
    )
}

fn string_ref(bytes: &'static [u8]) -> AotStringRef {
    AotStringRef {
        data: bytes.as_ptr(),
        len: bytes.len(),
    }
}

#[test]
fn test_store_expr_op_writes_immediate_and_add_result() {
    let mut vm = VirtualMachine::new();
    let module = Arc::new(ModuleObject::with_metadata(
        "__main__",
        None,
        None,
        Some("".into()),
    ));
    vm.bind_module(Arc::clone(&module));

    let store_value = AotStoreExprOp {
        target: string_ref(b"VALUE"),
        kind: AotStoreExprKind::Operand,
        reserved: [0; 7],
        left: AotOperand::immediate(AotImmediate::value_bits(
            Value::int(37).expect("small int").to_bits(),
        )),
        right: AotOperand::immediate(AotImmediate::value_bits(0)),
    };
    assert_eq!(
        prism_aot_op_store_expr(&mut vm, Arc::as_ptr(&module), &store_value),
        AotOpStatus::Ok
    );

    let store_result = AotStoreExprOp {
        target: string_ref(b"RESULT"),
        kind: AotStoreExprKind::Add,
        reserved: [0; 7],
        left: AotOperand::name(string_ref(b"VALUE")),
        right: AotOperand::immediate(AotImmediate::value_bits(
            Value::int(5).expect("small int").to_bits(),
        )),
    };
    assert_eq!(
        prism_aot_op_store_expr(&mut vm, Arc::as_ptr(&module), &store_result),
        AotOpStatus::Ok
    );

    assert_eq!(
        module.get_attr("VALUE").and_then(|value| value.as_int()),
        Some(37)
    );
    assert_eq!(
        module.get_attr("RESULT").and_then(|value| value.as_int()),
        Some(42)
    );
    assert!(vm.take_last_aot_error().is_none());
}

#[test]
fn test_import_module_op_binds_top_level_package_for_dotted_imports() {
    let mut vm = VirtualMachine::new();
    let module = Arc::new(ModuleObject::with_metadata(
        "__main__",
        None,
        None,
        Some("".into()),
    ));
    vm.bind_module(Arc::clone(&module));

    let op = AotImportModuleOp {
        target: string_ref(b"OS"),
        module_spec: string_ref(b"os.path"),
        binding: AotImportBinding::TopLevel,
        reserved: [0; 7],
    };

    assert_eq!(
        prism_aot_op_import_module(&mut vm, Arc::as_ptr(&module), &op),
        AotOpStatus::Ok
    );

    let os_value = module
        .get_attr("OS")
        .and_then(|value| value.as_object_ptr())
        .expect("imported module should be stored as a module object");
    let os_module = vm
        .import_resolver
        .module_from_ptr(os_value)
        .expect("stored module pointer should resolve");
    assert_eq!(os_module.name(), "os");
}

#[test]
fn test_import_from_op_supports_relative_imports() {
    let mut vm = VirtualMachine::new();
    vm.import_resolver.insert_frozen_module(
        "pkg",
        FrozenModuleSource::new(
            compile_module("PACKAGE = True\n", "<frozen:pkg.__init__>"),
            "<frozen:pkg.__init__>",
            "pkg",
            true,
        ),
    );
    vm.import_resolver.insert_frozen_module(
        "pkg.helper",
        FrozenModuleSource::new(
            compile_module("VALUE = 42\n", "<frozen:pkg.helper>"),
            "<frozen:pkg.helper>",
            "pkg",
            false,
        ),
    );

    let module = Arc::new(ModuleObject::with_metadata(
        "__main__",
        None,
        Some("<frozen:pkg.__main__>".into()),
        Some("pkg".into()),
    ));
    vm.bind_module(Arc::clone(&module));

    let op = AotImportFromOp {
        target: string_ref(b"RESULT"),
        module_spec: string_ref(b".helper"),
        attribute: string_ref(b"VALUE"),
    };

    assert_eq!(
        prism_aot_op_import_from(&mut vm, Arc::as_ptr(&module), &op),
        AotOpStatus::Ok
    );
    assert_eq!(
        module.get_attr("RESULT").and_then(|value| value.as_int()),
        Some(42)
    );
}

#[test]
fn test_store_expr_op_records_runtime_error() {
    let mut vm = VirtualMachine::new();
    let module = Arc::new(ModuleObject::with_metadata(
        "__main__",
        None,
        None,
        Some("".into()),
    ));
    vm.bind_module(Arc::clone(&module));

    let op = AotStoreExprOp {
        target: string_ref(b"RESULT"),
        kind: AotStoreExprKind::Operand,
        reserved: [0; 7],
        left: AotOperand::name(string_ref(b"MISSING")),
        right: AotOperand::immediate(AotImmediate::value_bits(0)),
    };

    assert_eq!(
        prism_aot_op_store_expr(&mut vm, Arc::as_ptr(&module), &op),
        AotOpStatus::Error
    );

    let err = vm
        .take_last_aot_error()
        .expect("failing AOT helper should record an error");
    assert!(matches!(
        err.kind,
        crate::error::RuntimeErrorKind::NameError { .. }
    ));
}
