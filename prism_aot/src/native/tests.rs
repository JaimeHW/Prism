use super::*;
use prism_parser::parse;

fn parse_module(source: &str) -> Module {
    parse(source).expect("source should parse")
}

#[test]
fn test_lower_module_supports_docstring_imports_and_assignments() {
    let module = parse_module(
        "\"doc\"\nimport pkg.helper\nfrom .tools import VALUE\nRESULT = VALUE + 5\nNAME = 'prism'\n",
    );

    let plan = NativeModuleInitPlan::lower("pkg.__main__", &module)
        .expect("module should lower successfully");

    assert_eq!(plan.symbol_name, native_init_symbol("pkg.__main__"));
    assert_eq!(plan.operations.len(), 5);
    assert!(matches!(
        &plan.operations[0],
        NativeInitOperation::StoreExpr { target, .. } if target == "__doc__"
    ));
    assert!(matches!(
        &plan.operations[1],
        NativeInitOperation::ImportModule { target, binding, .. }
            if target == "pkg" && *binding == AotImportBinding::TopLevel
    ));
    assert!(matches!(
        &plan.operations[2],
        NativeInitOperation::ImportFrom { target, module_spec, attribute }
            if target == "VALUE" && module_spec == ".tools" && attribute == "VALUE"
    ));
    assert!(matches!(
        &plan.operations[3],
        NativeInitOperation::StoreExpr {
            target,
            expr: NativeExpr::Add { .. }
        } if target == "RESULT"
    ));
}

#[test]
fn test_lower_module_supports_negative_numeric_literals() {
    let module = parse_module("NEG = -5\nFLOAT = -3.5\n");
    let plan =
        NativeModuleInitPlan::lower("__main__", &module).expect("negative literals should lower");

    assert_eq!(plan.operations.len(), 2);
    assert!(matches!(
        &plan.operations[0],
        NativeInitOperation::StoreExpr {
            expr: NativeExpr::Operand(NativeOperand::Immediate(NativeImmediate::ValueBits(_))),
            ..
        }
    ));
}

#[test]
fn test_lower_module_rejects_unsupported_statements() {
    let module = parse_module("def helper():\n    return 1\n");
    let err = NativeModuleInitPlan::lower("__main__", &module)
        .expect_err("function definitions should not lower yet");

    assert!(err.to_string().contains("cannot lower statement"));
}

#[test]
fn test_native_init_symbol_is_stable_hex_encoding() {
    assert_eq!(
        native_init_symbol("pkg.__main__"),
        "prism_aot_init_706b672e5f5f6d61696e5f5f"
    );
}
