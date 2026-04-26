use super::*;

#[test]
fn test_compile_source_module_returns_ast_and_code() {
    let compilation = compile_source_module("value = 42\n", "<test>", OptimizationLevel::Basic)
        .expect("source compilation should succeed");

    assert_eq!(compilation.module.body.len(), 1);
    assert!(!compilation.code.instructions.is_empty());
}

#[test]
fn test_compile_source_code_returns_parse_error_variant() {
    let err = compile_source_code("def\n", "<test>", OptimizationLevel::None)
        .expect_err("invalid syntax should fail parsing");

    assert!(err.as_parse_error().is_some());
    assert!(err.as_compile_error().is_none());
}

#[test]
fn test_compile_source_code_returns_compile_error_variant() {
    let err = compile_source_code("continue\n", "<test>", OptimizationLevel::None)
        .expect_err("invalid control flow should fail compilation");

    assert!(err.as_parse_error().is_none());
    let compile_error = err
        .as_compile_error()
        .expect("expected compilation error variant");
    assert!(compile_error.message.contains("continue"));
}

#[test]
fn test_compile_source_code_with_namespace_mode_emits_bytecode() {
    let code = compile_source_code_with_namespace_mode(
        "x = 1\n",
        "<test>",
        OptimizationLevel::Basic,
        ModuleNamespaceMode::DynamicLocals,
    )
    .expect("dynamic locals compilation should succeed");

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_compile_source_code_maps_byte_offsets_to_source_lines() {
    let source = "\n\nclass C:\n    def run(self):\n        raise ValueError('x')\n";
    let code = compile_source_code(source, "lineprobe.py", OptimizationLevel::None)
        .expect("source compilation should succeed");

    let class_code = code
        .nested_code_objects
        .first()
        .expect("class body code object should be emitted");
    let function_code = class_code
        .nested_code_objects
        .first()
        .expect("method code object should be emitted");

    assert_eq!(class_code.first_lineno, 3);
    assert_eq!(function_code.first_lineno, 4);
    assert!(
        function_code.line_table.iter().any(|entry| entry.line == 5),
        "method line table should point at the raise statement: {:?}",
        function_code.line_table
    );
    assert!(
        function_code
            .line_table
            .iter()
            .all(|entry| entry.line < 100),
        "byte offsets must not leak into line-table entries: {:?}",
        function_code.line_table
    );
}
