use prism_compiler::{Opcode, OptimizationLevel, compile_source_code};

fn compiled_opcodes(source: &str) -> Vec<Opcode> {
    compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile")
        .instructions
        .iter()
        .map(|instruction| Opcode::from_u8(instruction.opcode()).expect("known opcode"))
        .collect()
}

#[test]
fn compiles_except_star_handlers() {
    let opcodes = compiled_opcodes(
        r#"
try:
    1 / 0
except* ZeroDivisionError as err:
    handled = True
else:
    handled = False
"#,
    );

    assert!(
        opcodes.contains(&Opcode::ExceptionMatch),
        "except* should compile through the exception matcher: {opcodes:?}"
    );
}

#[test]
fn rejects_bare_except_star() {
    let err = compile_source_code(
        r#"
try:
    pass
except*:
    pass
"#,
        "<test>",
        OptimizationLevel::None,
    )
    .expect_err("bare except* is invalid Python syntax");

    assert!(
        err.to_string()
            .contains("expected expression after 'except*'")
    );
}

#[test]
fn rejects_mixed_except_and_except_star() {
    let err = compile_source_code(
        r#"
try:
    pass
except ValueError:
    pass
except* TypeError:
    pass
"#,
        "<test>",
        OptimizationLevel::None,
    )
    .expect_err("regular except and except* cannot share one try statement");

    assert!(
        err.to_string()
            .contains("cannot have both 'except' and 'except*' on the same 'try'")
    );
}
