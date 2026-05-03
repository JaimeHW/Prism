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
fn except_handler_reraise_runs_enclosing_finally() {
    let opcodes = compiled_opcodes(
        r#"
try:
    raise RuntimeError()
except RuntimeError:
    raise
finally:
    cleanup = True
"#,
    );

    let abort_except = opcodes
        .iter()
        .position(|opcode| *opcode == Opcode::AbortExcept)
        .expect("except handler should have an abort path");

    assert_eq!(
        opcodes.get(abort_except + 1),
        Some(&Opcode::Jump),
        "exceptions escaping an except handler must enter the enclosing finally body: {opcodes:?}"
    );
    assert!(
        opcodes[abort_except + 1..].contains(&Opcode::EndFinally),
        "the abort path should route through compiled finally cleanup: {opcodes:?}"
    );
}
