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
fn compiles_single_starred_subscript_key_as_tuple_unpack() {
    let opcodes = compiled_opcodes(
        r#"
T = tuple
xs = ()
result = T[*xs]
"#,
    );

    assert!(
        opcodes
            .windows(2)
            .any(|window| matches!(window, [Opcode::BuildTupleUnpack, Opcode::CallKwEx])),
        "starred subscription should materialize an unpacked tuple key: {opcodes:?}"
    );
    assert!(opcodes.contains(&Opcode::GetItem));
}

#[test]
fn compiles_mixed_starred_subscript_key_as_tuple_unpack() {
    let opcodes = compiled_opcodes(
        r#"
T = tuple
xs = ()
result = T[int, *xs]
"#,
    );

    assert!(
        opcodes
            .windows(2)
            .any(|window| matches!(window, [Opcode::BuildTupleUnpack, Opcode::CallKwEx])),
        "mixed starred subscription should materialize an unpacked tuple key: {opcodes:?}"
    );
    assert!(opcodes.contains(&Opcode::GetItem));
}
