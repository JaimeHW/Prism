use std::process::Command;

#[test]
fn test_command_mode_accepts_semicolon_separated_simple_statements() {
    let output = Command::new(env!("CARGO_BIN_EXE_prism"))
        .arg("-X")
        .arg("nojit")
        .arg("-B")
        .arg("-c")
        .arg("import sys; value = 40; print(value + 2); print(len(sys.argv))")
        .output()
        .expect("failed to execute prism binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "command mode failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert_eq!(stdout, "42\n1\n");
}

#[test]
fn test_command_mode_exposes_doc_binding_for_multiline_c() {
    let output = Command::new(env!("CARGO_BIN_EXE_prism"))
        .arg("-X")
        .arg("nojit")
        .arg("-B")
        .arg("-c")
        .arg("print(__doc__)\nvalue = 40\nprint(value + 2)")
        .output()
        .expect("failed to execute prism binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "command mode failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert_eq!(stdout, "None\n42\n");
}
