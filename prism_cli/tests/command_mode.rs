use std::io::Write;
use std::process::{Command, Stdio};

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

#[test]
fn test_command_mode_sys_exit_zero_exits_successfully_without_traceback() {
    let output = Command::new(env!("CARGO_BIN_EXE_prism"))
        .arg("-X")
        .arg("nojit")
        .arg("-c")
        .arg("import sys; sys.exit(0)")
        .output()
        .expect("failed to execute prism binary");

    assert_eq!(output.status.code(), Some(0));
    assert!(
        output.stderr.is_empty(),
        "sys.exit(0) should be silent, got stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_command_mode_sys_exit_string_prints_payload_and_fails() {
    let output = Command::new(env!("CARGO_BIN_EXE_prism"))
        .arg("-X")
        .arg("nojit")
        .arg("-c")
        .arg("import sys; sys.exit('bye')")
        .output()
        .expect("failed to execute prism binary");

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&output.stderr), "bye\n");
}

#[test]
fn test_command_mode_accepts_isolated_flag_and_preserves_module_execution() {
    let output = Command::new(env!("CARGO_BIN_EXE_prism"))
        .arg("-X")
        .arg("nojit")
        .arg("-I")
        .arg("-c")
        .arg("print('isolated-ok')")
        .output()
        .expect("failed to execute prism binary");

    assert!(
        output.status.success(),
        "isolated command mode failed with stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "isolated-ok\n");
}

#[test]
fn test_command_mode_print_uses_live_sys_stdout_binding() {
    let output = Command::new(env!("CARGO_BIN_EXE_prism"))
        .arg("-X")
        .arg("nojit")
        .arg("-B")
        .arg("-c")
        .arg("import sys; sys.stdout = sys.stderr; print('usage: redirected')")
        .output()
        .expect("failed to execute prism binary");

    assert!(
        output.status.success(),
        "command mode failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "redirected print should not write stdout, got:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        "usage: redirected\n"
    );
}

#[test]
fn test_command_mode_standard_stream_buffers_read_and_write_bytes() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_prism"))
        .arg("-X")
        .arg("nojit")
        .arg("-c")
        .arg(
            "import sys\n".to_owned()
                + "data = sys.stdin.buffer.read()\n"
                + "sys.stdout.buffer.write(data.upper())\n",
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to execute prism binary");

    child
        .stdin
        .as_mut()
        .expect("stdin should be piped")
        .write_all(b"abc\n")
        .expect("failed to write test stdin");
    drop(child.stdin.take());

    let output = child
        .wait_with_output()
        .expect("failed to wait for prism binary");
    assert!(
        output.status.success(),
        "buffer command mode failed with stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"ABC\n");
}
