use std::process::{Command, Output};

fn run_command(script: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_prism"))
        .arg("-X")
        .arg("nojit")
        .arg("-B")
        .arg("-c")
        .arg(script)
        .output()
        .expect("failed to execute prism binary")
}

fn assert_success(script: &str, expected_stdout: &str) {
    let output = run_command(script);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "command mode failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert_eq!(stdout, expected_stdout);
}

#[test]
fn test_yield_from_delegates_generator_values() {
    assert_success(
        r#"
def inner():
    yield 1
    yield 2

def outer():
    yield from inner()

print(list(outer()))
"#,
        "[1, 2]\n",
    );
}

#[test]
fn test_yield_from_delegates_plain_iterables() {
    assert_success(
        r#"
def outer():
    yield from [1, 2, 3]

print(list(outer()))
"#,
        "[1, 2, 3]\n",
    );
}

#[test]
fn test_yield_from_propagates_stop_iteration_value() {
    assert_success(
        r#"
def inner():
    yield 1
    return 99

def outer():
    result = yield from inner()
    print(result)

gen = outer()
print(next(gen))
try:
    next(gen)
except StopIteration:
    pass
"#,
        "1\n99\n",
    );
}
