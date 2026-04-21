use std::path::PathBuf;
use std::process::Command;

fn cpython_root() -> PathBuf {
    std::env::var_os("PRISM_CPYTHON_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\James\Desktop\cpython-3.12"))
}

fn prism_test_exe() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_prism"));
    path.set_file_name(format!("prism-test{}", std::env::consts::EXE_SUFFIX));
    path
}

fn run_suite(test_name: &str) {
    let output = Command::new(prism_test_exe())
        .arg("--cpython-root")
        .arg(cpython_root())
        .arg("--runner")
        .arg("suite")
        .arg("--timeout")
        .arg("90")
        .arg(test_name)
        .output()
        .expect("failed to execute prism-test binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "CPython suite {test_name} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
}

#[test]
fn test_cpython_test_bool_suite_passes_in_prism_test() {
    run_suite("test_bool");
}

#[test]
fn test_cpython_test_keyword_suite_passes_in_prism_test() {
    run_suite("test_keyword");
}
