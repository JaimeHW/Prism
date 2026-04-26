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

fn prism_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_prism"))
}

fn cpython_lib() -> PathBuf {
    cpython_root().join("Lib")
}

fn run_suite(test_name: &str) {
    run_suite_with_timeout(test_name, "90");
}

fn run_suite_with_timeout(test_name: &str, timeout: &str) {
    let output = Command::new(prism_test_exe())
        .arg("--cpython-root")
        .arg(cpython_root())
        .arg("--runner")
        .arg("suite")
        .arg("--timeout")
        .arg(timeout)
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

#[test]
fn test_cpython_test_base64_suite_passes_threaded_subprocess_paths() {
    run_suite_with_timeout("test_base64", "120");
}

#[test]
fn test_cpython_threading_atexit_callbacks_run_at_prism_shutdown() {
    let output = Command::new(prism_exe())
        .env("PYTHONPATH", cpython_lib())
        .arg("-X")
        .arg("nojit")
        .arg("-c")
        .arg("import threading\ndef run_last():\n    print('parrot')\nthreading._register_atexit(run_last)\n")
        .output()
        .expect("failed to execute prism binary");

    assert!(
        output.status.success(),
        "threading atexit callback failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "parrot\n");
    assert!(
        output.stderr.is_empty(),
        "successful threading atexit callback should be silent on stderr, got:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cpython_atexit_callbacks_run_with_clean_exception_context_after_sys_exit() {
    let output = Command::new(prism_exe())
        .env("PYTHONPATH", cpython_lib())
        .arg("-X")
        .arg("nojit")
        .arg("-c")
        .arg("import logging, sys\nsys.exit(0)\n")
        .output()
        .expect("failed to execute prism binary");

    assert_eq!(output.status.code(), Some(0));
    assert!(
        output.stderr.is_empty(),
        "sys.exit(0) should not leak SystemExit into atexit callbacks, got:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_cpython_threading_rejects_atexit_registration_during_prism_shutdown() {
    let output = Command::new(prism_exe())
        .env("PYTHONPATH", cpython_lib())
        .arg("-X")
        .arg("nojit")
        .arg("-c")
        .arg("import threading\ndef func():\n    pass\ndef run_last():\n    threading._register_atexit(func)\nthreading._register_atexit(run_last)\n")
        .output()
        .expect("failed to execute prism binary");

    assert!(
        output.status.success(),
        "threading shutdown errors should not change process status\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("RuntimeError: can't register atexit after shutdown"),
        "stderr should report late threading atexit registration, got:\n{stderr}"
    );
}
