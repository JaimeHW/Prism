use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

fn cpython_root() -> PathBuf {
    std::env::var_os("PRISM_CPYTHON_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\James\Desktop\cpython-3.12"))
}

fn cpython_lib_dir() -> PathBuf {
    let lib_dir = cpython_root().join("Lib");
    assert!(
        lib_dir.is_dir(),
        "CPython Lib directory not found at {}. Set PRISM_CPYTHON_ROOT to override.",
        lib_dir.display()
    );
    lib_dir
}

fn run_fixture(script_path: &Path) {
    let lib_dir = cpython_lib_dir();
    let output = Command::new(env!("CARGO_BIN_EXE_prism"))
        .arg("-X")
        .arg("nojit")
        .arg("-B")
        .arg(script_path)
        .env("PYTHONPATH", OsString::from(lib_dir.as_os_str()))
        .output()
        .expect("failed to execute prism binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "fixture {} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        script_path.display(),
        output.status.code(),
        stdout,
        stderr
    );
}

#[test]
fn test_cpython_future_test1_fixture_passes_in_prism_cli() {
    run_fixture(
        &cpython_lib_dir()
            .join("test")
            .join("test_future_stmt")
            .join("future_test1.py"),
    );
}

#[test]
fn test_cpython_future_test2_fixture_passes_in_prism_cli() {
    run_fixture(
        &cpython_lib_dir()
            .join("test")
            .join("test_future_stmt")
            .join("future_test2.py"),
    );
}
