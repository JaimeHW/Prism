use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::VirtualMachine;

fn execute(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");
    let mut vm = VirtualMachine::new();
    vm.execute_runtime(code).expect("source should execute");
}

fn python_path(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "\\\\")
}

#[test]
fn binary_file_readinto_updates_bytearray_and_position() {
    let path = std::env::temp_dir().join(format!(
        "prism-readinto-{}-{}.bin",
        std::process::id(),
        "bytearray"
    ));
    std::fs::write(&path, b"abcdef").expect("test file should be writable");

    execute(&format!(
        r#"
f = open("{}", "rb")
buf = bytearray(4)
n = f.readinto(buf)

if n != 4:
    raise RuntimeError(n)
if f.tell() != 4:
    raise RuntimeError(f.tell())
if buf[0] != 97 or buf[1] != 98 or buf[2] != 99 or buf[3] != 100:
    raise RuntimeError(buf)
f.close()
"#,
        python_path(&path)
    ));

    let _ = std::fs::remove_file(path);
}

#[test]
fn open_file_exposes_public_mode() {
    let path = std::env::temp_dir().join(format!(
        "prism-readinto-{}-{}.txt",
        std::process::id(),
        "mode"
    ));
    std::fs::write(&path, b"abcdef").expect("test file should be writable");

    execute(&format!(
        r#"
binary = open("{}", "rb")
if binary.mode != "rb":
    raise RuntimeError(binary.mode)
binary.close()

text = open("{}", "r")
if text.mode != "r":
    raise RuntimeError(text.mode)
text.close()
"#,
        python_path(&path),
        python_path(&path)
    ));

    let _ = std::fs::remove_file(path);
}
