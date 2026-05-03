use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::{JitConfig, VirtualMachine};

fn execute(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");
    let mut vm = VirtualMachine::with_jit_config(JitConfig::disabled());
    vm.execute_runtime(code).expect("source should execute");
}

#[test]
fn f_string_formatting_ignores_shadowed_builtin_names() {
    execute(
        r#"
def format(value, spec=None):
    raise RuntimeError("shadowed format called")

def str(value):
    raise RuntimeError("shadowed str called")

def repr(value):
    raise RuntimeError("shadowed repr called")

def ascii(value):
    raise RuntimeError("shadowed ascii called")

value = "world"

plain = f"hello {value}"
converted = f"{value!s}|{value!r}|{value!a}"
padded = f"{7:03d}"

if plain != "hello world":
    raise RuntimeError(plain)
if converted != "world|'world'|'world'":
    raise RuntimeError(converted)
if padded != "007":
    raise RuntimeError(padded)
"#,
    );
}
