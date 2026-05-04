use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::{JitConfig, VirtualMachine};

fn execute_with_deterministic_jit(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");

    let mut config = JitConfig::benchmark();
    config.tier1_threshold = 5;
    config.tier2_threshold = u64::MAX;
    config.enable_osr = false;
    config.background_compilation = false;
    config.eager_compilation = true;

    let mut vm = VirtualMachine::with_jit_config(config);
    vm.execute_runtime(code).expect("source should execute");
}

#[test]
fn nested_jit_entries_preserve_the_outer_frame_state() {
    execute_with_deterministic_jit(
        r#"
class C:
    def __init__(self):
        self.x = None

    def advance(self):
        self.x = True

    def get(self):
        self.advance()
        return self.x

def run():
    c = C()
    return c.get()

for _ in range(200):
    if run() is not True:
        raise RuntimeError("bad result")
"#,
    );
}

#[test]
fn nested_jit_exceptions_unwind_the_compiled_target_frame() {
    execute_with_deterministic_jit(
        r#"
class SubPattern:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

def add_pattern(pattern):
    total = 0
    for op, av in pattern:
        total = total + op
    return total

pattern = SubPattern([(1, None), (2, None)])

for _ in range(50):
    result = add_pattern(pattern)
    if result != 3:
        raise RuntimeError(result)
"#,
    );
}
