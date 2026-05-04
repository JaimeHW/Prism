use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::{JitConfig, VirtualMachine};

fn execute_with_osr_jit(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");

    let mut config = JitConfig::benchmark();
    config.enable_osr = true;
    config.tier1_threshold = 1_000;
    config.tier2_threshold = u64::MAX;
    config.background_compilation = false;
    config.eager_compilation = true;

    let mut vm = VirtualMachine::with_jit_config(config);
    vm.execute_runtime(code).expect("source should execute");
}

#[test]
fn osr_skips_iterator_protocol_headers_until_live_ins_are_validated() {
    execute_with_osr_jit(
        r#"
class SubPattern:
    def __init__(self, data=None):
        if data is None:
            data = []
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SubPattern(self.data[index])
        return self.data[index]

LITERAL_CODES = (16,)
REPEATING_CODES = {44: (23, 18, 24)}
ANY = 2
SUCCESS = 1

def simple(pattern):
    return True

def compile_pattern(code, pattern, flags):
    emit = code.append
    _len = len
    literal_codes = LITERAL_CODES
    repeating_codes = REPEATING_CODES
    for op, av in pattern:
        if op in literal_codes:
            emit(op)
            emit(av)
        elif op is ANY:
            emit(ANY)
        elif op in repeating_codes:
            if simple(av[2]):
                emit(repeating_codes[op][2])
                skip = _len(code)
                emit(0)
                emit(av[0])
                emit(av[1])
                compile_pattern(code, av[2], flags)
                emit(SUCCESS)
                code[skip] = _len(code) - skip
            else:
                emit(repeating_codes[op][0])
                skip = _len(code)
                emit(0)
                emit(av[0])
                emit(av[1])
                compile_pattern(code, av[2], flags)
                code[skip] = _len(code) - skip
                emit(repeating_codes[op][1])
    return code

nested = SubPattern([(2, None)])
pattern = [
    (16, 69), (16, 120), (16, 99), (16, 101), (16, 112), (16, 116),
    (16, 105), (16, 111), (16, 110), (16, 32), (16, 105), (16, 110),
    (16, 32), (16, 99), (16, 97), (16, 108), (16, 108), (16, 98),
    (16, 97), (16, 99), (16, 107), (44, (0, 4294967295, nested)),
    (16, 122), (16, 101), (16, 114), (16, 111),
]

for _ in range(600):
    code = []
    result = compile_pattern(code, pattern, 16)
    if len(result) != 56:
        raise RuntimeError(result)
"#,
    );
}
