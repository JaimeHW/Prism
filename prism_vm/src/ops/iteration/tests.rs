use super::*;
use prism_compiler::Compiler;
use prism_parser::parse;
use std::sync::Arc;

fn execute(source: &str) -> Result<Value, String> {
    let module = parse(source).map_err(|err| format!("parse error: {err:?}"))?;
    let code = Compiler::compile_module(&module, "<iteration-test>")
        .map_err(|err| format!("compile error: {err:?}"))?;

    let mut vm = VirtualMachine::new();
    vm.execute_runtime(Arc::new(code))
        .map_err(|err| format!("runtime error: {err:?}"))
}

#[test]
fn test_sequence_getitem_exhaustion_does_not_leak_exc_info() {
    let result = execute(
        r#"
import sys

class Seq:
    def __getitem__(self, index):
        if index < 2:
            return index
        raise IndexError("done")

seen = []
for value in Seq():
    seen.append(value)

assert seen == [0, 1]
assert sys.exc_info() == (None, None, None)
"#,
    );

    assert!(
        result.is_ok(),
        "sequence fallback should clear exhaustion: {result:?}"
    );
}

#[test]
fn test_sequence_getitem_exhaustion_is_clean_for_type_constructors() {
    let result = execute(
        r#"
import sys

class Seq:
    def __getitem__(self, index):
        if index < 2:
            return index
        raise IndexError("done")

assert list(Seq()) == [0, 1]
assert tuple(Seq()) == (0, 1)
assert sys.exc_info() == (None, None, None)
"#,
    );

    assert!(
        result.is_ok(),
        "sequence fallback should make constructor collection clean: {result:?}"
    );
}

#[test]
fn test_protocol_next_exhaustion_does_not_leak_exc_info() {
    let result = execute(
        r#"
import sys

class It:
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration

for value in It():
    raise AssertionError("iterator should be empty")

assert next(It(), "sentinel") == "sentinel"
assert sys.exc_info() == (None, None, None)
"#,
    );

    assert!(
        result.is_ok(),
        "protocol StopIteration should be consumed cleanly: {result:?}"
    );
}
