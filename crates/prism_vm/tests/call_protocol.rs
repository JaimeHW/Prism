use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::VirtualMachine;

fn execute(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");
    let mut vm = VirtualMachine::new();
    vm.execute_runtime(code).expect("source should execute");
}

#[test]
fn callable_objects_expose_dunder_call() {
    execute(
        r#"
def plus_one(value):
    return value + 1

if getattr(plus_one, "__call__", None) is None:
    raise RuntimeError("function __call__ missing")
if plus_one.__call__(41) != 42:
    raise RuntimeError("function __call__ did not dispatch")

class Box:
    def method(self, value):
        return value + 2

bound = Box().method
if getattr(bound, "__call__", None) is None:
    raise RuntimeError("method __call__ missing")
if bound.__call__(40) != 42:
    raise RuntimeError("method __call__ did not dispatch")

if getattr(len, "__call__", None) is None:
    raise RuntimeError("builtin __call__ missing")
if len.__call__([1, 2, 3]) != 3:
    raise RuntimeError("builtin __call__ did not dispatch")

class Meta(type):
    def __new__(mcls, name, bases, namespace, **kwargs):
        return super().__new__(mcls, name, bases, namespace, **kwargs)

class UsesMeta(metaclass=Meta):
    pass
"#,
    );
}
