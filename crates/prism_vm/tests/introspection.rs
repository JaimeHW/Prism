use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::VirtualMachine;

fn execute(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");
    let mut vm = VirtualMachine::new();
    vm.execute_runtime(code).expect("source should execute");
}

#[test]
fn code_type_dir_exports_code_members() {
    execute(
        r#"
from types import CodeType

names = dir(CodeType)
for name in ("co_argcount", "co_flags", "co_posonlyargcount", "co_kwonlyargcount",
             "co_varnames", "co_names", "co_freevars", "co_cellvars", "co_positions"):
    if name not in names:
        raise RuntimeError(name)
"#,
    );
}

#[test]
fn code_flags_use_python_visible_bits() {
    execute(
        r#"
import inspect

def plain():
    pass

async def coroutine(*args, **kwargs):
    pass

if plain.__code__.co_flags & inspect.CO_OPTIMIZED != inspect.CO_OPTIMIZED:
    raise RuntimeError(plain.__code__.co_flags)
if plain.__code__.co_flags & inspect.CO_NEWLOCALS != inspect.CO_NEWLOCALS:
    raise RuntimeError(plain.__code__.co_flags)
if coroutine.__code__.co_flags & inspect.CO_COROUTINE != inspect.CO_COROUTINE:
    raise RuntimeError(coroutine.__code__.co_flags)
if coroutine.__code__.co_flags & inspect.CO_VARARGS != inspect.CO_VARARGS:
    raise RuntimeError(coroutine.__code__.co_flags)
if coroutine.__code__.co_flags & inspect.CO_VARKEYWORDS != inspect.CO_VARKEYWORDS:
    raise RuntimeError(coroutine.__code__.co_flags)
"#,
    );
}

#[test]
fn coroutine_function_detection_accepts_functionlike_code_flags() {
    execute(
        r#"
import inspect

class Code:
    pass

class FunctionLike:
    pass

function_like = FunctionLike()
function_like.__code__ = Code()
function_like.__code__.co_flags = inspect.CO_COROUTINE

if not inspect.iscoroutinefunction(function_like):
    raise RuntimeError("function-like coroutine was not detected")

function_like.__code__.co_flags = 0
if inspect.iscoroutinefunction(function_like):
    raise RuntimeError("non-coroutine function-like object was detected")
"#,
    );
}
