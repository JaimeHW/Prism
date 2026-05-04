use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::VirtualMachine;

fn execute(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");
    let mut vm = VirtualMachine::new();
    vm.execute_runtime(code).expect("source should execute");
}

#[test]
fn direct_base_exception_subclass_preserves_active_exception_value() {
    execute(
        r#"
class MyBase(BaseException):
    pass

try:
    raise MyBase("x")
except BaseException as exc:
    seen_name = type(exc).__name__
    seen_args = exc.args
else:
    raise RuntimeError("handler did not run")

if seen_name != "MyBase":
    raise RuntimeError(seen_name)
if seen_args != ("x",):
    raise RuntimeError(seen_args)
"#,
    );
}

#[test]
fn raise_from_preserves_direct_base_exception_subclass_cause() {
    execute(
        r#"
class MyBase(BaseException):
    pass

try:
    try:
        raise MyBase("cancelled")
    except MyBase as exc:
        raise TimeoutError() from exc
except TimeoutError as exc:
    cause = exc.__cause__
else:
    raise RuntimeError("chained exception did not run")

if type(cause).__name__ != "MyBase":
    raise RuntimeError(type(cause).__name__)
if cause.args != ("cancelled",):
    raise RuntimeError(cause.args)
"#,
    );
}

#[test]
fn exception_group_exposes_message_and_exceptions_tuple() {
    execute(
        r#"
err = OSError("network")
group = ExceptionGroup("connect failed", [err])

if group.message != "connect failed":
    raise RuntimeError(group.message)
if len(group.exceptions) != 1:
    raise RuntimeError(group.exceptions)
if group.exceptions[0] is not err:
    raise RuntimeError(group.exceptions[0])
"#,
    );
}

#[test]
fn os_error_subclasses_inherit_errno_attributes() {
    execute(
        r#"
class Err(OSError):
    pass

exc = Err(7, "cannot bind", "addr")

if exc.errno != 7:
    raise RuntimeError(exc.errno)
if exc.strerror != "cannot bind":
    raise RuntimeError(exc.strerror)
if exc.filename != "addr":
    raise RuntimeError(exc.filename)
if exc.winerror is not None:
    raise RuntimeError(exc.winerror)
"#,
    );
}
