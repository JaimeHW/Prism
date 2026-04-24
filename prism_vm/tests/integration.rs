//! End-to-end integration tests for Prism VM.
//!
//! These tests verify the complete parse → compile → execute pipeline.

use prism_compiler::Compiler;
use prism_core::Value;
use prism_core::intern::interned_by_ptr;
use prism_parser::parse;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use prism_vm::VirtualMachine;
use prism_vm::import::ModuleObject;
use prism_vm::stdlib::re::RegexFlags;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Helper to run Python source code and return result.
fn execute(source: &str) -> Result<Value, String> {
    execute_with_search_paths(source, &[])
}

/// Helper to run Python source code with additional import search paths.
fn execute_with_search_paths(source: &str, search_paths: &[&Path]) -> Result<Value, String> {
    execute_with_search_paths_and_step_limit(source, search_paths, None)
}

fn execute_with_search_paths_and_step_limit(
    source: &str,
    search_paths: &[&Path],
    step_limit: Option<u64>,
) -> Result<Value, String> {
    // Parse
    let module = parse(source).map_err(|e| format!("Parse error: {:?}", e))?;

    // Compile
    let code = Compiler::compile_module(&module, "<test>")
        .map_err(|e| format!("Compile error: {:?}", e))?;

    // Execute
    let mut vm = VirtualMachine::new();
    for path in search_paths {
        let path = Arc::<str>::from(path.to_string_lossy().into_owned());
        vm.import_resolver.add_search_path(path);
    }
    vm.set_execution_step_limit(step_limit);
    vm.execute(Arc::new(code))
        .map_err(|e| format!("Runtime error: {:?}", e))
}

fn cpython_lib_dir() -> std::path::PathBuf {
    let root = std::env::var_os("PRISM_CPYTHON_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Users\James\Desktop\cpython-3.12"));
    let lib_dir = root.join("Lib");
    assert!(
        lib_dir.is_dir(),
        "CPython Lib directory not found at {}. Set PRISM_CPYTHON_ROOT to override.",
        lib_dir.display()
    );
    lib_dir
}

fn execute_with_cpython_lib(source: &str) -> Result<Value, String> {
    let lib_dir = cpython_lib_dir();
    execute_with_search_paths(source, &[lib_dir.as_path()])
}

fn execute_with_cpython_lib_and_step_limit(source: &str, step_limit: u64) -> Result<Value, String> {
    let lib_dir = cpython_lib_dir();
    execute_with_search_paths_and_step_limit(source, &[lib_dir.as_path()], Some(step_limit))
}

fn execute_in_main_module_with_search_paths(
    source: &str,
    search_paths: &[&Path],
) -> Result<(VirtualMachine, Arc<ModuleObject>), String> {
    let module = parse(source).map_err(|e| format!("Parse error: {:?}", e))?;
    let code = Compiler::compile_module(&module, "<test>")
        .map_err(|e| format!("Compile error: {:?}", e))?;

    let mut vm = VirtualMachine::new();
    for path in search_paths {
        let path = Arc::<str>::from(path.to_string_lossy().into_owned());
        vm.import_resolver.add_search_path(path);
    }

    let main = Arc::new(ModuleObject::new("__main__"));
    vm.execute_in_module(Arc::new(code), Arc::clone(&main))
        .map_err(|e| format!("Runtime error: {:?}", e))?;
    Ok((vm, main))
}

fn value_is_python_string(value: Value, expected: &str) -> bool {
    if value.is_string() {
        let Some(ptr) = value.as_string_object_ptr() else {
            return false;
        };
        return interned_by_ptr(ptr as *const u8)
            .is_some_and(|interned| interned.as_str() == expected);
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::STR {
        return false;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    string.as_str() == expected
}

fn python_string_value(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        return interned_by_ptr(ptr as *const u8).map(|interned| interned.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::STR {
        return None;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Some(string.as_str().to_string())
}

fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("prism_{label}_{}_{}", std::process::id(), nonce))
}

// =============================================================================
// VM Builtin Tests
// =============================================================================

#[test]
fn test_vm_builtins_are_initialized() {
    let vm = VirtualMachine::new();

    // Verify len builtin exists and is an object_ptr
    let len_val = vm.builtins.get("len").expect("len should exist");
    assert!(
        len_val.as_object_ptr().is_some(),
        "len should be object_ptr, got: bits = {:#x}",
        unsafe { std::mem::transmute::<Value, u64>(len_val) }
    );

    // Verify range builtin
    let range_val = vm.builtins.get("range").expect("range should exist");
    assert!(
        range_val.as_object_ptr().is_some(),
        "range should be object_ptr"
    );
}

// =============================================================================
// Arithmetic Tests
// =============================================================================

#[test]
fn test_integer_addition() {
    let result = execute("1 + 2");
    // Module execution returns None (last statement is expression, not returned)
    // For now, just verify it doesn't crash
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_integer_multiplication() {
    let result = execute("3 * 4");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_constructor_exception_after_successful_instance_is_not_masked_by_stale_return_register() {
    let result = execute(
        r#"
class Example:
    def __init__(self, ok):
        if ok:
            self.value = 1
        else:
            raise ValueError("boom")

first = Example(True)
assert first.value == 1

try:
    Example(False)
except ValueError as exc:
    assert exc.args == ("boom",)
else:
    raise AssertionError("expected ValueError")

third = Example(True)
assert third.value == 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_nonreflexive_values_preserve_identity_inside_builtin_containers() {
    let result = execute_with_cpython_lib(
        r#"
from collections import deque
from test.support import NEVER_EQ

values = float('nan'), 1, None, 'abc', NEVER_EQ
constructors = list, tuple, dict.fromkeys, set, frozenset, deque

for constructor in constructors:
    container = constructor(values)
    for elem in container:
        assert elem in container
    assert container == constructor(values)
    assert container == container
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_main_module_exposes_doc_binding_without_a_docstring() {
    let result = execute(
        r#"
assert __doc__ is None
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_imported_source_module_exposes_doc_binding_without_a_docstring() {
    let temp_dir = unique_temp_dir("module_doc_binding");
    fs::create_dir_all(&temp_dir).expect("failed to create temp dir");
    fs::write(temp_dir.join("docless_mod.py"), "VALUE = __doc__\n")
        .expect("failed to write temp module");

    let result = execute_with_search_paths(
        r#"
import docless_mod

assert docless_mod.VALUE is None
"#,
        &[temp_dir.as_path()],
    );

    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_native_chainmap_supports_unittest_style_ordered_subtest_params() {
    let result = execute_with_cpython_lib(
        r#"
from collections import ChainMap

class _OrderedChainMap(ChainMap):
    def __iter__(self):
        d = {}
        for mapping in reversed(self.maps):
            d.update(dict.fromkeys(mapping))
        return iter(d)

params = _OrderedChainMap({'scope': 'x'})
assert list(params.items()) == [('scope', 'x')]

child = params.new_child({'phase': 'y'})
assert list(child.items()) == [('scope', 'x'), ('phase', 'y')]
assert child['scope'] == 'x'
assert child['phase'] == 'y'
assert list(child.parents.items()) == [('scope', 'x')]

created = ChainMap.fromkeys(['alpha', 'beta'], 5)
assert list(created.items()) == [('alpha', 5), ('beta', 5)]
"#,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_compound_arithmetic() {
    let result = execute("2 + 3 * 4");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_float_division() {
    let result = execute("10.0 / 3.0");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_unary_plus_preserves_python_numeric_semantics() {
    let result = execute(
        r#"
assert +False == 0
assert (+False) is not False
assert +True == 1
assert (+True) is not True
assert +1.5 == 1.5
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_unary_plus_dispatches_dunder_pos() {
    let result = execute(
        r#"
class Signed:
    def __pos__(self):
        return 42

assert +Signed() == 42
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_bool_invert_emits_deprecation_warning() {
    let result = execute_with_cpython_lib(
        r#"
import warnings

with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter("always", DeprecationWarning)
    flag = False
    assert ~flag == -1
    assert len(seen) == 1
    assert seen[0].category is DeprecationWarning
    assert "Bitwise inversion '~' on bool is deprecated" in str(seen[0].message)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_eval_bool_invert_emits_deprecation_warning() {
    let result = execute_with_cpython_lib(
        r#"
import warnings

with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter("always", DeprecationWarning)
    assert eval("~False") == -1
    assert len(seen) == 1
    assert seen[0].category is DeprecationWarning
    assert "Bitwise inversion '~' on bool is deprecated" in str(seen[0].message)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_struct_module_supports_pickle_bootstrap_formats() {
    let result = execute_with_cpython_lib(
        r#"
import struct

assert struct.calcsize("<Q") == 8
assert struct.pack("<I", 0x12345678) == b"\x78\x56\x34\x12"
assert struct.unpack("<I", b"\x78\x56\x34\x12") == (0x12345678,)
assert struct.pack(">d", 1.5) == b"\x3f\xf8\x00\x00\x00\x00\x00\x00"
assert struct.unpack(">d", b"\x3f\xf8\x00\x00\x00\x00\x00\x00") == (1.5,)

buf = bytearray(b"\x00" * 10)
struct.pack_into("<H", buf, 4, 0x1234)
assert bytes(buf) == b"\x00\x00\x00\x00\x34\x12\x00\x00\x00\x00"
assert struct.unpack_from("<H", buf, 4) == (0x1234,)
assert list(struct.iter_unpack("<H", b"\x01\x00\x02\x00")) == [(1,), (2,)]

compiled = struct.Struct("<I")
assert compiled.size == 4
assert compiled.pack(7) == b"\x07\x00\x00\x00"
assert compiled.unpack(b"\x07\x00\x00\x00") == (7,)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_pickle_bool_round_trip_uses_native_struct_bootstrap() {
    let result = execute_with_cpython_lib(
        r#"
import pickle

for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    assert pickle.loads(pickle.dumps(True, proto)) is True
    assert pickle.loads(pickle.dumps(False, proto)) is False

assert pickle.dumps(True, protocol=0) == b"I01\n."
assert pickle.dumps(False, protocol=0) == b"I00\n."
assert pickle.dumps(True, protocol=1) == b"I01\n."
assert pickle.dumps(False, protocol=1) == b"I00\n."
assert pickle.dumps(True, protocol=2) == b"\x80\x02\x88."
assert pickle.dumps(False, protocol=2) == b"\x80\x02\x89."
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_platform_uname_pickle_round_trips_all_protocols_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import pickle
import platform

original = platform.uname()
for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    restored = pickle.loads(pickle.dumps(original, proto))
    assert restored == original
    assert tuple(restored) == tuple(original)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_bytesio_getbuffer_exposes_readable_buffer_contents() {
    let result = execute_with_cpython_lib(
        r#"
import io

buf = io.BytesIO(b"frame-data")
view = buf.getbuffer()

assert len(view) == 10
assert view == b"frame-data"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_for_iter_handles_large_bool_loop_exit_offsets() {
    let mut body = String::new();
    for _ in 0..48 {
        body.push_str("    acc += flag & 1\n");
        body.push_str("    acc += flag | 0\n");
        body.push_str("    acc += flag ^ 0\n");
    }
    let source = format!("acc = 0\nfor flag in (False, True):\n{body}result = acc\n");

    let (_vm, main) = execute_in_main_module_with_search_paths(&source, &[])
        .expect("large bool loop should execute");

    let result = main
        .get_attr("result")
        .expect("result binding should exist");
    assert_eq!(result.as_int(), Some(144));
}

#[test]
fn test_string_addition() {
    let result = execute(
        r#"
value = "hello" + " world"
assert value == "hello world"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_string_repetition() {
    let result = execute(
        r#"
value = "ab" * 3
assert value == "ababab"
assert "ab" * -1 == ""
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_fstring_values_can_drive_dynamic_exec_source() {
    let result = execute(
        r#"
key = "False"
source = f"{key} = 42"
assert source == "False = 42"

try:
    exec(source)
except SyntaxError:
    pass
else:
    raise AssertionError("expected SyntaxError from keyword assignment")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_fstring_numeric_format_specs_render_runtime_strings() {
    let result = execute(
        r#"
value = 255
width = 4
assert f"{value:{width}}" == " 255"
assert f"{value:08X}" == "000000FF"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_exec_without_explicit_namespaces_preserves_caller_frame() {
    let result = execute(
        r#"
marker = "alive"
exec("_ = 42")
assert marker == "alive"
assert _ == 42
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_exec_exceptions_can_transfer_control_to_surrounding_handlers() {
    let result = execute(
        r#"
handled = False
try:
    exec("raise ValueError('boom')")
except ValueError:
    handled = True

assert handled is True
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_with_cleanup_passes_exception_classes_into_unittest_assert_raises() {
    let result = execute_with_cpython_lib(
        r#"
import sys
import unittest

case = unittest.TestCase()
with case.assertRaises(SyntaxError):
    exec("False = 42")

assert sys.exc_info() == (None, None, None)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_with_cleanup_clears_exc_info_after_truthy_exit() {
    let result = execute(
        r#"
import sys

class SuppressSyntax:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert exc_type is SyntaxError
        assert exc is not None
        return True

with SuppressSyntax():
    exec("False = 42")

assert sys.exc_info() == (None, None, None)

with SuppressSyntax():
    exec("False = 42")

assert sys.exc_info() == (None, None, None)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_exception_with_traceback_round_trips_and_returns_same_instance() {
    let result = execute(
        r#"
try:
    raise ValueError("boom")
except ValueError as caught:
    tb = caught.__traceback__

rebound = BaseException().with_traceback(tb)
assert rebound.__traceback__ is tb
assert rebound.with_traceback(None) is rebound
assert rebound.__traceback__ is None
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_exception_traceback_assignment_validates_input() {
    let result = execute(
        r#"
try:
    raise ValueError("boom")
except ValueError as caught:
    tb = caught.__traceback__

sink = BaseException()
sink.__traceback__ = tb
assert sink.__traceback__ is tb
sink.__traceback__ = None
assert sink.__traceback__ is None

try:
    sink.__traceback__ = 1
except TypeError as exc:
    assert "__traceback__ must be a traceback" in str(exc)
else:
    raise AssertionError("expected traceback assignment to reject non-traceback values")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_traceback_extract_tb_reads_code_positions() {
    let result = execute_with_cpython_lib(
        r#"import traceback

def boom():
    raise ValueError("boom")

try:
    boom()
except ValueError as exc:
    frames = traceback.extract_tb(exc.__traceback__)
    assert len(frames) == 2
    outer = frames[0]
    assert outer.name == "<module>"
    assert outer.filename == "<test>"
    assert outer.lineno == 6
    frame = frames[1]
    assert frame.name == "boom"
    assert frame.filename == "<test>"
    assert frame.lineno == 3
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_exception_group_builtin_hierarchy_matches_cpython_surface() {
    let result = execute(
        r#"assert issubclass(BaseExceptionGroup, BaseException)
assert issubclass(ExceptionGroup, BaseExceptionGroup)
assert issubclass(ExceptionGroup, Exception)
assert BaseExceptionGroup.__bases__ == (BaseException,)
assert ExceptionGroup.__bases__ == (BaseExceptionGroup, Exception)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_generator_yield_comma_tuple_expression() {
    let result = execute(
        r#"def probe():
    yield 1, 2

g = probe()
assert next(g) == (1, 2)
try:
    next(g)
except StopIteration:
    pass
else:
    raise AssertionError("generator should be exhausted")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_function_descriptor_surface_matches_python_semantics() {
    let result = execute(
        r#"
class Box:
    pass

box = Box()
box.value = 41

def method(self):
    return self.value

assert hasattr(method, "__get__")
assert method.__get__(None, Box) is method
bound = method.__get__(box, Box)
assert bound() == 41
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Variable Tests
// =============================================================================

#[test]
fn test_assignment() {
    let result = execute("x = 42");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_assignment_and_use() {
    let result = execute(
        r#"
x = 10
y = 20
z = x + y
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_augmented_assignment() {
    let result = execute(
        r#"
x = 5
x += 3
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Control Flow Tests
// =============================================================================

#[test]
fn test_if_statement_true() {
    let result = execute(
        r#"
x = 10
if x > 5:
    y = 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_if_else_statement() {
    let result = execute(
        r#"
x = 3
if x > 5:
    y = 1
else:
    y = 0
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_while_loop() {
    let result = execute(
        r#"
x = 0
while x < 5:
    x = x + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Container Tests
// =============================================================================

#[test]
fn test_list_literal() {
    let result = execute("[1, 2, 3]");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_dict_literal() {
    let result = execute("{\"a\": 1, \"b\": 2}");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_tuple_literal() {
    let result = execute("(1, 2, 3)");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_set_literal() {
    let result = execute("{1, 2, 3}");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Comparison Tests
// =============================================================================

#[test]
fn test_comparison_operators() {
    let sources = ["1 < 2", "2 <= 2", "3 > 2", "3 >= 3", "1 == 1", "1 != 2"];

    for source in sources {
        let result = execute(source);
        assert!(result.is_ok(), "Failed for '{}': {:?}", source, result);
    }
}

#[test]
fn test_boolean_operators() {
    let sources = ["True and True", "True or False", "not False"];

    for source in sources {
        let result = execute(source);
        assert!(result.is_ok(), "Failed for '{}': {:?}", source, result);
    }
}

// =============================================================================
// Expression Tests
// =============================================================================

#[test]
fn test_parenthesized_expression() {
    let result = execute("(2 + 3) * 4");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_generator_lambda_creation_does_not_escape_yield_to_module_scope() {
    let result = execute("generator_type = type((lambda: (yield))())");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_generator_throw_injects_exception_into_suspended_generator() {
    let result = execute(
        r#"
def probe():
    try:
        yield 1
    except ValueError as exc:
        assert exc is marker
        observed.append(exc is marker)
        return

g = probe()
observed = []
marker = ValueError("boom")
assert next(g) == 1
try:
    g.throw(marker)
except StopIteration:
    assert observed == [True]
else:
    raise AssertionError("throw() should exhaust the generator after it is handled")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_generator_throw_on_unstarted_generator_raises_original_exception() {
    let result = execute(
        r#"
def probe():
    yield 1

g = probe()
marker = ValueError("boom")
try:
    g.throw(marker)
except ValueError as exc:
    assert exc is marker
else:
    raise AssertionError("throw() should propagate into an unstarted generator")

try:
    next(g)
except StopIteration:
    pass
else:
    raise AssertionError("unstarted throw() should exhaust the generator")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_explicit_metaclass_methods_are_visible_on_created_classes() {
    let result = execute(
        r#"
class Meta(type):
    def register(cls, value):
        return value

class Example(metaclass=Meta):
    pass

Example.register(1)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_metaclass_is_inherited_from_base_classes() {
    let result = execute(
        r#"
class Meta(type):
    def ping(cls):
        return 1

class Base(metaclass=Meta):
    pass

class Derived(Base):
    pass

Derived.ping()
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_type_new_builtin_survives_handled_descriptor_callback_exception() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
ERROR_TYPE = None
ERROR_TEXT = None
OBSERVED_RESULT_IS_NONE = None
OBSERVED_RESULT_TYPE = None
OBSERVED_MARKER = None

class Descriptor:
    def __set_name__(self, owner, name):
        try:
            {}["missing"]
        except KeyError:
            pass
        owner.marker = name

class Meta(type):
    pass

namespace = {"field": Descriptor()}

try:
    RESULT_CLASS = type.__new__(Meta, "Example", (), namespace)
except Exception as exc:
    ERROR_TYPE = type(exc).__name__
    ERROR_TEXT = str(exc)
else:
    OBSERVED_RESULT_IS_NONE = RESULT_CLASS is None
    OBSERVED_RESULT_TYPE = type(RESULT_CLASS).__name__
    OBSERVED_MARKER = RESULT_CLASS.marker
"#,
        &[],
    )
    .expect("direct type.__new__ probe should execute");

    assert_eq!(
        main.get_attr("ERROR_TYPE").and_then(python_string_value),
        None,
        "direct type.__new__ should not fail, got {:?}: {:?}",
        main.get_attr("ERROR_TYPE").and_then(python_string_value),
        main.get_attr("ERROR_TEXT").and_then(python_string_value),
    );
    assert_eq!(
        main.get_attr("OBSERVED_RESULT_IS_NONE")
            .and_then(|value| value.as_bool()),
        Some(false),
        "type.__new__ should produce a class object",
    );
    assert_eq!(
        main.get_attr("OBSERVED_RESULT_TYPE")
            .and_then(python_string_value),
        Some("Meta".to_string()),
        "direct type.__new__ should preserve the metaclass identity",
    );
    assert_eq!(
        main.get_attr("OBSERVED_MARKER")
            .and_then(python_string_value),
        Some("field".to_string()),
        "descriptor __set_name__ should run against the created class",
    );
}

#[test]
fn test_descriptor_set_name_handled_exception_does_not_poison_metaclass_super_new() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
ERROR_TYPE = None
ERROR_TEXT = None
OBSERVED_CLS_IS_NONE = None
OBSERVED_CLS_TYPE = None
OBSERVED_CLS_MARKER = None

class Descriptor:
    def __set_name__(self, owner, name):
        try:
            {}["missing"]
        except KeyError:
            pass
        owner.marker = name

class Meta(type):
    observed_cls_is_none = None
    observed_cls_type = None
    observed_cls_marker = None

    def __new__(metacls, name, bases, namespace):
        cls = super().__new__(metacls, name, bases, namespace)
        metacls.observed_cls_is_none = cls is None
        metacls.observed_cls_type = type(cls).__name__
        metacls.observed_cls_marker = getattr(cls, "marker", None)
        return cls

try:
    class Example(metaclass=Meta):
        field = Descriptor()
except Exception as exc:
    ERROR_TYPE = type(exc).__name__
    ERROR_TEXT = str(exc)
else:
    RESULT_EXAMPLE = Example
    OBSERVED_CLS_IS_NONE = Meta.observed_cls_is_none
    OBSERVED_CLS_TYPE = Meta.observed_cls_type
    OBSERVED_CLS_MARKER = Meta.observed_cls_marker
"#,
        &[],
    )
    .expect("descriptor callback probe should execute");

    assert_eq!(
        main.get_attr("ERROR_TYPE").and_then(python_string_value),
        None,
        "class creation should not fail, got {:?}: {:?}",
        main.get_attr("ERROR_TYPE").and_then(python_string_value),
        main.get_attr("ERROR_TEXT").and_then(python_string_value),
    );
    assert_eq!(
        main.get_attr("OBSERVED_CLS_IS_NONE")
            .and_then(|value| value.as_bool()),
        Some(false),
        "metaclass super().__new__ should produce a class object",
    );
    assert_eq!(
        main.get_attr("OBSERVED_CLS_TYPE")
            .and_then(python_string_value),
        Some("Meta".to_string()),
        "metaclass result should preserve the heap metaclass identity",
    );
    assert_eq!(
        main.get_attr("OBSERVED_CLS_MARKER")
            .and_then(python_string_value),
        Some("field".to_string()),
        "descriptor __set_name__ should see the created owner class",
    );
}

#[test]
fn test_prepare_callback_with_keywords_restores_exception_context_after_success() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
ERROR_TYPE = None
ERROR_TEXT = None
OBSERVED_PREPARED_BOUNDARY = None
OBSERVED_CLASS_BOUNDARY = None
OBSERVED_DERIVED_IS_NONE = None
OBSERVED_DERIVED_TYPE = None

class Meta(type):
    @classmethod
    def __prepare__(mcls, name, bases, boundary=None):
        try:
            {}["missing"]
        except KeyError:
            pass
        namespace = {}
        namespace["prepared_boundary"] = boundary
        return namespace

    def __new__(mcls, name, bases, namespace, boundary=None):
        cls = super().__new__(mcls, name, bases, namespace)
        cls.boundary = boundary
        return cls

try:
    class Derived(metaclass=Meta, boundary=7):
        value = 1
except Exception as exc:
    ERROR_TYPE = type(exc).__name__
    ERROR_TEXT = str(exc)
else:
    OBSERVED_DERIVED_IS_NONE = Derived is None
    OBSERVED_DERIVED_TYPE = type(Derived).__name__
    OBSERVED_PREPARED_BOUNDARY = Derived.prepared_boundary
    OBSERVED_CLASS_BOUNDARY = Derived.boundary
"#,
        &[],
    )
    .expect("__prepare__ keyword callback probe should execute");

    assert_eq!(
        main.get_attr("ERROR_TYPE").and_then(python_string_value),
        None,
        "class creation via __prepare__ should not fail, got {:?}: {:?}",
        main.get_attr("ERROR_TYPE").and_then(python_string_value),
        main.get_attr("ERROR_TEXT").and_then(python_string_value),
    );
    assert_eq!(
        main.get_attr("OBSERVED_PREPARED_BOUNDARY")
            .and_then(|value| value.as_int()),
        Some(7),
        "__prepare__ keyword callback should preserve its namespace return",
    );
    assert_eq!(
        main.get_attr("OBSERVED_DERIVED_IS_NONE")
            .and_then(|value| value.as_bool()),
        Some(false),
        "class statement should bind the created type object, got {:?}",
        main.get_attr("OBSERVED_DERIVED_TYPE")
            .and_then(python_string_value),
    );
    assert_eq!(
        main.get_attr("OBSERVED_CLASS_BOUNDARY")
            .and_then(|value| value.as_int()),
        Some(7),
        "metaclass __new__ should preserve keyword constructor state after callback success",
    );
}

#[test]
fn test_heap_type_reflection_exposes_mro_and_dict_to_python() {
    let result = execute(
        r#"
class Base:
    base_token = 1

class Child(Base):
    token = 7

assert Child.__mro__[0] is Child
assert Child.__mro__[1] is Base
assert Child.__mro__[2] is object
assert getattr(Child, "__mro__")[1] is Base
assert "token" in Child.__dict__
assert Child.__dict__["token"] == 7
assert "token" in getattr(Child, "__dict__")
assert getattr(Child, "__dict__")["token"] == 7
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_builtin_type_reflection_exposes_mro_and_mappingproxy_membership_to_python() {
    let result = execute(
        r#"
assert getattr(bool, "__mro__")[0] is bool
assert getattr(bool, "__mro__")[1] is int
assert getattr(bool, "__mro__")[2] is object
assert "fromkeys" in dict.__dict__
assert dict.__dict__["fromkeys"] is not None
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_mappingproxy_methods_support_class_dict_access_and_update() {
    let result = execute(
        r#"
class Sample:
    token = 7
    label = "ready"

proxy = Sample.__dict__
keys = list(proxy.keys())
assert "token" in keys
assert "label" in keys
assert proxy.get("token") == 7
assert proxy.get("missing", 11) == 11
assert len(proxy) >= 2

copied = {}
copied.update(proxy)
assert copied["token"] == 7
assert copied["label"] == "ready"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_mappingproxy_type_wraps_plain_dicts() {
    let result = execute_with_cpython_lib(
        r#"
from types import MappingProxyType

source = {"alpha": 1}
proxy = MappingProxyType(source)
assert proxy["alpha"] == 1

source["beta"] = 2
assert proxy["beta"] == 2
assert list(proxy.keys()) == ["alpha", "beta"]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_abc_abstractmethod_with_defaults_does_not_break_following_classmethod() {
    let result = execute_with_cpython_lib(
        r#"
from abc import ABCMeta, abstractmethod

class Example(metaclass=ABCMeta):
    @abstractmethod
    def trouble(self, value, fallback=None, extra=None):
        return value

    @classmethod
    def hook(cls, arg):
        return cls is Example and arg == 1

assert Example.hook(1)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_async_abc_subclasshook_survives_abstractmethod_with_defaults() {
    let result = execute_with_cpython_lib(
        r#"
from abc import ABCMeta, abstractmethod
from types import GenericAlias

def _check_methods(C, *methods):
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True

class AsyncIterable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __aiter__(self):
        return self

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsyncIterable:
            return _check_methods(C, "__aiter__")
        return NotImplemented

    __class_getitem__ = classmethod(GenericAlias)

class AsyncIterator(AsyncIterable):
    __slots__ = ()

    @abstractmethod
    async def __anext__(self):
        raise StopAsyncIteration

    def __aiter__(self):
        return self

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsyncIterator:
            return _check_methods(C, "__anext__", "__aiter__")
        return NotImplemented

class AsyncGenerator(AsyncIterator):
    __slots__ = ()

    @abstractmethod
    async def athrow(self, typ, val=None, tb=None):
        raise StopAsyncIteration

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsyncGenerator:
            return _check_methods(C, "__aiter__", "__anext__", "athrow")
        return NotImplemented

assert AsyncGenerator.__subclasshook__(AsyncGenerator) is True
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_collections_abc_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import _collections_abc
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_keyword_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import keyword

assert keyword.iskeyword("while")
assert not keyword.iskeyword("prism_runtime")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_test_support_os_helper_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
from test.support import os_helper

assert isinstance(os_helper.TESTFN_ASCII, str)
assert os_helper.TESTFN
assert os_helper.FS_NONASCII == "" or os_helper.FS_NONASCII in os_helper.TESTFN_NONASCII
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_thread_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import _thread

ident = _thread.get_ident()
assert isinstance(ident, int)
assert ident == _thread.get_ident()
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_collections_with_native_weakref_bootstrap() {
    let result = execute_with_cpython_lib(
        r#"
import collections
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_signal_with_cpython_stdlib_roundtrips_sigint_handler() {
    let result = execute_with_cpython_lib(
        r#"
import signal

previous = signal.getsignal(signal.SIGINT)
assert previous == signal.SIG_DFL
returned = signal.signal(signal.SIGINT, signal.SIG_IGN)
assert returned == signal.SIG_DFL
assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
signal.signal(signal.SIGINT, previous)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_weakref_with_cpython_stdlib_constructs_mapping() {
    let result = execute_with_cpython_lib(
        r#"
import weakref

cache = weakref.WeakKeyDictionary()
assert isinstance(cache, dict)
cache["ready"] = 1
assert cache["ready"] == 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_str_join_accepts_generator_expression() {
    let result = execute(
        r#"
joined = ", ".join(str(x) for x in (1, 2, 3))
assert joined == "1, 2, 3"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_list_extend_accepts_generator_expression() {
    let result = execute(
        r#"
values = []
values.extend(x * 2 for x in (1, 2, 3))
assert values == [2, 4, 6]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_generator_expression_captures_enclosing_function_local() {
    let result = execute(
        r#"
def outer(scale, values):
    return tuple(scale * value for value in values)

assert outer(10, (1, 2, 3)) == (10, 20, 30)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_list_equality_uses_contents() {
    let result = execute(
        r#"
assert ["1", "2", "3"] == ["1", "2", "3"]
assert ["1", "2", "3"] != ["1", "2"]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_tuple_equality_uses_contents() {
    let result = execute(
        r#"
assert ("1", "2", "3") == ("1", "2", "3")
assert ("1", "2", "3") != ("1", "2", "4")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_list_accepts_map_iterator() {
    let result = execute(
        r#"
values = list(map(str, (1, 2, 3)))
assert values == ["1", "2", "3"]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_tuple_accepts_map_iterator() {
    let result = execute(
        r#"
values = tuple(map(str, (1, 2, 3)))
assert values == ("1", "2", "3")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_list_map_iterator_materializes_string_items() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
values = list(map(str, (1, 2, 3)))
"#,
        &[],
    )
    .expect("list(map(str, ...)) should execute");

    let values = main
        .get_attr("values")
        .expect("values binding should exist");
    let ptr = values
        .as_object_ptr()
        .expect("values should be a heap list object");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::LIST);

    let list = unsafe { &*(ptr as *const ListObject) };
    assert_eq!(list.len(), 3);
    assert!(value_is_python_string(list.get(0).unwrap(), "1"));
    assert!(value_is_python_string(list.get(1).unwrap(), "2"));
    assert!(value_is_python_string(list.get(2).unwrap(), "3"));
}

#[test]
fn test_tuple_map_iterator_materializes_string_items() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
values = tuple(map(str, (1, 2, 3)))
"#,
        &[],
    )
    .expect("tuple(map(str, ...)) should execute");

    let values = main
        .get_attr("values")
        .expect("values binding should exist");
    let ptr = values
        .as_object_ptr()
        .expect("values should be a heap tuple object");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::TUPLE);

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    assert_eq!(tuple.len(), 3);
    assert!(value_is_python_string(tuple.get(0).unwrap(), "1"));
    assert!(value_is_python_string(tuple.get(1).unwrap(), "2"));
    assert!(value_is_python_string(tuple.get(2).unwrap(), "3"));
}

#[test]
fn test_namedtuple_creation_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
point = Point(1, 2)
assert point.x == 1
assert point.y == 2
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_copy_uses_user_reduce_via_object_reduce_ex_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import copy

def factory():
    return "ok"

class Reducible:
    def __reduce__(self):
        return (factory, ())

assert copy.copy(Reducible()) == "ok"
assert copy.deepcopy(Reducible()) == "ok"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_dis_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import dis

assert dis._Instruction.opname.__doc__ == "Human readable name for operation"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_functools_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import functools
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_codecs_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import codecs
assert codecs.lookup("utf-8")
assert codecs.lookup_error("strict")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_importlib_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import importlib
RESULT = importlib.import_module("keyword").iskeyword("for")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
    assert_eq!(
        result
            .expect("importlib import should return a value")
            .as_bool(),
        Some(true)
    );
}

#[test]
fn test_import_warnings_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import warnings
RESULT = warnings.defaultaction
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
    assert!(value_is_python_string(
        result.expect("warnings import should return a value"),
        "default"
    ));
}

#[test]
fn test_type_constructor_keywords_dispatch_to_heap_metaclass() {
    let result = execute(
        r#"
class Meta(type):
    def __new__(mcls, name, bases, namespace, token=None, simple=False):
        cls = super().__new__(mcls, name, bases, namespace)
        cls.token = token
        cls.simple = simple
        return cls

class Base(metaclass=Meta):
    pass

Created = type("Created", (Base,), {"value": 1}, token=7, simple=True)
assert Created.token == 7
assert Created.simple is True
assert Created.value == 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_class_definition_keywords_dispatch_to_metaclass_protocol() {
    let result = execute(
        r#"
class Meta(type):
    @classmethod
    def __prepare__(mcls, name, bases, boundary=None):
        namespace = {}
        namespace["prepared_boundary"] = boundary
        return namespace

    def __new__(mcls, name, bases, namespace, boundary=None):
        cls = super().__new__(mcls, name, bases, namespace)
        cls.boundary = boundary
        cls.prepared_boundary = namespace["prepared_boundary"]
        return cls

class Base(metaclass=Meta):
    pass

class Derived(Base, boundary=7):
    value = 1

assert Derived.boundary == 7
assert Derived.prepared_boundary == 7
assert Derived.value == 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_enum_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import enum
RESULT = enum.KEEP.name
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
    assert!(value_is_python_string(
        result.expect("enum import should return a value"),
        "KEEP"
    ));
}

#[test]
fn test_import_enum_prefix_through_reprenum() {
    let temp_dir = unique_temp_dir("enum_prefix_reprenum");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("mini_enum_base.py");
    let enum_source =
        fs::read_to_string(cpython_lib_dir().join("enum.py")).expect("failed to read enum.py");
    let prefix = enum_source
        .lines()
        .take(1296)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&module_path, prefix).expect("failed to write temp enum prefix");

    let cpython_lib = cpython_lib_dir();
    let result = execute_with_search_paths(
        r#"
import mini_enum_base

class ReprEnum(mini_enum_base.Enum):
    pass
"#,
        &[temp_dir.as_path(), cpython_lib.as_path()],
    );

    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_enum_prefix_through_strenum() {
    let temp_dir = unique_temp_dir("enum_prefix_strenum");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("mini_enum_str.py");
    let enum_source =
        fs::read_to_string(cpython_lib_dir().join("enum.py")).expect("failed to read enum.py");
    let prefix = enum_source
        .lines()
        .take(1341)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&module_path, prefix).expect("failed to write temp enum prefix");

    let cpython_lib = cpython_lib_dir();
    let result = execute_with_search_paths(
        r#"
import mini_enum_str
"#,
        &[temp_dir.as_path(), cpython_lib.as_path()],
    );

    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_enum_prefix_strenum_registers_heap_classes_globally() {
    let temp_dir = unique_temp_dir("enum_prefix_strenum_registry");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("mini_enum_str_registry.py");
    let enum_source =
        fs::read_to_string(cpython_lib_dir().join("enum.py")).expect("failed to read enum.py");
    let prefix = enum_source
        .lines()
        .take(1341)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&module_path, prefix).expect("failed to write temp enum prefix");

    let cpython_lib = cpython_lib_dir();
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
import mini_enum_str_registry

StrEnum = mini_enum_str_registry.StrEnum
ReprEnum = mini_enum_str_registry.ReprEnum
"#,
        &[temp_dir.as_path(), cpython_lib.as_path()],
    )
    .expect("enum prefix import should execute");

    let str_enum = main.get_attr("StrEnum").expect("StrEnum should be bound");
    let repr_enum = main.get_attr("ReprEnum").expect("ReprEnum should be bound");

    let str_enum_ptr = str_enum.as_object_ptr().expect("StrEnum should be a class");
    let repr_enum_ptr = repr_enum
        .as_object_ptr()
        .expect("ReprEnum should be a class");
    let str_enum_class = unsafe { &*(str_enum_ptr as *const PyClassObject) };
    let repr_enum_class = unsafe { &*(repr_enum_ptr as *const PyClassObject) };

    assert!(
        global_class(str_enum_class.class_id()).is_some(),
        "StrEnum should be globally registered"
    );
    assert!(
        global_class(repr_enum_class.class_id()).is_some(),
        "ReprEnum should be globally registered"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_import_enum_prefix_strenum_exposes_heap_mro() {
    let temp_dir = unique_temp_dir("enum_prefix_strenum_mro");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("mini_enum_str_mro.py");
    let enum_source =
        fs::read_to_string(cpython_lib_dir().join("enum.py")).expect("failed to read enum.py");
    let prefix = enum_source
        .lines()
        .take(1341)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&module_path, prefix).expect("failed to write temp enum prefix");

    let cpython_lib = cpython_lib_dir();
    let result = execute_with_search_paths(
        r#"
import mini_enum_str_mro

assert mini_enum_str_mro.StrEnum.__mro__[0] is mini_enum_str_mro.StrEnum
assert mini_enum_str_mro.StrEnum.__mro__[1] is str
assert mini_enum_str_mro.StrEnum.__mro__[2] is mini_enum_str_mro.ReprEnum
"#,
        &[temp_dir.as_path(), cpython_lib.as_path()],
    );

    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_enum_prefix_strenum_generate_next_value_diagnostics() {
    let temp_dir = unique_temp_dir("enum_prefix_strenum_diag");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("mini_enum_diag.py");
    let enum_source =
        fs::read_to_string(cpython_lib_dir().join("enum.py")).expect("failed to read enum.py");
    let prefix = enum_source
        .lines()
        .take(1308)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&module_path, prefix).expect("failed to write temp enum prefix");

    let cpython_lib = cpython_lib_dir();
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
import mini_enum_diag

EVENT_COUNT = 0
SECOND_IS_SUNDER = None
SECOND_PRESENT = None
SECOND_IS_STATICMETHOD = None
CAUGHT_HAS_GENERATE_NEXT_VALUE_ATTR = None
RESULT_OK = False
RESULT_ERROR = None

_original_setitem = mini_enum_diag._EnumDict.__setitem__

def _trace_setitem(self, key, value):
    global EVENT_COUNT, SECOND_IS_SUNDER, SECOND_PRESENT, SECOND_IS_STATICMETHOD
    global CAUGHT_HAS_GENERATE_NEXT_VALUE_ATTR
    if key == "_generate_next_value_":
        EVENT_COUNT += 1
        if EVENT_COUNT == 2:
            SECOND_IS_SUNDER = mini_enum_diag._is_sunder(key)
            SECOND_PRESENT = key in self
            SECOND_IS_STATICMETHOD = isinstance(value, staticmethod)
    try:
        return _original_setitem(self, key, value)
    except Exception:
        if key == "_generate_next_value_":
            CAUGHT_HAS_GENERATE_NEXT_VALUE_ATTR = hasattr(self, "_generate_next_value")
        raise

mini_enum_diag._EnumDict.__setitem__ = _trace_setitem

try:
    class StrEnum(str, mini_enum_diag.ReprEnum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()
except Exception as exc:
    RESULT_ERROR = str(exc)
else:
    RESULT_OK = True
"#,
        &[temp_dir.as_path(), cpython_lib.as_path()],
    )
    .expect("diagnostic enum probe should execute");

    let _ = fs::remove_dir_all(&temp_dir);
    assert_eq!(
        main.get_attr("EVENT_COUNT")
            .and_then(|value| value.as_int()),
        Some(2)
    );
    assert_eq!(
        main.get_attr("SECOND_IS_SUNDER")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("SECOND_PRESENT")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("SECOND_IS_STATICMETHOD")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("CAUGHT_HAS_GENERATE_NEXT_VALUE_ATTR")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("RESULT_OK").and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[test]
fn test_import_enum_prefix_through_flag_boundary() {
    let temp_dir = unique_temp_dir("enum_prefix_flag_boundary");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("mini_enum_flag.py");
    let enum_source =
        fs::read_to_string(cpython_lib_dir().join("enum.py")).expect("failed to read enum.py");
    let prefix = enum_source
        .lines()
        .take(1365)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&module_path, prefix).expect("failed to write temp enum prefix");

    let cpython_lib = cpython_lib_dir();
    let result = execute_with_search_paths(
        r#"
import mini_enum_flag
"#,
        &[temp_dir.as_path(), cpython_lib.as_path()],
    );

    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_enum_prefix_strenum_subclass_with_members_initializes_cleanly() {
    let temp_dir = unique_temp_dir("enum_prefix_strenum_members");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("mini_enum_str_members.py");
    let enum_source =
        fs::read_to_string(cpython_lib_dir().join("enum.py")).expect("failed to read enum.py");
    let prefix = enum_source
        .lines()
        .take(1341)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&module_path, prefix).expect("failed to write temp enum prefix");

    let cpython_lib = cpython_lib_dir();
    let result = execute_with_search_paths(
        r#"
import mini_enum_str_members

class FlagBoundary(mini_enum_str_members.StrEnum):
    STRICT = mini_enum_str_members.auto()
    CONFORM = mini_enum_str_members.auto()
    EJECT = mini_enum_str_members.auto()
    KEEP = mini_enum_str_members.auto()

assert FlagBoundary.__mro__[0] is FlagBoundary
assert FlagBoundary.__mro__[1] is mini_enum_str_members.StrEnum
assert FlagBoundary.STRICT.value == "strict"
assert tuple(member.name for member in FlagBoundary) == ("STRICT", "CONFORM", "EJECT", "KEEP")
"#,
        &[temp_dir.as_path(), cpython_lib.as_path()],
    );

    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_enum_prefix_strenum_subclass_member_construction_reaches_class_completion() {
    let temp_dir = unique_temp_dir("enum_prefix_strenum_members_stage");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("mini_enum_str_members_stage.py");
    let enum_source =
        fs::read_to_string(cpython_lib_dir().join("enum.py")).expect("failed to read enum.py");
    let prefix = enum_source
        .lines()
        .take(1341)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&module_path, prefix).expect("failed to write temp enum prefix");

    let cpython_lib = cpython_lib_dir();
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
import mini_enum_str_members_stage

STAGE = "before_class"
ERROR = None
HEAD_OK = None
STRICT_OK = None
HOOK_ENTERED = False
HOOK_MRO_OK = False
HOOK_STAGE = "not_entered"
HOOK_LAST_BASE = None
HOOK_CLASS = None
TRACE_LINE = None
TRACE_NAME = None
TRACE_FILE = None
HOOK_CALLS = 0
HOOK_MEMBER = None

_original_set_name = mini_enum_str_members_stage._proto_member.__set_name__

def _traced_set_name(self, enum_class, member_name):
    global HOOK_ENTERED, HOOK_MRO_OK, HOOK_STAGE, HOOK_LAST_BASE, HOOK_CLASS, HOOK_CALLS, HOOK_MEMBER
    HOOK_ENTERED = True
    HOOK_CLASS = enum_class
    HOOK_CALLS += 1
    HOOK_MEMBER = member_name
    HOOK_STAGE = "entered"
    _ = enum_class.__mro__
    HOOK_MRO_OK = True
    HOOK_STAGE = "after_initial_mro"
    delattr(enum_class, member_name)
    HOOK_STAGE = "after_delattr"

    value = self.value
    HOOK_STAGE = "after_value"
    if not isinstance(value, tuple):
        args = (value,)
    else:
        args = value
    HOOK_STAGE = "after_args"
    if enum_class._member_type_ is tuple:
        args = (args,)
    HOOK_STAGE = "after_member_type_args"

    if not enum_class._use_args_:
        enum_member = enum_class._new_member_(enum_class)
    else:
        enum_member = enum_class._new_member_(enum_class, *args)
    HOOK_STAGE = "after_new_member"

    if not hasattr(enum_member, "_value_"):
        if enum_class._member_type_ is object:
            enum_member._value_ = value
        else:
            enum_member._value_ = enum_class._member_type_(*args)
    HOOK_STAGE = "after_value_init"

    value = enum_member._value_
    enum_member._name_ = member_name
    enum_member.__objclass__ = enum_class
    HOOK_STAGE = "after_member_identity"
    enum_member.__init__(*args)
    enum_member._sort_order_ = len(enum_class._member_names_)
    HOOK_STAGE = "after_member_init"

    try:
        try:
            enum_member = enum_class._value2member_map_[value]
            HOOK_STAGE = "after_fast_alias_lookup"
        except TypeError:
            HOOK_STAGE = "after_alias_lookup_typeerror"
            for name, canonical_member in enum_class._member_map_.items():
                if canonical_member._value_ == value:
                    enum_member = canonical_member
                    HOOK_STAGE = "after_alias_linear_match"
                    break
            else:
                HOOK_STAGE = "before_alias_keyerror"
                raise KeyError
    except KeyError:
        HOOK_STAGE = "after_alias_keyerror"
        enum_class._member_names_.append(member_name)

    HOOK_STAGE = "before_descriptor_scan"
    found_descriptor = None
    descriptor_type = None
    class_type = None
    mro_tail = enum_class.__mro__[1:]
    HOOK_STAGE = "after_descriptor_mro"
    for base in mro_tail:
        HOOK_LAST_BASE = getattr(base, "__name__", None)
        HOOK_STAGE = "scanning_base"
        attr = base.__dict__.get(member_name)
        if attr is not None:
            if isinstance(attr, (property, mini_enum_str_members_stage.DynamicClassAttribute)):
                found_descriptor = attr
                class_type = base
                descriptor_type = "enum"
                HOOK_STAGE = "found_enum_descriptor"
                break
            elif mini_enum_str_members_stage._is_descriptor(attr):
                found_descriptor = attr
                descriptor_type = descriptor_type or "desc"
                class_type = class_type or base
                HOOK_STAGE = "found_data_descriptor"
                continue
            else:
                descriptor_type = "attr"
                class_type = base
                HOOK_STAGE = "found_plain_attr"

    if found_descriptor:
        redirect = property()
        redirect.member = enum_member
        redirect.__set_name__(enum_class, member_name)
        if descriptor_type in ("enum", "desc"):
            redirect.fget = getattr(found_descriptor, "fget", None)
            redirect._get = getattr(found_descriptor, "__get__", None)
            redirect.fset = getattr(found_descriptor, "fset", None)
            redirect._set = getattr(found_descriptor, "__set__", None)
            redirect.fdel = getattr(found_descriptor, "fdel", None)
            redirect._del = getattr(found_descriptor, "__delete__", None)
        redirect._attr_type = descriptor_type
        redirect._cls_type = class_type
        setattr(enum_class, member_name, redirect)
        HOOK_STAGE = "after_redirect_setattr"
    else:
        setattr(enum_class, member_name, enum_member)
        HOOK_STAGE = "after_member_setattr"

    enum_class._member_map_[member_name] = enum_member
    HOOK_STAGE = "after_member_map"
    try:
        enum_class._value2member_map_.setdefault(value, enum_member)
        HOOK_STAGE = "after_value2member"
    except TypeError:
        enum_class._unhashable_values_.append(value)
        HOOK_STAGE = "after_unhashable_append"

    return None

mini_enum_str_members_stage._proto_member.__set_name__ = _traced_set_name

try:
    class FlagBoundary(mini_enum_str_members_stage.StrEnum):
        STRICT = mini_enum_str_members_stage.auto()
        CONFORM = mini_enum_str_members_stage.auto()
        EJECT = mini_enum_str_members_stage.auto()
        KEEP = mini_enum_str_members_stage.auto()

    STAGE = "after_class"
    HEAD_OK = FlagBoundary.__mro__[0] is FlagBoundary
    STAGE = "after_mro"
    STRICT_OK = FlagBoundary.STRICT.value == "strict"
    STAGE = "after_member"
except Exception as exc:
    ERROR = str(exc)
    tb = exc.__traceback__
    while tb is not None:
        TRACE_LINE = tb.tb_lineno
        TRACE_NAME = tb.tb_frame.f_code.co_name
        TRACE_FILE = tb.tb_frame.f_code.co_filename
        tb = tb.tb_next
"#,
        &[temp_dir.as_path(), cpython_lib.as_path()],
    )
    .expect("diagnostic enum stage probe should execute");

    let _ = fs::remove_dir_all(&temp_dir);
    let stage = python_string_value(main.get_attr("STAGE").expect("STAGE should be bound"))
        .expect("STAGE should be a python string");
    let error = main
        .get_attr("ERROR")
        .and_then(python_string_value)
        .unwrap_or_else(|| "<none>".to_string());
    let hook_entered = main
        .get_attr("HOOK_ENTERED")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let hook_mro_ok = main
        .get_attr("HOOK_MRO_OK")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let hook_stage = main
        .get_attr("HOOK_STAGE")
        .and_then(python_string_value)
        .unwrap_or_else(|| "<none>".to_string());
    let hook_last_base = main
        .get_attr("HOOK_LAST_BASE")
        .and_then(python_string_value)
        .unwrap_or_else(|| "<none>".to_string());
    let hook_class = main
        .get_attr("HOOK_CLASS")
        .expect("HOOK_CLASS should be bound");
    let hook_class_ptr = hook_class
        .as_object_ptr()
        .expect("HOOK_CLASS should reference the active enum class");
    let hook_class = unsafe { &*(hook_class_ptr as *const PyClassObject) };
    let hook_class_id = hook_class.class_id().0;
    let hook_mro_registry = hook_class
        .mro()
        .iter()
        .map(|class_id| {
            let registered = if class_id.0 < TypeId::FIRST_USER_TYPE {
                true
            } else {
                global_class(*class_id).is_some()
            };
            format!("{}={registered}", class_id.0)
        })
        .collect::<Vec<_>>()
        .join(", ");
    let trace_line = main
        .get_attr("TRACE_LINE")
        .and_then(|value| value.as_int())
        .unwrap_or(-1);
    let trace_name = main
        .get_attr("TRACE_NAME")
        .and_then(python_string_value)
        .unwrap_or_else(|| "<none>".to_string());
    let trace_file = main
        .get_attr("TRACE_FILE")
        .and_then(python_string_value)
        .unwrap_or_else(|| "<none>".to_string());
    let hook_calls = main
        .get_attr("HOOK_CALLS")
        .and_then(|value| value.as_int())
        .unwrap_or(-1);
    let hook_member = main
        .get_attr("HOOK_MEMBER")
        .and_then(python_string_value)
        .unwrap_or_else(|| "<none>".to_string());
    assert_eq!(
        stage, "after_member",
        "ERROR={error}; HOOK_ENTERED={hook_entered}; HOOK_MRO_OK={hook_mro_ok}; HOOK_STAGE={hook_stage}; HOOK_LAST_BASE={hook_last_base}; HOOK_CLASS_ID={hook_class_id}; HOOK_MRO_REGISTRY=[{hook_mro_registry}]; TRACE_LINE={trace_line}; TRACE_NAME={trace_name}; TRACE_FILE={trace_file}; HOOK_CALLS={hook_calls}; HOOK_MEMBER={hook_member}"
    );
    assert_eq!(
        main.get_attr("ERROR").map(|value| value.is_none()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("HEAD_OK").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("STRICT_OK").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("HOOK_ENTERED")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("HOOK_MRO_OK")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("HOOK_STAGE")
            .and_then(python_string_value)
            .as_deref(),
        Some("after_value2member")
    );
}

#[test]
fn test_prepare_mapping_observes_live_class_namespace_mutations() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class TrackingDict(dict):
    def __init__(self):
        super().__init__()
        self.events = []

    def __setitem__(self, key, value):
        self.events.append(("set", key, key in self))
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        self.events.append(("del", key, key in self))
        return super().__delitem__(key)

class Meta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        namespace = TrackingDict()
        namespace["seed"] = 41
        return namespace

    def __new__(mcls, name, bases, namespace):
        filtered = []
        for event in namespace.events:
            key = event[1]
            if key == "seed" or key == "before" or key == "after":
                filtered.append(event)
        cls = super().__new__(mcls, name, bases, namespace)
        cls.before = namespace["before"]
        cls.seed = namespace["seed"]
        cls.filtered_event_count = len(filtered)
        return cls

class Example(metaclass=Meta):
    before = seed
    seed = 42
    after = seed
    del after

CHECK_BEFORE = Example.before
CHECK_SEED = Example.seed
CHECK_AFTER_PRESENT = hasattr(Example, "after")
CHECK_FILTERED_EVENT_COUNT = Example.filtered_event_count
"#,
        &[],
    )
    .expect("prepared namespace mutation program should execute");

    assert_eq!(
        main.get_attr("CHECK_BEFORE")
            .and_then(|value| value.as_int()),
        Some(41)
    );
    assert_eq!(
        main.get_attr("CHECK_SEED").and_then(|value| value.as_int()),
        Some(42)
    );
    assert_eq!(
        main.get_attr("CHECK_AFTER_PRESENT")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        main.get_attr("CHECK_FILTERED_EVENT_COUNT")
            .and_then(|value| value.as_int()),
        Some(4)
    );
}

#[test]
fn test_prepare_mapping_does_not_replay_class_body_writes() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class RecordingDict(dict):
    def __init__(self):
        super().__init__()
        self.counts = {}

    def __setitem__(self, key, value):
        self.counts[key] = self.counts.get(key, 0) + 1
        return super().__setitem__(key, value)

class Meta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        namespace = RecordingDict()
        namespace["seed"] = 0
        return namespace

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        cls.seed_writes = namespace.counts["seed"]
        cls.value_writes = namespace.counts["value"]
        return cls

class Example(metaclass=Meta):
    seed = 1
    value = 2

RESULT_SEED_WRITES = Example.seed_writes
RESULT_VALUE_WRITES = Example.value_writes

"#,
        &[],
    )
    .expect("prepared namespace replay probe should execute");

    assert_eq!(
        main.get_attr("RESULT_SEED_WRITES")
            .and_then(|value| value.as_int()),
        Some(2)
    );
    assert_eq!(
        main.get_attr("RESULT_VALUE_WRITES")
            .and_then(|value| value.as_int()),
        Some(1)
    );
}

#[test]
fn test_heap_dict_subscript_overrides_bypass_native_fast_path() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class RecordingDict(dict):
    def __init__(self):
        super().__init__()
        self.set_count = 0
        self.get_count = 0
        self.del_count = 0

    def __setitem__(self, key, value):
        self.set_count += 1
        return super().__setitem__(key, value + 1)

    def __getitem__(self, key):
        self.get_count += 1
        return super().__getitem__(key) + 10

    def __delitem__(self, key):
        self.del_count += 1
        return super().__delitem__(key)

mapping = RecordingDict()
mapping["token"] = 5
RESULT_READ = mapping["token"]
del mapping["token"]
RESULT_AFTER_DELETE = mapping.get("token")
RESULT_SET_COUNT = mapping.set_count
RESULT_GET_COUNT = mapping.get_count
RESULT_DEL_COUNT = mapping.del_count
"#,
        &[],
    )
    .expect("dict subclass override probe should execute");

    assert_eq!(
        main.get_attr("RESULT_READ")
            .and_then(|value| value.as_int()),
        Some(16)
    );
    assert_eq!(
        main.get_attr("RESULT_SET_COUNT")
            .and_then(|value| value.as_int()),
        Some(1)
    );
    assert_eq!(
        main.get_attr("RESULT_GET_COUNT")
            .and_then(|value| value.as_int()),
        Some(1)
    );
    assert_eq!(
        main.get_attr("RESULT_DEL_COUNT")
            .and_then(|value| value.as_int()),
        Some(1)
    );
    assert!(
        main.get_attr("RESULT_AFTER_DELETE")
            .is_some_and(|value| value.is_none())
    );
}

#[test]
fn test_tuple_subscript_keys_round_trip_through_getitem() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Matrix:
    def __getitem__(self, key):
        return key

plain_key = Matrix()[1, 2]
slice_key = Matrix()[1:3, 4]

RESULT_PLAIN_LEN = len(plain_key)
RESULT_PLAIN_FIRST = plain_key[0]
RESULT_PLAIN_SECOND = plain_key[1]
RESULT_SLICE_LEN = len(slice_key)
RESULT_SLICE_IS_SLICE = type(slice_key[0]).__name__ == "slice"
RESULT_SLICE_SECOND = slice_key[1]
"#,
        &[],
    )
    .expect("tuple subscript probe should execute");

    assert_eq!(
        main.get_attr("RESULT_PLAIN_LEN")
            .and_then(|value| value.as_int()),
        Some(2)
    );
    assert_eq!(
        main.get_attr("RESULT_PLAIN_FIRST")
            .and_then(|value| value.as_int()),
        Some(1)
    );
    assert_eq!(
        main.get_attr("RESULT_PLAIN_SECOND")
            .and_then(|value| value.as_int()),
        Some(2)
    );
    assert_eq!(
        main.get_attr("RESULT_SLICE_IS_SLICE")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("RESULT_SLICE_LEN")
            .and_then(|value| value.as_int()),
        Some(2)
    );
    assert_eq!(
        main.get_attr("RESULT_SLICE_SECOND")
            .and_then(|value| value.as_int()),
        Some(4)
    );
}

#[test]
fn test_delete_attribute_executes_runtime_protocol() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Box:
    pass

box = Box()
box.value = 41
del box.value

try:
    box.value
except AttributeError:
    RESULT_MISSING = True
else:
    RESULT_MISSING = False
"#,
        &[],
    )
    .expect("attribute delete probe should execute");

    assert_eq!(
        main.get_attr("RESULT_MISSING")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[test]
fn test_prepare_mapping_reads_fall_back_to_globals_before_local_assignment() {
    let result = execute(
        r#"
seed = 7

class TrackingDict(dict):
    pass

class Meta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        return TrackingDict()

class Example(metaclass=Meta):
    before = seed
    seed = 9

assert Example.before == 7
assert Example.seed == 9
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_enum_style_sunder_detection_matches_cpython_semantics() {
    let result = execute(
        r#"
def _is_sunder(name):
    return (
        len(name) > 2 and
        name[0] == name[-1] == '_' and
        name[1:2] != '_' and
        name[-2:-1] != '_'
    )

assert _is_sunder("_generate_next_value_")
assert _is_sunder("_ignore_")
assert not _is_sunder("__dunder__")
assert not _is_sunder("plain")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_chained_string_comparison_with_negative_slice_bounds() {
    let result = execute(
        r#"
name = "_generate_next_value_"

assert name[0] == name[-1] == "_"
assert name[1:2] != "_"
assert name[-2:-1] != "_"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_prepare_mapping_supports_decorated_sunder_redefinition() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
def _is_sunder(name):
    return (
        len(name) > 2 and
        name[0] == name[-1] == "_" and
        name[1:2] != "_" and
        name[-2:-1] != "_"
    )

class EnumLikeDict(dict):
    def __init__(self):
        super().__init__()
        self._member_names = {}

    def __setitem__(self, key, value):
        if _is_sunder(key):
            if key == "_generate_next_value_":
                pass
        elif key in self._member_names:
            raise TypeError("%r already defined as %r" % (key, self[key]))
        else:
            if key in self:
                raise TypeError("%r already defined as %r" % (key, self[key]))
        return super().__setitem__(key, value)

class Meta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        namespace = EnumLikeDict()
        namespace["_generate_next_value_"] = object()
        return namespace

class Example(metaclass=Meta):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count

RESULT = Example._generate_next_value_("x", 1, 2, [])
"#,
        &[],
    )
    .expect("decorated sunder class body should execute");
    assert_eq!(
        main.get_attr("RESULT").and_then(|value| value.as_int()),
        Some(2)
    );
}

#[test]
fn test_nested_if_inside_elif_does_not_fall_through_outer_chain() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
events = []

def probe(key):
    if False:
        events.append("private")
    elif True:
        if False:
            events.append("reserved")
        if key == "x":
            events.append("inner")
        elif key == "y":
            events.append("ignore")
    elif key == "x":
        events.append("duplicate")
    else:
        events.append("else")

probe("x")
EVENT_COUNT = len(events)
EVENT_0 = events[0] if len(events) > 0 else None
EVENT_1 = events[1] if len(events) > 1 else None
HAS_INNER = "inner" in events
HAS_DUPLICATE = "duplicate" in events
"#,
        &[],
    )
    .expect("nested elif control-flow probe should execute");

    assert_eq!(
        main.get_attr("HAS_INNER").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("HAS_DUPLICATE")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        main.get_attr("EVENT_COUNT")
            .and_then(|value| value.as_int()),
        Some(1)
    );
    assert!(
        main.get_attr("EVENT_0")
            .is_some_and(|value| value_is_python_string(value, "inner"))
    );
    assert!(
        main.get_attr("EVENT_1")
            .is_some_and(|value| value.is_none())
    );
}

#[test]
fn test_prepare_mapping_supports_inherited_generate_next_value_redefinition() {
    let result = execute(
        r#"
def _is_sunder(name):
    return (
        len(name) > 2 and
        name[0] == name[-1] == "_" and
        name[1:2] != "_" and
        name[-2:-1] != "_"
    )

class EnumLikeDict(dict):
    def __init__(self):
        super().__init__()
        self._member_names = {}
        self._ignore = []
        self._auto_called = False
        self._cls_name = ""

    def __setitem__(self, key, value):
        if _is_sunder(key):
            if key == "_generate_next_value_":
                if self._auto_called:
                    raise TypeError("_generate_next_value_ must be defined before members")
                _gnv = value.__func__ if isinstance(value, staticmethod) else value
                self._generate_next_value = _gnv
        elif key in self._member_names:
            raise TypeError("%r already defined as %r" % (key, self[key]))
        else:
            if key in self:
                raise TypeError("%r already defined as %r" % (key, self[key]))
        return super().__setitem__(key, value)

class Meta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        namespace = EnumLikeDict()
        namespace._cls_name = name
        first = bases[0] if bases else None
        if first is not None:
            namespace["_generate_next_value_"] = getattr(first, "_generate_next_value_", None)
        return namespace

class Base(metaclass=Meta):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count

class Derived(Base):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count + 1

assert Derived._generate_next_value_("x", 1, 2, []) == 3
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_class_function_lookup_on_type_does_not_inject_class_argument() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Example:
    def combine(a, b):
        return a + b

RESULT = Example.combine(2, 3)
"#,
        &[],
    )
    .expect("plain function lookup on class should execute");
    assert_eq!(
        main.get_attr("RESULT").and_then(|value| value.as_int()),
        Some(5)
    );
}

#[test]
fn test_metaclass_method_lookup_still_binds_class_argument() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Meta(type):
    def describe(cls):
        return cls.__name__

class Example(metaclass=Meta):
    pass

RESULT = Example.describe()
"#,
        &[],
    )
    .expect("metaclass method lookup should execute");
    assert!(value_is_python_string(
        main.get_attr("RESULT").expect("RESULT should be bound"),
        "Example"
    ));
}

#[test]
fn test_import_re_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import re
RESULT = re.escape("a+b")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
    assert!(value_is_python_string(
        result.expect("re import should return a value"),
        "a\\+b"
    ));
}

#[test]
fn test_cpython_re_compile_uses_native_sre_bridge() {
    let result = execute_with_cpython_lib(
        r#"
import re

pattern = re.compile(r"(?P<word>ab)+", re.I)
match = pattern.match("ABab")
assert match is not None
assert match.group("word") == "ab"
assert pattern.groupindex["word"] == 1
RESULT = pattern.flags
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
    assert_eq!(
        result.expect("pattern flags should be returned").as_int(),
        Some(RegexFlags::IGNORECASE as i64 | RegexFlags::UNICODE as i64)
    );
}

#[test]
fn test_direct_sre_compile_supports_cpython_parser_output() {
    let result = execute_with_cpython_lib(
        r#"
from re import _compiler

pattern = _compiler.compile(r"(?P<word>ab)+", 0)
match = pattern.match("abab")
assert match is not None
RESULT = match.group("word")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
    assert!(value_is_python_string(
        result.expect("_sre bridge should return a working pattern"),
        "ab"
    ));
}

#[test]
fn test_import_tokenize_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import tokenize
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_dataclasses_with_cpython_stdlib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import dataclasses
"#,
        120_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_locale_with_cpython_stdlib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import locale
"#,
        200_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_argparse_with_cpython_stdlib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import argparse
"#,
        200_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_unittest_with_cpython_stdlib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import unittest
"#,
        120_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_unittest_text_runner_executes_basic_testcase_with_cpython_stdlib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import unittest

class Smoke(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(True)

suite = unittest.defaultTestLoader.loadTestsFromTestCase(Smoke)
result = unittest.TextTestRunner(verbosity=0).run(suite)

assert result.wasSuccessful()
assert result.testsRun == 1
assert len(result.failures) == 0
assert len(result.errors) == 0
"#,
        200_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_unpack_sequence_uses_metaclass_iter_protocol() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Meta(type):
    def __iter__(cls):
        return iter((10, 20))

class Values(metaclass=Meta):
    pass

first, second = Values
"#,
        &[],
    )
    .expect("metaclass unpack should execute");

    assert_eq!(
        main.get_attr("first")
            .and_then(|value| value.as_int())
            .expect("first should be bound"),
        10
    );
    assert_eq!(
        main.get_attr("second")
            .and_then(|value| value.as_int())
            .expect("second should be bound"),
        20
    );
}

#[test]
fn test_class_iteration_prefers_metaclass_special_method_over_base_instance_iter() {
    let result = execute(
        r#"
class Meta(type):
    def __iter__(cls):
        return iter((10, 20))

class Base:
    def __iter__(self):
        raise AssertionError("instance __iter__ should not be used for class iteration")

class Example(Base, metaclass=Meta):
    pass

assert list(Example) == [10, 20]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_metaclass_super_new_preserves_namespace_entries() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Meta(type):
    def __new__(metacls, name, bases, namespace):
        namespace["_member_names_"] = 7
        namespace["answer"] = 42
        return super().__new__(metacls, name, bases, namespace)

class Dynamic(metaclass=Meta):
    pass
"#,
        &[],
    )
    .expect("metaclass super().__new__ should execute");

    let dynamic = main.get_attr("Dynamic").expect("Dynamic should exist");
    let dynamic_ptr = dynamic.as_object_ptr().expect("Dynamic should be a class");
    let dynamic_class =
        unsafe { &*(dynamic_ptr as *const prism_runtime::object::class::PyClassObject) };
    assert_eq!(
        dynamic_class
            .get_attr(&prism_core::intern::intern("answer"))
            .and_then(|value| value.as_int()),
        Some(42)
    );
    assert!(
        dynamic_class
            .get_attr(&prism_core::intern::intern("_member_names_"))
            .is_some(),
        "metaclass namespace entries should survive type.__new__"
    );
}

#[test]
fn test_metaclass_super_new_invokes_descriptor_set_name_hooks() {
    let result = execute(
        r#"
events = []

class Descriptor:
    def __set_name__(self, owner, name):
        events.append((owner.__name__, name))

class Meta(type):
    def __new__(metacls, name, bases, namespace):
        return super().__new__(metacls, name, bases, namespace)

class Example(metaclass=Meta):
    field = Descriptor()

assert events == [("Example", "field")]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_metaclass_super_new_exposes_fresh_class_mro_during_set_name_hooks() {
    let result = execute(
        r#"
events = []

class Descriptor:
    def __set_name__(self, owner, name):
        assert owner.__mro__[0] is owner
        assert owner.__mro__[1] is object
        events.append((owner.__name__, name))

class Meta(type):
    def __new__(metacls, name, bases, namespace):
        return super().__new__(metacls, name, bases, namespace)

class Example(metaclass=Meta):
    field = Descriptor()

assert events == [("Example", "field")]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_metaclass_super_new_preserves_descriptor_set_name_order() {
    let result = execute(
        r#"
events = []

class Descriptor:
    def __init__(self, token):
        self.token = token
    def __set_name__(self, owner, name):
        events.append((self.token, owner.__name__, name))

class Meta(type):
    def __new__(metacls, name, bases, namespace):
        return super().__new__(metacls, name, bases, namespace)

class Example(metaclass=Meta):
    first = Descriptor("first")
    second = Descriptor("second")
    third = Descriptor("third")

assert events == [
    ("first", "Example", "first"),
    ("second", "Example", "second"),
    ("third", "Example", "third"),
]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_metaclass_inherited_type_new_invokes_descriptor_set_name_hooks() {
    let result = execute(
        r#"
events = []

class Descriptor:
    def __set_name__(self, owner, name):
        events.append((owner.__name__, name))

class Meta(type):
    pass

class Example(metaclass=Meta):
    field = Descriptor()

assert events == [("Example", "field")]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_none_dunder_new_matches_cpython_behavior() {
    let result = execute(
        r#"
sentinel = object()
value = getattr(None, "__new__", sentinel)
assert value is not sentinel
assert value == None.__new__
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_getattr_default_handles_none_in_enum_find_new_pattern() {
    let result = execute(
        r#"
seen = []
for method in ("__new_member__", "__new__"):
    for possible in (str, None):
        target = getattr(possible, method, None)
        seen.append((possible is None, method, target is None))

assert seen == [
    (False, "__new_member__", True),
    (True, "__new_member__", True),
    (False, "__new__", False),
    (True, "__new__", False),
]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_heap_class_dunder_new_repeated_lookup_compares_equal() {
    let result = execute(
        r#"
class Example:
    def __new__(cls, value=None):
        return object.__new__(cls)

first = Example.__new__
second = Example.__new__
assert first == second
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_heap_class_dunder_new_repeated_lookup_is_hash_stable() {
    let result = execute(
        r#"
class Example:
    def __new__(cls, value=None):
        return object.__new__(cls)

first = Example.__new__
sentinels = {None, None.__new__, object.__new__, Example.__new__}
assert Example.__new__ in {first}
assert first in sentinels
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_type_three_arg_form_invokes_descriptor_set_name_hooks() {
    let result = execute(
        r#"
events = []

class Descriptor:
    def __set_name__(self, owner, name):
        events.append((owner.__name__, name))

Example = type("Example", (), {"field": Descriptor()})

assert events == [("Example", "field")]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_set_name_hooks_can_invoke_inherited_init_on_fresh_instances() {
    let result = execute(
        r#"
events = []

class Descriptor:
    def __set_name__(self, owner, name):
        instance = owner()
        assert callable(instance.__init__)
        instance.__init__()
        events.append((owner.__name__, name))

class Example:
    field = Descriptor()

assert events == [("Example", "field")]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_set_name_hooks_can_delete_class_attr_without_breaking_fresh_mro() {
    let result = execute(
        r#"
events = []

class Descriptor:
    def __set_name__(self, owner, name):
        delattr(owner, name)
        assert owner.__mro__[0] is owner
        assert owner.__mro__[1] is object
        events.append(name)

class Example:
    field = Descriptor()

assert events == ["field"]
assert hasattr(Example, "field") is False
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_str_subclass_instances_preserve_native_string_semantics() {
    let result = execute(
        r#"
class StringChild(str):
    pass

value = str.__new__(StringChild, "Seed")
value.extra = 7

assert type(value) is StringChild
assert value == "Seed"
assert "Seed" == value
assert hash(value) == hash("Seed")
assert len(value) == 4
assert value.lower() == "seed"
assert callable(value.__init__)
value.__init__()
assert value.extra == 7
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_metaclass_new_sees_fresh_str_subclasses_as_native_str_types() {
    let result = execute(
        r#"
SEEN_SUBTYPE = None
CONSTRUCTED = None

class Meta(type):
    def __new__(metacls, name, bases, namespace):
        cls = super().__new__(metacls, name, bases, namespace)
        global SEEN_SUBTYPE, CONSTRUCTED
        SEEN_SUBTYPE = issubclass(cls, str)
        value = str.__new__(cls, "Seed")
        value.__init__()
        CONSTRUCTED = (type(value) is cls, value == "Seed", callable(value.__init__))
        return cls

class StringChild(str, metaclass=Meta):
    pass

assert SEEN_SUBTYPE is True
assert CONSTRUCTED == (True, True, True)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_zero_arg_super_calls_parent_method() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Base:
    def value(self):
        return "base"

class Child(Base):
    def value(self):
        return super().value() + "-child"

RESULT = Child().value()
"#,
        &[],
    )
    .expect("super() regression program should execute");
    let result = main.get_attr("RESULT").expect("RESULT should be bound");
    assert!(value_is_python_string(result, "base-child"));
}

#[test]
fn test_explicit_super_calls_parent_method() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Base:
    def answer(self):
        return 41

class Child(Base):
    def answer(self):
        return super(Child, self).answer() + 1

RESULT = Child().answer()
"#,
        &[],
    )
    .expect("explicit super regression program should execute");
    let result = main.get_attr("RESULT").expect("RESULT should be bound");
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_super_exposes_standard_binding_attributes() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Base:
    pass

class Child(Base):
    def inspect(self):
        proxy = super()
        assert proxy.__self__ is self
        assert proxy.__self_class__ is Child
        assert proxy.__thisclass__ is Child
        return "ok"

RESULT = Child().inspect()
"#,
        &[],
    )
    .expect("super binding regression program should execute");
    let result = main.get_attr("RESULT").expect("RESULT should be bound");
    assert!(value_is_python_string(result, "ok"));
}

#[test]
fn test_metaclass_direct_call_uses_custom_new() {
    let (_, main) = execute_in_main_module_with_search_paths(
        r#"
class Meta(type):
    def __new__(metacls, name, bases, namespace):
        namespace["answer"] = 42
        return type.__new__(metacls, name, bases, namespace)

Dynamic = Meta("Dynamic", (), {})
"#,
        &[],
    )
    .expect("direct metaclass call should execute");

    let dynamic = main.get_attr("Dynamic").expect("Dynamic should exist");
    let dynamic_ptr = dynamic.as_object_ptr().expect("Dynamic should be a class");
    let dynamic_class =
        unsafe { &*(dynamic_ptr as *const prism_runtime::object::class::PyClassObject) };
    assert_eq!(
        dynamic_class
            .get_attr(&prism_core::intern::intern("answer"))
            .and_then(|value| value.as_int()),
        Some(42)
    );
}

#[test]
fn test_user_defined_class_custom_new_executes_before_init() {
    let result = execute(
        r#"
class Token:
    def __new__(cls, value):
        instance = object.__new__(cls)
        instance.created = value
        return instance

    def __init__(self, value):
        self.inited = value + 1

token = Token(4)
assert token.created == 4
assert token.inited == 5
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_user_defined_class_init_executes_body_before_returning_instance() {
    let result = execute(
        r#"
class Marker:
    ran = False

    def __init__(self):
        Marker.ran = True

Marker()
assert Marker.ran is True
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_class_body_bindings_preserve_distinct_local_slots() {
    let result = execute(
        r#"
class Probe:
    x = 1
    y = 2

assert Probe.x == 1
assert Probe.y == 2
assert Probe.__dict__["x"] == 1
assert Probe.__dict__["y"] == 2
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_class_method_bindings_do_not_alias_init_slot() {
    let result = execute(
        r#"
class Probe:
    def __init__(self, value):
        pass

    def f(self, other):
        return other

assert Probe.__dict__["__init__"](None, 9) is None
assert Probe.__dict__["f"](None, 7) == 7
assert isinstance(Probe(1), Probe)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_nested_class_definition_binds_into_outer_class_namespace() {
    let result = execute(
        r#"
class Outer:
    class Inner:
        token = 3

assert Outer.Inner.token == 3
assert Outer.__dict__["Inner"].token == 3
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_keyword_only_function_call_binds_and_executes() {
    let result = execute(
        r#"
def configure(*, maxlevel=6):
    return maxlevel

assert configure() == 6
assert configure(maxlevel=4) == 4
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_keyword_only_function_can_mutate_constructed_instance() {
    let result = execute(
        r#"
class Holder:
    pass

def configure(self, *, maxlevel=6):
    self.maxlevel = maxlevel

value = Holder()
configure(value)
assert value.maxlevel == 6
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_keyword_only_function_mutation_persists_in_main_module_instance() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
class Holder:
    pass

def configure(self, *, maxlevel=6):
    self.maxlevel = maxlevel

value = Holder()
configure(value)
"#,
        &[],
    )
    .expect("program should execute");

    let value = main.get_attr("value").expect("value binding should exist");
    let value_ptr = value
        .as_object_ptr()
        .expect("value binding should be an object pointer");
    let header = unsafe { &*(value_ptr as *const ObjectHeader) };
    assert!(
        header.type_id.raw() >= TypeId::FIRST_USER_TYPE,
        "expected a user-defined instance, got {:?}",
        header.type_id
    );

    let shaped =
        unsafe { &*(value_ptr as *const prism_runtime::object::shaped_object::ShapedObject) };
    assert_eq!(
        shaped
            .get_property("maxlevel")
            .and_then(|value| value.as_int()),
        Some(6),
        "instance mutation should persist on the constructed object"
    );
}

#[test]
fn test_user_defined_class_init_runs_with_keyword_only_defaults() {
    let result = execute(
        r#"
class ReprStyle:
    def __init__(self, *, maxlevel=6):
        self.maxlevel = maxlevel

value = ReprStyle()
assert value.maxlevel == 6
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_user_defined_class_init_binds_positional_and_keyword_args() {
    let result = execute(
        r#"
class Example:
    def __init__(self, left, *, right=2):
        self.total = left + right

value = Example(3, right=4)
assert value.total == 7
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_user_defined_class_init_varkw_preserves_string_keys() {
    let result = execute(
        r#"
class Example:
    def __init__(self, **kw):
        assert kw["x"] == 1
        assert "x" in kw
        self.kw = kw

value = Example(x=1)
assert value.kw["x"] == 1
assert "x" in value.kw
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_functools_partial_keyword_binding_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import functools

p = functools.partial(lambda **kw: kw["x"], x=1)
assert p() == 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_unittest_result_with_cpython_stdlib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import unittest.result
"#,
        120_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_method_call_uses_positional_default_after_binding_self() {
    let result = execute(
        r#"
class Probe:
    def value(self, amount=1):
        return amount

assert Probe().value() == 1
assert Probe().value(7) == 7
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_method_call_collects_varargs_after_binding_self() {
    let result = execute(
        r#"
class Probe:
    def collect(self, *args):
        return args

assert Probe().collect(1, 2, 3) == (1, 2, 3)
assert Probe().collect() == ()
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_method_call_uses_keyword_only_default_after_binding_self() {
    let result = execute(
        r#"
class Probe:
    def flag(self, *, enabled=True):
        return enabled

assert Probe().flag() is True
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_contextlib_contextmanager_decorator_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import contextlib

@contextlib.contextmanager
def token():
    yield 7
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_contextlib_contextmanager_exit_preserves_thrown_exception_identity() {
    let lib_dir = cpython_lib_dir();
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
import contextlib

events = []
INNER_SAME = None
OUTER_SAME = None
OUTER_TYPE = None
EVENTS_MATCH = None

@contextlib.contextmanager
def manager():
    global INNER_SAME
    try:
        yield
    except KeyError as exc:
        INNER_SAME = exc is marker
        events.append(exc)
        raise

marker = KeyError("boom")
try:
    with manager():
        raise marker
except KeyError as exc:
    OUTER_SAME = exc is marker
    OUTER_TYPE = type(exc).__name__
else:
    OUTER_TYPE = "missing"

EVENTS_MATCH = len(events) == 1 and events[0] is marker
"#,
        &[lib_dir.as_path()],
    )
    .expect("contextlib throw() probe should execute");

    assert_eq!(
        main.get_attr("INNER_SAME")
            .and_then(|value| value.as_bool()),
        Some(true),
        "generator handler should receive the original exception object",
    );
    assert_eq!(
        main.get_attr("OUTER_SAME")
            .and_then(|value| value.as_bool()),
        Some(true),
        "contextlib should re-raise the original exception object",
    );
    assert_eq!(
        main.get_attr("OUTER_TYPE").and_then(python_string_value),
        Some("KeyError".to_string()),
        "contextlib should re-raise KeyError after throw() propagation",
    );
    assert_eq!(
        main.get_attr("EVENTS_MATCH")
            .and_then(|value| value.as_bool()),
        Some(true),
        "the handler-observed exception should match the original object",
    );
}

#[test]
fn test_complex_constructor_matches_bool_numeric_semantics() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
ZERO_EQ_LITERAL = complex(False) == 0j
ZERO_EQ_BOOL = complex(False) == False
ONE_EQ_LITERAL = complex(True) == 1 + 0j
ONE_EQ_BOOL = complex(True) == True
ZERO_TEXT = str(0j)
ONE_TEXT = str(1 + 0j)
"#,
        &[],
    )
    .expect("complex numeric script should execute");

    assert_eq!(
        main.get_attr("ZERO_EQ_LITERAL")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("ZERO_EQ_BOOL")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("ONE_EQ_LITERAL")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("ONE_EQ_BOOL")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        main.get_attr("ZERO_TEXT").and_then(python_string_value),
        Some("0j".to_string())
    );
    assert_eq!(
        main.get_attr("ONE_TEXT").and_then(python_string_value),
        Some("(1+0j)".to_string())
    );
}

#[test]
fn test_complex_literals_bypass_shadowed_builtin_name() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
complex = lambda *args: 99
VALUE = 1 + 0j
VALUE_TEXT = str(VALUE)
VALUE_MATCH = VALUE == 1 + 0j
"#,
        &[],
    )
    .expect("complex literal script should execute");

    assert_eq!(
        main.get_attr("VALUE_TEXT").and_then(python_string_value),
        Some("(1+0j)".to_string())
    );
    assert_eq!(
        main.get_attr("VALUE_MATCH")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[test]
fn test_call_ex_keywords_preserve_locals_across_try_except() {
    let result = execute(
        r#"
def copy_metadata(wrapper, wrapped):
    for attr in ("__annotations__",):
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    assert wrapper is not None
    return wrapper

def wrapped():
    yield 1

def wrapper():
    return 1

args = (wrapper,)
kwargs = {"wrapped": wrapped}
assert copy_metadata(*args, **kwargs) is wrapper
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_call_ex_starargs_expands_generator_iterable() {
    let result = execute(
        r#"
def values():
    yield 1
    yield 2

def collect(*items):
    return items

assert collect(*values()) == (1, 2)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_functools_wraps_decorator_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import functools

def wrapped():
    yield 1

@functools.wraps(wrapped)
def wrapper():
    return 7

assert wrapper.__name__ == "wrapped"
assert wrapper.__wrapped__ is wrapped
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_reprlib_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import reprlib
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_dict_item_assignment_uses_compiler_set_item_encoding() {
    let result = execute(
        r#"
d = {}
d["answer"] = 42
assert d["answer"] == 42
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_sys_modules_mapping_supports_item_assignment() {
    let result = execute(
        r#"
import sys
sys.modules["prism_alias"] = sys
assert sys.modules["prism_alias"] is sys
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_frozenset_membership_uses_containment_fast_path() {
    let result = execute(
        r#"
flags = frozenset(["HAVE_LSTAT", "MS_WINDOWS"])
assert "HAVE_LSTAT" in flags
assert "chmod" not in flags
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_set_add_method_call_mutates_receiver() {
    let result = execute(
        r#"
seen = set()
seen.add("nt")
assert "nt" in seen
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_set_pop_method_call_returns_member_and_updates_membership() {
    let result = execute(
        r#"
seen = {"enum"}
value = seen.pop()
assert value == "enum"
assert "enum" not in seen
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_set_and_frozenset_support_subset_and_superset_comparisons() {
    let result = execute(
        r#"
left = {1, 2}
right = {1, 2, 3}
frozen = frozenset([1, 2])
same = frozenset([1, 2, 3])

assert left <= right
assert left < right
assert right >= left
assert right > frozen
assert frozen <= right
assert right >= same
assert not right < same
assert not left > right
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_string_upper_method_call_on_return_value() {
    let result = execute(
        r#"
def check_str(value):
    return value

assert check_str("Path").upper() == "PATH"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_function_call_preserves_returned_string_value() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
def check_str(value):
    return value

result = check_str("Path")
"#,
        &[],
    )
    .expect("function call should execute");

    let result = main
        .get_attr("result")
        .expect("result binding should exist");
    assert!(
        value_is_python_string(result, "Path"),
        "function should preserve returned string values, got {:?}",
        result
    );
}

#[test]
fn test_nested_function_reads_outer_cell_after_alias_assignment() {
    let result = execute(
        r#"
def outer():
    def check_str(value):
        return value

    encode = check_str

    def encodekey(key):
        return encode(key).upper()

    return encodekey("Path")

assert outer() == "PATH"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_nested_closure_cells_are_fresh_per_call() {
    let result = execute(
        r#"
def outer(value):
    def inner():
        return value

    return inner

first = outer("first")
second = outer("second")

assert outer.__closure__ is None
assert len(first.__closure__) == 1
assert len(second.__closure__) == 1
assert first.__closure__[0].cell_contents == "first"
assert second.__closure__[0].cell_contents == "second"
assert first() == "first"
assert second() == "second"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_functools_wraps_closures_keep_distinct_wrapped_functions() {
    let result = execute(
        r#"
import functools

def first():
    return "first"

def second():
    return "second"

def make_wrapper(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)
    return inner

a = make_wrapper(first)
b = make_wrapper(second)

assert a() == "first"
assert b() == "second"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_unittest_smoke_suite_runs_with_cpython_lib() {
    let result = execute_with_cpython_lib(
        r#"
import unittest

class Smoke(unittest.TestCase):
    def test_truth(self):
        self.assertTrue(True)

suite = unittest.defaultTestLoader.loadTestsFromTestCase(Smoke)
result = unittest.TestResult()
suite.run(result)
assert result.wasSuccessful()
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_super_init_subclass_resolves_object_hook_for_type_bindings() {
    let result = execute(
        r#"
class Base:
    seen = None

    def __init_subclass__(cls):
        cls.seen = cls.__name__
        super().__init_subclass__()

class Child(Base):
    pass

assert Child.seen == "Child"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_os_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import os
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_object_new_attribute_loads_as_builtin_function() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
value = object.__new__
"#,
        &[],
    )
    .expect("object.__new__ load should execute");

    let value = main.get_attr("value").expect("value binding should exist");
    let value_ptr = value
        .as_object_ptr()
        .expect("object.__new__ should be stored as an object pointer");
    let header = unsafe { &*(value_ptr as *const ObjectHeader) };
    assert_eq!(
        header.type_id,
        TypeId::BUILTIN_FUNCTION,
        "object.__new__ should resolve to a builtin function object",
    );
}

#[test]
fn test_object_new_class_decorator_creates_instance_sentinel() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
@object.__new__
class AllowMissing:
    pass
"#,
        &[],
    )
    .expect("object.__new__ class decorator should execute");

    let sentinel = main
        .get_attr("AllowMissing")
        .expect("decorated class binding should exist");
    let sentinel_ptr = sentinel
        .as_object_ptr()
        .expect("decorated class binding should be an object pointer");
    let header = unsafe { &*(sentinel_ptr as *const ObjectHeader) };
    assert!(
        header.type_id.raw() >= TypeId::FIRST_USER_TYPE,
        "decorated class binding should be an instance of the defined class",
    );
}

#[test]
fn test_loaded_object_new_callable_creates_object_instance() {
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
decorator = object.__new__
value = decorator(object)
"#,
        &[],
    )
    .expect("loaded object.__new__ call should execute");

    let value = main.get_attr("value").expect("value binding should exist");
    let value_ptr = value
        .as_object_ptr()
        .expect("object.__new__(object) should return an object instance");
    let header = unsafe { &*(value_ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::OBJECT);
}

#[test]
fn test_plain_python_class_decorator_executes() {
    let result = execute(
        r#"
def identity(value):
    return value

@identity
class Example:
    pass
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_keyword_binds_registered_module_object() {
    let lib_dir = cpython_lib_dir();
    let (vm, main) = execute_in_main_module_with_search_paths(
        r#"
import keyword
"#,
        &[lib_dir.as_path()],
    )
    .expect("keyword import should execute");

    let keyword_value = main
        .get_attr("keyword")
        .expect("keyword binding should exist");
    let keyword_ptr = keyword_value
        .as_object_ptr()
        .expect("keyword binding should be an object pointer");
    let keyword_module = vm
        .import_resolver
        .module_from_ptr(keyword_ptr)
        .expect("keyword binding should point at a registered module");
    assert_eq!(keyword_module.name(), "keyword");
}

#[test]
fn test_import_keyword_exposes_bound_builtin_iskeyword() {
    let lib_dir = cpython_lib_dir();
    let (vm, main) = execute_in_main_module_with_search_paths(
        r#"
import keyword
"#,
        &[lib_dir.as_path()],
    )
    .expect("keyword import should execute");

    let keyword_value = main
        .get_attr("keyword")
        .expect("keyword binding should exist");
    let keyword_ptr = keyword_value
        .as_object_ptr()
        .expect("keyword binding should be an object pointer");
    let keyword_module = vm
        .import_resolver
        .module_from_ptr(keyword_ptr)
        .expect("keyword binding should point at a registered module");

    let iskeyword = keyword_module
        .get_attr("iskeyword")
        .expect("keyword.iskeyword should exist");
    let iskeyword_ptr = iskeyword
        .as_object_ptr()
        .expect("keyword.iskeyword should be callable");
    let type_id = unsafe { (*(iskeyword_ptr as *const ObjectHeader)).type_id };
    assert_eq!(
        type_id,
        TypeId::BUILTIN_FUNCTION,
        "keyword.iskeyword should be a bound builtin method, got {}",
        type_id.name()
    );
}

#[test]
fn test_import_keyword_can_call_iskeyword_without_assert() {
    let result = execute_with_cpython_lib(
        r#"
import keyword

result = keyword.iskeyword("while")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_keyword_asserts_positive_iskeyword() {
    let result = execute_with_cpython_lib(
        r#"
import keyword

assert keyword.iskeyword("while")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_keyword_asserts_negative_iskeyword() {
    let result = execute_with_cpython_lib(
        r#"
import keyword

assert not keyword.iskeyword("prism_runtime")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_plain_builtin_class_attributes_remain_unbound() {
    let result = execute(
        r#"
class Probe:
    helper = len

assert Probe().helper("abc") == 3
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_random_module_binds_native_heaptype_methods() {
    let result = execute(
        r#"
import _random

rng = _random.Random()
seed = rng.seed
assert seed.__self__ is rng
seed(123)
assert rng.getstate() == 123
seed()
assert isinstance(rng.getstate(), int)
assert isinstance(rng.random(), float)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_random_stdlib_subclass_inherits_bound_native_methods() {
    let result = execute_with_cpython_lib(
        r#"
import random

rng = random.Random()
seed = rng.seed
assert seed.__self__ is rng
seed(123)
assert isinstance(rng.getstate(), tuple)
assert rng.getstate()[1] == 123
seed()
assert isinstance(rng.random(), float)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_sysconfig_supports_dict_union_operators() {
    let result = execute_with_cpython_lib(
        r#"
import sysconfig

paths = sysconfig.get_paths()
assert sysconfig.get_default_scheme() in sysconfig.get_scheme_names()
assert "stdlib" in paths
assert "platlib" in paths
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_class_body_delete_removes_class_binding() {
    let result = execute(
        r#"
class Namespace:
    token = 1
    del token

assert not hasattr(Namespace, "token")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_keyword_positive_call_stores_true_bool() {
    let lib_dir = cpython_lib_dir();
    let (_vm, main) = execute_in_main_module_with_search_paths(
        r#"
import keyword

result = keyword.iskeyword("while")
"#,
        &[lib_dir.as_path()],
    )
    .expect("keyword positive call should execute");

    let result = main
        .get_attr("result")
        .expect("result binding should exist");
    assert_eq!(
        result.as_bool(),
        Some(true),
        "expected bool True, got {result:?}"
    );
}

#[test]
fn test_test_support_package_exposes_path_and_import_helper_with_cpython_lib() {
    let result = execute_with_cpython_lib(
        r#"
import test.support as support
from test.support import import_helper, warnings_helper

assert hasattr(support, "__path__")
assert import_helper is not None
assert warnings_helper is not None
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_winreg_with_cpython_lib_bootstrap_surface() {
    if !cfg!(windows) {
        return;
    }

    let result = execute_with_cpython_lib(
        r#"
import winreg

assert winreg.HKEY_CURRENT_USER >= 0
assert winreg.HKEY_LOCAL_MACHINE >= 0
assert callable(winreg.OpenKey)
assert callable(winreg.QueryValue)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_textwrap_with_cpython_lib() {
    let result = execute_with_cpython_lib(
        r#"
import textwrap
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_collections_namedtuple_accepts_keyword_module_and_defaults() {
    let result = execute(
        r#"
from collections import namedtuple

Point = namedtuple("Point", "x y", module="demo.point", defaults=[7])
assert Point.__module__ == "demo.point"
assert Point._field_defaults["y"] == 7
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_unittest_with_cpython_lib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import unittest

loader = unittest.TestLoader()
assert loader.__class__.__name__ == "TestLoader"
"#,
        200_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_io_stringio_supports_core_text_buffer_protocol() {
    let result = execute(
        r#"
from io import StringIO

buffer = StringIO("seed")
assert buffer.getvalue() == "seed"
assert buffer.tell() == 0

buffer.seek(4)
assert buffer.write("!")
assert buffer.getvalue() == "seed!"

buffer.seek(0)
assert buffer.read() == "seed!"
buffer.seek(0)
assert buffer.readline() == "seed!"

buffer.seek(0)
buffer.truncate(2)
assert buffer.getvalue() == "se"
assert buffer.writable()
assert buffer.readable()
assert buffer.seekable()
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_io_memory_buffers_are_subclassable_from_python() {
    let result = execute(
        r#"
from io import BytesIO, StringIO

class TextBuffer(StringIO):
    pass

class BinaryBuffer(BytesIO):
    pass

text = TextBuffer("seed")
text.seek(4)
assert text.write("!")
assert text.getvalue() == "seed!"

binary = BinaryBuffer(b"seed")
binary.seek(4)
assert binary.write(b"!")
assert binary.getvalue() == b"seed!"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_super_binds_builtin_object_init_from_base_class() {
    let result = execute(
        r#"
class Base:
    pass

class Derived(Base):
    def __init__(self):
        super(Derived, self).__init__()
        self.ready = True

instance = Derived()
assert instance.ready is True
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_import_doctest_with_cpython_lib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import doctest

assert doctest._newline_convert("a\r\nb\rc\n") == "a\nb\nc\n"
"#,
        200_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_bound_builtin_callbacks_stored_on_classes_remain_unbound() {
    let result = execute(
        r#"
import re

class Parser:
    matcher = re.compile(r"^a+$").match

parser = Parser()
assert Parser.matcher("aaa") is not None
assert parser.matcher("aaa") is not None
assert parser.matcher("bbb") is None
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_doctest_finder_handles_named_group_stdlib_parsing() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import doctest
import test.test_listcomps as module

finder = doctest.DocTestFinder()
tests = finder.find(module)
assert len(tests) >= 1
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_decimal_bootstrap_uses_contextvars_for_localcontext() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import decimal

base = decimal.getcontext().prec
with decimal.localcontext() as ctx:
    ctx.prec = base + 2
    assert decimal.getcontext().prec == base + 2

assert decimal.getcontext().prec == base
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_large_i64_literals_survive_compile_and_execute() {
    let result = execute(
        r#"
value = 2305843009213693952
assert value == 2305843009213693952
assert value - 1 == 2305843009213693951
assert value + 1 == 2305843009213693953
assert -value == -2305843009213693952
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_bigint_literals_survive_compile_and_execute() {
    let result = execute(
        r#"
value = 1267650600228229401496703205376
assert value == 1267650600228229401496703205376
assert value - 1 == 1267650600228229401496703205375
assert value + 1 == 1267650600228229401496703205377
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_sys_hash_info_modulus_supports_bigint_arithmetic() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import sys

assert sys.hash_info.modulus - 2 == 2305843009213693949
assert sys.hash_info.modulus + 1 == 2305843009213693952
assert -sys.hash_info.modulus == -2305843009213693951
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_doctest_parser_blank_or_comment_callback_uses_regex_receiver() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import doctest

parser = doctest.DocTestParser()
assert parser._IS_BLANK_OR_COMMENT("   # comment") is not None
assert parser._IS_BLANK_OR_COMMENT("   ") is not None
assert parser._IS_BLANK_OR_COMMENT("value") is None
"#,
        200_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_class_bodies_seed_module_and_qualname_metadata() {
    let result = execute(
        r#"
class Outer:
    seen_module = __module__
    seen_qualname = __qualname__

    class Inner:
        seen_module = __module__
        seen_qualname = __qualname__

def factory():
    class Local:
        seen_module = __module__
        seen_qualname = __qualname__
    return Local

Local = factory()

assert Outer.__module__ == "__main__"
assert Outer.__qualname__ == "Outer"
assert Outer.seen_module == "__main__"
assert Outer.seen_qualname == "Outer"

assert Outer.Inner.__module__ == "__main__"
assert Outer.Inner.__qualname__ == "Outer.Inner"
assert Outer.Inner.seen_module == "__main__"
assert Outer.Inner.seen_qualname == "Outer.Inner"

assert Local.__module__ == "__main__"
assert Local.__qualname__ == "factory.<locals>.Local"
assert Local.seen_module == "__main__"
assert Local.seen_qualname == "factory.<locals>.Local"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_doctest_finder_reads_cpython_source_lines_as_text() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import inspect
import linecache
import re
import test.test_listcomps as module

source_file = inspect.getsourcefile(module.ListComprehensionTest)
if not isinstance(source_file, str):
    raise AssertionError(f"source_file={source_file!r} type={type(source_file).__name__}")

source_lines = linecache.getlines(source_file, module.__dict__)
if not isinstance(source_lines, list):
    raise AssertionError(f"source_lines_type={type(source_lines).__name__}")
if len(source_lines) <= 100:
    raise AssertionError(f"source_lines_len={len(source_lines)}")
line_types = [type(line).__name__ for line in source_lines[:10]]
if not all(isinstance(line, str) for line in source_lines[:10]):
    raise AssertionError(f"source_line_types={line_types!r}")

pat = re.compile(r'(^|.*:)\s*\w*("|\')')
for line in source_lines[:50]:
    pat.match(line)
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_doctest_finder_computes_lineno_for_cpython_class() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import doctest
import inspect
import linecache
import test.test_listcomps as module

finder = doctest.DocTestFinder()
source_file = inspect.getsourcefile(module.ListComprehensionTest)
source_lines = linecache.getlines(source_file, module.__dict__)

class_lineno = finder._find_lineno(module.ListComprehensionTest, source_lines)
if class_lineno is not None:
    raise AssertionError(f"class_lineno={class_lineno!r}")

load_tests_lineno = finder._find_lineno(module.load_tests, source_lines)
if not isinstance(load_tests_lineno, int):
    raise AssertionError(
        f"load_tests_lineno={load_tests_lineno!r} type={type(load_tests_lineno).__name__}"
    )
if load_tests_lineno < 0:
    raise AssertionError(f"load_tests_lineno={load_tests_lineno}")
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_doctest_finder_extracts_known_test_listcomps_members() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import doctest
import inspect
import linecache
import test.test_listcomps as module

finder = doctest.DocTestFinder()
source_file = inspect.getsourcefile(module)
source_lines = linecache.getlines(source_file, module.__dict__)
globs = module.__dict__.copy()

targets = [
    ("module", module),
    ("ListComprehensionTest", module.ListComprehensionTest),
    ("load_tests", module.load_tests),
    ("doctests", module.__test__["doctests"]),
]

for label, obj in targets:
    try:
        finder._get_test(obj, label, module, globs, source_lines)
    except Exception as exc:
        raise AssertionError(f"{label}: {type(exc).__name__}: {exc}")
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_doctest_parser_named_groups_produce_text_segments() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import doctest
import test.test_listcomps as module

doc = module.__test__["doctests"]
match = next(doctest.DocTestParser._EXAMPLE_RE.finditer(doc))

source = match.group("source")
want = match.group("want")
if not isinstance(source, str):
    raise AssertionError(f"source_type={type(source).__name__}")
if not isinstance(want, str):
    raise AssertionError(f"want_type={type(want).__name__}")

source_lines = source.split("\n")
want_lines = want.split("\n")
if not all(isinstance(line, str) for line in source_lines):
    raise AssertionError(f"source_line_types={[type(line).__name__ for line in source_lines]!r}")
if not all(isinstance(line, str) for line in want_lines):
    raise AssertionError(f"want_line_types={[type(line).__name__ for line in want_lines]!r}")
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_doctest_parser_parses_each_stdlib_listcomps_example() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import doctest
import test.test_listcomps as module

parser = doctest.DocTestParser()
doc = module.__test__["doctests"]
lineno = 0

for match in doctest.DocTestParser._EXAMPLE_RE.finditer(doc):
    try:
        parser._parse_example(match, "test.test_listcomps.__test__.doctests", lineno)
    except Exception as exc:
        raise AssertionError(f"lineno={lineno}: {type(exc).__name__}: {exc}")
    lineno += doc.count("\n", match.start(), match.end())
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_doctest_parser_replays_known_failing_example_steps() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import doctest
import re
import test.test_listcomps as module

doc = module.__test__["doctests"]
lineno = 0
target = None

for match in doctest.DocTestParser._EXAMPLE_RE.finditer(doc):
    if lineno == 21:
        target = match
        break
    lineno += doc.count("\n", match.start(), match.end())

assert target is not None

indent_text = target.group("indent")
source = target.group("source")
want = target.group("want")

if not isinstance(indent_text, str):
    raise AssertionError(f"indent_type={type(indent_text).__name__}")
if not isinstance(source, str):
    raise AssertionError(f"source_type={type(source).__name__}")
if not isinstance(want, str):
    raise AssertionError(f"want_type={type(want).__name__}")

indent = len(indent_text)
source_lines = source.split("\n")
want_lines = want.split("\n")
if not all(isinstance(line, str) for line in source_lines):
    raise AssertionError(f"source_line_types={[type(line).__name__ for line in source_lines]!r}")
if not all(isinstance(line, str) for line in want_lines):
    raise AssertionError(f"want_line_types={[type(line).__name__ for line in want_lines]!r}")

try:
    re.match(r" *$", want_lines[-1])
except Exception as exc:
    raise AssertionError(f"module_re_match: {type(exc).__name__}: {exc}")

stripped_source_lines = [line[indent + 4:] for line in source_lines]
stripped_want_lines = [line[indent:] for line in want_lines]
if not all(isinstance(line, str) for line in stripped_source_lines):
    raise AssertionError(
        f"stripped_source_line_types={[type(line).__name__ for line in stripped_source_lines]!r}"
    )
if not all(isinstance(line, str) for line in stripped_want_lines):
    raise AssertionError(
        f"stripped_want_line_types={[type(line).__name__ for line in stripped_want_lines]!r}"
    )

stripped_source = "\n".join(stripped_source_lines)
stripped_want = "\n".join(stripped_want_lines)
if not isinstance(stripped_source, str):
    raise AssertionError(f"stripped_source_type={type(stripped_source).__name__}")
if not isinstance(stripped_want, str):
    raise AssertionError(f"stripped_want_type={type(stripped_want).__name__}")

try:
    doctest.DocTestParser._EXCEPTION_RE.match(stripped_want)
except Exception as exc:
    raise AssertionError(f"exception_re_match: {type(exc).__name__}: {exc}")

try:
    option_matches = list(doctest.DocTestParser._OPTION_DIRECTIVE_RE.finditer(stripped_source))
except Exception as exc:
    raise AssertionError(f"option_finditer: {type(exc).__name__}: {exc}")

if option_matches:
    try:
        parser = doctest.DocTestParser()
        parser._IS_BLANK_OR_COMMENT(stripped_source)
    except Exception as exc:
        raise AssertionError(f"blank_or_comment: {type(exc).__name__}: {exc}")

parser = doctest.DocTestParser()
try:
    parser._check_prompt_blank(source_lines, indent, "test", lineno)
    parser._check_prefix(source_lines[1:], " " * indent + ".", "test", lineno)
    parser._check_prefix(want_lines, " " * indent, "test", lineno + len(source_lines))
except Exception as exc:
    raise AssertionError(f"parser_checks: {type(exc).__name__}: {exc}")

try:
    parser._parse_example(target, "test.test_listcomps.__test__.doctests", lineno)
except Exception as exc:
    raise AssertionError(f"parse_example: {type(exc).__name__}: {exc}")
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_inspect_bootstrap_reports_source_metadata_for_cpython_test_modules() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import inspect
import test.test_listcomps as module

assert inspect.ismodule(module)
assert inspect.getmodule(module) is module
assert inspect.isclass(module.ListComprehensionTest)
assert inspect.isfunction(module.load_tests)
assert inspect.getmodule(module.ListComprehensionTest) is module
assert inspect.getmodule(module.load_tests) is module
assert inspect.getfile(module).endswith("test_listcomps.py")
assert inspect.getsourcefile(module.ListComprehensionTest).endswith("test_listcomps.py")
assert inspect.getsourcefile(module.load_tests).endswith("test_listcomps.py")
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_unittest_loader_handles_doctest_load_tests_modules() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import unittest
import test.test_listcomps as module

suite = unittest.defaultTestLoader.loadTestsFromModule(module)
assert suite.countTestCases() >= 1
"#,
        300_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_sys_standard_streams_expose_text_runner_methods() {
    let result = execute(
        r#"
import sys

assert sys.stdout is sys.__stdout__
assert sys.stderr is sys.__stderr__
assert sys.stdin is sys.__stdin__

assert sys.stdout.write("") == 0
assert sys.stderr.write("") == 0
assert sys.stdout.flush() is None
assert sys.stderr.flush() is None
assert hasattr(sys.stdin, "readline")
assert sys.stdout.closed is False
assert sys.stderr.closed is False
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_open_context_manager_tracks_closed_state_like_cpython_bool_suite() {
    let result = execute_with_cpython_lib(
        r#"
from test.support import os_helper
import os

try:
    with open(os_helper.TESTFN, "w", encoding="utf-8") as f:
        assert f.closed is False
    assert f.closed is True
finally:
    if os.path.exists(os_helper.TESTFN):
        os.remove(os_helper.TESTFN)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_open_supports_text_round_trip_with_keyword_encoding() {
    let result = execute_with_cpython_lib(
        r#"
from test.support import os_helper
import os

try:
    with open(os_helper.TESTFN, "w", encoding="utf-8") as f:
        assert f.write("hello")
        assert f.closed is False

    with open(os_helper.TESTFN, "r", encoding="utf-8") as f:
        assert f.read() == "hello"
        assert f.readline() == ""
finally:
    if os.path.exists(os_helper.TESTFN):
        os.remove(os_helper.TESTFN)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_dir_lists_unittest_module_and_heap_testcase_attrs() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import unittest

class Smoke(unittest.TestCase):
    marker = 1

    def test_ok(self):
        pass

module_names = dir(unittest)
class_names = dir(Smoke)

assert "TestCase" in module_names
assert "TextTestRunner" in module_names
assert "marker" in class_names
assert "test_ok" in class_names
assert "assertTrue" in class_names
"#,
        120_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_unittest_loader_finds_testcase_methods_with_cpython_stdlib() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import unittest

class Smoke(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(True)

suite = unittest.defaultTestLoader.loadTestsFromTestCase(Smoke)
assert suite.countTestCases() == 1
"#,
        160_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_imported_class_method_call_preserves_defining_module_globals() {
    let temp_dir = unique_temp_dir("imported_class_method_globals");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("modprobe.py");
    fs::write(
        &module_path,
        r#"
from _abc import _abc_register

class Meta(type):
    def register(cls, subclass):
        return _abc_register

class Example(metaclass=Meta):
    pass
"#,
    )
    .expect("failed to write temp module");

    let result = execute_with_search_paths(
        r#"
import modprobe
Example = modprobe.Example
Example.register(None)
"#,
        &[temp_dir.as_path()],
    );

    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_source_defined_metaclass_isinstance_tracks_heap_metaclass_relationship() {
    let temp_dir = unique_temp_dir("source_defined_metaclass_isinstance");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("metaprobe.py");
    fs::write(
        &module_path,
        r#"
class Meta(type):
    pass

class Example(metaclass=Meta):
    pass
"#,
    )
    .expect("failed to write temp module");

    let result = execute_with_search_paths(
        r#"
import metaprobe

assert type(metaprobe.Example) is metaprobe.Meta
assert isinstance(metaprobe.Example, metaprobe.Meta)
assert isinstance(metaprobe.Meta, type)
"#,
        &[temp_dir.as_path()],
    );

    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_imported_class_direct_attribute_call_constructs_instance() {
    let temp_dir = unique_temp_dir("imported_class_direct_attribute_call");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    let module_path = temp_dir.join("callprobe.py");
    fs::write(
        &module_path,
        r#"
class C(dict):
    pass
"#,
    )
    .expect("failed to write temp module");

    let result = execute_with_search_paths(
        r#"
import callprobe

instance = callprobe.C()
assert type(instance) is callprobe.C
assert isinstance(instance, callprobe.C)
assert isinstance(instance, dict)
"#,
        &[temp_dir.as_path()],
    );

    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_unary_operators() {
    let sources = ["-5", "+5", "not True", "~15"];

    for source in sources {
        let result = execute(source);
        assert!(result.is_ok(), "Failed for '{}': {:?}", source, result);
    }
}

// =============================================================================
// String Tests
// =============================================================================

#[test]
fn test_string_literal() {
    let result = execute("\"hello\"");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_string_assignment() {
    let result = execute("s = \"world\"");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// None and Boolean Tests
// =============================================================================

#[test]
fn test_none_literal() {
    let result = execute("None");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_boolean_literals() {
    let result = execute("True");
    assert!(result.is_ok(), "True failed: {:?}", result);

    let result = execute("False");
    assert!(result.is_ok(), "False failed: {:?}", result);
}

#[test]
fn test_bool_type_rejects_subclassing_and_int_new_receiver() {
    let result = execute(
        r#"
try:
    class C(bool):
        pass
except TypeError as exc:
    BASE_ERROR = str(exc)
else:
    raise AssertionError("bool should not be subclassable")

try:
    int.__new__(bool, 0)
except TypeError as exc:
    NEW_ERROR = str(exc)
else:
    raise AssertionError("int.__new__(bool, 0) should fail")

assert BASE_ERROR.endswith("type 'bool' is not an acceptable base type")
assert NEW_ERROR.endswith("int.__new__(bool) is not safe, use bool.__new__()")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_marshal_round_trips_bool_like_cpython_bool_suite() {
    let result = execute(
        r#"
import marshal

assert marshal.loads(marshal.dumps(True)) is True
assert marshal.loads(marshal.dumps(False)) is False
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_bool_from_bytes_matches_cpython_semantics() {
    let result = execute(
        r#"
assert bool.from_bytes(b"\x00" * 8, "big") is False
assert bool.from_bytes(b"abcd", "little") is True
assert bool.from_bytes([], "big") is False
assert bool.from_bytes([0, 1], byteorder="big") is True
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_bool_real_and_imag_project_to_int_values() {
    let result = execute(
        r#"
assert True.real == 1
assert True.imag == 0
assert type(True.real) is int
assert type(True.imag) is int
assert False.real == 0
assert False.imag == 0
assert type(False.real) is int
assert type(False.imag) is int
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Complex Programs
// =============================================================================

#[test]
fn test_sum_to_n() {
    let result = execute(
        r#"
total = 0
n = 10
i = 1
while i <= n:
    total = total + i
    i = i + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_fibonacci() {
    let result = execute(
        r#"
a = 0
b = 1
n = 10
i = 0
while i < n:
    temp = a + b
    a = b
    b = temp
    i = i + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_nested_loops() {
    let result = execute(
        r#"
result = 0
i = 0
while i < 3:
    j = 0
    while j < 3:
        result = result + 1
        j = j + 1
    i = i + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Builtin Call Tests
// =============================================================================

#[test]
fn test_builtin_len_call() {
    // Test calling len() builtin
    let result = execute("x = len([1, 2, 3])");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_builtin_abs_call() {
    // Test calling abs() builtin
    let result = execute("x = abs(-5)");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_builtin_any_all_accept_generator_expressions() {
    let result = execute(
        r#"
assert any(x for x in [0, 0, 1])
assert not any(x for x in [0, 0, 0])
assert all(x for x in [1, 2, 3])
assert not all(x for x in [1, 0, 3])
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_builtin_sorted_and_sum_accept_generator_expressions() {
    let result = execute(
        r#"
assert sorted((x for x in [3, 1, 2])) == [1, 2, 3]
assert sorted((x for x in [3, 1, 2]), None, True) == [3, 2, 1]
assert sum(x for x in [1, 2, 3]) == 6
assert sum((x for x in [1, 2, 3]), 10) == 16
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_heap_list_subclass_inherits_core_list_protocols() {
    let result = execute(
        r#"
class L(list):
    pass

items = L()
assert not items
assert len(items) == 0
items.append(1)
items.append(2)
assert items
assert len(items) == 2
assert items[0] == 1
assert items[1] == 2
assert list(items) == [1, 2]
assert repr(items) == "[1, 2]"
items[1] = 5
assert list(items) == [1, 5]
del items[0]
assert list(items) == [5]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// For-Loop Tests
// =============================================================================

#[test]
fn test_for_loop_range() {
    // Test for-loop over range
    let result = execute(
        r#"
total = 0
for i in range(5):
    total = total + i
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_for_loop_list() {
    // Test for-loop over list
    let result = execute(
        r#"
total = 0
nums = [1, 2, 3, 4, 5]
for n in nums:
    total = total + n
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_nested_for_loops() {
    // Test nested for-loops
    let result = execute(
        r#"
result = 0
for i in range(3):
    for j in range(3):
        result = result + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_user_defined_descriptors_bind_on_class_and_instance_access() {
    let result = execute(
        r#"
class Descriptor:
    def __init__(self, label):
        self.label = label

    def __get__(self, instance, owner):
        if instance is None:
            return ('class', owner.__name__, self.label)
        return ('instance', owner.__name__, self.label)

class Example:
    value = Descriptor('token')

assert Example.value == ('class', 'Example', 'token')
assert Example().value == ('instance', 'Example', 'token')
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_user_defined_data_descriptors_support_get_set_and_delete() {
    let result = execute(
        r#"
events = []

class Descriptor:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        events.append(('get', owner.__name__))
        return 'managed'

    def __set__(self, instance, value):
        events.append(('set', value))

    def __delete__(self, instance):
        events.append(('delete', type(instance).__name__))

class Example:
    value = Descriptor()

example = Example()
example.value = 10
assert example.value == 'managed'
del example.value
assert events == [('set', 10), ('get', 'Example'), ('delete', 'Example')]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_wrapped_descriptor_inside_classmethod_matches_cpython() {
    let result = execute(
        r#"
class BoundWrapper:
    def __init__(self, wrapped):
        self.__wrapped__ = wrapped

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

class Wrapper:
    def __init__(self, wrapped):
        self.__wrapped__ = wrapped

    def __get__(self, instance, owner):
        bound_function = self.__wrapped__.__get__(instance, owner)
        return BoundWrapper(bound_function)

def decorator(wrapped):
    return Wrapper(wrapped)

class Example:
    @decorator
    @classmethod
    def inner(cls):
        return 'spam'

    @classmethod
    @decorator
    def outer(cls):
        return 'eggs'

assert Example.inner() == 'spam'
assert Example.outer() == 'eggs'
assert Example().inner() == 'spam'
assert Example().outer() == 'eggs'
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_bound_function_inside_classmethod_matches_cpython() {
    let result = execute(
        r#"
class A:
    def foo(self, cls):
        return 'spam'

class B:
    bar = classmethod(A().foo)

assert B.bar() == 'spam'
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_eval_defaults_to_current_frame_locals_for_varargs_and_kwargs() {
    let result = execute(
        r#"
def outer():
    def check(*args, **kwds):
        return (
            eval("args[1] is not None", None, None),
            eval("kwds['token']", None, None),
        )

    return check(1, 2, token=3)

assert outer() == (True, 3)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_eval_compiled_code_object_uses_current_frame_locals_with_explicit_none_namespaces() {
    let result = execute(
        r#"
expr = compile("args[1] is not None", "<probe>", "eval")

def check(*args, **kwds):
    return eval(expr, None, None)

assert check(1, 2, token=3) is True
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_types_method_type_matches_cpython_binding_contract() {
    let result = execute_with_cpython_lib(
        r#"
from types import MethodType

class C:
    def f(self, value):
        return (self, value)

obj = C()
bound = MethodType(C.f, obj)

assert bound.__self__ is obj
assert bound.__func__ is C.f
assert bound(7) == (obj, 7)

try:
    MethodType(C.f, None)
except TypeError as exc:
    assert str(exc) == "instance must not be None"
else:
    raise AssertionError("MethodType should reject None instance")

try:
    MethodType(42, obj)
except TypeError as exc:
    assert str(exc) == "first argument must be callable"
else:
    raise AssertionError("MethodType should reject non-callable receiver")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_types_method_type_preserves_raw_exception_payloads() {
    let result = execute_with_cpython_lib(
        r#"
from types import MethodType

class C:
    def f(self):
        return self

obj = C()

try:
    MethodType(C.f, None)
except TypeError as exc:
    assert str(exc) == "instance must not be None"
    assert exc.args == ("instance must not be None",)
else:
    raise AssertionError("MethodType(None) should raise TypeError")

try:
    MethodType(42, obj)
except TypeError as exc:
    assert str(exc) == "first argument must be callable"
    assert exc.args == ("first argument must be callable",)
else:
    raise AssertionError("MethodType(noncallable) should raise TypeError")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_types_method_type_wraps_callable_instances() {
    let result = execute_with_cpython_lib(
        r#"
from types import MethodType

class Callable:
    def __call__(self, owner, value):
        return (owner, value)

callable_obj = Callable()
owner = object()
bound = MethodType(callable_obj, owner)

assert bound.__self__ is owner
assert bound.__func__ is callable_obj
assert bound(9) == (owner, 9)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_heap_exception_subclass_can_call_base_exception_init_explicitly() {
    let result = execute(
        r#"
class DbcheckError(Exception):
    def __init__(self, exprstr, func, args, kwds):
        Exception.__init__(
            self,
            "dbcheck %r failed (func=%s args=%s kwds=%s)" % (exprstr, func, args, kwds),
        )

err = DbcheckError("x", "f", (1, None), {})

assert err.args == ("dbcheck 'x' failed (func=f args=(1, None) kwds={})",)
assert str(err) == "dbcheck 'x' failed (func=f args=(1, None) kwds={})"
assert repr(err) == "DbcheckError('dbcheck \\'x\\' failed (func=f args=(1, None) kwds={})')"
assert BaseException.__str__(err) == str(err)
assert BaseException.__repr__(err) == repr(err)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_base_exception_class_surface_exposes_unbound_base_methods() {
    let result = execute(
        r#"
str_method = BaseException.__str__
repr_method = BaseException.__repr__

assert str_method.__qualname__ == "BaseException.__str__"
assert str_method.__self__ is None
assert repr_method.__qualname__ == "BaseException.__repr__"
assert repr_method.__self__ is None

err = Exception("boom")
assert str_method(err) == "boom"
assert repr_method(err) == "Exception('boom')"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_heap_exception_subclass_new_initializes_empty_base_state() {
    let result = execute(
        r#"
class CustomError(Exception):
    def __init__(self, marker):
        self.marker = marker

err = CustomError(7)

assert err.marker == 7
assert err.args == ()
assert str(err) == ""
assert repr(err) == "CustomError()"
assert err.__traceback__ is None
assert err.__cause__ is None
assert err.__context__ is None
assert err.__suppress_context__ is False
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_heap_exception_subclass_with_traceback_round_trips() {
    let result = execute(
        r#"
class CustomError(Exception):
    pass

try:
    raise ValueError("boom")
except ValueError as exc:
    tb = exc.__traceback__

err = CustomError("payload").with_traceback(tb)

assert err.args == ("payload",)
assert err.__traceback__ is tb
assert str(err) == "payload"
assert repr(err) == "CustomError('payload')"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_wrapped_classmethod_inside_classmethod_matches_cpython() {
    let result = execute_with_cpython_lib(
        r#"
from types import MethodType

class MyClassMethod1:
    def __init__(self, func):
        self.func = func

    def __call__(self, cls):
        if hasattr(self.func, '__get__'):
            return self.func.__get__(cls, cls)()
        return self.func(cls)

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return MethodType(self, owner)

class MyClassMethod2:
    def __init__(self, func):
        if isinstance(func, classmethod):
            func = func.__func__
        self.func = func

    def __call__(self, cls):
        return self.func(cls)

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return MethodType(self, owner)

for myclassmethod in [MyClassMethod1, MyClassMethod2]:
    class A:
        @myclassmethod
        def f1(cls):
            return cls

        @classmethod
        @myclassmethod
        def f2(cls):
            return cls

        @myclassmethod
        @classmethod
        def f3(cls):
            return cls

        @classmethod
        @classmethod
        def f4(cls):
            return cls

        @myclassmethod
        @MyClassMethod1
        def f5(cls):
            return cls

        @myclassmethod
        @MyClassMethod2
        def f6(cls):
            return cls

    assert A.f1() is A
    assert A.f2() is A
    assert A.f3() is A
    assert A.f4() is A
    assert A.f5() is A
    assert A.f6() is A

    a = A()
    assert a.f1() is A
    assert a.f2() is A
    assert a.f3() is A
    assert a.f4() is A
    assert a.f5() is A
    assert a.f6() is A

    def f(cls):
        return cls

    assert myclassmethod(f).__get__(a)() is A
    assert myclassmethod(f).__get__(a, A)() is A
    assert myclassmethod(f).__get__(A, A)() is A
    assert myclassmethod(f).__get__(A)() is type(A)
    assert classmethod(f).__get__(a)() is A
    assert classmethod(f).__get__(a, A)() is A
    assert classmethod(f).__get__(A, A)() is A
    assert classmethod(f).__get__(A)() is type(A)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}
