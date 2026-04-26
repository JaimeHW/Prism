use super::*;

// =============================================================================
// VM Builtin Tests
// =============================================================================

#[test]
fn test_vm_builtins_are_initialized() {
    let vm = VirtualMachine::new();

    // Verify len builtin exists and is an object_ptr
    let len_val = vm.builtin_value("len").expect("len should exist");
    assert!(
        len_val.as_object_ptr().is_some(),
        "len should be object_ptr, got: bits = {:#x}",
        unsafe { std::mem::transmute::<Value, u64>(len_val) }
    );

    // Verify range builtin
    let range_val = vm.builtin_value("range").expect("range should exist");
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
fn test_platform_win32_ver_uses_native_windows_version_and_registry_data() {
    let result = execute_with_cpython_lib(
        r#"
import os
import platform
import sys

release, version, csd, ptype = platform.win32_ver('a', 'b', 'c', 'd')
if sys.platform == 'win32':
    assert release != 'a'
    assert version and all(part.isdigit() for part in version.split('.'))
    assert not csd or csd.startswith('SP')
    assert not ptype or 'Multiprocessor' in ptype or 'Uniprocessor' in ptype
    count = os.cpu_count()
    assert count is None or count >= 1
else:
    assert (release, version, csd, ptype) == ('a', 'b', 'c', 'd')
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
fn test_class_body_loads_later_method_names_from_globals_until_assigned() {
    let result = execute(
        r#"
class date:
    __slots__ = ("date_slot",)

class time:
    __slots__ = ("time_slot",)

class datetime(date):
    __slots__ = date.__slots__ + time.__slots__

    def date(self):
        return "method-date"

    def time(self):
        return "method-time"

assert datetime.__slots__ == ("date_slot", "time_slot")
assert datetime.date(None) == "method-date"
assert datetime.time(None) == "method-time"
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
fn test_builtin_container_instances_expose_class_attribute_to_python() {
    let result = execute(
        r#"
assert [].__class__ is list
assert {}.__class__ is dict
assert ().__class__ is tuple
assert {1}.__class__ is set
assert b"abc".__class__ is bytes
assert bytearray(b"abc").__class__ is bytearray
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
