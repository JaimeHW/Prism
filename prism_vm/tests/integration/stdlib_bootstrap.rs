use super::*;

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
fn test_comprehension_reads_explicit_global_without_creating_cell() {
    let result = execute(
        r#"
seed = [10]
cache = None

def build():
    global cache, seed
    if cache is None:
        cache = [x + seed[0] for x in range(2)]
    return cache[0]

assert build() == 10
assert build() == 10
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
fn test_os_fspath_uses_pathlike_protocol_with_cpython_stdlib() {
    let result = execute_with_cpython_lib(
        r#"
import os

assert os.fspath("alpha") == "alpha"
assert os.fspath(b"alpha") == b"alpha"

class PathLike:
    def __fspath__(self):
        return "beta"

assert os.fspath(PathLike()) == "beta"

class BadReturn:
    def __fspath__(self):
        return 123

try:
    os.fspath(BadReturn())
except TypeError as exc:
    assert "__fspath__" in str(exc)
else:
    raise AssertionError("expected TypeError for invalid __fspath__ result")

class InstanceOnly:
    pass

instance = InstanceOnly()
instance.__fspath__ = lambda: "ignored"
try:
    os.fspath(instance)
except TypeError:
    pass
else:
    raise AssertionError("instance __fspath__ must not satisfy the path protocol")
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

