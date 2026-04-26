use super::*;

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
fn test_dict_constructor_and_update_accept_mapping_protocol_objects() {
    let result = execute_with_cpython_lib(
        r#"
import os

class MappingOnly:
    def __init__(self):
        self.values = {"alpha": 1, "beta": 2}

    def keys(self):
        return self.values.keys()

    def __getitem__(self, key):
        return self.values[key]

mapping = MappingOnly()
assert dict(mapping) == {"alpha": 1, "beta": 2}

target = {"alpha": 0}
target.update(mapping)
assert target == {"alpha": 1, "beta": 2}

copied = os.environ.copy()
assert isinstance(copied, dict)
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
fn test_namedtuple_subclass_custom_new_does_not_reinitialize_tuple_payload() {
    let result = execute_with_cpython_lib(
        r#"
from collections import namedtuple

Base = namedtuple("Base", "nid shortname longname oid")

class ASN1Object(Base):
    __slots__ = ()

    def __new__(cls, oid):
        return Base.__new__(cls, 129, "serverAuth", "TLS Web Server Authentication", oid)

value = ASN1Object("1.3.6.1.5.5.7.3.1")
assert value.nid == 129
assert value.shortname == "serverAuth"
assert value.longname == "TLS Web Server Authentication"
assert value.oid == "1.3.6.1.5.5.7.3.1"
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
fn test_cpython_base64_uses_native_binascii() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import base64
import binascii
from array import array
from io import BytesIO

assert binascii.hexlify(b"\xb9\x01\xef") == b"b901ef"
assert binascii.hexlify(array("B", b"\x01\x02\xef")) == b"0102ef"
assert b"==payload==\n".rstrip(b"=\n") == b"==payload"
assert bytearray(b"\tdata\n").strip() == bytearray(b"data")
assert b"abca".translate(bytes.maketrans(b"a", b"z")) == b"zbcz"
assert b"abc".translate(None, b"b") == b"ac"
assert bytes.maketrans(array("B", b"+/"), array("B", b"-_"))[ord("+")] == ord("-")
assert base64.b64encode(b"hello") == b"aGVsbG8="
assert base64.b64decode(b"aGVsbG8=") == b"hello"
assert base64.b16encode(array("B", b"\x01\x02\xef")) == b"0102EF"
assert array("B", b"abcdef")[1:5:2].tobytes() == b"bd"
assert base64.encodebytes(array("B", b"abc")) == b"YWJj\n"
outfp = BytesIO()
base64.decode(BytesIO(b"d3d3LnB5dGhvbi5vcmc="), outfp)
assert outfp.getvalue() == b"www.python.org"
try:
    base64.decodebytes(memoryview(b"1234").cast("B", (2, 2)))
    raise AssertionError("multidimensional memoryview should be rejected")
except TypeError:
    pass
"#,
        200_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_memoryview_cast_preserves_shape_metadata() {
    let source = r#"
view = memoryview(b"1234").cast("B", (2, 2))
assert view.ndim == 2
assert view.shape == (2, 2)
assert view.strides == (2, 1)
assert memoryview(view).ndim == 2
assert memoryview(view).shape == (2, 2)
"#;

    execute(source).unwrap();
}

#[test]
fn test_named_expression_in_while_condition_assigns_loop_value() {
    let source = r#"
calls = 0
seen = []

def read():
    global calls
    calls = calls + 1
    if calls == 1:
        return b"payload"
    return b""

while line := read():
    seen.append(line)

assert seen == [b"payload"]
assert line == b""
assert calls == 2
"#;

    execute(source).unwrap();
}

#[test]
fn test_cpython_http_server_imports_with_native_http_stack() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
from http.server import HTTPServer

assert HTTPServer.__name__ == "HTTPServer"
"#,
        12_000_000,
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
