use super::*;

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
assert sys.stdout.buffer.write(b"") == 0
assert sys.stderr.buffer.write(b"") == 0
assert sys.stdout.flush() is None
assert sys.stderr.flush() is None
assert sys.stdout.buffer.flush() is None
assert sys.stderr.buffer.flush() is None
assert hasattr(sys.stdin, "readline")
assert hasattr(sys.stdin, "buffer")
assert hasattr(sys.stdin.buffer, "read")
assert sys.stdin.buffer.readable() is True
assert sys.stdout.buffer.writable() is True
assert sys.stderr.buffer.writable() is True
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

