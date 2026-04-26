use super::*;

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

