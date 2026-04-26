use super::*;

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
fn test_int_from_bytes_consumes_lazy_python_iterables_with_vm_context() {
    let result = execute(
        r#"
assert int.from_bytes(map(int, ["1", "2", "3", "4"]), "big") == 16909060
assert bool.from_bytes(map(int, ["0"]), "big") is False
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_int_constructor_uses_python_numeric_conversion_protocols() {
    let result = execute(
        r#"
class IntLike:
    def __int__(self):
        return 3221225985

class IndexLike:
    def __index__(self):
        return 33

class TruncatedIndex:
    def __index__(self):
        return 44

class TruncLike:
    def __trunc__(self):
        return TruncatedIndex()

assert int(IntLike()) == 3221225985
assert int(IndexLike()) == 33
assert int(TruncLike()) == 44
assert int("ff", base=16) == 255
assert int(memoryview(b"123")) == 123
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_int_constructor_rejects_invalid_numeric_protocol_results() {
    let result = execute(
        r#"
class BadInt:
    def __int__(self):
        return "not an int"

class BadIndex:
    def __index__(self):
        return "not an int"

class BadTrunc:
    def __trunc__(self):
        return object()

for cls, expected in [
    (BadInt, "__int__ returned non-int"),
    (BadIndex, "__index__ returned non-int"),
    (BadTrunc, "__trunc__ returned non-Integral"),
]:
    try:
        int(cls())
    except TypeError as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError(expected)
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
fn test_generator_throw_accepts_heap_exception_instance() {
    let result = execute(
        r#"
class CustomSkip(Exception):
    pass

def gen():
    try:
        yield "ready"
    except CustomSkip as exc:
        yield (str(exc), isinstance(exc, CustomSkip), isinstance(exc, Exception))

g = gen()
assert next(g) == "ready"
assert g.throw(CustomSkip("payload")) == ("payload", True, True)
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_generator_throw_accepts_heap_exception_class_and_instance() {
    let result = execute(
        r#"
class CustomSkip(Exception):
    pass

def gen():
    try:
        yield "ready"
    except CustomSkip as exc:
        yield exc

thrown = CustomSkip("payload")
g = gen()
assert next(g) == "ready"
assert g.throw(CustomSkip, thrown) is thrown
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_cpython_unittest_skiptest_uses_heap_exception_throw_protocol() {
    let result = execute_with_cpython_lib(
        r#"
import unittest

class Case(unittest.TestCase):
    def runTest(self):
        self.skipTest("threading compatibility")

result = unittest.TestResult()
Case().run(result)

if not result.wasSuccessful():
    raise RuntimeError(
        f"errors={len(result.errors)} failures={len(result.failures)} skipped={len(result.skipped)}"
    )
assert len(result.skipped) == 1
assert result.skipped[0][1] == "threading compatibility"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_bound_generator_method_with_keywords_returns_generator_object() {
    let result = execute(
        r#"
class Producer:
    def values(self, first, *, second=2):
        yield first
        yield second

producer = Producer()
generated = producer.values(1, second=3)
assert list(generated) == [1, 3]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_cpython_traceback_exception_format_generator_method_with_keywords() {
    let result = execute_with_cpython_lib(
        r#"
import traceback

try:
    raise AssertionError("x")
except BaseException as exc:
    rendered = list(traceback.TracebackException(
        type(exc),
        exc,
        exc.__traceback__,
        compact=True,
    ).format(chain=True))

assert rendered[0] == "Traceback (most recent call last):\n"
assert rendered[-1] == "AssertionError: x\n"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_native_functools_partial_keeps_callable_attributes_as_state() {
    let result = execute(
        r#"
from _functools import partial

class CallableWithPartialNames:
    def __init__(self):
        self.func = self
        self.args = object()
        self.keywords = object()
        self.calls = 0

    def __call__(self):
        self.calls += 1

target = CallableWithPartialNames()
callback = partial(target)

assert callback.func is target
assert callback.args == ()
callback()
assert target.calls == 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_native_functools_partial_merges_flattened_args_and_keywords() {
    let result = execute(
        r#"
from _functools import partial

def combine(a, b, *, c=0):
    return a + b + c

first = partial(combine, 2, c=5)
second = partial(first, 3, c=7)

assert second.func is combine
assert second.args == (2, 3)
assert second() == 12
assert first(4, c=9) == 15
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_cpython_functools_import_uses_native_partial_for_mock_callbacks() {
    let result = execute_with_cpython_lib(
        r#"
import _functools
from functools import partial
from unittest.mock import Mock

assert partial is _functools.partial

mock = Mock()
callback = partial(mock)
assert callback.func is mock
assert callback.args == ()

mock.assert_not_called()
callback()
mock.assert_called_once()
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
