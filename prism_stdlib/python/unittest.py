"""Small unittest compatibility layer for Prism's CPython regression harness."""

try:
    import _warnings as _prism_warnings
except Exception:
    _prism_warnings = None


class SkipTest(Exception):
    pass


def skip(reason):
    def decorate(obj):
        setattr(obj, "__unittest_skip__", True)
        setattr(obj, "__unittest_skip_why__", reason)
        return obj

    return decorate


class _Outcome:
    def __init__(self):
        self.failures = []
        self.errors = []
        self.skipped = []

    def addFailure(self, test, err):
        self.failures.append((test, err))

    def addError(self, test, err):
        self.errors.append((test, err))

    def addSkip(self, test, reason):
        self.skipped.append((test, reason))

    def wasSuccessful(self):
        return len(self.failures) == 0 and len(self.errors) == 0


class _AssertRaisesContext:
    def __init__(self, expected, test_case, expected_regex=None):
        self.expected = expected
        self.test_case = test_case
        self.expected_regex = expected_regex
        self.exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.test_case.fail("expected exception was not raised")
        if not issubclass(exc_type, self.expected):
            return False
        if self.expected_regex is not None and self.expected_regex not in str(exc):
            self.test_case.fail("exception message did not match")
        self.exception = exc
        return True


class _AssertWarnsContext:
    def __init__(self, expected, test_case):
        self.expected = expected
        self.test_case = test_case
        self.warning = None

    def __enter__(self):
        if _prism_warnings is not None:
            _prism_warnings._prism_begin_capture(self.expected)
        return self

    def __exit__(self, exc_type, exc, tb):
        matched = 0
        if _prism_warnings is not None:
            matched = _prism_warnings._prism_end_capture()
        if exc_type is not None:
            return False
        if matched <= 0:
            self.test_case.fail("expected warning was not triggered")
        return False


class _SubTestContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestCase:
    failureException = AssertionError

    def __init__(self, methodName="runTest"):
        self._testMethodName = methodName
        self._cleanups = []

    def __call__(self):
        return self.run()

    def __repr__(self):
        return type(self).__name__ + "." + self._testMethodName

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def addCleanup(self, function, *args):
        self._cleanups.append((function, args))

    def doCleanups(self):
        ok = True
        while self._cleanups:
            function, args = self._cleanups.pop()
            function(*args)
        return ok

    def run(self, result=None):
        if result is None:
            result = _Outcome()

        if getattr(type(self), "__unittest_skip__", False):
            result.addSkip(self, getattr(type(self), "__unittest_skip_why__", ""))
            return result

        method = getattr(self, self._testMethodName)
        if getattr(method, "__unittest_skip__", False):
            result.addSkip(self, getattr(method, "__unittest_skip_why__", ""))
            return result

        try:
            self.setUp()
            method()
            self.tearDown()
        except SkipTest as exc:
            result.addSkip(self, str(exc))
        except AssertionError as exc:
            result.addFailure(self, exc)
        except Exception as exc:
            result.addError(self, exc)
        finally:
            try:
                self.doCleanups()
            except Exception as exc:
                result.addError(self, exc)

        return result

    def fail(self, msg=None):
        if msg is None:
            msg = "test failed"
        raise self.failureException(msg)

    def assertTrue(self, expr, msg=None):
        if not expr:
            self.fail(msg or "expression is not true")

    def assertFalse(self, expr, msg=None):
        if expr:
            self.fail(msg or "expression is not false")

    def assertEqual(self, first, second, msg=None):
        if not first == second:
            self.fail(msg or (repr(first) + " != " + repr(second)))

    def assertNotEqual(self, first, second, msg=None):
        if first == second:
            self.fail(msg or (repr(first) + " == " + repr(second)))

    def assertListEqual(self, first, second, msg=None):
        self.assertEqual(first, second, msg)

    def assertSequenceEqual(self, first, second, msg=None):
        self.assertEqual(first, second, msg)

    def assertIs(self, first, second, msg=None):
        if first is not second:
            self.fail(msg or (repr(first) + " is not " + repr(second)))

    def assertIsNot(self, first, second, msg=None):
        if first is second:
            self.fail(msg or (repr(first) + " is " + repr(second)))

    def assertIsInstance(self, obj, cls, msg=None):
        if not isinstance(obj, cls):
            self.fail(msg or "object is not an instance")

    def assertNotIsInstance(self, obj, cls, msg=None):
        if isinstance(obj, cls):
            self.fail(msg or "object is an instance")

    def assertGreaterEqual(self, first, second, msg=None):
        if not first >= second:
            self.fail(msg or (repr(first) + " is less than " + repr(second)))

    def assertIn(self, member, container, msg=None):
        if member not in container:
            self.fail(msg or (repr(member) + " not found"))

    def assertNotIn(self, member, container, msg=None):
        if member in container:
            self.fail(msg or (repr(member) + " unexpectedly found"))

    def assertRaises(self, expected_exception, callable_obj=None, *args):
        context = _AssertRaisesContext(expected_exception, self)
        if callable_obj is None:
            return context
        with context:
            callable_obj(*args)
        return context

    def assertRaisesRegex(self, expected_exception, expected_regex, callable_obj=None, *args):
        context = _AssertRaisesContext(expected_exception, self, expected_regex)
        if callable_obj is None:
            return context
        with context:
            callable_obj(*args)
        return context

    def assertWarns(self, expected_warning):
        return _AssertWarnsContext(expected_warning, self)

    def subTest(self, msg=None, **params):
        return _SubTestContext()


class TestSuite:
    def __init__(self, tests=None):
        self._tests = []
        if tests is not None:
            for test in tests:
                self.addTest(test)

    def addTest(self, test):
        self._tests.append(test)

    def __iter__(self):
        return iter(self._tests)


class TestLoader:
    testMethodPrefix = "test"

    def loadTestsFromTestCase(self, testCaseClass):
        suite = TestSuite()
        for name in dir(testCaseClass):
            if name.startswith(self.testMethodPrefix):
                suite.addTest(testCaseClass(name))
        return suite

    def loadTestsFromModule(self, module):
        suite = TestSuite()
        for name in dir(module):
            obj = getattr(module, name)
            try:
                is_case = issubclass(obj, TestCase)
            except TypeError:
                is_case = False
            if is_case and obj is not TestCase:
                for test in self.loadTestsFromTestCase(obj):
                    suite.addTest(test)
        return suite


class TextTestRunner:
    def __init__(self, stream=None, descriptions=True, verbosity=1):
        self.stream = stream
        self.descriptions = descriptions
        self.verbosity = verbosity

    def run(self, test):
        result = _Outcome()
        for case in test:
            case.run(result)
        if self.verbosity:
            if result.wasSuccessful():
                print("OK")
            else:
                print("FAILED")
                for case, err in result.failures:
                    print("FAIL:", repr(case), str(err))
                for case, err in result.errors:
                    print("ERROR:", repr(case), str(err))
        return result


defaultTestLoader = TestLoader()


def main(module=None):
    if module is None:
        module = __import__("__main__")
    result = TextTestRunner().run(defaultTestLoader.loadTestsFromModule(module))
    if not result.wasSuccessful():
        raise SystemExit(1)
