use super::*;

#[test]
fn test_list_subclass_iterates_with_native_storage() {
    let result = execute(
        r#"
class L(list):
    pass

items = L()
items.append(2)
items.append(3)

seen = []
for value in items:
    seen.append(value)

assert len(items) == 2
assert seen == [2, 3]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_list_slice_assignment_accepts_sequence_getitem_protocol() {
    let result = execute(
        r#"
class SequenceOnly:
    def __init__(self):
        self.values = [10, 11]

    def __getitem__(self, index):
        return self.values[index]

items = [0, 1, 2, 3]
items[1:3] = SequenceOnly()
assert items == [0, 10, 11, 3]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_sequence_getitem_iteration_preserves_outer_except_context() {
    let result = execute(
        r#"
import sys

class SequenceOnly:
    def __getitem__(self, index):
        if index == 0:
            return "value"
        raise IndexError("done")

seen = []

try:
    raise ValueError("outer")
except ValueError:
    iterator = iter(SequenceOnly())
    seen.append(next(iterator))
    try:
        next(iterator)
    except StopIteration:
        seen.append("stop")
    assert sys.exc_info()[0] is ValueError

seen.append("after")
assert sys.exc_info() == (None, None, None)
assert seen == ["value", "stop", "after"]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_uncaught_sequence_getitem_callback_exception_reaches_python_handler() {
    let result = execute(
        r#"
class BadSequence:
    def __getitem__(self, index):
        raise RuntimeError("boom")

caught = False
try:
    iter(BadSequence())
except RuntimeError as exc:
    caught = str(exc) == "boom"

assert caught
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_cpython_threading_thread_runs_and_joins_native_worker() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import threading

seen = []

def worker():
    seen.append(42)

t = threading.Thread(target=worker)
t.start()
t.join(5)

assert seen == [42]
assert not t.is_alive()
"#,
        20_000_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_native_threads_share_sys_modules_with_spawning_interpreter() {
    let result = execute_with_search_paths_and_step_limit(
        r#"
import _thread
import sys
import time

marker = object()
sys.modules["shared_thread_marker"] = marker
seen = []

def worker():
    import sys
    seen.append(sys.modules.get("shared_thread_marker") is marker)

_thread.start_new_thread(worker, ())

for _ in range(1000):
    if seen:
        break
    time.sleep(0.001)

assert seen == [True], seen
"#,
        &[],
        Some(5_000_000),
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_module_namespace_builtins_return_live_module_dicts() {
    let result = execute(
        r#"
import _thread

module_dict = vars(_thread)
assert module_dict is _thread.__dict__
assert "_is_main_interpreter" in module_dict

copied = {}
copied.update(module_dict)
assert copied["_is_main_interpreter"] is module_dict["_is_main_interpreter"]

globals()["from_globals_dict"] = 123
assert from_globals_dict == 123

vars()["from_vars_dict"] = 456
assert from_vars_dict == 456
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_str_replace_accepts_keywords_in_python_code() {
    let result = execute(
        r#"
assert "banana".replace(old="na", new="NA") == "baNANA"
assert "banana".replace("na", new="NA", count=1) == "baNAna"
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_starred_list_literal_expands_iterables() {
    let result = execute(
        r#"
items = [0, *(1, 2), 3]
assert len(items) == 4
assert items[0] == 0
assert items[1] == 1
assert items[2] == 2
assert items[3] == 3
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_starred_tuple_literal_expands_iterables() {
    let result = execute(
        r#"
items = (0, *[1, 2], 3)
assert len(items) == 4
assert items[0] == 0
assert items[1] == 1
assert items[2] == 2
assert items[3] == 3
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_starred_set_literal_expands_iterables() {
    let result = execute(
        r#"
items = {*[1, 2], 2, 3}
assert len(items) == 3
assert 1 in items
assert 2 in items
assert 3 in items
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_cpython_threading_custom_excepthook_receives_args() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import threading

seen = []

def hook(args):
    seen.append(args.exc_type)
    seen.append(str(args.exc_value))
    seen.append(args.thread)

threading.excepthook = hook

class ThreadRunFail(threading.Thread):
    def run(self):
        raise ValueError("run failed")

thread = ThreadRunFail()
thread.start()
thread.join(5)

assert seen[0] is ValueError
assert seen[1] == "run failed"
assert seen[2] is thread
"#,
        20_000_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_cpython_threading_excepthook_prints_source_line() {
    let temp_dir = unique_temp_dir("threading_excepthook_source_line");
    fs::create_dir_all(&temp_dir).expect("failed to create temp module dir");
    fs::write(
        temp_dir.join("line_thread_probe.py"),
        r#"import threading

class ThreadRunFail(threading.Thread):
    def run(self):
        raise ValueError("run failed")

def run():
    thread = ThreadRunFail(name="line-thread")
    thread.start()
    thread.join(5)
"#,
    )
    .expect("failed to write temp module");

    let cpython_lib = cpython_lib_dir();
    let result = execute_with_search_paths_and_step_limit(
        r#"
import io
import sys
import line_thread_probe

stderr = io.StringIO()
old_stderr = sys.stderr
sys.stderr = stderr
try:
    line_thread_probe.run()
finally:
    sys.stderr = old_stderr

rendered = stderr.getvalue()
assert 'Exception in thread line-thread:' in rendered, rendered
assert '  File "' in rendered, rendered
assert 'line 5, in run' in rendered, rendered
assert '  raise ValueError("run failed")' in rendered, rendered
assert 'ValueError: run failed' in rendered, rendered
"#,
        &[temp_dir.as_path(), cpython_lib.as_path()],
        Some(20_000_000),
    );
    let _ = fs::remove_dir_all(&temp_dir);
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_gc_collect_preserves_regex_patterns_used_by_tokenize() {
    let result = execute_with_cpython_lib_and_step_limit(
        r##"
import gc
import re
import tokenize

pattern = re.compile("x")
cookie_re = tokenize.cookie_re
blank_re = tokenize.blank_re

for _ in range(5):
    gc.collect()

assert pattern.match("x")
assert tokenize.cookie_re is cookie_re
assert tokenize.blank_re is blank_re
blank_re.match
cookie_re.match
"##,
        20_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_cpython_threading_all_uses_native_thread_support_metadata() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import threading
from test.test_threading import MiscTestCase

assert _thread._is_main_interpreter() is True
assert threading.local.__module__ == "_thread"

MiscTestCase("test__all__").test__all__()
"#,
        20_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_cpython_subinterp_threading_joins_workers_before_returning() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import os
import test.support

r, w = os.pipe()
try:
    code = (
        "import os, threading\n"
        f"w = {w}\n"
        "def worker():\n"
        "    os.write(w, b'x')\n"
        "threading.Thread(target=worker).start()\n"
    )
    assert test.support.run_in_subinterp(code) == 0
    assert os.read(r, 1) == b"x"
finally:
    os.close(r)
    os.close(w)
"#,
        20_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_cpython_subinterp_threading_does_not_poison_later_heap_exception_matching() {
    let first = execute_with_cpython_lib_and_step_limit(
        r#"
import os
import test.support

r, w = os.pipe()
try:
    code = (
        "import os, threading\n"
        f"w = {w}\n"
        "class LocalThread(threading.Thread):\n"
        "    pass\n"
        "def worker():\n"
        "    os.write(w, b'x')\n"
        "LocalThread(target=worker).start()\n"
    )
    assert test.support.run_in_subinterp(code) == 0
    assert os.read(r, 1) == b"x"
finally:
    os.close(r)
    os.close(w)
"#,
        20_000_000,
    );
    assert!(first.is_ok(), "Failed: {:?}", first);

    let second = execute_with_cpython_lib(
        r#"
import unittest

class Case(unittest.TestCase):
    def runTest(self):
        self.skipTest("subinterpreter teardown")

result = unittest.TestResult()
Case().run(result)

if len(result.errors) != 0:
    raise RuntimeError(result.errors[0][1])
assert len(result.failures) == 0, result.failures
assert len(result.skipped) == 1
assert result.skipped[0][1] == "subinterpreter teardown"
"#,
    );
    assert!(second.is_ok(), "Failed: {:?}", second);
}

#[test]
fn test_cpython_tracemalloc_reports_inactive_native_state() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _tracemalloc
import tracemalloc

assert _tracemalloc.is_tracing() is False
assert tracemalloc.is_tracing() is False
assert tracemalloc.get_traced_memory() == (0, 0)
"#,
        5_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_re_sub_accepts_callable_replacements() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import re

assert re.sub(r'\d+', lambda match: '[' + match.group(0) + ']', 'a12b3') == 'a[12]b[3]'
assert re.subn(r'\d+', lambda match: match.group(0), 'a12b3') == ('a12b3', 2)
assert re.sub(br'\d+', lambda match: b'[' + match.group(0) + b']', b'a12b3') == b'a[12]b[3]'

named = re.search(r'(?P<word>\w+)(?:-(\d+))?', 'abc')
assert named.lastindex == 1
assert named.lastgroup == 'word'
assert named[0] == 'abc'
assert named['word'] == 'abc'

plain = re.search(r'\w+', 'abc')
assert plain.lastindex is None
assert plain.lastgroup is None
assert plain[0] == 'abc'

assert re.sub(r'(?P<num>\d+)', lambda match: match.lastgroup + str(match.lastindex), 'x12') == 'xnum1'
assert re.sub(br'(?P<num>\d+)', lambda match: b'ok' if match.lastgroup == 'num' and match.lastindex == 1 else b'bad', b'x12') == b'xok'
assert re.search(br'(?P<word>\w+)', b'abc')[0] == b'abc'
assert re.search(br'(?P<word>\w+)', b'abc')['word'] == b'abc'
"#,
        5_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_thread_interrupt_main_raises_keyboard_interrupt_on_main_thread() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread

caught = False
try:
    _thread.interrupt_main()
except KeyboardInterrupt:
    caught = True

assert caught
"#,
        5_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_thread_interrupt_main_from_worker_raises_on_main_join() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import threading

def call_interrupt():
    _thread.interrupt_main()

thread = threading.Thread(target=call_interrupt)
caught = False
try:
    thread.start()
    thread.join(5)
except KeyboardInterrupt:
    caught = True

thread.join(5)
assert caught
"#,
        20_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_thread_interrupt_main_delivers_python_signal_handler() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import signal

seen = []

def handler(signum, frame):
    seen.append(signum)
    1 / 0

old = signal.signal(signal.SIGINT, handler)
caught = False
try:
    try:
        _thread.interrupt_main()
    except ZeroDivisionError:
        caught = True
finally:
    signal.signal(signal.SIGINT, old)

assert caught
assert seen == [signal.SIGINT], seen
"#,
        5_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_thread_interrupt_main_honors_ignored_and_default_signals() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import signal

for signum in (signal.SIGINT, signal.SIGTERM):
    old = signal.signal(signum, signal.SIG_IGN)
    try:
        _thread.interrupt_main(signum)
        signal.signal(signum, signal.SIG_DFL)
        _thread.interrupt_main(signum)
    finally:
        signal.signal(signum, old)
"#,
        5_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_thread_interrupt_main_rejects_invalid_signal_numbers() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import signal

for signum in (-1, signal.NSIG, 1000000):
    try:
        _thread.interrupt_main(signum)
    except ValueError:
        pass
    else:
        raise AssertionError(signum)
"#,
        5_000_000,
    );

    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_return_runs_finally_before_leaving_function() {
    let result = execute(
        r#"
events = []

def f():
    try:
        return 7
    finally:
        events.append(1)

assert f() == 7
assert events == [1]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_loop_control_flow_runs_finally_cleanup() {
    let result = execute(
        r#"
events = []

for i in [0, 1, 2]:
    try:
        if i == 0:
            continue
        break
    finally:
        events.append(i)

assert events == [0, 1]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_return_runs_with_exit_before_leaving_function() {
    let result = execute(
        r#"
events = []

class Manager:
    def __enter__(self):
        events.append("enter")
        return self
    def __exit__(self, exc_type, exc, tb):
        events.append(("exit", exc_type is None, exc is None, tb is None))
        return False

def f():
    with Manager():
        return 7

assert f() == 7
assert events == ["enter", ("exit", True, True, True)]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_loop_control_flow_runs_with_exit_cleanup() {
    let result = execute(
        r#"
events = []

class Manager:
    def __init__(self, value):
        self.value = value
    def __enter__(self):
        events.append(("enter", self.value))
        return self
    def __exit__(self, exc_type, exc, tb):
        events.append(("exit", self.value, exc_type is None))
        return False

for i in [0, 1, 2]:
    with Manager(i):
        if i == 0:
            continue
        break

assert events == [
    ("enter", 0), ("exit", 0, True),
    ("enter", 1), ("exit", 1, True),
]
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_loop_break_inside_outer_with_keeps_context_active() {
    let result = execute(
        r#"
events = []

class Manager:
    def __enter__(self):
        events.append("enter")
        return self
    def __exit__(self, exc_type, exc, tb):
        events.append(("exit", exc_type is None, exc is None, tb is None))
        return False

with Manager():
    events.append("before")
    for item in [1, 2]:
        events.append(("loop", item))
        break
    events.append("after")

events.append("done")

assert events == [
    "enter",
    "before",
    ("loop", 1),
    "after",
    ("exit", True, True, True),
    "done",
], events
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_loop_continue_inside_outer_with_keeps_context_active() {
    let result = execute(
        r#"
events = []

class Manager:
    def __enter__(self):
        events.append("enter")
        return self
    def __exit__(self, exc_type, exc, tb):
        events.append(("exit", exc_type is None, exc is None, tb is None))
        return False

with Manager():
    for item in [1, 2]:
        events.append(("loop", item))
        if item == 1:
            continue
        events.append(("body", item))
    events.append("after")

events.append("done")

assert events == [
    "enter",
    ("loop", 1),
    ("loop", 2),
    ("body", 2),
    "after",
    ("exit", True, True, True),
    "done",
], events
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_condition_wait_reacquires_lock_after_return_finally() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import threading
import time

cond = threading.Condition(threading.Lock())
seen = []
errors = []

def worker():
    try:
        with cond:
            seen.append(1)
            cond.wait()
            seen.append(2)
        seen.append(3)
    except BaseException as exc:
        errors.append(type(exc).__name__)

_thread.start_new_thread(worker, ())

for _ in [0] * 100:
    if seen:
        break
    time.sleep(0.01)

with cond:
    cond.notify_all()

for _ in [0] * 100:
    if _thread._count() == 1:
        break
    time.sleep(0.01)

assert errors == [], errors
assert seen == [1, 2, 3], seen
assert _thread._count() == 1
"#,
        20_000_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_barrier_shared_state_finishes_all_workers() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import threading
import time

class Holder:
    pass

holder = Holder()
holder.n = 5
holder.barrier = threading.Barrier(holder.n, timeout=2.0)
holder.results = []
holder.errors = []

def worker():
    try:
        holder.results.append(holder.barrier.wait())
    except BaseException as exc:
        holder.errors.append(type(exc).__name__)

for _ in range(holder.n):
    _thread.start_new_thread(worker, ())

for _ in range(300):
    if _thread._count() == 1 and len(holder.results) + len(holder.errors) == holder.n:
        break
    time.sleep(0.01)

assert holder.errors == [], holder.errors
assert sorted(holder.results) == [0, 1, 2, 3, 4], holder.results
assert _thread._count() == 1
"#,
        50_000_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_thread_target_preserves_nested_function_closure() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import time

def make_worker():
    token = 17
    seen = []
    errors = []

    def worker():
        try:
            seen.append(token)
        except BaseException as exc:
            errors.append((type(exc).__name__, str(exc)))

    return worker, seen, errors

worker, seen, errors = make_worker()
_thread.start_new_thread(worker, ())

for _ in range(200):
    if _thread._count() == 1:
        break
    time.sleep(0.01)

assert errors == [], errors
assert seen == [17], seen
assert _thread._count() == 1
"#,
        20_000_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_barrier_repr_uses_python_dunder_repr() {
    let result = execute_with_cpython_lib(
        r#"
import threading

barrier = threading.Barrier(3)
text = repr(barrier)

assert "threading.Barrier" in text, text
assert "waiters=0/3" in text, text
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_native_lock_repr_and_keyword_timeout() {
    let result = execute_with_cpython_lib(
        r#"
import _thread

lock = _thread.allocate_lock()
text = repr(lock)
assert text.startswith("<unlocked _thread.lock object at 0x"), text

try:
    lock.acquire(False, 1)
except ValueError:
    pass
else:
    raise AssertionError("non-blocking acquire with timeout should fail")

try:
    lock.acquire(timeout=-2)
except ValueError:
    pass
else:
    raise AssertionError("invalid timeout should fail")

assert lock.acquire(timeout=0.0)
assert repr(lock).startswith("<locked _thread.lock object at 0x")
lock.release()
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_worker_heap_allocations_survive_thread_exit() {
    let result = execute_with_search_paths_and_step_limit(
        r#"
import _thread
import time

seen = []

def worker():
    seen.append((True, 2))

_thread.start_new_thread(worker, ())

for _ in range(200):
    if _thread._count() == 1 and seen:
        break
    time.sleep(0.01)

assert seen == [(True, 2)], seen
assert _thread._count() == 1
"#,
        &[],
        Some(10_000_000),
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_native_locks_clear_weakrefs_after_gc_collect() {
    let result = execute(
        r#"
import _thread
import gc
import weakref

lock = _thread.RLock()
ref = weakref.ref(lock)
assert ref() is lock
del lock
gc.collect()
assert ref() is None
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_native_lock_weakref_preserves_live_target_after_gc_collect() {
    let result = execute(
        r#"
import _thread
import gc
import weakref

lock = _thread.RLock()
ref = weakref.ref(lock)
gc.collect()
assert ref() is lock
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_condition_weakref_clears_after_gc_collect() {
    let result = execute_with_cpython_lib(
        r#"
import gc
import threading
import weakref

condition = threading.Condition()
ref = weakref.ref(condition)
assert ref() is condition
del condition
gc.collect()
assert ref() is None
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_weakref_preserves_live_heap_object_after_gc_collect() {
    let result = execute(
        r#"
import gc
import weakref

class Box:
    pass

box = Box()
ref = weakref.ref(box)
gc.collect()
assert ref() is box
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_condition_uses_native_rlock_private_hooks() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import threading
import time

lock = threading.RLock()
cond = threading.Condition(lock)
seen = []
errors = []

def worker():
    try:
        with cond:
            lock.acquire()
            seen.append((lock._is_owned(), lock._recursion_count()))
            cond.wait(1.0)
            seen.append((lock._is_owned(), lock._recursion_count()))
            lock.release()
    except BaseException as exc:
        errors.append((type(exc).__name__, str(exc)))

_thread.start_new_thread(worker, ())

for _ in range(200):
    if seen:
        break
    time.sleep(0.01)

with cond:
    cond.notify_all()

for _ in range(200):
    if _thread._count() == 1:
        break
    time.sleep(0.01)

assert errors == [], errors
assert seen == [(True, 2), (True, 2)], seen
assert _thread._count() == 1
"#,
        40_000_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_threading_barrier_repr_flow_with_bunch_trips_barrier() {
    let result = execute_with_cpython_lib_and_step_limit(
        r#"
import _thread
import threading
import time
from test.lock_tests import Bunch

barrier = threading.Barrier(3)
indexes = []

def worker():
    indexes.append(barrier.wait(2.0))

with Bunch(worker, 2):
    for _ in range(100):
        if barrier.n_waiting >= 2:
            break
        time.sleep(0.01)
    assert "waiters=2/3" in repr(barrier), repr(barrier)
    indexes.append(barrier.wait(2.0))

assert sorted(indexes) == [0, 1, 2], indexes
assert _thread._count() == 1
"#,
        50_000_000,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_repr_honors_python_dunder_repr_result_contract() {
    let result = execute(
        r#"
class Custom:
    def __repr__(self):
        return "custom-repr"

class Bad:
    def __repr__(self):
        return 7

assert repr(Custom()) == "custom-repr"
assert ascii(Custom()) == "custom-repr"

try:
    repr(Bad())
except TypeError as exc:
    assert "__repr__ returned non-string" in str(exc), str(exc)
else:
    raise AssertionError("repr() should reject non-string __repr__ results")
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}
