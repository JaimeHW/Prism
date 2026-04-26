//! Native Python thread execution support.
//!
//! Prism currently uses a process-wide Python execution lock while bytecode or
//! Python-visible runtime objects are active. The lock is deliberately
//! reentrant so nested interpreter entry points stay cheap and correct, and it
//! is released around known blocking native operations so other Python threads
//! can keep making progress.

use std::cell::Cell;
use std::sync::{Condvar, LazyLock, Mutex};
use std::thread::{self, ThreadId};

const CHECKPOINT_INTERVAL_OPCODES: u32 = 64;

static EXECUTION_LOCK: LazyLock<(Mutex<ExecutionLockState>, Condvar)> =
    LazyLock::new(|| (Mutex::new(ExecutionLockState::default()), Condvar::new()));

thread_local! {
    static CHECKPOINT_TICKER: Cell<u32> = const { Cell::new(CHECKPOINT_INTERVAL_OPCODES) };
}

#[derive(Debug, Default)]
struct ExecutionLockState {
    owner: Option<ThreadId>,
    depth: usize,
}

#[must_use = "dropping the guard leaves the Python execution region"]
#[derive(Debug)]
pub(crate) struct ExecutionRegionGuard {
    active: bool,
}

impl Drop for ExecutionRegionGuard {
    #[inline]
    fn drop(&mut self) {
        if self.active {
            release_execution_lock();
        }
    }
}

#[must_use = "dropping the guard reacquires the Python execution lock"]
struct BlockingReleaseGuard {
    depth: usize,
}

impl Drop for BlockingReleaseGuard {
    #[inline]
    fn drop(&mut self) {
        if self.depth != 0 {
            acquire_execution_lock_at_depth(self.depth);
        }
    }
}

/// Enter Python bytecode execution on the current OS thread.
///
/// The lock is reentrant on a per-thread basis because nested interpreter loops
/// are common: imports, descriptor calls, `eval`, generators, and builtin
/// callbacks can all re-enter the VM while a caller is already executing Python.
#[inline]
pub(crate) fn enter_execution_region() -> ExecutionRegionGuard {
    acquire_execution_lock_at_depth(1);
    ExecutionRegionGuard { active: true }
}

/// Cooperative scheduling checkpoint for long bytecode loops.
///
/// The fast path is a thread-local countdown. When the countdown expires, the
/// current thread briefly releases the execution lock and yields so a waiting
/// Python thread can acquire it without forcing every opcode through a mutex.
#[inline]
pub(crate) fn checkpoint() {
    let should_yield = CHECKPOINT_TICKER.with(|ticker| {
        let remaining = ticker.get();
        if remaining > 1 {
            ticker.set(remaining - 1);
            false
        } else {
            ticker.set(CHECKPOINT_INTERVAL_OPCODES);
            true
        }
    });

    if should_yield {
        let _released = release_for_blocking_operation();
        thread::yield_now();
    }
}

/// Run a blocking native operation.
///
/// If the current thread owns the Python execution lock, this releases the full
/// reentrant depth for the duration of `body` and restores it before returning
/// or unwinding. Calls made outside Python execution simply run `body`.
#[inline]
pub(crate) fn blocking_operation<R>(body: impl FnOnce() -> R) -> R {
    let released = release_for_blocking_operation();
    let result = body();
    drop(released);
    result
}

fn acquire_execution_lock_at_depth(depth: usize) {
    debug_assert!(depth > 0);

    let current = thread::current().id();
    let (state_lock, available) = &*EXECUTION_LOCK;
    let mut state = state_lock
        .lock()
        .expect("Python execution lock should not be poisoned");

    loop {
        match state.owner {
            None => {
                state.owner = Some(current);
                state.depth = depth;
                return;
            }
            Some(owner) if owner == current => {
                state.depth = state
                    .depth
                    .checked_add(depth)
                    .expect("Python execution lock recursion depth overflowed");
                return;
            }
            Some(_) => {
                state = available
                    .wait(state)
                    .expect("Python execution lock wait should not be poisoned");
            }
        }
    }
}

fn release_execution_lock() {
    let current = thread::current().id();
    let (state_lock, available) = &*EXECUTION_LOCK;
    let mut state = state_lock
        .lock()
        .expect("Python execution lock should not be poisoned");

    debug_assert_eq!(
        state.owner,
        Some(current),
        "current thread must own the Python execution lock"
    );
    debug_assert!(state.depth > 0);

    if state.depth > 1 {
        state.depth -= 1;
        return;
    }

    state.owner = None;
    state.depth = 0;
    available.notify_one();
}

fn release_for_blocking_operation() -> BlockingReleaseGuard {
    let current = thread::current().id();
    let (state_lock, available) = &*EXECUTION_LOCK;
    let mut state = state_lock
        .lock()
        .expect("Python execution lock should not be poisoned");

    if state.owner != Some(current) {
        return BlockingReleaseGuard { depth: 0 };
    }

    let depth = state.depth;
    state.owner = None;
    state.depth = 0;
    available.notify_one();
    BlockingReleaseGuard { depth }
}

#[cfg(test)]
mod tests {
    use super::{blocking_operation, checkpoint, enter_execution_region};
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn test_execution_regions_serialize_python_threads() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let first_tx = entered_tx.clone();
        let first = std::thread::spawn(move || {
            let _region = enter_execution_region();
            first_tx.send("first").expect("receiver should be alive");
            release_rx.recv().expect("release sender should be alive");
        });

        assert_eq!(
            entered_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("first thread should enter"),
            "first"
        );

        let second = std::thread::spawn(move || {
            let _region = enter_execution_region();
            entered_tx.send("second").expect("receiver should be alive");
        });

        assert!(
            entered_rx.recv_timeout(Duration::from_millis(100)).is_err(),
            "second thread should wait until the first execution region exits"
        );

        release_tx.send(()).expect("first thread should be alive");
        first.join().expect("first thread should finish");
        assert_eq!(
            entered_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("second thread should enter after the first exits"),
            "second"
        );
        second.join().expect("second thread should finish");
    }

    #[test]
    fn test_blocking_operation_releases_execution_gate() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let _region = enter_execution_region();

        let value = blocking_operation(|| {
            let second = std::thread::spawn(move || {
                let _region = enter_execution_region();
                entered_tx.send("second").expect("receiver should be alive");
            });

            assert_eq!(
                entered_rx
                    .recv_timeout(Duration::from_secs(2))
                    .expect("blocking operation should release the execution lock"),
                "second"
            );
            second.join().expect("second thread should finish");
            42
        });

        assert_eq!(value, 42);
    }

    #[test]
    fn test_checkpoint_hands_off_execution_gate() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let _region = enter_execution_region();

        let second = std::thread::spawn(move || {
            let _region = enter_execution_region();
            entered_tx.send(()).expect("receiver should be alive");
        });

        for _ in 0..(super::CHECKPOINT_INTERVAL_OPCODES * 4) {
            checkpoint();
            if entered_rx.try_recv().is_ok() {
                second.join().expect("second thread should finish");
                return;
            }
        }

        panic!("checkpoint should periodically release the execution lock");
    }
}
