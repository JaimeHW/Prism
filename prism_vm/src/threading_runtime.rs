//! Native Python thread execution support.
//!
//! Prism intentionally does not use a global interpreter lock. Python worker
//! threads run on independent OS threads and may execute bytecode at the same
//! time. Shared runtime state must be protected at the owning data structure:
//! heap metadata, import tables, native locks, and blocking I/O each provide
//! their own synchronization without a process-wide execution mutex.

use std::cell::Cell;
use std::thread;

const CHECKPOINT_INTERVAL_OPCODES: u32 = 64;

thread_local! {
    static CHECKPOINT_TICKER: Cell<u32> = const { Cell::new(CHECKPOINT_INTERVAL_OPCODES) };
}

#[must_use = "dropping the guard leaves the Python execution region"]
#[derive(Debug, Default)]
pub(crate) struct ExecutionRegionGuard;

/// Enter Python bytecode execution on the current OS thread.
///
/// This is deliberately not a lock acquisition. It gives call sites a clear
/// marker for Python execution lifetime while preserving Prism's no-GIL
/// threading model.
#[inline]
pub(crate) fn enter_execution_region() -> ExecutionRegionGuard {
    ExecutionRegionGuard
}

/// Cooperative scheduling checkpoint for long bytecode loops.
///
/// With no GIL there is no mutex to hand off. The checkpoint only gives the OS
/// scheduler an occasional opportunity to run another ready thread and remains
/// cheap on the hot path.
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
        thread::yield_now();
    }
}

/// Run a blocking native operation.
///
/// This wrapper exists so blocking sites document their concurrency boundary.
/// It performs no global unlock because Prism does not hold a global execution
/// lock in the first place.
#[inline]
pub(crate) fn blocking_operation<R>(body: impl FnOnce() -> R) -> R {
    body()
}

#[cfg(test)]
mod tests {
    use super::{blocking_operation, checkpoint, enter_execution_region};
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn test_execution_regions_do_not_serialize_threads() {
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

        assert_eq!(
            entered_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("second thread should enter while first is still running"),
            "second"
        );

        release_tx.send(()).expect("first thread should be alive");
        first.join().expect("first thread should finish");
        second.join().expect("second thread should finish");
    }

    #[test]
    fn test_blocking_operation_has_no_execution_gate_to_release() {
        let _region = enter_execution_region();
        assert_eq!(blocking_operation(|| 42), 42);
    }

    #[test]
    fn test_checkpoint_is_non_blocking() {
        for _ in 0..(super::CHECKPOINT_INTERVAL_OPCODES * 2) {
            checkpoint();
        }
    }
}
