use super::{blocking_operation, checkpoint, enter_execution_region};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const THREAD_ENTRY_TIMEOUT: Duration = Duration::from_secs(10);
const SHOULD_REMAIN_BLOCKED_TIMEOUT: Duration = Duration::from_millis(100);

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
            .recv_timeout(THREAD_ENTRY_TIMEOUT)
            .expect("first thread should enter"),
        "first"
    );

    let second = std::thread::spawn(move || {
        let _region = enter_execution_region();
        entered_tx.send("second").expect("receiver should be alive");
    });

    assert!(
        entered_rx
            .recv_timeout(SHOULD_REMAIN_BLOCKED_TIMEOUT)
            .is_err(),
        "second thread should wait until the first execution region exits"
    );

    release_tx.send(()).expect("first thread should be alive");
    first.join().expect("first thread should finish");
    assert_eq!(
        entered_rx
            .recv_timeout(THREAD_ENTRY_TIMEOUT)
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
                .recv_timeout(THREAD_ENTRY_TIMEOUT)
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

    let deadline = Instant::now() + THREAD_ENTRY_TIMEOUT;
    while super::waiting_thread_count_for_test() == 0 {
        assert!(
            Instant::now() < deadline,
            "second thread should block on the execution gate"
        );
        std::thread::yield_now();
    }

    while Instant::now() < deadline {
        checkpoint();
        if entered_rx.try_recv().is_ok() {
            second.join().expect("second thread should finish");
            return;
        }
        std::thread::yield_now();
    }

    panic!("checkpoint should periodically release the execution lock");
}
