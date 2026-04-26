//! Background compilation queue for non-blocking JIT compilation.
//!
//! Implements a channel-based compilation queue with a dedicated worker thread.
//! The mutator thread enqueues compilation requests via MPSC channel, and the
//! worker thread processes them, inserting results into the shared code cache.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐    MPSC Channel    ┌─────────────────┐
//! │  Mutator(s)  │ ─────────────────► │  Worker Thread   │
//! │              │  CompilationReq    │                  │
//! │  .enqueue()  │                    │  owns:           │
//! │              │                    │  - TemplateComp  │
//! └──────────────┘                    │  - CodeCache ref │
//!                                     └────────┬────────┘
//!                                              │
//!                                     ┌────────▼────────┐
//!                                     │   Code Cache     │
//!                                     │  (Arc, shared)   │
//!                                     └─────────────────┘
//! ```
//!
//! # Performance
//!
//! - **Enqueue**: O(1) channel send, non-blocking
//! - **Compilation**: Happens off the critical path, does not block execution
//! - **Lookup**: After compilation, code is available via the shared cache

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use prism_code::CodeObject;
#[cfg(test)]
use prism_code::Opcode;
use prism_jit::runtime::{CodeCache, CompiledEntry};
use prism_jit::tier1::codegen::TemplateCompiler;

use crate::jit_bridge::{compile_tier1_entry, compile_tier2_entry};

/// A request to compile a function in the background.
struct CompilationRequest {
    /// The code object to compile.
    code: Arc<CodeObject>,
    /// The target compilation tier.
    tier: u8,
}

/// Statistics for the compilation queue.
#[derive(Debug, Default)]
pub struct CompilationQueueStats {
    /// Number of requests enqueued.
    pub enqueued: AtomicU64,
    /// Number of requests completed successfully.
    pub completed: AtomicU64,
    /// Number of requests that failed.
    pub failed: AtomicU64,
    /// Number of requests dropped (queue full or duplicate).
    pub dropped: AtomicU64,
}

impl CompilationQueueStats {
    /// Get snapshot of current stats.
    pub fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.enqueued.load(Ordering::Relaxed),
            self.completed.load(Ordering::Relaxed),
            self.failed.load(Ordering::Relaxed),
            self.dropped.load(Ordering::Relaxed),
        )
    }
}

/// Background compilation queue with a dedicated worker thread.
///
/// Decouples JIT compilation from the interpreter's critical path.
/// The worker thread owns its own `TemplateCompiler` instance for Tier 1 work
/// and runs Tier 2 compilation directly from the optimizing pipeline.
pub struct CompilationQueue {
    /// Channel sender for enqueuing requests.
    sender: mpsc::Sender<CompilationRequest>,
    /// Handle to the worker thread (for graceful shutdown).
    _worker: Option<thread::JoinHandle<()>>,
    /// Shared flag to signal shutdown.
    shutdown: Arc<AtomicBool>,
    /// Shared code cache for dedup checks on enqueue.
    code_cache: Arc<CodeCache>,
    /// Queue statistics.
    stats: Arc<CompilationQueueStats>,
    /// Approximate queue depth (sender doesn't expose len()).
    pending: Arc<AtomicUsize>,
    /// Maximum queue depth before dropping requests.
    max_queue_size: usize,
}

impl CompilationQueue {
    /// Create a new compilation queue with the given code cache and queue size.
    ///
    /// Spawns a dedicated worker thread that processes compilation requests.
    pub fn new(code_cache: Arc<CodeCache>, max_queue_size: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(CompilationQueueStats::default());
        let pending = Arc::new(AtomicUsize::new(0));

        let worker = {
            let code_cache = Arc::clone(&code_cache);
            let shutdown = Arc::clone(&shutdown);
            let stats = Arc::clone(&stats);
            let pending = Arc::clone(&pending);

            thread::Builder::new()
                .name("prism-jit-compiler".to_string())
                .spawn(move || {
                    Self::worker_loop(receiver, code_cache, shutdown, stats, pending);
                })
                .expect("Failed to spawn JIT compilation thread")
        };

        Self {
            sender,
            _worker: Some(worker),
            shutdown,
            code_cache,
            stats,
            pending,
            max_queue_size,
        }
    }

    /// Enqueue a compilation request.
    ///
    /// Returns `true` if the request was enqueued, `false` if the queue
    /// is full or the function is already compiled.
    pub fn enqueue(&self, code: Arc<CodeObject>, tier: u8) -> bool {
        let code_id = Arc::as_ptr(&code) as u64;

        // Fast path: already compiled at this tier or higher
        if let Some(entry) = self.code_cache.lookup(code_id) {
            if entry.tier() >= tier {
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        }

        // Check approximate queue depth
        if self.pending.load(Ordering::Relaxed) >= self.max_queue_size {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Send request
        let request = CompilationRequest { code, tier };
        match self.sender.send(request) {
            Ok(()) => {
                self.pending.fetch_add(1, Ordering::Relaxed);
                self.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => {
                // Channel disconnected — worker died
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Get the approximate queue depth.
    pub fn queue_depth(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }

    /// Get compilation statistics.
    pub fn stats(&self) -> &CompilationQueueStats {
        &self.stats
    }

    /// Check if the queue is empty (approximate).
    pub fn is_empty(&self) -> bool {
        self.queue_depth() == 0
    }

    /// Get maximum queue size.
    pub fn max_queue_size(&self) -> usize {
        self.max_queue_size
    }

    /// The worker thread's main loop.
    ///
    /// Processes compilation requests until shutdown or channel disconnect.
    fn worker_loop(
        receiver: mpsc::Receiver<CompilationRequest>,
        code_cache: Arc<CodeCache>,
        shutdown: Arc<AtomicBool>,
        stats: Arc<CompilationQueueStats>,
        pending: Arc<AtomicUsize>,
    ) {
        // Worker owns its own compiler instance — no contention with mutator
        let mut compiler = TemplateCompiler::new_runtime();

        loop {
            // Check shutdown flag
            if shutdown.load(Ordering::Acquire) {
                break;
            }

            // Block waiting for next request (with timeout for shutdown checks)
            match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(request) => {
                    pending.fetch_sub(1, Ordering::Relaxed);
                    Self::process_request(&mut compiler, &code_cache, &stats, request);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Check shutdown and loop
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Channel closed, exit
                    break;
                }
            }
        }
    }

    /// Process a single compilation request.
    fn process_request(
        compiler: &mut TemplateCompiler,
        code_cache: &CodeCache,
        stats: &CompilationQueueStats,
        request: CompilationRequest,
    ) {
        let code_id = Arc::as_ptr(&request.code) as u64;

        // Double-check: might have been compiled while waiting in queue
        if let Some(entry) = code_cache.lookup(code_id) {
            if entry.tier() >= request.tier {
                stats.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }

        // Compile at the requested tier (>=2 means optimize with Tier 2 pipeline).
        let compiled = if request.tier >= 2 {
            compile_tier2_entry(&request.code)
        } else {
            compile_tier1_entry(&request.code, compiler)
        };

        match compiled {
            Ok(entry) => {
                code_cache.insert(entry);
                stats.completed.fetch_add(1, Ordering::Relaxed);
            }
            Err(_e) => {
                if std::env::var_os("PRISM_JIT_LOG_FAILURES").is_some() {
                    eprintln!("Background compilation failed (codegen): {}", _e);
                }
                stats.failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for CompilationQueue {
    fn drop(&mut self) {
        // Signal worker to shut down
        self.shutdown.store(true, Ordering::Release);

        // Take the worker handle and join it
        if let Some(handle) = self._worker.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests;
