use super::*;

#[test]
fn test_compilation_queue_creation() {
    let code_cache = Arc::new(CodeCache::new(1024 * 1024));
    let queue = CompilationQueue::new(code_cache, 16);
    assert!(queue.is_empty());
    assert_eq!(queue.max_queue_size(), 16);
}

#[test]
fn test_compilation_queue_stats_initial() {
    let code_cache = Arc::new(CodeCache::new(1024 * 1024));
    let queue = CompilationQueue::new(code_cache, 16);
    let (enqueued, completed, failed, dropped) = queue.stats().snapshot();
    assert_eq!(enqueued, 0);
    assert_eq!(completed, 0);
    assert_eq!(failed, 0);
    assert_eq!(dropped, 0);
}

#[test]
fn test_compilation_queue_enqueue_and_compile() {
    let code_cache = Arc::new(CodeCache::new(1024 * 1024));
    let queue = CompilationQueue::new(Arc::clone(&code_cache), 16);

    // Create a simple code object with a return instruction
    let mut code = CodeObject::new("test_func", "<test>");
    code.register_count = 1;
    use prism_code::Instruction;
    code.instructions = vec![
        Instruction::op_d(Opcode::LoadNone, prism_code::Register::new(0)),
        Instruction::op_d(Opcode::Return, prism_code::Register::new(0)),
    ]
    .into_boxed_slice();

    let code = Arc::new(code);
    let code_id = Arc::as_ptr(&code) as u64;

    // Enqueue
    assert!(queue.enqueue(Arc::clone(&code), 1));

    // Wait for compilation to complete
    let mut attempts = 0;
    while code_cache.lookup(code_id).is_none() && attempts < 50 {
        std::thread::sleep(std::time::Duration::from_millis(10));
        attempts += 1;
    }

    // Verify compiled
    assert!(
        code_cache.lookup(code_id).is_some(),
        "Compilation should complete within 500ms"
    );

    let entry = code_cache.lookup(code_id).unwrap();
    assert_eq!(entry.tier(), 1);

    // Stats should reflect completion
    let (enqueued, completed, _failed, _dropped) = queue.stats().snapshot();
    assert_eq!(enqueued, 1);
    assert_eq!(completed, 1);
}

#[test]
fn test_compilation_queue_duplicate_skip() {
    let code_cache = Arc::new(CodeCache::new(1024 * 1024));
    let queue = CompilationQueue::new(Arc::clone(&code_cache), 16);

    // Create a code object
    let mut code = CodeObject::new("already_compiled", "<test>");
    code.register_count = 1;
    use prism_code::Instruction;
    code.instructions = vec![
        Instruction::op_d(Opcode::LoadNone, prism_code::Register::new(0)),
        Instruction::op_d(Opcode::Return, prism_code::Register::new(0)),
    ]
    .into_boxed_slice();
    let code = Arc::new(code);
    let code_id = Arc::as_ptr(&code) as u64;

    // Pre-insert a tier 1 entry using raw pointer constructor (no ExecutableBuffer needed)
    let fake_entry = CompiledEntry::new(code_id, 0x10000 as *const u8, 1).with_tier(1);
    code_cache.insert(fake_entry);

    // Enqueue at same tier — should be dropped
    assert!(!queue.enqueue(Arc::clone(&code), 1));

    let (_enqueued, _completed, _failed, dropped) = queue.stats().snapshot();
    assert_eq!(dropped, 1);
}

#[test]
fn test_compilation_queue_graceful_shutdown() {
    let code_cache = Arc::new(CodeCache::new(1024 * 1024));
    let queue = CompilationQueue::new(code_cache, 16);

    // Drop should join worker thread without hanging
    drop(queue);
    // If we get here, shutdown was graceful
}

#[test]
fn test_compilation_queue_multiple_functions() {
    let code_cache = Arc::new(CodeCache::new(1024 * 1024));
    let queue = CompilationQueue::new(Arc::clone(&code_cache), 16);

    let mut code_ids = Vec::new();

    // Enqueue multiple functions
    for i in 0..5 {
        let name: Arc<str> = Arc::from(format!("func_{}", i).as_str());
        let mut code = CodeObject::new(name, "<test>");
        code.register_count = 1;
        use prism_code::Instruction;
        code.instructions = vec![
            Instruction::op_d(Opcode::LoadNone, prism_code::Register::new(0)),
            Instruction::op_d(Opcode::Return, prism_code::Register::new(0)),
        ]
        .into_boxed_slice();
        let code = Arc::new(code);
        code_ids.push(Arc::as_ptr(&code) as u64);
        queue.enqueue(code, 1);
    }

    // Wait for all compilations
    let mut attempts = 0;
    while code_ids.iter().any(|id| code_cache.lookup(*id).is_none()) && attempts < 100 {
        std::thread::sleep(std::time::Duration::from_millis(10));
        attempts += 1;
    }

    // All should be compiled
    for code_id in &code_ids {
        assert!(
            code_cache.lookup(*code_id).is_some(),
            "Function {:x} should be compiled",
            code_id
        );
    }

    let (enqueued, completed, _failed, _dropped) = queue.stats().snapshot();
    assert_eq!(enqueued, 5);
    assert_eq!(completed, 5);
}

#[test]
fn test_compilation_queue_queue_depth_tracking() {
    let code_cache = Arc::new(CodeCache::new(1024 * 1024));
    let queue = CompilationQueue::new(code_cache, 64);

    // Initially empty
    assert_eq!(queue.queue_depth(), 0);
    assert!(queue.is_empty());
}

#[test]
fn test_compilation_queue_stats_snapshot() {
    let stats = CompilationQueueStats::default();
    stats.enqueued.store(10, Ordering::Relaxed);
    stats.completed.store(8, Ordering::Relaxed);
    stats.failed.store(1, Ordering::Relaxed);
    stats.dropped.store(1, Ordering::Relaxed);

    let (e, c, f, d) = stats.snapshot();
    assert_eq!(e, 10);
    assert_eq!(c, 8);
    assert_eq!(f, 1);
    assert_eq!(d, 1);
}

#[test]
fn test_compilation_queue_tier_upgrade() {
    use prism_jit::runtime::ReturnAbi;

    let code_cache = Arc::new(CodeCache::new(1024 * 1024));
    let queue = CompilationQueue::new(Arc::clone(&code_cache), 16);

    // Create a code object
    let mut code = CodeObject::new("upgrade_func", "<test>");
    code.register_count = 1;
    use prism_code::Instruction;
    code.instructions = vec![
        Instruction::op_d(Opcode::LoadNone, prism_code::Register::new(0)),
        Instruction::op_d(Opcode::Return, prism_code::Register::new(0)),
    ]
    .into_boxed_slice();
    let code = Arc::new(code);
    let code_id = Arc::as_ptr(&code) as u64;

    // Pre-insert a tier 1 entry using raw pointer constructor (no ExecutableBuffer needed)
    let fake_entry = CompiledEntry::new(code_id, 0x10000 as *const u8, 1).with_tier(1);
    code_cache.insert(fake_entry);

    // Enqueue at tier 2 - should compile with the optimizing pipeline.
    assert!(queue.enqueue(Arc::clone(&code), 2));

    // Wait for the tier upgrade to complete.
    let mut attempts = 0;
    while attempts < 100 {
        if let Some(entry) = code_cache.lookup(code_id) {
            if entry.tier() >= 2 {
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        attempts += 1;
    }

    let entry = code_cache
        .lookup(code_id)
        .expect("entry should remain present");
    assert_eq!(entry.tier(), 2);
    assert_eq!(entry.return_abi(), ReturnAbi::RawValueBits);

    let (enqueued, completed, failed, _dropped) = queue.stats().snapshot();
    assert_eq!(enqueued, 1);
    assert_eq!(completed, 1);
    assert_eq!(failed, 0);
}
