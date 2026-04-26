use super::*;

#[test]
fn test_frame_size() {
    // Frame should be approximately 2KB + overhead
    let size = std::mem::size_of::<Frame>();
    // 256 * 8 (registers) + code arc + ip + return info + closure
    assert!(size >= REGISTER_COUNT * 8);
    assert!(size < 4096, "Frame too large: {} bytes", size);
}

#[test]
fn test_register_access() {
    let code = Arc::new(CodeObject::new("test", "test.py"));
    let mut frame = Frame::new(code, None, 0);

    // Test set/get
    frame.set_reg(0, Value::int(42).unwrap());
    assert_eq!(frame.get_reg(0).as_int(), Some(42));

    // Test boundary registers
    frame.set_reg(255, Value::float(3.14));
    assert!((frame.get_reg(255).as_float().unwrap() - 3.14).abs() < 0.001);
}

#[test]
fn test_register_multi_access() {
    let code = Arc::new(CodeObject::new("test", "test.py"));
    let mut frame = Frame::new(code, None, 0);

    frame.set_reg(1, Value::int(10).unwrap());
    frame.set_reg(2, Value::int(20).unwrap());
    frame.set_reg(3, Value::int(30).unwrap());

    let (a, b) = frame.get_regs2(1, 2);
    assert_eq!(a.as_int(), Some(10));
    assert_eq!(b.as_int(), Some(20));

    let (x, y, z) = frame.get_regs3(1, 2, 3);
    assert_eq!(x.as_int(), Some(10));
    assert_eq!(y.as_int(), Some(20));
    assert_eq!(z.as_int(), Some(30));
}

#[test]
fn test_frame_tracks_active_register_count() {
    let mut code = CodeObject::new("test", "test.py");
    code.register_count = 7;
    let frame = Frame::new(Arc::new(code), None, 0);
    assert_eq!(frame.active_register_count(), 7);
}

#[test]
fn test_frame_pool_reuses_and_clears_live_window() {
    let mut pool = FramePool::new();
    let mut code = CodeObject::new("pool", "test.py");
    code.register_count = 2;

    let mut frame = pool.acquire(Arc::new(code), None, 0, None, None);
    frame.set_reg(0, Value::int(41).unwrap());
    frame.set_reg(1, Value::int(99).unwrap());
    pool.release(frame);

    assert_eq!(pool.len(), 1);

    let mut code = CodeObject::new("pool-reuse", "test.py");
    code.register_count = 2;
    let frame = pool.acquire(Arc::new(code), None, 0, None, None);
    assert!(frame.get_reg(0).is_none());
    assert!(frame.get_reg(1).is_none());
    assert!(!frame.reg_is_written(0));
    assert!(!frame.reg_is_written(1));
}

#[test]
fn test_mark_reg_written_preserves_value() {
    let code = Arc::new(CodeObject::new("test", "test.py"));
    let mut frame = Frame::new(code, None, 0);

    assert!(frame.get_reg(0).is_none());
    assert!(!frame.reg_is_written(0));

    frame.mark_reg_written(0);

    assert!(frame.get_reg(0).is_none());
    assert!(frame.reg_is_written(0));
}

#[test]
fn test_frame_pool_clears_registers_before_storing_frame() {
    let mut pool = FramePool::new();
    let mut code = CodeObject::new("pool-store", "test.py");
    code.register_count = 3;

    let mut frame = pool.acquire(Arc::new(code), None, 0, None, None);
    frame.set_reg(0, Value::int(1).unwrap());
    frame.set_reg(2, Value::int(3).unwrap());
    pool.release(frame);

    let pooled = pool
        .free_frames
        .last()
        .expect("frame should be stored in pool");
    assert!(pooled.get_reg(0).is_none());
    assert!(pooled.get_reg(2).is_none());
    assert!(!pooled.reg_is_written(0));
    assert!(!pooled.reg_is_written(2));
    assert_eq!(pooled.active_register_count(), 0);
}

#[test]
fn test_frame_pool_clears_larger_future_window() {
    let mut pool = FramePool::new();

    let mut large_code = CodeObject::new("large", "test.py");
    large_code.register_count = 4;
    let mut frame = pool.acquire(Arc::new(large_code), None, 0, None, None);
    frame.set_reg(3, Value::int(7).unwrap());
    pool.release(frame);

    let mut small_code = CodeObject::new("small", "test.py");
    small_code.register_count = 1;
    let frame = pool.acquire(Arc::new(small_code), None, 0, None, None);
    assert!(frame.get_reg(3).is_none());
}

#[test]
fn test_closure_env_inline() {
    // Test inline storage (≤4 cells)
    let env = ClosureEnv::with_unbound_cells(3);
    assert!(env.is_inline());
    assert_eq!(env.len(), 3);

    // Set values through cells
    env.set(0, Value::int(100).unwrap());
    env.set(1, Value::float(2.5));
    env.set(2, Value::none());

    assert_eq!(env.get(0).as_int(), Some(100));
    assert!((env.get(1).as_float().unwrap() - 2.5).abs() < 0.001);
    assert!(env.get(2).is_none());
}

#[test]
fn test_closure_env_overflow() {
    // Test overflow storage (>4 cells)
    let env = ClosureEnv::with_unbound_cells(6);
    assert!(!env.is_inline());
    assert_eq!(env.len(), 6);

    for i in 0..6 {
        env.set(i, Value::int(i as i64).unwrap());
    }

    for i in 0..6 {
        assert_eq!(env.get(i).as_int(), Some(i as i64));
    }
}

#[test]
fn test_closure_env_empty() {
    let env = ClosureEnv::empty();
    assert!(env.is_empty());
    assert_eq!(env.len(), 0);
    assert!(env.is_inline());
}

#[test]
fn test_closure_env_shared_mutation() {
    // Test that cells are shared (mutations visible across closures)
    let cell = Arc::new(prism_runtime::types::Cell::new(Value::int(42).unwrap()));
    let cells = vec![Arc::clone(&cell)];
    let env = ClosureEnv::new(cells);

    // Modify through env
    env.set(0, Value::int(100).unwrap());

    // Should be visible through original cell
    assert_eq!(cell.get().unwrap().as_int(), Some(100));
}

#[test]
fn test_closure_env_clone() {
    let env1 = ClosureEnv::with_unbound_cells(2);
    env1.set(0, Value::int(42).unwrap());

    let env2 = env1.clone();

    // Cloned env shares the same cells
    assert_eq!(env2.get(0).as_int(), Some(42));

    // Mutation in env2 visible in env1 (shared cells)
    env2.set(0, Value::int(99).unwrap());
    assert_eq!(env1.get(0).as_int(), Some(99));
}
