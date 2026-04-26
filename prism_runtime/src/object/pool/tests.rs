use super::*;

// =========================================================================
// InstancePool Tests
// =========================================================================

#[test]
fn test_instance_pool_new() {
    let pool = InstancePool::new(ClassId(100), TypeId::from_raw(100), 4);

    assert_eq!(pool.class_id(), ClassId(100));
    assert_eq!(pool.type_id(), TypeId::from_raw(100));
    assert_eq!(pool.inline_slot_count(), 4);
    assert_eq!(pool.pooled_count(), 0);
    assert_eq!(pool.alloc_count(), 0);
    assert_eq!(pool.dealloc_count(), 0);
    assert_eq!(pool.max_pool_size(), InstancePool::DEFAULT_MAX_POOL_SIZE);
}

#[test]
fn test_instance_pool_with_max_size() {
    let pool = InstancePool::with_max_size(ClassId(200), TypeId::from_raw(200), 2, 16);

    assert_eq!(pool.max_pool_size(), 16);
}

#[test]
fn test_instance_pool_alloc_empty() {
    let pool = InstancePool::new(ClassId(100), TypeId::from_raw(100), 4);

    // Pool is empty, alloc should return None
    assert!(pool.alloc().is_none());
}

#[test]
fn test_instance_pool_dealloc_and_alloc() {
    let pool = InstancePool::new(ClassId(100), TypeId::from_raw(100), 4);

    // Create an instance to pool
    let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
    let ptr = NonNull::new(Box::into_raw(instance)).unwrap();

    // Dealloc to pool
    assert!(pool.dealloc(ptr));
    assert_eq!(pool.pooled_count(), 1);
    assert_eq!(pool.dealloc_count(), 1);

    // Alloc from pool
    let allocated = pool.alloc();
    assert!(allocated.is_some());
    assert_eq!(pool.pooled_count(), 0);
    assert_eq!(pool.alloc_count(), 1);

    // Clean up
    unsafe { drop(Box::from_raw(allocated.unwrap().as_ptr())) };
}

#[test]
fn test_instance_pool_multiple_dealloc_alloc() {
    let pool = InstancePool::with_max_size(ClassId(100), TypeId::from_raw(100), 4, 10);

    let mut instances = Vec::new();
    for _ in 0..5 {
        let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
        instances.push(NonNull::new(Box::into_raw(instance)).unwrap());
    }

    // Dealloc all to pool
    for ptr in &instances {
        assert!(pool.dealloc(*ptr));
    }
    assert_eq!(pool.pooled_count(), 5);

    // Alloc all from pool
    let mut allocated = Vec::new();
    for _ in 0..5 {
        allocated.push(pool.alloc().unwrap());
    }
    assert_eq!(pool.pooled_count(), 0);

    // Clean up
    for ptr in allocated {
        unsafe { drop(Box::from_raw(ptr.as_ptr())) };
    }
}

#[test]
fn test_instance_pool_capacity_limit() {
    let pool = InstancePool::with_max_size(ClassId(100), TypeId::from_raw(100), 4, 3);

    let mut instances = Vec::new();
    for _ in 0..5 {
        let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
        instances.push(NonNull::new(Box::into_raw(instance)).unwrap());
    }

    // Dealloc 5 but only 3 should be accepted
    let mut accepted = 0;
    for ptr in &instances {
        if pool.dealloc(*ptr) {
            accepted += 1;
        }
    }
    assert_eq!(accepted, 3);
    assert_eq!(pool.pooled_count(), 3);

    // Clean up the ones that weren't pooled
    for i in 3..5 {
        unsafe { drop(Box::from_raw(instances[i].as_ptr())) };
    }

    // Alloc the 3 pooled ones and clean up
    for _ in 0..3 {
        if let Some(ptr) = pool.alloc() {
            unsafe { drop(Box::from_raw(ptr.as_ptr())) };
        }
    }
}

#[test]
fn test_instance_pool_statistics() {
    let pool = InstancePool::new(ClassId(100), TypeId::from_raw(100), 4);

    // Create and pool several instances
    let mut instances = Vec::new();
    for _ in 0..3 {
        let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
        let ptr = NonNull::new(Box::into_raw(instance)).unwrap();
        pool.dealloc(ptr);
        instances.push(());
    }
    assert_eq!(pool.dealloc_count(), 3);

    // Alloc from pool
    let mut allocated = Vec::new();
    for _ in 0..3 {
        if let Some(ptr) = pool.alloc() {
            allocated.push(ptr);
        }
    }
    assert_eq!(pool.alloc_count(), 3);

    // Clean up
    for ptr in allocated {
        unsafe { drop(Box::from_raw(ptr.as_ptr())) };
    }
}

#[test]
fn test_instance_pool_debug() {
    let pool = InstancePool::new(ClassId(42), TypeId::from_raw(42), 4);

    let debug_str = format!("{:?}", pool);
    assert!(debug_str.contains("InstancePool"));
    assert!(debug_str.contains("class_id"));
}

#[test]
fn test_instance_pool_is_empty() {
    let pool = InstancePool::new(ClassId(100), TypeId::from_raw(100), 4);
    assert!(pool.is_empty());

    let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
    let ptr = NonNull::new(Box::into_raw(instance)).unwrap();
    pool.dealloc(ptr);

    assert!(!pool.is_empty());

    // Clean up
    if let Some(p) = pool.alloc() {
        unsafe { drop(Box::from_raw(p.as_ptr())) };
    }
}

#[test]
fn test_instance_pool_is_full() {
    let pool = InstancePool::with_max_size(ClassId(100), TypeId::from_raw(100), 4, 2);
    assert!(!pool.is_full());

    // Fill the pool
    for _ in 0..2 {
        let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
        let ptr = NonNull::new(Box::into_raw(instance)).unwrap();
        pool.dealloc(ptr);
    }

    assert!(pool.is_full());

    // Clean up
    while let Some(p) = pool.alloc() {
        unsafe { drop(Box::from_raw(p.as_ptr())) };
    }
}

// =========================================================================
// PoolManager Tests
// =========================================================================

#[test]
fn test_pool_manager_new() {
    let manager = PoolManager::new();

    assert_eq!(manager.pool_count(), 0);
    assert_eq!(
        manager.pool_threshold(),
        PoolManager::DEFAULT_POOL_THRESHOLD
    );
}

#[test]
fn test_pool_manager_with_threshold() {
    let manager = PoolManager::with_threshold(50);
    assert_eq!(manager.pool_threshold(), 50);
}

#[test]
fn test_pool_manager_create_pool() {
    let manager = PoolManager::new();

    assert!(manager.create_pool(ClassId(100), TypeId::from_raw(100), 4));
    assert_eq!(manager.pool_count(), 1);
    assert!(manager.has_pool(ClassId(100)));

    // Creating again should return false
    assert!(!manager.create_pool(ClassId(100), TypeId::from_raw(100), 4));
}

#[test]
fn test_pool_manager_remove_pool() {
    let manager = PoolManager::new();

    manager.create_pool(ClassId(100), TypeId::from_raw(100), 4);
    assert!(manager.remove_pool(ClassId(100)));
    assert_eq!(manager.pool_count(), 0);
    assert!(!manager.has_pool(ClassId(100)));

    // Removing non-existent pool should return false
    assert!(!manager.remove_pool(ClassId(100)));
}

#[test]
fn test_pool_manager_get_pool() {
    let manager = PoolManager::new();

    manager.create_pool(ClassId(100), TypeId::from_raw(100), 4);
    let pool = manager.get_pool(ClassId(100));
    assert!(pool.is_some());
    assert_eq!(pool.unwrap().class_id(), ClassId(100));
}

#[test]
fn test_pool_manager_multiple_pools() {
    let manager = PoolManager::new();

    manager.create_pool(ClassId(100), TypeId::from_raw(100), 4);
    manager.create_pool(ClassId(200), TypeId::from_raw(200), 2);
    manager.create_pool(ClassId(300), TypeId::from_raw(300), 8);

    assert_eq!(manager.pool_count(), 3);
    assert!(manager.has_pool(ClassId(100)));
    assert!(manager.has_pool(ClassId(200)));
    assert!(manager.has_pool(ClassId(300)));
}

#[test]
fn test_pool_manager_try_alloc_no_pool() {
    let manager = PoolManager::new();

    // No pool exists
    assert!(manager.try_alloc(ClassId(100)).is_none());
}

#[test]
fn test_pool_manager_try_alloc_empty_pool() {
    let manager = PoolManager::new();

    manager.create_pool(ClassId(100), TypeId::from_raw(100), 4);
    // Pool exists but is empty
    assert!(manager.try_alloc(ClassId(100)).is_none());
}

#[test]
fn test_pool_manager_try_dealloc_no_pool() {
    let manager = PoolManager::new();

    let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
    let ptr = NonNull::new(Box::into_raw(instance)).unwrap();

    // No pool exists
    assert!(!manager.try_dealloc(ClassId(100), ptr));

    // Clean up
    unsafe { drop(Box::from_raw(ptr.as_ptr())) };
}

#[test]
fn test_pool_manager_try_alloc_dealloc() {
    let manager = PoolManager::new();

    manager.create_pool(ClassId(100), TypeId::from_raw(100), 4);

    let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
    let ptr = NonNull::new(Box::into_raw(instance)).unwrap();

    // Dealloc to pool
    assert!(manager.try_dealloc(ClassId(100), ptr));

    // Alloc from pool
    let allocated = manager.try_alloc(ClassId(100));
    assert!(allocated.is_some());

    // Clean up
    unsafe { drop(Box::from_raw(allocated.unwrap().as_ptr())) };
}

#[test]
fn test_pool_manager_clear() {
    let manager = PoolManager::new();

    manager.create_pool(ClassId(100), TypeId::from_raw(100), 4);
    manager.create_pool(ClassId(200), TypeId::from_raw(200), 4);

    assert_eq!(manager.pool_count(), 2);

    manager.clear();
    assert_eq!(manager.pool_count(), 0);
}

#[test]
fn test_pool_manager_debug() {
    let manager = PoolManager::new();

    let debug_str = format!("{:?}", manager);
    assert!(debug_str.contains("PoolManager"));
    assert!(debug_str.contains("pool_count"));
}

// =========================================================================
// InstancePool Stress Tests
// =========================================================================

#[test]
fn test_instance_pool_lifo_order() {
    let pool = InstancePool::new(ClassId(100), TypeId::from_raw(100), 4);

    // Create and pool several instances
    let mut ptrs = Vec::new();
    for _ in 0..5 {
        let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
        let ptr = NonNull::new(Box::into_raw(instance)).unwrap();
        ptrs.push(ptr);
        pool.dealloc(ptr);
    }

    // Alloc should return in LIFO order
    let mut allocated = Vec::new();
    for _ in 0..5 {
        if let Some(ptr) = pool.alloc() {
            allocated.push(ptr);
        }
    }

    // Compare in reverse order (LIFO)
    for (i, ptr) in allocated.iter().enumerate() {
        assert_eq!(*ptr, ptrs[4 - i]);
    }

    // Clean up
    for ptr in allocated {
        unsafe { drop(Box::from_raw(ptr.as_ptr())) };
    }
}

#[test]
fn test_instance_pool_many_cycles() {
    let pool = InstancePool::with_max_size(ClassId(100), TypeId::from_raw(100), 4, 8);

    // Run multiple alloc/dealloc cycles
    for _cycle in 0..10 {
        // Dealloc phase
        let mut instances = Vec::new();
        for _ in 0..4 {
            let instance = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
            let ptr = NonNull::new(Box::into_raw(instance)).unwrap();
            pool.dealloc(ptr);
            instances.push(());
        }

        // Alloc phase
        let mut allocated = Vec::new();
        while let Some(ptr) = pool.alloc() {
            allocated.push(ptr);
        }

        // Clean up
        for ptr in allocated {
            unsafe { drop(Box::from_raw(ptr.as_ptr())) };
        }
    }

    // Pool should be empty after all cycles
    assert!(pool.is_empty());
}

#[test]
fn test_instance_pool_max_size_one() {
    let pool = InstancePool::with_max_size(ClassId(100), TypeId::from_raw(100), 4, 1);

    let i1 = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
    let p1 = NonNull::new(Box::into_raw(i1)).unwrap();

    let i2 = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
    let p2 = NonNull::new(Box::into_raw(i2)).unwrap();

    // First should succeed
    assert!(pool.dealloc(p1));
    assert_eq!(pool.pooled_count(), 1);

    // Second should fail (max size 1)
    assert!(!pool.dealloc(p2));
    assert_eq!(pool.pooled_count(), 1);

    // Clean up p2 manually
    unsafe { drop(Box::from_raw(p2.as_ptr())) };

    // Alloc the pooled one
    let allocated = pool.alloc().unwrap();
    unsafe { drop(Box::from_raw(allocated.as_ptr())) };
}

#[test]
fn test_instance_pool_is_empty_is_full() {
    let pool = InstancePool::with_max_size(ClassId(100), TypeId::from_raw(100), 4, 2);

    assert!(pool.is_empty());
    assert!(!pool.is_full());

    // Add one
    let i1 = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
    let p1 = NonNull::new(Box::into_raw(i1)).unwrap();
    pool.dealloc(p1);

    assert!(!pool.is_empty());
    assert!(!pool.is_full());

    // Add second (now full)
    let i2 = Box::new(PyInstanceObject::new(ClassId(100), TypeId::from_raw(100)));
    let p2 = NonNull::new(Box::into_raw(i2)).unwrap();
    pool.dealloc(p2);

    assert!(!pool.is_empty());
    assert!(pool.is_full());

    // Clean up
    while let Some(p) = pool.alloc() {
        unsafe { drop(Box::from_raw(p.as_ptr())) };
    }
}
