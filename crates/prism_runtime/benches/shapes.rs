//! Shape System Performance Benchmarks
//!
//! Comprehensive benchmarks for the Hidden Classes (Shapes) implementation
//! measuring property access, shape transitions, and inline cache performance.
//!
//! # Benchmark Categories
//!
//! 1. **Property Access**: O(1) inline slot access vs O(n) shape chain lookup
//! 2. **Shape Transitions**: Cost of adding properties and reusing transitions
//! 3. **Shape Sharing**: Benefits of shared shapes across objects
//! 4. **IC Fast Paths**: Cached vs uncached property access performance
//! 5. **Overflow Storage**: Performance degradation beyond inline slots
//!
//! # Performance Targets
//!
//! - Inline slot access: < 10ns
//! - Shape lookup (existing property): < 50ns
//! - Shape transition: < 100ns with cached transition
//! - IC hit: Near-zero overhead over direct access

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::shape::{MAX_INLINE_SLOTS, ShapeRegistry};
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;

// =============================================================================
// Benchmark Helpers
// =============================================================================

/// Create a ShapedObject with N properties named "prop0", "prop1", etc.
fn create_object_with_n_properties(registry: &ShapeRegistry, n: usize) -> ShapedObject {
    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
    for i in 0..n {
        let name = intern(&format!("prop{}", i));
        obj.set_property(name, Value::int(i as i64).unwrap(), registry);
    }
    obj
}

/// Pre-intern property names for consistent benchmarking
fn intern_property_names(count: usize) -> Vec<prism_core::intern::InternedString> {
    (0..count).map(|i| intern(&format!("prop{}", i))).collect()
}

// =============================================================================
// Property Access Benchmarks
// =============================================================================

fn bench_property_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_access");

    // Inline slot access (direct path)
    group.bench_function("inline_slot_direct", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, 4);
        let name = intern("prop2");

        // Warm up to establish shape
        let _ = obj.get_property_interned(&name);

        b.iter(|| black_box(obj.get_property_interned(&name)))
    });

    // Inline slot via string lookup
    group.bench_function("inline_slot_str_lookup", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, 4);

        b.iter(|| black_box(obj.get_property("prop2")))
    });

    // Cached slot access (IC fast path)
    group.bench_function("cached_slot_access", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, 4);
        let slot_index = obj.shape().lookup("prop2").unwrap();

        b.iter(|| black_box(obj.get_property_cached(slot_index)))
    });

    // Compare inline access at different slot positions
    for slot in [0, 3, 7].iter() {
        if *slot < MAX_INLINE_SLOTS {
            group.bench_with_input(BenchmarkId::new("slot_position", slot), slot, |b, &slot| {
                let registry = ShapeRegistry::new();
                let obj = create_object_with_n_properties(&registry, MAX_INLINE_SLOTS);
                let prop_name = format!("prop{}", slot);

                b.iter(|| black_box(obj.get_property(&prop_name)))
            });
        }
    }

    group.finish();
}

// =============================================================================
// Property Write Benchmarks
// =============================================================================

fn bench_property_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_write");

    // Update existing property (no transition)
    group.bench_function("update_existing", |b| {
        let registry = ShapeRegistry::new();
        let mut obj = create_object_with_n_properties(&registry, 4);
        let name = intern("prop2");
        let value = Value::int(999).unwrap();

        b.iter(|| {
            obj.set_property(name.clone(), black_box(value), &registry);
        })
    });

    // Cached write (IC fast path)
    group.bench_function("cached_write", |b| {
        let registry = ShapeRegistry::new();
        let mut obj = create_object_with_n_properties(&registry, 4);
        let slot_index = obj.shape().lookup("prop2").unwrap();
        let value = Value::int(999).unwrap();

        b.iter(|| {
            obj.set_property_cached(slot_index, black_box(value));
        })
    });

    // Add new property (requires transition)
    group.bench_function("add_new_property", |b| {
        let registry = ShapeRegistry::new();
        let names = intern_property_names(100);

        b.iter(|| {
            let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
            for (i, name) in names.iter().take(4).enumerate() {
                obj.set_property(name.clone(), Value::int(i as i64).unwrap(), &registry);
            }
            black_box(obj)
        })
    });

    group.finish();
}

// =============================================================================
// Shape Transition Benchmarks
// =============================================================================

fn bench_shape_transitions(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_transitions");

    // Cached transition (reuses existing shape)
    group.bench_function("cached_transition", |b| {
        let registry = ShapeRegistry::new();
        let names = intern_property_names(4);

        // Prime the transition cache by creating first object
        let _ = create_object_with_n_properties(&registry, 4);

        b.iter(|| {
            let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
            for (i, name) in names.iter().enumerate() {
                obj.set_property(name.clone(), Value::int(i as i64).unwrap(), &registry);
            }
            black_box(obj)
        })
    });

    // Unique transitions (new property names each time)
    group.bench_function("unique_transitions", |b| {
        let registry = ShapeRegistry::new();
        let mut counter = 0usize;

        b.iter(|| {
            let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
            for i in 0..4 {
                let name = intern(&format!("unique_{}_{}", counter, i));
                obj.set_property(name, Value::int(i as i64).unwrap(), &registry);
            }
            counter += 1;
            black_box(obj)
        })
    });

    // Property count scaling
    for count in [1, 4, 8, 12, 16].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(
            BenchmarkId::new("property_count", count),
            count,
            |b, &count| {
                let registry = ShapeRegistry::new();
                let names = intern_property_names(count);

                // Prime transition cache
                let _ = create_object_with_n_properties(&registry, count);

                b.iter(|| {
                    let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
                    for (i, name) in names.iter().enumerate() {
                        obj.set_property(name.clone(), Value::int(i as i64).unwrap(), &registry);
                    }
                    black_box(obj)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Shape Sharing Benchmarks
// =============================================================================

fn bench_shape_sharing(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_sharing");

    // Create many objects with same shape
    group.bench_function("create_shared_shape_objects", |b| {
        let registry = ShapeRegistry::new();
        let names = intern_property_names(4);

        // Prime the transition cache
        let _ = create_object_with_n_properties(&registry, 4);

        b.iter(|| {
            let mut objects = Vec::with_capacity(100);
            for obj_idx in 0..100 {
                let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
                for (i, name) in names.iter().enumerate() {
                    obj.set_property(
                        name.clone(),
                        Value::int((obj_idx * 4 + i) as i64).unwrap(),
                        &registry,
                    );
                }
                objects.push(obj);
            }
            black_box(objects)
        })
    });

    // Verify objects share shapes
    group.bench_function("verify_shape_sharing", |b| {
        let registry = ShapeRegistry::new();
        let names = intern_property_names(4);

        // Create 100 objects with same properties
        let objects: Vec<_> = (0..100)
            .map(|obj_idx| {
                let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
                for (i, name) in names.iter().enumerate() {
                    obj.set_property(
                        name.clone(),
                        Value::int((obj_idx * 4 + i) as i64).unwrap(),
                        &registry,
                    );
                }
                obj
            })
            .collect();

        // All objects should have same shape ID
        let expected_shape_id = objects[0].shape_id();

        b.iter(|| {
            let all_same = objects.iter().all(|o| o.shape_id() == expected_shape_id);
            black_box(all_same)
        })
    });

    group.finish();
}

// =============================================================================
// Shape Lookup Benchmarks
// =============================================================================

fn bench_shape_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_lookup");

    // Single property lookup
    group.bench_function("lookup_single", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, 8);
        let shape = obj.shape().clone();

        b.iter(|| black_box(shape.lookup("prop4")))
    });

    // Lookup with interned string (fast path)
    group.bench_function("lookup_interned", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, 8);
        let shape = obj.shape().clone();
        let name = intern("prop4");

        b.iter(|| black_box(shape.lookup_interned(&name)))
    });

    // Lookup miss (property doesn't exist)
    group.bench_function("lookup_miss", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, 8);
        let shape = obj.shape().clone();

        b.iter(|| black_box(shape.lookup("nonexistent")))
    });

    // Property count scaling for lookup
    for count in [1, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("lookup_chain_length", count),
            count,
            |b, &count| {
                let registry = ShapeRegistry::new();
                let obj = create_object_with_n_properties(&registry, count);
                let shape = obj.shape().clone();
                let target = format!("prop{}", count - 1);

                b.iter(|| black_box(shape.lookup(&target)))
            },
        );
    }

    group.finish();
}

// =============================================================================
// Overflow Storage Benchmarks
// =============================================================================

fn bench_overflow_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("overflow_storage");
    group.sample_size(50); // Reduce for expensive operations

    // Access property in overflow storage
    group.bench_function("overflow_access", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, MAX_INLINE_SLOTS + 4);

        // Property in overflow (beyond inline slots)
        let overflow_prop = format!("prop{}", MAX_INLINE_SLOTS + 2);

        b.iter(|| black_box(obj.get_property(&overflow_prop)))
    });

    // Write to overflow storage
    group.bench_function("overflow_write", |b| {
        let registry = ShapeRegistry::new();
        let mut obj = create_object_with_n_properties(&registry, MAX_INLINE_SLOTS + 4);
        let name = intern(&format!("prop{}", MAX_INLINE_SLOTS + 2));
        let value = Value::int(999).unwrap();

        b.iter(|| {
            obj.set_property(name.clone(), black_box(value), &registry);
        })
    });

    // Compare inline vs overflow access
    group.bench_function("inline_vs_overflow_inline", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, MAX_INLINE_SLOTS + 4);

        b.iter(|| black_box(obj.get_property("prop0")))
    });

    group.bench_function("inline_vs_overflow_overflow", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, MAX_INLINE_SLOTS + 4);
        let overflow_prop = format!("prop{}", MAX_INLINE_SLOTS);

        b.iter(|| black_box(obj.get_property(&overflow_prop)))
    });

    group.finish();
}

// =============================================================================
// IC Simulation Benchmarks
// =============================================================================

fn bench_ic_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ic_simulation");

    // Simulate IC hit: shape check + cached access
    group.bench_function("ic_hit", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, 4);
        let cached_shape_id = obj.shape_id();
        let cached_slot = obj.shape().lookup("prop2").unwrap();

        b.iter(|| {
            // IC check
            if obj.shape_id() == cached_shape_id {
                // Fast path
                black_box(obj.get_property_cached(cached_slot))
            } else {
                // Slow path (should not happen in this benchmark)
                black_box(obj.get_property("prop2").unwrap())
            }
        })
    });

    // Simulate IC miss: shape mismatch requiring slow path
    group.bench_function("ic_miss_slow_path", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, 4);

        // Simulate wrong cached shape
        let wrong_shape_id = registry.empty_shape().id();

        b.iter(|| {
            // IC check fails
            if obj.shape_id() == wrong_shape_id {
                // Would be fast path
                unreachable!()
            } else {
                // Slow path
                black_box(obj.get_property("prop2").unwrap())
            }
        })
    });

    // Compare IC overhead vs direct access
    group.bench_function("direct_access_baseline", |b| {
        let registry = ShapeRegistry::new();
        let obj = create_object_with_n_properties(&registry, 4);
        let slot = obj.shape().lookup("prop2").unwrap();

        b.iter(|| black_box(obj.get_property_cached(slot)))
    });

    group.finish();
}

// =============================================================================
// Registry Benchmarks
// =============================================================================

fn bench_registry(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry");

    // Transition with cached result
    group.bench_function("transition_cached", |b| {
        let registry = ShapeRegistry::new();
        let name = intern("x");

        // Prime the cache
        let empty = registry.empty_shape();
        let _ = registry.transition_default(&empty, name.clone());

        b.iter(|| {
            let empty = registry.empty_shape();
            black_box(registry.transition_default(&empty, name.clone()))
        })
    });

    // Transition creating new shape
    group.bench_function("transition_new", |b| {
        let registry = ShapeRegistry::new();
        let mut counter = 0usize;

        b.iter(|| {
            let name = intern(&format!("unique_{}", counter));
            counter += 1;
            let empty = registry.empty_shape();
            black_box(registry.transition_default(&empty, name))
        })
    });

    // Stats retrieval
    group.bench_function("get_stats", |b| {
        let registry = ShapeRegistry::new();

        // Create some shapes first
        for i in 0..100 {
            let name = intern(&format!("prop{}", i));
            let empty = registry.empty_shape();
            let _ = registry.transition_default(&empty, name);
        }

        b.iter(|| black_box(registry.stats()))
    });

    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    shape_benches,
    bench_property_access,
    bench_property_write,
    bench_shape_transitions,
    bench_shape_sharing,
    bench_shape_lookup,
    bench_overflow_storage,
    bench_ic_simulation,
    bench_registry,
);

criterion_main!(shape_benches);
