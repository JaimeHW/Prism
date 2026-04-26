# Performance Engineering

Prism's performance bar is not "fast for a prototype." It should be engineered
as a runtime where every allocation, branch, cache miss, barrier, and deopt point
has an owner.

## Optimization Priorities

1. Correct Python semantics.
2. Predictable low-latency interpreter execution.
3. Low-allocation native builtins and stdlib modules.
4. Stable profiling and tier-up decisions.
5. High-quality JIT lowering and deoptimization metadata.
6. GC throughput without compromising pause-time predictability.

Correctness comes first because a fast wrong runtime cannot be optimized into a
production runtime.

## Hot-Path Rules

Hot paths include opcode handlers, builtin dispatch, attribute lookup, call
binding, iterator stepping, allocation, barriers, inline caches, and JIT stubs.

Rules:

- Avoid heap allocation unless it is required by Python semantics.
- Prefer `SmallVec`, stack arrays, or preallocated scratch storage only when
  measurements show benefit.
- Keep cold error formatting out of the hot path.
- Mark intentionally cold helpers with `#[cold]` where it helps code layout.
- Use `#[inline]` for tiny wrappers and representation accessors; avoid
  blanket `#[inline(always)]` unless benchmarks prove it.
- Keep lookup keys compact and hash only once per operation when possible.
- Keep object-header loads minimal and reuse extracted `TypeId` values.
- Preserve monomorphic fast paths before falling back to generic protocol logic.

## Allocation Policy

Every new allocation in a hot path should answer:

1. Is this required by Python-visible semantics?
2. Can the allocation be delayed until escape?
3. Can the value be represented as an immediate?
4. Can an existing object, interned string, or cached descriptor be reused?
5. Does this allocation need a write barrier or root protection?

If the allocation is intentional, tests should cover both correctness and the
path that keeps the object alive across GC.

## Compiler Policy

The compiler should produce bytecode that is friendly to both the interpreter and
the JIT:

- Keep register pressure visible and bounded.
- Prefer canonical instruction sequences over many equivalent shapes.
- Emit precise line and exception tables.
- Keep call metadata compact and directly indexable.
- Avoid encoding runtime-only objects into code-format data when plain metadata
  would be sufficient.

## JIT Policy

JIT changes should define:

- Which tier owns the optimization.
- Which guards protect the specialization.
- Which deopt reason is used when a guard fails.
- Which interpreter state must be reconstructed.
- Which profiling signal justifies tier-up.

Every speculative optimization needs a deopt path and a test that exercises the
fallback.

## GC Policy

GC-sensitive changes should define:

- Object ownership domain.
- Trace implementation and invariants.
- Write-barrier responsibility.
- Root lifetime.
- Movement or pinning assumptions.
- Interaction with JIT stack maps or safepoints.

Missing a barrier is a correctness bug even if the benchmark improves.

## Benchmarking

Benchmarks should be grouped by question:

- Interpreter dispatch throughput.
- Call and method dispatch.
- Attribute access and inline caches.
- Container operations.
- String and bytes operations.
- GC allocation and pause behavior.
- Tier-up, deopt, and OSR.
- End-to-end Python workloads.

For performance PRs, capture:

- Commit SHA.
- Host CPU and OS.
- Rust version.
- Benchmark command.
- Before/after median.
- Noise estimate or confidence interval.
- Correctness tests run.

Do not accept a microbenchmark win that regresses end-to-end compatibility or
common latency paths.

## Release Profile

The workspace release profile is intentionally aggressive:

- Fat LTO.
- Single codegen unit.
- Abort on panic.
- Strip symbols.

This favors final runtime performance over compile time for release artifacts.
Use development builds for iteration and release builds for performance claims.

