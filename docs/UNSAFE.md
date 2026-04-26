# Unsafe Code Policy

Prism uses unsafe Rust in places where a high-performance Python runtime needs
precise control over values, object layout, GC, JIT code, and platform APIs. The
standard for unsafe code is therefore high: every unsafe block must have a local
reason and a testable invariant.

## Requirements

Each unsafe block or unsafe impl must document:

1. Which invariant makes the operation sound.
2. Who owns the pointer or memory region.
3. Whether the object can move.
4. Whether the value is rooted or otherwise protected from GC.
5. Whether aliasing or mutability assumptions are involved.
6. Which tests or runtime assertions cover the invariant.

Use `// SAFETY:` comments immediately before unsafe blocks. The comment should be
specific to that block, not a generic statement that the caller is responsible.

## GC Safety

When unsafe code touches GC-managed memory, verify:

- The object is initialized before it can be traced.
- All child references are traced.
- Pointer stores trigger the required write barrier.
- Raw pointers do not outlive their owning heap or standalone allocation.
- JIT and interpreter frames expose roots at safepoints.

If a type implements `Trace`, missing a reference is a memory-safety bug.

## Value Representation

When working with `Value` bit patterns:

- Keep tag masks and payload masks centralized.
- Use existing constructors and accessors unless a measured hot path needs a
  lower-level operation.
- Test boundary values for small integers, floats, strings, and object pointers.
- Avoid transmute when bit operations or pointer casts are clearer.

## JIT Safety

Generated code and code patching must define:

- Executable memory ownership.
- W^X or platform-equivalent transition behavior.
- Calling convention.
- Register preservation.
- Stack alignment.
- Safepoint metadata.
- Deopt state reconstruction.

Any new machine-code path needs tests for normal return, exception/deopt return,
and GC interaction when applicable.

## FFI and Platform APIs

Platform calls should be isolated behind narrow wrappers. The wrapper owns:

- Argument validation.
- Error conversion.
- Handle ownership.
- Encoding conversion.
- Cleanup on early returns.

Tests should cover both success and failure paths where the platform makes that
practical.

## Review Checklist

Before merging unsafe code:

- The unsafe region is as small as practical.
- The safety comment names concrete invariants.
- There is no avoidable safe alternative in the hot path.
- Tests exercise invalid inputs and edge cases.
- GC and JIT interactions are explicitly considered.
- Benchmarks justify any unsafe optimization whose only benefit is speed.

