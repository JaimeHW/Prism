# Prism Architecture

Prism is organized as a layered Rust workspace. The primary architectural goal is
to keep hot execution paths small, explicit, and locally optimizable while keeping
semantic ownership clear enough that compatibility work does not sprawl across
unrelated crates.

## Layering

The intended dependency direction is:

```text
prism_core
  -> prism_gc
  -> prism_code
  -> prism_parser
  -> prism_compiler
  -> prism_runtime
  -> prism_jit
  -> prism_vm
  -> prism_aot
  -> prism_cli
```

This diagram is conceptual, not a literal chain. Crates can depend sideways when
there is a clear ownership reason, but lower-level crates must not learn about
higher-level execution policy.

## Crate Ownership

`prism_core` owns representation-neutral primitives: `Value`, spans, interning,
shared errors, and low-level speculation hints.

`prism_gc` owns heap spaces, tracing, barriers, roots, and collector policy. GC
contracts should be expressed through small traits and pointer/value operations,
not through VM-specific objects.

`prism_code` owns immutable bytecode and code-object representation. It should
remain format-centric and avoid depending on collector, VM, parser, or runtime
policy wherever possible.

`prism_parser` owns lexing, tokens, AST, and parsing. It should not know about
bytecode or runtime object layout.

`prism_compiler` owns AST-to-bytecode lowering, scope analysis, line tables, and
exception table emission. It may depend on parser and code format, but it should
not depend on VM execution details.

`prism_runtime` owns Python object layout, type slots, descriptors, containers,
and runtime data structures that are independent of a particular interpreter
instance.

`prism_jit` owns IR, lowering, optimization, code generation, register
allocation, inline-cache support, safepoints, and JIT runtime state. It should
depend on code/runtime representation, not on VM orchestration.

`prism_vm` owns interpreter execution, frame management, imports, native module
realization, tier-up orchestration, and the bridge between bytecode and runtime
state.

`prism_aot` owns ahead-of-time planning, frozen module bundling, manifests, and
linkable artifacts.

`prism_cli` owns command-line UX and process-level orchestration.

## Boundary Rules

1. Lower layers expose data contracts; upper layers choose execution policy.
2. Hot paths should prefer concrete types over trait objects unless dynamic
   dispatch is intentionally part of the design.
3. Public modules are API. Prefer `pub(crate)` for implementation modules and
   re-export narrow facade types from crate roots.
4. A crate should not depend on another crate solely for a single marker trait.
   Move the marker down, move the type up, or introduce an explicit adapter.
5. Test utilities should live near the layer they exercise. Shared fixtures are
   better than copying large integration harnesses.

## Current Refactor Targets

The following areas should be treated as priority architecture debt:

1. Split native stdlib and builtin method implementations out of `prism_vm` once
   active VM changes settle. The target is a smaller VM crate that owns
   execution, with native module implementations behind a stable registration
   interface.
2. Remove the `prism_code -> prism_gc` dependency. `KwNamesTuple` is currently a
   code-format helper with GC tracing behavior. The clean target is to make the
   code object store keyword-name metadata as code data and let the runtime
   materialize managed objects only when needed.
3. Split files over 1,500 lines by semantic domain. Large files should become
   module directories with a small `mod.rs` facade.
4. Reduce `prism_vm/src/lib.rs` to a narrow public API. Most execution internals
   should be `pub(crate)` and tested through VM behavior or dedicated facade
   types.
5. Move reusable benchmark and integration-test fixtures into crate-local
   support modules.

## File Size Policy

These are review triggers, not mechanical laws:

- More than 1,000 lines: justify the grouping.
- More than 1,500 lines: split unless it is generated or table-driven data.
- More than 3,000 lines: must have an active split plan before more features
  are added.

For performance-critical files, split by hot-path locality. Do not split a tight
dispatch loop just to satisfy a line count if doing so harms instruction-cache
behavior or inlining.

## Public API Policy

Crate roots should expose stable concepts, not every implementation module.

Use:

```rust
pub use module::{ImportantType, important_function};
```

Prefer:

```rust
mod lowering;
pub use lowering::LoweringPipeline;
```

Avoid:

```rust
pub mod lowering;
```

unless downstream crates need the module as an intentional extension surface.

## Performance Architecture Principles

1. Keep value representation and object layout explicit.
2. Keep interpreter dispatch predictable and table-driven.
3. Avoid allocation in opcode handlers unless Python semantics require it.
4. Make slow paths explicit and cold.
5. Preserve profiling data in compact, cache-friendly structures.
6. Treat deoptimization metadata as a correctness contract, not a debug aid.
7. Prefer source-level structure that helps the optimizer inline hot operations.

## Compatibility Architecture Principles

1. CPython compatibility tests should describe the semantic contract.
2. Native fast paths must preserve Python-visible exceptions and edge cases.
3. Source-backed stdlib fallback policy belongs in metadata, not scattered
   conditionals.
4. Platform-specific behavior must be isolated behind modules or target cfgs.

