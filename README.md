<p align="center">
  <h1 align="center">Prism</h1>
  <p align="center">
    A high-performance Python 3.12 runtime written from scratch in Rust.
    <br />
    <em>NaN-boxed values · Register-based VM · Tiered JIT · Generational GC</em>
  </p>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#tooling">Tooling</a> ·
  <a href="#building">Building</a> ·
  <a href="#project-status">Status</a> ·
  <a href="#license">License</a>
</p>

---

Prism is a ground-up Python 3.12 implementation: parser, compiler, virtual machine, garbage collector, JIT compiler, standard library, and AOT tooling. The project is organized as an 11-crate workspace with explicit ownership boundaries for source compilation, runtime execution, and stdlib import metadata.

The codebase spans roughly 350k lines of Rust across the workspace.

## Quickstart

```bash
# Run a script
cargo run -p prism_cli --bin prism -- script.py

# Execute an inline expression
cargo run -p prism_cli --bin prism -- -c "print('hello from Prism')"

# Run a module
cargo run -p prism_cli --bin prism -- -m package.module

# Start the REPL
cargo run -p prism_cli --bin prism
```

## Architecture

```text
                  prism_cli
         prism / prismc / prism-test
                    |
        +-----------+-----------+
        |                       |
     prism_vm               prism_aot
  interpreter, imports      build planner
        |                       |
        |                  prism_stdlib
        |                 stdlib metadata
        |
  prism_runtime <---- prism_gc
   object model      collector
        |
    prism_jit
  tiered codegen

prism_parser -> prism_compiler -> prism_code
   lexer/AST     source frontend    bytecode format
                 scope analysis
                 code generation
```

### Crate Overview

| Crate | Purpose |
|---|---|
| **prism_core** | NaN-boxed value representation, string interning, error types, span tracking, speculation primitives |
| **prism_code** | Shared bytecode instruction set and `CodeObject` format |
| **prism_parser** | Full Python 3.12 lexer, parser, and AST |
| **prism_compiler** | Source frontend plus two-phase compiler: parse/compile entrypoints, scope analysis, and register-based bytecode emission |
| **prism_stdlib** | Static stdlib inventory and import-policy metadata shared by AOT planning and VM import machinery |
| **prism_runtime** | Python object model, type slots, descriptors, metaclasses, hidden classes, and GC integration |
| **prism_gc** | Generational garbage collector with nursery, tenured space, large-object handling, and write barriers |
| **prism_jit** | Tiered JIT with Sea-of-Nodes IR, optimization passes, register allocation, and x64 code generation |
| **prism_vm** | Bytecode interpreter, inline caching, profiling and tier-up, import system, and native stdlib execution |
| **prism_aot** | Ahead-of-time build planner, frozen-module bundler, manifest emitter, and object-file generation |
| **prism_cli** | `prism` interpreter, `prismc` AOT driver, and `prism-test` compatibility harness |

### Ownership Boundaries

- `prism_compiler` owns source-to-bytecode frontend entrypoints. CLI execution, REPL evaluation, VM source imports, and AOT planning all compile source through the same API.
- `prism_stdlib` owns static stdlib capability metadata. Planning code no longer reaches into VM runtime construction details just to answer "is this a native stdlib module?"
- `prism_vm` owns execution, import realization, and native module behavior. It consumes compiler output and stdlib metadata instead of being the source of both.

---

## Key Design Decisions

### NaN-Boxed Values

Every Python value fits in a single 64-bit word. IEEE 754 quiet-NaN bit patterns encode type tags for `None`, booleans, integers, floats, object pointers, and interned strings. That keeps the float fast path cheap while still supporting inline small integers.

### Register-Based VM

Prism uses a register-based bytecode format rather than a pure stack VM. The design favors straightforward lowering, fewer transient moves in hot paths, and a cleaner handoff to tiered JIT compilation.

### Tiered JIT Compilation

| Tier | Role |
|---|---|
| **Tier 0** | Bytecode interpreter with profiling counters and type feedback |
| **Tier 1** | Baseline JIT with fast template compilation and deoptimization guards |
| **Tier 2** | Optimizing JIT with Sea-of-Nodes IR and profile-guided optimization |

On-stack replacement (OSR) promotes hot loops from interpreter to JIT mid-execution. Deoptimization safely returns execution to the interpreter when speculative guards fail.

### Generational Garbage Collector

- Nursery collection uses bump-pointer allocation and copying evacuation.
- Tenured collection supports mark/sweep and compaction paths.
- Large objects are allocated out of line.
- Write barriers track old-to-young references for generational correctness.
- JIT-generated code participates through safepoints and precise stack maps.

### Hidden Classes

Objects share structural metadata through shape transitions, which lets inline caches resolve attribute offsets without a full dictionary lookup in the common case.

---

## Tooling

### `prism` - Interpreter

```bash
prism script.py
prism -c "expr"
prism -m module
prism
```

Supported flags include `-O`, `-OO`, `-B`, `-E`, `-s`, `-S`, `-u`, `-v`, `-W`, and `-X`. JIT controls are available through `-X jit=on`, `-X jit=off`, and `-X nojit`.

### `prismc` - AOT Compiler

```bash
prismc build app.py -O
prismc build -m pkg.tool -I vendor -OO
```

Outputs under `prism-build/` include:

- `build-plan.json` for the deterministic build manifest
- `frozen-modules.prism` for the serialized bytecode bundle
- `frozen-modules.obj` for the Windows object payload used by embedding flows

### `prism-test` - CPython Compatibility Harness

```bash
prism-test --cpython-root /path/to/cpython-3.12 test_grammar test_math
prism-test --cpython-root /path/to/cpython-3.12 --json-report results.json test_grammar
```

The harness supports multiple runner modes, per-test isolation, timeout control, and JSON reporting.

---

## Standard Library

Prism ships native Rust implementations for core modules and can fall back to filesystem-backed CPython stdlib source when it improves correctness.

Native builtin/import metadata is defined once in `prism_stdlib` and consumed by both the AOT planner and the VM import system. Modules such as `os`, `os.path`, `json`, and `functools` can prefer source-backed loading when appropriate, while modules such as `math`, `sys`, `time`, and `io` stay native.

The import system also supports frozen source modules so the AOT pipeline can bundle code without runtime filesystem access.

---

## Building

### Prerequisites

- Rust 1.94+ (edition 2024)
- x86-64 target support
- Optional: a CPython 3.12 source checkout for `prism-test`

### Build, Test, Bench

```bash
cargo build --workspace
cargo build --workspace --release
cargo test --workspace
cargo bench --workspace
```

## Engineering Docs

- [Architecture](docs/ARCHITECTURE.md) - crate ownership, layering, public API
  policy, and current refactor targets
- [Performance Engineering](docs/PERFORMANCE.md) - hot-path, allocation, JIT,
  GC, and benchmarking standards
- [Testing Strategy](docs/TESTING.md) - local checks, CI contract, compatibility
  tests, and benchmark expectations
- [Unsafe Code Policy](docs/UNSAFE.md) - required invariants for GC, value
  representation, JIT, and platform code
- [Contributing](CONTRIBUTING.md) - commit style, review expectations, and
  pre-PR checks

---

## Project Status

Prism is under active development. The runtime is architecturally broad, from parser through JIT, and it already runs meaningful end-to-end Python programs. Compatibility work continues across the standard library and CPython edge-case behavior.

### What Works Today

- Python 3.12 parsing and AST support across the language surface
- Source-to-bytecode compilation through shared frontend APIs
- Bytecode interpretation, profiling, and tier-up into JIT execution
- Generational GC with barriers and JIT safepoints
- Native stdlib coverage plus source-backed fallback imports
- AOT planning with frozen-module bundling
- CPython compatibility testing infrastructure

### In Progress

- Broader stdlib coverage and behavior parity
- ARM64 JIT backend work
- More concurrent GC infrastructure
- Standalone executable packaging from the AOT pipeline

For the most current runtime behavior, inspect the integration tests under `prism_vm/tests`, the CLI test suites under `prism_cli/tests`, and the `prism-test` harness.

---

## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.
