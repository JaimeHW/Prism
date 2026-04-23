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

Prism is a ground-up Python 3.12 implementation—parser, compiler, virtual machine, garbage collector, JIT compiler, standard library, and AOT tooling—designed to explore how far a Rust-native runtime can push Python performance.

The codebase spans **~350 k lines of Rust** across 11 workspace crates.

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
                    ┌──────────────────────────────────────────────────────┐
                    │                    prism_cli                        │
                    │         prism · prismc · prism-test                 │
                    └────────────┬──────────────┬──────────────┬──────────┘
                                 │              │              │
                    ┌────────────▼──┐    ┌──────▼──────┐  ┌────▼──────┐
                    │   prism_vm    │    │  prism_aot  │  │prism_code │
                    │  interpreter  │    │  AOT driver │  │ bytecode  │
                    │  stdlib · IC  │    └─────────────┘  │  format   │
                    │  import sys   │                     └───────────┘
                    └──┬─────┬─────┘
                       │     │
          ┌────────────▼┐  ┌─▼────────────┐
          │  prism_jit   │  │prism_runtime │
          │  Sea-of-Nodes│  │ object model │
          │  IR + x64    │  │ type system  │
          └──────┬───────┘  └──────┬───────┘
                 │                 │
          ┌──────▼─────────────────▼───────┐
          │           prism_gc             │
          │   generational collector       │
          └────────────┬───────────────────┘
                       │
          ┌────────────▼───────────────────┐
          │          prism_core            │
          │  NaN-boxed values · interning  │
          │  spans · errors · speculation  │
          └────────────────────────────────┘

  prism_parser ──► prism_compiler ──► prism_code
    lexer/AST      scope analysis      bytecode
                   code generation     code objects
```

### Crate Overview

| Crate | Purpose |
|---|---|
| **prism_core** | NaN-boxed value representation, string interning, error types, span tracking, speculation primitives |
| **prism_code** | Shared bytecode instruction set and `CodeObject` format |
| **prism_parser** | Full Python 3.12 lexer, parser, and AST (f-strings, pattern matching, type params) |
| **prism_compiler** | Two-phase compiler: scope analysis → register-based bytecode emission |
| **prism_runtime** | Python object model—`ObjectHeader`, type slots, descriptors, metaclasses, hidden classes (shapes), GC integration |
| **prism_gc** | Generational garbage collector: nursery (copying), tenured (mark-sweep/compact), large-object space, card-table write barriers, TLABs |
| **prism_jit** | Tiered JIT: Sea-of-Nodes IR, 18 optimization passes, graph-coloring register allocation, x64 native code generation, precise GC stack maps |
| **prism_vm** | Bytecode interpreter, inline caching, profiling & tier-up, JIT bridge & OSR, import system, 40+ native stdlib modules |
| **prism_aot** | Ahead-of-time build planner, frozen-module bundler, manifest emitter, Windows object-file generator |
| **prism_builtins** | Shared builtin runtime helpers |
| **prism_cli** | `prism` (interpreter), `prismc` (AOT driver), `prism-test` (CPython compat harness) |

---

## Key Design Decisions

### NaN-Boxed Values

Every Python value fits in a single 64-bit word. IEEE 754 quiet-NaN bit patterns encode type tags (None, bool, int, float, object pointer, interned string) with zero-cost unboxing for the hot float path and 48-bit inline small integers (±140 trillion).

### Register-Based VM

256 registers per call frame, inlined on the stack for L1-cache locality. Static dispatch table for O(1) opcode lookup. Monomorphic and polymorphic inline caches accelerate attribute access and call sites.

### Tiered JIT Compilation

| Tier | Role |
|---|---|
| **Tier 0** | Bytecode interpreter with profiling counters and type feedback |
| **Tier 1** | Baseline JIT—fast template compilation with deoptimization guards |
| **Tier 2** | Optimizing JIT—Sea-of-Nodes IR with full optimization pipeline and PGO |

The **optimization pipeline** runs 18 passes across six ordered phases:

| Phase | Passes |
|---|---|
| Canonicalization | SCCP, algebraic simplification, instruction combining |
| Profile-Guided | Branch probability estimation, hot/cold splitting |
| Local | Copy propagation, GVN, dead store elimination, PRE, strength reduction |
| Loop | LICM, loop unrolling, range-check elimination |
| Interprocedural | Inlining, escape analysis, tail-call optimization |
| Cleanup | Dead-code elimination |

On-stack replacement (OSR) promotes hot loops from interpreter to JIT mid-execution. Deoptimization safely falls back to the interpreter when speculative guards fail.

### Generational Garbage Collector

- **Nursery**: bump-pointer allocation, copying collection, TLAB support
- **Tenured**: mark-sweep with optional compaction (triggered at 30% fragmentation)
- **Large object space**: direct allocation for objects > 8 KB
- **Write barriers**: card-table tracked old→young references
- **Tri-color marking**: supports concurrent marking infrastructure
- **JIT integration**: precise GC via stack maps and safepoints; write-barrier emission in generated code

### Hidden Classes (Shapes)

Objects share structural metadata through shape transitions, enabling inline caches to resolve attribute offsets in constant time without dictionary lookups.

---

## Tooling

### `prism` — Interpreter

Drop-in CLI compatible with CPython's argument surface:

```bash
prism script.py                # Run a file
prism -c "expr"                # Execute a string
prism -m module                # Run a module
prism                          # Interactive REPL
```

Supported flags: `-O`, `-OO`, `-B`, `-E`, `-s`, `-S`, `-u`, `-v`, `-W`, `-X`.  
JIT controls via `-X jit=on`, `-X jit=off`, `-X nojit`.

### `prismc` — AOT Compiler

Build planner and frozen-module artifact generator:

```bash
prismc build app.py -O
prismc build -m pkg.tool -I vendor -OO
```

Outputs (under `prism-build/`):
- `build-plan.json` — deterministic build manifest
- `frozen-modules.prism` — serialized bytecode bundle
- `frozen-modules.obj` — Windows PE object for embedding

### `prism-test` — CPython Compatibility Harness

Runs Prism against a CPython 3.12 source checkout:

```bash
prism-test --cpython-root /path/to/cpython-3.12 test_grammar test_math
prism-test --cpython-root /path/to/cpython-3.12 --json-report results.json test_grammar
```

Supports multiple runner modes (`import`, `suite`, `test-main`), per-test isolation, timeout control, and JSON reporting.

---

## Standard Library

Prism ships native Rust implementations for core modules and can fall back to CPython's source stdlib when available on `sys.path`.

<details>
<summary><strong>Native modules (40+)</strong></summary>

**Core**: `builtins`, `sys`, `gc`, `marshal`, `inspect`, `typing`, `signal`, `atexit`

**Data & Math**: `math`, `itertools`, `collections`, `re`, `functools`, `fnmatch`

**I/O & System**: `io`, `time`, `os`, `nt`, `errno`, `json`

**Cryptography & Random**: `_sha2`, `_random`, `_secrets`

**Internals**: `_abc`, `_ast`, `_codecs`, `_contextvars`, `_imp`, `_string`, `_struct`, `_thread`, `_tokenize`, `_warnings`, `_weakref`, `weakref`

**Windows**: `nt`, `winreg`, `_winapi`

</details>

Modules like `os`, `os.path`, `json`, and `functools` can prefer filesystem-backed CPython source when it improves correctness. The import system supports frozen source modules, enabling the AOT pipeline to bundle code without runtime filesystem access.

---

## Building

### Prerequisites

- **Rust 1.85+** (2024 edition)
- **x86-64** target (primary platform; ARM64 backend is in progress)
- *Optional*: CPython 3.12 source checkout for `prism-test`

### Build, Test, Bench

```bash
cargo build --workspace            # Debug build
cargo build --workspace --release  # Optimized build (LTO + single codegen unit)
cargo test  --workspace            # Run test suite
cargo bench --workspace            # Criterion benchmarks
```

Benchmarks cover JIT speedup, GC pause times, deoptimization/OSR, speculation, string operations, hidden-class shapes, and small-integer caching.

---

## Project Status

Prism is under **active development**. The runtime is architecturally complete—parser through JIT—and runs meaningful end-to-end Python programs. Compatibility work continues across the broader standard library and CPython edge-case semantics.

**What works today:**
- Full Python 3.12 syntax (comprehensions, pattern matching, async/await, f-strings, decorators, metaclasses, generators, `yield from`)
- Tiered execution from interpreter through optimizing JIT
- Generational GC with write barriers and JIT safepoints
- 40+ native stdlib modules
- AOT build planning with frozen-module bundling
- CPython compatibility testing harness

**What's in progress:**
- Broader CPython stdlib coverage and behavioral edge cases
- ARM64 JIT backend
- Concurrent GC marking
- Standalone executable packaging from the AOT pipeline

For the most current picture of runtime behavior, see the integration tests in [`prism_vm/tests`](prism_vm/tests), CLI tests in [`prism_cli`](prism_cli/src/cpython_tests.rs), and the `prism-test` harness for CPython regression tracking.

---

## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.
