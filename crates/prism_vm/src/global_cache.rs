//! Version-guarded caches for name resolution.
//!
//! Python global lookup is on the critical path for ordinary code: function
//! calls, builtin helpers, imported constants, and module variables all start
//! with `LOAD_GLOBAL` or `LOAD_BUILTIN`. This cache keeps the common case to a
//! hash lookup keyed by bytecode site plus a few version comparisons while
//! preserving Python's dynamic shadowing semantics.

use prism_core::Value;
use rustc_hash::FxHashMap;

const MAX_GLOBAL_LOAD_CACHE_ENTRIES: usize = 1 << 15;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlobalLoadCacheKey {
    code: usize,
    name_index: u16,
    scope: usize,
}

impl GlobalLoadCacheKey {
    #[inline]
    pub(crate) fn new(code: usize, name_index: u16, scope: usize) -> Self {
        Self {
            code,
            name_index,
            scope,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GlobalLoadCacheContext {
    key: GlobalLoadCacheKey,
    namespace_version: u64,
    main_globals_version: u64,
    builtins_version: u64,
}

impl GlobalLoadCacheContext {
    #[inline]
    pub(crate) fn new(
        key: GlobalLoadCacheKey,
        namespace_version: u64,
        main_globals_version: u64,
        builtins_version: u64,
    ) -> Self {
        Self {
            key,
            namespace_version,
            main_globals_version,
            builtins_version,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GlobalLoadCacheEntry {
    namespace_version: u64,
    main_globals_version: u64,
    builtins_version: u64,
    value: Value,
}

/// Monomorphic global/builtin load cache.
#[derive(Debug, Default)]
pub(crate) struct GlobalLoadCache {
    entries: FxHashMap<GlobalLoadCacheKey, GlobalLoadCacheEntry>,
}

impl GlobalLoadCache {
    #[inline]
    pub(crate) fn get(&self, context: GlobalLoadCacheContext) -> Option<Value> {
        let entry = self.entries.get(&context.key)?;
        if entry.namespace_version == context.namespace_version
            && entry.main_globals_version == context.main_globals_version
            && entry.builtins_version == context.builtins_version
        {
            Some(entry.value)
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn insert(&mut self, context: GlobalLoadCacheContext, value: Value) {
        if self.entries.len() >= MAX_GLOBAL_LOAD_CACHE_ENTRIES {
            self.entries.clear();
        }
        self.entries.insert(
            context.key,
            GlobalLoadCacheEntry {
                namespace_version: context.namespace_version,
                main_globals_version: context.main_globals_version,
                builtins_version: context.builtins_version,
                value,
            },
        );
    }
}
