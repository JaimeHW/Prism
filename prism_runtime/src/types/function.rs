//! Function and closure objects.
//!
//! Implements Python function objects for the interpreter.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::Cell;
use crate::types::dict::DictObject;
use parking_lot::RwLock;
use prism_code::CodeObject;
use prism_core::Value;
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

// =============================================================================
// Closure Environment
// =============================================================================

/// Captured variable environment for closures.
///
/// Python closures capture cells, not plain values: sibling closures must see
/// mutations through the same cell object. The first four cells are stored
/// inline because small closures dominate Python code; larger environments use
/// a single shared overflow slice.
#[derive(Clone)]
pub struct ClosureEnv {
    /// Inline storage for small closure environments.
    inline_cells: [Option<Arc<Cell>>; 4],
    /// Number of cells in this environment.
    cell_count: usize,
    /// Overflow storage for environments with more than four cells.
    overflow: Option<Arc<[Arc<Cell>]>>,
    /// Parent closure environment (for nested closures).
    parent: Option<Arc<ClosureEnv>>,
}

impl ClosureEnv {
    /// Maximum number of cells stored inline.
    pub const INLINE_LIMIT: usize = 4;

    /// Create a new closure environment with captured cells.
    pub fn new(cells: Vec<Arc<Cell>>) -> Self {
        Self::with_parent(cells, None)
    }

    /// Create a closure environment with an explicit parent chain.
    pub fn with_parent(cells: Vec<Arc<Cell>>, parent: Option<Arc<ClosureEnv>>) -> Self {
        let cell_count = cells.len();

        if cells.len() <= Self::INLINE_LIMIT {
            let mut inline_cells: [Option<Arc<Cell>>; 4] = Default::default();
            for (idx, cell) in cells.into_iter().enumerate() {
                inline_cells[idx] = Some(cell);
            }
            Self {
                inline_cells,
                cell_count,
                overflow: None,
                parent,
            }
        } else {
            Self {
                inline_cells: Default::default(),
                cell_count,
                overflow: Some(cells.into()),
                parent,
            }
        }
    }

    /// Build an environment from values by wrapping each value in a cell.
    pub fn from_values(values: Box<[Value]>, parent: Option<Arc<ClosureEnv>>) -> Self {
        let cells = values
            .into_vec()
            .into_iter()
            .map(|value| Arc::new(Cell::new(value)))
            .collect();
        Self::with_parent(cells, parent)
    }

    /// Create an empty closure environment.
    pub fn empty() -> Self {
        Self {
            inline_cells: Default::default(),
            cell_count: 0,
            overflow: None,
            parent: None,
        }
    }

    /// Create an environment with pre-initialized unbound cells.
    pub fn with_unbound_cells(count: usize) -> Self {
        let cells = (0..count).map(|_| Arc::new(Cell::unbound())).collect();
        Self::new(cells)
    }

    /// Get a captured cell by index.
    #[inline]
    pub fn get_cell(&self, index: usize) -> &Arc<Cell> {
        self.try_get_cell(index).unwrap_or_else(|| {
            panic!(
                "closure cell index {index} out of bounds for {} cells",
                self.cell_count
            )
        })
    }

    #[inline]
    unsafe fn cell_at_unchecked(&self, index: usize) -> &Arc<Cell> {
        debug_assert!(index < self.cell_count);
        if self.overflow.is_none() {
            unsafe {
                self.inline_cells
                    .get_unchecked(index)
                    .as_ref()
                    .unwrap_unchecked()
            }
        } else {
            unsafe {
                self.overflow
                    .as_ref()
                    .unwrap_unchecked()
                    .get_unchecked(index)
            }
        }
    }

    /// Try to get a captured cell by index.
    #[inline]
    pub fn try_get_cell(&self, index: usize) -> Option<&Arc<Cell>> {
        if index >= self.len() {
            return None;
        }
        Some(unsafe { self.cell_at_unchecked(index) })
    }

    /// Get a captured value by index.
    #[inline]
    pub fn get(&self, index: usize) -> Value {
        self.get_cell(index).get_or_none()
    }

    /// Try to get a captured value by index.
    #[inline]
    pub fn try_get(&self, index: usize) -> Option<Value> {
        self.try_get_cell(index).map(|cell| cell.get_or_none())
    }

    /// Get a captured value by index, searching parent scopes.
    pub fn get_chain(&self, depth: usize, index: usize) -> Option<Value> {
        if depth == 0 {
            self.try_get(index)
        } else {
            self.parent.as_ref()?.get_chain(depth - 1, index)
        }
    }

    /// Set a captured value by index.
    #[inline]
    pub fn set(&self, index: usize, value: Value) -> bool {
        let Some(cell) = self.try_get_cell(index) else {
            return false;
        };
        cell.set(value);
        true
    }

    /// Get the number of captured values.
    #[inline]
    pub fn len(&self) -> usize {
        self.cell_count
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cell_count == 0
    }

    /// Whether all cells are stored inline.
    #[inline]
    pub fn is_inline(&self) -> bool {
        self.overflow.is_none()
    }

    /// Get parent environment.
    #[inline]
    pub fn parent(&self) -> Option<&Arc<ClosureEnv>> {
        self.parent.as_ref()
    }
}

impl std::fmt::Debug for ClosureEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosureEnv")
            .field("cell_count", &self.cell_count)
            .field("is_inline", &self.is_inline())
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

// =============================================================================
// Function Object
// =============================================================================

/// Python function object.
///
/// Represents a compiled function with its code, defaults, and closure.
#[derive(Default)]
struct FunctionAttrs {
    inline: FxHashMap<InternedString, Value>,
    dict_ptr: Option<NonNull<DictObject>>,
}

impl FunctionAttrs {
    #[inline]
    fn get(&self, name: &InternedString) -> Option<Value> {
        match self.dict_ptr {
            Some(ptr) => unsafe { ptr.as_ref() }.get(Value::string(name.clone())),
            None => self.inline.get(name).copied(),
        }
    }

    #[inline]
    fn set(&mut self, name: InternedString, value: Value) {
        match self.dict_ptr {
            Some(mut ptr) => unsafe { ptr.as_mut() }.set(Value::string(name), value),
            None => {
                self.inline.insert(name, value);
            }
        }
    }

    #[inline]
    fn remove(&mut self, name: &InternedString) -> Option<Value> {
        match self.dict_ptr {
            Some(mut ptr) => unsafe { ptr.as_mut() }.remove(Value::string(name.clone())),
            None => self.inline.remove(name),
        }
    }

    #[inline]
    fn contains(&self, name: &InternedString) -> bool {
        match self.dict_ptr {
            Some(ptr) => unsafe { ptr.as_ref() }.contains_key(Value::string(name.clone())),
            None => self.inline.contains_key(name),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self.dict_ptr {
            Some(ptr) => unsafe { ptr.as_ref() }.len(),
            None => self.inline.len(),
        }
    }

    #[inline]
    fn dict_ptr(&self) -> Option<*mut DictObject> {
        self.dict_ptr.map(NonNull::as_ptr)
    }

    fn materialize_dict<E, F>(&mut self, alloc: F) -> Result<*mut DictObject, E>
    where
        F: FnOnce(DictObject) -> Result<*mut DictObject, E>,
    {
        if let Some(ptr) = self.dict_ptr() {
            return Ok(ptr);
        }

        let mut dict = DictObject::with_capacity(self.inline.len());
        for (name, value) in &self.inline {
            dict.set(Value::string(name.clone()), *value);
        }

        let ptr = alloc(dict)?;
        let ptr = NonNull::new(ptr).expect("function attribute dict pointer must not be null");
        self.inline.clear();
        self.dict_ptr = Some(ptr);
        Ok(ptr.as_ptr())
    }

    fn for_each_value<F>(&self, mut f: F)
    where
        F: FnMut(Value),
    {
        match self.dict_ptr {
            Some(ptr) => {
                for value in unsafe { ptr.as_ref() }.values() {
                    f(value);
                }
            }
            None => {
                for value in self.inline.values() {
                    f(*value);
                }
            }
        }
    }
}

#[repr(C)]
pub struct FunctionObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Compiled bytecode.
    pub code: Arc<CodeObject>,
    /// Function name.
    pub name: Arc<str>,
    /// Default argument values.
    pub defaults: Option<Box<[Value]>>,
    /// Keyword-only defaults (name -> value).
    pub kwdefaults: Option<Box<[(Arc<str>, Value)]>>,
    /// Closure environment (captured variables).
    pub closure: Option<Arc<ClosureEnv>>,
    /// Global scope reference (module globals).
    /// This is a raw pointer to avoid Arc overhead on every call.
    /// The globals must outlive the function.
    globals_ptr: *const (),
    /// Lazily populated custom function attributes and optional live __dict__.
    attrs: RwLock<FunctionAttrs>,
    /// Native vectorcall override state used by C-API compatibility helpers.
    ///
    /// The hot function-call path reads this as one relaxed byte before the
    /// normal frame setup. Keeping it out of the attribute dictionary avoids a
    /// lock and keeps the default path branch-predictable.
    vectorcall_override: AtomicU8,
}

// Safety: FunctionObject is Send + Sync because:
// - All owned fields are Send + Sync (Arc, Box)
// - globals_ptr is only used for reading, and globals outlive functions
unsafe impl Send for FunctionObject {}
unsafe impl Sync for FunctionObject {}

impl FunctionObject {
    /// Create a new function object.
    pub fn new(
        code: Arc<CodeObject>,
        name: Arc<str>,
        defaults: Option<Box<[Value]>>,
        closure: Option<Arc<ClosureEnv>>,
    ) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::FUNCTION),
            code,
            name,
            defaults,
            kwdefaults: None,
            closure,
            globals_ptr: std::ptr::null(),
            attrs: RwLock::new(FunctionAttrs::default()),
            vectorcall_override: AtomicU8::new(0),
        }
    }

    /// Create a function with a specific globals pointer.
    ///
    /// # Safety
    /// The globals pointer must be valid for the lifetime of the function.
    pub unsafe fn with_globals(
        code: Arc<CodeObject>,
        name: Arc<str>,
        globals_ptr: *const (),
    ) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::FUNCTION),
            code,
            name,
            defaults: None,
            kwdefaults: None,
            closure: None,
            globals_ptr,
            attrs: RwLock::new(FunctionAttrs::default()),
            vectorcall_override: AtomicU8::new(0),
        }
    }

    /// Get the number of positional parameters.
    #[inline]
    pub fn arg_count(&self) -> u16 {
        self.code.arg_count
    }

    /// Get the number of keyword-only parameters.
    #[inline]
    pub fn kwonly_count(&self) -> u16 {
        self.code.kwonlyarg_count
    }

    /// Check if function takes *args.
    #[inline]
    pub fn has_varargs(&self) -> bool {
        self.code.flags.contains(prism_code::CodeFlags::VARARGS)
    }

    /// Check if function takes **kwargs.
    #[inline]
    pub fn has_varkw(&self) -> bool {
        self.code.flags.contains(prism_code::CodeFlags::VARKEYWORDS)
    }

    /// Get default value for parameter at index.
    pub fn get_default(&self, index: usize) -> Option<Value> {
        let defaults = self.defaults.as_ref()?;
        let num_required = (self.arg_count() as usize).saturating_sub(defaults.len());
        if index >= num_required {
            defaults.get(index - num_required).copied()
        } else {
            None
        }
    }

    /// Get the closure environment.
    #[inline]
    pub fn closure(&self) -> Option<&Arc<ClosureEnv>> {
        self.closure.as_ref()
    }

    /// Get a captured value from the closure.
    #[inline]
    pub fn get_closure_value(&self, index: usize) -> Option<Value> {
        self.closure.as_ref()?.try_get(index)
    }

    /// Get the raw module globals pointer captured when the function was defined.
    #[inline]
    pub fn globals_ptr(&self) -> *const () {
        self.globals_ptr
    }

    /// Get a custom function attribute.
    #[inline]
    pub fn get_attr(&self, name: &InternedString) -> Option<Value> {
        self.attrs.read().get(name)
    }

    /// Set a custom function attribute.
    #[inline]
    pub fn set_attr(&self, name: InternedString, value: Value) {
        self.attrs.write().set(name, value);
    }

    /// Delete a custom function attribute.
    #[inline]
    pub fn del_attr(&self, name: &InternedString) -> Option<Value> {
        self.attrs.write().remove(name)
    }

    /// Check whether a custom function attribute exists.
    #[inline]
    pub fn has_attr(&self, name: &InternedString) -> bool {
        self.attrs.read().contains(name)
    }

    /// Return the live function attribute dictionary if it has been materialized.
    #[inline]
    pub fn attr_dict_ptr(&self) -> Option<*mut DictObject> {
        self.attrs.read().dict_ptr()
    }

    /// Materialize and return the live function attribute dictionary.
    pub fn ensure_attr_dict<E, F>(&self, alloc: F) -> Result<*mut DictObject, E>
    where
        F: FnOnce(DictObject) -> Result<*mut DictObject, E>,
    {
        self.attrs.write().materialize_dict(alloc)
    }

    /// Visit each custom function attribute value.
    pub fn for_each_attr_value<F>(&self, mut f: F)
    where
        F: FnMut(Value),
    {
        self.attrs.read().for_each_value(|value| f(value));
    }

    /// Number of custom function attributes.
    #[inline]
    pub fn attr_len(&self) -> usize {
        self.attrs.read().len()
    }

    /// Update the function's module globals pointer.
    ///
    /// # Safety
    /// The pointer must remain valid for the lifetime of the function object.
    #[inline]
    pub unsafe fn set_globals_ptr(&mut self, globals_ptr: *const ()) {
        self.globals_ptr = globals_ptr;
    }

    /// Install the `_testcapi.function_setvectorcall` override.
    #[inline]
    pub fn set_test_vectorcall_override(&self) {
        self.vectorcall_override.store(1, Ordering::Release);
    }

    /// Whether calls should take the `_testcapi.function_setvectorcall` path.
    #[inline]
    pub fn has_test_vectorcall_override(&self) -> bool {
        self.vectorcall_override.load(Ordering::Acquire) != 0
    }
}

impl PyObject for FunctionObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Bound Method Object
// =============================================================================

/// Bound method - a function bound to an instance.
#[repr(C)]
pub struct BoundMethodObject {
    /// Object header.
    pub header: ObjectHeader,
    /// The underlying function.
    pub func: Arc<FunctionObject>,
    /// The bound instance (self).
    pub instance: Value,
}

impl BoundMethodObject {
    /// Create a new bound method.
    pub fn new(func: Arc<FunctionObject>, instance: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::METHOD),
            func,
            instance,
        }
    }
}

impl PyObject for BoundMethodObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}
