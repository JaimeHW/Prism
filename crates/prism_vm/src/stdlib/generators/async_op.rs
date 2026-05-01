//! Awaitable operation objects for native async generators.
//!
//! CPython exposes `agen.__anext__()`, `agen.asend(...)`, `agen.athrow(...)`,
//! and `agen.aclose()` as coroutine-like helper objects. Prism keeps the same
//! protocol shape so asyncio and `collections.abc.Coroutine` can drive async
//! generator operations through `send`, `throw`, `close`, and `__await__`.

use prism_core::Value;
use prism_gc::Trace;
use prism_gc::trace::Tracer;
use prism_runtime::gc_dispatch::{DispatchEntry, register_external_dispatch};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use std::mem;
use std::sync::Once;

static ASYNC_GENERATOR_OPERATION_GC_DISPATCH_ONCE: Once = Once::new();

/// The async-generator helper operation to perform when the object is driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AsyncGeneratorOperationKind {
    /// Resume the generator with an initial send value.
    ASend,
    /// Raise an exception into the generator.
    AThrow,
    /// Close the generator by raising `GeneratorExit` and suppressing it.
    AClose,
}

impl AsyncGeneratorOperationKind {
    #[inline(always)]
    pub fn type_id(self) -> TypeId {
        match self {
            Self::ASend => TypeId::ASYNC_GENERATOR_ASEND,
            Self::AThrow | Self::AClose => TypeId::ASYNC_GENERATOR_ATHROW,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum AsyncGeneratorOperationState {
    Created,
    Running,
    Closed,
}

/// Coroutine-like awaitable backing one async-generator operation.
#[repr(C)]
pub struct AsyncGeneratorOperationObject {
    pub header: ObjectHeader,
    generator: Value,
    send_value: Value,
    exception: Value,
    exception_type_id: u16,
    kind: AsyncGeneratorOperationKind,
    state: AsyncGeneratorOperationState,
    _pad: [u8; 4],
}

impl AsyncGeneratorOperationObject {
    #[inline]
    pub fn new_asend(generator: Value, send_value: Value) -> Self {
        Self::new(
            AsyncGeneratorOperationKind::ASend,
            generator,
            send_value,
            Value::none(),
            0,
        )
    }

    #[inline]
    pub fn new_athrow(generator: Value, exception: Value, exception_type_id: u16) -> Self {
        Self::new(
            AsyncGeneratorOperationKind::AThrow,
            generator,
            Value::none(),
            exception,
            exception_type_id,
        )
    }

    #[inline]
    pub fn new_aclose(generator: Value, exception: Value, exception_type_id: u16) -> Self {
        Self::new(
            AsyncGeneratorOperationKind::AClose,
            generator,
            Value::none(),
            exception,
            exception_type_id,
        )
    }

    #[inline]
    fn new(
        kind: AsyncGeneratorOperationKind,
        generator: Value,
        send_value: Value,
        exception: Value,
        exception_type_id: u16,
    ) -> Self {
        ensure_async_generator_operation_gc_dispatch_registered();
        Self {
            header: ObjectHeader::new(kind.type_id()),
            generator,
            send_value,
            exception,
            exception_type_id,
            kind,
            state: AsyncGeneratorOperationState::Created,
            _pad: [0; 4],
        }
    }

    #[inline]
    pub fn from_value(value: Value) -> Option<&'static Self> {
        let ptr = value.as_object_ptr()?;
        if !is_async_generator_operation_type_id(object_type_id(ptr)) {
            return None;
        }
        Some(unsafe { &*(ptr as *const Self) })
    }

    #[inline]
    pub fn from_value_mut(value: Value) -> Option<&'static mut Self> {
        let ptr = value.as_object_ptr()?;
        if !is_async_generator_operation_type_id(object_type_id(ptr)) {
            return None;
        }
        Some(unsafe { &mut *(ptr as *mut Self) })
    }

    #[inline(always)]
    pub fn generator(&self) -> Value {
        self.generator
    }

    #[inline(always)]
    pub fn send_value(&self) -> Value {
        self.send_value
    }

    #[inline(always)]
    pub fn exception(&self) -> Value {
        self.exception
    }

    #[inline(always)]
    pub fn exception_type_id(&self) -> u16 {
        self.exception_type_id
    }

    #[inline(always)]
    pub fn kind(&self) -> AsyncGeneratorOperationKind {
        self.kind
    }

    #[inline(always)]
    pub fn is_closed(&self) -> bool {
        self.state == AsyncGeneratorOperationState::Closed
    }

    #[inline]
    pub fn try_start(&mut self) -> Result<(), AsyncGeneratorOperationStartError> {
        match self.state {
            AsyncGeneratorOperationState::Created => {
                self.state = AsyncGeneratorOperationState::Running;
                Ok(())
            }
            AsyncGeneratorOperationState::Running => {
                Err(AsyncGeneratorOperationStartError::AlreadyRunning)
            }
            AsyncGeneratorOperationState::Closed => Err(AsyncGeneratorOperationStartError::Closed),
        }
    }

    #[inline(always)]
    pub fn finish(&mut self) {
        self.state = AsyncGeneratorOperationState::Closed;
    }
}

/// Start-state validation error for async-generator helper operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncGeneratorOperationStartError {
    AlreadyRunning,
    Closed,
}

#[inline(always)]
pub fn is_async_generator_operation_type_id(type_id: TypeId) -> bool {
    matches!(
        type_id,
        TypeId::ASYNC_GENERATOR_ASEND | TypeId::ASYNC_GENERATOR_ATHROW
    )
}

#[inline(always)]
fn object_type_id(ptr: *const ()) -> TypeId {
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
}

fn ensure_async_generator_operation_gc_dispatch_registered() {
    ASYNC_GENERATOR_OPERATION_GC_DISPATCH_ONCE.call_once(|| {
        let entry = DispatchEntry {
            trace: trace_async_generator_operation,
            size: size_async_generator_operation,
            finalize: finalize_async_generator_operation,
        };
        register_external_dispatch(TypeId::ASYNC_GENERATOR_ASEND, entry);
        register_external_dispatch(TypeId::ASYNC_GENERATOR_ATHROW, entry);
    });
}

unsafe fn trace_async_generator_operation(ptr: *const (), tracer: &mut dyn Tracer) {
    let object = unsafe { &*(ptr as *const AsyncGeneratorOperationObject) };
    object.trace(tracer);
}

unsafe fn size_async_generator_operation(_ptr: *const ()) -> usize {
    mem::size_of::<AsyncGeneratorOperationObject>()
}

unsafe fn finalize_async_generator_operation(ptr: *mut ()) {
    unsafe { std::ptr::drop_in_place(ptr as *mut AsyncGeneratorOperationObject) };
}

unsafe impl Send for AsyncGeneratorOperationObject {}

unsafe impl Trace for AsyncGeneratorOperationObject {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.generator.trace(tracer);
        self.send_value.trace(tracer);
        self.exception.trace(tracer);
    }
}
