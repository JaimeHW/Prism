//! Public exception inspection helpers for runtime errors.

use crate::builtins::ExceptionValue;
use crate::error::{RuntimeError, RuntimeErrorKind};
use prism_core::Value;

pub use crate::stdlib::exceptions::ExceptionTypeId;

/// Python exception information preserved on a runtime error.
#[derive(Debug, Clone)]
pub struct RuntimeException {
    type_id: ExceptionTypeId,
    display_text: String,
    args: Option<Box<[Value]>>,
    preserved_value: bool,
}

impl RuntimeException {
    #[inline]
    pub fn type_id(&self) -> ExceptionTypeId {
        self.type_id
    }

    #[inline]
    pub fn type_name(&self) -> &'static str {
        self.type_id.name()
    }

    #[inline]
    pub fn display_text(&self) -> &str {
        &self.display_text
    }

    #[inline]
    pub fn args(&self) -> Option<&[Value]> {
        self.args.as_deref()
    }

    #[inline]
    pub fn has_preserved_value(&self) -> bool {
        self.preserved_value
    }
}

/// Return the Python exception carried by a runtime error, if it has one.
pub fn runtime_exception(error: &RuntimeError) -> Option<RuntimeException> {
    if let Some(value) = error.raised_value
        && let Some(exception) = unsafe { ExceptionValue::from_value(value) }
    {
        return Some(RuntimeException {
            type_id: exception.type_id(),
            display_text: exception.display_text(),
            args: exception
                .args()
                .map(|args| args.to_vec().into_boxed_slice()),
            preserved_value: true,
        });
    }

    match error.kind() {
        RuntimeErrorKind::Exception { type_id, message } => {
            ExceptionTypeId::from_u8(*type_id as u8).map(|type_id| RuntimeException {
                type_id,
                display_text: message.to_string(),
                args: None,
                preserved_value: false,
            })
        }
        _ => None,
    }
}

/// Return the Python exception type carried by a runtime error.
#[inline]
pub fn exception_type_id(error: &RuntimeError) -> Option<ExceptionTypeId> {
    runtime_exception(error).map(|exception| exception.type_id())
}
