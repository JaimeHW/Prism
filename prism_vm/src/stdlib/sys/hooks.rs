//! System hooks for display, exception handling, and tracing.
//!
//! Provides customizable hooks that Python code can override
//! for output formatting and error handling.

use prism_core::Value;
use std::sync::Arc;

// =============================================================================
// Display Hook
// =============================================================================

/// Type alias for display hook functions.
/// Takes a value and returns the string to display (or None for default).
pub type DisplayHookFn = Arc<dyn Fn(Value) -> Option<String> + Send + Sync>;

/// Manages the display hook for interactive output.
#[derive(Clone)]
pub struct DisplayHook {
    /// Custom hook function, or None for default.
    custom: Option<DisplayHookFn>,
}

impl DisplayHook {
    /// Create with default display behavior.
    #[inline]
    pub fn new() -> Self {
        Self { custom: None }
    }

    /// Set a custom display hook.
    #[inline]
    pub fn set(&mut self, hook: DisplayHookFn) {
        self.custom = Some(hook);
    }

    /// Reset to default display behavior.
    #[inline]
    pub fn reset(&mut self) {
        self.custom = None;
    }

    /// Check if using custom hook.
    #[inline]
    pub fn is_custom(&self) -> bool {
        self.custom.is_some()
    }

    /// Display a value, returning the formatted string.
    pub fn display(&self, value: Value) -> String {
        if let Some(ref hook) = self.custom {
            if let Some(s) = hook(value) {
                return s;
            }
        }
        // Default display behavior
        default_display(value)
    }
}

impl Default for DisplayHook {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for DisplayHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DisplayHook")
            .field("custom", &self.custom.is_some())
            .finish()
    }
}

/// Default display implementation.
fn default_display(value: Value) -> String {
    // Don't display None
    if value.is_none() {
        return String::new();
    }

    // Use repr for other values
    if let Some(i) = value.as_int() {
        return i.to_string();
    }
    if let Some(f) = value.as_float() {
        return format!("{}", f);
    }
    if let Some(b) = value.as_bool() {
        return if b {
            "True".to_string()
        } else {
            "False".to_string()
        };
    }
    if value.is_string() {
        return "'<string>'".to_string(); // Placeholder
    }

    // Fallback
    "<object>".to_string()
}

// =============================================================================
// Exception Hook
// =============================================================================

/// Type alias for exception hook functions.
/// Takes exception type, value, and traceback, returns formatted string.
pub type ExceptHookFn = Arc<dyn Fn(&str, &str, &str) -> String + Send + Sync>;

/// Manages the exception hook for error display.
#[derive(Clone)]
pub struct ExceptHook {
    /// Custom hook function, or None for default.
    custom: Option<ExceptHookFn>,
}

impl ExceptHook {
    /// Create with default exception formatting.
    #[inline]
    pub fn new() -> Self {
        Self { custom: None }
    }

    /// Set a custom exception hook.
    #[inline]
    pub fn set(&mut self, hook: ExceptHookFn) {
        self.custom = Some(hook);
    }

    /// Reset to default behavior.
    #[inline]
    pub fn reset(&mut self) {
        self.custom = None;
    }

    /// Check if using custom hook.
    #[inline]
    pub fn is_custom(&self) -> bool {
        self.custom.is_some()
    }

    /// Format an exception.
    pub fn format(&self, exc_type: &str, exc_value: &str, traceback: &str) -> String {
        if let Some(ref hook) = self.custom {
            return hook(exc_type, exc_value, traceback);
        }
        default_except_format(exc_type, exc_value, traceback)
    }
}

impl Default for ExceptHook {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ExceptHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExceptHook")
            .field("custom", &self.custom.is_some())
            .finish()
    }
}

/// Default exception format.
fn default_except_format(exc_type: &str, exc_value: &str, traceback: &str) -> String {
    let mut result = String::new();
    if !traceback.is_empty() {
        result.push_str("Traceback (most recent call last):\n");
        result.push_str(traceback);
    }
    result.push_str(exc_type);
    if !exc_value.is_empty() {
        result.push_str(": ");
        result.push_str(exc_value);
    }
    result.push('\n');
    result
}

// =============================================================================
// Trace Function
// =============================================================================

/// Trace event types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceEvent {
    /// Function call.
    Call,
    /// Line execution.
    Line,
    /// Function return.
    Return,
    /// Exception raised.
    Exception,
    /// C function call.
    CCall,
    /// C function return.
    CReturn,
    /// C function exception.
    CException,
    /// Opcode execution (for settrace).
    Opcode,
}

impl TraceEvent {
    /// Get the Python event name.
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            TraceEvent::Call => "call",
            TraceEvent::Line => "line",
            TraceEvent::Return => "return",
            TraceEvent::Exception => "exception",
            TraceEvent::CCall => "c_call",
            TraceEvent::CReturn => "c_return",
            TraceEvent::CException => "c_exception",
            TraceEvent::Opcode => "opcode",
        }
    }
}

/// Type alias for trace functions.
pub type TraceFn = Arc<dyn Fn(TraceEvent, &str, u32) + Send + Sync>;

/// Manages the trace function for debugging.
#[derive(Clone)]
pub struct TraceHook {
    /// Trace function, or None if not tracing.
    trace_fn: Option<TraceFn>,
    /// Whether to trace opcodes.
    trace_opcodes: bool,
}

impl TraceHook {
    /// Create with no tracing.
    #[inline]
    pub fn new() -> Self {
        Self {
            trace_fn: None,
            trace_opcodes: false,
        }
    }

    /// Set trace function.
    #[inline]
    pub fn set(&mut self, func: TraceFn) {
        self.trace_fn = Some(func);
    }

    /// Clear trace function.
    #[inline]
    pub fn clear(&mut self) {
        self.trace_fn = None;
    }

    /// Check if tracing is enabled.
    #[inline]
    pub fn is_tracing(&self) -> bool {
        self.trace_fn.is_some()
    }

    /// Enable opcode tracing.
    #[inline]
    pub fn enable_opcodes(&mut self) {
        self.trace_opcodes = true;
    }

    /// Disable opcode tracing.
    #[inline]
    pub fn disable_opcodes(&mut self) {
        self.trace_opcodes = false;
    }

    /// Check if opcode tracing is enabled.
    #[inline]
    pub fn traces_opcodes(&self) -> bool {
        self.trace_opcodes && self.trace_fn.is_some()
    }

    /// Call the trace function.
    #[inline]
    pub fn trace(&self, event: TraceEvent, filename: &str, lineno: u32) {
        if let Some(ref func) = self.trace_fn {
            func(event, filename, lineno);
        }
    }
}

impl Default for TraceHook {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TraceHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraceHook")
            .field("enabled", &self.is_tracing())
            .field("opcodes", &self.trace_opcodes)
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
