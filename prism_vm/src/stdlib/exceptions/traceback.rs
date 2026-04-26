//! Traceback and frame information.
//!
//! This module provides lazy traceback construction for exceptions.
//! Line numbers are only computed when the traceback is accessed,
//! following CPython 3.11+ optimization patterns.
//!
//! # Performance Design
//!
//! - **Lazy line numbers**: Only computed when __traceback__ is accessed
//! - **SmallVec frames**: Stack-allocated for shallow traces (< 8 frames)
//! - **Deferred string formatting**: No allocations until str() called

use smallvec::SmallVec;
use std::fmt;
use std::sync::Arc;

// ============================================================================
// Frame Information
// ============================================================================

/// Information about a single stack frame in a traceback.
///
/// This is the raw frame data captured when an exception is raised.
/// Line numbers may be computed lazily from bytecode offsets.
#[derive(Clone)]
pub struct FrameInfo {
    /// Name of the function/method.
    pub func_name: Arc<str>,

    /// Filename (module path).
    pub filename: Arc<str>,

    /// Bytecode offset at time of exception.
    pub bytecode_offset: u32,

    /// Line number (0 = not yet computed, lazily resolved).
    pub line_number: u32,

    /// First line number of the function (for relative line computation).
    pub first_lineno: u32,
}

impl FrameInfo {
    /// Creates a new frame info with a known line number.
    #[inline]
    pub fn new(func_name: Arc<str>, filename: Arc<str>, line_number: u32) -> Self {
        Self {
            func_name,
            filename,
            bytecode_offset: 0,
            line_number,
            first_lineno: line_number,
        }
    }

    /// Creates a frame info with deferred line number resolution.
    #[inline]
    pub fn with_offset(
        func_name: Arc<str>,
        filename: Arc<str>,
        bytecode_offset: u32,
        first_lineno: u32,
    ) -> Self {
        Self {
            func_name,
            filename,
            bytecode_offset,
            line_number: 0, // Will be resolved lazily
            first_lineno,
        }
    }

    /// Returns true if the line number needs to be resolved.
    #[inline]
    pub const fn needs_resolution(&self) -> bool {
        self.line_number == 0
    }

    /// Sets the resolved line number.
    #[inline]
    pub fn set_line_number(&mut self, lineno: u32) {
        self.line_number = lineno;
    }

    /// Returns the line number, or 0 if not yet resolved.
    #[inline]
    pub const fn line_number(&self) -> u32 {
        self.line_number
    }
}

impl fmt::Debug for FrameInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameInfo")
            .field("func_name", &self.func_name)
            .field("filename", &self.filename)
            .field("line_number", &self.line_number)
            .finish()
    }
}

// ============================================================================
// Traceback Object
// ============================================================================

/// Number of frames stored inline (stack-allocated).
/// Most Python tracebacks are < 8 frames deep.
const INLINE_FRAMES: usize = 8;

/// Python traceback object.
///
/// Contains a list of stack frames from where an exception was raised.
/// Line numbers are resolved lazily for performance.
#[derive(Clone)]
pub struct TracebackObject {
    /// Stack frames, most recent last (matches Python's tb_next order).
    frames: SmallVec<[FrameInfo; INLINE_FRAMES]>,

    /// Cached formatted string (lazy).
    formatted_cache: Option<Arc<str>>,
}

impl TracebackObject {
    /// Creates an empty traceback.
    #[inline]
    pub fn empty() -> Self {
        Self {
            frames: SmallVec::new(),
            formatted_cache: None,
        }
    }

    /// Creates a traceback with a single frame.
    #[inline]
    pub fn single(frame: FrameInfo) -> Self {
        let mut frames = SmallVec::new();
        frames.push(frame);
        Self {
            frames,
            formatted_cache: None,
        }
    }

    /// Creates a traceback from an iterator of frames.
    pub fn from_frames(frames: impl IntoIterator<Item = FrameInfo>) -> Self {
        Self {
            frames: frames.into_iter().collect(),
            formatted_cache: None,
        }
    }

    /// Returns true if the traceback has no frames.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the number of frames.
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Appends a frame to the traceback.
    #[inline]
    pub fn push(&mut self, frame: FrameInfo) {
        self.frames.push(frame);
        self.formatted_cache = None; // Invalidate cache
    }

    /// Extends the traceback with frames from another traceback.
    pub fn extend(&mut self, other: &TracebackObject) {
        self.frames.extend(other.frames.iter().cloned());
        self.formatted_cache = None;
    }

    /// Returns an iterator over the frames.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &FrameInfo> {
        self.frames.iter()
    }

    /// Returns a mutable iterator over the frames.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut FrameInfo> {
        self.formatted_cache = None; // Invalidate on mutation
        self.frames.iter_mut()
    }

    /// Returns the most recent frame (where exception was raised).
    #[inline]
    pub fn innermost(&self) -> Option<&FrameInfo> {
        self.frames.last()
    }

    /// Returns the outermost frame (entry point).
    #[inline]
    pub fn outermost(&self) -> Option<&FrameInfo> {
        self.frames.first()
    }

    /// Formats the traceback as a string (cached).
    pub fn format(&mut self) -> Arc<str> {
        if let Some(cached) = &self.formatted_cache {
            return Arc::clone(cached);
        }

        let mut output = String::with_capacity(self.frames.len() * 80);
        output.push_str("Traceback (most recent call last):\n");

        for frame in &self.frames {
            output.push_str("  File \"");
            output.push_str(&frame.filename);
            output.push_str("\", line ");
            output.push_str(&frame.line_number.to_string());
            output.push_str(", in ");
            output.push_str(&frame.func_name);
            output.push('\n');
        }

        let formatted: Arc<str> = Arc::from(output);
        self.formatted_cache = Some(Arc::clone(&formatted));
        formatted
    }

    /// Clears the traceback.
    #[inline]
    pub fn clear(&mut self) {
        self.frames.clear();
        self.formatted_cache = None;
    }

    /// Returns true if all frames have resolved line numbers.
    pub fn is_resolved(&self) -> bool {
        self.frames.iter().all(|f| !f.needs_resolution())
    }
}

impl Default for TracebackObject {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Debug for TracebackObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracebackObject")
            .field("frames", &self.frames.len())
            .finish()
    }
}

impl fmt::Display for TracebackObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Traceback (most recent call last):\n")?;
        for frame in &self.frames {
            writeln!(
                f,
                "  File \"{}\", line {}, in {}",
                frame.filename, frame.line_number, frame.func_name
            )?;
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
