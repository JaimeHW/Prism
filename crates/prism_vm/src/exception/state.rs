//! Exception state machine for VM execution.
//!
//! This module provides the state machine that tracks exception handling
//! state during VM execution. It manages the transition between normal
//! execution, exception propagation, and handler execution.
//!
//! # State Machine
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                    Exception State Machine                            │
//! ├──────────────────────────────────────────────────────────────────────┤
//! │                                                                       │
//! │                     ┌─────────────┐                                   │
//! │                     │   Normal    │◀───────────────────────┐          │
//! │                     │ (executing) │                        │          │
//! │                     └──────┬──────┘                        │          │
//! │                            │                               │          │
//! │                     raise exception                   handled         │
//! │                            │                               │          │
//! │                            ▼                               │          │
//! │                     ┌─────────────┐                        │          │
//! │           ┌─────────│ Propagating │─────────┐              │          │
//! │           │         │ (unwinding) │         │              │          │
//! │           │         └──────┬──────┘         │              │          │
//! │           │                │                │              │          │
//! │      no handler     found handler      reraise             │          │
//! │           │                │                │              │          │
//! │           ▼                ▼                ▼              │          │
//! │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │          │
//! │    │ Unhandled   │  │  Handling   │  │  Finally    │       │          │
//! │    │ (propagate) │  │ (in handler)│  │ (cleanup)   │       │          │
//! │    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │          │
//! │           │                │                │              │          │
//! │       to caller     exception cleared   reraise/exit       │          │
//! │           │                │                │              │          │
//! │           ▼                └────────────────┴──────────────┘          │
//! │    ┌─────────────┐                                                    │
//! │    │   Exit      │                                                    │
//! │    └─────────────┘                                                    │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Transition | Cost |
//! |------------|------|
//! | Normal → Propagating | 3-5 cycles (state write) |
//! | Propagating → Handling | 5-10 cycles (handler lookup) |
//! | Handling → Normal | 2-3 cycles (state clear) |

use std::fmt;

// ============================================================================
// Exception State
// ============================================================================

/// The current exception handling state.
///
/// # Memory Layout (1 byte)
///
/// Uses a compact u8 representation for cache efficiency.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ExceptionState {
    /// Normal execution - no exception active.
    #[default]
    Normal = 0,

    /// Exception raised, propagating up the stack.
    Propagating = 1,

    /// Inside an exception handler (except block).
    Handling = 2,

    /// Inside a finally block.
    Finally = 3,

    /// Exception is unhandled, propagating to caller.
    Unhandled = 4,
}

impl ExceptionState {
    /// Creates from a raw u8 value.
    #[inline]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Normal),
            1 => Some(Self::Propagating),
            2 => Some(Self::Handling),
            3 => Some(Self::Finally),
            4 => Some(Self::Unhandled),
            _ => None,
        }
    }

    /// Returns the raw u8 value.
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns true if in normal execution.
    #[inline]
    pub const fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }

    /// Returns true if an exception is active.
    #[inline]
    pub const fn has_exception(self) -> bool {
        !matches!(self, Self::Normal)
    }

    /// Returns true if currently propagating an exception.
    #[inline]
    pub const fn is_propagating(self) -> bool {
        matches!(self, Self::Propagating)
    }

    /// Returns true if inside a handler.
    #[inline]
    pub const fn is_in_handler(self) -> bool {
        matches!(self, Self::Handling | Self::Finally)
    }

    /// Returns true if the exception is unhandled.
    #[inline]
    pub const fn is_unhandled(self) -> bool {
        matches!(self, Self::Unhandled)
    }

    /// Returns a human-readable name.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Propagating => "Propagating",
            Self::Handling => "Handling",
            Self::Finally => "Finally",
            Self::Unhandled => "Unhandled",
        }
    }
}

impl fmt::Debug for ExceptionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExceptionState::{}", self.name())
    }
}

impl fmt::Display for ExceptionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// Exception Context
// ============================================================================

/// Context for the current exception being handled.
///
/// # Memory Layout (16 bytes)
///
/// ```text
/// ┌────────────────────────────────────────────────────────────────────┐
/// │ state (1b) │ depth (1b) │ flags (1b) │ pad (1b) │ handler_pc (4b) │
/// ├────────────────────────────────────────────────────────────────────┤
/// │                       resume_pc (4b)                               │
/// ├────────────────────────────────────────────────────────────────────┤
/// │                       frame_id (4b)                                │
/// └────────────────────────────────────────────────────────────────────┘
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ExceptionContext {
    /// Current state.
    state: ExceptionState,

    /// Nesting depth of exception handling.
    depth: u8,

    /// Context flags.
    flags: ExceptionContextFlags,

    /// Padding for alignment.
    _pad: u8,

    /// PC of the current handler (or 0 if none).
    handler_pc: u32,

    /// PC to resume after handler completes.
    resume_pc: u32,

    /// Frame that owns the current exception.
    frame_id: u32,
}

/// Flags for exception context.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct ExceptionContextFlags(u8);

impl ExceptionContextFlags {
    /// No flags set.
    pub const EMPTY: Self = Self(0);

    /// Exception was explicitly raised (not from runtime).
    pub const EXPLICIT_RAISE: u8 = 1 << 0;

    /// Exception should be reraised after finally.
    pub const RERAISE_AFTER_FINALLY: u8 = 1 << 1;

    /// Exception has been handled (cleared).
    pub const HANDLED: u8 = 1 << 2;

    /// Handler has run (for finally blocks).
    pub const HANDLER_RAN: u8 = 1 << 3;

    /// Creates flags from raw value.
    #[inline]
    pub const fn from_raw(value: u8) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[inline]
    pub const fn as_raw(self) -> u8 {
        self.0
    }

    /// Returns true if explicitly raised.
    #[inline]
    pub const fn is_explicit_raise(self) -> bool {
        self.0 & Self::EXPLICIT_RAISE != 0
    }

    /// Returns true if should reraise after finally.
    #[inline]
    pub const fn should_reraise(self) -> bool {
        self.0 & Self::RERAISE_AFTER_FINALLY != 0
    }

    /// Returns true if exception has been handled.
    #[inline]
    pub const fn is_handled(self) -> bool {
        self.0 & Self::HANDLED != 0
    }

    /// Returns true if handler has run.
    #[inline]
    pub const fn handler_ran(self) -> bool {
        self.0 & Self::HANDLER_RAN != 0
    }

    /// Sets a flag.
    #[inline]
    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }

    /// Clears a flag.
    #[inline]
    pub fn clear(&mut self, flag: u8) {
        self.0 &= !flag;
    }
}

impl fmt::Debug for ExceptionContextFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut flags = Vec::new();
        if self.is_explicit_raise() {
            flags.push("EXPLICIT");
        }
        if self.should_reraise() {
            flags.push("RERAISE");
        }
        if self.is_handled() {
            flags.push("HANDLED");
        }
        if self.handler_ran() {
            flags.push("RAN");
        }

        if flags.is_empty() {
            write!(f, "Flags()")
        } else {
            write!(f, "Flags({})", flags.join("|"))
        }
    }
}

impl ExceptionContext {
    /// Creates a new context in normal state.
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: ExceptionState::Normal,
            depth: 0,
            flags: ExceptionContextFlags::EMPTY,
            _pad: 0,
            handler_pc: 0,
            resume_pc: 0,
            frame_id: 0,
        }
    }

    /// Returns the current state.
    #[inline]
    pub const fn state(&self) -> ExceptionState {
        self.state
    }

    /// Returns the nesting depth.
    #[inline]
    pub const fn depth(&self) -> u8 {
        self.depth
    }

    /// Returns the handler PC.
    #[inline]
    pub const fn handler_pc(&self) -> u32 {
        self.handler_pc
    }

    /// Returns the resume PC.
    #[inline]
    pub const fn resume_pc(&self) -> u32 {
        self.resume_pc
    }

    /// Returns the frame ID.
    #[inline]
    pub const fn frame_id(&self) -> u32 {
        self.frame_id
    }

    /// Returns the context flags.
    #[inline]
    pub const fn flags(&self) -> ExceptionContextFlags {
        self.flags
    }

    /// Returns true if in normal state.
    #[inline]
    pub const fn is_normal(&self) -> bool {
        self.state.is_normal()
    }

    /// Returns true if an exception is active.
    #[inline]
    pub const fn has_exception(&self) -> bool {
        self.state.has_exception()
    }

    /// Transitions to propagating state.
    #[inline]
    pub fn begin_propagation(&mut self, frame_id: u32) {
        self.state = ExceptionState::Propagating;
        self.frame_id = frame_id;
        self.depth = self.depth.saturating_add(1);
    }

    /// Transitions to handling state.
    #[inline]
    pub fn begin_handling(&mut self, handler_pc: u32, resume_pc: u32) {
        self.state = ExceptionState::Handling;
        self.handler_pc = handler_pc;
        self.resume_pc = resume_pc;
    }

    /// Transitions to finally state.
    #[inline]
    pub fn begin_finally(&mut self, handler_pc: u32, resume_pc: u32) {
        self.state = ExceptionState::Finally;
        self.handler_pc = handler_pc;
        self.resume_pc = resume_pc;
    }

    /// Marks exception as unhandled.
    #[inline]
    pub fn mark_unhandled(&mut self) {
        self.state = ExceptionState::Unhandled;
    }

    /// Clears the exception and returns to normal state.
    #[inline]
    pub fn clear(&mut self) {
        self.state = ExceptionState::Normal;
        self.flags.set(ExceptionContextFlags::HANDLED);
        self.handler_pc = 0;
        self.resume_pc = 0;
    }

    /// Resets completely to initial state.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Sets the explicit raise flag.
    #[inline]
    pub fn set_explicit_raise(&mut self) {
        self.flags.set(ExceptionContextFlags::EXPLICIT_RAISE);
    }

    /// Sets the reraise flag for finally blocks.
    #[inline]
    pub fn set_reraise(&mut self) {
        self.flags.set(ExceptionContextFlags::RERAISE_AFTER_FINALLY);
    }

    /// Clears the reraise flag.
    #[inline]
    pub fn clear_reraise(&mut self) {
        self.flags
            .clear(ExceptionContextFlags::RERAISE_AFTER_FINALLY);
    }

    /// Marks the handler as having run.
    #[inline]
    pub fn mark_handler_ran(&mut self) {
        self.flags.set(ExceptionContextFlags::HANDLER_RAN);
    }
}

impl Default for ExceptionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ExceptionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExceptionContext")
            .field("state", &self.state)
            .field("depth", &self.depth)
            .field("flags", &self.flags)
            .field("handler_pc", &self.handler_pc)
            .field("resume_pc", &self.resume_pc)
            .field("frame_id", &self.frame_id)
            .finish()
    }
}

// ============================================================================
// State Transition Result
// ============================================================================

/// Result of a state transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionResult {
    /// Transition completed successfully.
    Ok,

    /// Invalid transition from current state.
    InvalidTransition {
        from: ExceptionState,
        to: ExceptionState,
    },

    /// Maximum exception depth exceeded.
    DepthExceeded,
}

impl TransitionResult {
    /// Returns true if the transition was successful.
    #[inline]
    pub const fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Returns true if the transition failed.
    #[inline]
    pub const fn is_err(&self) -> bool {
        !self.is_ok()
    }
}

// ============================================================================
// Exception State Machine
// ============================================================================

/// The complete exception state machine.
///
/// This struct manages all state related to exception handling in a VM.
pub struct ExceptionStateMachine {
    /// Current exception context.
    context: ExceptionContext,

    /// Statistics for exception handling.
    stats: ExceptionStats,
}

/// Statistics for exception handling.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExceptionStats {
    /// Total exceptions raised.
    pub raised: u64,

    /// Exceptions caught by handlers.
    pub caught: u64,

    /// Exceptions propagated to caller.
    pub propagated: u64,

    /// Finally blocks executed.
    pub finally_ran: u64,

    /// Exceptions reraised.
    pub reraised: u64,

    /// Maximum exception depth.
    pub max_depth: u8,
}

impl ExceptionStats {
    /// Creates new empty stats.
    #[inline]
    pub const fn new() -> Self {
        Self {
            raised: 0,
            caught: 0,
            propagated: 0,
            finally_ran: 0,
            reraised: 0,
            max_depth: 0,
        }
    }

    /// Resets all statistics.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the catch rate as a percentage.
    #[inline]
    pub fn catch_rate(&self) -> f64 {
        if self.raised == 0 {
            0.0
        } else {
            (self.caught as f64 / self.raised as f64) * 100.0
        }
    }
}

impl ExceptionStateMachine {
    /// Creates a new state machine.
    #[inline]
    pub fn new() -> Self {
        Self {
            context: ExceptionContext::new(),
            stats: ExceptionStats::new(),
        }
    }

    /// Returns the current context.
    #[inline]
    pub fn context(&self) -> &ExceptionContext {
        &self.context
    }

    /// Returns a mutable reference to the context.
    #[inline]
    pub fn context_mut(&mut self) -> &mut ExceptionContext {
        &mut self.context
    }

    /// Returns the statistics.
    #[inline]
    pub fn stats(&self) -> &ExceptionStats {
        &self.stats
    }

    /// Returns the current state.
    #[inline]
    pub fn state(&self) -> ExceptionState {
        self.context.state()
    }

    /// Returns true if in normal state.
    #[inline]
    pub fn is_normal(&self) -> bool {
        self.context.is_normal()
    }

    /// Raises an exception, transitioning to propagating state.
    #[inline]
    pub fn raise(&mut self, frame_id: u32) -> TransitionResult {
        // Can raise from Normal, Handling, or Finally
        match self.context.state() {
            ExceptionState::Normal | ExceptionState::Handling | ExceptionState::Finally => {
                self.context.begin_propagation(frame_id);
                self.stats.raised += 1;
                if self.context.depth() > self.stats.max_depth {
                    self.stats.max_depth = self.context.depth();
                }
                TransitionResult::Ok
            }
            from => TransitionResult::InvalidTransition {
                from,
                to: ExceptionState::Propagating,
            },
        }
    }

    /// Enters an exception handler.
    #[inline]
    pub fn enter_handler(&mut self, handler_pc: u32, resume_pc: u32) -> TransitionResult {
        if self.context.state() != ExceptionState::Propagating {
            return TransitionResult::InvalidTransition {
                from: self.context.state(),
                to: ExceptionState::Handling,
            };
        }
        self.context.begin_handling(handler_pc, resume_pc);
        self.stats.caught += 1;
        TransitionResult::Ok
    }

    /// Enters a finally block.
    #[inline]
    pub fn enter_finally(&mut self, handler_pc: u32, resume_pc: u32) -> TransitionResult {
        // Can enter finally from Propagating or Normal (for exit)
        match self.context.state() {
            ExceptionState::Propagating | ExceptionState::Normal => {
                self.context.begin_finally(handler_pc, resume_pc);
                self.stats.finally_ran += 1;
                TransitionResult::Ok
            }
            from => TransitionResult::InvalidTransition {
                from,
                to: ExceptionState::Finally,
            },
        }
    }

    /// Exits the current handler, clearing the exception.
    #[inline]
    pub fn exit_handler(&mut self) -> TransitionResult {
        if !self.context.state().is_in_handler() {
            return TransitionResult::InvalidTransition {
                from: self.context.state(),
                to: ExceptionState::Normal,
            };
        }
        self.context.clear();
        TransitionResult::Ok
    }

    /// Exits a finally block, potentially reraising.
    #[inline]
    pub fn exit_finally(&mut self) -> TransitionResult {
        if self.context.state() != ExceptionState::Finally {
            return TransitionResult::InvalidTransition {
                from: self.context.state(),
                to: ExceptionState::Normal,
            };
        }

        self.context.mark_handler_ran();

        if self.context.flags().should_reraise() {
            self.context.state = ExceptionState::Propagating;
            self.context.clear_reraise();
            self.stats.reraised += 1;
            TransitionResult::Ok
        } else {
            self.context.clear();
            TransitionResult::Ok
        }
    }

    /// Marks exception as unhandled (propagating to caller).
    #[inline]
    pub fn mark_unhandled(&mut self) -> TransitionResult {
        if self.context.state() != ExceptionState::Propagating {
            return TransitionResult::InvalidTransition {
                from: self.context.state(),
                to: ExceptionState::Unhandled,
            };
        }
        self.context.mark_unhandled();
        self.stats.propagated += 1;
        TransitionResult::Ok
    }

    /// Resets the state machine completely.
    #[inline]
    pub fn reset(&mut self) {
        self.context.reset();
    }

    /// Resets statistics.
    #[inline]
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }
}

impl Default for ExceptionStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ExceptionStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExceptionStateMachine")
            .field("context", &self.context)
            .field("stats", &self.stats)
            .finish()
    }
}
