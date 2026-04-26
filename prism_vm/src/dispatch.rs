//! Dispatch table and main execution loop.
//!
//! Uses a static function pointer table for O(1) opcode dispatch.
//! Each opcode maps to a handler function that returns control flow.

use crate::VirtualMachine;
use crate::error::RuntimeError;
use prism_code::{CodeObject, Instruction, Opcode};
use prism_core::Value;
use std::sync::Arc;

/// Control flow result from opcode execution.
///
/// This enum represents all possible control flow outcomes from executing
/// a bytecode instruction. The VM dispatch loop uses this to determine
/// what action to take next.
#[derive(Debug, Clone)]
pub enum ControlFlow {
    // =========================================================================
    // Normal Execution
    // =========================================================================
    /// Continue to next instruction.
    Continue,

    /// Relative jump by signed offset.
    Jump(i16),

    /// Push new frame and call function.
    Call {
        code: Arc<CodeObject>,
        return_reg: u8,
    },

    /// Return value and pop frame.
    Return(Value),

    // =========================================================================
    // Exception Handling
    // =========================================================================
    /// Raise an exception.
    ///
    /// Triggers exception propagation: the VM will search for a handler,
    /// unwind the stack as needed, and either jump to a handler or
    /// propagate to the caller.
    Exception {
        /// Exception type ID for fast matching.
        type_id: u16,
        /// Handler PC if already located (0 if unknown).
        handler_pc: u32,
    },

    /// Re-raise the current exception.
    ///
    /// Used in except blocks to propagate an exception after partial handling.
    Reraise,

    /// Jump to exception handler.
    ///
    /// Called after handler lookup succeeds. Restores stack depth and
    /// transfers control to the handler code.
    EnterHandler {
        /// Handler bytecode address.
        handler_pc: u32,
        /// Stack depth to restore.
        stack_depth: u16,
    },

    /// Enter a finally block.
    ///
    /// Finally blocks execute unconditionally and may reraise after completion.
    EnterFinally {
        /// Finally block bytecode address.
        finally_pc: u32,
        /// Stack depth to restore.
        stack_depth: u16,
        /// Whether to reraise after finally completes.
        reraise: bool,
    },

    /// Exit exception handler.
    ///
    /// Pops the handler from the stack and resumes normal execution.
    ExitHandler,

    // =========================================================================
    // Generator Protocol
    // =========================================================================
    /// Yield a value from a generator.
    ///
    /// Suspends execution and returns the yielded value to the caller.
    /// The resume point is stored for later continuation.
    Yield {
        /// The value being yielded.
        value: Value,
        /// Bytecode offset to resume at when send() is called.
        resume_point: u32,
    },

    /// Resume a suspended generator.
    ///
    /// Continues execution from the saved resume point with the sent value.
    Resume {
        /// Value sent into the generator (or None for __next__).
        send_value: Value,
    },

    // =========================================================================
    // Error Handling
    // =========================================================================
    /// Runtime error occurred (non-exception error).
    Error(RuntimeError),
}

/// Opcode handler function signature.
pub type OpHandler = fn(&mut VirtualMachine, Instruction) -> ControlFlow;

/// Invalid opcode handler.
#[inline(always)]
fn op_invalid(_vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    ControlFlow::Error(RuntimeError::invalid_opcode(inst.opcode()))
}

// Import all opcode handlers
use crate::ops::arithmetic;
use crate::ops::calls;
use crate::ops::class;
use crate::ops::comparison;
use crate::ops::containers;
use crate::ops::context;
use crate::ops::control;
use crate::ops::coroutine;
use crate::ops::exception;
use crate::ops::load_store;
use crate::ops::r#match;
use crate::ops::method_dispatch;
use crate::ops::objects;
use crate::ops::subscript;
use crate::ops::unpack;

/// Build the static dispatch table.
/// Returns array of 256 function pointers indexed by opcode.
const fn build_dispatch_table() -> [OpHandler; 256] {
    let mut table: [OpHandler; 256] = [op_invalid; 256];

    // Control Flow (0x00-0x0F)
    table[Opcode::Nop as usize] = control::nop;
    table[Opcode::Return as usize] = control::return_value;
    table[Opcode::ReturnNone as usize] = control::return_none;
    table[Opcode::Jump as usize] = control::jump;
    table[Opcode::JumpIfFalse as usize] = control::jump_if_false;
    table[Opcode::JumpIfTrue as usize] = control::jump_if_true;
    table[Opcode::JumpIfNone as usize] = control::jump_if_none;
    table[Opcode::JumpIfNotNone as usize] = control::jump_if_not_none;
    table[Opcode::PopExceptHandler as usize] = exception::pop_except_handler;
    table[Opcode::Raise as usize] = exception::raise;
    table[Opcode::Reraise as usize] = exception::reraise;
    table[Opcode::EndFinally as usize] = exception::end_finally;
    table[Opcode::ExceptionMatch as usize] = exception::exception_match;
    table[Opcode::LoadException as usize] = exception::load_exception;
    table[Opcode::Yield as usize] = control::yield_value;
    table[Opcode::YieldFrom as usize] = control::yield_from;

    // Load/Store (0x10-0x1F)
    table[Opcode::LoadConst as usize] = load_store::load_const;
    table[Opcode::LoadNone as usize] = load_store::load_none;
    table[Opcode::LoadTrue as usize] = load_store::load_true;
    table[Opcode::LoadFalse as usize] = load_store::load_false;
    table[Opcode::LoadLocal as usize] = load_store::load_local;
    table[Opcode::StoreLocal as usize] = load_store::store_local;
    table[Opcode::LoadClosure as usize] = load_store::load_closure;
    table[Opcode::StoreClosure as usize] = load_store::store_closure;
    table[Opcode::LoadGlobal as usize] = load_store::load_global;
    table[Opcode::LoadBuiltin as usize] = load_store::load_builtin;
    table[Opcode::StoreGlobal as usize] = load_store::store_global;
    table[Opcode::DeleteLocal as usize] = load_store::delete_local;
    table[Opcode::DeleteGlobal as usize] = load_store::delete_global;
    table[Opcode::Move as usize] = load_store::move_reg;
    table[Opcode::DeleteClosure as usize] = load_store::delete_closure;
    table[Opcode::SetupAnnotations as usize] = load_store::setup_annotations;

    // Integer Arithmetic (0x20-0x2F)
    table[Opcode::AddInt as usize] = arithmetic::add_int;
    table[Opcode::SubInt as usize] = arithmetic::sub_int;
    table[Opcode::MulInt as usize] = arithmetic::mul_int;
    table[Opcode::FloorDivInt as usize] = arithmetic::floor_div_int;
    table[Opcode::ModInt as usize] = arithmetic::mod_int;
    table[Opcode::PowInt as usize] = arithmetic::pow_int;
    table[Opcode::NegInt as usize] = arithmetic::neg_int;
    table[Opcode::PosInt as usize] = arithmetic::pos_int;

    // Float Arithmetic (0x30-0x37)
    table[Opcode::AddFloat as usize] = arithmetic::add_float;
    table[Opcode::SubFloat as usize] = arithmetic::sub_float;
    table[Opcode::MulFloat as usize] = arithmetic::mul_float;
    table[Opcode::DivFloat as usize] = arithmetic::div_float;
    table[Opcode::FloorDivFloat as usize] = arithmetic::floor_div_float;
    table[Opcode::ModFloat as usize] = arithmetic::mod_float;
    table[Opcode::PowFloat as usize] = arithmetic::pow_float;
    table[Opcode::NegFloat as usize] = arithmetic::neg_float;

    // Generic Arithmetic (0x38-0x3F)
    table[Opcode::Add as usize] = arithmetic::add;
    table[Opcode::Sub as usize] = arithmetic::sub;
    table[Opcode::Mul as usize] = arithmetic::mul;
    table[Opcode::TrueDiv as usize] = arithmetic::true_div;
    table[Opcode::FloorDiv as usize] = arithmetic::floor_div;
    table[Opcode::Mod as usize] = arithmetic::modulo;
    table[Opcode::Pow as usize] = arithmetic::pow;
    table[Opcode::Neg as usize] = arithmetic::neg;
    table[Opcode::Pos as usize] = arithmetic::pos;

    // Comparison (0x40-0x4F)
    table[Opcode::Lt as usize] = comparison::lt;
    table[Opcode::Le as usize] = comparison::le;
    table[Opcode::Eq as usize] = comparison::eq;
    table[Opcode::Ne as usize] = comparison::ne;
    table[Opcode::Gt as usize] = comparison::gt;
    table[Opcode::Ge as usize] = comparison::ge;
    table[Opcode::Is as usize] = comparison::is;
    table[Opcode::IsNot as usize] = comparison::is_not;
    table[Opcode::In as usize] = comparison::in_op;
    table[Opcode::NotIn as usize] = comparison::not_in;

    // Logical/Bitwise (0x50-0x5F)
    table[Opcode::BitwiseAnd as usize] = comparison::bitwise_and;
    table[Opcode::BitwiseOr as usize] = comparison::bitwise_or;
    table[Opcode::BitwiseXor as usize] = comparison::bitwise_xor;
    table[Opcode::BitwiseNot as usize] = comparison::bitwise_not;
    table[Opcode::Shl as usize] = comparison::shl;
    table[Opcode::Shr as usize] = comparison::shr;
    table[Opcode::Not as usize] = comparison::not;

    // Object Operations (0x60-0x6F)
    table[Opcode::GetAttr as usize] = objects::get_attr;
    table[Opcode::SetAttr as usize] = objects::set_attr;
    table[Opcode::DelAttr as usize] = objects::del_attr;
    table[Opcode::GetItem as usize] = subscript::binary_subscr;
    table[Opcode::SetItem as usize] = subscript::store_subscr;
    table[Opcode::DelItem as usize] = subscript::delete_subscr;
    table[Opcode::GetIter as usize] = objects::get_iter;
    table[Opcode::ForIter as usize] = objects::for_iter;
    table[Opcode::Len as usize] = objects::len;
    table[Opcode::IsCallable as usize] = objects::is_callable;
    table[Opcode::LoadMethod as usize] = method_dispatch::load_method;
    table[Opcode::BuildClass as usize] = class::build_class;
    table[Opcode::BuildClassWithMeta as usize] = class::build_class_with_metaclass;
    table[Opcode::ClassMeta as usize] = class::class_meta;

    // Function Calls (0x70-0x7F)
    table[Opcode::Call as usize] = calls::call;
    table[Opcode::CallKw as usize] = calls::call_kw;
    table[Opcode::CallMethod as usize] = method_dispatch::call_method;
    table[Opcode::TailCall as usize] = calls::tail_call;
    table[Opcode::MakeFunction as usize] = calls::make_function;
    table[Opcode::MakeClosure as usize] = calls::make_closure;
    table[Opcode::CallKwEx as usize] = calls::call_kw_ex;
    table[Opcode::CallEx as usize] = unpack::call_ex;
    table[Opcode::BuildTupleUnpack as usize] = unpack::build_tuple_unpack;
    table[Opcode::BuildDictUnpack as usize] = unpack::build_dict_unpack;
    table[Opcode::SetFunctionDefaults as usize] = calls::set_function_defaults;

    // Container Operations (0x80-0x8F)
    table[Opcode::BuildList as usize] = containers::build_list;
    table[Opcode::BuildTuple as usize] = containers::build_tuple;
    table[Opcode::BuildSet as usize] = containers::build_set;
    table[Opcode::BuildDict as usize] = containers::build_dict;
    table[Opcode::BuildString as usize] = containers::build_string;
    table[Opcode::ListAppend as usize] = containers::list_append;
    table[Opcode::SetAdd as usize] = containers::set_add;
    table[Opcode::DictSet as usize] = containers::dict_set;
    table[Opcode::UnpackSequence as usize] = containers::unpack_sequence;
    table[Opcode::UnpackEx as usize] = containers::unpack_ex;
    table[Opcode::BuildSlice as usize] = containers::build_slice;
    table[Opcode::BuildListUnpack as usize] = unpack::build_list_unpack;
    table[Opcode::BuildSetUnpack as usize] = unpack::build_set_unpack;

    // Import (0x90-0x92)
    table[Opcode::ImportName as usize] = containers::import_name;
    table[Opcode::ImportFrom as usize] = containers::import_from;
    table[Opcode::ImportStar as usize] = containers::import_star;

    // Exception Handling (CPython 3.11+) (0x93-0x96, 0xAB)
    table[Opcode::Raise as usize] = exception::raise;
    table[Opcode::Reraise as usize] = exception::reraise;
    table[Opcode::RaiseFrom as usize] = exception::raise_from;
    table[Opcode::PushExcInfo as usize] = exception::push_exc_info;
    table[Opcode::PopExcInfo as usize] = exception::pop_exc_info;
    table[Opcode::HasExcInfo as usize] = exception::has_exc_info;
    table[Opcode::ClearException as usize] = exception::clear_exception;

    // Context Managers (0x97-0x99)
    table[Opcode::BeforeWith as usize] = context::before_with;
    table[Opcode::ExitWith as usize] = context::exit_with;
    table[Opcode::WithCleanup as usize] = context::with_cleanup;

    // Pattern Matching (0x9A-0x9F)
    table[Opcode::MatchClass as usize] = r#match::match_class;
    table[Opcode::MatchMapping as usize] = r#match::match_mapping;
    table[Opcode::MatchSequence as usize] = r#match::match_sequence;
    table[Opcode::MatchKeys as usize] = r#match::match_keys;
    table[Opcode::CopyDictWithoutKeys as usize] = r#match::copy_dict_without_keys;
    table[Opcode::GetMatchArgs as usize] = r#match::get_match_args;

    // Coroutine/Async Operations (0xA0-0xAF)
    table[Opcode::GetAwaitable as usize] = coroutine::get_awaitable;
    table[Opcode::GetAIter as usize] = coroutine::get_aiter;
    table[Opcode::GetANext as usize] = coroutine::get_anext;
    table[Opcode::EndAsyncFor as usize] = coroutine::end_async_for;
    table[Opcode::Send as usize] = coroutine::send;
    table[Opcode::EnterExcept as usize] = exception::enter_except;
    table[Opcode::ExitExcept as usize] = exception::exit_except;
    table[Opcode::AbortExcept as usize] = exception::abort_except;
    table[Opcode::AttrName as usize] = objects::attr_name;

    table
}

/// Static dispatch table - computed at compile time.
pub static DISPATCH_TABLE: [OpHandler; 256] = build_dispatch_table();

/// Get the handler for an opcode.
#[inline(always)]
pub fn get_handler(opcode: u8) -> OpHandler {
    // Safety: opcode is u8, so always in bounds for 256-element array
    unsafe { *DISPATCH_TABLE.get_unchecked(opcode as usize) }
}

#[cfg(test)]
mod tests;
