//! Inlining Transformation
//!
//! This module implements the actual inlining transformation that integrates
//! a callee graph into a caller graph at a specific call site.
//!
//! # Transformation Steps
//!
//! 1. **Clone callee graph** into caller with ID remapping
//! 2. **Substitute parameters** with actual arguments
//! 3. **Connect control flow**: caller control → callee body → caller continuation
//! 4. **Handle returns**: merge multiple returns, extract return value
//! 5. **Replace call node**: substitute with inlined result
//!
//! # Control Flow Integration
//!
//! ```text
//! Before:                          After:
//!                                  
//! ┌─────────┐                      ┌─────────┐
//! │  ...    │                      │  ...    │
//! └────┬────┘                      └────┬────┘
//!      │                                │
//! ┌────▼────┐                      ┌────▼────┐
//! │  Call   │         →           │ Inlined │
//! │  f(x)   │                      │  Body   │
//! └────┬────┘                      └────┬────┘
//!      │                                │
//! ┌────▼────┐                      ┌────▼────┐
//! │  ...    │                      │  ...    │
//! └─────────┘                      └─────────┘
//! ```

use super::clone::GraphCloner;
use super::CallSite;
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{ControlOp, Operator};
use crate::ir::types::ValueType;

// =============================================================================
// Inline Transform Error
// =============================================================================

/// Errors that can occur during inlining transformation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineError {
    /// The call site node is invalid or not a call.
    InvalidCallSite,
    /// The callee graph is malformed.
    MalformedCallee,
    /// The callee has no return nodes.
    NoReturns,
    /// Control flow connection failed.
    ControlFlowError,
    /// The transformation would create invalid IR.
    InvalidTransformation(String),
}

impl std::fmt::Display for InlineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InlineError::InvalidCallSite => write!(f, "Invalid call site"),
            InlineError::MalformedCallee => write!(f, "Malformed callee graph"),
            InlineError::NoReturns => write!(f, "Callee has no return nodes"),
            InlineError::ControlFlowError => write!(f, "Control flow connection failed"),
            InlineError::InvalidTransformation(msg) => write!(f, "Invalid transformation: {}", msg),
        }
    }
}

impl std::error::Error for InlineError {}

/// Result type for inline operations.
pub type InlineResult<T> = Result<T, InlineError>;

// =============================================================================
// Inline Transform
// =============================================================================

/// Performs the inlining transformation.
pub struct InlineTransform;

impl InlineTransform {
    /// Inline a callee at the given call site.
    ///
    /// This is the main entry point for inlining. It:
    /// 1. Validates the call site
    /// 2. Clones the callee graph
    /// 3. Connects control flow
    /// 4. Replaces the call with the inlined result
    pub fn inline(caller: &mut Graph, site: &CallSite, callee: &Graph) -> InlineResult<InlineInfo> {
        // Validate call site
        let call_node = caller
            .get(site.call_node)
            .ok_or(InlineError::InvalidCallSite)?;

        if !matches!(call_node.op, Operator::Call(_)) {
            return Err(InlineError::InvalidCallSite);
        }

        // Get control input to the call
        let control_input = site
            .control_input
            .or_else(|| call_node.inputs.get(0))
            .ok_or(InlineError::ControlFlowError)?;

        // Clone the callee graph into the caller
        let clone_result = GraphCloner::new(callee)
            .with_arguments(&site.arguments)
            .with_control_input(control_input)
            .clone_into(caller);

        // Handle returns - create merge point if multiple returns
        let (result_node, exit_control) =
            Self::handle_returns(caller, &clone_result.cloned_returns)?;

        // Replace uses of the call node with the inlined result
        if let Some(result) = result_node {
            caller.replace_all_uses(site.call_node, result);
        }

        // Update control flow: nodes that used call's control output
        // should now use the exit control from inlined code
        Self::reconnect_control(caller, site.call_node, exit_control)?;

        // Mark the original call as dead
        caller.kill(site.call_node);

        Ok(InlineInfo {
            nodes_added: clone_result.nodes_cloned,
            result_node,
            exit_control,
        })
    }

    /// Handle multiple return nodes by creating a merge point.
    fn handle_returns(
        caller: &mut Graph,
        returns: &[NodeId],
    ) -> InlineResult<(Option<NodeId>, NodeId)> {
        match returns.len() {
            0 => {
                // No returns - unusual but valid for functions that always throw
                // Return None for result, use start as control (will be fixed up)
                Err(InlineError::NoReturns)
            }
            1 => {
                // Single return - extract value and control directly
                let ret_node = caller.get(returns[0]).ok_or(InlineError::MalformedCallee)?;

                // Return node has (control, value) inputs
                let return_value = ret_node.inputs.get(1);

                // Create a region to serve as the exit point
                // (The return node's control input is the control from the callee body)
                let callee_control = ret_node
                    .inputs
                    .get(0)
                    .ok_or(InlineError::ControlFlowError)?;
                let exit_region = caller.region(&[callee_control]);

                Ok((return_value, exit_region))
            }
            _ => {
                // Multiple returns - need to merge control and values
                Self::merge_returns(caller, returns)
            }
        }
    }

    /// Merge multiple return paths into a single control/value.
    fn merge_returns(
        caller: &mut Graph,
        returns: &[NodeId],
    ) -> InlineResult<(Option<NodeId>, NodeId)> {
        // Collect control inputs from all returns
        let mut control_inputs: Vec<NodeId> = Vec::with_capacity(returns.len());
        let mut value_inputs: Vec<NodeId> = Vec::with_capacity(returns.len());
        let mut return_type = ValueType::None;

        for &ret_id in returns {
            let ret_node = caller.get(ret_id).ok_or(InlineError::MalformedCallee)?;

            // Return node: (control, value)
            if let Some(ctrl) = ret_node.inputs.get(0) {
                control_inputs.push(ctrl);
            }
            if let Some(val) = ret_node.inputs.get(1) {
                value_inputs.push(val);
                // Track return type (use first non-None type)
                let val_type = caller.get(val).map(|n| n.ty).unwrap_or(ValueType::None);
                if return_type == ValueType::None && val_type != ValueType::None {
                    return_type = val_type;
                }
            }
        }

        if control_inputs.is_empty() {
            return Err(InlineError::ControlFlowError);
        }

        // Create merge region for control
        let merge_region = caller.region(&control_inputs);

        // Create phi for values (if we have values)
        let result_node = if !value_inputs.is_empty() && value_inputs.len() == control_inputs.len()
        {
            Some(caller.phi(merge_region, &value_inputs, return_type))
        } else {
            None
        };

        Ok((result_node, merge_region))
    }

    /// Reconnect control flow from the call node to the exit control.
    fn reconnect_control(
        caller: &mut Graph,
        call_node: NodeId,
        exit_control: NodeId,
    ) -> InlineResult<()> {
        // Find all nodes that use the call node as a control input
        let users: Vec<NodeId> = caller.uses(call_node).to_vec();

        for user_id in users {
            let user = caller.get(user_id).ok_or(InlineError::ControlFlowError)?;

            // Check if this user takes control input (first input for control nodes)
            if user.is_control() || matches!(user.op, Operator::Phi | Operator::LoopPhi) {
                // Find which input is the call node and replace with exit_control
                let inputs = user.inputs.to_vec();
                for (i, &input) in inputs.iter().enumerate() {
                    if input == call_node {
                        caller.replace_input(user_id, i, exit_control);
                    }
                }
            }
        }

        Ok(())
    }

    /// Perform partial inlining: inline only the "hot" path of a function.
    ///
    /// This creates a guard that checks a condition, and only inlines
    /// the code for the expected case, calling the original function
    /// for the slow path.
    pub fn partial_inline(
        caller: &mut Graph,
        site: &CallSite,
        callee: &Graph,
        hot_path_condition: NodeId,
    ) -> InlineResult<InlineInfo> {
        // Get control input
        let call_node = caller
            .get(site.call_node)
            .ok_or(InlineError::InvalidCallSite)?;
        let control_input = site
            .control_input
            .or_else(|| call_node.inputs.get(0))
            .ok_or(InlineError::ControlFlowError)?;

        // Create if-then-else structure
        // if (hot_path_condition) { inlined_code } else { original_call }

        let if_node = caller.add_node_with_type(
            Operator::Control(ControlOp::If),
            InputList::Pair(control_input, hot_path_condition),
            ValueType::Control,
        );

        // True projection (hot path - will be inlined)
        let true_proj = caller.add_node_with_type(
            Operator::Projection(0), // True branch
            InputList::Single(if_node),
            ValueType::Control,
        );

        // False projection (cold path - original call)
        let false_proj = caller.add_node_with_type(
            Operator::Projection(1), // False branch
            InputList::Single(if_node),
            ValueType::Control,
        );

        // Clone callee for the hot path
        let clone_result = GraphCloner::new(callee)
            .with_arguments(&site.arguments)
            .with_control_input(true_proj)
            .clone_into(caller);

        // Handle hot path returns
        let (hot_result, hot_exit) = Self::handle_returns(caller, &clone_result.cloned_returns)?;

        // For cold path, keep the original call but with new control
        let cold_call = Self::create_cold_path_call(caller, site, false_proj)?;

        // Merge the two paths
        let merge = caller.region(&[hot_exit, cold_call.control]);

        // Create phi for result values
        let final_result = match (hot_result, cold_call.value) {
            (Some(hot), Some(cold)) => {
                let hot_type = caller.get(hot).map(|n| n.ty).unwrap_or(ValueType::Top);
                Some(caller.phi(merge, &[hot, cold], hot_type))
            }
            (Some(result), None) | (None, Some(result)) => Some(result),
            (None, None) => None,
        };

        // Replace original call with merged result
        if let Some(result) = final_result {
            caller.replace_all_uses(site.call_node, result);
        }

        // Update control users
        Self::reconnect_control(caller, site.call_node, merge)?;

        // Kill original call
        caller.kill(site.call_node);

        Ok(InlineInfo {
            nodes_added: clone_result.nodes_cloned + 5, // Clone + if/projections/merge
            result_node: final_result,
            exit_control: merge,
        })
    }

    /// Create a call node for the cold path of partial inlining.
    fn create_cold_path_call(
        caller: &mut Graph,
        site: &CallSite,
        control: NodeId,
    ) -> InlineResult<ColdPathCall> {
        // Reconstruct the call with new control input
        let original = caller
            .get(site.call_node)
            .ok_or(InlineError::InvalidCallSite)?;

        // Build new inputs: control + callee + args
        let mut new_inputs = vec![control];
        for input in original.inputs.iter().skip(1) {
            // Skip original control
            new_inputs.push(input);
        }

        let new_call =
            caller.add_node_with_type(original.op, InputList::from_slice(&new_inputs), original.ty);

        // The call produces both a control and value output
        // For simplicity, assume call node itself is the value
        // A proper implementation would use projections
        Ok(ColdPathCall {
            call: new_call,
            control: new_call, // Simplified: call is also control
            value: Some(new_call),
        })
    }
}

/// Information about a cold path call created during partial inlining.
struct ColdPathCall {
    /// The call node.
    call: NodeId,
    /// Control output of the call.
    control: NodeId,
    /// Value output of the call (if any).
    value: Option<NodeId>,
}

// =============================================================================
// Inline Info
// =============================================================================

/// Information about a completed inline transformation.
#[derive(Debug, Clone)]
pub struct InlineInfo {
    /// Number of nodes added by inlining.
    pub nodes_added: usize,
    /// The node representing the inlined function's result (if any).
    pub result_node: Option<NodeId>,
    /// The control flow exit point of the inlined code.
    pub exit_control: NodeId,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};
    use crate::ir::operators::CallKind;

    fn make_simple_callee() -> Graph {
        // Create: fn(a, b) { return a + b }
        let mut builder = GraphBuilder::new(8, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        builder.return_value(sum);
        builder.finish()
    }

    fn make_identity_callee() -> Graph {
        // Create: fn(x) { return x }
        let mut builder = GraphBuilder::new(4, 1);
        let p0 = builder.parameter(0).unwrap();
        builder.return_value(p0);
        builder.finish()
    }

    #[test]
    fn test_inline_error_display() {
        assert_eq!(
            InlineError::InvalidCallSite.to_string(),
            "Invalid call site"
        );
        assert_eq!(
            InlineError::NoReturns.to_string(),
            "Callee has no return nodes"
        );
    }

    #[test]
    fn test_handle_single_return() {
        let mut caller = Graph::new();
        let val = caller.const_int(42);
        let control = caller.start;

        // Create a return node manually
        let ret = caller.add_node_with_type(
            Operator::Control(ControlOp::Return),
            InputList::Pair(control, val),
            ValueType::Control,
        );

        let result = InlineTransform::handle_returns(&mut caller, &[ret]);
        assert!(result.is_ok());

        let (result_node, exit_control) = result.unwrap();
        assert!(result_node.is_some());
        assert_ne!(exit_control, caller.start);
    }

    #[test]
    fn test_handle_multiple_returns() {
        let mut caller = Graph::new();
        let val1 = caller.const_int(1);
        let val2 = caller.const_int(2);

        // Create a branch with two returns
        let if_cond = caller.const_bool(true);
        let if_node = caller.add_node_with_type(
            Operator::Control(ControlOp::If),
            InputList::Pair(caller.start, if_cond),
            ValueType::Control,
        );

        let true_proj = caller.add_node_with_type(
            Operator::Projection(0),
            InputList::Single(if_node),
            ValueType::Control,
        );
        let false_proj = caller.add_node_with_type(
            Operator::Projection(1),
            InputList::Single(if_node),
            ValueType::Control,
        );

        let ret1 = caller.add_node_with_type(
            Operator::Control(ControlOp::Return),
            InputList::Pair(true_proj, val1),
            ValueType::Control,
        );
        let ret2 = caller.add_node_with_type(
            Operator::Control(ControlOp::Return),
            InputList::Pair(false_proj, val2),
            ValueType::Control,
        );

        let result = InlineTransform::handle_returns(&mut caller, &[ret1, ret2]);
        assert!(result.is_ok());

        let (result_node, exit_control) = result.unwrap();
        // Should have created a phi for the merged values
        assert!(result_node.is_some());
        // The exit control should be a region
        let exit_node = caller.get(exit_control).unwrap();
        assert!(matches!(exit_node.op, Operator::Control(ControlOp::Region)));
    }

    #[test]
    fn test_handle_no_returns() {
        let caller = &mut Graph::new();
        let result = InlineTransform::handle_returns(caller, &[]);
        assert!(matches!(result, Err(InlineError::NoReturns)));
    }

    #[test]
    fn test_inline_info() {
        let info = InlineInfo {
            nodes_added: 10,
            result_node: Some(NodeId::new(5)),
            exit_control: NodeId::new(6),
        };

        assert_eq!(info.nodes_added, 10);
        assert!(info.result_node.is_some());
    }

    // Integration test for full inlining
    #[test]
    fn test_basic_inline_integration() {
        // This tests the overall structure without actually performing
        // the full inline (which requires proper callee provider setup)
        let callee = make_simple_callee();
        let mut caller = Graph::new();

        // Create arguments in caller
        let arg0 = caller.const_int(10);
        let arg1 = caller.const_int(20);

        // The actual inline would be done via Inline::run with proper setup
        // Here we just verify the callee is valid
        assert!(callee.len() > 2); // More than just start/end
    }
}
