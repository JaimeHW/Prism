//! Bytecode interpreter (Tier 0).
/// Bytecode interpreter with profiling.
pub struct Interpreter;
impl Interpreter {
    /// Execute bytecode.
    pub fn run(&mut self) -> prism_core::PrismResult<prism_core::Value> {
        Ok(prism_core::Value::none())
    }
}
