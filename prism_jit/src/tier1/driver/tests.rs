use super::*;
use crate::tier1::codegen::TemplateInstruction;
use prism_code::{Instruction, Opcode, Register};
use prism_core::speculation::NoSpeculation;

fn make_code(instructions: Vec<Instruction>) -> CodeObject {
    let mut code = CodeObject::new("test", "test.py");
    code.instructions = instructions.into_boxed_slice();
    code.register_count = 16;
    code
}

#[test]
fn test_compiler_new() {
    let speculation = Arc::new(NoSpeculation);
    let compiler = Tier1Compiler::new(speculation);

    assert_eq!(compiler.functions_compiled(), 0);
    assert_eq!(compiler.total_code_bytes(), 0);
    assert_eq!(compiler.total_ic_sites(), 0);
}

#[test]
fn test_compiler_with_config() {
    let speculation = Arc::new(NoSpeculation);
    let config = CompilationConfig::fast();
    let compiler = Tier1Compiler::with_config(speculation, config);

    assert!(!compiler.config().lowering.enable_speculation);
}

#[test]
fn test_compilation_config_default() {
    let config = CompilationConfig::default();

    assert!(config.lowering.enable_ic);
    assert!(config.lowering.enable_speculation);
    assert_eq!(config.max_code_size, 64 * 1024);
}

#[test]
fn test_compilation_config_fast() {
    let config = CompilationConfig::fast();

    assert!(config.lowering.enable_ic);
    assert!(!config.lowering.enable_speculation);
    assert_eq!(config.max_code_size, 32 * 1024);
}

#[test]
fn test_compilation_config_quality() {
    let config = CompilationConfig::quality();

    assert!(config.lowering.enable_ic);
    assert!(config.collect_stats);
    assert_eq!(config.max_code_size, 128 * 1024);
}

#[test]
fn test_compilation_config_without_ic() {
    let config = CompilationConfig::default().without_ic();

    assert!(!config.lowering.enable_ic);
}

#[test]
fn test_compilation_stats_default() {
    let stats = CompilationStats::default();

    assert_eq!(stats.lowering_ns, 0);
    assert_eq!(stats.codegen_ns, 0);
    assert_eq!(stats.ic_sites_allocated, 0);
}

#[test]
fn test_compilation_error_display() {
    let e = CompilationError::EmptyCode;
    assert_eq!(e.to_string(), "code object is empty");

    let e = CompilationError::CodeTooLarge { size: 100, max: 50 };
    assert_eq!(
        e.to_string(),
        "generated code (100 bytes) exceeds maximum (50 bytes)"
    );
}

#[test]
fn test_compiled_code_accessors() {
    // Create mock compiled function using the template compiler
    let compiler = TemplateCompiler::new_for_testing();
    let instructions = vec![TemplateInstruction::Nop { bc_offset: 0 }];
    let compiled_fn = compiler.compile(16, &instructions).expect("compile failed");

    let code = CompiledCode {
        compiled: compiled_fn,
        ic_manager: IcManager::new(ShapeVersion::new(1)),
        entry_offset: 0,
    };

    assert!(code.code_size() > 0);
    assert!(!code.entry_point().is_null());
    assert_eq!(code.deopt_info().len(), 0);
}

#[test]
fn test_compiler_refresh_shape_version() {
    let speculation = Arc::new(NoSpeculation);
    let mut compiler = Tier1Compiler::new(speculation);

    let v1 = compiler.shape_version;
    compiler.refresh_shape_version();
    // Version should be the same unless global was bumped
    assert_eq!(v1, compiler.shape_version);
}

// Note: Full compilation tests require TemplateCompiler::compile_with_deopt
// to be implemented. These tests verify the driver infrastructure.
