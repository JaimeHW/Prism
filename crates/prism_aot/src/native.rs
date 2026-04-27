use std::error::Error;
use std::fmt;

use prism_core::{AotImportBinding, Value};
use prism_parser::ast::{BinOp, Expr, ExprKind, Module, Stmt, StmtKind, UnaryOp};

/// One lowerable native module-init plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeModuleInitPlan {
    /// Canonical module name.
    pub module_name: String,
    /// Stable linker-visible symbol name for the module init stub.
    pub symbol_name: String,
    /// Ordered native operations executed for module initialization.
    pub operations: Vec<NativeInitOperation>,
}

/// One top-level native init operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeInitOperation {
    /// `import module`
    ImportModule {
        /// Target name bound in module scope.
        target: String,
        /// Raw module spec, including relative import dots.
        module_spec: String,
        /// Binding mode for dotted imports.
        binding: AotImportBinding,
    },
    /// `from module import name`
    ImportFrom {
        /// Target name bound in module scope.
        target: String,
        /// Raw module spec, including relative import dots.
        module_spec: String,
        /// Imported attribute or submodule name.
        attribute: String,
    },
    /// `target = expr`
    StoreExpr {
        /// Target name bound in module scope.
        target: String,
        /// Lowered expression.
        expr: NativeExpr,
    },
}

/// Lowerable expression subset for native module initialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeExpr {
    /// A single operand.
    Operand(NativeOperand),
    /// Binary addition with operand-only children.
    Add {
        /// Left operand.
        left: NativeOperand,
        /// Right operand.
        right: NativeOperand,
    },
}

/// One operand referenced by a native expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeOperand {
    /// Immediate literal value or string.
    Immediate(NativeImmediate),
    /// Name loaded from module scope.
    Name(String),
}

/// Immediate literal supported by native module initialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeImmediate {
    /// Fully encoded Prism value bits for scalar immediates.
    ValueBits(u64),
    /// UTF-8 string literal to intern at runtime.
    String(String),
}

/// Lowering failure for unsupported module-init syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLoweringError {
    message: String,
}

impl NativeLoweringError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for NativeLoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for NativeLoweringError {}

impl NativeModuleInitPlan {
    /// Lower a parsed module into the initial native module-init subset.
    pub fn lower(module_name: &str, module: &Module) -> Result<Self, NativeLoweringError> {
        let mut operations = Vec::new();

        for (index, stmt) in module.body.iter().enumerate() {
            lower_stmt(module_name, stmt, index, &mut operations)?;
        }

        Ok(Self {
            module_name: module_name.to_string(),
            symbol_name: native_init_symbol(module_name),
            operations,
        })
    }
}

/// Compute the stable native init symbol for a module.
pub fn native_init_symbol(module_name: &str) -> String {
    let mut encoded = String::with_capacity(module_name.len() * 2);
    for byte in module_name.as_bytes() {
        use fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    format!("prism_aot_init_{encoded}")
}

fn lower_stmt(
    module_name: &str,
    stmt: &Stmt,
    index: usize,
    operations: &mut Vec<NativeInitOperation>,
) -> Result<(), NativeLoweringError> {
    match &stmt.kind {
        StmtKind::Expr(expr) if index == 0 && matches!(expr.kind, ExprKind::String(_)) => {
            operations.push(NativeInitOperation::StoreExpr {
                target: "__doc__".to_string(),
                expr: NativeExpr::Operand(lower_operand(expr, module_name)?),
            });
            Ok(())
        }
        StmtKind::Pass | StmtKind::Global(_) => Ok(()),
        StmtKind::Import(aliases) => {
            for alias in aliases {
                let target = alias.asname.clone().unwrap_or_else(|| {
                    alias
                        .name
                        .split('.')
                        .next()
                        .unwrap_or(&alias.name)
                        .to_string()
                });
                let binding = if alias.asname.is_none() && alias.name.contains('.') {
                    AotImportBinding::TopLevel
                } else {
                    AotImportBinding::Exact
                };
                operations.push(NativeInitOperation::ImportModule {
                    target,
                    module_spec: alias.name.clone(),
                    binding,
                });
            }
            Ok(())
        }
        StmtKind::ImportFrom {
            module,
            names,
            level,
        } => {
            let module_spec = format!(
                "{}{}",
                ".".repeat(*level as usize),
                module.as_deref().unwrap_or("")
            );
            for alias in names {
                if alias.name == "*" {
                    return Err(unsupported(
                        module_name,
                        stmt,
                        "star imports are not yet supported by native module lowering",
                    ));
                }
                operations.push(NativeInitOperation::ImportFrom {
                    target: alias.asname.clone().unwrap_or_else(|| alias.name.clone()),
                    module_spec: module_spec.clone(),
                    attribute: alias.name.clone(),
                });
            }
            Ok(())
        }
        StmtKind::Assign { targets, value } => {
            let expr = lower_expr(value, module_name)?;
            for target in targets {
                let ExprKind::Name(name) = &target.kind else {
                    return Err(unsupported(
                        module_name,
                        stmt,
                        "only simple name assignments are supported by native module lowering",
                    ));
                };
                operations.push(NativeInitOperation::StoreExpr {
                    target: name.clone(),
                    expr: expr.clone(),
                });
            }
            Ok(())
        }
        _ => Err(unsupported(
            module_name,
            stmt,
            "statement is not yet supported by native module lowering",
        )),
    }
}

fn lower_expr(expr: &Expr, module_name: &str) -> Result<NativeExpr, NativeLoweringError> {
    match &expr.kind {
        ExprKind::BinOp { left, op, right } if *op == BinOp::Add => Ok(NativeExpr::Add {
            left: lower_operand(left, module_name)?,
            right: lower_operand(right, module_name)?,
        }),
        _ => Ok(NativeExpr::Operand(lower_operand(expr, module_name)?)),
    }
}

fn lower_operand(expr: &Expr, module_name: &str) -> Result<NativeOperand, NativeLoweringError> {
    match &expr.kind {
        ExprKind::Int(value) => Ok(NativeOperand::Immediate(NativeImmediate::ValueBits(
            small_int_bits(*value, module_name)?,
        ))),
        ExprKind::Float(value) => Ok(NativeOperand::Immediate(NativeImmediate::ValueBits(
            Value::float(*value).to_bits(),
        ))),
        ExprKind::Bool(value) => Ok(NativeOperand::Immediate(NativeImmediate::ValueBits(
            Value::bool(*value).to_bits(),
        ))),
        ExprKind::None => Ok(NativeOperand::Immediate(NativeImmediate::ValueBits(
            Value::none().to_bits(),
        ))),
        ExprKind::String(value) => Ok(NativeOperand::Immediate(NativeImmediate::String(
            value.value.clone(),
        ))),
        ExprKind::Name(name) => Ok(NativeOperand::Name(name.clone())),
        ExprKind::UnaryOp { op, operand } => lower_unary_operand(*op, operand, module_name),
        _ => Err(NativeLoweringError::new(format!(
            "module '{module_name}' uses an expression that is not yet supported by native module lowering"
        ))),
    }
}

fn lower_unary_operand(
    op: UnaryOp,
    operand: &Expr,
    module_name: &str,
) -> Result<NativeOperand, NativeLoweringError> {
    match (op, &operand.kind) {
        (UnaryOp::USub, ExprKind::Int(value)) => {
            Ok(NativeOperand::Immediate(NativeImmediate::ValueBits(small_int_bits(
                value.checked_neg().ok_or_else(|| {
                    NativeLoweringError::new(format!(
                        "module '{module_name}' uses an integer literal that overflows native lowering"
                    ))
                })?,
                module_name,
            )?)))
        }
        (UnaryOp::UAdd, ExprKind::Int(value)) => {
            Ok(NativeOperand::Immediate(NativeImmediate::ValueBits(small_int_bits(
                *value,
                module_name,
            )?)))
        }
        (UnaryOp::USub, ExprKind::Float(value)) => Ok(NativeOperand::Immediate(
            NativeImmediate::ValueBits(Value::float(-value).to_bits()),
        )),
        (UnaryOp::UAdd, ExprKind::Float(value)) => Ok(NativeOperand::Immediate(
            NativeImmediate::ValueBits(Value::float(*value).to_bits()),
        )),
        _ => Err(NativeLoweringError::new(format!(
            "module '{module_name}' uses a unary expression that is not yet supported by native module lowering"
        ))),
    }
}

fn small_int_bits(value: i64, module_name: &str) -> Result<u64, NativeLoweringError> {
    Value::int(value)
        .map(|value| value.to_bits())
        .ok_or_else(|| {
            NativeLoweringError::new(format!(
                "module '{module_name}' uses integer literal {value} that does not fit Prism's inline representation"
            ))
        })
}

fn unsupported(module_name: &str, stmt: &Stmt, message: &str) -> NativeLoweringError {
    NativeLoweringError::new(format!(
        "module '{module_name}' cannot lower statement at span {}..{}: {message}",
        stmt.span.start, stmt.span.end
    ))
}
