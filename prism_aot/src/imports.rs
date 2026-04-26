use std::collections::BTreeSet;

use prism_parser::ast::{Alias, MatchCase, Module, Stmt, StmtKind};

use crate::error::AotError;

/// Static imports discovered syntactically in a module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticImports {
    /// Modules that must exist for the import statement itself to resolve.
    pub required_modules: Vec<String>,
    /// Potential submodules referenced through `from ... import ...`.
    pub from_import_candidates: Vec<String>,
}

#[derive(Debug, Default)]
struct ImportCollector {
    required_modules: BTreeSet<String>,
    from_import_candidates: BTreeSet<String>,
}

impl ImportCollector {
    fn into_static_imports(self) -> StaticImports {
        StaticImports {
            required_modules: self.required_modules.into_iter().collect(),
            from_import_candidates: self.from_import_candidates.into_iter().collect(),
        }
    }
}

/// Collect all syntactic imports from a module, including nested scopes.
pub fn collect_static_imports(
    module: &Module,
    current_package: &str,
) -> Result<StaticImports, AotError> {
    let mut imports = ImportCollector::default();
    visit_statements(&module.body, current_package, &mut imports)?;
    Ok(imports.into_static_imports())
}

fn visit_statements(
    statements: &[Stmt],
    current_package: &str,
    imports: &mut ImportCollector,
) -> Result<(), AotError> {
    for statement in statements {
        visit_statement(statement, current_package, imports)?;
    }
    Ok(())
}

fn visit_statement(
    statement: &Stmt,
    current_package: &str,
    imports: &mut ImportCollector,
) -> Result<(), AotError> {
    match &statement.kind {
        StmtKind::Import(aliases) => {
            for alias in aliases {
                imports.required_modules.insert(alias.name.clone());
            }
        }
        StmtKind::ImportFrom {
            module,
            names,
            level,
        } => {
            let absolute_base =
                resolve_relative_import(module.as_deref().unwrap_or(""), *level, current_package)?;

            if !absolute_base.is_empty() {
                imports.required_modules.insert(absolute_base.clone());
            }

            if !is_star_import(names) && !absolute_base.is_empty() {
                for alias in names {
                    imports
                        .from_import_candidates
                        .insert(format!("{}.{}", absolute_base, alias.name));
                }
            }
        }
        StmtKind::If { body, orelse, .. }
        | StmtKind::For { body, orelse, .. }
        | StmtKind::AsyncFor { body, orelse, .. }
        | StmtKind::While { body, orelse, .. } => {
            visit_statements(body, current_package, imports)?;
            visit_statements(orelse, current_package, imports)?;
        }
        StmtKind::With { body, .. }
        | StmtKind::AsyncWith { body, .. }
        | StmtKind::FunctionDef { body, .. }
        | StmtKind::AsyncFunctionDef { body, .. }
        | StmtKind::ClassDef { body, .. } => {
            visit_statements(body, current_package, imports)?;
        }
        StmtKind::Try {
            body,
            handlers,
            orelse,
            finalbody,
        }
        | StmtKind::TryStar {
            body,
            handlers,
            orelse,
            finalbody,
        } => {
            visit_statements(body, current_package, imports)?;
            for handler in handlers {
                visit_statements(&handler.body, current_package, imports)?;
            }
            visit_statements(orelse, current_package, imports)?;
            visit_statements(finalbody, current_package, imports)?;
        }
        StmtKind::Match { cases, .. } => {
            for case in cases {
                visit_match_case(case, current_package, imports)?;
            }
        }
        StmtKind::Expr(_)
        | StmtKind::Assign { .. }
        | StmtKind::AugAssign { .. }
        | StmtKind::AnnAssign { .. }
        | StmtKind::Return(_)
        | StmtKind::Delete(_)
        | StmtKind::Pass
        | StmtKind::Break
        | StmtKind::Continue
        | StmtKind::Raise { .. }
        | StmtKind::Assert { .. }
        | StmtKind::Global(_)
        | StmtKind::Nonlocal(_)
        | StmtKind::TypeAlias { .. } => {}
    }

    Ok(())
}

fn visit_match_case(
    case: &MatchCase,
    current_package: &str,
    imports: &mut ImportCollector,
) -> Result<(), AotError> {
    visit_statements(&case.body, current_package, imports)
}

fn is_star_import(names: &[Alias]) -> bool {
    names.len() == 1 && names[0].name == "*"
}

fn resolve_relative_import(name: &str, level: u32, package: &str) -> Result<String, AotError> {
    if level == 0 {
        return Ok(name.to_string());
    }

    if package.is_empty() {
        return Err(AotError::InvalidEntrypoint {
            message: "attempted relative import in non-package".to_string(),
        });
    }

    let package_parts: Vec<&str> = package.split('.').collect();
    let level = level as usize;
    if level > package_parts.len() {
        return Err(AotError::InvalidEntrypoint {
            message: format!(
                "attempted relative import beyond top-level package (level={}, package depth={})",
                level,
                package_parts.len()
            ),
        });
    }

    let base_depth = package_parts.len() - level + 1;
    let base = package_parts[..base_depth].join(".");

    if name.is_empty() {
        Ok(base)
    } else {
        Ok(format!("{}.{}", base, name))
    }
}

#[cfg(test)]
mod tests;
