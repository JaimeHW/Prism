//! Abstract Syntax Tree definitions for Python 3.12.
//!
//! This module defines all AST node types that correspond to Python's grammar.
//! The structure closely follows CPython's AST for compatibility.

use prism_core::Span;

// =============================================================================
// Module Level
// =============================================================================

/// A module (file) in Python.
#[derive(Debug, Clone)]
pub struct Module {
    /// The statements in the module.
    pub body: Vec<Stmt>,
    /// Type ignores for this module.
    pub type_ignores: Vec<TypeIgnore>,
    /// Source span.
    pub span: Span,
}

impl Module {
    /// Create a new module.
    #[must_use]
    pub fn new(body: Vec<Stmt>, span: Span) -> Self {
        Self {
            body,
            type_ignores: Vec::new(),
            span,
        }
    }
}

/// A type ignore comment.
#[derive(Debug, Clone)]
pub struct TypeIgnore {
    /// Line number.
    pub lineno: u32,
    /// Optional tag.
    pub tag: String,
}

// =============================================================================
// Statements
// =============================================================================

/// A statement node.
#[derive(Debug, Clone)]
pub struct Stmt {
    /// The statement kind.
    pub kind: StmtKind,
    /// Source span.
    pub span: Span,
}

impl Stmt {
    /// Create a new statement.
    #[must_use]
    pub fn new(kind: StmtKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// Statement kinds.
#[derive(Debug, Clone)]
pub enum StmtKind {
    // === Simple Statements ===
    /// Expression statement.
    Expr(Box<Expr>),
    /// Assignment: `target = value`
    Assign {
        /// Assignment targets.
        targets: Vec<Expr>,
        /// Value being assigned.
        value: Box<Expr>,
    },
    /// Augmented assignment: `target += value`
    AugAssign {
        /// Target.
        target: Box<Expr>,
        /// Operator.
        op: AugOp,
        /// Value.
        value: Box<Expr>,
    },
    /// Annotated assignment: `target: annotation = value`
    AnnAssign {
        /// Target.
        target: Box<Expr>,
        /// Type annotation.
        annotation: Box<Expr>,
        /// Optional value.
        value: Option<Box<Expr>>,
        /// Whether this is a simple name target.
        simple: bool,
    },
    /// Return statement.
    Return(Option<Box<Expr>>),
    /// Delete statement.
    Delete(Vec<Expr>),
    /// Pass statement.
    Pass,
    /// Break statement.
    Break,
    /// Continue statement.
    Continue,
    /// Raise statement.
    Raise {
        /// Exception to raise.
        exc: Option<Box<Expr>>,
        /// Cause (for `raise X from Y`).
        cause: Option<Box<Expr>>,
    },
    /// Assert statement.
    Assert {
        /// Test expression.
        test: Box<Expr>,
        /// Optional message.
        msg: Option<Box<Expr>>,
    },
    /// Global statement.
    Global(Vec<String>),
    /// Nonlocal statement.
    Nonlocal(Vec<String>),

    // === Import Statements ===
    /// Import statement: `import x, y, z`
    Import(Vec<Alias>),
    /// From import: `from x import y, z`
    ImportFrom {
        /// Module name (None for relative imports like `from . import x`).
        module: Option<String>,
        /// Imported names.
        names: Vec<Alias>,
        /// Number of leading dots for relative imports.
        level: u32,
    },

    // === Compound Statements ===
    /// If statement.
    If {
        /// Test expression.
        test: Box<Expr>,
        /// Body.
        body: Vec<Stmt>,
        /// Elif and else clauses.
        orelse: Vec<Stmt>,
    },
    /// For loop.
    For {
        /// Loop variable.
        target: Box<Expr>,
        /// Iterator.
        iter: Box<Expr>,
        /// Body.
        body: Vec<Stmt>,
        /// Else clause.
        orelse: Vec<Stmt>,
    },
    /// Async for loop.
    AsyncFor {
        /// Loop variable.
        target: Box<Expr>,
        /// Iterator.
        iter: Box<Expr>,
        /// Body.
        body: Vec<Stmt>,
        /// Else clause.
        orelse: Vec<Stmt>,
    },
    /// While loop.
    While {
        /// Test expression.
        test: Box<Expr>,
        /// Body.
        body: Vec<Stmt>,
        /// Else clause.
        orelse: Vec<Stmt>,
    },
    /// With statement.
    With {
        /// Context items.
        items: Vec<WithItem>,
        /// Body.
        body: Vec<Stmt>,
    },
    /// Async with statement.
    AsyncWith {
        /// Context items.
        items: Vec<WithItem>,
        /// Body.
        body: Vec<Stmt>,
    },
    /// Try statement.
    Try {
        /// Try body.
        body: Vec<Stmt>,
        /// Exception handlers.
        handlers: Vec<ExceptHandler>,
        /// Else clause.
        orelse: Vec<Stmt>,
        /// Finally clause.
        finalbody: Vec<Stmt>,
    },
    /// Try-star statement (Python 3.11+).
    TryStar {
        /// Try body.
        body: Vec<Stmt>,
        /// Exception handlers.
        handlers: Vec<ExceptHandler>,
        /// Else clause.
        orelse: Vec<Stmt>,
        /// Finally clause.
        finalbody: Vec<Stmt>,
    },
    /// Match statement (Python 3.10+).
    Match {
        /// Subject expression.
        subject: Box<Expr>,
        /// Match cases.
        cases: Vec<MatchCase>,
    },

    // === Definitions ===
    /// Function definition.
    FunctionDef {
        /// Function name.
        name: String,
        /// Type parameters (Python 3.12+).
        type_params: Vec<TypeParam>,
        /// Arguments.
        args: Box<Arguments>,
        /// Body.
        body: Vec<Stmt>,
        /// Decorators.
        decorator_list: Vec<Expr>,
        /// Return type annotation.
        returns: Option<Box<Expr>>,
    },
    /// Async function definition.
    AsyncFunctionDef {
        /// Function name.
        name: String,
        /// Type parameters (Python 3.12+).
        type_params: Vec<TypeParam>,
        /// Arguments.
        args: Box<Arguments>,
        /// Body.
        body: Vec<Stmt>,
        /// Decorators.
        decorator_list: Vec<Expr>,
        /// Return type annotation.
        returns: Option<Box<Expr>>,
    },
    /// Class definition.
    ClassDef {
        /// Class name.
        name: String,
        /// Type parameters (Python 3.12+).
        type_params: Vec<TypeParam>,
        /// Base classes.
        bases: Vec<Expr>,
        /// Keyword arguments to metaclass.
        keywords: Vec<Keyword>,
        /// Body.
        body: Vec<Stmt>,
        /// Decorators.
        decorator_list: Vec<Expr>,
    },
    /// Type alias (Python 3.12+): `type X = int`
    TypeAlias {
        /// Name.
        name: Box<Expr>,
        /// Type parameters.
        type_params: Vec<TypeParam>,
        /// Value.
        value: Box<Expr>,
    },
}

// =============================================================================
// Expressions
// =============================================================================

/// An expression node.
#[derive(Debug, Clone)]
pub struct Expr {
    /// The expression kind.
    pub kind: ExprKind,
    /// Source span.
    pub span: Span,
}

impl Expr {
    /// Create a new expression.
    #[must_use]
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// Expression kinds.
#[derive(Debug, Clone)]
pub enum ExprKind {
    // === Literals ===
    /// Integer literal.
    Int(i64),
    /// Big integer literal (arbitrary precision).
    BigInt(String),
    /// Float literal.
    Float(f64),
    /// Complex literal.
    Complex { real: f64, imag: f64 },
    /// String literal (may be concatenation of multiple strings).
    String(StringLiteral),
    /// Bytes literal.
    Bytes(Vec<u8>),
    /// Bool literal.
    Bool(bool),
    /// None literal.
    None,
    /// Ellipsis literal.
    Ellipsis,

    // === Names ===
    /// Identifier reference.
    Name(String),
    /// Named expression (walrus operator): `x := value`
    NamedExpr {
        /// Target name.
        target: Box<Expr>,
        /// Value.
        value: Box<Expr>,
    },

    // === Container Literals ===
    /// List literal: `[1, 2, 3]`
    List(Vec<Expr>),
    /// Tuple literal: `(1, 2, 3)`
    Tuple(Vec<Expr>),
    /// Set literal: `{1, 2, 3}`
    Set(Vec<Expr>),
    /// Dict literal: `{k: v, ...}`
    Dict {
        /// Keys (None for `**d` unpacking).
        keys: Vec<Option<Expr>>,
        /// Values.
        values: Vec<Expr>,
    },

    // === Comprehensions ===
    /// List comprehension.
    ListComp {
        /// Element expression.
        elt: Box<Expr>,
        /// Generators.
        generators: Vec<Comprehension>,
    },
    /// Set comprehension.
    SetComp {
        /// Element expression.
        elt: Box<Expr>,
        /// Generators.
        generators: Vec<Comprehension>,
    },
    /// Dict comprehension.
    DictComp {
        /// Key expression.
        key: Box<Expr>,
        /// Value expression.
        value: Box<Expr>,
        /// Generators.
        generators: Vec<Comprehension>,
    },
    /// Generator expression.
    GeneratorExp {
        /// Element expression.
        elt: Box<Expr>,
        /// Generators.
        generators: Vec<Comprehension>,
    },

    // === Operations ===
    /// Binary operation.
    BinOp {
        /// Left operand.
        left: Box<Expr>,
        /// Operator.
        op: BinOp,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Unary operation.
    UnaryOp {
        /// Operator.
        op: UnaryOp,
        /// Operand.
        operand: Box<Expr>,
    },
    /// Boolean operation (and/or with short-circuit).
    BoolOp {
        /// Operator.
        op: BoolOp,
        /// Operands (at least 2).
        values: Vec<Expr>,
    },
    /// Comparison (chained).
    Compare {
        /// Left operand.
        left: Box<Expr>,
        /// Comparison operators.
        ops: Vec<CmpOp>,
        /// Comparison operands.
        comparators: Vec<Expr>,
    },

    // === Access ===
    /// Attribute access: `x.attr`
    Attribute {
        /// Object.
        value: Box<Expr>,
        /// Attribute name.
        attr: String,
    },
    /// Subscript: `x[index]`
    Subscript {
        /// Object.
        value: Box<Expr>,
        /// Index/slice.
        slice: Box<Expr>,
    },
    /// Slice: `start:stop:step`
    Slice {
        /// Start.
        lower: Option<Box<Expr>>,
        /// Stop.
        upper: Option<Box<Expr>>,
        /// Step.
        step: Option<Box<Expr>>,
    },
    /// Starred expression: `*args`
    Starred(Box<Expr>),

    // === Calls ===
    /// Function call.
    Call {
        /// Function.
        func: Box<Expr>,
        /// Positional arguments.
        args: Vec<Expr>,
        /// Keyword arguments.
        keywords: Vec<Keyword>,
    },

    // === Lambda and Conditionals ===
    /// Lambda expression.
    Lambda {
        /// Arguments.
        args: Box<Arguments>,
        /// Body.
        body: Box<Expr>,
    },
    /// Conditional expression: `a if test else b`
    IfExp {
        /// Test.
        test: Box<Expr>,
        /// Body (if true).
        body: Box<Expr>,
        /// Orelse (if false).
        orelse: Box<Expr>,
    },

    // === Async/Yield ===
    /// Await expression.
    Await(Box<Expr>),
    /// Yield expression.
    Yield(Option<Box<Expr>>),
    /// Yield from expression.
    YieldFrom(Box<Expr>),

    // === F-Strings ===
    /// Joined string (f-string with interpolations).
    JoinedStr(Vec<Expr>),
    /// Formatted value in f-string.
    FormattedValue {
        /// Value expression.
        value: Box<Expr>,
        /// Conversion character (-1 for none, 's', 'r', 'a').
        conversion: i8,
        /// Format spec as JoinedStr.
        format_spec: Option<Box<Expr>>,
    },
}

// =============================================================================
// Operators
// =============================================================================

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mult,
    /// `@`
    MatMult,
    /// `/`
    Div,
    /// `//`
    FloorDiv,
    /// `%`
    Mod,
    /// `**`
    Pow,
    /// `<<`
    LShift,
    /// `>>`
    RShift,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    /// `&`
    BitAnd,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `+x`
    UAdd,
    /// `-x`
    USub,
    /// `~x`
    Invert,
    /// `not x`
    Not,
}

/// Augmented assignment operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AugOp {
    /// `+=`
    Add,
    /// `-=`
    Sub,
    /// `*=`
    Mult,
    /// `@=`
    MatMult,
    /// `/=`
    Div,
    /// `//=`
    FloorDiv,
    /// `%=`
    Mod,
    /// `**=`
    Pow,
    /// `<<=`
    LShift,
    /// `>>=`
    RShift,
    /// `|=`
    BitOr,
    /// `^=`
    BitXor,
    /// `&=`
    BitAnd,
}

/// Boolean operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    /// `and`
    And,
    /// `or`
    Or,
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `==`
    Eq,
    /// `!=`
    NotEq,
    /// `<`
    Lt,
    /// `<=`
    LtE,
    /// `>`
    Gt,
    /// `>=`
    GtE,
    /// `is`
    Is,
    /// `is not`
    IsNot,
    /// `in`
    In,
    /// `not in`
    NotIn,
}

// =============================================================================
// String Literals
// =============================================================================

/// A string literal (possibly concatenated).
#[derive(Debug, Clone)]
pub struct StringLiteral {
    /// The string value.
    pub value: String,
    /// Whether this is a Unicode string.
    pub unicode: bool,
}

impl StringLiteral {
    /// Create a new string literal.
    #[must_use]
    pub fn new(value: String) -> Self {
        Self {
            value,
            unicode: true,
        }
    }
}

// =============================================================================
// Function Arguments
// =============================================================================

/// Function arguments specification.
#[derive(Debug, Clone, Default)]
pub struct Arguments {
    /// Position-only arguments (before `/`).
    pub posonlyargs: Vec<Arg>,
    /// Regular arguments.
    pub args: Vec<Arg>,
    /// *args.
    pub vararg: Option<Arg>,
    /// Keyword-only arguments (between `*` and `**`).
    pub kwonlyargs: Vec<Arg>,
    /// Defaults for keyword-only arguments.
    pub kw_defaults: Vec<Option<Expr>>,
    /// **kwargs.
    pub kwarg: Option<Arg>,
    /// Defaults for positional arguments.
    pub defaults: Vec<Expr>,
}

/// A single argument.
#[derive(Debug, Clone)]
pub struct Arg {
    /// Argument name.
    pub arg: String,
    /// Type annotation.
    pub annotation: Option<Box<Expr>>,
    /// Source span.
    pub span: Span,
}

/// A keyword argument in a call.
#[derive(Debug, Clone)]
pub struct Keyword {
    /// Argument name (None for `**kwargs`).
    pub arg: Option<String>,
    /// Value.
    pub value: Expr,
    /// Source span.
    pub span: Span,
}

// =============================================================================
// Import Aliases
// =============================================================================

/// An import alias: `name as asname`
#[derive(Debug, Clone)]
pub struct Alias {
    /// Original name.
    pub name: String,
    /// Alias (optional).
    pub asname: Option<String>,
    /// Source span.
    pub span: Span,
}

// =============================================================================
// Comprehensions
// =============================================================================

/// A comprehension clause.
#[derive(Debug, Clone)]
pub struct Comprehension {
    /// Loop variable.
    pub target: Expr,
    /// Iterator.
    pub iter: Expr,
    /// Filter conditions.
    pub ifs: Vec<Expr>,
    /// Whether this is an async comprehension.
    pub is_async: bool,
}

// =============================================================================
// Exception Handling
// =============================================================================

/// An exception handler.
#[derive(Debug, Clone)]
pub struct ExceptHandler {
    /// Exception type to catch.
    pub typ: Option<Expr>,
    /// Binding name.
    pub name: Option<String>,
    /// Handler body.
    pub body: Vec<Stmt>,
    /// Source span.
    pub span: Span,
}

// =============================================================================
// With Items
// =============================================================================

/// A context manager item in a with statement.
#[derive(Debug, Clone)]
pub struct WithItem {
    /// Context expression.
    pub context_expr: Expr,
    /// Optional as-binding.
    pub optional_vars: Option<Expr>,
}

// =============================================================================
// Match (Pattern Matching)
// =============================================================================

/// A match case.
#[derive(Debug, Clone)]
pub struct MatchCase {
    /// Pattern.
    pub pattern: Pattern,
    /// Guard expression.
    pub guard: Option<Expr>,
    /// Body.
    pub body: Vec<Stmt>,
}

/// A pattern in match statements.
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Pattern kind.
    pub kind: PatternKind,
    /// Source span.
    pub span: Span,
}

/// Pattern kinds.
#[derive(Debug, Clone)]
pub enum PatternKind {
    /// Match a literal value.
    MatchValue(Box<Expr>),
    /// Match singleton (True, False, None).
    MatchSingleton(Singleton),
    /// Match a sequence.
    MatchSequence(Vec<Pattern>),
    /// Match a mapping.
    MatchMapping {
        /// Keys.
        keys: Vec<Expr>,
        /// Patterns.
        patterns: Vec<Pattern>,
        /// Rest binding.
        rest: Option<String>,
    },
    /// Match a class.
    MatchClass {
        /// Class.
        cls: Box<Expr>,
        /// Positional patterns.
        patterns: Vec<Pattern>,
        /// Keyword patterns.
        kwd_attrs: Vec<String>,
        /// Keyword patterns.
        kwd_patterns: Vec<Pattern>,
    },
    /// Match with star (sequence unpacking).
    MatchStar(Option<String>),
    /// Match as (binding).
    MatchAs {
        /// Pattern (None for wildcard `_`).
        pattern: Option<Box<Pattern>>,
        /// Name to bind.
        name: Option<String>,
    },
    /// Match or (alternatives).
    MatchOr(Vec<Pattern>),
}

/// Singleton values for pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Singleton {
    /// True
    True,
    /// False
    False,
    /// None
    None,
}

// =============================================================================
// Type Parameters (Python 3.12+)
// =============================================================================

/// A type parameter.
#[derive(Debug, Clone)]
pub struct TypeParam {
    /// Type parameter kind.
    pub kind: TypeParamKind,
    /// Source span.
    pub span: Span,
}

/// Type parameter kinds.
#[derive(Debug, Clone)]
pub enum TypeParamKind {
    /// Type variable: `T`
    TypeVar {
        /// Name.
        name: String,
        /// Bound.
        bound: Option<Box<Expr>>,
    },
    /// Type var tuple: `*Ts`
    TypeVarTuple {
        /// Name.
        name: String,
    },
    /// Param spec: `**P`
    ParamSpec {
        /// Name.
        name: String,
    },
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        let module = Module::new(vec![], Span::dummy());
        assert!(module.body.is_empty());
    }

    #[test]
    fn test_stmt_creation() {
        let stmt = Stmt::new(StmtKind::Pass, Span::new(0, 4));
        assert!(matches!(stmt.kind, StmtKind::Pass));
    }

    #[test]
    fn test_expr_creation() {
        let expr = Expr::new(ExprKind::Int(42), Span::new(0, 2));
        assert!(matches!(expr.kind, ExprKind::Int(42)));
    }

    #[test]
    fn test_string_literal() {
        let lit = StringLiteral::new("hello".to_string());
        assert_eq!(lit.value, "hello");
        assert!(lit.unicode);
    }

    #[test]
    fn test_binop_variants() {
        let ops = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mult,
            BinOp::Div,
            BinOp::FloorDiv,
            BinOp::Mod,
            BinOp::Pow,
        ];
        assert_eq!(ops.len(), 7);
    }

    #[test]
    fn test_cmpop_variants() {
        let ops = [
            CmpOp::Eq,
            CmpOp::NotEq,
            CmpOp::Lt,
            CmpOp::LtE,
            CmpOp::Gt,
            CmpOp::GtE,
            CmpOp::Is,
            CmpOp::IsNot,
            CmpOp::In,
            CmpOp::NotIn,
        ];
        assert_eq!(ops.len(), 10);
    }

    #[test]
    fn test_default_arguments() {
        let args = Arguments::default();
        assert!(args.args.is_empty());
        assert!(args.vararg.is_none());
    }
}
