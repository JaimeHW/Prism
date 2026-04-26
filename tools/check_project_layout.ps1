$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")

$requiredFiles = @(
    ".github/workflows/ci.yml",
    ".cargo/config.toml",
    "CONTRIBUTING.md",
    "docs/ARCHITECTURE.md",
    "docs/PERFORMANCE.md",
    "docs/TESTING.md",
    "docs/UNSAFE.md",
    "rust-toolchain.toml",
    "rustfmt.toml"
)

$docLinks = @(
    "docs/ARCHITECTURE.md",
    "docs/PERFORMANCE.md",
    "docs/TESTING.md",
    "docs/UNSAFE.md",
    "CONTRIBUTING.md"
)

$maxNewRustFileLines = 3000
$knownLargeRustFiles = @{
    "prism_compiler/src/compiler.rs" = $true
    "prism_jit/src/tier1/lower.rs" = $true
    "prism_vm/src/builtins/types.rs" = $true
    "prism_vm/src/ops/calls.rs" = $true
    "prism_vm/src/ops/method_dispatch/builtin_methods.rs" = $true
    "prism_vm/src/ops/objects.rs" = $true
    "prism_vm/src/vm.rs" = $true
    "prism_vm/tests/integration.rs" = $true
}

$errors = New-Object System.Collections.Generic.List[string]

foreach ($relative in $requiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $relative) -PathType Leaf)) {
        $errors.Add("missing required project file: $relative")
    }
}

$readmePath = Join-Path $root "README.md"
$readme = Get-Content -LiteralPath $readmePath -Raw
foreach ($relative in $docLinks) {
    if (-not $readme.Contains($relative)) {
        $errors.Add("README.md does not link to $relative")
    }
}

$trackedFiles = git -C $root ls-files
foreach ($relative in $trackedFiles) {
    if ($relative.EndsWith(".rs")) {
        $path = Join-Path $root $relative
        $lineCount = (Get-Content -LiteralPath $path | Measure-Object -Line).Lines
        if ($lineCount -gt $maxNewRustFileLines -and -not $knownLargeRustFiles.ContainsKey($relative)) {
            $errors.Add("$relative has $lineCount lines; split it or add an explicit architecture waiver")
        }
    }

    $leaf = Split-Path -Leaf $relative
    if ($leaf -eq $relative -and ($relative.EndsWith(".log") -or $relative.EndsWith(".txt"))) {
        $errors.Add("scratch artifact is tracked at repository root: $relative")
    }
}

if ($errors.Count -gt 0) {
    foreach ($errorMessage in $errors) {
        Write-Error "layout-check: $errorMessage"
    }
    exit 1
}

Write-Host "layout-check: ok"
