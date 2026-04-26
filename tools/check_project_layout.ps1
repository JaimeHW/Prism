$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")

$requiredFiles = @(
    ".cargo/config.toml",
    "rust-toolchain.toml",
    "rustfmt.toml"
)

$maxNewRustFileLines = 3000
$knownLargeRustFiles = @{}

$errors = New-Object System.Collections.Generic.List[string]

foreach ($relative in $requiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $relative) -PathType Leaf)) {
        $errors.Add("missing required project file: $relative")
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
