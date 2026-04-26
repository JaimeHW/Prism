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

function Test-RustTestOnlyFile {
    param([string] $RelativePath)

    $parts = $RelativePath -split "[/\\]"
    $leaf = $parts[-1]

    if ($leaf -eq "tests.rs" -or $leaf.EndsWith("_tests.rs")) {
        return $true
    }

    return $parts -contains "tests"
}

function Test-ContainsInlineTestModule {
    param([string] $Path)

    $lines = Get-Content -LiteralPath $Path
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -notmatch "^\s*#\[cfg\(test\)\]") {
            continue
        }

        $limit = [Math]::Min($lines.Count - 1, $i + 4)
        for ($j = $i + 1; $j -le $limit; $j++) {
            if ($lines[$j] -match "^\s*mod\s+\w+\s*\{") {
                return $true
            }
        }
    }

    return $false
}

foreach ($relative in $requiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $relative) -PathType Leaf)) {
        $errors.Add("missing required project file: $relative")
    }
}

$trackedFiles = git -C $root ls-files --cached --others --exclude-standard
foreach ($relative in $trackedFiles) {
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        continue
    }

    if ($relative.EndsWith(".rs")) {
        $lineCount = (Get-Content -LiteralPath $path | Measure-Object -Line).Lines
        if ($lineCount -gt $maxNewRustFileLines -and -not $knownLargeRustFiles.ContainsKey($relative)) {
            $errors.Add("$relative has $lineCount lines; split it or add an explicit architecture waiver")
        }

        if (-not (Test-RustTestOnlyFile $relative) -and (Test-ContainsInlineTestModule $path)) {
            $errors.Add("$relative contains inline unit tests; move tests into a separate tests.rs module")
        }

        if (-not (Test-RustTestOnlyFile $relative)) {
            $testAttribute = Select-String -LiteralPath $path -Pattern "^\s*#\[(tokio::)?test\]" -Quiet
            if ($testAttribute) {
                $errors.Add("$relative contains test functions; move tests into a separate test-only file")
            }
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
