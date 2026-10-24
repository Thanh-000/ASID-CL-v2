param(
    [string]$VenvPath = "$env:LOCALAPPDATA\mm_hope_envs\phase0-py311",
    [int[]]$Seeds = @(2026),
    [double]$MemoryLearningRate = 0.001
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Phase-2 environment not found: $Python. Run .\run_mm_hope_phase1.ps1 first."
}

& $Python -m pip install --disable-pip-version-check `
    -r (Join-Path $WorkspaceRoot "experiments\mm_hope\requirements-phase1.txt")
& $Python -m pip install --disable-pip-version-check `
    --no-deps -e (Join-Path $WorkspaceRoot "external\nested_learning")
& $Python -m pytest (Join-Path $WorkspaceRoot "experiments\mm_hope\tests") -q
& $Python (Join-Path $WorkspaceRoot "run_mm_hope_phase2.py") `
    --seeds @Seeds --memory-learning-rate $MemoryLearningRate
