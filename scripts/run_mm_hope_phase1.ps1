param(
    [string]$VenvPath = "$env:LOCALAPPDATA\mm_hope_envs\phase0-py311",
    [string[]]$Models = @("B03", "B04", "B05", "B06")
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $VenvPath
}

& $Python -m pip install --disable-pip-version-check `
    -r (Join-Path $WorkspaceRoot "experiments\mm_hope\requirements-phase1.txt")
& $Python -m pip install --disable-pip-version-check `
    --no-deps -e (Join-Path $WorkspaceRoot "external\nested_learning")
& $Python -m pytest (Join-Path $WorkspaceRoot "experiments\mm_hope\tests") -q
& $Python (Join-Path $WorkspaceRoot "run_mm_hope_phase1.py") --models @Models
