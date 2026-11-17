param(
    [string]$DataPython = '',
    [string]$SolverPython = '',
    [string]$NodeExecutable = '',
    [switch]$SkipExperiments,
    [switch]$SkipExcel
)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies'
if (-not $DataPython) {
    $DataPython = Join-Path $runtimeDir 'python/python.exe'
    if (-not (Test-Path -LiteralPath $DataPython)) { $DataPython = 'python' }
}
if (-not $SolverPython) {
    $candidatePython = Join-Path $env:LOCALAPPDATA 'Programs/Python/Python312/python.exe'
    $SolverPython = if (Test-Path -LiteralPath $candidatePython) { $candidatePython } else { 'python' }
}
if (-not $NodeExecutable) {
    $NodeExecutable = Join-Path $runtimeDir 'node/bin/node.exe'
    if (-not (Test-Path -LiteralPath $NodeExecutable)) { $NodeExecutable = 'node' }
}
$env:PYTHONIOENCODING = 'utf-8'
function Run-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with code $LASTEXITCODE : $Executable" }
}
Run-Checked $DataPython @((Join-Path $PSScriptRoot 'data_forecast.py'))
Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'optimize_dispatch.py'), '--case', 'all')
if (-not $SkipExperiments) {
    Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'run_experiments.py'))
}
Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'make_paper_assets.py'))
if (-not $SkipExcel) {
    $modulePath = Join-Path $PSScriptRoot 'node_modules'
    $runtimeModules = Join-Path $runtimeDir 'node/node_modules'
    if (-not (Test-Path -LiteralPath $modulePath)) {
        if (-not (Test-Path -LiteralPath $runtimeModules)) {
            throw 'Excel export requires the bundled @oai/artifact-tool package.'
        }
        New-Item -ItemType Junction -Path $modulePath -Target $runtimeModules | Out-Null
    }
    Run-Checked $NodeExecutable @((Join-Path $PSScriptRoot 'export_workbooks.mjs'))
    Run-Checked $DataPython @((Join-Path $PSScriptRoot 'verify_workbooks.py'))
}
Write-Output 'Calculation, tables, figures and requested workbooks completed. No LaTeX compilation was run.'
