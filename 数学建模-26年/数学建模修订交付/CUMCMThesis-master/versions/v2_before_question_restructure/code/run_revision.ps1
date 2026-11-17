<#
Reproduce the settlement/control revision without compiling LaTeX.
Default: reuse the frozen January selection and prepared_v2 inputs.
Use -Recalibrate explicitly to rerun January forecast/controller selection.
The original run_all.ps1 remains the v1 entry point and is not modified here.
#>
param(
    [string]$DataPython = '',
    [string]$SolverPython = '',
    [string]$NodeExecutable = '',
    [ValidateSet(1, 2)][int]$Workers = 2,
    [switch]$Recalibrate,
    [switch]$SkipExperiments,
    [switch]$RecomputeDiagnostics,
    [switch]$SkipExcel
)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $projectDir 'data'
$runtimeDir = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies'
$frozenV1Dir = Join-Path $projectDir 'versions/v1_before_settlement_revision'
$cases = @('q2', 'q3', 'q4_2', 'q4_3')

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
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

function Run-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with code $LASTEXITCODE : $Executable $($Arguments -join ' ')"
    }
}
function Require-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required preserved input is missing: $Path"
    }
}
function Read-Json([string]$Path) {
    Require-File $Path
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

if ($Recalibrate -and $SkipExperiments) {
    throw '-Recalibrate changes the policy selection; remove -SkipExperiments to recompute its comparisons.'
}
foreach ($case in $cases) {
    # These arrays supply each case's common, already executed January prefix.
    Require-File (Join-Path $frozenV1Dir "data/${case}_full.npz")
}
Require-File (Join-Path $frozenV1Dir 'code/optimize_dispatch.py')
Require-File (Join-Path $frozenV1Dir 'data/prepared.npz')
Require-File (Join-Path $dataDir 'q1.json')

if ($Recalibrate) {
    if (-not (Test-Path -LiteralPath (Join-Path $dataDir 'prepared.npz'))) {
        Run-Checked $DataPython @((Join-Path $PSScriptRoot 'data_forecast.py'))
    }
    Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'calibrate_v2.py'))
}
Require-File (Join-Path $dataDir 'prepared.npz')
$selection = Read-Json (Join-Path $dataDir 'selection_v2.json')
if (-not $selection.frozen) {
    throw 'selection_v2.json is not frozen. Complete January calibration with -Recalibrate.'
}
foreach ($case in $cases) { Require-File (Join-Path $dataDir "prepared_v2_${case}.npz") }
Require-File (Join-Path $dataDir 'time_mapping.json')
# Validate the frozen mapping without regenerating labels or shifting arrays.
Run-Checked $DataPython @((Join-Path $PSScriptRoot 'time_index.py'))

if ($SkipExperiments) {
    $experiments = Read-Json (Join-Path $dataDir 'experiments_v2.json')
    Require-File (Join-Path $dataDir 'experiments_v2.npz')
    if (-not $experiments.completed) { throw 'Saved experiments are incomplete; remove -SkipExperiments.' }
    $checks = @{
        'selection_v2.json' = (Join-Path $dataDir 'selection_v2.json')
        'dispatch_v2.py' = (Join-Path $PSScriptRoot 'dispatch_v2.py')
    }
    # experiments_v2 compares Q2/Q3 only; its source manifest has no Q4 inputs.
    foreach ($case in @('q2', 'q3')) {
        $checks["prepared_v2_${case}.npz"] = Join-Path $dataDir "prepared_v2_${case}.npz"
    }
    foreach ($name in $checks.Keys) {
        $expected = $experiments.sources_sha256.PSObject.Properties[$name].Value
        $actual = (Get-FileHash -LiteralPath $checks[$name] -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $expected -or $actual -ne $expected) {
            throw "Saved experiment configuration/input changed ($name); remove -SkipExperiments."
        }
    }
}

Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'run_main_v2.py'), '--workers', [string]$Workers)
if (-not $SkipExperiments) {
    Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'experiments_v2.py'), '--workers', [string]$Workers)
}

# The original-policy diagnostic is fixed by the v1 snapshot. The inner-LP
# diagnostic also records current policy/data hashes; refresh when they differ.
$v1GapPath = Join-Path $dataDir 'execution_gap_v1.json'
if ($RecomputeDiagnostics -or -not (Test-Path -LiteralPath $v1GapPath)) {
    Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'diagnose_execution_gap.py'))
}
Require-File (Join-Path $dataDir 'execution_gap_v1.npz')
$innerGapPath = Join-Path $dataDir 'inner_gap_v2.json'
$refreshInnerGap = $RecomputeDiagnostics -or $Recalibrate -or -not (Test-Path -LiteralPath $innerGapPath)
if (-not $refreshInnerGap) {
    $innerGap = Read-Json $innerGapPath
    $refreshInnerGap = -not $innerGap.completed
    foreach ($property in $innerGap.sources_sha256.PSObject.Properties) {
        $sourcePath = if ($property.Name -like '*.py') {
            Join-Path $PSScriptRoot $property.Name
        } else { Join-Path $dataDir $property.Name }
        Require-File $sourcePath
        if ((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) {
            $refreshInnerGap = $true
        }
    }
}
if ($refreshInnerGap) {
    Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'diagnose_inner_v2.py'))
}

Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'make_paper_assets_v2.py'))
Run-Checked $SolverPython @((Join-Path $PSScriptRoot 'check_dispatch_v2.py'))
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
Write-Output 'Revision calculation and requested output checks completed. No LaTeX compilation or paper review was run.'
