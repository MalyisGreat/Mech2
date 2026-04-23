param(
    [int]$SmokePid = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\joshj\joseph-stroud-identity-stability-research"
$pythonExe = "C:\Users\joshj\miniconda3\python.exe"
$scriptPath = Join-Path $repoRoot "scripts\self_recognition_nearfoil.py"
$configPath = Join-Path $repoRoot "configs\identity_battery\self_recognition_nearfoil_qwen_holdout.yaml"
$outputDir = Join-Path $repoRoot "outputs\latest\qwen_holdout_full\self_recognition_nearfoil"
$logPath = Join-Path $repoRoot "outputs\latest\qwen_holdout_full\run.log"
$smokeSummaryPath = Join-Path $repoRoot "outputs\latest\qwen_holdout\self_recognition_nearfoil\summary.csv"

New-Item -ItemType Directory -Path (Split-Path $logPath) -Force | Out-Null

if ($SmokePid -gt 0) {
    try {
        Wait-Process -Id $SmokePid
    } catch {
        "Smoke wait warning: $($_.Exception.Message)" | Out-File -FilePath $logPath -Append -Encoding utf8
    }
}

if (-not (Test-Path $smokeSummaryPath)) {
    "Smoke did not produce $smokeSummaryPath. Full Qwen holdout not started." | Out-File -FilePath $logPath -Append -Encoding utf8
    exit 1
}

"Starting Qwen holdout at $(Get-Date -Format o)" | Out-File -FilePath $logPath -Encoding utf8
"Command: $pythonExe $scriptPath --config $configPath" | Out-File -FilePath $logPath -Append -Encoding utf8

& $pythonExe $scriptPath --config $configPath *>> $logPath
$exitCode = $LASTEXITCODE

"Finished Qwen holdout at $(Get-Date -Format o) with exit code $exitCode" | Out-File -FilePath $logPath -Append -Encoding utf8
exit $exitCode
