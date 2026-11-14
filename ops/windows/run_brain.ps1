# ONE command for the 4090: setup if needed -> pilot brain + planner -> exams -> verify -> freeze -> results zip. Safe to re-run: it resumes.
#   powershell -ExecutionPolicy Bypass -File ops\windows\run_brain.ps1                      # long: ~4 h of training (default, chosen by Imok)
#   powershell -ExecutionPolicy Bypass -File ops\windows\run_brain.ps1 -BudgetScale 1.0     # ~2 h
param([double]$BudgetScale = 2.0)
Set-Location (Join-Path $PSScriptRoot "..\..")
New-Item -ItemType Directory -Force runs | Out-Null
Start-Transcript -Path "runs\run_brain.log" -Append
if (-not (Test-Path ".venv\Scripts\python.exe")) { & powershell -ExecutionPolicy Bypass -File "ops\windows\setup_4090.ps1" }
$vpy = (Resolve-Path ".venv\Scripts\python.exe").Path
$env:PYTHONUTF8 = "1"
# keep the PC awake while plugged in (restore later with: powercfg /restoredefaultschemes)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
$names = "lander_pilot,planner_set,planner_logistic"
$fams = "lander,planner"
Write-Host "time budget scale $BudgetScale (1.0 = 2 h of training)"
& $vpy -m ops.overnight.run_all --profile brain4090 --run-id brain_4090 --resume --budget-scale $BudgetScale
& $vpy -m ops.verify after --names $names --families $fams
& $vpy -m ops.freeze --names $names --families $fams
& $vpy -m ops.freeze --check --names $names --families $fams
$files = @("runs") + (Get-ChildItem "data\models" -Filter "lander_pilot*.pt" | ForEach-Object { $_.FullName }) + (Get-ChildItem "data\models" -Filter "planner_*.pt" | ForEach-Object { $_.FullName })
if (Test-Path "data\planner") { $files += "data\planner" }
Compress-Archive -Force -Path $files -DestinationPath "adinfinitium_results_4090.zip"
Write-Host "ALL DONE: adinfinitium_results_4090.zip (see runs\overnight\brain_4090\RUN_REPORT.md)"
Stop-Transcript
