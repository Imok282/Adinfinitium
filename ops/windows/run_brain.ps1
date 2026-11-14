# ONE command for the 4090: setup if needed -> pilot brain + planner -> exams -> verify -> freeze -> results zip. Safe to re-run: it resumes.
#   powershell -ExecutionPolicy Bypass -File ops\windows\run_brain.ps1                      # ~2 h, many simulations in parallel (default)
#   powershell -ExecutionPolicy Bypass -File ops\windows\run_brain.ps1 -BudgetScale 2.0     # ~4 h
#   powershell -ExecutionPolicy Bypass -File ops\windows\run_brain.ps1 -RunProfile brain4090   # big GPU (4090 / 5070)
param([double]$BudgetScale = 1.0, [string]$RunProfile = "brain3050")
Set-Location (Join-Path $PSScriptRoot "..\..")
[Environment]::CurrentDirectory = (Get-Location).ProviderPath      # child programs (python, pip) start in the package folder too
New-Item -ItemType Directory -Force runs | Out-Null
Start-Transcript -Path "runs\run_brain.log" -Append
if (-not (Test-Path ".venv\Scripts\python.exe")) { & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "setup_4090.ps1") }
if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "SETUP FAILED: no .venv - read the messages above (antivirus? no internet?), fix, run again"; Stop-Transcript; exit 1 }
$vpy = (Resolve-Path ".venv\Scripts\python.exe").Path
& $vpy -c "import torch; assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) { Write-Host "SETUP FAILED: PyTorch cannot use the GPU - run ops\windows\setup_4090.ps1 again"; Stop-Transcript; exit 1 }
$env:PYTHONUTF8 = "1"
# keep the PC awake while plugged in (restore later with: powercfg /restoredefaultschemes)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
$names = "lander_pilot,planner_set,planner_logistic"
$fams = "lander,planner"
Write-Host "time budget scale $BudgetScale (1.0 = 2 h of training)"
& $vpy -m ops.overnight.run_all --profile $RunProfile --run-id brain_pc --resume --budget-scale $BudgetScale
& $vpy -m ops.verify after --names $names --families $fams
& $vpy -m ops.freeze --names $names --families $fams
& $vpy -m ops.freeze --check --names $names --families $fams
$files = @("runs") + (Get-ChildItem "data\models" -Filter "lander_pilot*.pt" | ForEach-Object { $_.FullName }) + (Get-ChildItem "data\models" -Filter "planner_*.pt" | ForEach-Object { $_.FullName })
if (Test-Path "data\planner") { $files += "data\planner" }
Compress-Archive -Force -Path $files -DestinationPath "adinfinitium_results_pc.zip"
Write-Host "ALL DONE: adinfinitium_results_pc.zip (see runs\overnight\brain_pc\RUN_REPORT.md)"
Stop-Transcript
