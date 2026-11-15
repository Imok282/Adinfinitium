# 4090 / 5070 PC setup (fresh Windows, no WSL). An RTX 50xx (Blackwell) needs a CUDA 12.8+ PyTorch build and a recent driver: tried first below.: Python 3.12 venv, CUDA PyTorch, libraries, seed data, machine check.
# Run from the package root in PowerShell:   powershell -ExecutionPolicy Bypass -File ops\windows\setup_4090.ps1
# UNTESTED on Windows by the author (written on a Mac): if a step fails, read its message, fix that step, re-run (every step is safe to repeat).
Set-Location (Join-Path $PSScriptRoot "..\..")
$cand = @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
          "C:\Program Files\Python312\python.exe", "C:\Program Files\Python311\python.exe")
$py = $cand | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) {
  Write-Host "Installing Python 3.12 with winget..."
  winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
  $py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
}
if (-not (Test-Path $py)) { Write-Error "Python 3.12 not found. Install it from python.org (tick 'Add python.exe to PATH') and re-run."; exit 1 }
Write-Host "using $py"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $py -m venv .venv }
$vpy = (Resolve-Path ".venv\Scripts\python.exe").Path
& $vpy -m pip install --upgrade pip
$ok = $false
foreach ($cu in @("cu130", "cu128", "cu126", "cu124")) {
  Write-Host "trying PyTorch build $cu ..."
  & $vpy -m pip uninstall -y torch | Out-Null
  & $vpy -m pip install torch --index-url "https://download.pytorch.org/whl/$cu"
  if ($LASTEXITCODE -eq 0) {
    & $vpy -c "import torch, sys; ok = torch.cuda.is_available(); print(torch.__version__, ok); x = torch.randn(256, 256, device='cuda') if ok else None; y = (x @ x).sum().item() if ok else None; print(torch.cuda.get_device_name(0) if ok else 'no GPU', 'compute OK' if ok else ''); sys.exit(0 if ok else 1)"
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
  }
}
if (-not $ok) { Write-Error "No CUDA build of PyTorch can see the GPU. Run 'nvidia-smi'; if it fails or is old, install the latest NVIDIA Game Ready / Studio driver, reboot, re-run."; exit 1 }
& $vpy -m pip install numpy scipy pydantic pyyaml spiceypy matplotlib pytest
& $vpy -m pip install rasterio
if ($LASTEXITCODE -ne 0) { Write-Warning "rasterio did not install: fine for the brain4090 run (it is only needed to download new NASA tiles)" }
& $vpy -m pip install -e . --no-deps
if ((Test-Path "seed\adinfinitium_seed_data.zip") -and -not (Test-Path "data\tiles")) {
  Write-Host "unpacking seed data..."
  Expand-Archive -Path "seed\adinfinitium_seed_data.zip" -DestinationPath "." -Force
}
New-Item -ItemType Directory -Force runs | Out-Null
& $vpy -m ops.verify before
Write-Host "setup done. Next:  powershell -ExecutionPolicy Bypass -File ops\windows\run_brain.ps1"
