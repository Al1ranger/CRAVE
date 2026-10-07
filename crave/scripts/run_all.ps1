param([int]$TrainTasks=4, [int]$EvalTasks=25, [int]$Budgets=2)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$workspaceRoot=Split-Path -Parent $taskRoot
$python=Join-Path $taskRoot '.venv\Scripts\python.exe'
$env:OPENBLAS_NUM_THREADS='2'
Set-Location -LiteralPath $workspaceRoot
& $python -m unittest discover -s crave/tests -v
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed' }
& $python -u -m crave.scripts.prepare all
if ($LASTEXITCODE -ne 0) { throw 'Asset preparation failed; resumable files preserved' }
& (Join-Path $PSScriptRoot 'start_server.ps1')
for ($attempt=0;$attempt -lt 120;$attempt++) {
    try { $health=Invoke-RestMethod -Uri 'http://127.0.0.1:8096/health'; if ($health.status -eq 'ok') { break } } catch { }
    Start-Sleep -Seconds 1
}
if ($attempt -eq 120) { throw 'Local server failed to become healthy; inspect runs/server.stderr.log' }
& $python -u -m crave.pipeline --train-tasks $TrainTasks --eval-tasks $EvalTasks --budgets $Budgets
if ($LASTEXITCODE -ne 0) { throw 'Pilot failed; inspect stage artifacts; final manuscript generation prohibited' }
