$ErrorActionPreference='Stop'
$taskRoot='D:\Study\Papers\p6'
$taskDev=Join-Path $taskRoot 'experiments\mive_revision\artifacts\dev4b'
$taskCompletion=Get-Content -LiteralPath (Join-Path $taskDev 'completion.json') -Raw | ConvertFrom-Json
if ($taskCompletion.status -ne 'complete') { throw 'Development audit has not finished collecting.' }
$taskPid=[int](Get-Content -LiteralPath (Join-Path $taskDev 'parallel_server.pid'))
$taskExisting=Get-CimInstance Win32_Process -Filter "ProcessId=$taskPid"
if ($null -ne $taskExisting) {
    if ($taskExisting.CommandLine -notlike '*Qwen3-4B-Q4_K_M.gguf*--port 8096*') { throw 'Task server identity does not match; refusing to stop it.' }
    taskkill.exe /PID $taskPid /F
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the task-owned server.' }
}
$taskOut=Join-Path $taskRoot 'experiments\mive_revision\artifacts\smoke_Qwen2.5-7B'
$taskArgs=@('-m',(Join-Path $taskRoot 'crave\models_data\Qwen2.5-7B\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf'),'--device','Vulkan1','-ngl','99','-c','4096','-np','1','-t','4','--host','127.0.0.1','--port','8098','--no-warmup')
$taskProcess=Start-Process -FilePath (Join-Path $taskRoot 'crave\runtime\llama-server.exe') -ArgumentList $taskArgs -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskOut 'server.stdout.log') -RedirectStandardError (Join-Path $taskOut 'server.stderr.log') -PassThru
$taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskOut 'server.pid')
$taskProcess.Id
