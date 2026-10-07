$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $taskRoot 'runtime\llama-server.exe'
$model = Join-Path $taskRoot 'models_data\Qwen3-4B-Q4_K_M.gguf'
if ((Get-Item -LiteralPath $model).Length -ne 2497280256) { throw 'Incomplete Qwen3-4B model file' }
$sha = (Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sha -ne '7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5') { throw 'Qwen3-4B model SHA256 mismatch' }
$logs = Join-Path $taskRoot 'runs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$argsList = @('-m', $model, '--device', 'Vulkan1', '-ngl', '99', '-c', '4096', '-np', '1', '-t', '4', '--host', '127.0.0.1', '--port', '8096', '--no-warmup')
$process = Start-Process -FilePath $exe -ArgumentList $argsList -WorkingDirectory (Join-Path $taskRoot 'runtime') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'server.stdout.log') -RedirectStandardError (Join-Path $logs 'server.stderr.log') -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $logs 'server.pid')
Write-Output "CRAVE local server PID: $($process.Id)"
