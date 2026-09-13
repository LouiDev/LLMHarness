# Creates the venv on first run, installs deps, starts Ollama if needed, starts the server and opens the browser.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$ollamaHost = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST.TrimEnd("/") } else { "http://127.0.0.1:11434" }
if ($ollamaHost -notmatch '^https?://') { $ollamaHost = "http://$ollamaHost" }

function Test-Ollama {
    try { Invoke-RestMethod "$ollamaHost/api/version" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

if (-not (Test-Ollama)) {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Write-Host "Ollama is not running - starting 'ollama serve' in a separate process..."
        # Detached process with its own (hidden) console: keeps running after this window closes.
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
        $deadline = (Get-Date).AddSeconds(20)
        while (-not (Test-Ollama) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
        if (Test-Ollama) { Write-Host "Ollama is up at $ollamaHost" }
        else { Write-Warning "Ollama did not respond at $ollamaHost within 20s - continuing anyway." }
    } else {
        Write-Warning "Ollama is not running at $ollamaHost and 'ollama' was not found on PATH. Install it from https://ollama.com or start it manually."
    }
}

if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python -m pip install -q -r requirements.txt
Start-Process "http://127.0.0.1:8766"
& .\.venv\Scripts\python server.py
