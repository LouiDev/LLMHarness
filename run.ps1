# Creates the venv on first run, installs deps, starts the server and opens the browser.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python -m pip install -q -r requirements.txt
Start-Process "http://127.0.0.1:8766"
& .\.venv\Scripts\python server.py
