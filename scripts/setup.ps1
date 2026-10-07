# SIGMA VI - setup and update for Windows ($0 configuration).
# Run from the SIGMA VI folder:
#     powershell -ExecutionPolicy Bypass -File .\setup.ps1
# Safe to re-run any time: it updates the code, never overwrites your .env, and skips finished steps.
# Add -SkipModels to skip the (large, one-time) local AI model downloads.

param([switch]$SkipModels)

$ErrorActionPreference = "Stop"
# Works whether this file sits in the project root or in scripts/.
$root = if ((Split-Path $PSScriptRoot -Leaf) -eq "scripts") { Split-Path $PSScriptRoot -Parent } else { $PSScriptRoot }
Set-Location -Path $root
function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# 1. Get or update the code (your .env and .venv are never touched)
Step "Getting the latest SIGMA VI code from GitHub"
if ((Test-Path ".\.git") -and (Get-Command git -ErrorAction SilentlyContinue)) {
    git pull --ff-only
} else {
    $tmp = Join-Path $env:TEMP ("sigma-vi-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp | Out-Null
    Invoke-WebRequest "https://github.com/bogdanseanwork/sigma-vi/archive/refs/heads/main.zip" -OutFile "$tmp\repo.zip"
    Expand-Archive "$tmp\repo.zip" -DestinationPath $tmp -Force
    $src = Join-Path $tmp "sigma-vi-main"
    # robocopy merges folders reliably; /XF and /XD protect your keys and the installed environment.
    robocopy $src . /E /XF .env setup-status.txt /XD .venv /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying the update failed (robocopy exit $LASTEXITCODE)." }
    $global:LASTEXITCODE = 0
    Copy-Item "$src\scripts\setup.ps1" -Destination $PSCommandPath -Force   # keep this script current
    Remove-Item $tmp -Recurse -Force
}

# 2. Python 3.12+
Step "Checking Python"
$py = $null
$check = "import sys; print(sys.version_info[:2] >= (3, 12))"
foreach ($cand in @(@("py", "-3.13"), @("py", "-3.12"), @("python"))) {
    try {
        $exe = $cand[0]; $rest = @($cand | Select-Object -Skip 1)
        $v = & $exe @rest -c $check 2>$null
        if ($v -eq "True") { $py = $cand; break }
    } catch {}
}
if (-not $py) {
    Write-Host "Python 3.12 or newer is required. Installing Python 3.12 with winget..."
    winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    Write-Host "`nPython installed. Close this window and run the setup command again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"; exit
}
Write-Host "Using: $($py -join ' ')"

# 3. Virtual environment + dependencies
Step "Installing dependencies"
if (-not (Test-Path ".\.venv")) {
    $exe = $py[0]; $rest = @($py | Select-Object -Skip 1)
    & $exe @rest -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
& .\.venv\Scripts\python.exe -m pip install -e ".[dev,data,ai]" --quiet
& .\.venv\Scripts\python.exe -m pip install uv --quiet   # also fixes the Serena plugin's missing 'uvx'

# 4. Tests
Step "Running the test suite"
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "Some tests failed - tell Claude what you see above." -ForegroundColor Red }

# 5. .env: create it, or add any new settings without changing your existing values
Step "Preparing your .env file"
if (-not (Test-Path ".\.env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env - opening it in Notepad. Paste each key after its '=' and save."
    Start-Process notepad.exe .env
} else {
    $have = @{}
    foreach ($line in Get-Content .env) { if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') { $have[$Matches[1]] = $true } }
    $new = @()
    foreach ($line in Get-Content .env.example) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=' -and -not $have.ContainsKey($Matches[1])) { $new += $line }
    }
    if ($new.Count -gt 0) {
        Add-Content .env ("`n# --- added by setup on " + (Get-Date -Format "yyyy-MM-dd") + " ---")
        Add-Content .env $new
        Write-Host "Added new settings to .env: $(($new | ForEach-Object { ($_ -split '=')[0] }) -join ', ')"
    } else {
        Write-Host ".env is up to date - your values were left untouched."
    }
}

# 6. Local AI models (free, run on this computer)
Step "Local AI models (Ollama)"
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe" }
if (-not (Test-Path $ollama)) {
    Write-Host "Installing Ollama with winget..."
    winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
}
if ($SkipModels) {
    Write-Host "Skipping model downloads (-SkipModels)."
} elseif (Test-Path $ollama) {
    Write-Host "Downloading gpt-oss:20b (about 14 GB) and qwen3.5:4b (about 3-4 GB). This is one-time and can take a while."
    & $ollama pull gpt-oss:20b
    & $ollama pull qwen3.5:4b
} else {
    Write-Host "Ollama isn't available yet. Restart the computer, then run setup again." -ForegroundColor Yellow
}

# 7. Status
Step "Integration status (shows which keys are present - never their values)"
$status = & .\.venv\Scripts\python.exe -m sigma.core.config
$status
$status | Out-File -FilePath setup-status.txt -Encoding utf8

Write-Host "`nDone. After filling in .env, run this again to re-check, then tell Claude 'keys are in'." -ForegroundColor Green
Read-Host "Press Enter to close"
