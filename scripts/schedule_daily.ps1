# Registers a Windows scheduled task that runs the SIGMA VI daily refresh every evening.
#   powershell -ExecutionPolicy Bypass -File .\scripts\schedule_daily.ps1            (install / update)
#   powershell -ExecutionPolicy Bypass -File .\scripts\schedule_daily.ps1 -Remove    (uninstall)
# Runs as you, only while you are logged in, so no password is stored. If the PC is asleep at the
# scheduled time, it runs as soon as the PC wakes (StartWhenAvailable). Output goes to daily_report.txt.
param([string]$At = "19:30", [switch]$Remove)
$ErrorActionPreference = "Stop"
$name = "SIGMA-VI daily refresh"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed '$name'."
    return
}

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Cannot find $py. Run scripts\setup.ps1 first." }

$action = New-ScheduledTaskAction -Execute $py -Argument "-m sigma.data.daily" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
    -Description "Pulls prices, filings, analyst estimates and macro data for the SIGMA VI watchlist." -Force | Out-Null
Write-Host "Scheduled '$name' every day at $At. Check daily_report.txt in $root after it runs."
Write-Host "To run it right now: Start-ScheduledTask -TaskName '$name'"
