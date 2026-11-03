<#
Start a long WSL job that outlives the launching shell, and keep Windows awake while it runs.

  powershell -File scripts\launch_detached.ps1 -Script scripts/phase0_pusht_full.sh [-ScriptArgs "50 5"]

Why: WSL shuts the whole Ubuntu VM down (killing even setsid/nohup jobs) as soon as no
wsl.exe session is open. So the job is the foreground command of a hidden wsl.exe
started through WMI, which puts it outside this shell's process tree and job object.
A second hidden process holds ES_SYSTEM_REQUIRED (idle sleep is blocked, nothing in
the power settings is changed) until the job ends. A closed lid or a manual Sleep
still suspends the machine.
#>
param(
    [Parameter(Mandatory)] [string] $Script,
    [string] $ScriptArgs = "",
    [string] $Distro = "Ubuntu-26.04",
    [string] $User = "aayush"
)

$repoWin = Split-Path -Parent $PSScriptRoot
$drive = $repoWin.Substring(0, 1).ToLower()
$repoWsl = "/mnt/$drive" + ($repoWin.Substring(2) -replace "\\", "/")

$inner = "cd '$repoWsl' && bash $Script $ScriptArgs"
$cmd = "wsl.exe -d $Distro -u $User -- bash -c `"$inner`""
$job = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
if ($job.ReturnValue -ne 0) { throw "could not start the job (WMI return code $($job.ReturnValue))" }

$guard = @"
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win.Power]::SetThreadExecutionState(0x80000001) | Out-Null
while (Get-Process -Id $($job.ProcessId) -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($guard))
$g = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "powershell.exe -NoProfile -WindowStyle Hidden -EncodedCommand $encoded" }

"job pid $($job.ProcessId), sleep-guard pid $($g.ProcessId)"
