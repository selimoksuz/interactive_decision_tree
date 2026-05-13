$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$PythonArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
    $PythonArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
} else {
    Write-Error "Python 3.10+ is required but was not found on PATH."
    exit 1
}

& $PythonExe @PythonArgs "$ProjectRoot\scripts\start_app.py" @args
exit $LASTEXITCODE
