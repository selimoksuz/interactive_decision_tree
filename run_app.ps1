$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    py -3 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\interactive_decision_tree_app.py --server.port 8501
