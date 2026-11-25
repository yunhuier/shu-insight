$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到 .venv，请先按 README 完成首次安装。"
}

& $python -m streamlit run `
    --server.showEmailPrompt false `
    --browser.gatherUsageStats false `
    (Join-Path $PSScriptRoot "app.py")
