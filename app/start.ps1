Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    Write-Host 'README.md の導入手順で Python と必要ライブラリを準備してください。'
    exit 1
}
& '.\.venv\Scripts\python.exe' -m mahjong_helper @args
