# run_chatbot.ps1  -  Run the Bangla RAG Chatbot from PowerShell
chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
python src\bangla_rag_chatbot.py
