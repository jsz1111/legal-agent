@echo off
chcp 65001 >nul
echo Starting legal-agent development environment...

start "FastAPI backend :8085" cmd /k "cd /d D:\learn\legal-agent && D:\develop\Miniconda\envs\legal\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8085 --reload"

echo Waiting for backend startup...
timeout /t 8 /nobreak >nul

start "Gradio test console :7864" cmd /k "cd /d D:\learn\legal-agent && set LEGAL_AGENT_API_BASE=http://127.0.0.1:8085 && set LEGAL_AGENT_GRADIO_PORT=7864 && D:\develop\Miniconda\envs\legal\python.exe scripts/gradio_chat_demo.py"
start "Legal web frontend :5173" cmd /k "cd /d D:\learn\legal-agent\frontend && npm run dev -- --host 127.0.0.1 --port 5173"

echo Web frontend: http://127.0.0.1:5173
echo Gradio test console: http://127.0.0.1:7864
