@echo off
chcp 65001 >nul
echo 启动法律多智能体平台开发环境...

:: 终端1：FastAPI 后端
start "FastAPI 后端 :8085" cmd /k "cd /d D:\learn\legal-agent && D:\develop\Miniconda\envs\legal\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8085 --reload"

:: 等待后端初始化（约8秒）
echo 等待后端启动...
timeout /t 8 /nobreak >nul

:: 终端2：Gradio 测试台
start "Gradio 测试台 :7864" cmd /k "cd /d D:\learn\legal-agent && set LEGAL_AGENT_API_BASE=http://127.0.0.1:8085 && set LEGAL_AGENT_GRADIO_PORT=7864 && D:\develop\Miniconda\envs\legal\python.exe scripts/gradio_chat_demo.py"

echo 两个窗口已启动，浏览器将自动打开 http://127.0.0.1:7864
