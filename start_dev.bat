@echo off
chcp 65001 >nul
echo 启动法律多智能体平台开发环境...

:: 终端1：FastAPI 后端
start "FastAPI 后端 :8001" cmd /k "cd /d D:\learn\legal-agent && D:\develop\Miniconda\envs\legal\python.exe -m uvicorn src.main:app --port 8001 --reload"

:: 等待后端初始化（约8秒）
echo 等待后端启动...
timeout /t 8 /nobreak >nul

:: 终端2：Gradio 测试台
start "Gradio 测试台 :7862" cmd /k "cd /d D:\learn\legal-agent && D:\develop\Miniconda\envs\legal\python.exe scripts/gradio_chat_demo.py"

echo 两个窗口已启动，浏览器将自动打开 http://localhost:7862
