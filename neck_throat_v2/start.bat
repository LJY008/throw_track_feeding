@echo off
REM 颈部咽喉追踪系统 v2 - 启动脚本
REM 
REM 使用方法:
REM   start.bat              - 正常启动（需要硬件相机）
REM   start.bat --simulate   - 模拟模式启动（无需硬件）
REM   start.bat --no-openpose - 禁用OpenPose
REM   start.bat --help       - 查看帮助

cd /d %~dp0
echo ====================================
echo   颈部咽喉追踪系统 v2 (模块化版本)
echo ====================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境，请确保已安装Python 3.8+
    pause
    exit /b 1
)

REM 运行主程序
python -m neck_throat_v2.main %*

echo.
echo 程序已退出
pause
