@echo off
echo Starting Oscilloscope MCP Server for Claude Desktop...
echo.

:: Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

:: Check if npm is available
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: npm is not available
    echo Please ensure Node.js is properly installed
    pause
    exit /b 1
)

:: Change to script directory
cd /d "%~dp0"

:: Install dependencies if needed
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

:: Set environment variables for microphone mode
set HARDWARE_INTERFACE=microphone
set AUDIO_SAMPLE_RATE=44100
set DEBUG=false

:: Build the project first
echo Building project...
npm run build
if %errorlevel% neq 0 (
    echo ERROR: Build failed
    pause
    exit /b 1
)
echo Build completed successfully

:: Start the MCP server
echo Starting MCP server with microphone support...
echo Hardware Interface: %HARDWARE_INTERFACE%
echo Audio Sample Rate: %AUDIO_SAMPLE_RATE%
echo Debug Mode: %DEBUG%
echo.
echo The server will run until you close this window.
echo Configure Claude Desktop to connect to this server.
echo.

npm run start

pause