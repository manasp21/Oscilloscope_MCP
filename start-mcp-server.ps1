#!/usr/bin/env pwsh
# Oscilloscope MCP Server Startup Script for Windows
# This script starts the MCP server with microphone support for Claude Desktop integration

Write-Host "🔬 Starting Oscilloscope MCP Server for Claude Desktop..." -ForegroundColor Green
Write-Host ""

# Check if Node.js is installed
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js detected: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Node.js is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if npm is available
try {
    $npmVersion = npm --version
    Write-Host "✅ npm detected: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: npm is not available" -ForegroundColor Red
    Write-Host "Please ensure Node.js is properly installed" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Change to script directory
Set-Location $PSScriptRoot

# Install dependencies if needed
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ ERROR: Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
}

# Set environment variables for microphone mode
$env:HARDWARE_INTERFACE = "microphone"
$env:AUDIO_SAMPLE_RATE = "44100"
$env:DEBUG = "false"

# Display configuration
Write-Host ""
Write-Host "🔧 Configuration:" -ForegroundColor Cyan
Write-Host "   Hardware Interface: $env:HARDWARE_INTERFACE" -ForegroundColor White
Write-Host "   Audio Sample Rate: $env:AUDIO_SAMPLE_RATE Hz" -ForegroundColor White
Write-Host "   Debug Mode: $env:DEBUG" -ForegroundColor White
Write-Host ""

# Display Claude Desktop configuration info
Write-Host "📋 Claude Desktop Configuration:" -ForegroundColor Cyan
Write-Host "   Add this to your claude_desktop_config.json:" -ForegroundColor White
Write-Host ""
Write-Host '@{
  "mcpServers": {
    "oscilloscope": {
      "command": "node",
      "args": ["' + (Get-Location).Path + '\\dist\\index.js"],
      "env": {
        "HARDWARE_INTERFACE": "microphone",
        "AUDIO_SAMPLE_RATE": "44100"
      }
    }
  }
}' -ForegroundColor Gray
Write-Host ""

# Build the project first
Write-Host "📦 Building project..." -ForegroundColor Cyan
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERROR: Build failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✅ Build completed successfully" -ForegroundColor Green

# Start the MCP server
Write-Host "🚀 Starting MCP server..." -ForegroundColor Green
Write-Host "   The server will run until you close this window." -ForegroundColor Yellow
Write-Host "   Configure Claude Desktop to connect to this server." -ForegroundColor Yellow
Write-Host ""

try {
    npm run start
} catch {
    Write-Host "❌ ERROR: Failed to start MCP server" -ForegroundColor Red
    Write-Host "Check the error messages above for details" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Server stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"