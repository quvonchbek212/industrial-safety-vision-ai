# Gasmask Loyiha PowerShell Startup Script
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "🤿 Gasmask & Head-Worn Verification AI Server ishga tushmoqda..." -ForegroundColor Green
Write-Host "📄 Swagger UI Dokumentatsiya: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "🌐 Veb Dashboard:              http://localhost:8000/" -ForegroundColor Yellow
Write-Host "🧠 Model:                     models/exp-3.pt" -ForegroundColor Gray
Write-Host "💻 Qurilma:                   CPU rejimi (Noutbuk)" -ForegroundColor Gray
Write-Host "👁️ Masofa sozlamasi:           imgsz=1280 (Uzoq masofa uchun)" -ForegroundColor Gray
Write-Host "⌨️ To'xtatish uchun:           Ctrl+C" -ForegroundColor Red
Write-Host "========================================================================" -ForegroundColor Cyan

python main.py
