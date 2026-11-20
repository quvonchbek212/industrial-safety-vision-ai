@echo off
chcp 65001 > nul
title Gasmask & Head Verification AI Server

echo ========================================================================
echo 🤿 Gasmask & Head-Worn Verification AI Server ishga tushmoqda...
echo 📄 Swagger UI Dokumentatsiya: http://localhost:8000/docs
echo 🌐 Veb Dashboard:              http://localhost:8000/
echo 🧠 Model:                     models/exp-3.pt
echo 💻 Qurilma:                   CPU rejimi (Noutbuk)
echo 👁️ Masofa sozlamasi:           imgsz=1280 (Uzoq masofa uchun)
echo ⌨️ To'xtatish uchun:           Ctrl+C
echo ========================================================================

python main.py
pause
