#!/bin/bash
# Gasmask Loyiha Startup Script

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_BIN="/home/tenzorsoft/dataset/venv/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="/home/tenzorsoft/Downloads/gasmask_camera/.venv/bin/python"
fi

cd "$DIR"

echo "========================================================================"
echo "🤿 Gasmask & Head-Worn Verification AI Server ishga tushmoqda..."
echo "📄 Swagger UI Dokumentatsiya: http://localhost:8000/docs"
echo "🌐 Veb Dashboard:              http://localhost:8000/"
echo "🧠 Model:                     models/exp-3.pt"
echo "👁️ Masofa sozlamasi:           imgsz=1280 (Uzoq masofa uchun)"
echo "⌨️ To'xtatish uchun:           Ctrl+C"
echo "========================================================================"

exec "$PYTHON_BIN" main.py
