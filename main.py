"""
Gasmask Loyiha — FastAPI Video & Image Analysis Server
Features:
- Swagger UI Documentation at /docs
- Video Upload and Processing with Head-Worn Gasmask Verification Algorithm
- Distant Object High-Resolution Inference (imgsz=1280)
- Person Tracking & Violations Analytics
- Web Dashboard
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
import torch
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from core.detector import GasmaskSystemDetector
from core.video_processor import VideoProcessor

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="🤿 Gasmask & Head Verification AI",
    description="""
### Real-Time Gasmask & Person Head Verification System

Bu tizim video va tasvirlarda odamlarni va gazmaskalarni aniqlab, **oldi, yon va orqa ko'rinish** bo'yicha tahlil qiladi:
* **gasmask:** `WEARING_GASMASK` (Yashil) — Boshida/yuzida gazmaska taqilgan (oldi va yon tomondan).
* **no-gasmask:** `NO_GASMASK_VIOLATION` (Qizil) — Odamning yuzi ko'rinib turibdi (oldi va yon tomondan), lekin maska yo'q.
* **Orqa tomondan:** Odam orqaga qaragan bo'lsa, hechnima chiqarilmaydi (inobatga olinmaydi).
* **Qo'lda ushlangan / yerda yotgan maska:** Boshga tegishli qilinmaydi, model aldanmaydi.
    """,
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize Detector and Video Processor
MODEL_PATH = MODELS_DIR / "exp-3.pt"
if not MODEL_PATH.exists():
    expert_fallback = MODELS_DIR / "expert.pt"
    if expert_fallback.exists():
        MODEL_PATH = expert_fallback

PERSON_MODEL_PATH = MODELS_DIR / "yolov8n-pose.pt"
if not PERSON_MODEL_PATH.exists():
    root_pose = BASE_DIR / "yolov8n-pose.pt"
    if root_pose.exists():
        PERSON_MODEL_PATH = root_pose

detector = GasmaskSystemDetector(
    gasmask_model_path=str(MODEL_PATH),
    person_model_path=str(PERSON_MODEL_PATH),
    default_imgsz=1280,
    default_conf=0.25
)
video_processor = VideoProcessor(detector=detector)


@app.get("/api/status", summary="Tizim va GPU holati", tags=["System"])
def get_system_status():
    """Tizim, GPU va faol model holatini tekshirish"""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if torch.cuda.is_available() else 0
    vram_used_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 2) if torch.cuda.is_available() else 0

    return {
        "status": "online",
        "device": detector.device,
        "gpu_name": gpu_name,
        "vram_total_gb": vram_total_gb,
        "vram_used_gb": vram_used_gb,
        "active_gasmask_model": str(MODEL_PATH.name),
        "model_classes": detector.gasmask_model.names,
        "high_res_default_imgsz": detector.default_imgsz,
        "head_worn_rule": "Oldi va yon tomondan gazmaska taqilgan bo'lsa 'gasmask', yuzi ochiq bo'lsa 'no-gasmask', orqa tomondan qaralganda hechnarsa chiqarilmaydi; bitta odamga qat'iy bitta yorliq"
    }


@app.post("/api/video/analyze", summary="Videoni tekshirish (Upload & Process)", tags=["Video Analysis"])
async def analyze_video(
    file: UploadFile = File(..., description="Tekshirish uchun video fayl (.mp4, .avi, .mov)"),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="Ishonchlilik chegarasi (kichik/uzoq obyektlar uchun 0.20-0.25 tavsiya)"),
    imgsz: int = Query(1280, enum=[640, 960, 1280, 1536], description="Inference o'lchami (Uzoqdagi odamlarni aniqlash uchun 1280 tavsiya)"),
    frame_stride: int = Query(2, ge=1, le=10, description="Har nechanchi kadrni tekshirish (1: har kadr, 2: 2 barobar tezroq, 3: 3 barobar tezroq)"),
    show_boxes: bool = Query(False, description="Qutilarni (bounding box) ko'rsatish (False = faqat toza 'gasmask / no-gasmask' yozuvlari)"),
    show_distance: bool = Query(False, description="Yozuv yonida masofani (masalan: ~3.2m) ko'rsatish")
):
    """
    Kameradan olingan videoni to'liq tekshirish:
    1. Barcha odamlar va gazmaskalarni yuqori aniqlikda aniqlaydi (`imgsz=1280`).
    2. Odamning faqat **yelkadan tepasi (bosh qismi)** tekshiriladi (qo'lida, belida yoki yerda bo'lsa hisoblamaydi).
    3. Bitta odam boshiga qat'iy 1 ta toza 'gasmask' yoki 'no-gasmask' belgisi chiqadi.
    4. Masofani hisoblaydi (Yaqin, O'rta, Uzoq).
    """
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in valid_exts:
        raise HTTPException(status_code=400, detail=f"Faqat video fayllar qabul qilinadi: {valid_exts}")

    file_id = str(uuid.uuid4())[:8]
    input_filename = f"upload_{file_id}_{file.filename}"
    output_filename = f"annotated_{file_id}_{Path(file.filename).stem}.mp4"

    input_path = UPLOAD_DIR / input_filename
    output_path = OUTPUT_DIR / output_filename

    # Save uploaded video
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run video processor
        results = video_processor.process_video(
            input_video_path=str(input_path),
            output_video_path=str(output_path),
            conf=conf,
            imgsz=imgsz,
            frame_stride=frame_stride,
            show_boxes=show_boxes,
            show_distance=show_distance
        )

        results["download_url"] = f"/api/video/download/{output_filename}"
        results["stream_url"] = f"/api/video/stream/{output_filename}"
        results["output_filename"] = output_filename

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Videoni qayta ishlashda xatolik: {str(e)}")


@app.get("/api/video/download/{filename}", summary="Natijaviy videoni yuklab olish", tags=["Video Analysis"])
def download_video(filename: str):
    """Qayta ishlangan va chizilgan videoni yuklab olish"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video topilmadi")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@app.get("/api/video/stream/{filename}", summary="Videoni brauzerda ko'rish (Stream)", tags=["Video Analysis"])
def stream_video(filename: str):
    """Natijaviy videoni brauzer pleyerida tomosha qilish"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video topilmadi")
    return FileResponse(file_path, media_type="video/mp4")


@app.post("/api/image/analyze", summary="Bitta rasm yoki kadrni tekshirish", tags=["Image Analysis"])
async def analyze_image(
    file: UploadFile = File(..., description="Tekshirish uchun rasm fayl"),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    imgsz: int = Query(1280, enum=[640, 960, 1280, 1536]),
    show_boxes: bool = Query(False, description="Qutilarni (bounding box) ko'rsatish (False = toza yozuvlar)"),
    show_distance: bool = Query(False, description="Masofani ko'rsatish"),
    return_image: bool = Query(False, description="True bo'lsa chizilgan JPEG tasvirni qaytaradi, False bo'lsa JSON")
):
    """Bitta rasm yoki kadrni kiritib, masofa va boshda gazmaska borligini tekshirish"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Rasm faylini ochib bo'lmadi")

    analysis = detector.detect_frame(frame, conf=conf, imgsz=imgsz)

    if return_image:
        annotated = video_processor.draw_annotations(
            frame.copy(),
            analysis,
            fps=0,
            frame_num=1,
            total_frames=1,
            show_boxes=show_boxes,
            show_distance=show_distance
        )
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        from io import BytesIO
        return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

    return analysis


# Mount static web dashboard
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 Gasmask & Head Verification Server ishga tushmoqda...")
    print("📄 Swagger UI Dokumentatsiya: http://localhost:8000/docs")
    print("🌐 Veb Dashboard:              http://localhost:8000/")
    print("="*70 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
