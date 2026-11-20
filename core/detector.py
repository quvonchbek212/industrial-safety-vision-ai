"""
Detector Module
Dual-inference pipeline:
1. Person detector (YOLOv8 COCO person class)
2. Gasmask detector (Custom fine-tuned exp-3.pt)
Runs high-resolution inference (imgsz=1280 by default) for distant subjects.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from ultralytics import YOLO

from .head_verifier import HeadVerifier


class GasmaskSystemDetector:
    def __init__(
        self,
        gasmask_model_path: Optional[str] = None,
        person_model_path: Optional[str] = None,
        device: Optional[str] = None,
        default_imgsz: int = 1280,
        default_conf: float = 0.25
    ):
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.default_imgsz = default_imgsz
        self.default_conf = default_conf
        self.verifier = HeadVerifier()

        print(f"[Detector] Initializing models on device: {self.device}...")
        
        # 1. Load Gasmask Model
        resolved_gasmask_path = None
        if gasmask_model_path and Path(gasmask_model_path).exists():
            resolved_gasmask_path = gasmask_model_path
        else:
            candidates = [
                models_dir / "exp-3.pt",
                models_dir / "expert.pt",
                base_dir / "exp-3.pt",
                base_dir / "expert.pt",
                Path("models/exp-3.pt"),
                Path("models/expert.pt")
            ]
            for cand in candidates:
                if cand.exists():
                    resolved_gasmask_path = str(cand)
                    break

        if not resolved_gasmask_path:
            resolved_gasmask_path = str(models_dir / "exp-3.pt")

        print(f"[Detector] Loading Gasmask model: {resolved_gasmask_path}")
        self.gasmask_model = YOLO(resolved_gasmask_path)
        
        # 2. Load Person Detector (Pose model provides both bounding boxes and head keypoints)
        resolved_person_model = None
        if person_model_path and Path(person_model_path).exists():
            resolved_person_model = person_model_path
        else:
            pose_candidates = [
                models_dir / "yolov8n-pose.pt",
                base_dir / "yolov8n-pose.pt",
                Path("models/yolov8n-pose.pt"),
                Path("yolov8n-pose.pt"),
                models_dir / "yolov8n.pt",
                base_dir / "yolov8n.pt",
                Path("yolov8n.pt")
            ]
            for cand in pose_candidates:
                if cand.exists():
                    resolved_person_model = str(cand)
                    break

        if not resolved_person_model:
            resolved_person_model = "yolov8n-pose.pt"

        print(f"[Detector] Loading Person/Pose model: {resolved_person_model}")
        self.person_model = YOLO(resolved_person_model)

        print("[Detector] Models successfully loaded and ready for high-resolution inference!")

    def detect_frame(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        require_head_only: bool = True
    ) -> Dict[str, Any]:
        """
        Runs detection on a single video frame or image.
        Returns combined analysis:
        - persons detected with head boxes and wear status
        - gasmasks on head, in hand, or on ground
        - distances and compliance metrics
        """
        h, w = frame.shape[:2]
        conf_val = conf if conf is not None else self.default_conf
        imgsz_val = imgsz if imgsz is not None else self.default_imgsz

        # 1. Detect Persons with Anatomical Keypoints
        # Sensitive threshold for distant persons (10-30m)
        person_conf_thresh = max(0.10, min(0.20, conf_val * 0.6))
        person_results = self.person_model(
            frame,
            conf=person_conf_thresh,
            imgsz=imgsz_val,
            device=self.device,
            verbose=False
        )

        detected_persons = []
        for r in person_results:
            has_kpts = hasattr(r, 'keypoints') and r.keypoints is not None and len(r.keypoints.data) > 0
            for idx, b in enumerate(r.boxes):
                if int(b.cls[0].cpu()) != 0:
                    continue
                xyxy = b.xyxy[0].cpu().tolist()
                score = float(b.conf[0].cpu())
                kpts = r.keypoints.data[idx].cpu().tolist() if has_kpts and idx < len(r.keypoints.data) else None
                detected_persons.append({
                    "box": xyxy,
                    "conf": score,
                    "kpts": kpts
                })

        # 2. Detect Gasmask & No-gasmask (exp-3.pt)
        # Sensitive threshold for distant masks
        mask_conf_thresh = max(0.12, min(0.22, conf_val * 0.75))
        mask_results = self.gasmask_model(
            frame,
            conf=mask_conf_thresh,
            imgsz=imgsz_val,
            device=self.device,
            verbose=False
        )

        detected_masks = []
        for r in mask_results:
            for b in r.boxes:
                cls_id = int(b.cls[0].cpu())
                score = float(b.conf[0].cpu())
                xyxy = b.xyxy[0].cpu().tolist()
                detected_masks.append({
                    "box": xyxy,
                    "cls_id": cls_id,
                    "name": self.gasmask_model.names.get(cls_id, f"class_{cls_id}"),
                    "conf": score
                })

        # 3. Associate and Verify (Head vs Hand vs Ground)
        analysis = self.verifier.associate_persons_and_masks(
            persons=detected_persons,
            gasmasks=detected_masks,
            frame_width=w,
            frame_height=h
        )

        # Raw counts for reference
        analysis["raw_detections"] = {
            "total_persons_raw": len(detected_persons),
            "total_gasmask_boxes_raw": sum(1 for m in detected_masks if m['cls_id'] == 0),
            "total_nogasmask_boxes_raw": sum(1 for m in detected_masks if m['cls_id'] == 1)
        }

        return analysis
