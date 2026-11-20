"""
Video Processor Module
Processes video files frame-by-frame with tracking, annotations,
heads-up display (HUD), timeline analytics, and H.264 MP4 export.
"""

import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import cv2
import numpy as np

from .detector import GasmaskSystemDetector


class SimpleTracker:
    """Lightweight IoU + Centroid tracker for persistent Person IDs"""
    def __init__(self, max_lost_frames: int = 15, iou_thresh: float = 0.3):
        self.next_id = 1
        self.tracks = {} # id -> {'box': [x1,y1,x2,y2], 'lost': 0}
        self.max_lost_frames = max_lost_frames
        self.iou_thresh = iou_thresh

    def update(self, detected_boxes: List[List[float]]) -> List[int]:
        # Increment lost count
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['lost'] += 1
            if self.tracks[tid]['lost'] > self.max_lost_frames:
                del self.tracks[tid]

        if not detected_boxes:
            return []

        matched_ids = [-1] * len(detected_boxes)
        available_tids = list(self.tracks.keys())

        if available_tids:
            # Calculate IoU matrix
            for b_idx, box in enumerate(detected_boxes):
                best_iou = 0.0
                best_tid = None
                for tid in available_tids:
                    iou = self._iou(box, self.tracks[tid]['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = tid

                if best_iou >= self.iou_thresh:
                    matched_ids[b_idx] = best_tid
                    self.tracks[best_tid]['box'] = box
                    self.tracks[best_tid]['lost'] = 0
                    available_tids.remove(best_tid)

        # Assign new IDs for unmatched
        for b_idx in range(len(detected_boxes)):
            if matched_ids[b_idx] == -1:
                new_id = self.next_id
                self.next_id += 1
                matched_ids[b_idx] = new_id
                self.tracks[new_id] = {'box': detected_boxes[b_idx], 'lost': 0}

        return matched_ids

    def _iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


class VideoProcessor:
    def __init__(self, detector: GasmaskSystemDetector):
        self.detector = detector

    def process_video(
        self,
        input_video_path: str,
        output_video_path: str,
        conf: float = 0.25,
        imgsz: int = 1280,
        frame_stride: int = 1, # 1 means every frame, 2 means every 2nd frame
        show_boxes: bool = False,
        show_distance: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Processes an entire video file, writes annotated output video,
        and returns comprehensive timeline and compliance analytics.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Accurately detect FPS, total frames, and duration (fixes 1000 FPS bug on WebM screencasts)
        orig_fps, probed_frames, duration_sec = self.probe_video_metadata(input_video_path, cap)
        total_frames = probed_frames if probed_frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or (duration_sec > 0 and total_frames > duration_sec * orig_fps * 1.5):
            if duration_sec > 0 and orig_fps > 0:
                total_frames = int(duration_sec * orig_fps)

        # Output video FPS: must match real-world playback speed (e.g. 25-30 FPS)
        output_fps = max(5.0, min(60.0, orig_fps / frame_stride))

        # Intermediate raw video
        temp_out = output_video_path.replace(".mp4", "_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(temp_out, fourcc, output_fps, (width, height))

        tracker = SimpleTracker(max_lost_frames=20, iou_thresh=0.25)
        
        frame_idx = 0
        timeline_events = []
        unique_persons = set()
        violations_count_total = 0
        safe_count_total = 0

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_stride != 0:
                continue

            current_sec = round(frame_idx / orig_fps, 1)

            # 1. Run detection
            analysis = self.detector.detect_frame(
                frame,
                conf=conf,
                imgsz=imgsz
            )

            # 2. Update tracking
            person_boxes = [p['person_box'] for p in analysis['persons']]
            track_ids = tracker.update(person_boxes)
            for idx, p in enumerate(analysis['persons']):
                p['person_id'] = track_ids[idx]
                unique_persons.add(track_ids[idx])

            # 3. Annotate frame
            annotated_frame = self.draw_annotations(
                frame.copy(),
                analysis,
                fps=round(orig_fps, 1),
                frame_num=frame_idx,
                total_frames=total_frames,
                show_boxes=show_boxes,
                show_distance=show_distance
            )
            out_writer.write(annotated_frame)

            # 4. Record timeline summary
            sm = analysis['summary']
            if sm['total_persons'] > 0:
                violations_count_total += sm['violations_no_gasmask']
                safe_count_total += sm['wearing_gasmask']

                # Log notable events (every second or when violation occurs)
                if frame_idx % int(orig_fps) == 0 or sm['violations_no_gasmask'] > 0:
                    timeline_events.append({
                        "second": current_sec,
                        "frame": frame_idx,
                        "total_persons": sm['total_persons'],
                        "wearing_gasmask": sm['wearing_gasmask'],
                        "violations": sm['violations_no_gasmask'],
                        "compliance_percent": sm['compliance_rate_percent']
                    })

            if progress_callback and total_frames > 0:
                progress_callback(min(1.0, frame_idx / total_frames))

        cap.release()
        out_writer.release()

        proc_time = round(time.time() - start_time, 2)
        actual_fps = round(frame_idx / max(0.1, proc_time), 1)

        # 5. Convert to web-compatible H.264 MP4 using ffmpeg at exact real-time playback speed
        self.convert_to_h264(
            input_path=temp_out,
            output_path=output_video_path,
            target_fps=output_fps,
            original_audio_path=input_video_path
        )
        if os.path.exists(temp_out):
            os.remove(temp_out)

        # Overall compliance
        total_evaluations = safe_count_total + violations_count_total
        final_compliance = round((safe_count_total / max(1, total_evaluations)) * 100, 1)

        return {
            "status": "success",
            "video_metadata": {
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "processed_frames": frame_idx,
                "video_fps": round(orig_fps, 2),
                "duration_seconds": round(duration_sec, 2),
                "processing_time_seconds": proc_time,
                "processing_speed_fps": actual_fps
            },
            "analytics": {
                "unique_persons_tracked": len(unique_persons),
                "overall_compliance_percent": final_compliance,
                "total_safe_detections": safe_count_total,
                "total_violations_no_mask": violations_count_total,
                "rule_applied": "Oldi va yon tomondan gazmaska taqilgan bo'lsa 'gasmask', yuzi ochiq bo'lsa 'no-gasmask', orqa tomondan qaralganda hechnarsa chiqarilmaydi"
            },
            "timeline": timeline_events[:50] # Top 50 key timestamps
        }

    def draw_annotations(
        self,
        frame: np.ndarray,
        analysis: Dict[str, Any],
        fps: float,
        frame_num: int,
        total_frames: int,
        show_boxes: bool = False,
        show_distance: bool = False
    ) -> np.ndarray:
        """
        Renders HUD dashboard and person status labels.
        If show_boxes is False: Only displays clean, floating status badges (Maska kiygan / Maska kiymagan)
        above each person without intrusive bounding box rectangles.
        If show_boxes is True: Also renders full person and head bounding boxes.
        """
        h, w = frame.shape[:2]

        # 1. Top HUD Bar
        hud_h = 46
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, hud_h), (w, hud_h), (51, 65, 85), 2)

        sm = analysis['summary']
        txt_hud_left = f"ODAMLAR: {sm['total_persons']}  |  GASMASK: {sm['wearing_gasmask']}  |  NO-GASMASK: {sm['violations_no_gasmask']}"
        txt_hud_right = f"KADR: {frame_num}/{total_frames} ({fps} FPS)"

        cv2.putText(frame, txt_hud_left, (16, 29), cv2.FONT_HERSHEY_DUPLEX, 0.58, (248, 250, 252), 1, cv2.LINE_AA)
        cv2.putText(frame, txt_hud_right, (max(16, w - 300), 29), cv2.FONT_HERSHEY_DUPLEX, 0.52, (203, 213, 225), 1, cv2.LINE_AA)

        # 2. Draw Persons (Strictly ONE compact badge per person)
        placed_badges = []
        for p in analysis['persons']:
            st = p.get('status', 'NO_GASMASK_VIOLATION')
            is_head_visible = p.get('is_head_visible', True)

            # Agar orqaga qaragan bo'lsa yoki bosh ko'rinmasa -> FRONTDA HECHNIMA CHIQMAYDI
            if st in ("FACING_AWAY", "HEAD_NOT_VISIBLE") or not is_head_visible or not p.get('is_drawn', True):
                continue

            px1, py1, px2, py2 = map(int, p['person_box'])
            hx1, hy1, hx2, hy2 = map(int, p['head_box'])
            pid = p.get('person_id', 0)
            dist_val = p.get('distance_meters', 0.0)

            # Ixcham yorliqlar: gasmask (Yashil) yoki no-gasmask (Qizil)
            if st == "WEARING_GASMASK":
                tag_bg = (16, 185, 129) # Emerald Green (BGR)
                base_text = "gasmask"
            else:
                tag_bg = (50, 50, 220) # Bright Red (BGR)
                base_text = "no-gasmask"

            if show_distance and dist_val > 0:
                badge_text = f"{base_text} (~{dist_val}m)"
            else:
                badge_text = base_text

            if show_boxes:
                # Debug / Advanced mode with bounding boxes
                cv2.rectangle(frame, (px1, py1), (px2, py2), tag_bg, 2)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 255), 1)

                full_tag = f"ID #{pid} | {badge_text}"
                (tw, th_text), _ = cv2.getTextSize(full_tag, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                tag_y1 = max(hud_h + 5, py1 - 24)
                tag_y2 = tag_y1 + 22
                cv2.rectangle(frame, (px1, tag_y1), (px1 + tw + 12, tag_y2), tag_bg, -1)
                cv2.putText(frame, full_tag, (px1 + 6, tag_y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Default Clean Mode: Bitta odam boshiga FAQAT bitta chiroyli ixcham badge!
                person_h = py2 - py1
                font_scale = 0.44 if person_h < 120 else 0.50
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
                pad_x = 7
                pad_y = 5
                bw = tw + pad_x * 2
                bh = th + pad_y * 2 + 2

                # Aniq bosh markaziga joylashtirish
                head_top = min(py1, hy1)
                cx = (hx1 + hx2) // 2
                bx1 = cx - bw // 2
                bx2 = bx1 + bw
                if bx1 < 6:
                    bx1 = 6
                    bx2 = bx1 + bw
                if bx2 > w - 6:
                    bx2 = w - 6
                    bx1 = bx2 - bw

                # Boshdan tepada bo'sh joy bo'lsa tepasiga, bo'lmasa ko'krak qismiga
                if head_top - bh - 6 > hud_h:
                    by2 = head_top - 6
                    by1 = by2 - bh
                    notch = True
                else:
                    by1 = max(hud_h + 4, head_top + 8)
                    by2 = by1 + bh
                    notch = False

                # Stagger vertically if overlapping with an already placed badge in the same frame
                for pb in placed_badges:
                    if not (bx2 < pb['bx1'] or bx1 > pb['bx2']):
                        if abs(by1 - pb['by1']) < bh + 4:
                            by1 = pb['by2'] + 4
                            by2 = by1 + bh
                            notch = False

                placed_badges.append({'bx1': bx1, 'bx2': bx2, 'by1': by1, 'by2': by2})

                # Draw small indicator notch pointing to person head
                if notch and by2 < py1:
                    tri = np.array([[cx - 4, by2], [cx + 4, by2], [cx, min(py1 - 1, by2 + 5)]], np.int32)
                    cv2.fillPoly(frame, [tri], tag_bg)

                # Draw filled badge with crisp white outline
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), tag_bg, -1)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    badge_text,
                    (bx1 + pad_x, by1 + pad_y + th),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # 3. Ground/Table Masks (Only shown if show_boxes=True)
        if show_boxes:
            for gm in analysis['ground_masks']:
                gx1, gy1, gx2, gy2 = map(int, gm['box'])
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (100, 116, 139), 2)
                label = "Yerda yotgan gasmask (inobatga olinmaydi)"
                cv2.putText(frame, label, (gx1, max(hud_h + 10, gy1 - 6)), cv2.FONT_HERSHEY_DUPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

        return frame

    @staticmethod
    def probe_video_metadata(path: str, cap: Optional[cv2.VideoCapture] = None) -> Tuple[float, int, float]:
        """
        Accurately extracts video FPS, total frames, and duration in seconds.
        Handles WebM / MKV screencasts where OpenCV falsely reports 1000 FPS.
        Returns: (fps, total_frames, duration_seconds)
        """
        fps = 25.0
        total_frames = 0
        duration = 0.0

        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames:format=duration",
                "-of", "json",
                path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                duration = float(data.get("format", {}).get("duration", 0.0))
                streams = data.get("streams", [])
                if streams:
                    st = streams[0]
                    # 1. Try avg_frame_rate
                    avg_fr = st.get("avg_frame_rate", "0/0")
                    if "/" in avg_fr and avg_fr != "0/0":
                        num, den = map(float, avg_fr.split("/"))
                        if den > 0:
                            cand = num / den
                            if 5.0 <= cand <= 120.0:
                                fps = cand

                    # 2. Try r_frame_rate if avg_fr not set
                    if fps == 25.0:
                        r_fr = st.get("r_frame_rate", "0/0")
                        if "/" in r_fr and r_fr != "0/0":
                            num, den = map(float, r_fr.split("/"))
                            if den > 0:
                                cand = num / den
                                if 5.0 <= cand <= 120.0:
                                    fps = cand

                    # 3. Check nb_frames in stream
                    nb_str = st.get("nb_frames")
                    if nb_str and nb_str.isdigit():
                        total_frames = int(nb_str)

            # If fps is still default or 1000, and duration is known (common in WebM screencasts)
            if (fps == 25.0 or fps >= 100.0) and duration > 0:
                cmd2 = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-count_packets",
                    "-show_entries", "stream=nb_read_packets",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    path
                ]
                res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=5)
                if res2.returncode == 0 and res2.stdout.strip().isdigit():
                    nb_p = int(res2.stdout.strip())
                    if nb_p > 0:
                        calc = nb_p / duration
                        if 5.0 <= calc <= 120.0:
                            fps = round(calc, 2)
                            total_frames = nb_p
        except Exception:
            pass

        # If ffprobe didn't get valid fps, try cap
        if fps == 25.0 and cap is not None:
            cv_fps = cap.get(cv2.CAP_PROP_FPS)
            if cv_fps and 5.0 <= cv_fps <= 120.0:
                fps = round(cv_fps, 2)

        # Calculate duration if missing
        if duration <= 0 and total_frames > 0 and fps > 0:
            duration = round(total_frames / fps, 2)

        return round(fps, 2), total_frames, round(duration, 2)

    @staticmethod
    def _get_ffmpeg_bin() -> str:
        import shutil
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def convert_to_h264(
        self,
        input_path: str,
        output_path: str,
        target_fps: float = 25.0,
        original_audio_path: Optional[str] = None
    ) -> None:
        """
        Uses ffmpeg to transcode video to standard H.264 (avc1) + yuv420p
        with explicit frame rate and synchronized audio so it plays at natural 1.0x real-life speed.
        """
        ffmpeg_bin = self._get_ffmpeg_bin()
        fps_str = str(round(target_fps, 2))
        cmd = [ffmpeg_bin, "-y", "-i", input_path]

        if original_audio_path and os.path.exists(original_audio_path):
            cmd.extend([
                "-i", original_audio_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-r", fps_str,
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                "-movflags", "+faststart",
                output_path
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-r", fps_str,
                "-movflags", "+faststart",
                output_path
            ])

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            if os.path.exists(input_path):
                import shutil
                shutil.copy2(input_path, output_path)
