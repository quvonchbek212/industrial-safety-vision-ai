"""
Head Verifier Module
Anatomical and spatial algorithm to verify whether a gasmask is properly
worn on a person's head vs held in hand or lying on the ground.
"""

from typing import List, Dict, Any, Tuple, Optional
import math


class HeadVerifier:
    def __init__(
        self,
        head_top_ratio: float = 0.04,    # Extra headroom above person bbox (for helmet/hood)
        head_bottom_ratio: float = 0.22, # Anatomical cutoff at shoulder level (20-22% of standing person height)
        head_side_margin: float = 0.08,  # Lateral margin for head width
        min_overlap_ratio: float = 0.15, # Minimum intersection over mask area to be on head
        focal_length_factor: float = 0.95 # Approximate focal length factor for distance estimation
    ):
        self.head_top_ratio = head_top_ratio
        self.head_bottom_ratio = head_bottom_ratio
        self.head_side_margin = head_side_margin
        self.min_overlap_ratio = min_overlap_ratio
        self.focal_length_factor = focal_length_factor

    def compute_iou(self, box_a: List[float], box_b: List[float]) -> float:
        """Calculates standard Intersection-over-Union (IoU)."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter + 1e-6)

    def compute_head_box(
        self,
        person_box: List[float],
        kpts: Optional[List[List[float]]] = None
    ) -> Tuple[List[float], bool]:
        """
        Calculates the anatomical head region coordinates [hx1, hy1, hx2, hy2].
        STRICT ANATOMICAL RULE:
        Only includes region from top of head down to the shoulders (yelkadan tepasi).
        Never extends below the shoulder line into chest, arms, torso, or hands!
        Returns: (head_box, is_head_valid)
        """
        px1, py1, px2, py2 = person_box
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)

        shoulder_y = None
        is_head_valid = True

        # Check pose keypoints: nose(0), eyes(1,2), ears(3,4), shoulders(5,6)
        if kpts is not None and len(kpts) >= 7:
            face_confs = [kpts[i][2] for i in range(min(5, len(kpts)))]
            max_face = max(face_confs) if face_confs else 0.0

            shoulders = []
            if kpts[5][2] > 0.25:
                shoulders.append(kpts[5][1])
            if kpts[6][2] > 0.25:
                shoulders.append(kpts[6][1])

            if shoulders:
                # Shoulders provide the exact anatomical cutoff level
                # Add slight margin of 2% ph to reach the base of neck / collar
                shoulder_y = max(shoulders) + (ph * 0.02)

            # If no face features detected and no shoulders detected,
            # and person box starts near top border -> Headless / truncated body!
            if max_face < 0.20 and len(shoulders) == 0:
                if py1 <= 30 or (ph / pw) < 1.0:
                    is_head_valid = False

        if shoulder_y is None:
            # Fallback anatomical ratio: adult head+neck is ~20-24% of height
            ratio = 0.20 if (ph / pw) > 1.8 else 0.24
            shoulder_y = py1 + ph * ratio

            if py1 <= 25 and ph > 200:
                is_head_valid = False

        hy1 = max(0.0, py1 - ph * self.head_top_ratio)
        hy2 = shoulder_y

        if hy2 <= hy1:
            hy2 = hy1 + ph * 0.20

        # Lateral margins: head width is approximately 70-80% of shoulder span
        hx1 = max(0.0, px1 + pw * self.head_side_margin)
        hx2 = px2 - pw * self.head_side_margin
        if hx2 <= hx1:
            hx1, hx2 = px1, px2

        head_box = [round(hx1, 1), round(hy1, 1), round(hx2, 1), round(hy2, 1)]
        return head_box, is_head_valid

    def estimate_distance(self, person_box: List[float], frame_height: int) -> Tuple[float, str]:
        """
        Estimates the distance in meters to a person based on bounding box height
        assuming average adult human height ~ 1.75m.
        Returns: (distance_meters, category_label)
        """
        _, py1, _, py2 = person_box
        box_h_pixels = max(1.0, py2 - py1)

        focal_px = self.focal_length_factor * frame_height
        distance_m = (focal_px * 1.75) / box_h_pixels

        if distance_m < 5.0:
            category = "Yaqin (<5m)"
        elif distance_m < 15.0:
            category = "O'rta (5-15m)"
        else:
            category = "Uzoq (>15m)"

        return round(distance_m, 1), category

    def deduplicate_persons(self, persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Eliminates duplicate / overlapping person bounding boxes for the SAME person.
        Ensures strictly ONE detection per human in the frame.
        """
        if len(persons) <= 1:
            return persons

        # Sort by confidence descending
        sorted_persons = sorted(persons, key=lambda p: p.get('conf', 0.0), reverse=True)
        unique_persons = []

        for cand in sorted_persons:
            c_box = cand['box']
            c_kpts = cand.get('kpts')
            c_head, c_valid = self.compute_head_box(c_box, c_kpts)
            cand['head_box'] = c_head
            cand['is_head_valid'] = c_valid

            is_duplicate = False
            for kept in unique_persons:
                k_box = kept['box']
                k_head = kept['head_box']

                # 1. Body IoU check (>0.60 means strong overlap of same detection)
                if self.compute_iou(c_box, k_box) > 0.60:
                    is_duplicate = True
                    break

                # 2. Head IoU check (>0.55 means heads strongly overlap)
                if self.compute_iou(c_head, k_head) > 0.55:
                    is_duplicate = True
                    break

                # 3. Head center distance check (only if IoU is also significant)
                c_cx = (c_head[0] + c_head[2]) / 2.0
                c_cy = (c_head[1] + c_head[3]) / 2.0
                k_cx = (k_head[0] + k_head[2]) / 2.0
                k_cy = (k_head[1] + k_head[3]) / 2.0
                hw = max(1.0, k_head[2] - k_head[0])
                hh = max(1.0, k_head[3] - k_head[1])

                if abs(c_cx - k_cx) < hw * 0.35 and abs(c_cy - k_cy) < hh * 0.35 and self.compute_iou(c_head, k_head) > 0.45:
                    is_duplicate = True
                    break

                # 4. Pose keypoints nose proximity
                if c_kpts and kept.get('kpts'):
                    cn = c_kpts[0]
                    kn = kept['kpts'][0]
                    if cn[2] > 0.25 and kn[2] > 0.25:
                        if math.hypot(cn[0] - kn[0], cn[1] - kn[1]) < hw * 0.5:
                            is_duplicate = True
                            break

            if not is_duplicate:
                unique_persons.append(cand)

        return unique_persons

    def analyze_face_and_orientation(
        self,
        person_box: List[float],
        kpts: Optional[List[List[float]]],
        head_box: List[float]
    ) -> Dict[str, Any]:
        """
        Anatomical orientation & face visibility classifier.
        Distinguishes:
        - FRONT: Face facing camera (both eyes or nose+eye visible)
        - SIDE: Profile view (one eye or nose+one ear visible)
        - BACK: Seen from behind / facing away (no face features, rear shoulder/ear alignment)
        - UNCERTAIN: Inconclusive keypoints
        """
        if kpts is None or len(kpts) < 7:
            return {
                "is_face_visible": False,
                "is_back_view": False,
                "orientation": "UNCERTAIN",
                "face_conf": 0.0,
                "details": "Nuqtalar mavjud emas"
            }

        nose = kpts[0]
        leye = kpts[1]
        reye = kpts[2]
        lear = kpts[3]
        rear = kpts[4]
        lsh = kpts[5]
        rsh = kpts[6]

        c_nose = float(nose[2]) if len(nose) > 2 else 0.0
        c_leye = float(leye[2]) if len(leye) > 2 else 0.0
        c_reye = float(reye[2]) if len(reye) > 2 else 0.0
        c_lear = float(lear[2]) if len(lear) > 2 else 0.0
        c_rear = float(rear[2]) if len(rear) > 2 else 0.0
        c_lsh = float(lsh[2]) if len(lsh) > 2 else 0.0
        c_rsh = float(rsh[2]) if len(rsh) > 2 else 0.0

        face_conf = max(c_nose, c_leye, c_reye)
        has_nose = c_nose >= 0.20
        has_leye = c_leye >= 0.20
        has_reye = c_reye >= 0.20
        has_lear = c_lear >= 0.20
        has_rear = c_rear >= 0.20
        has_lsh = c_lsh >= 0.20
        has_rsh = c_rsh >= 0.20

        face_kpts_count = sum([has_nose, has_leye, has_reye])

        # Anatomical shoulder orientation:
        # COCO 5 = Left Shoulder, 6 = Right Shoulder
        # Facing camera (FRONT): Right shoulder has smaller X than Left shoulder (rsh[0] < lsh[0])
        # Facing away (BACK): Left shoulder has smaller X than Right shoulder (lsh[0] < rsh[0])
        shoulder_orientation = None
        if has_lsh and has_rsh and abs(lsh[0] - rsh[0]) > 8:
            shoulder_orientation = "FRONT" if rsh[0] < lsh[0] else "BACK"

        # Anatomical ear orientation:
        # COCO 3 = Left Ear, 4 = Right Ear
        # Facing camera: Right ear has smaller X (rear[0] < lear[0])
        # Facing away: Left ear has smaller X (lear[0] < rear[0])
        ear_orientation = None
        if has_lear and has_rear and abs(lear[0] - rear[0]) > 6:
            ear_orientation = "FRONT" if rear[0] < lear[0] else "BACK"

        # 1. FRONT VIEW:
        if face_kpts_count >= 2 or (has_nose and has_lear and has_rear) or (shoulder_orientation == "FRONT" and (face_kpts_count >= 1 or face_conf >= 0.18)):
            return {
                "is_face_visible": True,
                "is_back_view": False,
                "orientation": "FRONT",
                "face_conf": round(face_conf, 3),
                "details": "Oldi tomondan (yuz ko'rinib turibdi)"
            }

        # 2. SIDE VIEW (Profile):
        if (has_nose and (has_lear ^ has_rear)) or (has_leye ^ has_reye) or (face_conf >= 0.20 and (has_lear or has_rear)):
            return {
                "is_face_visible": True,
                "is_back_view": False,
                "orientation": "SIDE",
                "face_conf": round(face_conf, 3),
                "details": "Yon tomondan (yuz profili ko'rinib turibdi)"
            }

        # 3. BACK VIEW (Rear view):
        # Strict requirement: ONLY mark BACK if shoulders or ears explicitly prove facing away!
        if shoulder_orientation == "BACK" or ear_orientation == "BACK":
            if face_kpts_count == 0:
                return {
                    "is_face_visible": False,
                    "is_back_view": True,
                    "orientation": "BACK",
                    "face_conf": round(face_conf, 3),
                    "details": "Orqa tomondan (yelka/quloq geometriyasi orqaga qaragan)"
                }

        # If shoulders are facing camera (FRONT) or neutral:
        if shoulder_orientation == "FRONT":
            return {
                "is_face_visible": True,
                "is_back_view": False,
                "orientation": "FRONT",
                "face_conf": round(face_conf, 3),
                "details": "Oldi tomondan (yelka kameraga qaragan)"
            }

        # Distant or normal view: Default to visible facing unless strictly proven back!
        return {
            "is_face_visible": True,
            "is_back_view": False,
            "orientation": "FRONT",
            "face_conf": round(face_conf, 3),
            "details": "Oldi / masofaviy ko'rinish"
        }

    def associate_persons_and_masks(
        self,
        persons: List[Dict[str, Any]],
        gasmasks: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> Dict[str, Any]:
        """
        Associates detected people and detected gasmasks/no-gasmasks using
        strict 1-to-1 bipartite assignment to avoid false gasmask inheritance.
        """
        # Step 1: Eliminate duplicate person detections
        deduped_persons = self.deduplicate_persons(persons)

        # Precompute head geometries and face centers for all persons
        person_info = []
        for p_idx, p in enumerate(deduped_persons):
            p_box = p['box']
            p_kpts = p.get('kpts')
            if p.get('head_box'):
                head_box = p['head_box']
                is_head_valid = p.get('is_head_valid', True)
            else:
                head_box, is_head_valid = self.compute_head_box(p_box, p_kpts)
            hx1, hy1, hx2, hy2 = head_box
            hw = max(1.0, hx2 - hx1)
            hh = max(1.0, hy2 - hy1)

            # Anatomical face center (nose/eyes or head center)
            fc_x, fc_y = None, None
            if p_kpts and len(p_kpts) >= 7:
                if p_kpts[0][2] > 0.18:
                    fc_x, fc_y = p_kpts[0][0], p_kpts[0][1]
                elif p_kpts[1][2] > 0.18 or p_kpts[2][2] > 0.18:
                    eyes = [p_kpts[i] for i in (1, 2) if p_kpts[i][2] > 0.18]
                    fc_x = sum(e[0] for e in eyes) / len(eyes)
                    fc_y = sum(e[1] for e in eyes) / len(eyes)

            if fc_x is None:
                fc_x = (hx1 + hx2) / 2.0
                fc_y = (hy1 + hy2) / 2.0

            dist_m, dist_cat = self.estimate_distance(p_box, frame_height)
            face_analysis = self.analyze_face_and_orientation(p_box, p_kpts, head_box)

            person_info.append({
                'person': p,
                'head_box': head_box,
                'is_head_valid': is_head_valid,
                'face_center': (fc_x, fc_y),
                'hw': hw,
                'hh': hh,
                'dist_m': dist_m,
                'dist_cat': dist_cat,
                'face_analysis': face_analysis
            })

        # Step 2: Global 1-to-1 Bipartite Matching
        candidate_matches = []
        for p_idx, pi in enumerate(person_info):
            hx1, hy1, hx2, hy2 = pi['head_box']
            hw, hh = pi['hw'], pi['hh']
            fc_x, fc_y = pi['face_center']

            for m_idx, m in enumerate(gasmasks):
                mb = m['box']
                mc_x = (mb[0] + mb[2]) / 2.0
                mc_y = (mb[1] + mb[3]) / 2.0

                # Strict Rule: Must be shoulder-up (never down in hands, waist, or chest!)
                if mc_y > hy2 + (hh * 0.12):
                    continue
                # Horizontal containment with head
                if not (hx1 - hw * 0.35 <= mc_x <= hx2 + hw * 0.35):
                    continue

                iou = self.compute_iou(mb, pi['head_box'])
                dist_to_fc = math.hypot(mc_x - fc_x, mc_y - fc_y)
                norm_dist = dist_to_fc / hh

                # Proximity check
                if norm_dist > 1.35 and iou < 0.05:
                    continue

                # Match quality score (higher is stronger link):
                # 1. IoU with head
                # 2. Closeness to anatomical face center (nose/eyes)
                # 3. Model detection confidence
                score = (iou * 2.5) + max(0.0, 1.8 - norm_dist * 1.5) + m.get('conf', 0.5)

                # Bare face directly centered on nose is definitive evidence of no-gasmask!
                if m.get('cls_id', 0) == 1 and norm_dist < 0.45:
                    score += 0.8

                candidate_matches.append({
                    'p_idx': p_idx,
                    'm_idx': m_idx,
                    'score': score,
                    'cls_id': m.get('cls_id', 0)
                })

        # Sort candidate matches by score descending (greedy optimal assignment)
        candidate_matches.sort(key=lambda x: x['score'], reverse=True)

        assigned_persons = {}
        assigned_mask_indices = set()

        for match in candidate_matches:
            pi_idx = match['p_idx']
            mi_idx = match['m_idx']
            if pi_idx not in assigned_persons and mi_idx not in assigned_mask_indices:
                assigned_persons[pi_idx] = (mi_idx, gasmasks[mi_idx])
                assigned_mask_indices.add(mi_idx)

        # Step 3: Resolve Final Status for each person
        person_results = []
        for p_idx, pi in enumerate(person_info):
            p = pi['person']
            p_box = p['box']
            p_conf = p.get('conf', 1.0)
            p_id = p.get('track_id', p_idx + 1)
            head_box = pi['head_box']
            is_head_valid = pi['is_head_valid']
            dist_m = pi['dist_m']
            dist_cat = pi['dist_cat']
            face_analysis = pi['face_analysis']
            is_face_visible = face_analysis['is_face_visible']
            is_back_view = face_analysis['is_back_view']
            orientation = face_analysis['orientation']

            assigned_pair = assigned_persons.get(p_idx)
            matched_detection = assigned_pair[1] if assigned_pair else None

            if matched_detection is not None:
                if matched_detection.get('cls_id', 0) == 0:
                    # 1. Boshda/yuzda to'g'ri taqilgan gazmaska
                    status = "WEARING_GASMASK"
                    status_label = "gasmask"
                    color = "#10b981"  # Emerald Green
                    is_safe = True
                    is_head_visible = True
                    is_drawn = True
                else:
                    # 2. Boshda ochiq yuz (no-gasmask) aniqlandi
                    status = "NO_GASMASK_VIOLATION"
                    status_label = "no-gasmask"
                    color = "#ef4444"  # Bright Red
                    is_safe = False
                    is_head_visible = True
                    is_drawn = True
            else:
                # Mask detektori (exp-3.pt) bu boshda hech narsa topmadi
                if is_face_visible or (not is_back_view and is_head_valid):
                    # Odamning yuzi ochiq (oldi yoki yon tomondan) va maska yo'q!
                    status = "NO_GASMASK_VIOLATION"
                    status_label = "no-gasmask"
                    color = "#ef4444"  # Bright Red
                    is_safe = False
                    is_head_visible = True
                    is_drawn = True
                elif is_back_view:
                    # Odam orqa tomondan ko'rindi (orqaga qaragan) -> Hechnima chiqarmaydi!
                    status = "FACING_AWAY"
                    status_label = "Orqa tomondan"
                    color = "#64748b"  # Slate Gray
                    is_safe = True     # Qoidabuzarlik hisoblanmaydi
                    is_head_visible = False
                    is_drawn = False   # Ekranga chizilmaydi
                else:
                    # Boshsiz / kesilgan tana
                    status = "HEAD_NOT_VISIBLE"
                    status_label = "Boshsiz tana"
                    color = "#64748b"
                    is_safe = True
                    is_head_visible = False
                    is_drawn = False

            person_results.append({
                "person_id": p_id,
                "person_box": [round(c, 1) for c in p_box],
                "head_box": [round(c, 1) for c in head_box],
                "confidence": round(p_conf, 3),
                "distance_meters": dist_m,
                "distance_category": dist_cat,
                "status": status,
                "status_label": status_label,
                "orientation": orientation,
                "is_face_visible": is_face_visible,
                "is_back_view": is_back_view,
                "is_safe": is_safe,
                "is_drawn": is_drawn,
                "color": color,
                "associated_detection": matched_detection
            })

        # Process ground / hand / unassigned masks
        ground_masks = []
        for m_idx, m in enumerate(gasmasks):
            if m_idx not in assigned_mask_indices and m.get('cls_id', 0) == 0:
                ground_masks.append({
                    "box": [round(c, 1) for c in m['box']],
                    "confidence": round(m.get('conf', 1.0), 3),
                    "status": "ON_GROUND_OR_HELD",
                    "status_label": "Boshda emas (inobatga olinmaydi)"
                })

        # Summary statistics
        total_persons = len(person_results)
        wearing_count = sum(1 for p in person_results if p['status'] == "WEARING_GASMASK")
        violation_count = sum(1 for p in person_results if p['status'] == "NO_GASMASK_VIOLATION")
        facing_away_count = sum(1 for p in person_results if p['status'] == "FACING_AWAY")
        headless_count = sum(1 for p in person_results if p['status'] == "HEAD_NOT_VISIBLE")
        evaluated_persons = wearing_count + violation_count
        compliance_pct = round((wearing_count / max(1, evaluated_persons)) * 100, 1) if evaluated_persons > 0 else (100.0 if wearing_count > 0 else 0.0)

        return {
            "persons": person_results,
            "ground_masks": ground_masks,
            "summary": {
                "total_persons": total_persons,
                "evaluated_persons": evaluated_persons,
                "wearing_gasmask": wearing_count,
                "violations_no_gasmask": violation_count,
                "facing_away_ignored": facing_away_count,
                "headless_ignored": headless_count,
                "ground_masks_ignored": len(ground_masks),
                "compliance_rate_percent": compliance_pct
            }
        }
