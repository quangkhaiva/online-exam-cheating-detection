"""
Crop vùng mặt từ bbox 'student' + (tuỳ chọn) refine bằng MediaPipe.
Cung cấp các hàm:
- crop_face_from_person(frame, xyxy, margin=0.25)
- refine_face_with_mediapipe(frame, xyxy)
- crop_head_top_half(frame, xyxy, margin=0.15)   # chỉ lấy nửa trên (đầu & vai)
- to_square(face_bgr)                             # pad về ảnh vuông (giữ tỷ lệ)

Tất cả hàm đều trả:
    (roi_bgr, (x1, y1, x2, y2))  hoặc (None, None) nếu không khả dụng.
"""

from __future__ import annotations
import cv2
import mediapipe as mp
from typing import Optional, Tuple

# Tạo 1 instance MediaPipe (nhẹ, dùng lại giữa các frame)
mp_fd = mp.solutions.face_detection.FaceDetection(
    model_selection=0,             # 0: near, 1: far
    min_detection_confidence=0.6
)

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def crop_face_from_person(frame, xyxy, margin: float = 0.25):
    """
    Cắt vùng quanh bbox 'student' có nới margin (đơn giản, chưa refine khuôn mặt).
    """
    if frame is None:
        return None, None
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None, None

    mx, my = int(w * margin), int(h * margin)
    nx1 = clamp(x1 - mx, 0, W - 1)
    ny1 = clamp(y1 - my, 0, H - 1)
    nx2 = clamp(x2 + mx, 0, W - 1)
    ny2 = clamp(y2 + my, 0, H - 1)
    if nx2 <= nx1 or ny2 <= ny1:
        return None, None

    face = frame[ny1:ny2, nx1:nx2]
    if face is None or face.size == 0:
        return None, None
    return face, (nx1, ny1, nx2, ny2)

def refine_face_with_mediapipe(frame, xyxy):
    """
    Dò tìm khuôn mặt thật sự bên trong bbox 'student' bằng MediaPipe.
    Hữu ích khi bbox 'student' rộng (gồm cả thân người).
    """
    if frame is None:
        return None, None
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = clamp(x1, 0, W - 1), clamp(y1, 0, H - 1)
    x2, y2 = clamp(x2, 0, W - 1), clamp(y2, 0, H - 1)
    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = frame[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None, None

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    res = mp_fd.process(rgb)
    if not res.detections:
        return None, None

    bb = res.detections[0].location_data.relative_bounding_box
    # Tính bbox mặt trong toạ độ gốc frame
    fx1 = x1 + int(bb.xmin * (x2 - x1))
    fy1 = y1 + int(bb.ymin * (y2 - y1))
    fx2 = fx1 + int(bb.width  * (x2 - x1))
    fy2 = fy1 + int(bb.height * (y2 - y1))

    fx1 = clamp(fx1, 0, W - 1)
    fy1 = clamp(fy1, 0, H - 1)
    fx2 = clamp(fx2, 0, W - 1)
    fy2 = clamp(fy2, 0, H - 1)
    if fx2 <= fx1 or fy2 <= fy1:
        return None, None

    face = frame[fy1:fy2, fx1:fx2]
    if face is None or face.size == 0:
        return None, None
    return face, (fx1, fy1, fx2, fy2)

def crop_head_top_half(frame, xyxy, margin: float = 0.15):
    """
    Lấy nửa trên của bbox 'student' (đầu & vai) để ổn định gaze/headpose.
    """
    if frame is None:
        return None, None
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None, None

    # Nới nhẹ trước, rồi cắt nửa trên
    mx, my = int(w * margin), int(h * margin)
    nx1 = clamp(x1 - mx, 0, W - 1)
    ny1 = clamp(y1 - my, 0, H - 1)
    nx2 = clamp(x2 + mx, 0, W - 1)
    ny2 = clamp(y2 + my, 0, H - 1)
    if nx2 <= nx1 or ny2 <= ny1:
        return None, None

    mid_y = ny1 + (ny2 - ny1) // 2
    fx1, fy1, fx2, fy2 = nx1, ny1, nx2, mid_y
    if fx2 <= fx1 or fy2 <= fy1:
        return None, None

    face = frame[fy1:fy2, fx1:fx2]
    if face is None or face.size == 0:
        return None, None
    return face, (fx1, fy1, fx2, fy2)

def to_square(face_bgr):
    """
    Pad ảnh về vuông (giữ tỷ lệ) để model ổn định hơn.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None
    h, w = face_bgr.shape[:2]
    side = max(h, w)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    sq = cv2.copyMakeBorder(face_bgr,
                            pad_y, side - h - pad_y,
                            pad_x, side - w - pad_x,
                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return sq
