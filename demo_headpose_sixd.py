# demo_headpose_sixd.py
import os
import cv2
import mediapipe as mp

from src.headpose_sixd import SixDRepONNX
from src.visualize_utils import draw_headpose_axes

# ===== CẤU HÌNH ĐƯỜNG DẪN =====
IMG_PATH   = "data_test/sample_frames/Vid1_frame_00015.jpg"  # hoặc Vid1_frame_00015.jpg
ONNX_PATH  = "models/SixDRepNet.onnx"   # đúng tên file .onnx của bạn
OUT_PATH   = "data_test/sample_frames/headpose_sixd_demo2.jpg"


def detect_face_bbox_mediapipe(img_bgr):
    """
    Dò khuôn mặt bằng MediaPipe Face Detection, trả về (x1,y1,x2,y2) hoặc None.
    """
    h, w = img_bgr.shape[:2]
    mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    )

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = mp_fd.process(rgb)
    if not res.detections:
        return None

    det = res.detections[0]
    bb = det.location_data.relative_bounding_box

    x1 = int(bb.xmin * w)
    y1 = int(bb.ymin * h)
    x2 = int((bb.xmin + bb.width) * w)
    y2 = int((bb.ymin + bb.height) * h)

    # clamp
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def main():
    if not os.path.exists(IMG_PATH):
        print(f"❌ Không tìm thấy ảnh: {IMG_PATH}")
        return
    if not os.path.exists(ONNX_PATH):
        print(f"❌ Không tìm thấy model SixDRepNet: {ONNX_PATH}")
        return

    # 1) Đọc ảnh
    img = cv2.imread(IMG_PATH)
    if img is None:
        print("❌ cv2.imread trả về None, check lại đường dẫn ảnh.")
        return

    # 2) Dò khuôn mặt để lấy bbox
    bbox = detect_face_bbox_mediapipe(img)
    if bbox is None:
        print("❌ Không phát hiện được khuôn mặt trong ảnh.")
        return

    x1, y1, x2, y2 = bbox
    face = img[y1:y2, x1:x2]

    # 3) Load SixDRepNet
    print("🔹 Đang load SixDRepNet ONNX...")
    headpose_model = SixDRepONNX(path=ONNX_PATH)

    # 4) Suy luận head pose
    ypr = headpose_model.infer(face)
    if ypr is None:
        print("❌ Model trả về None, không suy luận được tư thế đầu.")
        return

    yaw, pitch, roll = ypr
    print(f"✅ Head pose: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")

    # 5) Vẽ bbox + trục head pose
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # tâm khuôn mặt để đặt trục
    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2

    # vẽ 3 trục X/Y/Z
    vis = draw_headpose_axes(vis, (cx, cy), yaw, pitch, roll, length=80)

    # vẽ text góc
    text = f"yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}"
    cv2.putText(vis, text, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # 6) Lưu ảnh kết quả
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, vis)
    print(f"🖼  Đã lưu ảnh demo head pose tại: {OUT_PATH}")


if __name__ == "__main__":
    main()
