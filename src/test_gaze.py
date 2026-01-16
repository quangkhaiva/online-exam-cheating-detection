# src/test_gaze.py
from gaze_l2cs import L2CS
import cv2
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
onnx_path = BASE / "models" / "l2cs_gaze.onnx"
img_path  = BASE / "data_test" / "sample_frames" / "Vid1_frame_00015.jpg"

print("✅ Đang sử dụng mô hình:", onnx_path)
print("🖼️ Ảnh test:", img_path)

gaze = L2CS(str(onnx_path))
face = cv2.imread(str(img_path))

if face is None:
    print("⚠️ Không đọc được ảnh test")
else:
    res = gaze.infer(face)
    if res is not None:
        yaw, pitch = res
        print(f"🎯 Gaze direction: yaw={yaw:.2f}°, pitch={pitch:.2f}°")
    else:
        print("❌ Không phát hiện được gaze")
