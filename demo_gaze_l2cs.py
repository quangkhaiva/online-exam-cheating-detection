# demo_gaze_image.py
import os, sys, cv2
import numpy as np

# Cho phép import từ src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from gaze_l2cs import L2CS, pick_providers
from facemesh_mediapipe import FaceMeshExtractor
from visualize_utils import draw_gaze_vector

# ===== CHỌN ẢNH TRONG FOLDER data_test/sample_frames =====
IMG_DIR = "data_test/sample_frames"

def choose_image():
    files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png"))]
    if not files:
        print("❌ Không tìm thấy ảnh nào trong data_test/sample_frames/")
        exit()

    print("🔍 Danh sách ảnh có thể chọn:")
    for i, f in enumerate(files):
        print(f"   [{i}] {f}")

    idx = int(input("\n👉 Nhập số thứ tự ảnh muốn demo: "))
    return os.path.join(IMG_DIR, files[idx])


def main():
    img_path = choose_image()
    print(f"\n📌 Đang xử lý ảnh: {img_path}")

    # Load ảnh
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Không mở được ảnh!")
        return
    h, w = img.shape[:2]

    # Load model
    print("🔄 Load L2CS-Net ONNX...")
    gaze_model = L2CS("models/l2cs_gaze.onnx", providers=pick_providers())

    print("🔄 Khởi tạo MediaPipe FaceMesh...")
    fm = FaceMeshExtractor(max_faces=1, refine_landmarks=True)

    # Face landmarks
    lm2d, _ = fm.extract(img)
    if lm2d is None:
        print("❌ Không tìm thấy khuôn mặt trong ảnh!")
        return

    # BBox quanh mặt
    x_min = int(np.min(lm2d[:, 0]))
    x_max = int(np.max(lm2d[:, 0]))
    y_min = int(np.min(lm2d[:, 1]))
    y_max = int(np.max(lm2d[:, 1]))

    pad_x = int(0.15 * (x_max - x_min))
    pad_y = int(0.25 * (y_max - y_min))

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w - 1, x_max + pad_x)
    y2 = min(h - 1, y_max + pad_y)

    face = img[y1:y2, x1:x2].copy()
    if face.size == 0:
        print("❌ Không crop được mặt!")
        return

    # Gaze estimation
    gaze = gaze_model.infer(face)
    if gaze is None:
        print("❌ L2CS không trả ra kết quả!")
        return

    yaw, pitch = gaze
    print(f"🎯 Gaze Estimation → yaw={yaw:.2f}, pitch={pitch:.2f}")

    # Vẽ vector ánh nhìn
    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 3

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    draw_gaze_vector(img, (cx, cy), yaw, pitch, length=120)

    cv2.putText(img, f"yaw={yaw:.1f}, pitch={pitch:.1f}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)

    # Lưu ảnh kết quả
    os.makedirs("runs/gaze_demo", exist_ok=True)
    out_path = "runs/gaze_demo/output.jpg"
    cv2.imwrite(out_path, img)

    print(f"\n✅ Đã lưu ảnh kết quả tại: {out_path}")

    # Hiển thị
    cv2.imshow("Gaze Estimation - L2CS Demo", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
