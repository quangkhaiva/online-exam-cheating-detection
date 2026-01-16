# src/detect_yolo.py
from ultralytics import YOLO
import cv2
import os

COLOR_MAP = {
    "student":      (0, 255, 255),   # vàng/cyan
    "extra_person": (0, 0, 255),     # đỏ
    "phone":        (255, 0, 0),     # xanh dương
    "book":         (0, 255, 0),     # xanh lá
    "laptop":       (255, 0, 255),   # tím
    "absence":      (200, 200, 200), # xám nhạt (dùng cho rule nếu cần)
}

def load_model(weights: str = "models/yolov8c.pt"):
    """
    Load YOLO và trả về (model, class_names_dict).
    class_names: dict {id:int -> label:str}
    """
    model = YOLO(weights)
    class_names = model.names
    return model, class_names

def infer_image(model, class_names, img_bgr, conf_thres: float = 0.5, **kwargs):
    """
    Trả về list:
        [(label:str, conf:float, (x1,y1,x2,y2)), ...]
    """
    results = model.predict(img_bgr, conf=conf_thres, verbose=False, **kwargs)
    dets = []

    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = class_names.get(cls_id, f"class_{cls_id}")
            dets.append((label, conf, (x1, y1, x2, y2)))
    return dets

def draw_detections(img_bgr, dets):
    for label, conf, (x1, y1, x2, y2) in dets:
        color = COLOR_MAP.get(label, (0, 255, 0))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        cv2.rectangle(img_bgr, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            img_bgr, text,
            (x1 + 1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, lineType=cv2.LINE_AA
        )
    return img_bgr

# ====== Test nhanh ======
if __name__ == "__main__":
    model_path = "models/yolov8c.pt"
    test_img = "data_test/sample_frames/Cam2vid1_frame_00071.jpg"

    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại: {model_path}")
        raise SystemExit

    if not os.path.exists(test_img):
        print(f"❌ Không tìm thấy ảnh test: {test_img}")
        raise SystemExit

    print(f"🔹 Load model: {model_path}")
    model, class_names = load_model(model_path)

    print("🔍 class_names từ model:")
    for k, v in class_names.items():
        print(f"  {k}: {v}")

    img = cv2.imread(test_img)
    dets = infer_image(model, class_names, img, conf_thres=0.5, imgsz=640, iou=0.45)

    print("✅ Kết quả detect:")
    for cls, conf, bbox in dets:
        print(f"  - {cls:<12} {conf:.2f}  {bbox}")

    img_vis = draw_detections(img.copy(), dets)
    os.makedirs("runs/detect", exist_ok=True)
    out_path = "runs/detect/test_output.jpg"
    cv2.imwrite(out_path, img_vis)
    print(f"🖼️ Ảnh output đã lưu tại: {out_path}")

