import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# ================== CONFIG ==================
VIDEO_SOURCE = "data_test/test_videos/Vid21.mp4"
YOLO_WEIGHTS = "models/best.pt"
L2CS_WEIGHTS = "models/l2cs_gaze.onnx"
HEADPOSE_WEIGHTS = "models/SixDRepNet.onnx"

FACE_OUT_PATH = "runs/behavior_logs/face_crop_224x224_REAL.png"
OVERLAY_OUT_PATH = "runs/behavior_logs/overlay_gaze_headpose_REAL.png"

CONF_THRES = 0.4
IOU_THRES = 0.45
TARGET_CLASS_STUDENT = "student"
# ============================================


def load_models():
    print("[INFO] Loading YOLOv11...")
    yolo_model = YOLO(YOLO_WEIGHTS)

    print("[INFO] Loading L2CS-Net ONNX...")
    # fallback provider list: thử CUDA nếu có, nếu không thì CPU
    provider_list = ort.get_available_providers()
    providers_to_use = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in provider_list
        else ["CPUExecutionProvider"]
    )
    gaze_sess = ort.InferenceSession(L2CS_WEIGHTS, providers=providers_to_use)

    print("[INFO] Loading SixDRepNet ONNX...")
    headpose_sess = ort.InferenceSession(HEADPOSE_WEIGHTS, providers=providers_to_use)

    return yolo_model, gaze_sess, headpose_sess


def preprocess_face_for_models(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    img = face_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # (1,3,224,224)
    return img, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)  # trả về bản hiển thị BGR 224x224


# helper softmax cho numpy
def _np_softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)
np.softmax = _np_softmax


def infer_gaze(gaze_sess, face_input):
    """
    Giả định output:
    outs[0] = yaw_logits (1,90)
    outs[1] = pitch_logits (1,90)
    Nếu mô hình của bạn khác (ví dụ trả trực tiếp yaw,pitch), mình sẽ chỉnh tiếp dựa trên lỗi lần sau.
    """
    inputs = {gaze_sess.get_inputs()[0].name: face_input}
    outs = gaze_sess.run(None, inputs)

    yaw_logits = outs[0][0]   # shape (90,)
    pitch_logits = outs[1][0] # shape (90,)

    bins = np.arange(len(yaw_logits), dtype=np.float32)  # [0..89]
    yaw_deg = float((np.softmax(yaw_logits) * (bins - 45)).sum())
    pitch_deg = float((np.softmax(pitch_logits) * (bins - 45)).sum())

    return yaw_deg, pitch_deg


def infer_headpose(headpose_sess, face_input):
    """
    Nhiều bản SixDRepNet ONNX trả ra dạng [[yaw, pitch, roll]]
    hoặc đôi khi nhiều tensor.
    Ta sẽ đọc tất cả output, nối lại và ép float.
    """
    inputs = {headpose_sess.get_inputs()[0].name: face_input}
    outs = headpose_sess.run(None, inputs)

    # Trường hợp phổ biến nhất:
    # outs[0].shape == (1,3) -> [[yaw, pitch, roll]]
    hp = outs[0]
    if isinstance(hp, list):
        hp = np.array(hp)

    hp = np.array(hp).reshape(-1)  # (3,)
    yaw, pitch, roll = hp[0], hp[1], hp[2]

    # ép về float thuần (Python float) để format được
    return float(yaw), float(pitch), float(roll)


def draw_overlay(frame, face_bbox, gaze_yaw, gaze_pitch,
                 head_yaw, head_pitch, head_roll):
    (x1, y1, x2, y2) = face_bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    overlay = frame.copy()

    # ----- Vector gaze (mũi tên xanh lá) -----
    length = 150
    dx = int(length * np.cos(np.radians(gaze_pitch)) * np.sin(np.radians(gaze_yaw)))
    dy = int(length * np.sin(np.radians(gaze_pitch)) * -1)

    cv2.arrowedLine(
        overlay,
        (cx, cy),
        (cx + dx, cy + dy),
        (0, 255, 0),  # xanh lá cho gaze
        4,
        tipLength=0.25
    )
    cv2.putText(
        overlay,
        f"gaze yaw={gaze_yaw:.1f} pitch={gaze_pitch:.1f}",
        (cx + dx + 10, cy + dy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 120, 0),
        2,
        cv2.LINE_AA
    )

    # ----- Head pose axes -----
    axis_len = 100
    # X (đỏ)
    cv2.arrowedLine(
        overlay, (cx, cy), (cx + axis_len, cy),
        (0, 0, 255), 4, tipLength=0.25
    )
    # Y (xanh dương)
    cv2.arrowedLine(
        overlay, (cx, cy), (cx, cy - axis_len),
        (255, 0, 0), 4, tipLength=0.25
    )
    # Z (vàng)
    cv2.arrowedLine(
        overlay,
        (cx, cy),
        (cx - int(axis_len*0.6), cy + int(axis_len*0.4)),
        (0, 255, 255), 4, tipLength=0.25
    )

    cv2.putText(
        overlay,
        f"head yaw={head_yaw:.1f} pitch={head_pitch:.1f} roll={head_roll:.1f}",
        (cx - 160, cy - 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (50, 50, 50),
        2,
        cv2.LINE_AA
    )

    return overlay


def main():
    # 1. Load model
    yolo_model, gaze_sess, headpose_sess = load_models()

    # 2. Lấy frame đầu tiên từ video test
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Khong doc duoc frame tu VIDEO_SOURCE. Kiem tra duong dan/codec video.")

    # 3. YOLO detect
    results = yolo_model.predict(
        frame,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False
    )
    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy().astype(int)

    # 4. Tìm bbox student
    face_bbox = None
    for (cid, box_xyxy) in zip(cls_ids, xyxy):
        class_name = yolo_model.names[cid]
        if class_name == TARGET_CLASS_STUDENT:
            face_bbox = box_xyxy  # (x1,y1,x2,y2)
            break

    if face_bbox is None:
        raise RuntimeError("Khong tim thay 'student' trong frame. Thu video/frame khac hoac giam CONF_THRES.")

    x1, y1, x2, y2 = map(int, face_bbox)

    # 5. Lấy vùng upper-half làm "khuôn mặt"
    face_y2 = y1 + (y2 - y1)//2
    face_crop_bgr = frame[y1:face_y2, x1:x2].copy()

    # 6. Chuẩn hóa face cho model
    face_input, face_vis_224 = preprocess_face_for_models(face_crop_bgr)

    # 7. Gaze
    gaze_yaw, gaze_pitch = infer_gaze(gaze_sess, face_input)

    # 8. Head pose
    head_yaw, head_pitch, head_roll = infer_headpose(headpose_sess, face_input)

    # 9. Vẽ overlay
    overlay = draw_overlay(
        frame,
        (x1, y1, x2, face_y2),
        gaze_yaw, gaze_pitch,
        head_yaw, head_pitch, head_roll
    )

    # 10. Lưu ảnh output
    cv2.imwrite(FACE_OUT_PATH, face_vis_224)
    cv2.imwrite(OVERLAY_OUT_PATH, overlay)

    print("[DONE]")
    print("Face crop (224x224):", FACE_OUT_PATH)
    print("Overlay gaze/headpose:", OVERLAY_OUT_PATH)


if __name__ == "__main__":
    main()
