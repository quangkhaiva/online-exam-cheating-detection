import cv2
import time
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

from src.rules_engine import RuleEngine
from src.visualize_utils import draw_detections, draw_gaze, draw_headpose_axes, draw_status_text

# ====== Gaze Estimation wrapper (L2CS) ======
from src.gaze_l2cs import L2CS  # lớp bạn đã có: load onnx, infer(face_bgr)->(yaw_deg,pitch_deg)

# ====== Head Pose wrapper (SixDRepNet) ======
from src.headpose_sixd import SixDRepONNX  # lớp của bạn: infer(face_bgr)->(yaw,pitch,roll)

# ====== YOLO Loader ======
def load_yolo(weights_path):
    model = YOLO(weights_path)
    class_names = model.names  # dict {id: "label"}
    return model, class_names

def yolo_detect(model, class_names, frame_bgr, conf_thres=0.5):
    """
    Trả về list detections: [(label, conf, (x1,y1,x2,y2))]
    và box lớn nhất thuộc 'student' (để crop mặt)
    """
    results = model.predict(frame_bgr, conf=conf_thres, verbose=False)
    dets = []
    student_box = None

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = class_names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            dets.append((label, conf, (x1, y1, x2, y2)))

            # chọn student to nhất trong khung (ưu tiên theo area)
            if label == "student":
                area = (x2 - x1) * (y2 - y1)
                if (student_box is None) or (area > student_box["area"]):
                    student_box = {"bbox": (x1,y1,x2,y2), "area": area}

    return dets, (student_box["bbox"] if student_box else None)

def crop_face_from_student_box(frame_bgr, student_bbox, expand_ratio=0.4):
    """
    student_bbox = (x1,y1,x2,y2)
    Cắt vùng mặt gần phần trên bbox. Có thể tinh chỉnh sau.
    """
    if student_bbox is None:
        return None

    x1, y1, x2, y2 = student_bbox
    w = x2 - x1
    h = y2 - y1

    # Lấy phần trên cơ thể (đầu + vai)
    top = y1
    bottom = y1 + int(h * 0.5)
    left = x1 + int(w * 0.2)
    right = x2 - int(w * 0.2)

    # mở rộng 1 chút
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    half_w = int((right - left) * (0.5 + expand_ratio))
    half_h = int((bottom - top) * (0.5 + expand_ratio))

    fx1 = max(0, cx - half_w)
    fx2 = min(frame_bgr.shape[1], cx + half_w)
    fy1 = max(0, cy - half_h)
    fy2 = min(frame_bgr.shape[0], cy + half_h)

    face_crop = frame_bgr[fy1:fy2, fx1:fx2].copy()
    face_center = (cx, cy)  # dùng để vẽ vector gaze/headpose lên frame
    return face_crop, face_center

def main():
    # ====== Config đường dẫn model ======
    YOLO_WEIGHTS     = "models/best.pt"            # model có laptop
    GAZE_ONNX_PATH   = "models/l2cs_gaze.onnx"
    HEADPOSE_ONNX    = "models/SixDRepNet.onnx"    # đổi đúng tên file bạn export

    # ====== Load model ======
    yolo_model, class_names = load_yolo(YOLO_WEIGHTS)
    gaze_model = L2CS(GAZE_ONNX_PATH)
    head_model = SixDRepONNX(HEADPOSE_ONNX)

    # ====== Khởi tạo rule engine ======
    fps_estimate = 30  # giả định ban đầu
    rules = RuleEngine(fps=fps_estimate)

    # ====== Webcam ======
    cap = cv2.VideoCapture(0)  # 0 = webcam mặc định
    if not cap.isOpened():
        print("❌ Không mở được webcam")
        return

    prev_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Mất frame từ webcam")
            break

        # 1. YOLO detect
        dets, student_bbox = yolo_detect(yolo_model, class_names, frame, conf_thres=0.5)

        # 2. Crop mặt từ student
        face_crop = None
        face_center = None
        if student_bbox is not None:
            face_crop, face_center = crop_face_from_student_box(frame, student_bbox)

        # 3. Gaze + Head pose (nếu có face)
        gaze_yaw, gaze_pitch = None, None
        head_yaw, head_pitch, head_roll = None, None, None
        if face_crop is not None and face_crop.size > 0:
            try:
                gaze_yaw, gaze_pitch = gaze_model.infer(face_crop)  # độ
            except Exception as e:
                # nếu lỗi onnx hoặc crop quá nhỏ
                gaze_yaw, gaze_pitch = None, None

            try:
                head_yaw, head_pitch, head_roll = head_model.infer(face_crop)
            except Exception as e:
                head_yaw, head_pitch, head_roll = None, None, None

        # 4. Cập nhật rules
        # Lấy danh sách chỉ tên lớp để feed vào rule engine
        det_labels = [lbl for (lbl, _, _) in dets]
        now_t = time.time()
        dt = now_t - prev_t
        prev_t = now_t

        # Ước lượng timestamp hiện tại tính bằng giây kể từ start
        # (Bạn có thể duy trì counter frame index thay cho timestamp)
        current_time_s = rules.frame_idx / fps_estimate if fps_estimate > 0 else 0

        rules.update(
            frame_idx=rules.frame_idx,
            dets=dets,
            gaze=(gaze_yaw, gaze_pitch) if gaze_yaw is not None else None,
            headpose=(head_yaw, head_pitch, head_roll) if head_yaw is not None else None
        )
        # bạn có thể lấy các event mới sinh:
        recent_events = rules.recent_events() if hasattr(rules, "recent_events") else []

        # 5. Vẽ overlay
        # 5.1 Vẽ bbox nhiều màu
        from src.detect_yolo import COLOR_MAP, draw_detections
        frame = draw_detections(frame, dets)

        # 5.2 Vẽ gaze vector từ mặt
        if face_center is not None and gaze_yaw is not None and gaze_pitch is not None:
            frame = draw_gaze(
                frame,
                origin=(int(face_center[0]), int(face_center[1])),
                yaw=gaze_yaw,
                pitch=gaze_pitch,
                length=60,
                color=(0, 255, 255)  # vàng
            )

        # 5.3 Vẽ trục head pose
        if face_center is not None and head_yaw is not None:
            frame = draw_headpose_axes(
                frame,
                center=(int(face_center[0]), int(face_center[1])),
                yaw=head_yaw,
                pitch=head_pitch,
                roll=head_roll,
                size=60
            )

        # 5.4 Overlay cảnh báo (ví dụ "PHONE DETECTED", "EXTRA PERSON",...)
        if len(recent_events) > 0:
            alert_text = recent_events[-1][0]  # lấy event mới nhất, ví dụ "PHONE_USAGE"
            frame = draw_status_text(frame, alert_text, pos=(20,40), color=(0,0,255))

        # 6. Hiển thị FPS thô
        fps_txt = f"FPS ~ {1.0/max(dt,1e-6):.1f}"
        cv2.putText(frame, fps_txt, (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 7. Hiển thị khung hình
        cv2.imshow("Online Exam Proctoring AI - Demo", frame)

        # phím q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ====== Cleanup ======
    cap.release()
    cv2.destroyAllWindows()

    # Lưu log sự kiện gian lận (nếu muốn)
    rules.export_csv("runs/behavior_logs/realtime_events.csv")
    print("Đã lưu log sự kiện vào runs/behavior_logs/realtime_events.csv")

if __name__ == "__main__":
    main()
