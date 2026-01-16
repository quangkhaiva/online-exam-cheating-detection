# src/behavior_analyzer.py
import cv2, os

from detect_yolo import load_model, infer_image, draw_detections
from face_crop import crop_head_top_half
from gaze_l2cs import L2CS
from headpose_sixd import SixDRepONNX
from rules_engine import RuleEngine
from visualize_utils import draw_gaze_vector, draw_headpose_axes


def analyze_video(
    video_path: str,
    yolo_w: str = "models/yolov8c.pt",
    gaze_onnx: str = "models/l2cs_gaze.onnx",
    headpose_onnx: str = "models/SixDRepNet.onnx",
    out_dir: str = "runs/overlay_videos"
):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ===== Load models =====
    model_yolo, class_names = load_model(yolo_w)
    gaze = L2CS(gaze_onnx)
    headpose = SixDRepONNX(headpose_onnx)

    rules = RuleEngine(fps=fps)

    out = None
    frame_idx = 0

    print("🎥 Analyze video:", video_path)
    print("📸 FPS:", fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ===== YOLO detect =====
        dets = infer_image(
            model=model_yolo,
            class_names=class_names,
            img_bgr=frame,
            conf_thres=0.5,
            imgsz=640,
            iou=0.45
        )

        # ===== Lấy bbox student lớn nhất =====
        student_bbox = None
        max_area = 0
        for label, conf, (x1, y1, x2, y2) in dets:
            if label == "student":
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    student_bbox = (x1, y1, x2, y2)

        # ===== Crop face =====
        face = None
        if student_bbox is not None:
            face, _ = crop_head_top_half(frame, student_bbox)

        # ===== Gaze & Headpose =====
        g = gaze.infer(face) if face is not None else None      # (yaw, pitch)
        h = headpose.infer(face) if face is not None else None  # (yaw, pitch, roll)

        # ===== Rule Engine =====
        added_events = rules.update(
            frame_idx=frame_idx,
            dets=dets,
            gaze=g,
            headpose=h
        )

        # ===== Debug event mới =====
        if added_events:
            print(
                f"[FRAME {frame_idx}] EVENTS:",
                [e["type"] for e in added_events]
            )

        # ===== Draw YOLO bbox =====
        frame = draw_detections(frame, dets)

        # ===== Draw gaze & headpose =====
        if student_bbox is not None:
            sx1, sy1, sx2, sy2 = student_bbox

            cx_gaze = sx1 + (sx2 - sx1) // 2
            cy_gaze = sy1 + (sy2 - sy1) // 3

            cx_head = sx1 + (sx2 - sx1) // 2
            cy_head = sy1 + (sy2 - sy1) // 2

            if g is not None:
                yaw, pitch = g
                draw_gaze_vector(frame, (cx_gaze, cy_gaze), yaw, pitch)

            if h is not None:
                hyaw, hpitch, hroll = h
                draw_headpose_axes(frame, (cx_head, cy_head), hyaw, hpitch, hroll)

        # ===== Init video writer =====
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                os.path.join(out_dir, "output_overlay.mp4"),
                fourcc,
                fps,
                (frame.shape[1], frame.shape[0])
            )

        out.write(frame)

    cap.release()
    if out:
        out.release()

    # ===== Export log =====
    os.makedirs("runs/behavior_logs", exist_ok=True)
    rules.export_csv(os.path.join("runs/behavior_logs", "events.csv"))

    print("✅ Done analyzing:", video_path)
    print("📄 Log saved to runs/behavior_logs/events.csv")


if __name__ == "__main__":
    test_video = "data_test/test_videos/video_demo.mp4"
    if os.path.exists(test_video):
        analyze_video(test_video)
    else:
        print("❌ Không tìm thấy video demo:", test_video)
