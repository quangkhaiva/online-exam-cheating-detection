import os, sys, cv2, time, numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

# ===== Cho phép import từ thư mục src/ =====
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.detect_yolo import load_model, infer_image, draw_detections
from src.gaze_l2cs import L2CS, pick_providers
from src.headpose_sixd import SixDRepONNX
from src.face_crop import refine_face_with_mediapipe, crop_head_top_half, to_square
from src.rules_engine import RuleEngine

app = Flask(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR  = os.path.join(BASE_DIR, "web_demo", "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
RUNS_DIR    = os.path.join(BASE_DIR, "runs")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.join(RUNS_DIR, "logs"), exist_ok=True)

# ========== MODEL ZOO ==========
MODEL_ZOO = {
    "yolov8s": "models/best8s.pt",
    "yolov8c": "models/best8c.pt",
}

# ========== LOAD MODELS ==========
gaze_model     = L2CS("models/l2cs_gaze.onnx", providers=pick_providers())
headpose_model = SixDRepONNX("models/sixdrepnet.onnx", providers=pick_providers())

current_yolo_key = "yolov8s"
yolo_model, class_names = load_model(MODEL_ZOO[current_yolo_key])

# ===== Rule Engine (ABSENCE > 3s) =====
realtime_rules = RuleEngine(
    fps=30,
    absence_time_s=3.0   # 🔥 KHÔNG THẤY STUDENT > 3s MỚI BÁO
)

realtime_events_cache = []


def draw_overlay(img, face_box, gaze=None, headpose=None):
    if not face_box:
        return img

    x1, y1, x2, y2 = map(int, face_box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if gaze:
        yaw, pitch = gaze
        dx = int(np.sin(np.radians(yaw)) * 100)
        dy = int(np.sin(np.radians(pitch)) * 100)
        cv2.arrowedLine(img, (cx, cy), (cx + dx, cy - dy),
                        (255, 255, 0), 2, tipLength=0.2)

    if headpose:
        yaw_h, pitch_h, roll_h = headpose
        cv2.putText(
            img,
            f"head y={yaw_h:.1f} p={pitch_h:.1f} r={roll_h:.1f}",
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 1
        )
    return img


def gen_frames():
    global realtime_rules, realtime_events_cache, yolo_model, class_names

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    realtime_rules = RuleEngine(
        fps=fps,
        absence_time_s=3.0   # 🔥 realtime: >3s
    )

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # ===== YOLO =====
        dets = infer_image(
            model=yolo_model,
            class_names=class_names,
            img_bgr=frame,
            conf_thres=0.5,
            imgsz=640,
            iou=0.45
        )

        frame_vis = draw_detections(frame.copy(), dets)

        # ===== Tìm student =====
        student_bbox = None
        for label, _, bbox in dets:
            if label == "student":
                student_bbox = bbox
                break

        gaze = None
        headpose = None
        face_box = None

        # ===== Nếu có student → infer gaze & headpose =====
        if student_bbox is not None:
            face_head, face_box = crop_head_top_half(frame, student_bbox, margin=0.15)
            face_ref, face_box_ref = refine_face_with_mediapipe(frame, face_box)

            if face_ref is not None:
                face_head, face_box = face_ref, face_box_ref

            if face_head is not None and face_head.size > 0:
                face_sq = to_square(face_head)
                try:
                    gaze     = gaze_model.infer(face_sq)
                    headpose = headpose_model.infer(face_sq)
                except Exception:
                    gaze = headpose = None

                if gaze or headpose:
                    frame_vis = draw_overlay(frame_vis, face_box, gaze, headpose)

        # ===== 🔥 LUÔN UPDATE RULE ENGINE =====
        added_events = realtime_rules.update(
            frame_idx=frame_idx,
            dets=dets,
            gaze=gaze,
            headpose=headpose
        )

        if added_events:
            print("[REALTIME EVENTS]", [e["type"] for e in added_events])

        # cache 20 event gần nhất
        realtime_events_cache = realtime_rules.events[-20:]

        # ===== Stream frame =====
        ok, buf = cv2.imencode(".jpg", frame_vis)
        if ok:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes() +
                b"\r\n"
            )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream_page():
    global current_yolo_key, yolo_model, class_names

    key = request.args.get("model", "yolov8s")
    if key in MODEL_ZOO:
        current_yolo_key = key
        yolo_model, class_names = load_model(MODEL_ZOO[key])
        print(f"[INFO] Webcam model: {key}")

    model_name = "YOLOv8s (pretrained)" if key == "yolov8s" else "YOLOv8 custom"
    return render_template("stream.html", model_name=model_name)


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/events_feed")
def events_feed():
    return jsonify([
        {
            "type": ev["type"],
            "time_s": round(ev["t_rel"], 2),
            "time_str": ev["t_abs"]
        }
        for ev in realtime_events_cache
    ])


# ================= OFFLINE VIDEO =================

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return "No file", 400

    model_key = request.form.get("model_choice", "yolov8s")
    yolo_path = MODEL_ZOO.get(model_key, MODEL_ZOO["yolov8s"])

    model_yolo, class_names = load_model(yolo_path)

    f = request.files["video"]
    ts = int(time.time())
    in_path  = os.path.join(UPLOADS_DIR, f"input_{ts}.mp4")
    out_path = os.path.join(UPLOADS_DIR, f"result_{ts}.mp4")
    csv_path = os.path.join(RUNS_DIR, "logs", f"log_{ts}.csv")
    f.save(in_path)

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    rules = RuleEngine(
        fps=fps,
        absence_time_s=3.0
    )

    writer = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        dets = infer_image(model_yolo, class_names, frame,
                           conf_thres=0.5, imgsz=640, iou=0.45)

        vis = draw_detections(frame.copy(), dets)

        rules.update(frame_idx, dets)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h)
            )

        writer.write(vis)

    cap.release()
    writer.release()
    rules.export_csv(csv_path)

    return render_template(
        "result.html",
        model_name=model_key,
        processed_filename=os.path.basename(out_path),
        log_path=os.path.basename(csv_path)
    )


@app.route("/static/uploads/<path:filename>")
def serve_processed_video(filename):
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/logs/<path:filename>")
def serve_logfile(filename):
    return send_from_directory(os.path.join(RUNS_DIR, "logs"),
                               filename, mimetype="text/csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
