import cv2
import numpy as np
import os

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "mediapipe chưa được cài. Cài bằng:\n"
        "    pip install mediapipe==0.10.11\n"
        f"Lỗi gốc: {e}"
    )

class FaceMeshExtractor:
    """
    Dùng MediaPipe Face Mesh để trích xuất 468 landmarks khuôn mặt.
    - extract(): trả về (landmarks_2d, landmarks_3d)
    - draw_full_mesh(): vẽ lưới mesh trực tiếp lên frame
    """

    def __init__(self,
                 static_mode=False,
                 max_faces=1,
                 refine_landmarks=True,
                 min_det_conf=0.5,
                 min_track_conf=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf
        )

        self.drawing_utils = mp.solutions.drawing_utils
        self.draw_spec_point = self.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1
        )
        self.draw_spec_conn = self.drawing_utils.DrawingSpec(
            color=(0, 100, 255), thickness=1, circle_radius=1
        )

    def extract(self, frame_bgr):
        """
        frame_bgr: ảnh BGR (numpy HxWx3)
        return:
            lm2d: (468, 2) toạ độ pixel x,y
            lm3d: (468, 3) toạ độ chuẩn hoá x,y,z
        hoặc (None, None) nếu không thấy mặt
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return None, None

        face_landmarks = result.multi_face_landmarks[0].landmark

        lm2d = []
        lm3d = []
        for lm in face_landmarks:
            x_px = lm.x * w
            y_px = lm.y * h
            lm2d.append([x_px, y_px])
            lm3d.append([lm.x, lm.y, lm.z])

        lm2d = np.array(lm2d, dtype=np.float32)
        lm3d = np.array(lm3d, dtype=np.float32)

        return lm2d, lm3d

    def draw_full_mesh(self, frame_bgr):
        """
        Vẽ mesh (tessellation) lên frame và trả về frame_bgr đã vẽ.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(frame_rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                self.drawing_utils.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.draw_spec_point,
                    connection_drawing_spec=self.draw_spec_conn
                )
        return frame_bgr


def _test_single_image(input_path, output_debug_path="facemesh_debug.jpg"):
    """
    Hàm test nội bộ:
    - Đọc ảnh input_path
    - Chạy FaceMeshExtractor
    - In số landmark
    - Vẽ mesh và LƯU ra output_debug_path
    - Không dùng cv2.imshow() để tránh crash môi trường headless
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh test: {input_path}")

    print(f"[INFO] Đang đọc ảnh: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError("cv2.imread trả về None (ảnh hỏng hoặc đường dẫn sai?)")

    fm = FaceMeshExtractor(static_mode=True, max_faces=1, refine_landmarks=True)

    lm2d, lm3d = fm.extract(img)

    if lm2d is None:
        print("[WARN] Không phát hiện khuôn mặt.")
    else:
        print(f"[OK] Số landmarks: {lm2d.shape[0]}")
        print(f"[OK] Landmark đầu tiên (pixel): {lm2d[0]}")

    # vẽ mesh và lưu ra file
    debug_img = fm.draw_full_mesh(img.copy())
    cv2.imwrite(output_debug_path, debug_img)
    print(f"[INFO] Đã lưu ảnh debug mesh vào: {output_debug_path}")


if __name__ == "__main__":
    # đường dẫn ảnh test mặc định dựa theo cấu trúc của bạn
    default_img = "data_test/sample_frames/Vid1_frame_00015.jpg"
    _test_single_image(default_img)
