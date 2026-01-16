"""
SixDRepNet ONNX loader (yaw, pitch, roll)

- Tiền xử lý: 224x224, RGB, ImageNet mean/std, NCHW (float32).
- Tự động chọn ONNXRuntime providers (CUDA nếu có).
- Xử lý an toàn các biến thể shape output: (1,3) / (3,) / (1,1,3) ...
- Trả về tuple (yaw, pitch, roll) dạng float hoặc None nếu lỗi.
"""

from __future__ import annotations
import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple

# Tái dùng provider + hằng số từ L2CS để đồng bộ
from src.gaze_l2cs import pick_providers, IMAGENET_MEAN, IMAGENET_STD


class SixDRepONNX:
    def __init__(self, path: str = "models/sixdrepnet.onnx", providers=None):
        """
        path: đường dẫn ONNX của SixDRepNet (giữ mặc định chữ thường để khớp app.py)
        providers: list providers cho onnxruntime; mặc định chọn CUDA nếu có
        """
        self.sess = ort.InferenceSession(path, providers=providers or pick_providers())
        self.input_name = self.sess.get_inputs()[0].name

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Resize 224, BGR->RGB, chuẩn hoá ImageNet, HWC->NCHW, thêm batch."""
        img = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,224,224)
        return img.astype(np.float32)

    @staticmethod
    def _shape_to_ypr(out_any) -> Optional[Tuple[float, float, float]]:
        """
        Chuẩn hoá output của onnxruntime về (yaw, pitch, roll).
        Hỗ trợ các biến thể shape như: (1,3), (3,), (1,1,3), ...
        """
        if out_any is None:
            return None
        ypr = np.asarray(out_any, dtype=np.float32).reshape(-1)
        if ypr.size < 3:
            return None
        yaw, pitch, roll = float(ypr[0]), float(ypr[1]), float(ypr[2])
        # (tuỳ chọn) kẹp phạm vi “hợp lý” để tránh giá trị bậy bạ do lỗi model
        # yaw = float(np.clip(yaw, -180.0, 180.0))
        # pitch = float(np.clip(pitch, -90.0, 90.0))
        # roll = float(np.clip(roll, -180.0, 180.0))
        return yaw, pitch, roll

    def infer(self, face_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Trả về (yaw, pitch, roll) hoặc None nếu input/onnx lỗi.
        """
        try:
            if face_bgr is None or face_bgr.size == 0:
                return None
            h, w = face_bgr.shape[:2]
            if h < 10 or w < 10:
                return None  # vùng mặt quá nhỏ

            x = self._preprocess(face_bgr)
            outs = self.sess.run(None, {self.input_name: x})

            # Lấy output đầu tiên (hầu hết model export có 1 output)
            if not outs:
                return None
            return self._shape_to_ypr(outs[0])

        except Exception:
            # Fail an toàn để không làm vỡ stream
            return None
