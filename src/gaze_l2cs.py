"""
L2CS-Net ONNX loader (yaw, pitch)
- Tự suy luận NBINS từ ONNX (66, 90, ...): dùng step phù hợp (3°/4°) nếu nhận ra.
- Hỗ trợ cả 2 output (yaw_logits, pitch_logits) và 1 output (2, NBINS) / (NBINS, 2) / (1,2,NBINS)...
- Tiền xử lý: 224x224, RGB, ImageNet mean/std, NCHW.
- Trả về: (yaw_deg, pitch_deg) hoặc None.
"""

from __future__ import annotations
import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple, List


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_providers() -> List[str]:
    """Ưu tiên CUDA nếu có, fallback CPU."""
    avail = ort.get_available_providers()
    return (['CUDAExecutionProvider', 'CPUExecutionProvider']
            if 'CUDAExecutionProvider' in avail else ['CPUExecutionProvider'])


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _make_bins(n_bins: int) -> np.ndarray:
    """
    Tạo vector bins theo số lượng bin:
    - 66  -> bước 3°  (≈ -99 .. +96)
    - 90  -> bước 4°  (≈ -180 .. +176)
    - khác -> chia đều [-180, +180)
    """
    if n_bins == 66:
        return np.arange(66, dtype=np.float32) * 3.0 - 99.0
    if n_bins == 90:
        return np.arange(90, dtype=np.float32) * 4.0 - 180.0
    step = 360.0 / float(n_bins)
    start = -180.0
    return (np.arange(n_bins, dtype=np.float32) * step + start).astype(np.float32)


class L2CS:
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        self.sess = ort.InferenceSession(onnx_path, providers=providers or pick_providers())
        self.input_name = self.sess.get_inputs()[0].name

        # Suy NBINS từ outputs
        outs = self.sess.get_outputs()
        if len(outs) == 2:
            # phổ biến: 2 output (yaw_logits, pitch_logits), NBINS là trục cuối
            n_bins = outs[0].shape[-1]
        else:
            # 1 output: cố gắng suy ra NBINS từ shape
            s = tuple(d for d in outs[0].shape if d is not None)  # bỏ None (batch)
            if len(s) == 3 and s[1] == 2:          # (1,2,NBINS) hoặc (2,NBINS,?)
                n_bins = s[2]
            elif len(s) == 2 and s[0] == 2:        # (2,NBINS)
                n_bins = s[1]
            elif len(s) == 2 and s[1] == 2:        # (NBINS,2)
                n_bins = s[0]
            else:                                   # fallback: trục cuối
                n_bins = s[-1]

        self.BINS = _make_bins(int(n_bins))

    # ---------- preprocessing ----------
    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
        return img.astype(np.float32)

    def _expected_angle(self, logits: np.ndarray) -> float:
        prob = _softmax(logits.astype(np.float32), axis=-1)
        return float(np.sum(prob * self.BINS))

    # ---------- inference ----------
    def infer(self, face_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Trả về (yaw_deg, pitch_deg) hoặc None nếu input không hợp lệ
        hoặc ONNX trả về shape/out bất thường.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        x = self._preprocess(face_bgr)
        outs = self.sess.run(None, {self.input_name: x})

        try:
            # Case 1: 2 output (yaw_logits, pitch_logits)
            if len(outs) == 2:
                yaw_logits   = np.squeeze(outs[0])
                pitch_logits = np.squeeze(outs[1])
                if yaw_logits.ndim   > 1: yaw_logits   = yaw_logits.reshape(-1)
                if pitch_logits.ndim > 1: pitch_logits = pitch_logits.reshape(-1)
                yaw   = self._expected_angle(yaw_logits)
                pitch = self._expected_angle(pitch_logits)
                return yaw, pitch

            # Case 2: 1 output
            out = np.squeeze(outs[0])
            # Chuẩn hoá về (2, NBINS)
            if out.ndim == 2:
                if out.shape[0] == 2:
                    yaw_logits, pitch_logits = out[0], out[1]
                elif out.shape[1] == 2:
                    yaw_logits, pitch_logits = out[:, 0], out[:, 1]
                else:
                    # không đoán được 2 nhánh -> fail an toàn
                    return None
            elif out.ndim == 3 and out.shape[0] == 2:
                # (2, NBINS, ?) -> lấy trục NBINS = dim1
                yaw_logits, pitch_logits = out[0], out[1]
            else:
                return None

            yaw   = self._expected_angle(np.asarray(yaw_logits))
            pitch = self._expected_angle(np.asarray(pitch_logits))
            return yaw, pitch

        except Exception:
            # giữ app chạy realtime ổn định
            return None
