# import cv2, numpy as np

# def draw_gaze_vector(img, center, yaw, pitch, length=80):
#     x = int(center[0] + length * -np.sin(np.deg2rad(yaw)))
#     y = int(center[1] + length * -np.sin(np.deg2rad(pitch)))
#     cv2.arrowedLine(img, center, (x,y), (255,0,0), 2, tipLength=0.2)

# def draw_headpose_axes(img, center, yaw, pitch, roll, size=50):
#     yaw, pitch, roll = np.deg2rad([yaw,pitch,roll])
#     R = np.array([
#         [np.cos(yaw)*np.cos(roll),
#          np.sin(roll)*np.sin(pitch)*np.cos(yaw)-np.sin(yaw)*np.cos(pitch),
#          np.sin(roll)*np.cos(pitch)*np.cos(yaw)+np.sin(yaw)*np.sin(pitch)],
#         [np.sin(yaw)*np.cos(roll),
#          np.sin(roll)*np.sin(pitch)*np.sin(yaw)+np.cos(yaw)*np.cos(pitch),
#          np.sin(roll)*np.cos(pitch)*np.sin(yaw)-np.cos(yaw)*np.sin(pitch)],
#         [-np.sin(roll), np.cos(roll)*np.sin(pitch), np.cos(roll)*np.cos(pitch)]
#     ])
#     axes = np.float32([[size,0,0],[0,size,0],[0,0,size]])
#     proj = R @ axes.T
#     proj = proj.T[:, :2].astype(int)
#     origin = np.array(center, dtype=int)
#     cv2.line(img, tuple(origin), tuple(origin + proj[0]), (0,0,255), 2)
#     cv2.line(img, tuple(origin), tuple(origin + proj[1]), (0,255,0), 2)
#     cv2.line(img, tuple(origin), tuple(origin + proj[2]), (255,0,0), 2)
# COLOR_MAP = {
#     "student": (0, 255, 255),
#     "extra_person": (255, 0, 0),
#     "phone": (0, 0, 255),
#     "book": (255, 255, 0),
#     "absence": (180, 180, 180),
#     "laptop": (128, 0, 255)
# }
# src/visualize_utils.py
import cv2
import numpy as np

COLOR_MAP = {
    "student":      (0, 255, 255),
    "extra_person": (255,   0,   0),
    "phone":        (0,     0, 255),
    "book":         (255, 255,   0),
    "absence":      (180, 180, 180),
    "laptop":       (128,   0, 255),
}


def draw_gaze_vector(
    img,
    origin,
    yaw_deg: float,
    pitch_deg: float,
    length: int = 120,
    color=(0, 255, 0),
    thickness: int = 2
):
    """
    Vẽ vector gaze (mũi tên màu xanh) từ điểm origin.
    yaw > 0: nhìn sang phải; pitch > 0: nhìn lên (theo quy ước L2CS).
    """
    ox, oy = origin
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # đơn giản hoá: chiếu 3D -> 2D tương đối
    dx = length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)

    end = (int(ox + dx), int(oy + dy))
    cv2.arrowedLine(img, (ox, oy), end, color, thickness, tipLength=0.25)
    return img


def draw_headpose_axes(
    img,
    origin,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    length: int = 80
):
    """
    Vẽ 3 trục head pose x (đỏ), y (xanh lá), z (xanh dương) đơn giản.
    Đây là minh hoạ trực quan, không phải phép chiếu camera chính xác.
    """
    ox, oy = origin

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    # ma trận quay yaw-pitch-roll (Rz*Ry*Rx)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [0,            1,           0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0,                     0,    1]])
    R = Rz @ Ry @ Rx

    axes = np.float32([[length, 0, 0],
                       [0, length, 0],
                       [0, 0, length]])

    proj = R @ axes.T  # (3,3)

    # x: đỏ, y: xanh lá, z: xanh dương
    x_end = (int(ox + proj[0, 0]), int(oy - proj[1, 0]))
    y_end = (int(ox + proj[0, 1]), int(oy - proj[1, 1]))
    z_end = (int(ox + proj[0, 2]), int(oy - proj[1, 2]))

    cv2.line(img, (ox, oy), x_end, (0, 0, 255), 2)   # X – đỏ
    cv2.line(img, (ox, oy), y_end, (0, 255, 0), 2)   # Y – xanh lá
    cv2.line(img, (ox, oy), z_end, (255, 0, 0), 2)   # Z – xanh dương

    return img
