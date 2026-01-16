# # src/rules_engine.py
# import csv, os
# import datetime
# from typing import List, Tuple, Optional, Dict

# class RuleEngine:
#     def __init__(
#         self,
#         fps: float = 30.0,
#         gaze_yaw_thresh: float = 15.0,
#         head_pitch_thresh: float = -10.0,
#         cooldown_s: float = 2.0
#     ):
#         self.fps = fps
#         # events: list dict {type, t_rel, t_abs}
#         self.events: List[Dict] = []
#         self._last_event_ts: Dict[str, float] = {}
#         self.gaze_yaw_thresh = float(gaze_yaw_thresh)
#         self.head_pitch_thresh = float(head_pitch_thresh)
#         self.cooldown_s = float(cooldown_s)

#     def reset(self):
#         self.events.clear()
#         self._last_event_ts.clear()

#     def _add_event(self, etype: str, t_rel: float):
#         last = self._last_event_ts.get(etype, -1e9)
#         if (t_rel - last) >= self.cooldown_s:
#             now = datetime.datetime.now()
#             time_str = now.strftime("%d/%m/%Y %H:%M:%S")

#             ev = {
#                 "type": etype,
#                 "t_rel": float(t_rel),
#                 "t_abs": time_str,
#             }
#             self.events.append(ev)
#             self._last_event_ts[etype] = t_rel

#             print(f"[EVENT] {etype} @ {time_str} (t={t_rel:.2f}s)")

#     def update(
#         self,
#         frame_idx: int,
#         dets,
#         gaze: Optional[Tuple[float, float]] = None,
#         headpose: Optional[Tuple[float, float, float]] = None
#     ):
#         frame_time = frame_idx / max(self.fps, 1.0)
#         labels = [cls for cls, *_ in dets]
#         added_now: List[Dict] = []

#         # 1) Không thấy student → ABSENCE
#         if "student" not in labels:
#             self._add_event("ABSENCE", frame_time)

#         # 2) Có extra_person → EXTRA_PERSON
#         if "extra_person" in labels:
#             self._add_event("EXTRA_PERSON", frame_time)

#         # 3) Có phone → PHONE_USAGE
#         if "phone" in labels:
#             self._add_event("PHONE_USAGE", frame_time)

#         # 4) Cúi đầu
#         if headpose is not None:
#             _, pitch, _ = headpose
#             if pitch < self.head_pitch_thresh:
#                 if "laptop" in labels:
#                     self._add_event("LOOKING_AT_LAPTOP", frame_time)
#                 else:
#                     self._add_event("HEAD_DOWN", frame_time)

#         # 5) Nhìn lệch
#         if gaze is not None:
#             yaw, _ = gaze
#             if abs(yaw) > self.gaze_yaw_thresh:
#                 self._add_event("LOOKING_AWAY", frame_time)

#         for ev in reversed(self.events):
#             if abs(ev["t_rel"] - frame_time) < 1e-6:
#                 added_now.append(ev)
#             else:
#                 break
#         return added_now

#     def export_csv(self, path: str):
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         with open(path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow(["type", "time_s", "time_real"])
#             for ev in self.events:
#                 writer.writerow([
#                     ev["type"],
#                     f"{ev['t_rel']:.2f}",
#                     ev["t_abs"],
#                 ])
# src/rules_engine.py
import csv, os
import datetime
from typing import List, Tuple, Optional, Dict

class RuleEngine:
    def __init__(
        self,
        fps: float = 30.0,
        gaze_yaw_thresh: float = 15.0,
        head_pitch_thresh: float = -10.0,
        cooldown_s: float = 2.0,
        absence_time_s: float = 2.0,          # ✅ mất student liên tục bao lâu thì tính ABSENCE
        material_requires_student: bool = True # ✅ book chỉ tính khi có student (tránh book ngoài khung)
    ):
        self.fps = fps
        self.events: List[Dict] = []
        self._last_event_ts: Dict[str, float] = {}

        self.gaze_yaw_thresh = float(gaze_yaw_thresh)
        self.head_pitch_thresh = float(head_pitch_thresh)
        self.cooldown_s = float(cooldown_s)

        self.absence_time_s = float(absence_time_s)
        self.material_requires_student = bool(material_requires_student)

        # ✅ counter cho ABSENCE
        self._no_student_frames = 0

    def reset(self):
        self.events.clear()
        self._last_event_ts.clear()
        self._no_student_frames = 0

    def _add_event(self, etype: str, t_rel: float) -> Optional[Dict]:
        """Add event with cooldown; return the event dict if added, else None."""
        last = self._last_event_ts.get(etype, -1e9)
        if (t_rel - last) >= self.cooldown_s:
            now = datetime.datetime.now()
            time_str = now.strftime("%d/%m/%Y %H:%M:%S")
            ev = {"type": etype, "t_rel": float(t_rel), "t_abs": time_str}
            self.events.append(ev)
            self._last_event_ts[etype] = t_rel
            print(f"[EVENT] {etype} @ {time_str} (t={t_rel:.2f}s)")
            return ev
        return None

    def update(
        self,
        frame_idx: int,
        dets,
        gaze: Optional[Tuple[float, float]] = None,
        headpose: Optional[Tuple[float, float, float]] = None
    ):
        frame_time = frame_idx / max(self.fps, 1.0)

        # dets: [(cls, conf, x1,y1,x2,y2), ...] hoặc tương tự
        labels = [cls for cls, *_ in dets]

        new_events: List[Dict] = []

        has_student = ("student" in labels)

        # =========================
        # 1) ABSENCE: mất student liên tục >= absence_time_s
        # =========================
        if not has_student:
            self._no_student_frames += 1
        else:
            self._no_student_frames = 0

        absence_frames_thresh = int(self.absence_time_s * self.fps)
        if self._no_student_frames >= absence_frames_thresh:
            ev = self._add_event("ABSENCE", frame_time)
            if ev: new_events.append(ev)

            # ✅ Khi ABSENCE xảy ra, thường không xét gaze/headpose nữa (tránh log sai)
            # Nếu bạn muốn vẫn log các lỗi khác, comment dòng return này.
            return new_events

        # =========================
        # 2) EXTRA_PERSON
        # =========================
        if "extra_person" in labels:
            ev = self._add_event("EXTRA_PERSON", frame_time)
            if ev: new_events.append(ev)

        # =========================
        # 3) PHONE
        # =========================
        if "phone" in labels:
            ev = self._add_event("PHONE_USAGE", frame_time)
            if ev: new_events.append(ev)

        # =========================
        # 4) BOOK -> USING_MATERIAL (✅ mới)
        # =========================
        if "book" in labels:
            if (not self.material_requires_student) or has_student:
                ev = self._add_event("USING_MATERIAL", frame_time)
                if ev: new_events.append(ev)

        # =========================
        # 5) HEADPOSE: chỉ xét khi có student
        # =========================
        if has_student and headpose is not None:
            _, pitch, _ = headpose
            if pitch < self.head_pitch_thresh:
                if "laptop" in labels:
                    ev = self._add_event("LOOKING_AT_LAPTOP", frame_time)
                else:
                    ev = self._add_event("HEAD_DOWN", frame_time)
                if ev: new_events.append(ev)

        # =========================
        # 6) GAZE: chỉ xét khi có student
        # =========================
        if has_student and gaze is not None:
            yaw, _ = gaze
            if abs(yaw) > self.gaze_yaw_thresh:
                ev = self._add_event("LOOKING_AWAY", frame_time)
                if ev: new_events.append(ev)

        return new_events

    def export_csv(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "time_s", "time_real"])
            for ev in self.events:
                writer.writerow([ev["type"], f"{ev['t_rel']:.2f}", ev["t_abs"]])
