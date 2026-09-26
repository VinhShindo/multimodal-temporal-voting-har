"""
collector.py — Bộ điều khiển thu data đồng bộ Camera + Watch
              cho 20 hành động (10 movement + 10 gesture/grasp).

Cách dùng:
    python tools/collector.py \
        --watch http://192.168.0.105 \
        --session subject_001_session_01 \
        --out data/raw \
        --group movement \
        --reps 3 \
        --action-duration 5.0
"""
import argparse
import csv
import json
import platform
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests

from actions import ACTIONS, NO_ACTION_ID, get_by_group
from imu_server import ImuServer


# ============ WATCH CLIENT ============
class WatchClient:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self) -> dict:
        return requests.get(f"{self.base_url}/status",
                            timeout=self.timeout).json()

    def start(self, session_id: str, start_utc_ms: int) -> dict:
        r = requests.post(f"{self.base_url}/start_recording",
                          json={"session_id": session_id,
                                "start_utc_ms": start_utc_ms},
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def stop(self) -> dict:
        return requests.post(f"{self.base_url}/stop_recording",
                             timeout=self.timeout).json()

    def reset(self) -> dict:
        return requests.post(f"{self.base_url}/reset_state",
                             timeout=self.timeout).json()

    def set_action(self, action_id: int) -> dict:
        return requests.post(f"{self.base_url}/set_action",
                             json={"action_id": action_id},
                             timeout=self.timeout).json()


# ============ CAMERA ============
def open_camera(cam_index: int, width: int, height: int, target_fps: int,
                codec: str = "MJPG"):
    """Mở camera với DSHOW (Windows) / V4L2 (Linux)."""
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2

    cap = cv2.VideoCapture(cam_index, backend)
    if not cap.isOpened():
        print("[Camera] Fallback default backend...")
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera index={cam_index}")

    # Đợi camera init
    time.sleep(0.5)

    # Order: codec → resolution → fps, mỗi bước đợi một chút
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
    time.sleep(0.15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(0.15)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(0.3)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)

    print(f"[Camera] Request {width}x{height}@{target_fps} {codec} "
          f"→ Got {aw}x{ah}@{af:.1f}")
    return cap, aw, ah


def warmup_and_measure(cap, warmup_frames: int = 30,
                       measure_sec: float = 1.5):
    """Bỏ N frame đầu rồi đo FPS thực."""
    t0 = time.time()
    for _ in range(warmup_frames):
        cap.read()
    warm_time = time.time() - t0
    print(f"[Camera] Warmup {warmup_frames} frames = {warm_time:.2f}s")

    t0 = time.time()
    n = 0
    while time.time() - t0 < measure_sec:
        ret, _ = cap.read()
        if ret:
            n += 1
    elapsed = time.time() - t0
    fps = n / elapsed if elapsed > 0 else 0
    return fps


def record_camera(session_id: str, out_dir: Path,
                  fps: int, width: int, height: int,
                  duration_sec: float, stop_event: threading.Event,
                  cam_index: int = 0):
    """
    Ghi camera với fallback chain:
      1. Request như user chỉ định (default 1920x1080)
      2. Nếu FPS < 25 → thử 640x480 YUYV
    """
    # --- Bước 1: request như user yêu cầu ---
    cap, aw, ah = open_camera(cam_index, width, height, fps, "MJPG")
    measured_fps = warmup_and_measure(cap, 30, 1.5)
    print(f"[Camera] FPS thực: {measured_fps:.1f}")

    # --- Fallback nếu FPS thấp ---
    if measured_fps < 25:
        print(f"[Camera] ⚠️ FPS < 25, fallback 640x480 YUYV...")
        cap.release()
        time.sleep(0.5)
        cap, aw, ah = open_camera(cam_index, 640, 480, fps, "YUYV")
        measured_fps = warmup_and_measure(cap, 30, 1.5)
        print(f"[Camera] Fallback 640x480: {measured_fps:.1f} fps")

    use_fps = max(1, int(round(measured_fps)))
    print(f"[Camera] VideoWriter FPS = {use_fps}  ({aw}x{ah})")

    # --- Writer ---
    video_path = out_dir / f"{session_id}.mp4"
    writer = cv2.VideoWriter(str(video_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             use_fps, (aw, ah))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không tạo được VideoWriter {video_path}")

    # --- Ghi ---
    frames = []
    t0 = time.time()
    last_log = t0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("\n[Camera] ⚠️ read() thất bại — dừng")
            break
        writer.write(frame)
        frames.append((len(frames), int(time.time() * 1000)))

        if time.time() - last_log >= 2.0:
            elapsed = time.time() - t0
            live_fps = len(frames) / elapsed if elapsed > 0 else 0
            print(f"[Camera] {len(frames)} frames | "
                  f"{live_fps:.1f} fps | {elapsed:.0f}s", end="\r")
            last_log = time.time()

        if duration_sec > 0 and (time.time() - t0) >= duration_sec:
            break

    cap.release()
    writer.release()

    total_dur = time.time() - t0
    real_fps = len(frames) / total_dur if total_dur > 0 else 0
    print(f"\n[Camera] {len(frames)} frames / {total_dur:.1f}s "
          f"= {real_fps:.1f} fps (writer={use_fps})")
    print(f"[Camera] Video: {video_path}")

    # Lưu timestamps
    frames_csv = out_dir / f"{session_id}_frames.csv"
    with open(frames_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_utc_ms"])
        w.writerows(frames)
    print(f"[Camera] Timestamps: {frames_csv}")

    if real_fps < use_fps * 0.85:
        print(f"[Camera] ⚠️ FPS giảm giữa session: "
              f"{real_fps:.1f} < {use_fps}")


# ============ MAIN ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--group", choices=["movement", "gesture", "all"],
                    default="all")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--action-duration", type=float, default=5.0)
    ap.add_argument("--rest-duration", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--imu-port", type=int, default=5248)
    ap.add_argument("--lead-sec", type=float, default=3.0)
    ap.add_argument("--cam-index", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out) / args.session
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.group == "all":
        actions = list(ACTIONS)
    else:
        actions = get_by_group(args.group)
    if not actions:
        print("[LỖI] Không có action nào.")
        return

    print(f"\n=== SESSION: {args.session} ===")
    print(f"Số action: {len(actions)} × {args.reps} lần")
    total_est = len(actions) * args.reps * (
        args.action_duration + args.rest_duration)
    print(f"Tổng thời gian ước tính: {total_est:.0f}s\n")

    # ============ 1) CHUẨN BỊ CAMERA TRƯỚC ============
    # Mở + warmup + measure + tạo writer TRƯỚC khi tính t_start
    print("[Camera] Chuẩn bị (warmup + measure)...")
    cap, aw, ah = open_camera(args.cam_index, args.width, args.height,
                              args.fps, "MJPG")
    measured_fps = warmup_and_measure(cap, 30, 1.5)
    print(f"[Camera] FPS thực: {measured_fps:.1f}")

    if measured_fps < 25:
        print(f"[Camera] ⚠️ FPS < 25, fallback 640x480 YUYV...")
        cap.release()
        time.sleep(0.5)
        cap, aw, ah = open_camera(args.cam_index, 640, 480, args.fps, "YUYV")
        measured_fps = warmup_and_measure(cap, 30, 1.5)
        print(f"[Camera] Fallback 640x480: {measured_fps:.1f} fps")

    use_fps = max(1, int(round(measured_fps)))
    video_path = out_dir / f"{args.session}.mp4"
    writer = cv2.VideoWriter(str(video_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             use_fps, (aw, ah))
    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được VideoWriter {video_path}")
    print(f"[Camera] Sẵn sàng: {aw}x{ah} @ {use_fps} fps")

    # ============ 2) IMU server ============
    imu_server = ImuServer(args.imu_port, Path(args.out))
    imu_server.start()
    time.sleep(0.3)

    # ============ 3) Watch ============
    watch = WatchClient(args.watch)
    try:
        st = watch.status()
        print(f"[Watch] {st}")
    except Exception as e:
        print(f"[LỖI] Không kết nối được watch: {e}")
        cap.release(); writer.release()
        return

    try:
        watch.reset()
    except Exception as e:
        print(f"[Watch] reset lỗi (bỏ qua): {e}")

    # ============ 4) t_start ============
    now_utc_ms = int(time.time() * 1000)
    t_start_utc_ms = now_utc_ms + int(args.lead_sec * 1000)
    print(f"[Sync] Bắt đầu sau {args.lead_sec:.1f}s | "
          f"t_start_utc_ms = {t_start_utc_ms}")

    # ============ 5) Start watch ============
    try:
        print(f"[Watch] start → {watch.start(args.session, t_start_utc_ms)}")
    except Exception as e:
        print(f"[LỖI] start thất bại: {e}")
        cap.release(); writer.release()
        return

    # ============ 6) Chờ t_start ============
    while time.time() * 1000 < t_start_utc_ms:
        if t_start_utc_ms - time.time() * 1000 > 500:
            time.sleep(0.1)
        else:
            time.sleep(0.002)
    print("[Camera] ▶ BẮT ĐẦU GHI\n")

    # ============ 7) Camera thread — chỉ đọc + ghi ============
    stop_event = threading.Event()
    cam_duration = total_est + 60

    def cam_loop():
        frames = []
        t0 = time.time()
        last_log = t0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("\n[Camera] read() thất bại")
                break
            writer.write(frame)
            frames.append((len(frames), int(time.time() * 1000)))
            if time.time() - last_log >= 2.0:
                elapsed = time.time() - t0
                live_fps = len(frames) / elapsed if elapsed > 0 else 0
                print(f"[Camera] {len(frames)} frames | "
                      f"{live_fps:.1f} fps | {elapsed:.0f}s", end="\r")
                last_log = time.time()
            if duration_sec := (cam_duration):
                if (time.time() - t0) >= duration_sec:
                    break
        # Trả về frames list
        return frames

    cam_frames_holder = {}

    def cam_thread_fn():
        cam_frames_holder["frames"] = cam_loop()
        cam_frames_holder["t_end"] = time.time()

    cam_thread = threading.Thread(target=cam_thread_fn)
    cam_thread.start()

    # ============ 8) Action loop ============
    annotations = []
    session_start_utc_ms = int(time.time() * 1000)

    try:
        for aid, name, group in actions:
            for rep in range(1, args.reps + 1):
                print(f"\n[{aid:02d}|{group:8s}] {name} "
                      f"(lần {rep}/{args.reps})")
                print(f"  → Nghỉ {args.rest_duration:.0f}s.")
                watch.set_action(NO_ACTION_ID)
                time.sleep(args.rest_duration)

                watch.set_action(aid)
                start_utc_ms = int(time.time() * 1000)
                print(f"  ▶ BẮT ĐẦU {name.upper()} "
                      f"({args.action_duration:.0f}s)...")

                t_end = time.time() + args.action_duration
                while time.time() < t_end:
                    rem = t_end - time.time()
                    print(f"     ... {rem:4.1f}s   ", end="\r")
                    time.sleep(0.2)

                end_utc_ms = int(time.time() * 1000)
                print(f"\n  ■ KẾT THÚC {name} "
                      f"(dur={(end_utc_ms-start_utc_ms)/1000:.2f}s)")

                annotations.append({
                    "class_id":   aid,
                    "class_name": name,
                    "group":      group,
                    "rep":        rep,
                    "start_utc_ms": start_utc_ms,
                    "end_utc_ms":   end_utc_ms,
                })
    except KeyboardInterrupt:
        print("\n[!] Người dùng dừng giữa chừng.")
    finally:
        # ★ DỪNG WATCH TRƯỚC để cắt đuôi IMU
        print("\n[Watch] ■ DỪNG GHI (ưu tiên)")
        try:
            watch.stop()
        except Exception as e:
            print(f"[Watch] stop lỗi: {e}")
        try:
            watch.set_action(NO_ACTION_ID)
        except Exception:
            pass

        # Sau đó dừng camera
        print("[Camera] ■ DỪNG GHI")
        stop_event.set()
        cam_thread.join(timeout=5)

        # Giải phóng tài nguyên
        cap.release()
        writer.release()

        # Lấy frames và lưu CSV
        frames = cam_frames_holder.get("frames", [])
        total_dur = time.time() - session_start_utc_ms / 1000.0
        real_fps = len(frames) / total_dur if total_dur > 0 else 0
        print(f"\n[Camera] {len(frames)} frames / {total_dur:.1f}s = "
              f"{real_fps:.1f} fps (writer={use_fps})")
        print(f"[Camera] Video: {video_path}")

        frames_csv = out_dir / f"{args.session}_frames.csv"
        with open(frames_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "timestamp_utc_ms"])
            w.writerows(frames)
        print(f"[Camera] Timestamps: {frames_csv}")

        time.sleep(0.5)
        imu_server.close()

        # Annotation
        ann_path = out_dir / f"{args.session}_annotation.csv"
        with open(ann_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_id", "class_name", "group", "rep",
                        "start_time", "end_time",
                        "start_utc_ms", "end_utc_ms"])
            t0 = session_start_utc_ms
            for a in annotations:
                w.writerow([
                    a["class_id"], a["class_name"], a["group"], a["rep"],
                    (a["start_utc_ms"] - t0) / 1000.0,
                    (a["end_utc_ms"]   - t0) / 1000.0,
                    a["start_utc_ms"], a["end_utc_ms"],
                ])
        print(f"[Done] Annotation → {ann_path}")

        meta = {
            "session_id":         args.session,
            "t_start_utc_ms":     t_start_utc_ms,
            "t_start_iso":        datetime.fromtimestamp(
                                      t_start_utc_ms / 1000,
                                      tz=timezone.utc).isoformat(),
            "group":              args.group,
            "reps":               args.reps,
            "action_duration":    args.action_duration,
            "rest_duration":      args.rest_duration,
            "fps_target":         args.fps,
            "fps_actual":         use_fps,
            "resolution":         [aw, ah],
            "watch_url":          args.watch,
            "num_actions":        len(actions),
            "num_annotations":    len(annotations),
            "num_frames":         len(frames),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Done] Metadata → {out_dir}/metadata.json")


if __name__ == "__main__":
    main()