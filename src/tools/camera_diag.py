"""
camera_diag.py — Chẩn đoán camera: thử nhiều backend + resolution,
                 đo FPS thực cho từng cấu hình, tìm cấu hình tối ưu.

Chạy:
    python tools/camera_diag.py
"""
import cv2
import time
import sys


BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF",  cv2.CAP_MSMF),
    ("ANY",   cv2.CAP_ANY),
]

# (width, height) — thử từ cao xuống thấp
RESOLUTIONS = [
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (848, 480),
    (640, 480),
    (640, 360),
]

TARGET_FPS = 30
MEASURE_SEC = 2.0


def fourcc_str(v: int) -> str:
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def measure(cap, duration=MEASURE_SEC):
    """Đo FPS thực + kiểm tra frame có hợp lệ."""
    t0 = time.time()
    n = 0
    fails = 0
    while time.time() - t0 < duration:
        ret, frame = cap.read()
        if ret and frame is not None:
            n += 1
        else:
            fails += 1
    elapsed = time.time() - t0
    return n / elapsed if elapsed > 0 else 0, n, fails


def try_config(backend_name: str, backend_id: int,
               width: int, height: int, codec: str = "MJPG"):
    print(f"\n--- {backend_name} | {width}x{height} | {codec} ---")
    cap = cv2.VideoCapture(0, backend_id)
    if not cap.isOpened():
        print("  ❌ Không mở được")
        return None

    # Thử set theo nhiều thứ tự khác nhau
    # Order 1: FOURCC trước, resolution sau, FPS cuối
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # Đọc lại
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)
    fcc = fourcc_str(int(cap.get(cv2.CAP_PROP_FOURCC)))

    print(f"  Reported: {aw}x{ah} @ {afps:.1f} fps, codec={fcc}")

    # Warmup — bỏ 5 frame đầu
    for _ in range(5):
        cap.read()

    real_fps, n, fails = measure(cap)
    cap.release()

    status = "✅" if real_fps >= 25 else ("⚠️" if real_fps >= 15 else "❌")
    print(f"  {status} Real FPS: {real_fps:.1f}  "
          f"(frames={n}, fails={fails})")
    return real_fps


def main():
    print("=" * 60)
    print("CAMERA DIAGNOSTIC")
    print("=" * 60)

    results = []

    # Pass 1: DSHOW, thử từng resolution
    for w, h in RESOLUTIONS:
        fps = try_config("DSHOW", cv2.CAP_DSHOW, w, h, "MJPG")
        if fps is not None:
            results.append((f"DSHOW-{w}x{h}-MJPG", fps))
        if fps and fps >= 25:
            print(f"\n✅ Đã tìm được cấu hình tốt: DSHOW {w}x{h}")
            break

    # Pass 2: Nếu DSHOW không đủ, thử MSMF ở 720p
    if not any("DSHOW" in r[0] and r[1] >= 25 for r in results):
        print("\n[!] DSHOW không đạt. Thử MSMF...")
        for w, h in [(1280, 720), (960, 540), (640, 480)]:
            fps = try_config("MSMF", cv2.CAP_MSMF, w, h, "MJPG")
            if fps is not None:
                results.append((f"MSMF-{w}x{h}-MJPG", fps))
            if fps and fps >= 25:
                print(f"\n✅ MSMF {w}x{h} đạt")
                break

    # Pass 3: Thử YUYV ở resolution thấp (một số cam thích YUYV hơn)
    print("\n[!] Thử YUYV ở resolution thấp...")
    for w, h in [(640, 480), (320, 240)]:
        fps = try_config("DSHOW", cv2.CAP_DSHOW, w, h, "YUYV")
        if fps is not None:
            results.append((f"DSHOW-{w}x{h}-YUYV", fps))

    # Tổng kết
    print("\n" + "=" * 60)
    print("TỔNG KẾT")
    print("=" * 60)
    results.sort(key=lambda x: -x[1])
    for name, fps in results:
        print(f"  {fps:5.1f} fps  {name}")

    if not results or results[0][1] < 25:
        print("\n❌ Không có cấu hình nào đạt ≥ 25 FPS.")
        print("\nNguyên nhân có thể:")
        print("  1. Webcam phần cứng chỉ hỗ trợ 10-15 FPS ở 720p")
        print("  2. Webcam đang cắm vào USB hub (không đủ băng thông)")
        print("  3. Webcam cũ/giá rẻ giới hạn FPS")
        print("  4. Driver webcam cần update")
        print("\n→ Thử: cắm trực tiếp vào USB 3.0 port (màu xanh)")
        print("→ Hoặc chấp nhận 720p@15fps, hoặc hạ xuống 640x480@30fps")
        sys.exit(1)

    print(f"\n✅ Khuyến nghị: {results[0][0]}")
    print(f"   → Sửa collector.py dùng cấu hình này")


if __name__ == "__main__":
    main()