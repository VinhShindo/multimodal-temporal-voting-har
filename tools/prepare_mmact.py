"""
prepare_mmact.py — Chuyển MMAct Challenge 2021 raw → processed NPY + manifest.

Raw layout (dưới data/raw/):
  trimmed_pose/pose/{split}/trainval/subject{N}/cam{N}/scene{N}/session{N}/{action}.json
  trimmed_sensor/sensor/acc_watch_clip/subject{N}/scene{N}/session{N}/{action}.csv
  trimmed_splits/splits/trimmed_x-{view,scene}_{train,val}.txt

Output:
  data/processed/MMAct/
  ├── skeleton/{split}/{train,val}/{clip_id}.npy     (T, 17, 3)  float32
  ├── imu/{split}/{train,val}/{clip_id}.npy          (T, C)     float32  @50Hz
  └── source_manifests/{train,val}_{split}.csv

Usage:
  python tools/prepare_mmact.py --split cross_view
  python tools/prepare_mmact.py --split cross_view --sensor acc_watch_clip
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============= CONSTANTS =============
RAW_ROOT = Path("data/raw")
RAW = (RAW_ROOT / "trimmed"
       if (RAW_ROOT / "trimmed" / "trimmed_splits").is_dir()
       else RAW_ROOT)
OUT = Path("data/processed/MMAct")
NUM_KP = 17
CLASSES = [
    # 32 classes gốc
    "carrying", "checking_time", "closing", "crouching", "drinking",
    "entering", "exiting", "fall", "jumping", "kicking", "loitering",
    "looking_around", "opening", "picking_up", "pocket_in", "pocket_out",
    "pointing", "pulling", "pushing", "running", "sitting", "sitting_down",
    "standing", "standing_up", "talking", "talking_on_phone", "throwing",
    "transferring_object", "using_pc", "using_phone", "walking", "waving_hand",
    # 3 classes mở rộng
    "carrying_heavy", "carrying_light", "setting_down",
]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)  # = 35


def normalize_action(name: str) -> str:
    """
    Normalize action name:
      - lowercase
      - strip whitespace
      - handle case inconsistency (Carrying_light → carrying_light)
    """
    a = name.lower().strip()
    # Fix typo: 'loitering' vs 'loitering' — keep as-is
    return a

HIP_L, HIP_R = 11, 12
SHO_L, SHO_R = 5, 6
TARGET_IMU_HZ = 50.0


# ============= HELPERS =============
def parse_split_file(path: Path) -> set:
    """Parse split txt → set of (subject, cam, scene, session)."""
    keys = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split("/")
        if len(parts) < 6:
            continue
        # cross_view/trainval/subject1/cam1/scene1/session1
        subj, cam, scene, sess = parts[2], parts[3], parts[4], parts[5]
        keys.add((subj, cam, scene, sess))
    return keys


def load_pose_json(path: Path) -> np.ndarray:
    """Đọc pose JSON → (T, 17, 3) float32."""
    d = json.loads(path.read_text(encoding="utf-8"))
    frames = sorted(d.keys())
    T = len(frames)
    kps = np.zeros((T, NUM_KP, 3), dtype=np.float32)

    for t, fk in enumerate(frames):
        persons = d[fk]
        if not persons:
            continue
        pkey = "person1" if "person1" in persons else next(iter(persons))
        flat = persons[pkey]
        arr = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
        n = min(NUM_KP, arr.shape[0])
        kps[t, :n] = arr[:n]
    return kps


def normalize_skeleton(kps: np.ndarray) -> np.ndarray:
    """Root-centered (hip center) + scale-normalized (shoulder distance)."""
    out = kps.copy()
    # Hip center làm gốc
    hip = (out[:, HIP_L, :2] + out[:, HIP_R, :2]) / 2.0    # (T, 2)
    out[:, :, 0] -= hip[:, 0:1]
    out[:, :, 1] -= hip[:, 1:2]

    # Scale theo khoảng cách 2 vai
    sho_dist = np.linalg.norm(out[:, SHO_L, :2] - out[:, SHO_R, :2],
                              axis=-1, keepdims=True)       # (T, 1)
    sho_dist = np.clip(sho_dist, 1e-3, None)
    out[:, :, :2] /= sho_dist[..., None]
    return out


def parse_sensor_csv(path: Path):
    """Đọc CSV → (ts_unix_ms, sig) với sig (T, C) hoặc (None, None) nếu lỗi."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                dt = datetime.strptime(parts[0], "%Y%m%d_%H:%M:%S.%f")
                vals = [float(x) for x in parts[1:]]
            except (ValueError, IndexError):
                continue
            rows.append((dt, vals))

    if not rows:
        return None, None

    ts = np.array([int(r[0].timestamp() * 1000) for r in rows], dtype=np.int64)
    sig = np.array([r[1] for r in rows], dtype=np.float32)
    return ts, sig


def resample_sensor(ts: np.ndarray, sig: np.ndarray,
                    target_hz: float = TARGET_IMU_HZ) -> np.ndarray:
    """Linear-interpolation resample về target_hz."""
    t0, t1 = ts[0], ts[-1]
    dur_s = (t1 - t0) / 1000.0
    if dur_s <= 0:
        return sig
    n = int(round(dur_s * target_hz)) + 1
    t_new = t0 + np.arange(n) * (1000.0 / target_hz)
    t_new = np.clip(t_new, t0, t1)

    out = np.zeros((n, sig.shape[1]), dtype=np.float32)
    for c in range(sig.shape[1]):
        out[:, c] = np.interp(t_new, ts, sig[:, c])
    return out


# ============= MAIN =============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["cross_view", "cross_scene"],
                    default="cross_view")
    ap.add_argument("--sensor", default="acc_watch_clip",
                    choices=["acc_watch_clip", "acc_phone_clip",
                             "gyro_clip", "orientation_clip"])
    ap.add_argument("--imu-hz", type=float, default=TARGET_IMU_HZ)
    ap.add_argument("--skip-skeleton", action="store_true")
    ap.add_argument("--skip-imu", action="store_true")
    args = ap.parse_args()

    SPLIT = args.split

    # ─── Load splits ───
    if SPLIT == "cross_view":
        train_txt = RAW / "trimmed_splits/splits/trimmed_x-view_train.txt"
        val_txt   = RAW / "trimmed_splits/splits/trimmed_x-view_val.txt"
    else:
        train_txt = RAW / "trimmed_splits/splits/trimmed_x-scene_train.txt"
        val_txt   = RAW / "trimmed_splits/splits/trimmed_x-scene_val.txt"

    if not train_txt.exists() or not val_txt.exists():
        raise FileNotFoundError(f"Không tìm thấy split file: {train_txt} / {val_txt}")

    train_keys = parse_split_file(train_txt)
    val_keys   = parse_split_file(val_txt)
    print(f"[Split] train sessions = {len(train_keys)}, val = {len(val_keys)}")

    # ─── Prep output dirs ───
    skel_dir = OUT / "skeleton" / SPLIT
    imu_dir  = OUT / "imu" / SPLIT
    man_dir  = OUT / "source_manifests"
    for sub in ["train", "val"]:
        (skel_dir / sub).mkdir(parents=True, exist_ok=True)
        (imu_dir  / sub).mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    # ─── Roots ───
    pose_root   = RAW / "trimmed_pose" / "pose" / SPLIT / "trainval"
    sensor_root = RAW / "trimmed_sensor" / "sensor" / args.sensor

    if not pose_root.exists():
        print(f"[WARN] Không thấy pose root: {pose_root}")
    if not sensor_root.exists():
        print(f"[WARN] Không thấy sensor root: {sensor_root}")

    manifests = {"train": [], "val": []}

    for split_name, keys in [("train", train_keys), ("val", val_keys)]:
        for (subj, cam, scene, sess) in tqdm(sorted(keys), desc=split_name):
            pose_dir = pose_root / subj / cam / scene / sess
            if not pose_dir.exists():
                continue

            for pose_json in sorted(pose_dir.glob("*.json")):
                action_raw = pose_json.stem
                action = normalize_action(action_raw)  # ← normalize
                if action not in CLASS2ID:
                    print(f"  ⚠️  Skip action: {action_raw} (không trong CLASS2ID)")
                    continue

                clip_id = f"{subj}_{cam}_{scene}_{sess}_{action}"
                class_id = CLASS2ID[action]

                # ─── Skeleton ───
                skel_path_str = ""
                num_frames = 0
                if not args.skip_skeleton:
                    kps = load_pose_json(pose_json)
                    if kps.shape[0] < 5:
                        continue
                    kps = normalize_skeleton(kps)
                    skel_path = skel_dir / split_name / f"{clip_id}.npy"
                    np.save(skel_path, kps)
                    skel_path_str = str(skel_path)
                    num_frames = int(kps.shape[0])

                # ─── IMU (không có cam trong path) ───
                imu_path_str = ""
                if not args.skip_imu:
                    sensor_csv = sensor_root / subj / scene / sess / f"{action_raw}.csv"
                    # Thử cả lowercase
                    if not sensor_csv.exists():
                        sensor_csv = sensor_root / subj / scene / sess / f"{action}.csv"
                    if sensor_csv.exists():
                        ts, sig = parse_sensor_csv(sensor_csv)
                        if ts is not None and len(ts) >= 5:
                            sig_rs = resample_sensor(ts, sig, args.imu_hz)
                            imu_clip_id = f"{subj}_{scene}_{sess}_{action}"
                            imu_path = imu_dir / split_name / f"{imu_clip_id}.npy"
                            np.save(imu_path, sig_rs)
                            imu_path_str = str(imu_path)

                manifests[split_name].append({
                    "clip_id":       clip_id,
                    "split":         SPLIT,
                    "subset":        split_name,
                    "subject":       subj,
                    "cam":           cam,
                    "scene":         scene,
                    "session":       sess,
                    "action":        action,
                    "class_id":      class_id,
                    "num_frames":    num_frames,
                    "skeleton_path": skel_path_str,
                    "imu_path":      imu_path_str,
                })

    # ─── Save manifests ───
    for split_name, rows in manifests.items():
        df = pd.DataFrame(rows)
        out_csv = man_dir / f"{split_name}_{SPLIT}.csv"
        df.to_csv(out_csv, index=False)
        n_sk = (df["skeleton_path"] != "").sum() if len(df) else 0
        n_imu = (df["imu_path"] != "").sum() if len(df) else 0
        print(f"[{split_name}] {len(df)} clips  |  "
              f"skeleton={n_sk}  imu={n_imu}  →  {out_csv}")

    print(f"\n✓ Done. Output tại {OUT}")


if __name__ == "__main__":
    main()