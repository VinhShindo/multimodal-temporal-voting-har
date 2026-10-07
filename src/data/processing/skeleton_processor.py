"""Skeleton processor — pose JSON → normalized NPY.

Pipeline (đã sửa):
  1. Parse JSON (dict frame_id → {person1: [x,y,c]×17})  → (T, 17, 3)
  2. Làm sạch: joint thiếu / conf thấp → nội suy theo thời gian
  3. Tâm = median hip của CẢ CHUỖI (giữ nguyên quỹ đạo di chuyển)
     (keep_trajectory=False → trừ hip từng frame như bản cũ, dùng để ablation)
  4. Scale = median độ dài thân (mid-shoulder → mid-hip) của cả chuỗi
  5. Clip outlier về [-clip_range, clip_range]
  6. Lưu float32 NPY (T, 17, 3) = (x, y, conf)

Lưu ý: dataset.py KHÔNG normalize lại nữa, nên phải chạy lại processor với force=True.
"""
from pathlib import Path

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.processing.base import (
    BaseProcessor, StatsCollector, save_npy, load_manifest, save_manifest,
)


# COCO 17 indices
HIP_L, HIP_R = 11, 12
SHO_L, SHO_R = 5, 6
NUM_KP = 17


class SkeletonProcessor(BaseProcessor):
    name = "skeleton"

    def __init__(self,
                 root_joints: tuple = (HIP_L, HIP_R),
                 scale_joints: tuple = (SHO_L, SHO_R),
                 clip_range: float = 10.0,
                 min_frames: int = 5,
                 conf_thr: float = 0.1,
                 keep_trajectory: bool = True):
        self.root_joints = tuple(root_joints)
        self.scale_joints = tuple(scale_joints)
        self.clip_range = clip_range
        self.min_frames = min_frames
        self.conf_thr = conf_thr
        self.keep_trajectory = keep_trajectory

    # ---------- Core ----------
    def load_pose_json(self, path: Path) -> np.ndarray | None:
        """Parse MMAct pose JSON → (T, 17, 3) float32."""
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not isinstance(d, dict) or not d:
            return None

        try:
            frames = sorted(d.keys())
        except Exception:
            return None

        T = len(frames)
        kps = np.zeros((T, NUM_KP, 3), dtype=np.float32)

        for t, fk in enumerate(frames):
            persons = d[fk]
            if not persons:
                continue
            pkey = "person1" if "person1" in persons else next(iter(persons))
            flat = persons[pkey]
            try:
                arr = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
            except Exception:
                continue
            n = min(NUM_KP, arr.shape[0])
            kps[t, :n] = arr[:n]

        return kps

    def normalize(self, kps: np.ndarray) -> np.ndarray | None:
        """Nội suy joint thiếu → center/scale theo cả chuỗi → clip. Trả None nếu pose không dùng được."""
        xy = kps[..., :2].astype(np.float32).copy()
        conf = kps[..., 2].astype(np.float32).copy()
        T = len(xy)
        t = np.arange(T)

        # Joint hợp lệ: toạ độ khác (0,0) và (nếu file có conf) conf > ngưỡng
        valid = np.abs(xy).sum(-1) > 0
        if conf.max() > 0:                       # an toàn: nếu conf toàn 0 thì bỏ qua conf
            valid &= conf > self.conf_thr

        never = np.zeros(NUM_KP, dtype=bool)     # joint không bao giờ hợp lệ
        for v in range(NUM_KP):
            ok = valid[:, v]
            n_ok = int(ok.sum())
            if n_ok == 0:
                never[v] = True
                continue
            if n_ok < T:
                for dim in range(2):
                    xy[:, v, dim] = np.interp(t, t[ok], xy[ok, v, dim])

        core = list(self.root_joints) + list(self.scale_joints)
        if never[core].any():
            return None

        mid_hip = xy[:, list(self.root_joints)].mean(axis=1)       # (T, 2)
        mid_sho = xy[:, list(self.scale_joints)].mean(axis=1)      # (T, 2)
        torso = np.linalg.norm(mid_sho - mid_hip, axis=-1)
        good = torso > 1e-3
        if not good.any():
            return None
        scale = max(float(np.median(torso[good])), 1e-3)

        if self.keep_trajectory:
            origin = np.median(mid_hip, axis=0, keepdims=True)     # (1, 2) cố định cả chuỗi
        else:
            origin = mid_hip                                       # (T, 2) như bản cũ
        xy = (xy - origin[:, None, :]) / scale

        if self.clip_range > 0:
            xy = np.clip(xy, -self.clip_range, self.clip_range)

        xy[:, never] = 0.0
        conf[:, never] = 0.0

        out = np.concatenate([xy, conf[..., None]], axis=-1)
        return out.astype(np.float32)

    def process_one(self, raw_path: Path, out_path: Path) -> dict:
        kps = self.load_pose_json(raw_path)
        if kps is None:
            return {"ok": False, "reason": "json_load_failed"}
        if kps.shape[0] < self.min_frames:
            return {"ok": False, "reason": f"too_short ({kps.shape[0]})"}

        kps = self.normalize(kps)
        if kps is None:
            return {"ok": False, "reason": "no_valid_pose"}
        save_npy(kps, out_path)
        return {"ok": True, "shape": kps.shape}

    # ---------- Manifest ----------
    def process_manifest(self, manifest_csv: Path,
                         out_dir: Path,
                         split: str,
                         force: bool = False,
                         log_fn=None) -> tuple[pd.DataFrame, dict]:
        def log(m):
            (log_fn or print)(m)

        df = load_manifest(manifest_csv)
        log(f"  Loaded manifest: {len(df)} rows")

        stats = StatsCollector()
        new_paths = []

        for _, row in tqdm(df.iterrows(),
                            desc=f"  skeleton/{split}",
                            disable=log_fn is None):
            raw = row.get("pose_path", None)
            if raw is None or pd.isna(raw) or raw == "":
                raw = self._build_raw_path(row)
            raw = Path(raw)
            out = self.build_output_path(row, out_dir, split)

            if not force and out.exists() and out.stat().st_size > 0:
                new_paths.append(str(out))
                stats.record_ok((0,))
                continue

            if not raw.exists():
                new_paths.append("")
                stats.record_skip("missing", str(raw))
                continue

            try:
                res = self.process_one(raw, out)
                if res.get("ok"):
                    new_paths.append(str(out))
                    stats.record_ok(res["shape"])
                else:
                    new_paths.append("")
                    stats.record_skip("bad", f"{row['clip_id']}: {res['reason']}")
            except Exception as e:
                new_paths.append("")
                stats.record_error(f"{row['clip_id']}: {e}")

        df = df.copy()
        df["skeleton_path"] = new_paths
        save_manifest(df, manifest_csv)

        summary = stats.summary()
        log(f"  → OK={summary['ok']}  skip_missing={summary['skip_missing']}  "
            f"skip_bad={summary['skip_bad']}")
        return df, summary

    @staticmethod
    def _build_raw_path(row: pd.Series) -> str:
        root = "data/raw/trimmed_pose/pose/cross_view/trainval"
        return (f"{root}/{row['subject']}/{row['cam']}/{row['scene']}/"
                f"{row['session']}/{row['action']}.json")