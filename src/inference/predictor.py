"""
Load 3 model + chạy inference + fusion.

★ FIXED:
  1. IMU dùng global stats (giống training) thay vì per-window stats
  2. RGB resize 224 (đúng với train) thay vì 112
  3. Fallback an toàn nếu load stats fail
"""
from pathlib import Path

import numpy as np
import torch


class ContinuousPredictor:
    """Load 3 best.pt từ outputs/runs/, predict on window."""

    def __init__(self, runs_root="outputs/runs", device="cuda",
                 train_manifest="data/processed/MMAct/manifests/train_cross_view.csv"):
        self.device = device
        self.runs_root = Path(runs_root)
        self.train_manifest = train_manifest
        self.models = {}

        self._load_skeleton()
        self._load_imu()
        self._load_rgb()

    # ============================================================
    # Load models
    # ============================================================
    def _load_skeleton(self):
        try:
            from src.models.skeleton.ctrgcn import build_ctrgcn

            runs_root = self.runs_root / "skeleton"
            if not runs_root.exists():
                print("[Pred] Skeleton: không có run dir")
                return

            runs = sorted(runs_root.iterdir(),
                          key=lambda d: d.stat().st_mtime, reverse=True)
            run_dir = next(r for r in runs if (r / "best.pt").exists())

            ckpt = torch.load(run_dir / "best.pt",
                              map_location=self.device, weights_only=False)
            in_ch = ckpt.get("in_channels", 6)
            stream = ckpt.get("stream", "joint_bone_motion")

            m = build_ctrgcn(num_classes=32, num_joints=17, in_channels=in_ch)
            m = m.to(self.device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()

            self.models["skeleton"] = {
                "model": m,
                "in_ch": in_ch,
                "stream": stream,
                "run_dir": run_dir,
            }
            print(f"[Pred] Skeleton: {run_dir.name}  "
                  f"(in_ch={in_ch}, stream={stream})")
        except Exception as e:
            print(f"[Pred] Skeleton load failed: {e}")

    def _load_imu(self):
        try:
            from src.models.imu.cnn_bilstm import CNNBiLSTM

            runs_root = self.runs_root / "imu"
            if not runs_root.exists():
                print("[Pred] IMU: không có run dir")
                return

            runs = sorted(runs_root.iterdir(),
                          key=lambda d: d.stat().st_mtime, reverse=True)
            run_dir = next(r for r in runs if (r / "best.pt").exists())

            ckpt = torch.load(run_dir / "best.pt",
                              map_location=self.device, weights_only=False)
            # Checkpoint có thể là state_dict trực tiếp hoặc có wrapper
            state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

            m = CNNBiLSTM(in_channels=3, num_classes=35, dropout=0.3)
            m = m.to(self.device)
            m.load_state_dict(state)
            m.eval()

            self.models["imu"] = {
                "model": m,
                "run_dir": run_dir,
                "mu": None,
                "std": None,
            }
            print(f"[Pred] IMU: {run_dir.name}")

            # ★ Load global IMU stats từ train manifest
            try:
                from src.data.imu.dataset import compute_stats

                manifest = Path(self.train_manifest)
                if manifest.exists():
                    stats = compute_stats(str(manifest))
                    self.models["imu"]["mu"] = stats["mu"]
                    self.models["imu"]["std"] = stats["std"]
                    print(f"[Pred] IMU stats loaded: "
                          f"mu={stats['mu']}, std={stats['std']}")
                else:
                    print(f"[Pred] IMU stats: manifest not found {manifest}, "
                          f"dùng per-window fallback")
            except Exception as e:
                print(f"[Pred] IMU stats load failed: {e}, "
                      f"dùng per-window fallback")
        except Exception as e:
            print(f"[Pred] IMU load failed: {e}")

    def _load_rgb(self):
        try:
            from src.models.rgb.video_backbone import build_video_model

            runs_root = self.runs_root / "rgb"
            if not runs_root.exists():
                print("[Pred] RGB: không có run dir")
                return

            runs = sorted(runs_root.iterdir(),
                          key=lambda d: d.stat().st_mtime, reverse=True)
            run_dir = next(r for r in runs if (r / "best.pt").exists())

            ckpt = torch.load(run_dir / "best.pt",
                              map_location=self.device, weights_only=False)
            arch = ckpt.get("arch", "mc3_18")

            m = build_video_model(arch=arch, num_classes=32,
                                  pretrained=False, dropout=0.3)
            m = m.to(self.device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()

            self.models["rgb"] = {
                "model": m,
                "arch": arch,
                "run_dir": run_dir,
            }
            print(f"[Pred] RGB: {run_dir.name}  ({arch})")
        except Exception as e:
            print(f"[Pred] RGB load failed: {e}")

    # ============================================================
    # Predict
    # ============================================================
    @torch.no_grad()
    def predict_skeleton(self, skel_w):
        """skel_w: (T, 17, 3) → probs (32,)."""
        if "skeleton" not in self.models:
            return None
        if skel_w.shape[0] < 5:
            return None

        e = self.models["skeleton"]
        m = e["model"]
        stream = e["stream"]

        kps = self._normalize_skel(skel_w)
        joint = kps[:, :, :2]

        streams = [joint]
        if "bone" in stream:
            streams.append(self._bone(joint))
        if "motion" in stream:
            streams.append(self._motion(joint))

        feat = np.concatenate(streams, axis=-1)   # (T, 17, C)
        feat = self._resample_t(feat, 64)          # (64, 17, C)

        tensor = torch.from_numpy(feat).float()    # (64, 17, C)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)   # (1, C, 64, 17)
        tensor = tensor.to(self.device)

        logits = m(tensor)
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    @torch.no_grad()
    def predict_imu(self, imu_w):
        """imu_w: (T, 3) → probs (32,)."""
        if "imu" not in self.models:
            return None
        if imu_w.shape[0] < 5:
            return None

        m = self.models["imu"]["model"]

        # Lấy mu/sd trước để pad với mean (không phải zeros)
        mu = self.models["imu"].get("mu")
        sd = self.models["imu"].get("std")

        target = 200
        T = imu_w.shape[0]

        if T < target:
            # ★ Pad SYMMETRIC với mu (không phải zeros)
            if mu is None:
                mu_pad = imu_w.mean(axis=0, keepdims=True)
            else:
                mu_pad = mu

            total_pad = target - T
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            pad_l = np.broadcast_to(mu_pad, (pad_left, imu_w.shape[1]))
            pad_r = np.broadcast_to(mu_pad, (pad_right, imu_w.shape[1]))
            imu_w = np.concatenate([pad_l, imu_w, pad_r], axis=0)
        elif T > target:
            # ★ Center crop
            s = (T - target) // 2
            imu_w = imu_w[s:s + target]

        # Normalize
        if mu is None or sd is None:
            mu = imu_w.mean(axis=0, keepdims=True)
            sd = imu_w.std(axis=0, keepdims=True) + 1e-6

        sig = (imu_w - mu) / np.clip(sd, 1e-6, None)
        sig = sig.astype(np.float32)

        tensor = torch.from_numpy(sig).float().permute(1, 0).unsqueeze(0)
        tensor = tensor.to(self.device)

        logits = m(tensor)
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    @torch.no_grad()
    def predict_rgb(self, frames):
        """frames: (T, H, W, 3) uint8 BGR → probs (32,).

        ★ FIX: size=224 (đúng với train), trước là 112.
        """
        if "rgb" not in self.models:
            return None
        if frames is None or len(frames) < 2:
            return None

        import cv2
        m = self.models["rgb"]["model"]

        T_target = 8
        size = 224   # ★ 112 → 224

        idx = np.linspace(0, len(frames) - 1, T_target).astype(int)
        sampled = frames[idx]

        proc = []
        for f in sampled:
            # frames từ LazyVideoReader là BGR
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if f.shape[0] != size or f.shape[1] != size:
                f = cv2.resize(f, (size, size),
                               interpolation=cv2.INTER_LINEAR)
            proc.append(f)

        arr = np.stack(proc, axis=0).astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        tensor = torch.from_numpy(arr).float()   # (T, H, W, 3)
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)   # (1, T, 3, H, W)
        tensor = tensor.to(self.device)

        logits = m(tensor)
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # ============================================================
    # Skeleton helpers
    # ============================================================
    @staticmethod
    def _normalize_skel(kps):
        """Hip-center + shoulder-scale + clip — giống training."""
        out = kps.copy()
        hip = (out[:, 11, :2] + out[:, 12, :2]) / 2.0
        out[:, :, 0] -= hip[:, 0:1]
        out[:, :, 1] -= hip[:, 1:2]

        sho = np.linalg.norm(out[:, 5, :2] - out[:, 6, :2],
                             axis=-1, keepdims=True)
        sho = np.clip(sho, 1e-3, None)
        out[:, :, :2] /= sho[..., None]
        out[:, :, :2] = np.clip(out[:, :, :2], -3.0, 3.0)
        return out.astype(np.float32)

    @staticmethod
    def _bone(joint):
        """Tính bone vector từ joint-parent."""
        parent = [-1, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
        b = np.zeros_like(joint)
        for v in range(joint.shape[1]):
            p = parent[v]
            if p >= 0:
                b[:, v] = joint[:, v] - joint[:, p]
        return b

    @staticmethod
    def _motion(joint):
        """Temporal diff."""
        m = np.zeros_like(joint)
        m[1:] = joint[1:] - joint[:-1]
        return m

    @staticmethod
    def _resample_t(arr, T_new):
        """Resample time dimension (T_old → T_new) linear."""
        T_old = arr.shape[0]
        if T_old == T_new:
            return arr
        idx = np.linspace(0, T_old - 1, T_new)
        lo = np.floor(idx).astype(int)
        hi = np.clip(lo + 1, 0, T_old - 1)
        w = (idx - lo).reshape(-1, *([1] * (arr.ndim - 1)))
        return (arr[lo] * (1 - w) + arr[hi] * w).astype(np.float32)


# ============================================================
# Fusion
# ============================================================
def fuse_probs(probs_dict, weights=None):
    """probs_dict: {"skeleton": (32,), "imu": (32,), "rgb": (32,)}."""
    if weights is None:
        weights = {"skeleton": 0.2, "imu": 0.5, "rgb": 0.3}

    fused = None
    total_w = 0.0
    for mod, p in probs_dict.items():
        if p is None:
            continue
        w = weights.get(mod, 0.0)
        if w <= 0:
            continue
        if fused is None:
            fused = np.zeros_like(p)
        fused += w * p
        total_w += w

    if fused is None or total_w <= 0:
        return None
    return fused / total_w