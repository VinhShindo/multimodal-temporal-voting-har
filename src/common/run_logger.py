"""
Run logger — quản lý folder run + log console/file.
Mỗi lần train tạo folder riêng: outputs/runs/{model}/{timestamp}/
"""
import csv
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_class_names(classes_yaml: str = "configs/classes_mmact.yaml") -> list:
    """Đọc tên 32 class theo thứ tự class_id."""
    with open(classes_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    id2name = {int(k): v for k, v in cfg["classes"].items()}
    n = int(cfg["num_classes"])
    return [id2name[i] for i in range(n)]


class RunLogger:
    """
    Tạo run folder + logging + save history.
    Usage:
        run = RunLogger("skeleton", config=cfg)
        run.info("Epoch 1...")
        run.save_history(history)
        run.copy_checkpoint("outputs/checkpoints/skeleton/best.pt")
    """

    def __init__(self, model_name: str,
                 output_root: str = "outputs/runs",
                 config: dict = None,
                 tag: str = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        self.run_name = f"{ts}{suffix}"
        self.model_name = model_name
        self.run_dir = Path(output_root) / model_name / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        if config is not None:
            self._save_config(config)

    def _setup_logging(self):
        self.logger = logging.getLogger(
            f"run_{self.model_name}_{self.run_name}"
        )
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.propagate = False

        # File handler
        fh = logging.FileHandler(self.run_dir / "log.txt", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s",
                                           "%H:%M:%S"))
        self.logger.addHandler(fh)

        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(ch)

    def _save_config(self, config: dict):
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    def info(self, msg: str = ""):
        self.logger.info(msg)

    def save_history(self, history: list):
        """history: list[dict] với keys nhất quán."""
        if not history:
            return
        keys = list(history[0].keys())
        path = self.run_dir / "history.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(history)

    def copy_checkpoint(self, src, name: str = "best.pt"):
        src = Path(src)
        if src.exists():
            shutil.copy2(src, self.run_dir / name)

    def save_best_from_model(self, model, epoch: int, best_f1: float,
                              arch: str = None, extra: dict = None,
                              name: str = "best.pt"):
        """Lưu checkpoint best với metadata đầy đủ."""
        import torch
        ckpt = {
            "epoch":      epoch,
            "best_f1":    float(best_f1),
            "model_state": model.state_dict(),
        }
        if arch:
            ckpt["arch"] = arch
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, self.run_dir / name)

    @property
    def path(self) -> Path:
        return self.run_dir


# ============================================================
# Timer tiện dụng
# ============================================================
class Timer:
    """Đo thời gian — dùng: with Timer() as t: ...; print(t.elapsed)"""
    def __enter__(self):
        import time
        self._t0 = time.time()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self._t0

    @property
    def elapsed_str(self) -> str:
        e = getattr(self, "elapsed", 0)
        if e < 60:
            return f"{e:.1f}s"
        m, s = divmod(e, 60)
        return f"{int(m)}m{int(s)}s"