"""
prepare_rgb_mmact.py — Thêm cột rgb_path vào manifest skeleton/IMU.

Usage:
    python tools/prepare_rgb_mmact.py
    python tools/prepare_rgb_mmact.py --verify-only
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================
# PATTERN ĐÚNG (đã verify từ check_rgb_structure)
# ============================================================
TRAIN_VIDEO_TPL = (
    "data/raw/cross_view_video/"
    "trimmed_x-view_{subj}/video/cross_view/trainval/"
    "{subj}/{cam}/{scene}/{session}/{action}.mp4"
)

MANIFEST_DIR = "data/processed/MMAct/manifests"


def resolve_train_rgb(row, root: Path):
    """Tìm video RGB tương ứng với 1 manifest row."""
    subj = row["subject"]          # "subject1"
    cam = row["cam"]               # "cam1"
    scene = row["scene"]           # "scene1"
    session = row["session"]       # "session1"
    action = row["action"]         # "carrying"

    rel_path = TRAIN_VIDEO_TPL.format(
        subj=subj, cam=cam, scene=scene, session=session, action=action
    )
    p = root / rel_path
    if p.is_file():
        return str(p)

    if p.parent.is_dir():
        match = next((candidate for candidate in p.parent.iterdir()
                      if candidate.is_file()
                      and candidate.suffix.lower() == p.suffix.lower()
                      and candidate.stem.casefold() == p.stem.casefold()), None)
        if match is not None:
            return str(match)
    return ""


def process_manifest(csv_path: Path, root: Path, verify_only: bool = False):
    print(f"\n{'='*70}")
    print(f"  {csv_path.name}")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")

    if "rgb_path" in df.columns and verify_only:
        paths = df["rgb_path"].fillna("").astype(str)
        exists = paths.map(lambda value: Path(value).is_file() if value else False)
        print(f"  RGB file tồn tại: {exists.sum()}/{len(df)}")
        return

    # Resolve từng row
    rgb_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="resolving"):
        rgb_paths.append(resolve_train_rgb(row, root))

    df["rgb_path"] = rgb_paths

    # Stats
    n_ok = sum(1 for p in rgb_paths if p)
    n_missing = len(df) - n_ok
    pct = 100.0 * n_ok / len(df) if len(df) else 0

    print(f"  ✓ Found:   {n_ok}/{len(df)}  ({pct:.1f}%)")
    print(f"  ✗ Missing: {n_missing}/{len(df)}")

    # Sample missing
    if n_missing > 0:
        missing = df[df["rgb_path"] == ""].head(5)
        print(f"\n  Ví dụ missing (5 dòng đầu):")
        for _, r in missing.iterrows():
            print(f"    · {r['clip_id']}")

    # Save
    if not verify_only:
        df.to_csv(csv_path, index=False)
        print(f"\n  ✓ Đã lưu: {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="Chỉ kiểm tra, không ghi lại manifest")
    args = ap.parse_args()

    root = Path.cwd()
    if not (root / "src").exists():
        print("✗ Chạy script từ project root.")
        sys.exit(1)

    mf_dir = root / MANIFEST_DIR
    if not mf_dir.exists():
        print(f"✗ Không có {mf_dir}")
        sys.exit(1)

    # Backup manifest cũ
    if not args.verify_only:
        backup_dir = mf_dir / "_backup"
        backup_dir.mkdir(exist_ok=True)
        for csv in mf_dir.glob("*.csv"):
            backup = backup_dir / csv.name
            if not backup.exists():
                csv.rename(backup)  # chuyển cũ vào backup
                csv.write_text(backup.read_text(), encoding="utf-8")
        print(f"✓ Backup manifest cũ → {backup_dir.relative_to(root)}")

    # Process cả 2 manifest
    for csv_name in ["train_cross_view.csv", "val_cross_view.csv"]:
        csv_path = mf_dir / csv_name
        if not csv_path.exists():
            print(f"⚠️  Bỏ qua: {csv_name} không tồn tại")
            continue
        process_manifest(csv_path, root, args.verify_only)

    # Verify tổng kết
    print(f"\n{'='*70}")
    print("  KẾT LUẬN")
    print(f"{'='*70}")

    for csv_name in ["train_cross_view.csv", "val_cross_view.csv"]:
        csv_path = mf_dir / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "rgb_path" not in df.columns:
            print(f"  {csv_name}: chưa có cột rgb_path")
            continue
        paths = df["rgb_path"].fillna("").astype(str)
        n_ok = paths.map(lambda value: Path(value).is_file() if value else False).sum()
        print(f"  {csv_name}: {n_ok}/{len(df)} clip có RGB")

    print()
    print("  → Sẵn sàng train RGB model!")


if __name__ == "__main__":
    main()