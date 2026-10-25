"""Chạy continuous trên tất cả sessions (bước 20)."""
import argparse
import csv as _csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.run_logger import RunLogger, Timer, load_class_names
from src.inference.predictor import ContinuousPredictor
from src.inference.continuous_runner import run_continuous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-root", default="data/processed/MMAct/continuous")
    ap.add_argument("--runs-root", default="outputs/runs")
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--step-sec", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--imu-hz", type=float, default=50.0)
    ap.add_argument("--smooth-kernel", type=int, default=5)
    ap.add_argument("--t-avg", type=float, default=0.8)
    ap.add_argument("--min-duration", type=float, default=0.3)
    ap.add_argument("--no-boundary", action="store_true")
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = {"sessions_root": args.sessions_root, "window_sec": args.window_sec,
           "step_sec": args.step_sec, "smooth_kernel": args.smooth_kernel,
           "t_avg": args.t_avg, "min_duration": args.min_duration,
           "use_boundary": not args.no_boundary}
    run = RunLogger("continuous", config=cfg, tag=args.tag)
    run.info("=" * 70)
    run.info(f"  CONTINUOUS — {run.run_name}")
    run.info("=" * 70)
    run.info(f"  Sessions:  {args.sessions_root}")
    run.info(f"  Window:    {args.window_sec}s × {args.step_sec}s")
    run.info(f"  Boundary:  {not args.no_boundary}")
    run.info(f"  Run dir:   {run.path}")
    run.info("")

    class_names = load_class_names(args.classes_cfg)

    run.info("Loading models...")
    predictor = ContinuousPredictor(runs_root=args.runs_root, device=args.device)
    run.info("")

    sessions_root = Path(args.sessions_root)
    session_dirs = sorted([d for d in sessions_root.iterdir()
                            if d.is_dir() and (d / "video.mp4").exists()])
    if args.max_sessions:
        session_dirs = session_dirs[:args.max_sessions]
    run.info(f"Sessions: {len(session_dirs)}\n")

    all_results = {}
    for sdir in session_dirs:
        run.info("=" * 70)
        run.info(f"  {sdir.name}")
        run.info("=" * 70)
        with Timer() as t:
            try:
                res = run_continuous(
                    session_dir=sdir, predictor=predictor, class_names=class_names,
                    window_sec=args.window_sec, step_sec=args.step_sec,
                    fps=args.fps, imu_hz=args.imu_hz,
                    use_boundary=not args.no_boundary,
                    smooth_kernel=args.smooth_kernel,
                    t_avg=args.t_avg, min_duration=args.min_duration,
                    log_fn=run.info,
                )
            except Exception as e:
                run.info(f"  ✗ FAILED: {e}")
                import traceback; run.info(traceback.format_exc())
                continue

        run.info(f"  → {len(res['segments_final'])} segments  ({t.elapsed_str})")

        out_dir = run.path / sdir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "windows.csv", "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["window_id", "start", "end", "class_id", "confidence"])
            for p in res["windows"]:
                w.writerow([p["window_id"], p["start"], p["end"],
                             p["class_id"], p["confidence"]])

        with open(out_dir / "segments.csv", "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["start", "end", "class_id", "class_name", "confidence"])
            for s in res["segments_final"]:
                w.writerow([s.start, s.end, s.class_id, s.class_name, s.confidence])

        with open(out_dir / "boundaries.json", "w", encoding="utf-8") as f:
            json.dump({"predicted": res["boundaries"]}, f, indent=2)

        all_results[sdir.name] = {
            "n_windows": len(res["windows"]),
            "n_boundaries": len(res["boundaries"]),
            "n_segments_raw": len(res["segments_raw"]),
            "n_segments_final": len(res["segments_final"]),
            "duration": res["duration"],
        }

    (run.path / "all_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8")
    run.info("")
    run.info("=" * 70)
    run.info(f"  DONE. {len(all_results)}/{len(session_dirs)} sessions OK")
    run.info(f"  Run dir: {run.path}")


if __name__ == "__main__":
    main()