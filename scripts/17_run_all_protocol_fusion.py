"""Generate OOF predictions and run fusion for all MMAct protocols."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.manifest_paths import MM_ACT_PROTOCOLS


ROOT = Path(__file__).resolve().parent.parent
MODALITIES = ("skeleton", "imu", "rgb")
PREDICTION_FILES = (
    "oof_train_probs.npy",
    "oof_train_labels.npy",
    "oof_train_clip_ids.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OOF and benchmark fusion across MMAct protocols"
    )
    parser.add_argument("--config", default="configs/fusion/voting.yaml")
    parser.add_argument(
        "--protocol",
        choices=("all", *MM_ACT_PROTOCOLS),
        default="all",
        help="Run all six protocols by default, or select one protocol.",
    )
    parser.add_argument(
        "--stage",
        choices=("both", "oof", "fusion"),
        default="both",
        help="Run OOF, fusion, or both stages in order.",
    )
    parser.add_argument("--benchmark-root", default="outputs/benchmark")
    parser.add_argument("--runs-root", default="outputs/runs")
    parser.add_argument("--oof-root", default="outputs/oof")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--rgb-epochs",
        type=int,
        default=None,
        help="Override RGB training epochs for each OOF fold.",
    )
    parser.add_argument(
        "--allow-incomplete-audit",
        action="store_true",
        help="Explicitly allow the current manifest audit status even if it "
             "is not PASS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve checkpoints and print commands without training.",
    )
    parser.add_argument(
        "--force-oof",
        action="store_true",
        help="Regenerate OOF even when all expected artifacts already exist.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def selected_architectures(config: dict) -> dict[str, str]:
    configured = config.get("fusion", {}).get("architectures", {})
    selected = {}
    for modality in MODALITIES:
        architectures = configured.get(modality, [])
        if isinstance(architectures, str):
            architectures = [architectures]
        if len(architectures) != 1:
            raise ValueError(
                f"Expected exactly one selected {modality} architecture in "
                f"fusion config; found {architectures!r}"
            )
        selected[modality] = str(architectures[0])
    return selected


def select_run_dir(
    benchmark_root: Path,
    runs_root: Path,
    protocol: str,
    modality: str,
    architecture: str,
) -> Path:
    reports = sorted(
        (benchmark_root / modality).glob(f"{protocol}_*/results.csv"),
        key=lambda item: item.stat().st_mtime,
    )
    if not reports:
        raise FileNotFoundError(
            f"No benchmark results for {protocol}/{modality} under "
            f"{benchmark_root / modality}"
        )
    latest_report = None
    rows = []
    for report in reversed(reports):
        with report.open(newline="", encoding="utf-8") as stream:
            matching_rows = [
                row for row in csv.DictReader(stream)
                if row.get("architecture") == architecture
                and row.get("split") == protocol
            ]
        if matching_rows:
            latest_report = report
            rows = matching_rows
            break
    if latest_report is None or len(rows) != 1:
        raise ValueError(
            f"Expected one {architecture} result for {protocol} in "
            f"the available benchmark reports, found {len(rows)} in "
            f"{latest_report}"
        )
    raw_run_dir = Path(rows[0]["run_dir"])
    run_dir = raw_run_dir if raw_run_dir.is_absolute() else ROOT / raw_run_dir
    try:
        run_dir.relative_to(runs_root)
    except ValueError as error:
        raise ValueError(
            f"Selected run {run_dir} is outside configured runs root {runs_root}"
        ) from error
    required = ("val_probs.npy", "val_labels.npy")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Selected run {run_dir} is missing required artifacts: {missing}"
        )
    return run_dir


def audit_status(manifest_root: Path) -> str | None:
    audit = manifest_root / "protocol_manifest_audit.json"
    if not audit.is_file():
        return None
    import json

    return str(json.loads(audit.read_text(encoding="utf-8")).get("status"))


def run_command(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.rgb_epochs is not None and args.rgb_epochs < 1:
        raise ValueError("--rgb-epochs must be positive")

    config_path = resolve_path(args.config)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    architectures = selected_architectures(config)
    paths = config.get("paths", {}) or {}
    manifest_root = resolve_path(
        paths.get(
            "manifest_root",
            "data/processed/MMAct/manifests/protocols",
        )
    )
    benchmark_root = resolve_path(args.benchmark_root)
    runs_root = resolve_path(args.runs_root)
    oof_root = resolve_path(args.oof_root)
    protocols = MM_ACT_PROTOCOLS if args.protocol == "all" else (args.protocol,)

    plan = []
    for protocol in protocols:
        protocol_manifests = manifest_root / protocol
        for split in ("train", "val"):
            for modality in (*MODALITIES, "alignment"):
                manifest = protocol_manifests / modality / f"{split}.csv"
                if not manifest.is_file():
                    raise FileNotFoundError(f"Missing protocol manifest: {manifest}")
        status = audit_status(manifest_root)
        if status and status != "PASS" and not args.allow_incomplete_audit:
            raise RuntimeError(
                f"Manifest audit status is {status}. Verify upload completeness "
                "or pass --allow-incomplete-audit to proceed knowingly."
            )
        if status and status != "PASS":
            print(
                f"WARNING: {protocol} will run with manifest audit status "
                f"{status} by explicit request.",
                flush=True,
            )
        run_dirs = {
            modality: select_run_dir(
                benchmark_root, runs_root, protocol, modality,
                architectures[modality],
            )
            for modality in MODALITIES
        }
        plan.append((protocol, run_dirs))

    if args.stage == "fusion" and not args.dry_run:
        incomplete = []
        for protocol, run_dirs in plan:
            missing = [
                f"{modality}/{filename}"
                for modality in MODALITIES
                for filename in PREDICTION_FILES
                if not (run_dirs[modality] / filename).is_file()
            ]
            if missing:
                incomplete.append(f"{protocol}: {', '.join(missing)}")
        if incomplete:
            raise FileNotFoundError(
                "Fusion-only requires complete OOF artifacts for every "
                "selected protocol:\n" + "\n".join(incomplete)
            )

    for protocol, run_dirs in plan:
        print(f"\n===== {protocol} =====", flush=True)
        for modality in MODALITIES:
            print(f"{modality}: {run_dirs[modality]}", flush=True)

        oof_dir = oof_root / protocol
        oof_complete = all(
            (run_dirs[modality] / filename).is_file()
            for modality in MODALITIES
            for filename in PREDICTION_FILES
        )
        summary = oof_dir / "oof_summary.json"
        oof_complete = oof_complete and summary.is_file()
        should_run_oof = args.stage in {"both", "oof"} and (
            args.force_oof or not oof_complete
        )

        if should_run_oof:
            command = [
                sys.executable,
                "scripts/15_generate_oof_predictions.py",
                "--protocol", protocol,
                "--manifest-dir", str(manifest_root / protocol),
                "--output-dir", str(oof_dir),
                "--folds", str(args.folds),
                "--num-workers", str(args.num_workers),
            ]
            if args.rgb_epochs is not None:
                command += ["--rgb-epochs", str(args.rgb_epochs)]
            for modality in MODALITIES:
                command += [
                    f"--{modality}-run-dir",
                    str(run_dirs[modality]),
                ]
            if args.allow_incomplete_audit:
                command.append("--allow-incomplete-audit")
            run_command(command, args.dry_run)
        elif args.stage in {"both", "oof"}:
            print(f"OOF already complete for {protocol}; skipping generation.")

    if args.stage in {"both", "fusion"} and not args.dry_run:
        incomplete = []
        for protocol, run_dirs in plan:
            missing = [
                f"{modality}/{filename}"
                for modality in MODALITIES
                for filename in PREDICTION_FILES
                if not (run_dirs[modality] / filename).is_file()
            ]
            if missing:
                incomplete.append(f"{protocol}: {', '.join(missing)}")
        if incomplete:
            raise FileNotFoundError(
                "Fusion requires complete OOF artifacts for every selected "
                "protocol:\n" + "\n".join(incomplete)
            )

    if args.stage in {"both", "fusion"}:
        for protocol, run_dirs in plan:
            print(f"\n===== Fusion: {protocol} =====", flush=True)
            command = [
                sys.executable,
                "scripts/16_benchmark_fusion.py",
                "--config", str(config_path),
                "--protocol", protocol,
                "--runs-root", str(runs_root),
                "--fusion-mode", "both",
                "--tag", f"all_protocols_{protocol}",
            ]
            for modality in MODALITIES:
                command += [
                    "--run-dir",
                    f"{modality}={run_dirs[modality]}",
                ]
            if args.allow_incomplete_audit:
                command.append("--allow-incomplete-audit")
            run_command(command, args.dry_run)


if __name__ == "__main__":
    main()
