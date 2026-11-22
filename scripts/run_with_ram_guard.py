#!/usr/bin/env python3
"""Run a command and stop its process tree when system RAM gets low."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


GIB = 1024 ** 3


def read_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            kib = int(line.split()[1])
            return kib * 1024 / GIB
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def read_process_info(pid: int) -> tuple[int, int] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None
    fields = stat[stat.rfind(")") + 2 :].split()
    if len(fields) <= 19:
        return None
    return int(fields[1]), int(fields[19])


def process_tree(root_pid: int) -> list[tuple[int, int]]:
    processes = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            info = read_process_info(pid)
        except ValueError:
            continue
        if info is not None:
            processes[pid] = info

    children = {}
    for pid, (parent_pid, _) in processes.items():
        children.setdefault(parent_pid, []).append(pid)

    result = []
    pending = [root_pid]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen or pid not in processes:
            continue
        seen.add(pid)
        result.append((pid, processes[pid][1]))
        pending.extend(children.get(pid, ()))
    return result


def process_matches(pid: int, start_time: int) -> bool:
    try:
        info = read_process_info(pid)
    except ValueError:
        return False
    return info is not None and info[1] == start_time


def signal_process_tree(root_pid: int, grace_seconds: float) -> None:
    targets = {}
    term_sent = set()
    deadline = time.monotonic() + grace_seconds

    while True:
        targets.update(process_tree(root_pid))
        for pid, start_time in targets.items():
            identity = (pid, start_time)
            if identity in term_sent or not process_matches(pid, start_time):
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            term_sent.add(identity)

        alive = [
            (pid, start_time)
            for pid, start_time in targets.items()
            if process_matches(pid, start_time)
        ]
        if not alive or time.monotonic() >= deadline:
            break
        time.sleep(0.1)

    targets.update(process_tree(root_pid))
    for pid, start_time in targets.items():
        if not process_matches(pid, start_time):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a command with a system MemAvailable safety limit."
    )
    parser.add_argument(
        "--min-available-gib",
        type=float,
        default=32.0,
        help="stop if MemAvailable falls below this value (default: 32 GiB)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="RAM check interval (default: 1 second)",
    )
    parser.add_argument(
        "--grace-seconds",
        type=float,
        default=3.0,
        help="time to allow graceful shutdown before SIGKILL (default: 3)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    if args.min_available_gib <= 0:
        parser.error("--min-available-gib must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.grace_seconds < 0:
        parser.error("--grace-seconds cannot be negative")

    available = read_available_gib()
    if available < args.min_available_gib:
        print(
            f"[RAM GUARD] Not starting: MemAvailable is {available:.1f} GiB, "
            f"below the {args.min_available_gib:.1f} GiB limit.",
            file=sys.stderr,
            flush=True,
        )
        return 125

    print(
        f"[RAM GUARD] Starting with {available:.1f} GiB available; "
        f"limit is {args.min_available_gib:.1f} GiB.",
        flush=True,
    )
    process = subprocess.Popen(command)
    min_available = available

    try:
        while process.poll() is None:
            time.sleep(args.poll_seconds)
            available = read_available_gib()
            min_available = min(min_available, available)
            if available < args.min_available_gib:
                print(
                    f"[RAM GUARD] MemAvailable fell to {available:.1f} GiB "
                    f"(limit {args.min_available_gib:.1f} GiB); stopping "
                    "the benchmark process tree.",
                    file=sys.stderr,
                    flush=True,
                )
                signal_process_tree(process.pid, args.grace_seconds)
                try:
                    process.wait(timeout=args.grace_seconds + 5.0)
                except subprocess.TimeoutExpired:
                    print(
                        "[RAM GUARD] Benchmark has not exited after SIGKILL.",
                        file=sys.stderr,
                        flush=True,
                    )
                return 125
    except KeyboardInterrupt:
        signal_process_tree(process.pid, args.grace_seconds)
        try:
            process.wait(timeout=args.grace_seconds + 5.0)
        except subprocess.TimeoutExpired:
            pass
        return 130

    return_code = process.wait()
    print(f"[RAM GUARD] Finished; minimum MemAvailable was {min_available:.1f} GiB.")
    return return_code if return_code >= 0 else 128 - return_code


if __name__ == "__main__":
    raise SystemExit(main())
