from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional


def _build_child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start FineTree Pod API (6666) + Playground (5555).")
    parser.add_argument("--config", default=None, help="FineTree YAML config path.")
    parser.add_argument("--api-port", type=int, default=6666, help="Pod API port.")
    parser.add_argument("--gradio-port", type=int, default=5555, help="Playground app port.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("FINETREE_MAX_CONCURRENCY", "2")),
        help="Max concurrent API requests.",
    )
    return parser.parse_args(argv)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    env = _build_child_env()

    api_cmd = [
        sys.executable,
        "-m",
        "finetree_annotator.deploy.pod_api",
        "--port",
        str(args.api_port),
        "--max-concurrency",
        str(args.max_concurrency),
    ]
    gradio_cmd = [
        sys.executable,
        "-m",
        "finetree_annotator.deploy.pod_gradio",
        "--port",
        str(args.gradio_port),
    ]
    if args.config:
        api_cmd.extend(["--config", str(args.config)])
        gradio_cmd.extend(["--config", str(args.config)])

    api_proc = subprocess.Popen(api_cmd, env=env)
    gradio_proc = subprocess.Popen(gradio_cmd, env=env)
    procs = [api_proc, gradio_proc]

    def _handle_signal(_sig, _frame) -> None:
        for proc in procs:
            _terminate(proc)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    exit_code = 0
    try:
        while True:
            for proc in procs:
                code = proc.poll()
                if code is not None:
                    exit_code = code if code != 0 else exit_code
                    for sibling in procs:
                        if sibling is not proc:
                            _terminate(sibling)
                    return exit_code
            time.sleep(0.5)
    finally:
        for proc in procs:
            _terminate(proc)


if __name__ == "__main__":
    raise SystemExit(main())
