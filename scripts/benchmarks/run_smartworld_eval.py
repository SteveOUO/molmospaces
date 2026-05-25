import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


EVAL_CONFIG = "molmo_spaces.evaluation.configs.evaluation_configs:SmartWorldPolicyEvalConfig"
DEFAULT_EGL_VENDOR_JSON = Path(__file__).with_name("10_nvidia.json")

BENCHMARKS = {
    "close": "benchmarks/molmospaces-bench-v1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark",
    "open": "benchmarks/molmospaces-bench-v1/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark",
    "pick": "benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231",
    "pick_place": "benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark",
}
ABLATION_TASKS = ("pick", "pick_place")
ABLATION_EPISODES_PER_TASK = 20


def _read_overall(csv_path: Path) -> dict[str, str]:
    rows = []
    with csv_path.open(newline="") as f:
        filtered = (line for line in f if not line.startswith("#"))
        rows = list(csv.DictReader(filtered))
    for row in rows:
        if row.get("category") == "OVERALL":
            return row
    raise RuntimeError(f"No OVERALL row found in {csv_path}")


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MolmoSpaces benchmarks against a running SmartWorld websocket server."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["pick", "pick_place"],
        choices=sorted(BENCHMARKS),
        help="Benchmark tasks to run. Use all four for MolmoSpaces Bench v1 Combined.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all MolmoSpaces Bench v1 tasks: close/open/pick/pick_place.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the SmartWorld ablation set: pick and pick_place, 20 episodes each.",
    )
    parser.add_argument(
        "--assets-dir",
        default=os.environ.get("MLSPACES_ASSETS_DIR", "assets"),
        help="MolmoSpaces assets root. Usually /mnt/project/world_model/molmospaces_assets.",
    )
    parser.add_argument(
        "--output-root",
        default="eval_output/smartworld",
        help="Directory for per-task eval outputs and CSV summaries.",
    )
    parser.add_argument("--host", default=os.environ.get("SMARTWORLD_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SMARTWORLD_SERVER_PORT", "7777")))
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("SMARTWORLD_CHUNK_SIZE", "8")))
    parser.add_argument("--grasping-type", default=os.environ.get("SMARTWORLD_GRASPING_TYPE", "binary"))
    parser.add_argument(
        "--grasping-threshold",
        type=float,
        default=float(os.environ.get("SMARTWORLD_GRASPING_THRESHOLD", "0.5")),
    )
    parser.add_argument("--policy-name", default="smartworld")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--idx", type=int, default=None, help="Run one episode index for smoke tests.")
    parser.add_argument("--task-horizon-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--camera-system",
        default=os.environ.get("SMARTWORLD_CAMERA_SYSTEM", "smartworld_droid_three_view"),
        choices=["benchmark", "franka_eval", "smartworld_droid_three_view"],
        help="Camera system override passed to MolmoSpaces eval_main.",
    )
    parser.add_argument("--success-condition", default="both", choices=["at-end", "oracle", "both"])
    parser.add_argument("--dt", type=float, default=66.0 / 1000.0)
    parser.add_argument(
        "--no-wandb",
        dest="no_wandb",
        action="store_true",
        default=True,
        help="Disable wandb logging for smoke and automated eval runs.",
    )
    parser.add_argument(
        "--wandb",
        dest="no_wandb",
        action="store_false",
        help="Enable wandb logging if the environment is already logged in.",
    )
    args = parser.parse_args()

    if args.ablation:
        tasks = list(ABLATION_TASKS)
        if args.max_episodes is None:
            args.max_episodes = ABLATION_EPISODES_PER_TASK
    elif args.all:
        tasks = list(BENCHMARKS)
    else:
        tasks = args.tasks
    assets_dir = Path(args.assets_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SMARTWORLD_SERVER_HOST"] = args.host
    env["SMARTWORLD_SERVER_PORT"] = str(args.port)
    env["SMARTWORLD_CHUNK_SIZE"] = str(args.chunk_size)
    env["SMARTWORLD_GRASPING_TYPE"] = args.grasping_type
    env["SMARTWORLD_GRASPING_THRESHOLD"] = str(args.grasping_threshold)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    if DEFAULT_EGL_VENDOR_JSON.exists():
        env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(DEFAULT_EGL_VENDOR_JSON))
    if args.no_wandb:
        env.setdefault("WANDB_MODE", "disabled")
    env["MLSPACES_ASSETS_DIR"] = str(assets_dir)

    summary_rows: list[dict[str, str]] = []
    for task in tasks:
        benchmark_dir = assets_dir / BENCHMARKS[task]
        if not benchmark_dir.exists():
            raise FileNotFoundError(
                f"Missing benchmark for {task}: {benchmark_dir}. "
                "Run `python -m molmo_spaces.molmo_spaces_constants` first."
            )

        task_output = output_root / task
        csv_path = output_root / f"{task}.csv"

        eval_cmd = [
            sys.executable,
            "molmo_spaces/evaluation/eval_main.py",
            EVAL_CONFIG,
            "--benchmark_dir",
            str(benchmark_dir),
            "--output_dir",
            str(task_output),
            "--num_workers",
            str(args.num_workers),
            "--camera_system",
            args.camera_system,
        ]
        if args.no_wandb:
            eval_cmd.append("--no_wandb")
        if args.max_episodes is not None:
            eval_cmd.extend(["--max_episodes", str(args.max_episodes)])
        if args.idx is not None:
            eval_cmd.extend(["--idx", str(args.idx)])
        if args.task_horizon_steps is not None:
            eval_cmd.extend(["--task_horizon_steps", str(args.task_horizon_steps)])

        _run(eval_cmd, env)

        csv_cmd = [
            sys.executable,
            "scripts/benchmarks/eval_to_csv.py",
            str(task_output),
            args.policy_name,
            "--success-condition",
            args.success_condition,
            "--output-csv",
            str(csv_path),
            "--dt",
            str(args.dt),
        ]
        _run(csv_cmd, env)

        overall = _read_overall(csv_path)
        overall["task"] = task
        overall["csv_path"] = str(csv_path)
        summary_rows.append(overall)

    summary_path = output_root / "summary.csv"
    fieldnames = [
        "task",
        "policy",
        "successes",
        "total",
        "success_rate_pct",
        "oracle_successes",
        "oracle_rate_pct",
        "jerk_joint_mean",
        "csv_path",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {summary_path}")
    for row in summary_rows:
        print(
            f"{row['task']}: at-end={row.get('success_rate_pct', 'n/a')} "
            f"oracle={row.get('oracle_rate_pct', 'n/a')} total={row.get('total', 'n/a')}"
        )


if __name__ == "__main__":
    main()
