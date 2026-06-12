"""
Usage:
python scripts/eval_policy_sim.py --checkpoint path/to/ckpt -o path/to/output_dir
python scripts/eval_policy_sim.py -c ep-0500.ckpt -o out/ep-0500 --n-test-per-task 50 -n 3 --test-start-seed 1000
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import wandb
import json
import numpy as np
import yaml
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from typing import Any, Dict, List, Optional


def ensure_libero_runtime() -> None:
    """LIBERO sim import: submodule on PYTHONPATH + non-interactive ~/.libero/config.yaml."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    libero_repo = repo_root / "third_party" / "LIBERO"
    if libero_repo.is_dir() and str(libero_repo) not in sys.path:
        sys.path.insert(0, str(libero_repo))

    libero_pkg = libero_repo / "libero" / "libero"
    config_dir = pathlib.Path(
        os.environ.get("LIBERO_CONFIG_PATH", pathlib.Path.home() / ".libero")
    )
    config_file = config_dir / "config.yaml"
    if config_file.is_file():
        return

    benchmark_root = str(libero_pkg.resolve())
    default_paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": os.path.join(benchmark_root, "bddl_files"),
        "init_states": os.path.join(benchmark_root, "init_files"),
        "datasets": os.path.join(benchmark_root, "../datasets"),
        "assets": os.path.join(benchmark_root, "assets"),
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    with config_file.open("w", encoding="utf-8") as f:
        yaml.dump(default_paths, f)
    print(f"[eval] wrote LIBERO config: {config_file}")


def resolve_n_test(
    cfg,
    n_test: Optional[int],
    n_test_per_task: Optional[int],
) -> int:
    num_tasks = int(cfg.task.policy.get("num_tasks", 10))
    if n_test_per_task is not None:
        return int(n_test_per_task) * num_tasks
    if n_test is not None:
        return int(n_test)
    return int(cfg.task.policy.env_runner.n_test)


def build_runner_kwargs(
    cfg,
    output_dir: str,
    n_test: int,
    test_start_seed: int,
    n_test_vis_cap: Optional[int] = None,
) -> Dict[str, Any]:
    runner_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "n_test": n_test,
        "test_start_seed": test_start_seed,
    }
    cfg_n_test_vis = cfg.task.policy.env_runner.get("n_test_vis", None)
    if cfg_n_test_vis is not None:
        cap = int(cfg_n_test_vis) if n_test_vis_cap is None else n_test_vis_cap
        runner_kwargs["n_test_vis"] = min(int(cfg_n_test_vis), cap, n_test)
    return runner_kwargs


@click.command()
@click.option('-c', '--checkpoint', required=True, help="either a .ckpt file or a directory containing .ckpt files")
@click.option('-o', '--output_dir', required=True, help="output directory for eval info dump")
@click.option('-n', '--num_exp', default=1, help="num experiments to run (independent seed blocks)")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('--n-test', default=None, type=int, help="total rollout episodes across all LIBERO-10 tasks")
@click.option(
    '--n-test-per-task',
    default=None,
    type=int,
    help="episodes per task; total n_test = n_test_per_task * num_tasks (preferred for interpretability)",
)
@click.option('--test-start-seed', default=None, type=int, help="override LiberoRunner.test_start_seed (default from ckpt cfg, usually 1000)")
@click.option('--seed-stride', default=None, type=int, help="offset between num_exp runs (default: resolved n_test)")
@click.option('--temperature', default=None, type=float, help="temperature for policy inference")
@click.option('--topk', default=None, type=int, help="topk for policy inference")
@click.option('--use_k_tokens', default=None, type=int, help="number of tokens to use for policy inference")
@click.option('--use-blockwise', is_flag=True, default=False, help="use Blockwise-OAT inference path")
@click.option('--blockwise-prefix-len', default=None, type=int, help="prefix length P for blockwise inference")
@click.option('--blockwise-refine-iters', default=None, type=int, help="tail refinement iterations for blockwise inference")
@click.option(
    '--overwrite',
    is_flag=True,
    default=False,
    help="remove existing output_dir without interactive prompt (required for tmux/batch)",
)
def eval_policy_sim(
    checkpoint: str,
    output_dir: str,
    num_exp: int = 1,
    device: str = 'cuda:0',
    n_test: Optional[int] = None,
    n_test_per_task: Optional[int] = None,
    test_start_seed: Optional[int] = None,
    seed_stride: Optional[int] = None,
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    use_k_tokens: Optional[int] = None,
    use_blockwise: bool = False,
    blockwise_prefix_len: Optional[int] = None,
    blockwise_refine_iters: Optional[int] = None,
    overwrite: bool = False,
):
    if n_test is not None and n_test_per_task is not None:
        raise click.ClickException("Use only one of --n-test or --n-test-per-task")

    ensure_libero_runtime()

    if os.path.exists(output_dir):
        if not overwrite:
            click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        import shutil
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    ckpts: List[str]
    if os.path.isdir(checkpoint):
        ckpts = [
            os.path.join(checkpoint, f)
            for f in os.listdir(checkpoint)
            if f.endswith('.ckpt') and f != 'latest.ckpt'
        ]
    else:
        ckpts = [checkpoint,]

    base_output_dir = output_dir
    for ckpt in ckpts:
        if len(ckpts) > 1:
            ckpt_name = os.path.basename(ckpt).replace('.ckpt', '')
            output_dir = os.path.join(base_output_dir, ckpt_name)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir

        policy, cfg = BasePolicy.from_checkpoint(ckpt, return_configuration=True)

        device_t = torch.device(device)
        policy.to(device_t)
        policy.eval()

        resolved_n_test = resolve_n_test(cfg, n_test, n_test_per_task)
        num_tasks = int(cfg.task.policy.get("num_tasks", 10))
        episodes_per_task = resolved_n_test / num_tasks
        base_seed = int(
            test_start_seed
            if test_start_seed is not None
            else cfg.task.policy.env_runner.get("test_start_seed", 1000)
        )
        stride = int(seed_stride if seed_stride is not None else resolved_n_test)

        print(f"Running evaluation on {ckpt}")
        print(
            f"  task_suite={cfg.task.policy.env_runner.task_name} "
            f"num_tasks={num_tasks} n_test={resolved_n_test} "
            f"(~{episodes_per_task:.1f} episodes/task)"
        )
        print(f"  test_start_seed={base_seed} num_exp={num_exp} seed_stride={stride}")

        infer_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            infer_kwargs['temperature'] = temperature
        if topk is not None:
            infer_kwargs['topk'] = topk
        if use_k_tokens is not None:
            infer_kwargs['use_k_tokens'] = use_k_tokens
        if use_blockwise:
            infer_kwargs['use_blockwise'] = True
        if blockwise_prefix_len is not None:
            infer_kwargs['blockwise_prefix_len'] = int(blockwise_prefix_len)
        if blockwise_refine_iters is not None:
            infer_kwargs['blockwise_refine_iters'] = int(blockwise_refine_iters)

        all_runs: List[Dict[str, Any]] = []
        runner_log = None

        for exp_idx in range(num_exp):
            exp_seed = base_seed + exp_idx * stride
            runner_kwargs = build_runner_kwargs(
                cfg, output_dir, resolved_n_test, exp_seed, n_test_vis_cap=0 if exp_idx > 0 else None
            )
            print(f"  exp {exp_idx + 1}/{num_exp}: test_start_seed={exp_seed}")

            env_runner: BaseRunner = hydra.utils.instantiate(
                cfg.task.policy.env_runner,
                **runner_kwargs,
            )
            this_log = env_runner.run(policy, **infer_kwargs)
            env_runner.close()

            for key, value in list(this_log.items()):
                if isinstance(value, wandb.sdk.data_types.video.Video):
                    this_log[key] = [value]
            all_runs.append({k: v for k, v in this_log.items() if not isinstance(v, list)})
            print(f"Exp {exp_idx + 1}: success rate = {this_log['mean_success_rate']}")
            if runner_log is None:
                runner_log = this_log

        assert runner_log is not None
        numeric_keys = [k for k in all_runs[0].keys()]
        mean_log = {}
        std_log = {}
        for key in numeric_keys:
            values = [run[key] for run in all_runs]
            mean_log[key] = np.mean(values)
            if num_exp > 1:
                std_log[key] = np.std(values, ddof=1)

        json_log: Dict[str, Any] = {
            'checkpoint': ckpt,
            'num_exp': num_exp,
            'n_test': resolved_n_test,
            'n_test_per_task': n_test_per_task,
            'num_tasks': num_tasks,
            'episodes_per_task_approx': episodes_per_task,
            'test_start_seed': base_seed,
            'seed_stride': stride,
            'task_suite': cfg.task.policy.env_runner.task_name,
            'use_blockwise': bool(use_blockwise),
            'blockwise_prefix_len': blockwise_prefix_len,
            'blockwise_refine_iters': blockwise_refine_iters,
        }

        for key, value in mean_log.items():
            json_log[f'{key}_mean'] = float(value)
        if num_exp > 1:
            for key, value in std_log.items():
                json_log[f'{key}_std'] = float(value)
                json_log[f'{key}_stderr'] = float(value / np.sqrt(num_exp))

        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    if isinstance(video, wandb.sdk.data_types.video.Video):
                        json_log[f'{key}_{i}'] = video._path

        out_path = os.path.join(output_dir, 'eval_log.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(json_log, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    eval_policy_sim()
