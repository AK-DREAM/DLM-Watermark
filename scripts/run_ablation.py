"""
多GPU异步任务调度器：
将bash脚本中的参数组合生成任务队列，
自动检测可用GPU数量并异步分配任务到各GPU上执行。

用法:
    python scripts/run_ablation.py --script BDLM
    python scripts/run_ablation.py --script KGW
    python scripts/run_ablation.py --script ourWatermark
    python scripts/run_ablation.py --script all
    python scripts/run_ablation.py --script BDLM --gpus 0,1,2,3
    python scripts/run_ablation.py --script BDLM --max_concurrent 2
"""

import argparse
import asyncio
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AblationTask:
    """单个消融实验任务"""
    config: str
    name: str
    delta: float
    kernel: str
    topk: int
    seeding_scheme: str
    num_samples: int
    extra_args: List[str] = field(default_factory=list)

    def to_cmd(self) -> List[str]:
        cmd = [
            sys.executable, "scripts/ablate_watermark.py",
            "--delta", str(self.delta),
            "--kernel", self.kernel,
            "--topk", str(self.topk),
            "--config", self.config,
            "--name", self.name,
            "--seeding_scheme", self.seeding_scheme,
            "--num_samples", str(self.num_samples),
        ]
        cmd.extend(self.extra_args)
        return cmd

    def __str__(self):
        return f"[config={os.path.basename(self.config)}, delta={self.delta}, kernel={self.kernel}, topk={self.topk}, seeding={self.seeding_scheme}]"


# ─── 任务定义（等价于原始bash脚本） ────────────────────────

SCRIPT_CONFIGS = {
    "BDLM": {
        "configs": ["configs/main/Llada/BDLM_llada8b_instruct.yaml"],
        "names": ["Llada/OurWatermark"],
        "deltas": [2],
        "kernels": ["[-1]"],
        "topks": [100],
        "seeding_schemes": ["sumhash"],
        "num_samples": 250,
    },
    "KGW": {
        "configs": ["configs/main/Llada/KGW_llada8b_instruct.yaml"],
        "names": ["Llada/OurWatermark"],
        "deltas": [1, 2, 2.5, 3, 4, 5],
        "kernels": ["[-1]"],
        "topks": [100],
        "seeding_schemes": ["sumhash"],
        "num_samples": 250,
    },
    "ourWatermark": {
        "configs": ["configs/main/Llada/ourWatermark_llada8b_instruct.yaml"],
        "names": ["Llada/OurWatermark"],
        "deltas": [1, 2, 2.5, 3, 4, 5],
        "kernels": ["[-1]"],
        "topks": [100],
        "seeding_schemes": ["sumhash"],
        "num_samples": 250,
    },
}


def generate_tasks(script_name: str) -> List[AblationTask]:
    """根据脚本名称生成任务列表"""
    cfg = SCRIPT_CONFIGS[script_name]
    tasks = []

    for (config, name), delta, kernel, topk, seeding in itertools.product(
        zip(cfg["configs"], cfg["names"]),
        cfg["deltas"],
        cfg["kernels"],
        cfg["topks"],
        cfg["seeding_schemes"],
    ):
        tasks.append(AblationTask(
            config=config,
            name=name,
            delta=delta,
            kernel=kernel,
            topk=topk,
            seeding_scheme=seeding,
            num_samples=cfg["num_samples"],
        ))

    return tasks


def get_available_gpus(gpu_arg: Optional[str] = None) -> List[int]:
    """
    获取可用GPU列表。
    如果指定了 --gpus 参数则使用指定的GPU；
    否则通过 CUDA_VISIBLE_DEVICES 或 nvidia-smi 自动检测。
    """
    if gpu_arg:
        return [int(g.strip()) for g in gpu_arg.split(",")]

    # 尝试从 CUDA_VISIBLE_DEVICES 获取
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return [int(g.strip()) for g in cuda_visible.split(",")]

    # 通过 nvidia-smi 检测
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  无法检测GPU数量，默认使用 GPU 0")
        return [0]


class GPUTaskScheduler:
    """
    异步GPU任务调度器：
    维护一个GPU可用信号量，将任务队列中的任务分配到空闲GPU上执行。
    """

    def __init__(self, gpus: List[int], max_concurrent: Optional[int] = None):
        self.gpus = gpus
        self.max_concurrent = max_concurrent or len(gpus)
        self.gpu_queue: asyncio.Queue = asyncio.Queue()
        self.results: List[dict] = []

    async def initialize(self):
        """将所有可用GPU放入队列"""
        for gpu_id in self.gpus:
            await self.gpu_queue.put(gpu_id)

    async def run_task(self, task: AblationTask, task_idx: int, total: int):
        """在空闲GPU上运行单个任务"""
        gpu_id = await self.gpu_queue.get()

        try:
            cmd = task.to_cmd()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # 确保 src 在 PYTHONPATH 中
            old_pypath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"src{os.pathsep}{old_pypath}" if old_pypath else "src"

            print(f"🚀 [{task_idx + 1}/{total}] GPU {gpu_id}: {task}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print(f"✅ [{task_idx + 1}/{total}] GPU {gpu_id}: {task} 完成")
                self.results.append({"task": str(task), "gpu": gpu_id, "status": "success"})
            else:
                print(f"❌ [{task_idx + 1}/{total}] GPU {gpu_id}: {task} 失败 (code={process.returncode})")
                if stderr:
                    print(f"   stderr: {stderr.decode()[-500:]}")
                self.results.append({"task": str(task), "gpu": gpu_id, "status": "failed", "returncode": process.returncode})

        except Exception as e:
            print(f"❌ [{task_idx + 1}/{total}] GPU {gpu_id}: {task} 异常: {e}")
            self.results.append({"task": str(task), "gpu": gpu_id, "status": "error", "error": str(e)})

        finally:
            # 归还 GPU
            await self.gpu_queue.put(gpu_id)

    async def run_all(self, tasks: List[AblationTask]):
        """异步运行所有任务"""
        await self.initialize()

        total = len(tasks)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_task(task, idx):
            async with semaphore:
                await self.run_task(task, idx, total)

        aws = [limited_task(task, i) for i, task in enumerate(tasks)]
        await asyncio.gather(*aws)

        # 打印摘要
        success = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] != "success")
        print(f"\n{'='*60}")
        print(f"📊 执行摘要: {success} 成功, {failed} 失败, 共 {total} 个任务")
        print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="多GPU异步消融实验运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_ablation.py --script BDLM
  python scripts/run_ablation.py --script KGW --gpus 0,1,2,3
  python scripts/run_ablation.py --script ourWatermark --max_concurrent 2
  python scripts/run_ablation.py --script all
        """,
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        choices=list(SCRIPT_CONFIGS.keys()) + ["all"],
        help="要运行的实验脚本名称，或 'all' 运行全部",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="要使用的GPU编号，逗号分隔（如 '0,1,2,3'）。默认自动检测所有可用GPU。",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=None,
        help="最大并发任务数（默认等于GPU数量）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印任务列表，不实际执行",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 生成任务
    if args.script == "all":
        tasks = []
        for script_name in SCRIPT_CONFIGS:
            tasks.extend(generate_tasks(script_name))
    else:
        tasks = generate_tasks(args.script)

    gpus = get_available_gpus(args.gpus)

    print(f"📋 共 {len(tasks)} 个任务")
    print(f"🎮 可用 GPU: {gpus}")
    print(f"⚡ 最大并发: {args.max_concurrent or len(gpus)}")
    print()

    if args.dry_run:
        print("🔍 Dry run — 任务列表:")
        for i, task in enumerate(tasks):
            print(f"  [{i + 1}/{len(tasks)}] {task}")
            print(f"    cmd: {' '.join(task.to_cmd())}")
        return

    scheduler = GPUTaskScheduler(gpus, max_concurrent=args.max_concurrent)
    asyncio.run(scheduler.run_all(tasks))


if __name__ == "__main__":
    main()
