"""
多GPU异步任务调度器（v2 — 可组合配置 + 参数扫描）：

支持两种模式：
  1. 分层模式（推荐）：自动发现 configs/decoupled/ 下的模型、水印、数据集片段，
     按笛卡尔积生成任务，并支持参数扫描和耦合覆盖。
  2. 旧版模式（向后兼容）：使用 LEGACY_SCRIPT_CONFIGS 中的硬编码配置。

用法:
    # 新分层模式
    python scripts/run_ablation.py --experiment dlm
    python scripts/run_ablation.py --experiment bdlm_block_sweep --gpus 0,1,2,3
    python scripts/run_ablation.py --experiment all
    python scripts/run_ablation.py --experiment dlm --filter_model llada --filter_dataset longform

    # 旧版模式
    python scripts/run_ablation.py --script BDLM
    python scripts/run_ablation.py --script all

    # 通用选项
    python scripts/run_ablation.py --experiment dlm --max_concurrent 2 --dry_run
"""

import argparse
import asyncio
import glob
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════
#  配置发现
# ═══════════════════════════════════════════════════════════════

DECOUPLED_BASE = os.path.join(os.path.dirname(__file__), "..", "configs", "decoupled")


def auto_discover(directory: str) -> List[str]:
    """扫描目录下所有 .yaml 文件，返回排序后的绝对路径列表。"""
    abs_dir = os.path.abspath(directory)
    paths = sorted(glob.glob(os.path.join(abs_dir, "*.yaml")))
    if not paths:
        print(f"⚠️  No .yaml files found in {abs_dir}")
    return paths


def stem(path: str) -> str:
    """提取文件名（不含扩展名），用于过滤和命名。"""
    return os.path.splitext(os.path.basename(path))[0]


# ═══════════════════════════════════════════════════════════════
#  实验矩阵定义（分层模式）
# ═══════════════════════════════════════════════════════════════

EXPERIMENT_MATRIX: Dict[str, Dict[str, Any]] = {
    # ── DLM (Ours) delta 扫描 ──
    "dlm": {
        "model_layers": auto_discover(os.path.join(DECOUPLED_BASE, "models")),
        "watermark": os.path.abspath(os.path.join(DECOUPLED_BASE, "watermarks", "dlm.yaml")),
        "dataset_layers": auto_discover(os.path.join(DECOUPLED_BASE, "datasets")),
        "sweep": {
            "watermark_config.delta": [1, 2, 2.5, 3, 4, 5],
        },
        "coupled_overrides": {},
        "defaults": {"num_samples": 200},
    },

    # ── BDLM 固定参数 ──
    "bdlm": {
        "model_layers": [os.path.abspath(os.path.join(DECOUPLED_BASE, "models", "llada_8b.yaml"))],
        "watermark": os.path.abspath(os.path.join(DECOUPLED_BASE, "watermarks", "bdlm.yaml")),
        "dataset_layers": auto_discover(os.path.join(DECOUPLED_BASE, "datasets")),
        "sweep": {
            "watermark_config.delta": [1, 2, 2.5, 3, 4, 5],
        },
        "coupled_overrides": {},
        "defaults": {"num_samples": 200},
    },

    # ── BDLM block_length 扫描（自动同步 offset 和 context_len） ──
    "bdlm_block_sweep": {
        "model_layers": [os.path.abspath(os.path.join(DECOUPLED_BASE, "models", "llada_8b.yaml"))],
        "watermark": os.path.abspath(os.path.join(DECOUPLED_BASE, "watermarks", "bdlm.yaml")),
        "dataset_layers": auto_discover(os.path.join(DECOUPLED_BASE, "datasets")),
        "sweep": {
            "model_configuration.model_specific_arguments.block_length": [25, 50, 100, 150],
        },
        "coupled_overrides": {
            "model_configuration.model_specific_arguments.block_length": [
                "watermark_config.offset",
                "watermark_config.context_len",
            ],
        },
        "defaults": {"num_samples": 200},
    },

    # ── KGW delta 扫描 ──
    "kgw": {
        "model_layers": auto_discover(os.path.join(DECOUPLED_BASE, "models")),
        "watermark": os.path.abspath(os.path.join(DECOUPLED_BASE, "watermarks", "kgw.yaml")),
        "dataset_layers": auto_discover(os.path.join(DECOUPLED_BASE, "datasets")),
        "sweep": {
            "watermark_config.delta": [1, 2, 2.5, 3, 4, 5],
        },
        "coupled_overrides": {},
        "defaults": {"num_samples": 200},
    },

    # ── Unigram delta 扫描 ──
    "unigram": {
        "model_layers": [os.path.abspath(os.path.join(DECOUPLED_BASE, "models", "llada_8b.yaml"))],
        "watermark": os.path.abspath(os.path.join(DECOUPLED_BASE, "watermarks", "unigram.yaml")),
        "dataset_layers": auto_discover(os.path.join(DECOUPLED_BASE, "datasets")),
        "sweep": {
            "watermark_config.delta": [1, 2, 2.5, 3, 4, 5],
        },
        "coupled_overrides": {},
        "defaults": {"num_samples": 200},
    },

    # ── 无水印 baseline ──
    "no_watermark": {
        "model_layers": auto_discover(os.path.join(DECOUPLED_BASE, "models")),
        "watermark": os.path.abspath(os.path.join(DECOUPLED_BASE, "watermarks", "none.yaml")),
        "dataset_layers": auto_discover(os.path.join(DECOUPLED_BASE, "datasets")),
        "sweep": {},
        "coupled_overrides": {},
        "defaults": {"num_samples": 200},
    },

    # ── Baseline 水印方法（KTH, AAR, OA）——仅 LLaDA ──
    "baselines": {
        "model_layers": [os.path.abspath(os.path.join(DECOUPLED_BASE, "models", "llada_8b.yaml"))],
        "watermark": "__multi__",  # 特殊标记：遍历多个水印
        "_watermark_list": [
            os.path.abspath(os.path.join(DECOUPLED_BASE, "watermarks", w))
            for w in ["kth.yaml", "aar.yaml", "oa.yaml"]
        ],
        "dataset_layers": auto_discover(os.path.join(DECOUPLED_BASE, "datasets")),
        "sweep": {},
        "coupled_overrides": {},
        "defaults": {"num_samples": 200},
    },
}


# ═══════════════════════════════════════════════════════════════
#  旧版脚本配置（向后兼容）
# ═══════════════════════════════════════════════════════════════

LEGACY_SCRIPT_CONFIGS = {
    "BDLM": {
        "configs": ["configs/main/Llada/BDLM_llada8b_instruct.yaml"],
        "names": ["Llada/BDLM"],
        "deltas": [1, 2, 2.5, 3, 4, 5],
        "kernels": ["[-1]"],
        "topks": [100],
        "seeding_schemes": ["sumhash"],
        "num_samples": 200,
    },
    "KGW": {
        "configs": ["configs/main/Llada/KGW_llada8b_instruct.yaml"],
        "names": ["Llada/KGW"],
        "deltas": [1, 2, 2.5, 3, 4, 5],
        "kernels": ["[-1]"],
        "topks": [100],
        "seeding_schemes": ["sumhash"],
        "num_samples": 200,
    },
    "ourWatermark": {
        "configs": ["configs/main/Llada/ourWatermark_llada8b_instruct.yaml"],
        "names": ["Llada/DLM"],
        "deltas": [1, 2, 2.5, 3, 4, 5],
        "kernels": ["[-1]"],
        "topks": [100],
        "seeding_schemes": ["sumhash"],
        "num_samples": 200,
    },
}


# ═══════════════════════════════════════════════════════════════
#  任务数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class AblationTask:
    """单个消融实验任务（支持分层和旧版两种模式）"""

    # --- 分层模式字段 ---
    layers: List[str] = field(default_factory=list)
    overrides: List[str] = field(default_factory=list)
    num_samples: int = 200

    # --- 旧版模式字段 ---
    config: str = ""
    name: str = ""
    delta: float = 0.0
    kernel: str = "[-1]"
    topk: int = 100
    seeding_scheme: str = "sumhash"
    extra_args: List[str] = field(default_factory=list)

    # --- 模式标记 ---
    use_layers: bool = False

    def to_cmd(self) -> List[str]:
        if self.use_layers:
            cmd = [
                sys.executable, "scripts/ablate_watermark.py",
                "--layers", *self.layers,
            ]
            for ov in self.overrides:
                cmd.extend(["--override", ov])
            cmd.extend(["--num_samples", str(self.num_samples)])
            return cmd
        else:
            # 旧版命令
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

    def short_label(self) -> str:
        """生成简短的任务标签用于日志。"""
        if self.use_layers:
            parts = [stem(l) for l in self.layers]
            ov_str = ", ".join(self.overrides[:3])
            if len(self.overrides) > 3:
                ov_str += f" +{len(self.overrides)-3} more"
            return f"[layers={'+'.join(parts)}, overrides=({ov_str})]"
        else:
            return f"[config={os.path.basename(self.config)}, delta={self.delta}, kernel={self.kernel}, topk={self.topk}, seeding={self.seeding_scheme}]"

    def __str__(self):
        return self.short_label()


# ═══════════════════════════════════════════════════════════════
#  任务生成
# ═══════════════════════════════════════════════════════════════

def _cartesian_sweep(sweep: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    生成扫描轴的笛卡尔积。
    如果 sweep 为空，返回一个空字典的列表（生成一个无额外覆盖的任务）。
    """
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = list(sweep.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def generate_experiment_tasks(
    experiment_name: str,
    filter_model: Optional[str] = None,
    filter_dataset: Optional[str] = None,
    filter_watermark: Optional[str] = None,
) -> List[AblationTask]:
    """
    根据 EXPERIMENT_MATRIX 中的实验定义，生成任务列表。
    每个任务 = 1 model × 1 watermark × 1 dataset × 1 sweep_point。
    """
    exp = EXPERIMENT_MATRIX[experiment_name]
    tasks: List[AblationTask] = []

    model_layers = exp["model_layers"]
    dataset_layers = exp["dataset_layers"]
    sweep = exp.get("sweep", {})
    coupled = exp.get("coupled_overrides", {})
    defaults = exp.get("defaults", {})

    # 处理多水印模式
    if exp.get("watermark") == "__multi__":
        watermark_list = exp["_watermark_list"]
    else:
        watermark_list = [exp["watermark"]]

    # 过滤
    if filter_model:
        model_layers = [m for m in model_layers if filter_model.lower() in stem(m).lower()]
    if filter_dataset:
        dataset_layers = [d for d in dataset_layers if filter_dataset.lower() in stem(d).lower()]
    if filter_watermark:
        watermark_list = [w for w in watermark_list if filter_watermark.lower() in stem(w).lower()]

    sweep_combos = _cartesian_sweep(sweep)

    for model_layer in model_layers:
        for watermark_path in watermark_list:
            for dataset_layer in dataset_layers:
                for sweep_combo in sweep_combos:
                    overrides = []
                    for key, val in sweep_combo.items():
                        overrides.append(f"{key}={val}")
                        # 应用耦合规则
                        if key in coupled:
                            for coupled_key in coupled[key]:
                                overrides.append(f"{coupled_key}={val}")

                    # 添加默认覆盖
                    if "num_samples" in defaults:
                        overrides.append(f"evaluation_config.num_samples={defaults['num_samples']}")

                    tasks.append(AblationTask(
                        layers=[model_layer, watermark_path, dataset_layer],
                        overrides=overrides,
                        num_samples=defaults.get("num_samples", 200),
                        use_layers=True,
                    ))

    return tasks


def generate_legacy_tasks(script_name: str) -> List[AblationTask]:
    """旧版模式：根据 LEGACY_SCRIPT_CONFIGS 生成任务。"""
    cfg = LEGACY_SCRIPT_CONFIGS[script_name]
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
            use_layers=False,
        ))

    return tasks


# ═══════════════════════════════════════════════════════════════
#  GPU 检测与调度
# ═══════════════════════════════════════════════════════════════

def get_available_gpus(gpu_arg: Optional[str] = None) -> List[int]:
    """
    获取可用GPU列表。
    如果指定了 --gpus 参数则使用指定的GPU；
    否则通过 CUDA_VISIBLE_DEVICES 或 nvidia-smi 自动检测。
    """
    if gpu_arg:
        return [int(g.strip()) for g in gpu_arg.split(",")]

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return [int(g.strip()) for g in cuda_visible.split(",")]

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
        for gpu_id in self.gpus:
            await self.gpu_queue.put(gpu_id)

    async def run_task(self, task: AblationTask, task_idx: int, total: int):
        gpu_id = await self.gpu_queue.get()

        try:
            cmd = task.to_cmd()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
            await self.gpu_queue.put(gpu_id)

    async def run_all(self, tasks: List[AblationTask]):
        await self.initialize()

        total = len(tasks)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_task(task, idx):
            async with semaphore:
                await self.run_task(task, idx, total)

        aws = [limited_task(task, i) for i, task in enumerate(tasks)]
        await asyncio.gather(*aws)

        success = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] != "success")
        print(f"\n{'='*60}")
        print(f"📊 执行摘要: {success} 成功, {failed} 失败, 共 {total} 个任务")
        print(f"{'='*60}")

        if failed > 0:
            print("\n❌ 失败的任务:")
            for r in self.results:
                if r["status"] != "success":
                    print(f"   {r['task']}")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="多GPU异步消融实验运行器（v2 — 支持分层配置和参数扫描）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
分层模式示例:
  python scripts/run_ablation.py --experiment dlm
  python scripts/run_ablation.py --experiment bdlm_block_sweep --gpus 0,1,2,3
  python scripts/run_ablation.py --experiment dlm --filter_model llada --filter_dataset longform
  python scripts/run_ablation.py --experiment all

旧版模式示例（向后兼容）:
  python scripts/run_ablation.py --script BDLM
  python scripts/run_ablation.py --script KGW --gpus 0,1,2,3
  python scripts/run_ablation.py --script all
        """,
    )

    # 二选一：分层模式 vs 旧版模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=f"分层实验名称。可选值: {', '.join(list(EXPERIMENT_MATRIX.keys()) + ['all'])}",
    )
    mode_group.add_argument(
        "--script",
        type=str,
        default=None,
        help=f"旧版脚本名称（向后兼容）。可选值: {', '.join(list(LEGACY_SCRIPT_CONFIGS.keys()) + ['all'])}",
    )

    # 过滤选项（仅分层模式有效）
    parser.add_argument("--filter_model", type=str, default=None, help="按模型名过滤（子字符串匹配）")
    parser.add_argument("--filter_dataset", type=str, default=None, help="按数据集名过滤（子字符串匹配）")
    parser.add_argument("--filter_watermark", type=str, default=None, help="按水印名过滤（子字符串匹配，仅 baselines 实验有效）")

    # 通用选项
    parser.add_argument("--gpus", type=str, default=None, help="要使用的GPU编号，逗号分隔（如 '0,1,2,3'）。默认自动检测。")
    parser.add_argument("--max_concurrent", type=int, default=None, help="最大并发任务数（默认等于GPU数量）")
    parser.add_argument("--dry_run", action="store_true", help="仅打印任务列表，不实际执行")

    return parser.parse_args()


def main():
    args = parse_args()

    # 生成任务
    tasks: List[AblationTask] = []

    if args.experiment:
        # 分层模式
        if args.experiment == "all":
            for exp_name in EXPERIMENT_MATRIX:
                tasks.extend(generate_experiment_tasks(
                    exp_name,
                    filter_model=args.filter_model,
                    filter_dataset=args.filter_dataset,
                    filter_watermark=args.filter_watermark,
                ))
        else:
            if args.experiment not in EXPERIMENT_MATRIX:
                print(f"❌ Unknown experiment: '{args.experiment}'")
                print(f"   Available: {', '.join(EXPERIMENT_MATRIX.keys())}")
                sys.exit(1)
            tasks = generate_experiment_tasks(
                args.experiment,
                filter_model=args.filter_model,
                filter_dataset=args.filter_dataset,
                filter_watermark=args.filter_watermark,
            )
    else:
        # 旧版模式
        if args.script == "all":
            for script_name in LEGACY_SCRIPT_CONFIGS:
                tasks.extend(generate_legacy_tasks(script_name))
        else:
            if args.script not in LEGACY_SCRIPT_CONFIGS:
                print(f"❌ Unknown script: '{args.script}'")
                print(f"   Available: {', '.join(LEGACY_SCRIPT_CONFIGS.keys())}")
                sys.exit(1)
            tasks = generate_legacy_tasks(args.script)

    if not tasks:
        print("⚠️  No tasks generated. Check your filters and config directories.")
        return

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
