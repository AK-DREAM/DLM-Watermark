"""
文件IO工具模块：
1. 提供 WATERMARK_OUTPUT_BASE_DIR 环境变量支持
2. 提供基于文件锁的安全并发写入
"""

import os
import fcntl
import json
import time
import contextlib
import pandas as pd
from pathlib import Path


def get_output_base_dir(fallback: str = ".") -> str:
    """
    从环境变量 WATERMARK_OUTPUT_BASE_DIR 获取输出根目录。
    如果未设置则使用 fallback（默认为当前目录）。
    """
    return os.environ.get("WATERMARK_OUTPUT_BASE_DIR", fallback)


def resolve_output_path(relative_path: str, fallback_base: str = ".") -> str:
    """
    将相对路径与 WATERMARK_OUTPUT_BASE_DIR 拼接。
    例如: relative_path="output/exp1/results.jsonl"
    如果 WATERMARK_OUTPUT_BASE_DIR="/data/results"，
    则返回 "/data/results/output/exp1/results.jsonl"
    """
    base = get_output_base_dir(fallback_base)
    return os.path.join(base, relative_path)


@contextlib.contextmanager
def file_lock(filepath: str, timeout: float = 60.0, poll_interval: float = 0.1):
    """
    基于文件的排他锁，用于防止多进程并发写入冲突。
    使用 fcntl.flock 实现，带超时机制。

    Args:
        filepath: 需要加锁的文件路径
        timeout: 获取锁的最大等待时间（秒），默认60秒
        poll_interval: 轮询间隔（秒），默认0.1秒

    Raises:
        TimeoutError: 超时仍未获取到锁时抛出

    用法:
        with file_lock("/path/to/data.jsonl"):
            # 在此处安全地读写文件
            ...
    """
    lock_path = filepath + ".lock"
    os.makedirs(os.path.dirname(lock_path) if os.path.dirname(lock_path) else ".", exist_ok=True)
    lock_fd = open(lock_path, "w")
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # 成功获取锁
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    lock_fd.close()
                    raise TimeoutError(
                        f"无法在 {timeout} 秒内获取文件锁: {lock_path}"
                    )
                time.sleep(poll_interval)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def safe_append_jsonl(filepath: str, results: list, transform_fn=None):
    """
    安全地将结果追加写入 .jsonl 文件（带文件锁）。

    Args:
        filepath: 目标 .jsonl 文件路径
        results: 待写入的字典列表
        transform_fn: 可选的变换函数，对每一行进行处理
    """
    with file_lock(filepath):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "a") as f:
            for result in results:
                line = transform_fn(result) if transform_fn else result
                f.write(json.dumps(line) + "\n")


def safe_write_df_json(filepath: str, df: pd.DataFrame):
    """
    安全地将 DataFrame 写入 .jsonl 文件（带文件锁，完整覆盖写入）。

    Args:
        filepath: 目标文件路径
        df: 待写入的 DataFrame
    """
    with file_lock(filepath):
        df.to_json(filepath, lines=True, orient="records")


def safe_write_df_csv(filepath: str, df: pd.DataFrame, **kwargs):
    """
    安全地将 DataFrame 写入 .csv 文件（带文件锁）。

    Args:
        filepath: 目标文件路径
        df: 待写入的 DataFrame
    """
    with file_lock(filepath):
        df.to_csv(filepath, **kwargs)
