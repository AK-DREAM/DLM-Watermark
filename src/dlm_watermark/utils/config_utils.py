"""
配置工具模块：
1. 深度合并多个 YAML 配置片段（composable layers）
2. 支持点标记法覆盖任意配置字段（--override）
3. 自动发现配置目录中的可用片段
4. 统一的配置加载入口，兼容单体配置和分层配置
"""

import copy
import glob
import json
import os
from typing import Any, Dict, List, Optional

import yaml

from dlm_watermark.configs import MainConfiguration


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归深度合并两个字典。override 中的值覆盖 base 中的同名键。
    对于嵌套字典会递归合并，非字典值直接覆盖。
    
    Args:
        base: 基础字典
        override: 覆盖字典（优先级更高）
    
    Returns:
        合并后的新字典（不修改原始输入）
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_yaml_files(*paths: str) -> Dict[str, Any]:
    """
    加载并深度合并多个 YAML 文件。后面的文件覆盖前面的。
    
    Args:
        *paths: YAML 文件路径列表
    
    Returns:
        合并后的字典
    """
    merged: Dict[str, Any] = {}
    for path in paths:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        merged = deep_merge(merged, data)
    return merged


def _cast_value(value_str: str) -> Any:
    """
    智能类型转换：将字符串值转换为合适的 Python 类型。
    支持 int, float, bool, None, JSON list/dict, 及普通字符串。
    """
    # None
    if value_str.lower() in ("none", "null"):
        return None
    
    # bool
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    
    # int
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # JSON (list or dict)
    if value_str.startswith(("[", "{")):
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass
    
    # 普通字符串
    return value_str


def apply_overrides(raw_config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    应用点标记法覆盖。
    
    例如:
        overrides = ["watermark_config.delta=3.0", "model_configuration.steps=500"]
    
    支持嵌套键（用 . 分隔），值会自动进行类型转换。
    
    Args:
        raw_config: 原始配置字典
        overrides: "key=value" 格式的覆盖列表
    
    Returns:
        应用覆盖后的配置字典（会修改并返回原始字典）
    """
    for override_str in overrides:
        if "=" not in override_str:
            raise ValueError(
                f"Invalid override format: '{override_str}'. Expected 'key.path=value'"
            )
        
        key_path, value_str = override_str.split("=", 1)
        keys = key_path.strip().split(".")
        value = _cast_value(value_str.strip())
        
        # 逐层导航到目标位置
        target = raw_config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
    
    return raw_config


def auto_discover(directory: str) -> List[str]:
    """
    扫描目录中的所有 .yaml 文件并返回排序后的路径列表。
    
    Args:
        directory: 目标目录路径
    
    Returns:
        排序后的 YAML 文件路径列表
    """
    pattern = os.path.join(directory, "*.yaml")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"⚠️  No .yaml files found in {directory}")
    return paths


def load_config(
    base: Optional[str] = None,
    layers: Optional[List[str]] = None,
    overrides: Optional[List[str]] = None,
) -> MainConfiguration:
    """
    统一的配置加载入口，支持三种模式：
    
    1. 单体模式: load_config(base="path/to/monolithic.yaml")
    2. 分层模式: load_config(layers=["model.yaml", "watermark.yaml", "dataset.yaml"])
    3. 混合模式: load_config(base="base.yaml", layers=["extra.yaml"], overrides=["key=val"])
    
    加载顺序: base → layers（按序） → overrides
    
    Args:
        base: 单体配置文件路径（可选）
        layers: 分层配置文件路径列表（可选）
        overrides: 点标记法覆盖列表（可选）
    
    Returns:
        验证后的 MainConfiguration 实例
    """
    raw_config: Dict[str, Any] = {}
    
    # 加载单体配置
    if base:
        with open(base, "r") as f:
            raw_config = yaml.safe_load(f) or {}
    
    # 按序合并分层配置
    if layers:
        for layer_path in layers:
            with open(layer_path, "r") as f:
                layer_data = yaml.safe_load(f) or {}
            raw_config = deep_merge(raw_config, layer_data)
    
    # 应用覆盖
    if overrides:
        raw_config = apply_overrides(raw_config, overrides)
    
    # 验证并返回
    return MainConfiguration(**raw_config)
