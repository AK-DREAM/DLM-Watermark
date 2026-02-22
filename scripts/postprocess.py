"""
Data post-processing module for watermark experiment results.

Usage:
    python scripts/postprocess.py list
    python scripts/postprocess.py eval_negatives [--device cpu|cuda]
    python scripts/postprocess.py roc  <config_name1> <config_name2> ...
    python scripts/postprocess.py delta <method1> <method2> ...

Examples:
    python scripts/postprocess.py list
    python scripts/postprocess.py eval_negatives --device cuda
    python scripts/postprocess.py roc bdlm-delta-2.0-random kth-random unigram-delta-3.0-random
    python scripts/postprocess.py delta bdlm-random kgw-random unigram-random

The 'eval_negatives' command must be run before 'roc' or 'delta' to compute
real negative detection scores by running each watermark detector on the
unwatermarked ('none') completions. Results are cached via resolve_output_path.
Without this step, synthetic scores from theoretical null distributions will be used
as a fallback (with a warning).
"""

import json
import os
import sys
import hashlib
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import math

from dlm_watermark.utils.file_io import resolve_output_path

# ================= Configuration =================

DATA_FILE = resolve_output_path('watermark_results.jsonl')

# Cache file for negative detection scores (computed by eval_negatives)
NEG_CACHE_FILE = resolve_output_path('neg_detection_cache.json')

# Minimum token count for filtering
MIN_TOKENS = 200

# Default parameters for each watermark type (used for naming: omit if value matches default)
DEFAULT_PARAMS = {
    'BDLM': {
        'gamma': 0.5,
        'offset': 25.0,
        'context_len': 25.0,
        'topk': 40.0,
    },
    'Unigram': {
        'gamma': 0.25,
        'seed': 42.0,
    },
    'KTH': {
        'seed': 42.0,
        'key_len': 200.0,
    },
    'DiffusionKGW_Optimal_Gaussian': {
        'topk': 50.0,
        'seeding_scheme': 'sumhash',
        'greenlist_type': 'bernoulli',
        'enforce_kl': 0.0,
        'n_iter': 1.0,
    },
    'none': {},
}

# Short names for watermark types
TYPE_SHORT_NAMES = {
    'BDLM': 'bdlm',
    'Unigram': 'unigram',
    'KTH': 'kth',
    'DiffusionKGW_Optimal_Gaussian': 'opt_kgw',
    'none': 'none',
}

# Parameters that vary per-method and should appear in the config name
# (delta varies as sub-config for delta plots, remasking is always shown)
CONFIG_PARAMS_ORDER = ['delta', 'gamma', 'seed', 'key_len', 'context_len',
                       'offset', 'topk', 'seeding_scheme', 'greenlist_type',
                       'enforce_kl', 'n_iter']

# Color palette for plotting
COLORS = [
    '#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3',
    '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd',
]

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']


# ================= Data Loading =================

def load_all_data(filepath=None):
    """Load all records from the JSONL file."""
    if filepath is None:
        filepath = DATA_FILE
    filepath = os.path.abspath(filepath)

    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _get_config_dict(record):
    """
    Extract the configuration-relevant fields from a record.
    Returns a dict of non-None config parameters.
    """
    fields = ['watermark_type', 'delta', 'gamma', 'seed', 'key_len',
              'context_len', 'offset', 'topk', 'seeding_scheme',
              'convolution_kernel', 'greenlist_type', 'model_name',
              'remasking', 'greenlist_params', 'enforce_kl', 'n_iter']
    cfg = {}
    for f in fields:
        v = record.get(f)
        if v is not None:
            # Normalize list/dict for comparison
            if isinstance(v, list):
                v = tuple(v)
            elif isinstance(v, dict):
                v = tuple(sorted(v.items()))
            cfg[f] = v
    return cfg


def _config_name(cfg):
    """
    Generate a human-readable config name from config dict.
    Omits default parameters for the watermark type.
    Format: <short_type>[-<param>-<value>]*[-<remasking>]
    """
    wt = cfg.get('watermark_type', 'unknown')
    short = TYPE_SHORT_NAMES.get(wt, wt.lower())
    defaults = DEFAULT_PARAMS.get(wt, {})
    remasking = cfg.get('remasking', 'random')

    if wt == 'none':
        return f"none-{remasking}"

    parts = [short]

    # Add non-default parameters in a defined order
    for param in CONFIG_PARAMS_ORDER:
        val = cfg.get(param)
        if val is None:
            continue
        default_val = defaults.get(param)
        if default_val is not None and val == default_val:
            continue
        # Format the value
        if isinstance(val, float) and val == int(val):
            val_str = f"{val:.1f}"
        elif isinstance(val, float):
            val_str = f"{val:.1f}"
        else:
            val_str = str(val)
        parts.append(f"{param}-{val_str}")

    # Always append remasking strategy
    parts.append(remasking)

    return '-'.join(parts)


def _method_name(cfg):
    """
    Generate a method-level name (without delta) for delta plots.
    Format: <short_type>[-<non-default-non-delta-params>]*[-<remasking>]
    """
    wt = cfg.get('watermark_type', 'unknown')
    short = TYPE_SHORT_NAMES.get(wt, wt.lower())
    defaults = DEFAULT_PARAMS.get(wt, {})
    remasking = cfg.get('remasking', 'random')

    if wt == 'none':
        return f"none-{remasking}"

    parts = [short]

    for param in CONFIG_PARAMS_ORDER:
        if param == 'delta':
            continue  # Skip delta for method-level grouping
        val = cfg.get(param)
        if val is None:
            continue
        default_val = defaults.get(param)
        if default_val is not None and val == default_val:
            continue
        if isinstance(val, float) and val == int(val):
            val_str = f"{val:.1f}"
        elif isinstance(val, float):
            val_str = f"{val:.1f}"
        else:
            val_str = str(val)
        parts.append(f"{param}-{val_str}")

    parts.append(remasking)
    return '-'.join(parts)


def build_config_index(records):
    """
    Build an index: config_name -> list of records.
    Also returns config_name -> config_dict mapping.
    """
    index = defaultdict(list)
    cfg_map = {}

    for rec in records:
        cfg = _get_config_dict(rec)
        name = _config_name(cfg)
        index[name].append(rec)
        if name not in cfg_map:
            cfg_map[name] = cfg

    return dict(index), cfg_map


def build_method_index(records):
    """
    Build an index: method_name -> { delta_value -> list of records }.
    Also returns method_name -> config_dict (without delta) mapping.
    """
    index = defaultdict(lambda: defaultdict(list))
    cfg_map = {}

    for rec in records:
        cfg = _get_config_dict(rec)
        method = _method_name(cfg)
        delta = cfg.get('delta', None)
        index[method][delta].append(rec)
        if method not in cfg_map:
            cfg_map[method] = cfg

    return {k: dict(v) for k, v in index.items()}, cfg_map


# ================= Feature 1: List Configurations =================

def list_configs(records=None, filepath=None):
    """List all available configurations and sample counts."""
    if records is None:
        records = load_all_data(filepath)

    config_index, cfg_map = build_config_index(records)

    print(f"\n{'='*70}")
    print(f"{'Configuration Name':<50} {'Samples':>8}")
    print(f"{'='*70}")

    # Sort by watermark type then by name
    sorted_names = sorted(config_index.keys())
    for name in sorted_names:
        count = len(config_index[name])
        print(f"{name:<50} {count:>8}")

    print(f"{'='*70}")
    print(f"{'Total':<50} {len(records):>8}")
    print()

    # Also list method-level groupings for delta plots
    method_index, _ = build_method_index(records)
    print(f"\n{'='*70}")
    print(f"{'Method Name (for delta plots)':<40} {'Deltas':>30}")
    print(f"{'='*70}")
    sorted_methods = sorted(method_index.keys())
    for method in sorted_methods:
        deltas = sorted([d for d in method_index[method].keys() if d is not None])
        delta_str = ', '.join(f"{d:.1f}" for d in deltas)
        if None in method_index[method]:
            delta_str = 'N/A (no delta)' + (f', {delta_str}' if delta_str else '')
        print(f"{method:<40} {delta_str:>30}")
    print(f"{'='*70}")

    return config_index


# ================= Detection Score Utilities =================

# Watermark types that use p_value (lower = more watermarked) instead of z_score.
P_VALUE_TYPES = {'KTH'}

# Mapping from watermark_type strings stored in JSONL (via get_key_params())
# to WatermarkType enum values used by watermark_factory.
_JSONL_TO_ENUM = {
    'BDLM': 'BDLM',
    'KTH': 'KTH',
    'Unigram': 'Unigram',
    'DiffusionKGW_Optimal_Gaussian': 'Ours',
    'KGW': 'KGW',
    'none': 'None',
}


def _filter_records(records, min_tokens=MIN_TOKENS):
    """Filter records by minimum length (completion length in tokens)."""
    filtered = []
    for rec in records:
        length = rec.get('length', 0)
        if length is not None and length >= min_tokens:
            filtered.append(rec)
    return filtered


def _get_detection_scores(records, watermark_type):
    """
    Extract detection scores from records, normalized so higher = more watermarked.
    - For z_score-based methods: returns z_scores directly.
    - For p_value-based methods (e.g. KTH): returns (1 - p_value).
    """
    if watermark_type in P_VALUE_TYPES:
        return [1.0 - r['p_value'] for r in records if r.get('p_value') is not None]
    else:
        return [r['z_score'] for r in records if r.get('z_score') is not None]


def _completion_hash(text):
    """Compute a short hash for a completion text to use as cache key."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _detector_cache_key(cfg):
    """
    Build the cache key for negative score storage from a config dict.
    Delta does NOT affect detection, so the key is watermark_type|remasking.
    This means all delta variants of the same method share one cache entry.

    Accepts the full config dict for maintainability — if additional fields
    ever become relevant to the detector identity, only this function needs
    to change.
    """
    wt = cfg.get('watermark_type', 'unknown')
    remasking = cfg.get('remasking', 'random')
    return f"{wt}|{remasking}"


def _load_neg_cache():
    """
    Load cached negative detection scores from disk.

    Cache structure (sequence-level):
    {
        "BDLM|random": {
            "watermark_type": "BDLM",
            "remasking": "random",
            "score_field": "z_score",
            "scores": {
                "<completion_hash>": <score_float>,
                ...
            }
        },
        ...
    }
    """
    cache_path = NEG_CACHE_FILE
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, 'r') as f:
        return json.load(f)


def _save_neg_cache(cache):
    """Save negative detection scores cache to disk."""
    cache_path = NEG_CACHE_FILE
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


def _build_detector_from_record(cfg, tokenizer, device='cpu'):
    """
    Build a watermark detector from a config dict (extracted from JSONL records)
    by reusing the existing watermark_factory.

    Steps:
    1. Map the JSONL watermark_type to WatermarkType enum
    2. Construct the matching Configuration dataclass from record params
    3. Call load_watermark_from_config()
    """
    from dlm_watermark.configs import (
        WatermarkType,
        BDLMConfiguration,
        UnigramConfiguration,
        KTHConfiguration,
        OurWatermarkConfiguration,
        KGWConfiguration,
    )
    from dlm_watermark.watermarks.watermark_factory import load_watermark_from_config

    wt_str = cfg.get('watermark_type')
    enum_val = _JSONL_TO_ENUM.get(wt_str)
    if enum_val is None:
        raise ValueError(f"Unknown watermark_type in record: {wt_str}")

    watermark_type = WatermarkType(enum_val)

    # Build the appropriate configuration object from record fields
    if wt_str == 'BDLM':
        wm_config = BDLMConfiguration(
            delta=float(cfg.get('delta', 2.0)),
            gamma=float(cfg.get('gamma', 0.25)),
            offset=int(cfg.get('offset', 32)),
            context_len=int(cfg.get('context_len', 32)),
            topk=int(cfg.get('topk', 40)),
        )
    elif wt_str == 'Unigram':
        wm_config = UnigramConfiguration(
            delta=float(cfg.get('delta', 2.0)),
            gamma=float(cfg.get('gamma', 0.25)),
            seed=int(cfg.get('seed', 42)),
        )
    elif wt_str == 'KTH':
        wm_config = KTHConfiguration(
            key_len=int(cfg.get('key_len', 200)),
            seed=int(cfg.get('seed', 42)),
        )
    elif wt_str == 'DiffusionKGW_Optimal_Gaussian':
        greenlist_params = cfg.get('greenlist_params')
        if isinstance(greenlist_params, tuple):
            greenlist_params = dict(greenlist_params)
        elif greenlist_params is None:
            greenlist_params = {"gamma": 0.25}
        conv_kernel = cfg.get('convolution_kernel', (-1,))
        if isinstance(conv_kernel, tuple):
            conv_kernel = list(conv_kernel)
        wm_config = OurWatermarkConfiguration(
            delta=float(cfg.get('delta', 2.0)),
            enforce_kl=bool(cfg.get('enforce_kl', True)),
            convolution_kernel=conv_kernel,
            greenlist_type=str(cfg.get('greenlist_type', 'bernoulli')),
            greenlist_params=greenlist_params,
            topk=int(cfg.get('topk', 100)),
            n_iter=int(cfg.get('n_iter', 1)),
            seeding_scheme=str(cfg.get('seeding_scheme', 'sumhash')),
        )
    elif wt_str == 'KGW':
        conv_kernel = cfg.get('convolution_kernel', (-2, -1))
        if isinstance(conv_kernel, tuple):
            conv_kernel = list(conv_kernel)
        wm_config = KGWConfiguration(
            gamma=float(cfg.get('gamma', 0.25)),
            delta=float(cfg.get('delta', 2.0)),
            convolution_kernel=conv_kernel,
            seeding_scheme=str(cfg.get('seeding_scheme', 'sumhash')),
        )
    else:
        raise ValueError(f"Cannot build config for watermark_type: {wt_str}")

    return load_watermark_from_config(
        config=wm_config,
        tokenizer=tokenizer,
        watermark_type=watermark_type,
        device=device,
    )


def eval_negatives(records=None, filepath=None, device='cpu'):
    """
    Evaluate all watermark detectors on 'none' (unwatermarked) completions
    to obtain real negative detection scores. Results are cached to disk.

    Cache is keyed by (watermark_type, remasking) — delta does NOT affect
    detection, so all delta variants of a method share one cache entry.
    Scores are stored per-completion (keyed by content hash), enabling
    incremental updates when new negative samples are added.
    """
    import torch
    from transformers import AutoTokenizer as HFAutoTokenizer
    from tqdm import tqdm

    if records is None:
        records = load_all_data(filepath)

    config_index, cfg_map = build_config_index(records)

    # Deduplicate configs at detector level: group by (watermark_type, remasking).
    # Different delta values produce the same detector, so we only need one
    # representative config per (watermark_type, remasking) pair.
    detector_configs = {}  # detector_cache_key -> (representative_config_name, cfg)
    for name, cfg in cfg_map.items():
        if cfg.get('watermark_type') == 'none':
            continue
        wt = cfg.get('watermark_type')
        remasking = cfg.get('remasking', 'random')
        det_key = _detector_cache_key(cfg)
        if det_key not in detector_configs:
            detector_configs[det_key] = (name, cfg)

    if not detector_configs:
        print("No watermark configs found in data.")
        return

    # Load existing cache
    cache = _load_neg_cache()

    # Group by model_name to share tokenizer
    model_groups = defaultdict(list)
    for det_key, (rep_name, cfg) in detector_configs.items():
        model_name = cfg.get('model_name', 'unknown')
        model_groups[model_name].append((det_key, rep_name, cfg))

    for model_name, det_entries in model_groups.items():
        print(f"\nLoading tokenizer for model: {model_name}")
        try:
            tokenizer = HFAutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"  Failed to load tokenizer for {model_name}: {e}")
            print(f"  Skipping {len(det_entries)} detector configs.")
            continue

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for det_key, rep_name, cfg in det_entries:
            remasking = cfg.get('remasking', 'random')
            wt = cfg.get('watermark_type')
            neg_config_name = f"none-{remasking}"
            score_field = 'p_value' if wt in P_VALUE_TYPES else 'z_score'

            # Get 'none' completions for this remasking strategy
            if neg_config_name not in config_index:
                print(f"  {det_key}: no '{neg_config_name}' records found. Skipping.")
                continue

            none_records = config_index[neg_config_name]
            completions = [r['completion'] for r in none_records if r.get('completion')]

            if not completions:
                print(f"  {det_key}: no completions in '{neg_config_name}'. Skipping.")
                continue

            # Load or initialize cache entry for this detector
            if det_key not in cache:
                cache[det_key] = {
                    'watermark_type': wt,
                    'remasking': remasking,
                    'score_field': score_field,
                    'scores': {},
                }
            cached_scores = cache[det_key]['scores']

            # Identify new completions that are not yet cached
            new_completions = []
            for comp in completions:
                h = _completion_hash(comp)
                if h not in cached_scores:
                    new_completions.append((h, comp))

            if not new_completions:
                print(f"  {det_key}: all {len(completions)} completions already cached. Skipping.")
                continue

            print(f"  {det_key} (via {rep_name}): {len(new_completions)} new / "
                  f"{len(completions)} total completions to evaluate...")

            # Build detector via watermark_factory (using representative config)
            try:
                detector = _build_detector_from_record(cfg, tokenizer, device=device)
            except Exception as e:
                print(f"  {det_key}: failed to build detector: {e}")
                continue

            computed = 0
            for h, comp in tqdm(new_completions, desc=f"    {det_key}", leave=False):
                try:
                    inputs = tokenizer(comp, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    detection_output = detector.detect(**inputs)
                    score = detection_output.get(score_field)
                    if score is not None:
                        cached_scores[h] = float(score)
                        computed += 1
                except Exception:
                    continue

            print(f"    -> {computed} new scores computed "
                  f"(total cached: {len(cached_scores)}).")

            # Save after each detector (in case of interruption)
            _save_neg_cache(cache)

    _save_neg_cache(cache)
    print(f"\nNegative detection cache saved to: {NEG_CACHE_FILE}")
    print(f"Total cached detectors: {len(cache)}")


def _get_neg_scores(config_index, remasking, min_tokens, watermark_type, n_samples=None,
                    config_name=None):
    """
    Get negative detection scores for ROC/threshold computation.

    Priority order:
    1. Cached real detection scores (from eval_negatives) — looked up by
       the canonical detector key (watermark_type|remasking).
    2. Real z_scores/p_values from 'none' records (if they exist in the data)
    3. Synthetic fallback from theoretical null distribution (with warning)

    Args:
        config_index: Config name -> records mapping.
        remasking: Remasking strategy string.
        min_tokens: Minimum token filter.
        watermark_type: Watermark type string.
        n_samples: Number of synthetic samples (fallback only).
        config_name: For logging/fallback purposes only. Cache lookup uses
                     (watermark_type, remasking) directly.
    """
    neg_name = f"none-{remasking}"
    neg_records = []
    if neg_name in config_index:
        neg_records = _filter_records(config_index[neg_name], min_tokens)

    # 1. Try cached real detection scores (canonical key lookup)
    cache = _load_neg_cache()
    det_key = _detector_cache_key({'watermark_type': watermark_type, 'remasking': remasking})
    cached_entry = cache.get(det_key)
    if cached_entry is not None:
        scores_dict = cached_entry.get('scores', {})
        if scores_dict:
            scores = list(scores_dict.values())
            # Convert p_value scores to (1 - p_value) for consistency
            if watermark_type in P_VALUE_TYPES:
                scores = [1.0 - s for s in scores]
            return scores, neg_records

    # 2. Try real scores from 'none' records in data
    if neg_records:
        neg_scores = _get_detection_scores(neg_records, watermark_type)
        if neg_scores:
            return neg_scores, neg_records

    # 3. Synthetic fallback (with warning)
    print(f"  Warning: Using synthetic negative scores for {config_name or watermark_type}. "
          f"Run 'eval_negatives' for accurate results.")
    if n_samples is None:
        n_samples = 1000
    rng = np.random.RandomState(42)
    if watermark_type in P_VALUE_TYPES:
        neg_scores = rng.uniform(0, 1, n_samples).tolist()
    else:
        neg_scores = rng.standard_normal(n_samples).tolist()

    return neg_scores, neg_records


# ================= Feature 2: ROC Plot =================


def draw_roc(config_names, records=None, filepath=None, output='roc_comparison.png',
             min_tokens=MIN_TOKENS):
    """
    Draw ROC curves for the specified configurations.

    Each watermarked config is compared against synthetic N(0,1) negatives
    (or real 'none' z_scores if available) to compute the ROC curve.

    Args:
        config_names: List of config name strings (as shown by list_configs).
        records: Pre-loaded records (optional).
        filepath: Path to JSONL file (optional).
        output: Output image filename.
        min_tokens: Minimum token/length threshold.
    """
    if records is None:
        records = load_all_data(filepath)

    config_index, cfg_map = build_config_index(records)

    # Validate config names
    available = set(config_index.keys())
    for name in config_names:
        if name not in available:
            print(f"Warning: Config '{name}' not found. Available configs:")
            suggestions = [n for n in available if name.split('-')[0] in n]
            for s in suggestions[:10]:
                print(f"  {s}")
            return

    plt.figure(figsize=(10, 8))

    for i, name in enumerate(config_names):
        cfg = cfg_map[name]
        remasking = cfg.get('remasking', 'random')
        wt = cfg.get('watermark_type', 'unknown')

        # Get positive samples (watermarked)
        pos_records = _filter_records(config_index[name], min_tokens)
        pos_scores = _get_detection_scores(pos_records, wt)

        if not pos_scores:
            print(f"Warning: No detection scores for '{name}' after filtering. Skipping.")
            continue

        # Get negative detection scores
        neg_scores, neg_records = _get_neg_scores(
            config_index, remasking, min_tokens, wt, n_samples=len(pos_scores),
            config_name=name,
        )

        # Compute ROC
        y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
        y_scores = pos_scores + neg_scores
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute average PPL for the label
        pos_ppls = [r['ppl'] for r in pos_records if r.get('ppl') is not None]
        avg_ppl = np.mean(pos_ppls) if pos_ppls else 0.0

        color = COLORS[i % len(COLORS)]
        label = f"{name}\nAUC: {roc_auc:.4f} | PPL: {avg_ppl:.2f}"

        plt.semilogx(fpr, tpr, color=color, lw=2.5, label=label)

    # Random guess baseline
    random_fpr = np.logspace(-2.5, 0, 100)
    # Get baseline PPL from 'none' configs
    none_ppls = []
    for cname in config_index:
        if cname.startswith('none-'):
            for rec in _filter_records(config_index[cname], min_tokens):
                if rec.get('ppl') is not None:
                    none_ppls.append(rec['ppl'])
    baseline_ppl = np.mean(none_ppls) if none_ppls else 0.0

    label_random = f"Unwatermarked (TPR=FPR)\nBaseline PPL: {baseline_ppl:.2f}"
    plt.semilogx(random_fpr, random_fpr, color='gray', linestyle='--', lw=2, label=label_random)

    # Style
    plt.xlabel("False Positive Rate (Log Scale)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title(f"Watermark Detection ROC Comparison\n(Filter: Length >= {min_tokens})", fontsize=14)
    plt.xlim([math.pow(10, -2.5), 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    plt.legend(loc="lower right", fontsize=9, frameon=True, shadow=True, borderpad=1)
    plt.tight_layout()
    output_path = resolve_output_path(output)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC plot saved to: {output_path}")
    plt.close()


# ================= Feature 3: Delta Plot =================

def draw_delta(method_names, records=None, filepath=None, output='delta_comparison.png',
               min_tokens=MIN_TOKENS):
    """
    Draw delta plot: x-axis = log(PPL), y-axis = TPR@1%FPR.
    Each method is a curve, each delta value is a point.

    Args:
        method_names: List of method name strings (as shown by list_configs method listing).
        records: Pre-loaded records (optional).
        filepath: Path to JSONL file (optional).
        output: Output image filename.
        min_tokens: Minimum token/length threshold.
    """
    if records is None:
        records = load_all_data(filepath)

    method_index, method_cfg_map = build_method_index(records)
    config_index, _ = build_config_index(records)

    # Validate method names
    available = set(method_index.keys())
    for name in method_names:
        if name not in available:
            print(f"Warning: Method '{name}' not found. Available methods:")
            suggestions = [n for n in available if name.split('-')[0] in n]
            for s in suggestions[:10]:
                print(f"  {s}")
            return

    plt.figure(figsize=(7, 5))
    all_neg_ppls = []

    for i, method in enumerate(method_names):
        delta_groups = method_index[method]
        cfg = method_cfg_map[method]
        remasking = cfg.get('remasking', 'random')

        wt = cfg.get('watermark_type', 'unknown')

        # Get negative detection scores for threshold calculation
        neg_scores, neg_records = _get_neg_scores(
            config_index, remasking, min_tokens, wt, n_samples=1000,
            config_name=method,
        )
        neg_ppls = [r['ppl'] for r in neg_records if r.get('ppl') is not None]

        if not neg_scores:
            print(f"Warning: Empty negative data for {method}. Skipping.")
            continue

        all_neg_ppls.extend(neg_ppls)

        # Threshold @ 1% FPR (99th percentile since higher score = more watermarked)
        threshold = np.percentile(neg_scores, 99)

        x_values = []  # log(PPL)
        y_values = []  # TPR

        # Sort by delta
        deltas = sorted([d for d in delta_groups.keys() if d is not None])
        for delta in deltas:
            pos_records = _filter_records(delta_groups[delta], min_tokens)
            pos_scores = _get_detection_scores(pos_records, wt)
            pos_ppls = [r['ppl'] for r in pos_records if r.get('ppl') is not None]

            if not pos_scores or not pos_ppls:
                print(f"  Skipping {method} delta={delta:.1f}: empty data after filtering.")
                continue

            tpr = np.sum(np.array(pos_scores) > threshold) / len(pos_scores)
            avg_ppl = np.mean(pos_ppls)
            log_ppl = np.log(avg_ppl)

            x_values.append(log_ppl)
            y_values.append(tpr)

            print(f"  {method} delta={delta:.1f}: TPR={tpr:.4f}, log(PPL)={log_ppl:.4f}")

        if x_values:
            color = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]
            plt.plot(x_values, y_values,
                     label=method,
                     color=color,
                     marker=marker,
                     linewidth=2,
                     markersize=7,
                     alpha=0.9)

    # Unwatermarked baseline vertical line
    if all_neg_ppls:
        baseline_log_ppl = np.log(np.mean(all_neg_ppls))
        plt.axvline(x=baseline_log_ppl,
                     color='red',
                     linestyle='--',
                     linewidth=2,
                     label='log(Perplexity)\nUnwatermarked LLM')

    # Style
    ax = plt.gca()
    ax.grid(True, linestyle='-', alpha=0.2, color='gray')
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$\leftarrow$ log(Perplexity)", fontsize=14)
    ax.set_ylabel(r"TPR @ 1% FPR $\rightarrow$", fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(frameon=False, fontsize=10, loc='lower right')
    plt.tight_layout()
    output_path = resolve_output_path(output)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Delta plot saved to: {output_path}")
    plt.close()


# ================= CLI =================

def main():
    parser = argparse.ArgumentParser(
        description='Watermark experiment results post-processing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data', type=str, default=None,
                        help=f'Path to JSONL data file (default: {DATA_FILE})')
    parser.add_argument('--min-tokens', type=int, default=MIN_TOKENS,
                        help=f'Minimum token count filter (default: {MIN_TOKENS})')

    subparsers = parser.add_subparsers(dest='command', help='Sub-command')

    # list
    sub_list = subparsers.add_parser('list', help='List all configurations and sample counts')

    # eval_negatives
    sub_eval = subparsers.add_parser('eval_negatives',
                                     help='Evaluate watermark detectors on unwatermarked text')
    sub_eval.add_argument('--device', type=str, default='cpu',
                          help='Device for detection (default: cpu)')

    # roc
    sub_roc = subparsers.add_parser('roc', help='Draw ROC plot for specified configurations')
    sub_roc.add_argument('configs', nargs='+', help='Config names to plot (as shown by "list")')
    sub_roc.add_argument('-o', '--output', type=str, default='roc_comparison.png',
                         help='Output filename (default: roc_comparison.png)')

    # delta
    sub_delta = subparsers.add_parser('delta', help='Draw delta plot for specified methods')
    sub_delta.add_argument('methods', nargs='+', help='Method names to plot (as shown by "list")')
    sub_delta.add_argument('-o', '--output', type=str, default='delta_comparison.png',
                           help='Output filename (default: delta_comparison.png)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Load data once
    records = load_all_data(args.data)

    if args.command == 'list':
        list_configs(records)
    elif args.command == 'eval_negatives':
        eval_negatives(records=records, device=args.device)
    elif args.command == 'roc':
        draw_roc(args.configs, records=records, output=args.output, min_tokens=args.min_tokens)
    elif args.command == 'delta':
        draw_delta(args.methods, records=records, output=args.output, min_tokens=args.min_tokens)


if __name__ == '__main__':
    main()
