from dlm_watermark.configs import MainConfiguration
from dlm_watermark.watermarks.watermark_factory import load_watermark_from_config
from dlm_watermark.watermark_eval import Evaluator
from dlm_watermark.models.model_factory import load_model
from dlm_watermark.utils.file_io import resolve_output_path
from dlm_watermark.utils.config_utils import load_config
import yaml
import argparse
import json

def parse_args():
    
    parser = argparse.ArgumentParser(description="Ablate generation parameters")
    
    # --- 配置加载方式（三选一或组合使用） ---
    parser.add_argument("--config", type=str, default=None, help="Path to a monolithic configuration file (legacy mode)")
    parser.add_argument("--layers", type=str, nargs="*", default=None, help="Paths to decoupled config layer files to merge (model, watermark, dataset)")
    parser.add_argument("--override", action="append", default=[], help="Dot-notation override, e.g. 'watermark_config.delta=3.0'. Can be specified multiple times.")
    
    # --- 实验命名 ---
    parser.add_argument("--name", type=str, default=None, help="Name of the ablation experiment (used to construct output path)")
    
    # --- 旧版 CLI 参数（向后兼容，内部转为 override） ---
    parser.add_argument("--delta", type=float, default=None, help="Delta value for watermark")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma value for watermark")
    parser.add_argument("--kernel", type=str, default=None, help="Convolution kernel, e.g. '[-1]'")
    parser.add_argument("--topk", type=int, default=None, help="Top-k value for watermark")
    parser.add_argument("--topk_greenify", type=int, default=None, help="Top-k value for greenify")
    parser.add_argument("--topk_hashes", type=int, default=None, help="Top-k value for hashes")
    parser.add_argument("--n_iter", type=int, default=None)
    parser.add_argument("--seeding_scheme", type=str, default=None, help="Seeding scheme for watermark")
    parser.add_argument("--disable_generation", action="store_true", help="Disable generation during evaluation")
    parser.add_argument("--ppl", action="store_true", help="Evaluate perplexity")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    
    # OptimalKGWGeneral specific parameters
    parser.add_argument("--enforce_kl", action="store_true", help="Enforce KL divergence constraint in the optimization")
    parser.add_argument("--no_enforce_kl", action="store_true", help="Disable KL divergence constraint in the optimization")
    parser.add_argument("--greenlist_type", type=str, default=None, help="Type of greenlist to use. Options: 'bernoulli', 'gaussian'")
    parser.add_argument("--greenlist_params", type=str, default=None, help="Parameters for the greenlist as JSON string (e.g., '{\"gamma\": 0.25}')")
    
    # Generation specific parameters
    parser.add_argument("--temperature", type=float, default=None, help="Temperature parameter")
        
    args = parser.parse_args()
    return args


def build_legacy_overrides(args) -> list:
    """
    将旧版 CLI 参数转换为 --override 格式，保持向后兼容。
    只有非 None 的参数才会被转换。
    """
    overrides = []
    
    if args.delta is not None:
        overrides.append(f"watermark_config.delta={args.delta}")
    if args.gamma is not None:
        overrides.append(f"watermark_config.gamma={args.gamma}")
    if args.kernel is not None:
        overrides.append(f"watermark_config.convolution_kernel={args.kernel}")
    if args.topk is not None:
        overrides.append(f"watermark_config.topk={args.topk}")
    if args.n_iter is not None:
        overrides.append(f"watermark_config.n_iter={args.n_iter}")
    if args.seeding_scheme is not None:
        overrides.append(f"watermark_config.seeding_scheme={args.seeding_scheme}")
    if args.num_samples is not None:
        overrides.append(f"evaluation_config.num_samples={args.num_samples}")
    if args.temperature is not None:
        overrides.append(f"model_configuration.temperature={args.temperature}")
    if args.enforce_kl:
        overrides.append("watermark_config.enforce_kl=true")
    elif args.no_enforce_kl:
        overrides.append("watermark_config.enforce_kl=false")
    if args.greenlist_type is not None:
        overrides.append(f"watermark_config.greenlist_type={args.greenlist_type}")
    if args.greenlist_params is not None:
        overrides.append(f"watermark_config.greenlist_params={args.greenlist_params}")
    
    return overrides


def main():
    
    args = parse_args()
    
    # 合并旧版CLI参数和显式override
    legacy_overrides = build_legacy_overrides(args)
    all_overrides = legacy_overrides + (args.override or [])
    
    # 统一加载配置
    config = load_config(
        base=args.config,
        layers=args.layers,
        overrides=all_overrides if all_overrides else None,
    )
    
    # 设置输出路径
    if args.name:
        if args.delta == 0:
            config.evaluation_config.save_path = resolve_output_path(f"output/{args.name}/watermark_ablation_no_watermark.jsonl")
        else:
            config.evaluation_config.save_path = resolve_output_path(f"output/{args.name}/watermark_ablation.jsonl")
    
    if config.watermark_type.value == "None":
        print("No watermark type specified -- evaluating without watermark.")
    
    print(config.short_summary())

    additional_info = {
        "model_name": config.model_configuration.model_name,
    }

    evaluator = Evaluator(config=config.evaluation_config)
    
    if not args.disable_generation:   
        delta_val = getattr(config.watermark_config, 'delta', None)
        print(f"Evaluating with config: watermark_type={config.watermark_type.value}, delta={delta_val}")
        model, tokenizer = load_model(config.model_configuration, tokenizer_only=False)    

        # Generation parameters
        if args.temperature is not None:
            model.config.temperature = args.temperature

        if args.delta == 0:
            watermark = None
        else:
            watermark = load_watermark_from_config(config=config.watermark_config, tokenizer=tokenizer, watermark_type=config.watermark_type)
        evaluator.evaluate_watermark(model, tokenizer, watermark, additional_info=additional_info)
        
    if args.ppl:
        evaluator.evaluate_ppl()


if __name__ == "__main__":
    main()
