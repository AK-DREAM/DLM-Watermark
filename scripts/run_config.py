import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dlm_watermark.configs import MainConfiguration
from dlm_watermark.watermarks.watermark_factory import load_watermark_from_config
from dlm_watermark.watermark_eval import Evaluator
from dlm_watermark.models.model_factory import load_model
from dlm_watermark.utils.file_io import resolve_output_path
from dlm_watermark.utils.config_utils import load_config
import yaml
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="Run a single experiment from config")
    parser.add_argument("--config", type=str, default=None, help="Path to monolithic configuration file")
    parser.add_argument("--layers", type=str, nargs="*", default=None, help="Paths to decoupled config layer files to merge")
    parser.add_argument("--override", action="append", default=[], help="Dot-notation override, e.g. 'watermark_config.delta=3.0'")
    parser.add_argument("--output_path", type=str, help="Path to save the output results", default="outputs")
    parser.add_argument("--num_samples", type=int, default=None, help="Overwrite number of samples to evaluate.")
        
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    
    # 构建覆盖列表
    overrides = list(args.override or [])
    if args.num_samples is not None:
        overrides.append(f"evaluation_config.num_samples={args.num_samples}")
    
    config = load_config(
        base=args.config,
        layers=args.layers,
        overrides=overrides if overrides else None,
    )
    
    config.evaluation_config.save_path = resolve_output_path(f"{args.output_path}/results_ours_pos_new.jsonl")
    print(config.short_summary())

    additional_info = {
        "model_name": config.model_configuration.model_name,
    }

    evaluator = Evaluator(config=config.evaluation_config)
    
    model, tokenizer = load_model(config.model_configuration, tokenizer_only=False)
    watermark = load_watermark_from_config(config=config.watermark_config, tokenizer=tokenizer, watermark_type=config.watermark_type, device=config.model_configuration.device_map)
    evaluator.evaluate_watermark(model,tokenizer,watermark, additional_info=additional_info)


if __name__ == "__main__":
    main()
