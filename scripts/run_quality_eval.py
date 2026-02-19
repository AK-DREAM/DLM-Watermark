import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dlm_watermark.configs import MainConfiguration
from dlm_watermark.watermarks.watermark_factory import load_watermark_from_config
from dlm_watermark.models.model_factory import load_model
from dlm_watermark.watermark_eval import Evaluator
from dlm_watermark.utils.file_io import resolve_output_path
from dlm_watermark.utils.config_utils import load_config
import yaml
import argparse
import glob

def parse_args():
    
    parser = argparse.ArgumentParser(description="Run quality evaluation on generated results")
    parser.add_argument("--config", type=str, default=None, help="Path to monolithic configuration file")
    parser.add_argument("--layers", type=str, nargs="*", default=None, help="Paths to decoupled config layer files to merge")
    parser.add_argument("--override", action="append", default=[], help="Dot-notation override, e.g. 'watermark_config.delta=3.0'")
    parser.add_argument("--path", type=str, help="Path to the data to evaluate")
    parser.add_argument("--paths", type=str, help="Path to the folder to evaluate")
    parser.add_argument("--pval", action="store_true", help="Evaluate watermark detection")
    parser.add_argument("--ppl", action="store_true", help="Evaluate perplexity")
    parser.add_argument("--rep", action="store_true", help="Evaluate repetition")
    parser.add_argument("--gpt", action="store_true", help="Use GPT-4 as a judge")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    config = load_config(
        base=args.config,
        layers=args.layers,
        overrides=list(args.override) if args.override else None,
    )

    config.evaluation_config.save_path = resolve_output_path("outputs/eval.json")
    
    if args.overwrite:
        config.evaluation_config.skip_if_exists = False
    
    evaluator = Evaluator(config=config.evaluation_config)
    
    if args.paths:
        paths = glob.glob(args.paths + "/**/*.jsonl", recursive=True)
    else:
        paths = [args.path]
        
    print(paths)
    
    for path in paths:
        print(f"Evaluating {path}")
        config.evaluation_config.save_path = path
    
        if args.pval:
            print(f"Evaluating watermark detection on {path}")
            _, tokenizer = load_model(model_config=config.model_configuration,tokenizer_only=True)
            watermark = load_watermark_from_config(config=config.watermark_config, tokenizer=tokenizer, watermark_type=config.watermark_type)
            evaluator.evaluate_watermark_detection(tokenizer, watermark)
            
        if args.ppl:
            evaluator.evaluate_ppl()
            
        if args.rep:
            _, tokenizer = load_model(model_config=config.model_configuration,tokenizer_only=True)
            evaluator.evaluate_repetition(tokenizer)

        if args.gpt:
            evaluator.evaluate_gpt4_judge()


if __name__ == "__main__":
    main()
