from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import json

from .models.model import DiffusionLM
from .data_processing import tokenize_dataset_with_chat, convert_sft_dataset, tokenize_dataset
from .configs import EvaluationConfiguration, EvaluationDataset
from .quality_evaluations.ppl import compute_ppl
from .quality_evaluations.metrics import compute_seq_rep_n
from .quality_evaluations.judge import get_gpt4_grades
from .utils.file_io import file_lock, safe_append_jsonl, safe_write_df_json, safe_patch_df_columns


def get_dataset(dataset_config: EvaluationDataset):
    config_dict = dataset_config.model_dump()

    # Sanitize the config_dict
    fields_to_keep = ["path", "split", "streaming", "name"]
    config_dict = {k: v for k, v in config_dict.items() if k in fields_to_keep}
    
    data_path = config_dict["path"]
    if data_path.endswith(".jsonl"):
        ds = Dataset.from_json(
            data_path,
            split=config_dict.get("split", "train"),
            streaming=config_dict.get("streaming", False),
        )

    else: 
        try: # Dataset saved on disk are loaded with load_from_disk
            ds = load_dataset(**config_dict)
        except ValueError as e:
            config_dict["dataset_path"] = config_dict.pop("path", None)
            config_dict.pop("name", None)
            config_dict.pop("streaming", None)
            config_dict.pop("split", None)
            ds = load_from_disk(**config_dict)
            
        
    # Apply dataset specific transforms
    if dataset_config.path == "sentence-transformers/eli5":
        ds = ds.filter(
            lambda x: "?" in x["question"]
        )
        
        
    return ds


class Evaluator:
    def __init__(self, config: EvaluationConfiguration):
        self.config = config

    def prepare_dataset(self, dataset_config, tokenizer):
        dataset = get_dataset(dataset_config)
        user_prompt = dataset_config.user_prompt
    
    
        tokenizer_og_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"  # Ensure left padding for generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            

        if dataset_config.chat:
            if dataset_config.system_prompt:
                conversion_func = lambda example: {  # noqa: E731
                    "messages": [
                        {"role": "system", "content": dataset_config.system_prompt},
                        {"role": "user", "content": user_prompt.format(**example)},
                    ]
                }
            else:
                conversion_func = lambda example: {  # noqa: E731
                    "messages": [
                        {"role": "user", "content": user_prompt.format(**example)},
                    ]
                }
            dataset = convert_sft_dataset(dataset, conversion_func)
            dataset = tokenize_dataset_with_chat(
                dataset, tokenizer=tokenizer, max_length=dataset_config.max_length, pad = self.config.batch_size > 1
            )
        else:
            dataset = tokenize_dataset(
                dataset, tokenizer, user_prompt=user_prompt, length=dataset_config.max_length
            )
            
        tokenizer.padding_side = tokenizer_og_padding_side  # Restore original padding side
        
        # Filter according to min_length
        dataset = dataset.filter(
            lambda x: len(x["input_ids"]) >= dataset_config.min_length
        )
            
        return dataset

    def _evaluate_on_dataset(self, model: DiffusionLM, tokenizer, watermark, dataset, dataset_name):
                
        res = []
        device = model.device
        
        dataset = dataset.with_format("torch")
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
        )
        
        target_samples = self.config.num_samples
        target_batches = target_samples // self.config.batch_size

        # 原子性检查：在锁内读取已有行数，避免 TOCTOU 竞态
        with file_lock(self.config.save_path):
            existing = self.skip_if_exists(dataset_name)

        if existing >= target_samples:
            print(f"Skipping evaluation for {dataset_name} as results already exist.")
            return []

        pbar = tqdm(
            total=target_samples,
            initial=existing,
            desc=f"Evaluating on {dataset_name}",
            disable=not self.config.tqdm,
        )

        print(f"Saving results to: {os.path.abspath(self.config.save_path)}")
        generated_count = 0
        for i, row in enumerate(dataloader):
            # 每个 batch 开始前，原子性检查最新已有行数
            with file_lock(self.config.save_path):
                existing = self.skip_if_exists(dataset_name)
            
            if existing >= target_samples:
                break
            
            if i >= target_batches:
                break

            pbar.n = existing
            pbar.refresh()

            input_ids = row["input_ids"].to(device)
            attention_mask = row.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                
            # We generate a long enough sentence that is not repetitive -- current diffusion model tends to be repetitive
            is_valid_completion = False 
            failsafe_counter = 0
            while not is_valid_completion and failsafe_counter < 10:
                
                torch.cuda.empty_cache()
                
                generation_output = model.generate_diffusion(input_ids=input_ids, attention_mask=attention_mask, watermark=watermark)
                sequences = generation_output.output_sequences
                completion = tokenizer.batch_decode(sequences, skip_special_tokens=True)

                # Ensuring the output is valid
                is_valid_completion = True
                for batch_idx in range(len(completion)):
                    if not (len(completion[batch_idx]) >= 0.75 * sequences.shape[-1]): # Check it is long-enough
                        is_valid_completion = False
                        
                seq_rep_n = compute_seq_rep_n(completion, tokenizer, n=2)
                for batch_idx in range(len(seq_rep_n)): # Check it is diverse enough
                    if seq_rep_n[batch_idx] > 0.5:
                        is_valid_completion = False
                        break
                    
                if not is_valid_completion:
                    print(f"Re-generating for batch {i} due to invalid completion.")
                    print(f"Current completions: {completion}")
                    failsafe_counter += 1

            
            prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            encoded_completion = tokenizer(
                completion, return_tensors="pt", padding=True
            )["input_ids"]

            batch_res = []
            for batch_idx in range(len(input_ids)):   
                
                if watermark is None:
                    detection_output = {}
                else:
                    detection_output = watermark.detect(input_ids=sequences)
                
                line = {}
                line.update(detection_output)
                line.update({
                    "prompt": prompt[batch_idx],
                    "completion": completion[batch_idx],
                    "dataset_name": dataset_name,
                    "ages": generation_output.ages[batch_idx].detach().cpu().tolist(),
                    "length": len(encoded_completion[batch_idx]), 
                })
                
                batch_res.append(line)
            
            # 立即保存本 batch 的结果（save_results 内部持有文件锁）
            self.save_results(batch_res)
            res.extend(batch_res)
            generated_count += len(batch_res)

        pbar.n = existing + generated_count
        pbar.refresh()
        pbar.close()

        print(f"Saved {generated_count} results to: {os.path.abspath(self.config.save_path)}")
                
        return res

    def save_results(self, results):
        
        save_path = self.config.save_path
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except FileNotFoundError:
            pass
        # 使用文件锁安全追加写入，防止多进程并发冲突
        safe_append_jsonl(save_path, results, transform_fn=self.add_information)
                
    def skip_if_exists(self, dataset_name):
        if not self.config.skip_if_exists:
            return 0

        df = self.load_data_from_path()
        if df.empty:
            return 0

        # merge model + watermark config
        current_cfg = {**self.model_config_dict, **self.watermark_parameters_dict}

        # start with matching dataset_name
        masks = [df["dataset_name"] == dataset_name]

        for key, val in current_cfg.items():
            if key not in df.columns:
                continue

            col = df[key]
            # if column is numeric and val is int/float, compare numerically (with isclose for floats)
            if pd.api.types.is_numeric_dtype(col.dtype) and isinstance(val, (int, float)):
                # cast both to float, then use a small tolerance
                masks.append(np.isclose(col.astype(float), float(val), atol=1e-8))
            else:
                # fallback: compare as string
                masks.append(col.astype(str) == str(val))

        # combine all masks
        existing = np.logical_and.reduce(masks)
        return int(existing.sum())

    def add_information(self, result_line):
        result_line.update(self.watermark_parameters_dict)
        result_line.update(self.model_config_dict)
        
        return result_line
        
    def store_watermark_parameters(self, watermark):
        if watermark is None:
            self.watermark_parameters_dict = {"watermark_type": "none"}
            return
        self.watermark_parameters_dict = watermark.get_key_params()
        
    def store_generation_parameters(self, model: DiffusionLM):
        self.model_config_dict = model.config_dict()
        
        
    def store_additional_info(self, additional_info: dict = None):
        if additional_info is None:
            additional_info = {}
        else:
            additional_info = additional_info

        # Store additional info in the watermark parameters dict
        if not hasattr(self, 'watermark_parameters_dict'):
            self.watermark_parameters_dict = {}
        self.watermark_parameters_dict.update(additional_info)

    def evaluate_watermark(self, model, tokenizer, watermark = None, additional_info: dict = None):
        
        self.store_watermark_parameters(watermark)
        self.store_additional_info(additional_info)
        self.store_generation_parameters(model)
        
        for dataset_config in self.config.evaluation_datasets:
            dataset_name = dataset_config.path.split("/")[-1]
            dataset = self.prepare_dataset(dataset_config, tokenizer)
            res = self._evaluate_on_dataset(
                model,
                tokenizer,
                watermark,
                dataset,
                dataset_name,
            )
            if len(res) == 0:
                print(f"No (new) results for dataset {dataset_name}. Skipping.")
                continue
            
    def load_data_from_path(self):
        
        # Check if the file exists
        if not os.path.exists(self.config.save_path):
            return pd.DataFrame()
        
        df = pd.read_json(self.config.save_path, lines=True)
        return df

    def evaluate_ppl(self):
        
        # 使用文件锁保护读取阶段，获取需要计算的行索引和数据
        # 同时写入哨兵值标记这些行正在计算中，防止并发重复计算
        with file_lock(self.config.save_path):
            df = self.load_data_from_path()
            
            mask = pd.Series([True] * len(df))
            if self.config.skip_if_exists:
                if "ppl" in df.columns:
                    # 跳过已有值和正在计算中（哨兵值）的行
                    mask = df["ppl"].isnull()
                    if not mask.any():
                        print("PPL already evaluated. Skipping.")
                        return
            
            row_indices = df.index[mask].tolist()
            prompts = df.loc[row_indices, "prompt"].tolist()
            completions = df.loc[row_indices, "completion"].tolist()
            
            # 写入哨兵值标记为"正在计算"，防止其他进程重复计算同样的行
            if "ppl" not in df.columns:
                df["ppl"] = pd.NA
            df.loc[row_indices, "ppl"] = -1.0  # 哨兵值
            df.to_json(self.config.save_path, lines=True, orient="records")
        
        # 耗时计算在锁外进行，如果失败则清除哨兵值
        try:
            ppls = self._evaluate_ppl(prompts, completions)
        except Exception as e:
            # 清除哨兵值，将 ppl 列恢复为 NA，允许后续重试
            print(f"PPL evaluation failed: {e}. Clearing sentinel values.")
            safe_patch_df_columns(
                self.config.save_path,
                {"ppl": [pd.NA] * len(row_indices)},
                row_indices=row_indices,
            )
            raise

        # 写入时重新读取最新文件，只更新 ppl 列，避免覆盖并发修改
        safe_patch_df_columns(self.config.save_path, {"ppl": ppls}, row_indices=row_indices)

    def _evaluate_ppl(self, prompts, completions):
        return compute_ppl(
            self.config.ppl_model_name,
            prompts,
            completions,
            self.config.ppl_batch_size,
        )
        
    def evaluate_repetition(self, tokenizer):
        
        with file_lock(self.config.save_path):
            df = self.load_data_from_path()
            row_indices = df.index.tolist()
            completions = df.loc[row_indices, "completion"].tolist()
        
        # 耗时计算在锁外进行
        columns_to_patch = {}
        n_grams = [1,2,3]
        for n in n_grams:
            seq_rep_n = compute_seq_rep_n(completions, tokenizer, n=n)
            columns_to_patch[f"seq_rep_{n}"] = seq_rep_n

        # 写入时重新读取最新文件，使用显式 row_indices 避免行数不匹配
        safe_patch_df_columns(self.config.save_path, columns_to_patch, row_indices=row_indices)
        
    def evaluate_watermark_detection(self, tokenizer, watermark):
        
        with file_lock(self.config.save_path):
            df = self.load_data_from_path()
            row_indices = df.index.tolist()
            completions = df.loc[row_indices, "completion"].tolist()
        
        device = watermark.device
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 耗时计算在锁外进行
        results = []
        for completion in tqdm(completions, desc="Evaluating watermark detection", disable=not self.config.tqdm):

            # Evaluate the watermark
            inputs = tokenizer(completion, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            detection_output = watermark.detect(**inputs)
            results.append(detection_output)
            
        res_df = pd.DataFrame(results)
        
        # 写入时重新读取最新文件，使用显式 row_indices 避免行数不匹配
        columns_to_patch = {col: res_df[col].tolist() for col in res_df.columns}
        safe_patch_df_columns(self.config.save_path, columns_to_patch, row_indices=row_indices)

    def evaluate_gpt4_judge(self):
        
        with file_lock(self.config.save_path):
            df = self.load_data_from_path()
            
            if "gpt4_judge" in df.columns and not df["gpt4_judge"].isnull().all():
                print("GPT-4 evaluation already exists. Skipping.")
                return
            
            row_indices = df.index.tolist()
            prompts = df.loc[row_indices, "prompt"].tolist()
            completions = df.loc[row_indices, "completion"].tolist()
        
        # 耗时计算在锁外进行
        gpt4_scores = get_gpt4_grades(prompts, completions, is_completion_task=True)
        scores = []
        explanations = []

        for i, score_dict in enumerate(gpt4_scores):
            
            explanations.append(score_dict)
            
            comb_score = 0
            ctr = 0
            for key, val in score_dict.items():

                if key != "ethics":
                    if val["grade"] == -1:
                        continue
                    comb_score += val["grade"]
                    ctr += 1

            comb_score /= max(ctr, 1.0)

            scores.append(comb_score)
        
        # 写入时重新读取最新文件，使用显式 row_indices 避免行数不匹配
        safe_patch_df_columns(self.config.save_path, {"gpt4_judge": scores}, row_indices=row_indices)
