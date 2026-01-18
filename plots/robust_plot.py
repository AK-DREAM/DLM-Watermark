import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import json

# ================= 配置区域 =================
# 请在这里替换你的实际文件路径
FILE_PATHS = {
    'unwatermarked': 'outputs/results_ours_neg.jsonl', # 未加水印样本 (用于计算阈值)
    'original': 'csv/bdlm_baseline_original.csv',             # 加水印但在 ratio=0 时的样本 (基准)
    'deletion': 'csv/bdlm_baseline_deletion.csv',             # Deletion 攻击数据
    'substitution': 'csv/bdlm_baseline_substitution.csv'      # Substitution 攻击数据
}


# 绘图风格设置
STYLE_CONFIG = {
    'deletion': {'label': 'Deletion', 'marker': 'o', 'linestyle': '-', 'color': '#4c72b0'},     # 深蓝实线 + 圆点
    'substitution': {'label': 'Substitution', 'marker': 'x', 'linestyle': '--', 'color': '#4c72b0'}, # 深蓝虚线 + 叉号
}

# 过滤阈值
MIN_TOKENS = 200
# ===========================================

def parse_ratio(param_str):
    """
    解析 csv 中的 attack_params 列。
    """
    try:
        if pd.isna(param_str):
            return 0.0
        if isinstance(param_str, dict):
            return float(param_str.get('parameter', 0.0))
        d = ast.literal_eval(param_str)
        return float(d.get('parameter', 0.0))
    except:
        return 0.0

def get_tpr_at_fixed_fpr(watermarked_scores, threshold):
    """
    计算给定阈值下的 TPR
    """
    if len(watermarked_scores) == 0:
        return 0.0
    tp_count = np.sum(np.array(watermarked_scores) > threshold)
    return tp_count / len(watermarked_scores)

def load_and_process_attack_data(filepath, threshold, baseline_tpr=None):
    """
    读取攻击数据 CSV，过滤掉 total_tokens < 200 的行，并计算 TPR
    """
    df = pd.read_csv(filepath)
    
    # --- 新增过滤逻辑 ---
    initial_count = len(df)
    if 'total_tokens' in df.columns:
        df = df[df['total_tokens'] >= MIN_TOKENS]
    else:
        print(f"Warning: 'total_tokens' column not found in {filepath}")
    
    # print(f"File {filepath}: Filtered {initial_count - len(df)} rows (< {MIN_TOKENS} tokens). Remaining: {len(df)}")
    # ------------------
    
    df['ratio'] = df['attack_params'].apply(parse_ratio)
    
    results = []
    grouped = df.groupby('ratio')
    
    for ratio, group in grouped:
        tpr = get_tpr_at_fixed_fpr(group['z_score'].values, threshold)
        results.append((ratio, tpr))
    
    results.sort(key=lambda x: x[0])
    
    ratios = [x[0] for x in results]
    tprs = [x[1] for x in results]
    
    if 0.0 not in ratios and baseline_tpr is not None:
        ratios.insert(0, 0.0)
        tprs.insert(0, baseline_tpr)
        
    return ratios, tprs

def main():
    # 1. 计算阈值 (Threshold @ 1% FPR)
    print("正在加载未加水印数据以计算阈值...")
    
    unwatermarked_scores = []
    with open(FILE_PATHS['unwatermarked'], 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # --- 新增过滤逻辑 ---
                # 确保 total_tokens 存在且 >= 200
                if item.get('total_tokens', 0) >= MIN_TOKENS:
                    unwatermarked_scores.append(item['z_score'])
                # ------------------
            except json.JSONDecodeError:
                continue
    
    if not unwatermarked_scores:
        raise ValueError("没有找到符合条件的未加水印样本 (total_tokens >= 200)")

    print(f"有效未加水印样本数: {len(unwatermarked_scores)}")
    
    # 计算 99 分位数作为阈值
    threshold = np.percentile(unwatermarked_scores, 95)
    print(f"Threshold @ 5% FPR: {threshold:.4f}")

    # 2. 计算 Ratio = 0 (Original) 的 TPR
    print("正在计算基准 TPR (Original)...")
    df_orig = pd.read_csv(FILE_PATHS['original'])
    
    # --- 新增过滤逻辑 ---
    if 'total_tokens' in df_orig.columns:
        df_orig = df_orig[df_orig['total_tokens'] >= MIN_TOKENS]
    # ------------------
    
    baseline_tpr = get_tpr_at_fixed_fpr(df_orig['z_score'].values, threshold)
    print(f"Baseline TPR (Ratio 0): {baseline_tpr:.4f}")

    # 3. 准备绘图
    fig, ax = plt.subplots(figsize=(7, 5))

    # 4. 处理并绘制 Deletion 和 Substitution 曲线
    for attack_type in ['deletion', 'substitution']:
        print(f"正在处理 {attack_type} 数据...")
        file_path = FILE_PATHS[attack_type]
        style = STYLE_CONFIG[attack_type]
        
        ratios, tprs = load_and_process_attack_data(file_path, threshold, baseline_tpr)
        
        ax.plot(ratios, tprs, 
                label=style['label'], 
                marker=style['marker'], 
                linestyle=style['linestyle'], 
                color=style['color'],
                linewidth=2,
                markersize=6)

    # 5. 设置图表样式
    ax.grid(True, linestyle='-', alpha=0.2, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_xlim(-0.02, 0.52)
    # 确保 Y 轴范围合理
    y_min = 0.0
    ax.set_ylim(bottom=y_min, top=1.02)

    ax.set_xlabel("Ratio", fontsize=14)
    ax.set_ylabel(r"TPR @ 5% FPR $\rightarrow$", fontsize=14)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(frameon=False, fontsize=12, loc='lower left', bbox_to_anchor=(0.1, 0.15))

    plt.tight_layout()
    
    output_filename = 'watermark_robustness_filtered.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"绘图完成，已保存为 {output_filename}")
    # plt.show() # 如果在无界面环境运行，请注释掉此行

if __name__ == "__main__":
    main()