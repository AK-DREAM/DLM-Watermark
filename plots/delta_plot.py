import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================

# Delta 列表 (即使写成整数，代码里也会强制转为 1.0, 2.0 这种格式)
DELTAS = [1.0, 2.0, 2.5, 3.0, 4.0]

# 路径模板配置
CONFIGS = {
    'Ours': {
        # 注意这里 {} 会被填入保留一位小数的 delta，例如 "1.0"
        'dir_pattern': 'output_json/bdlm_delta_{}_gamma_0.5', 
        'color': '#4c72b0',  # 深蓝色
        'marker': 'o'
    },
    'Baseline': {
        'dir_pattern': 'output_json/dlm_delta_{}_gamma_0.5',
        'color': '#dd8452',  # 橙色
        'marker': 'o'
    }
}

MIN_TOKENS = 200

# ===========================================

def load_data(filepath):
    """
    读取 jsonl 文件，返回 z_scores 和 ppls 列表。
    执行 total_tokens >= 200 的过滤。
    """
    z_scores = []
    ppls = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return np.array([]), np.array([])

    with open(filepath, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # 过滤短文本
                tokens = item.get('total_tokens', 0)
                if tokens is None or tokens < MIN_TOKENS:
                    continue
                
                z_scores.append(item.get('z_score', 0))
                # 兼容可能的键名 'ppl' 或 'perplexity'
                ppl = item.get('ppl', item.get('perplexity', None))
                if ppl is not None:
                    ppls.append(ppl)
            except json.JSONDecodeError:
                continue
                
    return np.array(z_scores), np.array(ppls)

def main():
    plt.figure(figsize=(7, 5))
    
    # 用于收集未加水印的 PPL 以绘制红色竖线
    all_neg_ppls = []

    # 遍历两种方法 (Ours, Baseline)
    for label, config in CONFIGS.items():
        x_values = [] # log(PPL)
        y_values = [] # TPR
        
        print(f"Processing {label}...")

        for delta in DELTAS:
            # --- 核心修改：强制格式化为一位小数 ---
            # 例如 delta=1 -> "1.0", delta=2.5 -> "2.5"
            delta_str = f"{delta:.1f}"
            
            # 构造文件夹路径
            dir_path = config['dir_pattern'].format(delta_str)
            pos_path = os.path.join(dir_path, 'pos.jsonl')
            neg_path = os.path.join(dir_path, 'neg.jsonl')

            print(f"  Reading from: {dir_path}")

            # 1. 读取数据
            pos_z, pos_ppl = load_data(pos_path)
            neg_z, neg_ppl = load_data(neg_path)

            if len(pos_z) == 0 or len(neg_z) == 0:
                print(f"    Skipping delta={delta_str} due to empty data or file not found.")
                continue

            # 收集 neg ppl 用于计算基准线
            all_neg_ppls.extend(neg_ppl)

            # 2. 计算 Threshold @ 1% FPR
            threshold = np.percentile(neg_z, 99)

            # 3. 计算 TPR (Recall)
            tpr = np.sum(pos_z > threshold) / len(pos_z)
            y_values.append(tpr)

            # 4. 计算 log(Perplexity)
            avg_ppl = np.mean(pos_ppl)
            log_ppl = np.log(avg_ppl) # 自然对数
            x_values.append(log_ppl)
            
            print(f"    Delta {delta_str}: TPR={tpr:.4f}, log(PPL)={log_ppl:.4f}")

        # 绘制曲线
        plt.plot(x_values, y_values, 
                 label=label,
                 color=config['color'],
                 marker=config['marker'],
                 linewidth=2,
                 markersize=7,
                 alpha=0.9)

    # ================= 绘制红色竖线 (Unwatermarked Baseline) =================
    if all_neg_ppls:
        # 计算所有 neg 样本的平均 PPL 的 log 值
        baseline_log_ppl = np.log(np.mean(all_neg_ppls))
        
        plt.axvline(x=baseline_log_ppl, 
                    color='red', 
                    linestyle='--', 
                    linewidth=2, 
                    label='log(Perplexity)\nUnwatermarked LLM')

    # ================= 样式设置 =================
    ax = plt.gca()
    
    ax.grid(True, linestyle='-', alpha=0.2, color='gray')
    ax.set_axisbelow(True)

    # 坐标轴标签 (箭头指向左边表示越小越好)
    ax.set_xlabel(r"$\leftarrow$ log(Perplexity)", fontsize=14) 
    ax.set_ylabel(r"TPR @ 1% FPR $\rightarrow$", fontsize=14)
    
    # 这里的范围你可以根据实际画出来的图微调
    # plt.ylim(-0.02, 1.05)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(axis='both', labelsize=12)

    plt.legend(frameon=False, fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig('ppl_vs_tpr_curve_fixed.png', dpi=300, bbox_inches='tight')
    print("绘图完成: ppl_vs_tpr_curve_fixed.png")
    # plt.show()

if __name__ == "__main__":
    main()