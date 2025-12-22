import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import math

# --- 配置区域 ---
# 第一个方法 (Baseline A)
method1_config = {
    "name": "BDLM-Watermark (Ours)",
    "pos_file": "outputs/results_ours_pos.jsonl",
    "neg_file": "outputs/results_ours_neg.jsonl",
    "color": "tab:red"
}

# 第二个方法 (Baseline B)
method2_config = {
    "name": "Baseline-Watermark", # 也就是你跑的另一个 baseline
    "pos_file": "outputs/results_base_pos.jsonl",
    "neg_file": "outputs/results_base_neg.jsonl",
    "color": "tab:blue"
}

# 过滤长度小于此值的样本
MIN_LENGTH = 200
# ----------------
def load_data(pos_path, neg_path):
    """
    读取正负样本，返回:
    1. pos_z, neg_z (用于算 AUC)
    2. pos_ppl_avg (用于图例)
    3. neg_ppl_avg (用于第三条线的图例)
    """
    def _read_file(path):
        z_scores, ppls = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if data.get("length", 0) < MIN_LENGTH: continue
                
                if "z_score" in data: z_scores.append(data["z_score"])
                if "ppl" in data: ppls.append(data["ppl"])
        return z_scores, ppls

    p_z, p_ppl_list = _read_file(pos_path)
    n_z, n_ppl_list = _read_file(neg_path)
    
    avg_pos_ppl = np.mean(p_ppl_list) if p_ppl_list else 0.0
    avg_neg_ppl = np.mean(n_ppl_list) if n_ppl_list else 0.0
    
    return p_z, n_z, avg_pos_ppl, avg_neg_ppl

def get_roc_metrics(pos_z, neg_z):
    y_true = [1] * len(pos_z) + [0] * len(neg_z)
    y_scores = pos_z + neg_z
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_combined_log_roc():
    plt.figure(figsize=(10, 8))

    # ==========================================
    # 1. 处理 Method 1 (Ours)
    # ==========================================
    p_z1, n_z1, p_ppl1, n_ppl1 = load_data(method1_config["pos_file"], method1_config["neg_file"])
    # 这里的 n_ppl1 就是我们要展示的 Unwatermarked PPL (因为负样本是一样的)
    baseline_ppl_val = n_ppl1 
    
    if p_z1 and n_z1:
        fpr1, tpr1, auc1 = get_roc_metrics(p_z1, n_z1)
        # 图例只写 AUC 和 它自己的 PPL
        label1 = (f"{method1_config['name']}\n"
                  f"AUC: {auc1:.4f} | PPL: {p_ppl1:.2f}")
        plt.semilogx(fpr1, tpr1, color=method1_config["color"], lw=3, label=label1)

    # ==========================================
    # 2. 处理 Method 2 (Baseline)
    # ==========================================
    p_z2, n_z2, p_ppl2, _ = load_data(method2_config["pos_file"], method2_config["neg_file"])
    
    if p_z2 and n_z2:
        fpr2, tpr2, auc2 = get_roc_metrics(p_z2, n_z2)
        # 图例只写 AUC 和 它自己的 PPL
        label2 = (f"{method2_config['name']}\n"
                  f"AUC: {auc2:.4f} | PPL: {p_ppl2:.2f}")
        plt.semilogx(fpr2, tpr2, color=method2_config["color"], lw=2.5, linestyle='-.', label=label2)

    # ==========================================
    # 3. 处理 Unwatermarked (Random Guess)
    # ==========================================
    # 生成 Log 轴下的对角线数据
    random_fpr = np.logspace(-2.5, 0, 100)
    
    # 这里的图例专门展示 Baseline PPL
    label_random = (f"Unwatermarked (TPR=FPR)\n"
                    f"Baseline PPL: {baseline_ppl_val:.2f}")
    
    plt.semilogx(random_fpr, random_fpr, color='gray', linestyle='--', lw=2, label=label_random)

    # ==========================================
    # 4. 美化图表
    # ==========================================
    plt.xlabel("False Positive Rate (Log Scale)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title(f"Watermark Detection & Quality Comparison\n(Filter: Length >= {MIN_LENGTH})", fontsize=14)
    
    # 设置 Log 轴范围 (从 0.0001 到 1)
    plt.xlim([math.pow(10, -2.5), 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    # 调整图例位置和样式
    plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True, borderpad=1)
    
    plt.tight_layout()
    plt.savefig("final_comparison_roc.png", dpi=300)
    print("绘图完成: final_comparison_roc.png")
    plt.show()

if __name__ == "__main__":
    # 测试用 (如果有真实文件请注释掉下面几行)
    # import random
    # def mock(f, mu_z, mu_ppl):
    #     with open(f, 'w') as F:
    #         for _ in range(500): 
    #             F.write(json.dumps({"z_score": random.gauss(mu_z, 2), "ppl": random.gauss(mu_ppl, 1), "length": 250})+"\n")
    # mock("method_a_pos.jsonl", 5.0, 12.5) # Ours: PPL稍高，检测极强
    # mock("method_a_neg.jsonl", 0.0, 10.0) # Neg: PPL低
    # mock("method_b_pos.jsonl", 2.5, 14.0) # Baseline: PPL更高，检测较弱
    # mock("method_b_neg.jsonl", 0.0, 10.0) 

    plot_combined_log_roc()