import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import glob
import subprocess
from dlm_watermark.utils.file_io import resolve_output_path

def main():
    # 定义目录路径（受 WATERMARK_OUTPUT_BASE_DIR 环境变量控制）
    output_dir = resolve_output_path("outputs_ulti")
    config_dir = resolve_output_path("configs_ulti")

    # 1. 检查目录是否存在
    if not os.path.exists(output_dir):
        print(f"❌ 错误: 找不到输出数据文件夹 '{output_dir}'")
        print("请确保你在项目根目录下运行此脚本，或者该文件夹确实存在。")
        return
    if not os.path.exists(config_dir):
        print(f"❌ 错误: 找不到配置文件夹 '{config_dir}'")
        return

    # 2. 扫描所有的 .jsonl 文件
    jsonl_pattern = os.path.join(output_dir, "*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    
    jsonl_files.sort() # 排序，保证执行顺序
    
    total_files = len(jsonl_files)
    print(f"📂 在 '{output_dir}' 中找到了 {total_files} 个 .jsonl 文件。\n")

    if total_files == 0:
        return

    # 3. 遍历并执行
    for i, jsonl_path in enumerate(jsonl_files):
        filename = os.path.basename(jsonl_path)
        basename = os.path.splitext(filename)[0] # 去掉 .jsonl 后缀
        
        # 对应的 yaml 配置文件路径
        yaml_path = os.path.join(config_dir, f"{basename}.yaml")
        
        # 检查输出文件是否已存在，如果存在则跳过
        expected_outputs = [
            os.path.join(output_dir, f"{basename}_original.csv"),
            os.path.join(output_dir, f"{basename}_deletion.csv"),
            os.path.join(output_dir, f"{basename}_substitution.csv")
        ]
        if all(os.path.exists(f) for f in expected_outputs):
            print(f"⏭️ [已存在] {filename}: 所有结果文件 (.csv) 已生成，跳过。")
            continue
        
        # 检查配置文件是否存在
        if not os.path.exists(yaml_path):
            print(f"⚠️ [跳过 {i+1}/{total_files}] {filename}: 在 '{config_dir}' 中未找到对应的 config 文件 '{basename}.yaml'")
            continue

        print(f"🚀 [正在处理 {i+1}/{total_files}] {filename} ...")
        print(f"   Config: {yaml_path}")

        # 构建命令
        # PYTHONPATH=src python scripts/run_robustness_eval.py --path ... --config ... ...
        cmd = [
            sys.executable, "scripts/run_robustness_eval.py",
            "--path", jsonl_path,
            "--config", yaml_path,
            "--original",
            "--deletion",
            "--substitution"#,
            # "--ca_substitution"
        ]

        # 设置环境变量 PYTHONPATH
        env = os.environ.copy()
        #如果你在 src 外面运行，需要把 src 加入 PYTHONPATH
        old_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"src{os.pathsep}{old_path}" if old_path else "src"

        try:
            # 运行命令，check=True 表示如果命令返回非0状态码引发异常
            subprocess.run(cmd, env=env, check=True)
            print(f"✅ 完成 {filename}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ [失败] 处理 {filename} 时命令返回错误代码: {e.returncode}")
            # 这里选择继续处理下一个，如果想遇到错误即停止，可以取消这一行的注释:
            # sys.exit(1)
            print("\n")
            
        except KeyboardInterrupt:
            print("\n🛑 用户强制中断。正在停止...")
            sys.exit(0)

if __name__ == "__main__":
    main()
