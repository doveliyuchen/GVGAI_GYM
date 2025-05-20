import pandas as pd
import json
import os

# ==== 获取 benchmark_results 下所有子目录 ====
base_dir = "benchmark_results"
folders = [
    os.path.join(base_dir, name)
    for name in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, name))
]

# ==== 提取游戏名、模型名和方法 ====
def extract_info_from_path(path):
    basename = os.path.basename(path)
    parts = basename.split("-")
    game = parts[1]
    llm = basename.split("_")[-1]
    if "contextual" in basename:
        method = "contextual"
    elif "zero-shot" in basename or "zeroshot" in basename:
        method = "zero-shot"
    else:
        method = "unknown"
    return game, llm, method

# ==== 处理每个子文件夹 ====
def process_single_result_folder(folder_path):
    json_file = os.path.join(folder_path, "step_metrics.json")
    csv_file = os.path.join(folder_path, "step_metrics.csv")

    game, llm, method = extract_info_from_path(folder_path)

    # 加载 JSON
    with open(json_file, "r") as f:
        meaningful_step_ratio = json.load(f).get("meaningful_step_ratio", "")

    # 加载 CSV
    df = pd.read_csv(csv_file)
    reward_sum = df["reward"].sum()
    step = len(df.columns) + 2

    return {
        "game": game,
        "method": method,
        "reward": reward_sum,
        "step": step,
        "llm": llm,
        "meaningful step": meaningful_step_ratio,
        "winrate": ""
    }

# ==== 批量处理并排序 ====
results = [process_single_result_folder(folder) for folder in folders]
df_result = pd.DataFrame(results)

# 设置排序：先按 method 排序（zero-shot 在前），再按 game 排序
method_order = {"zero-shot": 0, "contextual": 1, "unknown": 2}
df_result["method_order"] = df_result["method"].map(method_order)
df_result.sort_values(by=["method_order", "game"], inplace=True)
df_result.drop(columns=["method_order"], inplace=True)

# ==== 保存 CSV ====
df_result.to_csv("benchmark_summary.csv", index=False)
