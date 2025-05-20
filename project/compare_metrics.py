import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_metrics_from_directory(directory):
    """
    从指定目录加载指标数据
    
    Args:
        directory: 包含benchmark_results的目录路径
        
    Returns:
        包含指标的字典
    """
    metrics = {}
    
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return metrics
    
    # 加载step_metrics.json
    json_path = os.path.join(directory, "step_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            metrics['meaningful_step_ratio'] = json_data.get('meaningful_step_ratio', 0)
    
    # 加载step_metrics.csv
    csv_path = os.path.join(directory, "step_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        metrics['total_steps'] = len(df)
        metrics['total_reward'] = df['reward'].sum()
        
        # 检查游戏是否获胜
        # 根据用户反馈，判断游戏是否获胜的标准是：
        # 1. 如果总reward大于30，则认为游戏获胜
        # 2. 或者如果游戏过程中有任何一步的reward大于0，则认为游戏获胜
        if len(df) > 0:
            # 检查是否有done列，如果有，则使用done列来判断游戏是否结束
            if 'done' in df.columns:
                # 找到最后一个done=True的行
                done_rows = df[df['done'] == True]
                if not done_rows.empty:
                    last_done_row = done_rows.iloc[-1]
                    metrics['win'] = last_done_row['reward'] > 0
                else:
                    # 如果没有done=True的行，则检查是否有任何一步的reward大于0
                    metrics['win'] = (df['reward'] > 0).any()
            else:
                # 如果没有done列，则检查是否有任何一步的reward大于0
                metrics['win'] = (df['reward'] > 0).any()
        else:
            metrics['win'] = False
    
    return metrics

def collect_all_metrics(base_dir, is_dual_reward=False):
    """
    收集所有游戏、模型和模式的指标
    
    Args:
        base_dir: 基础目录
        is_dual_reward: 是否是dual_reward目录
        
    Returns:
        包含所有指标的嵌套字典
    """
    all_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # 遍历基础目录
    for game_mode_model in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, game_mode_model)):
            continue
            
        # 解析游戏名称、模式和模型
        parts = game_mode_model.split('_')
        if len(parts) < 3:
            continue
            
        # 处理不同格式的目录名
        if "gvgai-" in game_mode_model:
            # 处理benchmark_results目录中的格式
            game_parts = game_mode_model.split('_')
            if len(game_parts) >= 3:
                game = game_parts[0].replace('gvgai-', '')
                if game.endswith('-lvl1') or game.endswith('-lvl2'):
                    game = game.rsplit('-', 1)[0]
                mode = game_parts[1]
                model = '_'.join(game_parts[2:])
            else:
                continue
        else:
            # 处理experiment和dual_reward目录中的格式
            game = parts[0]
            mode = parts[1]
            model = '_'.join(parts[2:])
        
        # 构建benchmark_results路径
        if is_dual_reward:
            # 对于dual_reward目录，直接使用目录
            metrics = load_metrics_from_directory(os.path.join(base_dir, game_mode_model))
            if metrics:  # 只有当找到指标时才添加
                all_metrics[game][mode][model] = metrics
            else:
                # 尝试查找benchmark_results子目录
                benchmark_dir = os.path.join(base_dir, game_mode_model)
                if os.path.exists(benchmark_dir) and os.path.isdir(benchmark_dir):
                    for subdir in os.listdir(benchmark_dir):
                        if os.path.isdir(os.path.join(benchmark_dir, subdir)):
                            if "benchmark_results" in subdir:
                                results_dir = os.path.join(benchmark_dir, subdir)
                                for env_dir in os.listdir(results_dir):
                                    metrics = load_metrics_from_directory(os.path.join(results_dir, env_dir))
                                    if metrics:  # 只有当找到指标时才添加
                                        all_metrics[game][mode][model] = metrics
                                        break
                            else:
                                # 直接检查子目录
                                metrics = load_metrics_from_directory(os.path.join(benchmark_dir, subdir))
                                if metrics:  # 只有当找到指标时才添加
                                    all_metrics[game][mode][model] = metrics
                                    break
        else:
            # 对于experiment目录，直接使用目录
            metrics = load_metrics_from_directory(os.path.join(base_dir, game_mode_model))
            if metrics:  # 只有当找到指标时才添加
                all_metrics[game][mode][model] = metrics
            else:
                # 尝试查找benchmark_results子目录
                benchmark_dir = os.path.join(base_dir, game_mode_model)
                if os.path.exists(benchmark_dir) and os.path.isdir(benchmark_dir):
                    benchmark_results_dir = os.path.join(benchmark_dir, "benchmark_results")
                    if os.path.exists(benchmark_results_dir) and os.path.isdir(benchmark_results_dir):
                        for env_dir in os.listdir(benchmark_results_dir):
                            env_path = os.path.join(benchmark_results_dir, env_dir)
                            if os.path.isdir(env_path):
                                metrics = load_metrics_from_directory(env_path)
                                if metrics:  # 只有当找到指标时才添加
                                    # 从env_dir中提取游戏名称、模式和模型
                                    if "gvgai-" in env_dir:
                                        env_parts = env_dir.split('_')
                                        if len(env_parts) >= 3:
                                            env_game = env_parts[0].replace('gvgai-', '')
                                            if env_game.endswith('-lvl1') or env_game.endswith('-lvl2'):
                                                env_game = env_game.rsplit('-', 1)[0]
                                            env_mode = env_parts[1]
                                            env_model = '_'.join(env_parts[2:])
                                            all_metrics[env_game][env_mode][env_model] = metrics
    
    return all_metrics

def compare_metrics(dual_reward_dirs, experiment_dirs):
    """
    比较dual_reward和experiment目录中的指标
    
    Args:
        dual_reward_dirs: dual_reward目录列表
        experiment_dirs: experiment目录列表
        
    Returns:
        包含比较结果的DataFrame
    """
    # 收集所有指标
    all_metrics = {}
    
    # 收集dual_reward目录中的指标
    for dr_dir in dual_reward_dirs:
        metrics = collect_all_metrics(dr_dir, is_dual_reward=True)
        for game in metrics:
            for mode in metrics[game]:
                for model in metrics[game][mode]:
                    key = f"{game}_{mode}_{model}_dual"
                    all_metrics[key] = metrics[game][mode][model]
    
    # 收集experiment目录中的指标
    for exp_dir in experiment_dirs:
        metrics = collect_all_metrics(exp_dir, is_dual_reward=False)
        for game in metrics:
            for mode in metrics[game]:
                for model in metrics[game][mode]:
                    key = f"{game}_{mode}_{model}_no_dual"
                    all_metrics[key] = metrics[game][mode][model]
    
    # 合并所有指标
    all_data = []
    
    # 处理所有指标
    for key, metrics in all_metrics.items():
        parts = key.split('_')
        game = parts[0]
        mode = parts[1]
        model_parts = parts[2:-1]
        is_dual = parts[-1] == 'dual'
        
        # 根据用户的反馈，dual reward是指contextual和zero-shot这两种模式
        # 如果不是这两个就不是dual reward
        if mode in ["contextual", "zero-shot"]:
            dual_reward = True
        else:
            dual_reward = False
        
        # 处理model名称
        if len(model_parts) > 0:
            # 如果model_parts中包含"no"，则去掉它
            if model_parts[-1] == "no":
                model_parts = model_parts[:-1]
            
            # 如果model_parts中包含"portkey-"，则提取真正的模型名称
            if len(model_parts) > 0 and model_parts[0].startswith("portkey-"):
                model = model_parts[0].replace("portkey-", "")
            else:
                model = '_'.join(model_parts)
        else:
            model = "unknown"
        
        row = {
            'game': game,
            'mode': mode,
            'model': model,
            'dual_reward': dual_reward,
            'meaningful_step_ratio': metrics.get('meaningful_step_ratio', 0),
            'total_steps': metrics.get('total_steps', 0),
            'total_reward': metrics.get('total_reward', 0),
            'win': metrics.get('win', False)
        }
        all_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    
    # 计算winrate（按游戏、模式、模型和dual_reward分组）
    winrate_df = df.groupby(['game', 'mode', 'model', 'dual_reward'])['win'].mean().reset_index()
    winrate_df.rename(columns={'win': 'winrate'}, inplace=True)
    
    # 合并winrate到原始DataFrame
    df = pd.merge(df, winrate_df, on=['game', 'mode', 'model', 'dual_reward'])
    
    return df

def plot_metrics(df, output_dir):
    """
    绘制指标比较图
    
    Args:
        df: 包含指标的DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图表样式
    sns.set(style="whitegrid")
    
    # 按游戏分组绘制图表
    for game in df['game'].unique():
        game_df = df[df['game'] == game]
        
        # 创建一个2x2的子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Metrics Comparison for {game}', fontsize=16)
        
        # 1. Meaningful Step Ratio
        sns.barplot(x='model', y='meaningful_step_ratio', hue='mode', 
                   data=game_df, ax=axes[0, 0], palette='Set1')
        axes[0, 0].set_title('Meaningful Step Ratio')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Ratio')
        
        # 2. Winrate
        sns.barplot(x='model', y='winrate', hue='mode', 
                   data=game_df, ax=axes[0, 1], palette='Set1')
        axes[0, 1].set_title('Winrate')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Rate')
        
        # 3. Total Steps
        sns.barplot(x='model', y='total_steps', hue='mode', 
                   data=game_df, ax=axes[1, 0], palette='Set1')
        axes[1, 0].set_title('Total Steps')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Steps')
        
        # 4. Total Reward
        sns.barplot(x='model', y='total_reward', hue='mode', 
                   data=game_df, ax=axes[1, 1], palette='Set1')
        axes[1, 1].set_title('Total Reward')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Reward')
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f'{game}_metrics_comparison.png'))
        plt.close()
        
        # 绘制dual_reward vs no_dual_reward的比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Dual Reward Comparison for {game}', fontsize=16)
        
        # 1. Meaningful Step Ratio
        sns.barplot(x='model', y='meaningful_step_ratio', hue='dual_reward', 
                   data=game_df, ax=axes[0, 0], palette='Set2')
        axes[0, 0].set_title('Meaningful Step Ratio')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Ratio')
        
        # 2. Winrate
        sns.barplot(x='model', y='winrate', hue='dual_reward', 
                   data=game_df, ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('Winrate')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Rate')
        
        # 3. Total Steps
        sns.barplot(x='model', y='total_steps', hue='dual_reward', 
                   data=game_df, ax=axes[1, 0], palette='Set2')
        axes[1, 0].set_title('Total Steps')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Steps')
        
        # 4. Total Reward
        sns.barplot(x='model', y='total_reward', hue='dual_reward', 
                   data=game_df, ax=axes[1, 1], palette='Set2')
        axes[1, 1].set_title('Total Reward')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Reward')
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f'{game}_dual_reward_comparison.png'))
        plt.close()

def generate_win_input_csv(df, output_path):
    """
    生成一个用于手动填写输赢信息的CSV文件
    
    Args:
        df: 包含指标的DataFrame
        output_path: 输出文件路径
    """
    # 创建一个新的DataFrame，只包含游戏、模式、模型和dual_reward列
    win_df = df[['game', 'mode', 'model', 'dual_reward']].drop_duplicates()
    
    # 添加一个win列，默认为空，用于手动填写
    win_df['win'] = ""
    
    # 保存到CSV
    win_df.to_csv(output_path, index=False)
    
    print(f"已生成用于手动填写输赢信息的CSV文件: {output_path}")

def calculate_winrate_from_csv(input_path, output_path):
    """
    从手动填写的CSV文件中计算胜率
    
    Args:
        input_path: 输入文件路径（手动填写的CSV）
        output_path: 输出文件路径（计算胜率后的CSV）
    """
    # 读取CSV文件
    df = pd.read_csv(input_path)
    
    # 将win列转换为数值
    # 首先尝试直接转换为数值
    try:
        df['win'] = pd.to_numeric(df['win'])
    except ValueError:
        # 如果直接转换失败，则使用映射
        win_map = {
            'True': 1, 'False': 0, '1': 1, '0': 0, 
            'true': 1, 'false': 0, 'yes': 1, 'no': 0,
            'win': 1, 'lose': 0, '赢': 1, '输': 0
        }
        df['win'] = df['win'].map(win_map)
    
    # 确保win列是数值类型
    df['win'] = pd.to_numeric(df['win'], errors='coerce')
    
    # 计算每个游戏、模式、模型和dual_reward组合的胜率
    winrate_df = df.groupby(['game', 'mode', 'model', 'dual_reward'])['win'].agg(['mean', 'count']).reset_index()
    winrate_df.rename(columns={'mean': 'winrate'}, inplace=True)
    
    # 保存到CSV
    winrate_df.to_csv(output_path, index=False)
    
    print(f"已计算胜率并保存到: {output_path}")
    
    return winrate_df

def main():
    # === 从已整理好的 CSV 读取数据 ===
    csv_path = 'benchmark_summary.csv'  # 你自己准备的数据
    if not os.path.exists(csv_path):
        print(f"找不到输入文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # === 字段标准化（只改列名，不做计算）===
    df.rename(columns={
        "method": "mode",
        "llm": "model",
        "meaningful step": "meaningful_step_ratio",
        "step": "total_steps",
        "reward": "total_reward"
    }, inplace=True)

    # === 补充字段 ===
    df["dual_reward"] = df["mode"].isin(["contextual", "zero-shot"])
    df["win"] = df["winrate"].astype(int)  # 不再自己推算

    # === 保存中间清洗结果 ===
    df.to_csv("cleaned_metrics.csv", index=False)

    # === 绘图 ===
    plot_metrics(df, 'metrics_plots')

    # === 输出摘要 ===
    print("指标比较摘要:")
    print("====================")
    summary = df.groupby(['game', 'mode', 'dual_reward']).agg({
        'meaningful_step_ratio': 'mean',
        'winrate': 'mean',
        'total_steps': 'mean',
        'total_reward': 'mean'
    }).reset_index()

    print(summary)
    summary.to_csv('metrics_summary.csv', index=False)


if __name__ == "__main__":
    main()
