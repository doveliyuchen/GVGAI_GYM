#!/usr/bin/env python3
"""
扩展版分析LLM游戏结果并生成热力图的脚本
支持处理多个输入目录中的JSON结果文件
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 定义6个核心游戏
CORE_GAMES = {'aliens', 'boulderdash', 'escape', 'realsokoban', 'sokoban', 'zelda'}

def extract_game_info(env_name_full):
    """从完整环境名提取游戏名和关卡"""
    # 例如: gvgai-aliens-lvl0-v0 -> (aliens, 0)
    import re
    match = re.search(r'gvgai-(.+?)-lvl(\d+)-v\d+', env_name_full)
    if match:
        return match.group(1), int(match.group(2))
    return env_name_full, 0

def load_run_data(run_dir):
    """加载单次运行的数据"""
    step_metrics_path = os.path.join(run_dir, "benchmark_analysis.json", "step_metrics.json")
    
    if not os.path.exists(step_metrics_path):
        return None
    
    try:
        import traceback
        with open(step_metrics_path, 'r') as f:
            data = json.load(f)
        print(f"DEBUG: loaded {step_metrics_path}, keys={list(data.keys())}", flush=True)
        
        # 计算基本指标
        meaningful_steps = data.get('meaningful_steps', [])
        if meaningful_steps is None:
            meaningful_steps = []
        avatar_positions = data.get('avatar_positions', [])
        if avatar_positions is None:
            avatar_positions = []
        winner = data.get('winner', None)

        # 优先 meaningful_steps 字段，其次 meaningful_step_ratio
        if isinstance(meaningful_steps, list) and len(meaningful_steps) > 0:
            total_steps = len(meaningful_steps)
            meaningful_count = sum(meaningful_steps)
            meaningful_ratio = meaningful_count / total_steps if total_steps > 0 else 0
        elif "meaningful_step_ratio" in data:
            meaningful_ratio = data["meaningful_step_ratio"]
            total_steps = None
            meaningful_count = None
        else:
            total_steps = None
            meaningful_count = None
            meaningful_ratio = 0

        # 计算移动距离
        if avatar_positions and isinstance(avatar_positions, list) and len(avatar_positions) > 1:
            total_distance = 0
            for i in range(1, len(avatar_positions)):
                prev = avatar_positions[i-1]
                curr = avatar_positions[i]
                # 检查 prev 和 curr 是否为有效二维坐标
                if (
                    isinstance(prev, list) and len(prev) == 1 and isinstance(prev[0], list) and len(prev[0]) == 2 and
                    isinstance(curr, list) and len(curr) == 1 and isinstance(curr[0], list) and len(curr[0]) == 2
                ):
                    pos1 = prev[0]
                    pos2 = curr[0]
                    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    total_distance += distance
                else:
                    continue
        else:
            total_distance = 0

        return {
            'total_steps': total_steps,
            'meaningful_steps': meaningful_count,
            'meaningful_ratio': meaningful_ratio,
            'total_distance': total_distance,
            'avg_distance_per_step': total_distance / total_steps if total_steps > 0 else 0,
            'winner': winner
        }
    except Exception as e:
        import traceback
        print(f"Error loading {step_metrics_path}: {e}")
        traceback.print_exc()
        return None

def load_csv_data(run_dir, winner=None):
    """加载CSV数据获取奖励信息，如果CSV不存在则基于winner生成基本奖励信息"""
    csv_path = os.path.join(run_dir, "benchmark_analysis.json", "step_metrics.csv")
    
    if not os.path.exists(csv_path):
        # CSV不存在时，基于winner字段生成基本的奖励信息
        if winner is not None:
            won = (winner == "PLAYER_WINS")
            # 如果获胜，给一个基本的奖励值；否则为0
            if won:
                return {
                    'total_reward': 10.0,  # 基本获胜奖励
                    'reward_steps': 1,
                    'max_reward': 10.0,
                    'final_reward': 10.0,
                    'won': True
                }
            else:
                return {
                    'total_reward': 0.0,
                    'reward_steps': 0,
                    'max_reward': 0.0,
                    'final_reward': 0.0,
                    'won': False
                }
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        total_reward = df['reward'].sum()
        reward_steps = len(df[df['reward'] > 0])
        max_reward = df['reward'].max()
        
        # 优先用 winner 字段判定胜负
        if winner is not None:
            won = (winner == "PLAYER_WINS")
        else:
            # 检查游戏是否获胜 (通常最后一步有高奖励表示获胜)
            final_reward = df['reward'].iloc[-1] if len(df) > 0 else 0
            won = final_reward > 0 or total_reward > 10  # 简单的获胜判断逻辑
        
        return {
            'total_reward': total_reward,
            'reward_steps': reward_steps,
            'max_reward': max_reward,
            'final_reward': df['reward'].iloc[-1] if len(df) > 0 else 0,
            'won': won
        }
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def scan_results_directory(base_dir, source_label=""):
    """递归扫描所有 run_x/benchmark_analysis.json/step_metrics.json 文件，自动推断模型/游戏/关卡"""
    results = []
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Results directory {base_dir} does not exist!")
        return results

    print(f"Scanning results in {base_dir} ({source_label})...")

    # 递归查找所有 run_x/benchmark_analysis.json/step_metrics.json
    for step_json in base_path.rglob("run_*/benchmark_analysis.json/step_metrics.json"):
        run_dir = step_json.parent.parent
        rel_parts = run_dir.relative_to(base_path).parts
        
        # 根据source_label判断目录结构
        if source_label == "6games":
            # llm_agent_runs_output_portkey_6games: 模型/游戏-关卡/run_x/
            if len(rel_parts) >= 2:
                model_name = rel_parts[0]
                game_level_dir = rel_parts[1]
            else:
                continue  # 跳过结构不对的
        elif source_label in ["deepseek-6games", "deepseek-r1-6games"]:
            # deepseek目录: 游戏-关卡/run_x/ (没有模型层)
            if len(rel_parts) >= 1:
                model_name = source_label.replace("-6games", "")  # deepseek 或 deepseek-r1
                game_level_dir = rel_parts[0]
            else:
                continue  # 跳过结构不对的
        else:
            # 默认逻辑（兼容其他情况）
            if len(rel_parts) >= 2:
                model_name = rel_parts[0]
                game_level_dir = rel_parts[1]
            else:
                model_name = "default"
                game_level_dir = rel_parts[0]
            
            # 如果有source_label，添加到模型名中
            if source_label:
                model_name = f"{model_name}-{source_label}"
            
        run_id = run_dir.name.replace('run_', '')
        game_name, level = extract_game_info(f"gvgai-{game_level_dir}-v0")
        json_data = load_run_data(run_dir)
        if json_data is not None:
            winner = json_data.get('winner', None)
            csv_data = load_csv_data(run_dir, winner=winner)
            print(f"DEBUG: {run_dir} model={model_name} game_level={game_level_dir} json_data={json_data} csv_data={csv_data}")
            result = {
                'model': model_name,
                'game': game_name,
                'level': level,
                'run_id': int(run_id) if run_id.isdigit() else run_id,
                'game_level': f"{game_name}-lvl{level}",
                'source': source_label,
                **json_data
            }
            
            # 确保从JSON的winner字段计算won，即使没有CSV文件
            if 'won' not in result and winner is not None:
                result['won'] = (winner == "PLAYER_WINS")
            
            if csv_data is not None:
                result.update(csv_data)
            results.append(result)
        else:
            print(f"DEBUG: {run_dir} model={model_name} game_level={game_level_dir} json_data=None (skipped)")

    print(f"Collected {len(results)} results from {base_dir}")
    return results

def create_heatmaps(df, output_dir="heatmaps_extended"):
    """创建各种热力图"""
    os.makedirs(output_dir, exist_ok=True)

    # 保证 normalized_reward 字段存在
    if 'total_reward' in df.columns and 'normalized_reward' not in df.columns:
        df['normalized_reward'] = df.groupby('game_level')['total_reward'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
    
    # 设置图形样式
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # 定义 reasoner 和非 reasoner 模型分组
    def get_model_order_with_separator(models):
        """获取带分隔符的模型排序列表"""
        # 推理模型包括：DeepSeek-reasoner, Gemini-2.5-pro, GPT-o3-mini
        reasoner_models = [m for m in models if 'reasoner' in m.lower() or 'o3' in m.lower() or 'Gemini-2.5-pro' in m]
        non_reasoner_models = [m for m in models if m not in reasoner_models]
        
        # 排序各组内的模型
        non_reasoner_models.sort()
        reasoner_models.sort()
        
        # 创建完整的排序列表，包含分隔符
        ordered_models = []
        if non_reasoner_models:
            ordered_models.extend(non_reasoner_models)
        if reasoner_models:
            if non_reasoner_models:  # 如果有非reasoner模型，添加分隔符
                ordered_models.append("────── REASONER MODELS ──────")
            ordered_models.extend(reasoner_models)
        
        return ordered_models, len(non_reasoner_models)
    
    def create_ordered_pivot_with_separator(pivot_df, ordered_models, separator_pos):
        """创建带分隔符的排序数据透视表"""
        # 重新排序现有模型
        existing_models = [m for m in ordered_models if m != "--- REASONER MODELS ---" and m in pivot_df.index]
        ordered_pivot = pivot_df.reindex(existing_models, fill_value=0)
        
        # 插入分隔符行（如果需要）
        if "--- REASONER MODELS ---" in ordered_models and separator_pos > 0:
            # 为分隔符行创建空数据
            separator_row = pd.Series([np.nan] * len(pivot_df.columns), index=pivot_df.columns, name="--- REASONER MODELS ---")
            
            # 分割数据框
            top_part = ordered_pivot.iloc[:separator_pos]
            bottom_part = ordered_pivot.iloc[separator_pos:]
            
            # 创建新的DataFrame，插入分隔符行
            ordered_pivot = pd.concat([top_part, separator_row.to_frame().T, bottom_part])
        
        return ordered_pivot
    
    # 获取模型排序
    unique_models = df['model'].unique()
    ordered_models, separator_pos = get_model_order_with_separator(unique_models)
    
    # 1. 有意义步数比例热力图
    print("Creating meaningful ratio heatmap...")
    pivot_meaningful = df.groupby(['model', 'game_level'])['meaningful_ratio'].mean().unstack(fill_value=0)
    
    # 重新排序行并添加分隔符行
    pivot_meaningful_ordered = create_ordered_pivot_with_separator(pivot_meaningful, ordered_models, separator_pos)
    
    # 创建注释矩阵
    annot_meaningful = pivot_meaningful_ordered.applymap(
        lambda x: "" if pd.isna(x) else (f"{int(x)}" if float(x).is_integer() else f"{x:.2f}")
    )
    
    plt.figure(figsize=(20, 12))
    mask = pivot_meaningful_ordered.isna()
    sns.heatmap(pivot_meaningful_ordered, annot=annot_meaningful, fmt='', cmap='RdYlGn', 
                mask=mask, cbar_kws={'label': 'Meaningful Step Ratio', 'shrink': 0.8, 'pad': 0.01})
    plt.title('Average Meaningful Step Ratio by Model and Game')
    plt.xlabel('Game-Level')
    plt.ylabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/meaningful_ratio_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 总奖励热力图
    if 'total_reward' in df.columns:
        print("Creating total reward heatmap...")
        pivot_reward = df.groupby(['model', 'game_level'])['total_reward'].mean().unstack(fill_value=0)
        
        # 重新排序行并添加分隔符行
        pivot_reward_ordered = create_ordered_pivot_with_separator(pivot_reward, ordered_models, separator_pos)
        
        # 创建注释矩阵
        annot_reward = pivot_reward_ordered.applymap(
            lambda x: "" if pd.isna(x) else (f"{int(x)}" if float(x).is_integer() else f"{x:.2f}")
        )
        
        plt.figure(figsize=(20, 12))
        mask = pivot_reward_ordered.isna()
        sns.heatmap(pivot_reward_ordered, annot=annot_reward, fmt='', cmap='RdYlGn', 
                    mask=mask, cbar_kws={'label': 'Average Total Reward', 'shrink': 0.8, 'pad': 0.01})
        plt.title('Average Total Reward by Model and Game')
        plt.xlabel('Game-Level')
        plt.ylabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/total_reward_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2b. 归一化奖励热力图
    if 'normalized_reward' in df.columns:
        print("Creating normalized reward heatmap...")
        pivot_norm_reward = df.groupby(['model', 'game_level'])['normalized_reward'].mean().unstack(fill_value=0)
        
        # 重新排序行并添加分隔符行
        pivot_norm_reward_ordered = create_ordered_pivot_with_separator(pivot_norm_reward, ordered_models, separator_pos)
        
        # 创建注释矩阵
        annot_norm_reward = pivot_norm_reward_ordered.applymap(
            lambda x: "" if pd.isna(x) else (f"{int(x)}" if float(x).is_integer() else f"{x:.2f}")
        )
        
        plt.figure(figsize=(20, 12))
        mask = pivot_norm_reward_ordered.isna()
        sns.heatmap(pivot_norm_reward_ordered, annot=annot_norm_reward, fmt='', cmap='YlGnBu', 
                    vmin=0, vmax=1, mask=mask, cbar_kws={'label': 'Average Normalized Reward', 'shrink': 0.8, 'pad': 0.01})
        plt.title('Average Normalized Reward by Model and Game')
        plt.xlabel('Game-Level')
        plt.ylabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/normalized_reward_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 获胜率热力图
    if 'won' in df.columns:
        print("Creating win rate heatmap...")
        pivot_winrate = df.groupby(['model', 'game_level'])['won'].mean().unstack(fill_value=0)
        
        # 重新排序行并添加分隔符行
        pivot_winrate_ordered = create_ordered_pivot_with_separator(pivot_winrate, ordered_models, separator_pos)
        
        # 创建注释矩阵
        annot_winrate = pivot_winrate_ordered.applymap(
            lambda x: "" if pd.isna(x) else (f"{int(x)}" if float(x).is_integer() else f"{x:.2f}")
        )
        
        plt.figure(figsize=(20, 12))
        mask = pivot_winrate_ordered.isna()
        sns.heatmap(pivot_winrate_ordered, annot=annot_winrate, fmt='', cmap='RdYlGn', 
                    mask=mask, cbar_kws={'label': 'Win Rate', 'shrink': 0.8, 'pad': 0.01})
        plt.title('Win Rate by Model and Game')
        plt.xlabel('Game-Level')
        plt.ylabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/win_rate_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 平均步数热力图
    if 'total_steps' in df.columns:
        print("Creating average steps heatmap...")
        pivot_steps = df.groupby(['model', 'game_level'])['total_steps'].mean().unstack(fill_value=0)
        
        # 重新排序行并添加分隔符行
        pivot_steps_ordered = create_ordered_pivot_with_separator(pivot_steps, ordered_models, separator_pos)
        
        # 创建注释矩阵
        annot_steps = pivot_steps_ordered.applymap(
            lambda x: "" if pd.isna(x) else (f"{int(x)}" if float(x).is_integer() else f"{x:.2f}")
        )
        
        plt.figure(figsize=(20, 12))
        mask = pivot_steps_ordered.isna()
        sns.heatmap(pivot_steps_ordered, annot=annot_steps, fmt='', cmap='RdYlBu_r', 
                    mask=mask, cbar_kws={'label': 'Average Total Steps', 'shrink': 0.8, 'pad': 0.01})
        plt.title('Average Total Steps by Model and Game')
        plt.xlabel('Game-Level')
        plt.ylabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/total_steps_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. 综合性能热力图 (标准化后的多指标组合)
    print("Creating comprehensive performance heatmap...")
    
    # 选择关键指标进行标准化（不包含奖励数据，确保一致性）
    metrics = ['meaningful_ratio']
    if 'total_steps' in df.columns:
        metrics.append('total_steps')
    if 'won' in df.columns:
        metrics.append('won')
    # 注意：故意不包含奖励相关指标，确保所有模型使用相同的评分标准
    
    # 计算每个模型-游戏组合的平均值
    agg_data = df.groupby(['model', 'game_level'])[metrics].mean()
    
    # 标准化指标 (0-1范围)
    normalized_data = agg_data.copy()
    for metric in metrics:
        if metric == 'total_steps':
            # 使用step efficiency: 相对于每个游戏最高step数的百分比，更公平
            # 对于每个游戏级别，计算相对step效率
            for game_level in agg_data.index.get_level_values('game_level').unique():
                game_mask = agg_data.index.get_level_values('game_level') == game_level
                game_steps = agg_data.loc[game_mask, metric]
                max_steps = game_steps.max()
                if max_steps > 0:
                    # 计算step百分比，步数越多相对效率越高（在合理范围内）
                    normalized_data.loc[game_mask, metric] = game_steps / max_steps
                else:
                    normalized_data.loc[game_mask, metric] = 0
        elif metric == 'normalized_reward':
            # normalized_reward 已经是0-1，无需再归一化
            normalized_data[metric] = agg_data[metric]
        else:
            # 其他指标越高越好
            normalized_data[metric] = (agg_data[metric] - agg_data[metric].min()) / (agg_data[metric].max() - agg_data[metric].min() + 1e-8)
    
    # 计算综合得分
    comprehensive_score = normalized_data.mean(axis=1)
    pivot_comprehensive = comprehensive_score.unstack(fill_value=0)
    
    # 重新排序行并添加分隔符行
    pivot_comprehensive_ordered = create_ordered_pivot_with_separator(pivot_comprehensive, ordered_models, separator_pos)
    
    # 创建注释矩阵
    annot_comprehensive = pivot_comprehensive_ordered.applymap(
        lambda x: "" if pd.isna(x) else (f"{int(x)}" if float(x).is_integer() else f"{x:.2f}")
    )
    
    plt.figure(figsize=(20, 12))
    mask = pivot_comprehensive_ordered.isna()
    sns.heatmap(pivot_comprehensive_ordered, annot=annot_comprehensive, fmt='', cmap='RdYlGn', 
                mask=mask, cbar_kws={'label': 'Comprehensive Performance Score'})
    plt.title('Comprehensive Performance Score by Model and Game\n(Normalized combination of meaningful ratio, efficiency, reward, and win rate)')
    plt.xlabel('Game-Level')
    plt.ylabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_stats(df, output_dir="heatmaps_extended"):
    """创建汇总统计表"""
    print("Creating summary statistics...")
    
    # 按模型汇总
    model_columns = ['meaningful_ratio']
    if 'total_steps' in df.columns:
        model_columns.append('total_steps')
    if 'total_reward' in df.columns:
        model_columns.append('total_reward')
    if 'won' in df.columns:
        model_columns.append('won')
    
    model_summary = df.groupby('model')[model_columns].agg(['mean', 'std', 'count']).round(3)
    model_summary.to_csv(f"{output_dir}/model_summary.csv")
    
    # 按游戏汇总
    game_summary = df.groupby('game_level')[model_columns].agg(['mean', 'std', 'count']).round(3)
    game_summary.to_csv(f"{output_dir}/game_summary.csv")
    
    # 详细结果
    df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    
    print(f"Summary statistics saved to {output_dir}/")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析LLM游戏结果并生成热力图')
    parser.add_argument('--mode', choices=['6games', 'all'], default='all',
                        help='分析模式: 6games (分析6个游戏数据源) 或 all (分析全部游戏数据源)')
    args = parser.parse_args()
    
    print(f"运行模式: {'6个游戏数据源' if args.mode == '6games' else '全部游戏数据源'}")
    
    # 根据模式选择不同的数据源
    if args.mode == '6games':
        # 6个游戏模式：只使用真正只包含6个游戏的数据源
        data_sources = [
            {
                'path': '../llm_agent_runs_output_portkey_6games',
                'label': '6games'
            },
            {
                'path': 'llm_agent_runs_output/deepseek-r1',
                'label': 'deepseek-r1-6games'
            }
        ]
        print("数据源：../llm_agent_runs_output_portkey_6games, llm_agent_runs_output/deepseek-r1")
    else:
        # 全部游戏模式：只使用all games数据源
        data_sources = [
            {
                'path': 'llm_agent_runs_output/portkey_4o-mini',
                'label': 'allgames'
            }
        ]
        print("数据源：portkey_4o_mini_all_games_run")
    
    all_results = []
    
    # 扫描选定的数据源
    for source in data_sources:
        results = scan_results_directory(source['path'], source['label'])
        all_results.extend(results)
    
    if not all_results:
        print("No results found!")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 如果是6games模式，只保留核心游戏
    if args.mode == '6games':
        df = df[df['game'].isin(CORE_GAMES)]
        print(f"6games模式：过滤后保留 {len(df)} 个结果 (仅6个核心游戏)")
        print(f"保留的游戏: {sorted(df['game'].unique())}")
    
    # 模型名友好映射和清理
    model_map = {
        # GPT模型映射
        "portkey-4o-mini": "GPT-4o-mini",
        "portkey-4o-mini-6games": "GPT-4o-mini",
        "portkey-4o-mini-project": "GPT-4o-mini",
        "portkey-4o-mini-allgames": "GPT-4o-mini",
        "portkey": "GPT-o3-mini",
        "portkey-6games": "GPT-o3-mini",
        "portkey-project": "GPT-o3-mini",
        
        # Gemini模型映射
        "gemini": "gemini-2.0-exp.-flash",
        "gemini-6games": "gemini-2.0-exp.-flash",
        "gemini-allgames": "gemini-2.0-exp.-flash",
        "portkey-gemini-allgames": "gemini-2.0-exp.-flash",
        "gemini-pro": "Gemini-2.5-pro",
        "gemini-pro-6games": "Gemini-2.5-pro",
        
        # DeepSeek模型映射
        "deepseek": "DeepSeek-chat",
        "deepseek-project": "DeepSeek-chat",
        "deepseek-r1": "DeepSeek-reasoner",
        "deepseek-r1-project": "DeepSeek-reasoner",
        
        # 其他模型
        "claude-4-6games": "Claude-4"
    }
    
    df['model'] = df['model'].map(lambda x: model_map.get(x, x))
    
    # 数据类型清理
    if 'won' in df.columns:
        # 确保won列是数值类型
        df['won'] = df['won'].astype(float)
    
    # 按 game_level 归一化 total_reward，跨 model
    if 'total_reward' in df.columns:
        df['normalized_reward'] = df.groupby('game_level')['total_reward'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
    
    print(f"分析 {len(df)} 个结果，来自 {df['model'].nunique()} 个模型和 {df['game_level'].nunique()} 个游戏-关卡")
    print(f"模型列表: {sorted(df['model'].unique())}")
    
    # 根据模式创建不同的输出目录
    if args.mode == '6games':
        output_dir = "heatmaps_6games_extended"
        mode_desc = "6个游戏数据源"
    else:
        output_dir = "heatmaps_allgames_extended"
        mode_desc = "全部游戏数据源"
    
    print(f"输出目录: {output_dir}")
    
    # 创建热力图
    create_heatmaps(df, output_dir)
    
    # 创建汇总统计
    create_summary_stats(df, output_dir)
    
    print(f"\n{mode_desc}分析完成! 结果保存到 {output_dir}/")
    print("Generated files:")
    print("- meaningful_ratio_heatmap.png")
    print("- total_reward_heatmap.png (if reward data available)")
    print("- normalized_reward_heatmap.png (if reward data available)")
    print("- win_rate_heatmap.png (if win data available)")
    print("- total_steps_heatmap.png (if steps data available)")
    print("- comprehensive_performance_heatmap.png")
    print("- model_summary.csv")
    print("- game_summary.csv")
    print("- detailed_results.csv")

if __name__ == "__main__":
    main()
