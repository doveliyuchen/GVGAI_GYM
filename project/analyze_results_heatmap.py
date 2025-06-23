#!/usr/bin/env python3
"""
分析LLM游戏结果并生成热力图的脚本
处理 llm_agent_runs_output 目录中的JSON结果文件
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
        with open(step_metrics_path, 'r') as f:
            data = json.load(f)
        
        # 计算基本指标
        meaningful_steps = data.get('meaningful_steps', [])
        avatar_positions = data.get('avatar_positions', [])
        
        total_steps = len(meaningful_steps)
        meaningful_count = sum(meaningful_steps) if meaningful_steps else 0
        meaningful_ratio = meaningful_count / total_steps if total_steps > 0 else 0
        
        # 计算移动距离
        total_distance = 0
        if len(avatar_positions) > 1:
            for i in range(1, len(avatar_positions)):
                if avatar_positions[i] and avatar_positions[i-1]:
                    pos1 = avatar_positions[i-1][0] if avatar_positions[i-1] else [0, 0]
                    pos2 = avatar_positions[i][0] if avatar_positions[i] else [0, 0]
                    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])  # 曼哈顿距离
                    total_distance += distance
        
        return {
            'total_steps': total_steps,
            'meaningful_steps': meaningful_count,
            'meaningful_ratio': meaningful_ratio,
            'total_distance': total_distance,
            'avg_distance_per_step': total_distance / total_steps if total_steps > 0 else 0
        }
    except Exception as e:
        print(f"Error loading {step_metrics_path}: {e}")
        return None

def load_csv_data(run_dir):
    """加载CSV数据获取奖励信息"""
    csv_path = os.path.join(run_dir, "benchmark_analysis.json", "step_metrics.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        total_reward = df['reward'].sum()
        reward_steps = len(df[df['reward'] > 0])
        max_reward = df['reward'].max()
        
        # 检查游戏是否获胜 (通常最后一步有高奖励表示获胜)
        final_reward = df['reward'].iloc[-1] if len(df) > 0 else 0
        won = final_reward > 0 or total_reward > 10  # 简单的获胜判断逻辑
        
        return {
            'total_reward': total_reward,
            'reward_steps': reward_steps,
            'max_reward': max_reward,
            'final_reward': final_reward,
            'won': won
        }
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def scan_results_directory(base_dir):
    """扫描结果目录并收集所有数据"""
    results = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Results directory {base_dir} does not exist!")
        return results
    
    print(f"Scanning results in {base_dir}...")
    
    # 遍历模型目录
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        
        # 遍历游戏目录
        for game_dir in model_dir.iterdir():
            if not game_dir.is_dir():
                continue
                
            game_name_full = game_dir.name
            game_name, level = extract_game_info(f"gvgai-{game_name_full}-v0")
            
            # 遍历运行目录
            for run_dir in game_dir.iterdir():
                if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
                    continue
                    
                run_id = run_dir.name.replace('run_', '')
                
                # 加载数据
                json_data = load_run_data(run_dir)
                csv_data = load_csv_data(run_dir)
                
                if json_data is not None:
                    result = {
                        'model': model_name,
                        'game': game_name,
                        'level': level,
                        'run_id': int(run_id),
                        'game_level': f"{game_name}-lvl{level}",
                        **json_data
                    }
                    
                    if csv_data is not None:
                        result.update(csv_data)
                    
                    results.append(result)
    
    print(f"Collected {len(results)} results")
    return results

def create_heatmaps(df, output_dir="heatmaps"):
    """创建各种热力图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图形样式
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # 1. 有意义步数比例热力图
    print("Creating meaningful ratio heatmap...")
    pivot_meaningful = df.groupby(['model', 'game_level'])['meaningful_ratio'].mean().unstack(fill_value=0)
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_meaningful, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Meaningful Step Ratio'})
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
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_reward, annot=True, fmt='.1f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Average Total Reward'})
        plt.title('Average Total Reward by Model and Game')
        plt.xlabel('Game-Level')
        plt.ylabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/total_reward_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 获胜率热力图
    if 'won' in df.columns:
        print("Creating win rate heatmap...")
        pivot_winrate = df.groupby(['model', 'game_level'])['won'].mean().unstack(fill_value=0)
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_winrate, annot=True, fmt='.3f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Win Rate'})
        plt.title('Win Rate by Model and Game')
        plt.xlabel('Game-Level')
        plt.ylabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/win_rate_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 平均步数热力图
    print("Creating average steps heatmap...")
    pivot_steps = df.groupby(['model', 'game_level'])['total_steps'].mean().unstack(fill_value=0)
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_steps, annot=True, fmt='.0f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Average Total Steps'})
    plt.title('Average Total Steps by Model and Game')
    plt.xlabel('Game-Level')
    plt.ylabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_steps_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 综合性能热力图 (标准化后的多指标组合)
    print("Creating comprehensive performance heatmap...")
    
    # 选择关键指标进行标准化
    metrics = ['meaningful_ratio', 'total_steps']
    if 'total_reward' in df.columns:
        metrics.append('total_reward')
    if 'won' in df.columns:
        metrics.append('won')
    
    # 计算每个模型-游戏组合的平均值
    agg_data = df.groupby(['model', 'game_level'])[metrics].mean()
    
    # 标准化指标 (0-1范围)
    normalized_data = agg_data.copy()
    for metric in metrics:
        if metric == 'total_steps':
            # 步数越少越好，所以反转
            normalized_data[metric] = 1 - (agg_data[metric] - agg_data[metric].min()) / (agg_data[metric].max() - agg_data[metric].min() + 1e-8)
        else:
            # 其他指标越高越好
            normalized_data[metric] = (agg_data[metric] - agg_data[metric].min()) / (agg_data[metric].max() - agg_data[metric].min() + 1e-8)
    
    # 计算综合得分
    comprehensive_score = normalized_data.mean(axis=1)
    pivot_comprehensive = comprehensive_score.unstack(fill_value=0)
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_comprehensive, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Comprehensive Performance Score'})
    plt.title('Comprehensive Performance Score by Model and Game\n(Normalized combination of meaningful ratio, efficiency, reward, and win rate)')
    plt.xlabel('Game-Level')
    plt.ylabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_stats(df, output_dir="heatmaps"):
    """创建汇总统计表"""
    print("Creating summary statistics...")
    
    # 按模型汇总
    model_summary = df.groupby('model').agg({
        'meaningful_ratio': ['mean', 'std'],
        'total_steps': ['mean', 'std'],
        'total_reward': ['mean', 'std'] if 'total_reward' in df.columns else ['count'],
        'won': ['mean', 'count'] if 'won' in df.columns else ['count']
    }).round(3)
    
    model_summary.to_csv(f"{output_dir}/model_summary.csv")
    
    # 按游戏汇总
    game_summary = df.groupby('game_level').agg({
        'meaningful_ratio': ['mean', 'std'],
        'total_steps': ['mean', 'std'],
        'total_reward': ['mean', 'std'] if 'total_reward' in df.columns else ['count'],
        'won': ['mean', 'count'] if 'won' in df.columns else ['count']
    }).round(3)
    
    game_summary.to_csv(f"{output_dir}/game_summary.csv")
    
    # 详细结果
    df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    
    print(f"Summary statistics saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM game results and create heatmaps')
    parser.add_argument('--input_dir', type=str, default='llm_agent_runs_output',
                        help='Input directory containing results (default: llm_agent_runs_output)')
    parser.add_argument('--output_dir', type=str, default='heatmaps',
                        help='Output directory for heatmaps (default: heatmaps)')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Specific models to analyze (default: all)')
    parser.add_argument('--games', nargs='*', default=None,
                        help='Specific games to analyze (default: all)')
    
    args = parser.parse_args()
    
    # 扫描结果
    results = scan_results_directory(args.input_dir)
    
    if not results:
        print("No results found!")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 过滤模型和游戏
    if args.models:
        df = df[df['model'].isin(args.models)]
        print(f"Filtered to models: {args.models}")
    
    if args.games:
        df = df[df['game'].isin(args.games)]
        print(f"Filtered to games: {args.games}")
    
    if df.empty:
        print("No data remaining after filtering!")
        return
    
    print(f"Analyzing {len(df)} results from {df['model'].nunique()} models and {df['game_level'].nunique()} game-levels")
    
    # 创建热力图
    create_heatmaps(df, args.output_dir)
    
    # 创建汇总统计
    create_summary_stats(df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")
    print("Generated files:")
    print("- meaningful_ratio_heatmap.png")
    print("- total_reward_heatmap.png (if reward data available)")
    print("- win_rate_heatmap.png (if win data available)")
    print("- total_steps_heatmap.png")
    print("- comprehensive_performance_heatmap.png")
    print("- model_summary.csv")
    print("- game_summary.csv")
    print("- detailed_results.csv")

if __name__ == "__main__":
    main()
