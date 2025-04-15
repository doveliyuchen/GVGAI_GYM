import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from collections import Counter

# Parse into structured data from ./project directory
game_dir = os.listdir('./project')
txt_files = [f for f in game_dir if f.endswith('.txt')]
records = []

for txt_file in txt_files:
    with open(os.path.join('./project', txt_file), 'r') as file:
        lines = file.readlines()
        for line in lines:
            entry = dict(re.findall(r'(\w+): ([^,]+)', line))
            if entry:
                records.append(entry)

df = pd.DataFrame(records)
df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
df['step_count'] = pd.to_numeric(df['step_count'], errors='coerce')
df['api'] = df['api'].astype(str)
df['game_name'] = df['game_name'].astype(str)
df['winner'] = df['winner'].astype(str)

# Step 1: 标记 win 信息和有效记录
df['is_win'] = df['winner'] == 'PLAYER_WINS'
df['is_valid'] = df['winner'].isin(['PLAYER_WINS', 'PLAYER_LOSES'])
TIMEOUT_THRESHOLD = 1000
df['timed_out_no_reward'] = (df['step_count'] >= TIMEOUT_THRESHOLD) & (df['reward'] == 0)
df['is_unbeaten'] = (df['winner'] == 'NO_WINNER') | df['timed_out_no_reward']

# Step 2: 每个游戏是否至少赢过一次 / 是否所有记录都为“unbeaten”
game_win_flags = df.groupby('game_name')['is_win'].any().reset_index(name='has_win')
game_unbeaten_flags = df.groupby('game_name')['is_unbeaten'].all().reset_index(name='is_unbeaten')

# Step 3: 计算每个游戏的最大 step（用于归一化 win step）
game_max_steps = df.groupby('game_name')['step_count'].max().reset_index(name='max_step')

# Step 4: 筛选赢的记录用于计算 avg win step
win_df = df[df['is_win'] == True]
game_avg_win_steps = win_df.groupby('game_name')['step_count'].mean().reset_index(name='avg_win_step')

# Step 5: 合并所有信息到 game_stats
game_stats = df.groupby('game_name').agg(
    num_apis_tested=('api', 'nunique'),
    avg_reward=('reward', 'mean'),
    win_rate=('is_win', 'mean'),
    max_reward=('reward', 'max'),
    min_reward=('reward', 'min')
).reset_index()

game_stats = game_stats.merge(game_win_flags, on='game_name', how='left')
game_stats = game_stats.merge(game_unbeaten_flags, on='game_name', how='left')
game_stats = game_stats.merge(game_max_steps, on='game_name', how='left')
game_stats = game_stats.merge(game_avg_win_steps, on='game_name', how='left')

# Step 6: center-normalized clipped reward based on min/max only
rewards = game_stats['avg_reward'].values
sorted_rewards = np.sort(rewards)
clip_min = sorted_rewards[1] if len(sorted_rewards) > 1 else sorted_rewards[0]
clip_max = sorted_rewards[-2] if len(sorted_rewards) > 1 else sorted_rewards[-1]
p50 = np.median(rewards)

clipped_rewards = np.clip(rewards, clip_min, clip_max)
game_stats['normalized_reward'] = (clipped_rewards - p50) / (clip_max - clip_min + 1e-8) * 0.5 + 0.5

# Step 7: step score（只有赢的游戏才有值，否则为0）
game_stats['step_score'] = 1 - (game_stats['avg_win_step'] / game_stats['max_step'])
game_stats['step_score'] = game_stats['step_score'].fillna(0.0)

# Step 8: final score（正常保留所有评分）
game_stats['final_score'] = 0.6 * game_stats['win_rate'] + \
                             0.2 * game_stats['normalized_reward'] + \
                             0.2 * game_stats['step_score']

# Step 9: 打标签（仅对所有记录为 NO_WINNER 或 timeout 的游戏标记为 unbeaten）
def label_difficulty(row):
    if row['is_unbeaten']:
        return 'unbeaten'
    score = row['final_score']
    if score >= 0.8:
        return 'very_easy'
    elif score >= 0.6:
        return 'easy'
    elif score >= 0.4:
        return 'medium'
    elif score >= 0.2:
        return 'hard'
    else:
        return 'very_hard'

game_stats['difficulty_label'] = game_stats.apply(label_difficulty, axis=1)

# Step 10: 可视化评分结果并保存
plt.figure(figsize=(12, 6))
game_stats_sorted = game_stats.sort_values(by='final_score', ascending=False)
plt.barh(game_stats_sorted['game_name'], game_stats_sorted['final_score'], color='skyblue')
plt.xlabel('Final Score (0~1)')
plt.title('Cross-API Game Evaluation Score')
plt.tight_layout()
plt.savefig('gvgai_game_score_plot.png')
plt.show()

# Step 11: 保存打分表格到 CSV
game_stats.to_csv('gvgai_game_difficulty_scores.csv', index=False)

# Optional: print
print(game_stats.sort_values(by='final_score', ascending=False))