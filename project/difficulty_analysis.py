import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load uploaded CSV
df = pd.read_csv("project/game_log.csv")

# Clean reward column
df['reward'] = (
    df['reward']
    .astype(str)
    .str.replace('\r', '', regex=False)
    .str.replace('\n', '', regex=False)
    .str.replace('"', '', regex=False)
    .str.strip()
)
df['reward'] = pd.to_numeric(df['reward'], errors='coerce')

# Convert types
df['step_count'] = pd.to_numeric(df['step_count'], errors='coerce')
df['api'] = df['api'].astype(str)
df['game_name'] = df['game_name'].astype(str)
df['winner'] = df['winner'].astype(str)

# Step 1: Mark win and unbeaten
df['is_win'] = df['winner'] == 'PLAYER_WINS'
df['is_valid'] = df['winner'].isin(['PLAYER_WINS', 'PLAYER_LOSES'])
TIMEOUT_THRESHOLD = 1000
df['timed_out_no_reward'] = (df['step_count'] >= TIMEOUT_THRESHOLD) & (df['reward'] == 0)
df['is_unbeaten'] = (df['winner'] == 'NO_WINNER') | df['timed_out_no_reward']

# Step 2: Win and unbeaten flags per game
game_win_flags = df.groupby('game_name')['is_win'].any().reset_index(name='has_win')
game_unbeaten_flags = df.groupby('game_name')['is_unbeaten'].all().reset_index(name='is_unbeaten')

# Step 3: Max step count
game_max_steps = df.groupby('game_name')['step_count'].max().reset_index(name='max_step')

# Step 4: Avg win step
win_df = df[df['is_win'] == True]
game_avg_win_steps = win_df.groupby('game_name')['step_count'].mean().reset_index(name='avg_win_step')

# Step 5: Aggregate stats
game_stats = df.groupby('game_name').agg(
    num_apis_tested=('api', 'nunique'),
    avg_reward=('reward', 'mean'),
    win_rate=('is_win', 'mean'),
    max_reward=('reward', 'max'),
    min_reward=('reward', 'min')
).reset_index()

# Merge all info
game_stats = game_stats.merge(game_win_flags, on='game_name', how='left')
game_stats = game_stats.merge(game_unbeaten_flags, on='game_name', how='left')
game_stats = game_stats.merge(game_max_steps, on='game_name', how='left')
game_stats = game_stats.merge(game_avg_win_steps, on='game_name', how='left')

# Step 6: Normalized reward
rewards = game_stats['avg_reward'].dropna().values
if len(rewards) > 1:
    sorted_rewards = np.sort(rewards)
    clip_min = sorted_rewards[1] if len(sorted_rewards) > 1 else sorted_rewards[0]
    clip_max = sorted_rewards[-2] if len(sorted_rewards) > 1 else sorted_rewards[-1]
    p50 = np.median(rewards)
    clipped_rewards = np.clip(game_stats['avg_reward'].fillna(p50), clip_min, clip_max)
    game_stats['normalized_reward'] = (clipped_rewards - p50) / (clip_max - clip_min + 1e-8) * 0.5 + 0.5
else:
    game_stats['normalized_reward'] = 0.5

# Step 7: Step score
game_stats['step_score'] = 1 - (game_stats['avg_win_step'] / game_stats['max_step'])
game_stats['step_score'] = game_stats['step_score'].fillna(0.0)

# Step 8: Final score
game_stats['final_score'] = (
    0.6 * game_stats['win_rate'] +
    0.2 * game_stats['normalized_reward'] +
    0.2 * game_stats['step_score']
)

# Step 9: Difficulty label
def label_difficulty(row):
    if row.get('is_unbeaten', False):
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

# Save plot
plt.figure(figsize=(12, 6))
game_stats_sorted = game_stats.sort_values(by='final_score', ascending=False)
plt.barh(game_stats_sorted['game_name'], game_stats_sorted['final_score'], color='skyblue')
plt.xlabel('Final Score (0~1)')
plt.title('Cross-API Game Evaluation Score')
plt.tight_layout()
plot_path = "gvgai_game_score_plot.png"
plt.savefig(plot_path)
plt.close()

# Save results to CSV
csv_path = "gvgai_game_difficulty_scores.csv"
game_stats.to_csv(csv_path, index=False)


plot_path, csv_path
