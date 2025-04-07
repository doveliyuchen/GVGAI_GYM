import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取所有 txt 文件
txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
records = []

for txt_file in txt_files:
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            entry = dict(re.findall(r'(\w+): ([^,]+)', line))
            if entry:
                records.append(entry)

# 2. 构建 DataFrame
df = pd.DataFrame(records)
df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
df['step_count'] = pd.to_numeric(df['step_count'], errors='coerce')
df['api'] = df['api'].astype(str)
df['game_name'] = df['game_name'].astype(str)

# 3. 创建 win 字段
def compute_win(winner):
    if winner == 'PLAYER_WINS':
        return 1
    elif winner == 'PLAYER_LOSES':
        return 0
    else:
        return float('nan')  # NO_WINNER 为 NaN，灰色显示

df['win'] = df['winner'].apply(compute_win)

# 4. 构建透视表：game_name 为列，api 为行

winrate_matrix = df.pivot_table(index='api', columns='game_name', values='win', aggfunc='mean')
reward_matrix = df.pivot_table(index='api', columns='game_name', values='reward', aggfunc='mean')
fig_width = max(len(winrate_matrix.columns) * 0.3, 5)  # 最小宽度 12
fig_height = max(len(winrate_matrix.index) * 0.3, 5)   # 最小高度 10
# 5. 画胜率热力图
plt.figure(figsize=(fig_width, fig_height))
sns.heatmap(
    winrate_matrix,
    cmap='RdYlGn',
    linewidths=0.5,
    annot=False,
    cbar=True,
    mask=winrate_matrix.isna()
)
plt.title('Win Rate Heatmap (api × game)')
plt.xlabel('Game')
plt.ylabel('API')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('winrate_matrix_heatmap.png')
plt.close()

# 6. 画 reward 热力图
plt.figure(figsize=(fig_width, fig_height))
sns.heatmap(
    reward_matrix,
    cmap='RdYlGn',
    linewidths=0.5,
    annot=False,
    cbar=True,
    mask=reward_matrix.isna()
)
plt.title('Reward Heatmap (api × game)')
plt.xlabel('Game')
plt.ylabel('API')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('reward_matrix_heatmap.png')
plt.close()

print("2D heatmaps saved: 'winrate_matrix_heatmap.png' & 'reward_matrix_heatmap.png'")
