# test_game.py 使用说明

这个脚本允许你指定具体的模型路径来测试单个游戏的性能。

## 基本用法

```bash
python test_game.py <模型路径> <游戏名称>
```

## 参数说明

### 必需参数
- `model_path`: 训练好的模型文件路径 (.zip 格式)
- `game`: 要测试的游戏名称 (如 'aliens', 'zelda', 'sokoban')

### 可选参数
- `--levels`: 要测试的关卡列表 (默认: 0 1 2 3 4)
- `--episodes`: 每个关卡的测试回合数 (默认: 5)
- `--max-steps`: 每回合最大步数 (默认: 2000)
- `--algorithm`: 算法类型 (默认: auto，可选: DQN, PPO)
- `--output-prefix`: 输出文件前缀 (默认: 自动生成)
- `--quiet`: 减少输出详细度

## 使用示例

### 示例 1: 基本测试
```bash
# 测试 DQN 模型在 aliens 游戏上的表现
python test_game.py "dqn_gvgai_logs/best_model_gold_digger_lvl0.zip" golddigger
```

### 示例 2: 指定关卡和回合数
```bash
# 只测试关卡 0 和 1，每个关卡跑 10 次
python test_game.py "multi_game_dqn_logs/aliens_dqn_logs/best_dqn_model_aliens.zip" aliens --levels 0 1 --episodes 10
```

### 示例 3: 测试 PPO 模型
```bash
# 测试 PPO 模型
python test_game.py "multi_game_ppo_logs/zelda_ppo_logs/best_ppo_model_zelda.zip" zelda --algorithm PPO
```

### 示例 4: 快速测试（减少输出）
```bash
# 静默模式，只输出最终结果
python test_game.py "path/to/model.zip" sokoban --quiet --episodes 3
```

### 示例 5: 自定义输出文件名
```bash
# 指定输出文件前缀
python test_game.py "model.zip" aliens --output-prefix "my_test_results"
```

## 输出文件

脚本会生成两个文件：
1. `{prefix}.csv`: 详细的测试结果数据
2. `{prefix}_summary.txt`: 测试结果摘要

## 支持的游戏名称

常见的游戏名称包括：
- aliens
- zelda  
- sokoban
- boulderdash
- escape
- realsokoban
- golddigger
- 等等...

## 算法自动检测

如果不指定 `--algorithm` 参数，脚本会根据模型文件路径自动检测算法类型：
- 包含 "dqn" 的路径 → DQN
- 包含 "ppo" 的路径 → PPO
- 无法检测时默认为 DQN

## 错误处理

如果遇到以下错误：
1. 模型文件不存在 → 检查路径是否正确
2. 游戏环境无法创建 → 确认游戏名称正确
3. 模型加载失败 → 检查算法类型是否匹配

## 与其他脚本的区别

- `test_dqn_agent_6_games.py`: 自动发现并测试所有模型
- `test_game.py`: 测试指定的单个模型
- `example/load_trained_model.py`: 加载模型进行简单推理

选择合适的脚本根据你的需求使用。
