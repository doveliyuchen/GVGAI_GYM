import os
import openai
import gym
import gym_gvgai as gvgai
import numpy as np
import re
import pygame
import matplotlib.pyplot as plt
from llm.client import LLMClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
# **显示当前状态**
def show_state(env, step, name, info, vgdl_representation=None):
    plt.figure(3)
    plt.clf()

    try:
        # 尝试获取 RGB 渲染
        img = env.render(mode='rgb_array')
        plt.imshow(img)
    except Exception as e:
        print(f"⚠️ env.render() 失败，使用 VGDL 显示: {e}")
        if vgdl_representation:
            img = vgdl_to_image(vgdl_representation)
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, "无法获取图像", fontsize=14, ha='center')

    plt.title(f"{name} | Step: {step} {info}")
    plt.axis("off")

    # **确保 `imgs/` 目录存在**
    os.makedirs('imgs', exist_ok=True)

    # **自动递增文件名，防止覆盖**
    existing_files = [f for f in os.listdir('imgs') if f.startswith(name) and f.endswith('.png')]
    file_index = len(existing_files) + 1  # 计算新的文件编号
    path = f'imgs/{name}_{file_index}.png'

    plt.savefig(path)
    print(f" 截图已保存: {path}")


# **VGDL 转换成简单图像**
def vgdl_to_image(vgdl_representation):
    pygame.init()
    font = pygame.font.Font(None, 24)
    img_size = (300, 300)  # 固定大小
    surface = pygame.Surface(img_size)
    surface.fill((0, 0, 0))  # 黑色背景

    # 渲染 VGDL 关卡
    lines = vgdl_representation.split("\n")
    for i, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))  # 白色文字
        surface.blit(text, (10, i * 20))  # 每行间隔 20px

    return pygame.surfarray.array3d(surface)  # 转换成 numpy array 供 Matplotlib 显示


# **初始化 GVGAI 环境**
env = gvgai.make("gvgai-aliens-lvl0-v0")
state = env.reset()
done = False

# **解析 game_name**
game_name = env.spec.id.replace("gvgai-", "").split("-")[0] + "_v0"

# **路径配置**
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
game_dir = os.path.join(root_path, "gym_gvgai", "envs", "games", game_name)
vgdl_rule_file = None
level_layout_file = None

# **查找 VGDL 规则文件**
for file in os.listdir(game_dir):
    if file.endswith(".txt") and "lvl" not in file:
        vgdl_rule_file = os.path.join(game_dir, file)
        break

# **查找 Level Layout 文件**
for file in os.listdir(game_dir):
    if file.endswith(".txt") and "lvl" in file:
        level_layout_file = os.path.join(game_dir, file)
        break

# **检查文件是否存在**
if not vgdl_rule_file or not os.path.exists(vgdl_rule_file):
    raise FileNotFoundError(f"VGDL 文件未找到: {vgdl_rule_file}")

if not level_layout_file or not os.path.exists(level_layout_file):
    raise FileNotFoundError(f"Level Layout 未找到: {level_layout_file}")

# **读取 VGDL 规则**
with open(vgdl_rule_file, "r") as f:
    vgdl_rules = f.read()

# **读取 Level Layout**
with open(level_layout_file, "r") as f:
    vgdl_level = [line.strip() for line in f.readlines()]


# **解析 Level Layout**
def parse_vgdl_level(vgdl_level):
    max_width = max(len(row) for row in vgdl_level)
    padded_level = [row.ljust(max_width, ".") for row in vgdl_level]
    return np.array([list(row) for row in padded_level])


# **转换 VGDL Level**
vgdl_grid = parse_vgdl_level(vgdl_level)
h, w = vgdl_grid.shape
print("VGDL 关卡网格大小:", h, "x", w)


# **状态转 VGDL**
def position_to_vgdl(state, vgdl_grid):
    state_np = np.array(state)
    h, w, _ = state_np.shape
    vgdl_output = []

    for i in range(min(h, vgdl_grid.shape[0])):
        vgdl_row = []
        for j in range(min(w, vgdl_grid.shape[1])):
            vgdl_row.append(vgdl_grid[i, j])
        vgdl_output.append("".join(vgdl_row))

    return "\n".join(vgdl_output)


# **获取可用动作列表**
available_actions = list(range(env.action_space.n))

# **尝试获取动作名称**
try:
    action_meanings = env.unwrapped.get_action_meanings()
    action_mapping = {i: action_meanings[i] for i in available_actions}
except AttributeError:
    action_mapping = {i: f"Action {i}" for i in available_actions}

print("可用动作:", action_mapping)


# **查询 LLM 生成动作**
# 记录历史奖励
past_rewards = []


def query_llm_parallel(vgdl_rules, state_text, available_actions, action_mapping, reward):
    past_rewards.append(reward)
    recent_rewards = past_rewards[-5:]

    action_descriptions = "\n".join([f"{i}: {desc}" for i, desc in action_mapping.items()])
    reward_history = ", ".join(map(str, recent_rewards))

    prompt_template = """
    You are a player in this game, and your goal is to win. 
    The game is fully playable, and you must follow the given rules step by step to achieve victory.

    ### **Game Rules (VGDL Format)**
    {rules}

    ### **Current Game State**
    {state}

    ### **Available Actions**
    {actions}

    ### **Recent Rewards**
    {rewards}

    The game follows these rules, and "A" represents the player character.
    Based on the game state and recent rewards, please determine the **best next action** for "A" that will help achieve the **winning condition**.

    **Only return a single integer representing the action (without any extra text).**
    """

    # 准备并行请求
    def generate_action(client: LLMClient, model_name: str):
        try:
            start_time = time.time()
            response = client.query(prompt_template.format(
                rules=vgdl_rules,
                state=state_text,
                actions=action_descriptions,
                rewards=reward_history
            ))
            elapsed = time.time() - start_time

            match = re.search(r"\d+", response)
            if match:
                action = int(match.group(0))
                valid = action in available_actions
                print(
                    f"[{model_name.upper()}] 响应时间: {elapsed:.2f}s | 动作: {action} ({'有效' if valid else '无效'})")
                return action if valid else None
            return None
        except Exception as e:
            print(f"[{model_name.upper()}] 请求失败: {str(e)}")
            return None

    # 初始化客户端（假设已配置好.env）
    clients = [
        ("deepseek", LLMClient("deepseek")),
        ("openai", LLMClient("openai"))
    ]

    # 并行执行请求
    valid_actions = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(generate_action, client, name): name
            for name, client in clients
        }

        for future in as_completed(futures, timeout=5):
            model_name = futures[future]
            try:
                result = future.result()
                if result is not None:
                    valid_actions.append(result)
            except Exception as e:
                print(f"[{model_name.upper()}] 请求异常: {str(e)}")

    # 决策策略（优先选择共同建议，否则随机选择有效动作）
    if len(valid_actions) == 0:
        print("⚠️ 所有模型请求失败，默认执行动作 0")
        return 0

    # 统计动作出现频率
    action_counts = {}
    for action in valid_actions:
        action_counts[action] = action_counts.get(action, 0) + 1

    # 选择最高频的有效动作
    best_action = max(action_counts.items(), key=lambda x: x[1])[0]
    print(f"最终选择动作: {best_action} (得票数: {action_counts[best_action]}/2)")
    return best_action



# **初始化 `pygame`**
# pygame.init()
# screen = pygame.display.set_mode((640, 480))
# pygame.display.set_caption("GVGAI Visualization")
# **游戏主循环**
reward = 0
while not done:
    try:
        vgdl_representation = env.unwrapped.get_observation()  # 获取 VGDL 格式的 game state
    except AttributeError:
        vgdl_representation = position_to_vgdl(state, vgdl_grid)  # 退回像素转换

    print("当前游戏状态:")
    print(vgdl_representation)

    action = query_llm_parallel(vgdl_rules, vgdl_representation, available_actions, action_mapping, reward)
    print(f"LLM 选择的动作: {action} ({action_mapping.get(action, '未知动作')})")

    # **执行动作，获取新的状态和奖励**
    state, reward, done, info = env.step(action)
    show_state(env, step=len(info), name="game", info=f"Reward: {reward}")

# env.close()


# 关闭 pygame
# pygame.quit()
env.close()
