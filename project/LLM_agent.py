import os
import gym
import gym_gvgai as gvgai
import numpy as np
import re
import pygame
import matplotlib.pyplot as plt
from llm.client import LLMClient
from collections import defaultdict, Counter
from typing import Iterable


# 游戏状态可视化模块
def show_state(env, step, name, info, vgdl_representation=None):
    plt.figure(3)
    plt.clf()
    try:
        img = env.render(mode='rgb_array')
        plt.imshow(img)
    except Exception as e:
        if vgdl_representation:
            img = vgdl_to_image(vgdl_representation)
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, "无法获取图像", fontsize=14, ha='center')

    plt.title(f"{name} | Step: {step} {info}")
    plt.axis("off")
    os.makedirs('imgs', exist_ok=True)
    path = f'imgs/{name}_{len(os.listdir("imgs")) + 1}.png'
    plt.savefig(path)


def vgdl_to_image(vgdl_representation):
    pygame.init()
    font = pygame.font.Font(None, 24)
    surface = pygame.Surface((300, 300))
    surface.fill((0, 0, 0))
    for i, line in enumerate(vgdl_representation.split("\n")):
        text = font.render(line, True, (255, 255, 255))
        surface.blit(text, (10, i * 20))
    return pygame.surfarray.array3d(surface)


# 正确代码
def parse_vgdl_level(vgdl_level):
    max_width = max(len(row) for row in vgdl_level)
    padded_level = [row.ljust(max_width, ".") for row in vgdl_level]
    return np.array([list(row) for row in padded_level])




# 增强奖励系统核心模块
class EnhancedRewardSystem:
    def __init__(self, action_space_size: int, window_size=15):
        self.action_history = []
        self.reward_history = []
        self.action_efficacy = defaultdict(list)
        self.consecutive_zero_threshold = 3
        self.window_size = window_size

    def update(self, action: int, reward: float):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.action_efficacy[action].append(reward)

        if len(self.action_history) > self.window_size:
            removed_action = self.action_history.pop(0)
            self.reward_history.pop(0)
            if self.action_efficacy[removed_action]:
                self.action_efficacy[removed_action].pop(0)

    def generate_guidance(self) -> str:
        if self.get_zero_streak() >= self.consecutive_zero_threshold:
            return self._zero_reward_analysis()
        return self._performance_summary()

    def get_zero_streak(self) -> int:
        """计算连续零奖励次数（修复版本）"""
        return next(
            (i for i, r in enumerate(reversed(self.reward_history)) if r != 0),
            len(self.reward_history))

    def _zero_reward_analysis(self) -> str:
        recent_actions = self.action_history[-self.consecutive_zero_threshold:]
        action_stats = {a: (np.mean(self.action_efficacy[a]), len(self.action_efficacy[a]))
                        for a in set(recent_actions)}
        return f"""
    [系统警告] 连续{self.get_zero_streak()}次零奖励
    最近动作序列: {recent_actions}
    效能分析:
    {chr(10).join(f'- 动作{a}: 平均奖励{avg:.2f} (尝试次数{count})'
                  for a, (avg, count) in action_stats.items())}
    建议尝试新动作组合或检查规则合规性，注意reward会有一定的延迟，并且只有动作1能获得reward，但是组合动作2,3（改变位置）然后再释放动作1会更有概率获得reward。
    """

    def _performance_summary(self) -> str:
        top_actions = sorted([(a, np.mean(r)) for a, r in self.action_efficacy.items() if r],
                             key=lambda x: x[1], reverse=True)[:3]
        return f"""
    [策略分析] 最佳动作:
    {chr(10).join(f'- 动作{a}: 平均奖励{reward:.2f}' for a, reward in top_actions)}
    """

    # LLM交互模块
def build_enhanced_prompt(vgdl_rules: str, state: str,
                          action_map: dict, reward_system: EnhancedRewardSystem) -> str:
    base = f'''你正在控制游戏角色，请根据以下信息决策action, 并只返回action index：

    游戏规则：
    {vgdl_rules}

    当前状态：
    {state}

    可用动作：
    {chr(10).join(f'{k}: {v}' for k, v in action_map.items())}
    '''
    return base + reward_system.generate_guidance()


def query_llm(llm_client: LLMClient, vgdl_rules: str,
              current_state: str, action_map: dict,
              reward_system: EnhancedRewardSystem,
              step: int) -> int:  # 添加step参数
    prompt = build_enhanced_prompt(vgdl_rules, current_state, action_map, reward_system)
    try:
        response = llm_client.query(prompt)
        numbers = [int(m.group()) for m in re.finditer(r'\d+', response)]
        selected_action = next((n for n in numbers if n in action_map), 0)

        # 新增动作选择显示
        print(f"\n=== Step {step} ===")
        print(f"Selected Action: {selected_action} ({action_map[selected_action]})")
        print(f"Full Response: {response[:100]}...")  # 显示前100字符防止刷屏

        return selected_action
    except Exception as e:
        print(f"LLM请求异常: {str(e)}")
        return 0





def generate_report(system: EnhancedRewardSystem):
    print(f"\n=== 游戏分析报告 ===")
    print(f"总步数: {len(system.reward_history)}")
    print(f"总奖励: {sum(system.reward_history)}")
    print(f"最大零奖励连续步数: {system.get_zero_streak()}")

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(system.reward_history)
    plt.title("奖励变化趋势")

    plt.subplot(122)
    action_dist = Counter(system.action_history)
    plt.bar(action_dist.keys(), action_dist.values())
    plt.title("动作分布")
    plt.savefig("game_analysis.png")

if __name__ == "__main__":

    env = gvgai.make("gvgai-aliens-lvl0-v0")
    state = env.reset()
    done = False

    # VGDL规则加载模块
    game_name = env.spec.id.replace("gvgai-", "").split("-")[0] + "_v0"
    current_path = os.path.dirname(os.path.abspath(__file__))
    game_dir = os.path.join(os.path.dirname(current_path), "gym_gvgai", "envs", "games", game_name)

    vgdl_rule_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                           if f.endswith(".txt") and "lvl" not in f), None)
    level_layout_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                              if f.endswith(".txt") and "lvl" in f), None)

    if not vgdl_rule_file or not level_layout_file:
        raise FileNotFoundError("缺少游戏配置文件")

    with open(vgdl_rule_file, "r") as f:
        vgdl_rules = f.read()

        # **转换 VGDL Level**
    vgdl_grid = parse_vgdl_level(level_layout_file)
    h, w = vgdl_grid.shape
    print("VGDL 关卡网格大小:", h, "x", w)

    # 动作空间配置
    available_actions = list(range(env.action_space.n))
    try:
        action_mapping = {i: env.unwrapped.get_action_meanings()[i] for i in available_actions}
    except AttributeError:
        action_mapping = {i: f"Action {i}" for i in available_actions}

    # env = gvgai.make("gvgai-aliens-lvl0-v0")
    # state = env.reset()

    llm_client = LLMClient("openai")
    reward_system = EnhancedRewardSystem(env.action_space.n)

    try:
        step_count = 0
        while not done:
            try:
                game_state = env.unwrapped.get_observation()
            except AttributeError:
                game_state = "\n".join(["".join(row) for row in state[..., 0].astype(int).astype(str)])

            action = query_llm(llm_client, vgdl_rules, game_state, action_mapping, reward_system,step_count)

            next_state, reward, done, _ = env.step(action)
            reward_system.update(action, reward)

            print(f"Received Reward: {reward}")
            if reward != 0:
                print("Positive Reward Detected!")

            if reward_system.get_zero_streak() >= 5:
                print("Action Divergence")

            show_state(env, step_count,  # 使用统一step计数
                       "enhanced_agent",
                       f"Reward: {reward} | Action: {action}",  # 标题添加动作信息
                       game_state)

            state = next_state
            step_count += 1  # 递增步骤计数器
    finally:
        env.close()
        generate_report(reward_system)
