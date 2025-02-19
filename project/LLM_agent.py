import os
import gym
import gym_gvgai as gvgai
import numpy as np
import re
import pygame
import matplotlib.pyplot as plt
from llm.client import LLMClient
from collections import defaultdict, Counter
from typing import Iterable, Tuple


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


class ReflectionManager:
    def __init__(self, max_history=3):
        self.history = []
        self.max_history = max_history

    def add_reflection(self, reflection: str):
        """Added a reflection to the history"""
        if reflection:
            self.history.append(reflection)
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def get_formatted_history(self) -> str:
        """get the fomatted history"""
        return "\n".join(f"[History reflection{i + 1}] {r}"
                         for i, r in enumerate(self.history))


def parse_vgdl_level(vgdl_level):
    max_width = max(len(row) for row in vgdl_level)
    padded_level = [row.ljust(max_width, ".") for row in vgdl_level]

    # 添加avatar位置检测
    avatar_pos = None
    for y, row in enumerate(padded_level):
        if 'A' in row:
            x = row.index('A')
            avatar_pos = (x, y)
            break

    return np.array([list(row) for row in padded_level]), avatar_pos  # 返回元组


# 增强奖励系统核心模块
class EnhancedRewardSystem:
    def __init__(self, action_space_size: int, window_size=15):
        self.action_history = []
        self.reward_history = []
        self.action_efficacy = defaultdict(list)
        self.consecutive_zero_threshold = 3
        self.window_size = window_size
        self.total_reward = 0.0

    def update(self, action: int, reward: float):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.action_efficacy[action].append(reward)
        self.total_reward += reward

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
        return next(
            (i for i, r in enumerate(reversed(self.reward_history)) if r != 0),
            len(self.reward_history))

    def _zero_reward_analysis(self) -> str:
        recent_actions = self.action_history[-self.consecutive_zero_threshold:]
        action_stats = {a: (np.mean(self.action_efficacy[a]), len(self.action_efficacy[a]))
                        for a in set(recent_actions)}
        return f"""
   Zero reward for {self.get_zero_streak()} consecutive times.
    Recent action sequence: {recent_actions}
    Performance analysis:
    {chr(10).join(f'- Action {a}: Average reward {avg:.2f} (Attempt count {count})'
                  for a, (avg, count) in action_stats.items())}
    It is recommended to try new action combinations or check rule compliance. 
    Note that there may be a certain delay in rewards, and only Action 1 can yield rewards. 
    Combining Actions 2 and 3 (changing position) before executing Action 1 will increase the probability of obtaining a reward.
    Recommended strategy:

    Avoid repetitive patterns
    Get the reward
    """

    def _performance_summary(self) -> str:
        top_actions = sorted([(a, np.mean(r)) for a, r in self.action_efficacy.items() if r],
                             key=lambda x: x[1], reverse=True)[:3]
        return f"""
    [策略分析] 最佳动作:
    {chr(10).join(f'- 动作{a}: 平均奖励{reward:.2f}' for a, reward in top_actions)}
    """

    # LLM交互模块


def build_enhanced_prompt(vgdl_rules: str,
                          state: str,
                          action_map: dict,
                          reward_system: EnhancedRewardSystem,
                          reflection_mgr: ReflectionManager) -> str:
    """Build English prompt with layout and reflection history"""
    # 获取最近一次动作
    last_action = None
    last_reward = None
    if reward_system.action_history:
        last_action = reward_system.action_history[-1]
        last_reward = reward_system.reward_history[-1]
    last_action_desc = action_map.get(last_action, "None") if last_action is not None else "None"
    last_reward_desc = action_map.get(last_reward, "None") if last_reward is not None else "None"
    base = f'''
    You are controlling avatar A. Respond in this format:
    Action: <action number>
    Reflection: ```<your strategy reflection>```


    === Game Rules ===
    {vgdl_rules}

    === Current State ===
    {state}

    === Last Action ===
    {last_action} ({last_action_desc})

    === Available Actions ===
    {chr(10).join(f'{k}: {v}' for k, v in action_map.items())}
    '''

    reflection_section = ""
    if reflection_mgr.history:
        reflection_section = f"\n=== Reflection History ===\n{reflection_mgr.get_formatted_history()}"

    # 添加射击提醒逻辑
    reward_reminder = ""
    if last_reward == 0 and last_action is not None:
        reward_reminder = "\n* The reward may delay from the action, please analyse the rule and think about the strategy. "
    elif last_action is None:
        reward_reminder = "\n* The final goal is win the game."

    guidance = f'''
    {reward_system.generate_guidance()} 
    * Critical Insight: Only some action may direct rewards
    * Strategic Priority: 
      1. Avoid using same action repeatedly
    {reward_reminder}
    '''

    return f"{base}{reflection_section}\n{guidance}"


def query_llm(llm_client: LLMClient,
              vgdl_rules: str,
              current_state: str,
              action_map: dict,
              reward_system: EnhancedRewardSystem,
              reflection_mgr: ReflectionManager,
              step: int) -> Tuple[int, str]:

    prompt = build_enhanced_prompt(vgdl_rules, current_state, action_map,
                                   reward_system, reflection_mgr)
    try:
        response = llm_client.query(prompt)


        action_match = re.search(r"Action:\s*(\d+)", response)
        reflection_match = re.search(r"Reflection:\s*```(.*?)```", response, re.DOTALL)

        action = int(action_match.group(1)) if action_match else 0
        reflection = reflection_match.group(1).strip() if reflection_match else ""

        action = action if action in action_map else 0

        print(f"\n=== Step {step} ===")
        print(f"Selected Action: {action} ({action_map.get(action, 'Unknown')})")
        if reflection:
            print(f"Strategy Reflection: {reflection[:200]}...")

        return action, reflection

    except Exception as e:
        print(f"LLM query error: {str(e)}")
        return 0, ""





def generate_report(system: EnhancedRewardSystem):
    print(f"\n=== Game analysis ===")
    print(f"Total steps: {len(system.reward_history)}")
    print(f"Total reward: {sum(system.reward_history)}")
    print(f"Zero Streak: {system.get_zero_streak()}")

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(system.reward_history)
    plt.title("Reward trend")

    plt.subplot(122)
    action_dist = Counter(system.action_history)
    plt.bar(action_dist.keys(), action_dist.values())
    plt.title("Action distribution")
    plt.savefig("game_analysis.png")

if __name__ == "__main__":

    env = gvgai.make("gvgai-angelsdemons-lvl0-v0")
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
        raise FileNotFoundError("No file detected")

    with open(vgdl_rule_file, "r") as f:
        vgdl_rules = f.read()

        # **转换 VGDL Level**
    vgdl_grid, avatar_pos = parse_vgdl_level(level_layout_file)
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
    reflection_mgr = ReflectionManager()
    reward_system = EnhancedRewardSystem(env.action_space.n)

    try:
        step_count = 0
        while not done:
            try:
                game_state = env.unwrapped.get_observation()
            except AttributeError:
                game_state = parse_vgdl_level(level_layout_file)

            action, reflection = query_llm(llm_client, vgdl_rules, game_state,action_mapping, reward_system,reflection_mgr, step_count)
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
