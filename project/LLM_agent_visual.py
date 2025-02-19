# llm_agent_vision_integrated.py
import os
import re
import base64
import gym
import gym_gvgai
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
# 从旧文件导入核心功能
from LLM_agent import (
    EnhancedRewardSystem,
    ReflectionManager,
    parse_vgdl_level,
    vgdl_to_image,
    LLMClient
)


class VisionClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.usage_log = []
        self.clent = LLMClient("openai")

    def get_vision_response(self, messages: list) -> Tuple[str, dict]:
        """获取视觉模型的响应"""
        response = self.clent.query(messages)
        return response


def capture_game_frame(env, step: int, info: str, vgdl_representation=None) -> str:
    """捕获游戏画面并返回base64编码"""
    plt.figure(figsize=(8, 8))

    try:
        img_array = env.render(mode='rgb_array')
        plt.imshow(img_array)
    except Exception as render_error:
        if vgdl_representation:
            plt.imshow(vgdl_to_image(vgdl_representation))
        else:
            plt.text(0.5, 0.5, "Render Unavailable",
                     ha='center', va='center', fontsize=16)

    # 保存到内存缓冲区
    img_buffer = BytesIO()
    plt.title(f"Step: {step} | {info}")
    plt.axis('off')
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    plt.close()

    # 转换为base64
    img_buffer.seek(0)
    return base64.b64encode(img_buffer.getvalue()).decode('utf-8')


def build_vision_message(vgdl_rules: str, game_state: str, image_b64: str) -> list:
    """构建视觉API请求消息"""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""你是一个专业游戏AI，请根据以下游戏规则和当前状态选择最佳动作：

                    === 游戏规则 ===
                    {vgdl_rules}

                    === 当前状态 ===
                    {game_state}

                    请按照以下格式响应：
                    Action: <动作编号>
                    Analysis: <策略分析>
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "low"  # 可选：low/medium/high
                    }
                }
            ]
        }
    ]


def parse_response(response_text: str) -> Tuple[int, str]:
    """解析模型响应"""
    action_match = re.search(r"Action:\s*(\d+)", response_text)
    analysis_match = re.search(r"Analysis:\s*(.+?)(?=\nAction|\Z)", response_text, re.DOTALL)

    action = int(action_match.group(1)) if action_match else 0
    analysis = analysis_match.group(1).strip() if analysis_match else ""

    return action, analysis


if __name__ == "__main__":
    # 初始化环境
    env = gym.make("gvgai-angelsdemons-lvl0-v0")
    state = env.reset()

    # 初始化模块
    vision_client = VisionClient()
    reward_system = EnhancedRewardSystem(env.action_space.n)
    reflection_mgr = ReflectionManager()

    # 加载游戏规则
    game_name = env.spec.id.replace("gvgai-", "").split("-")[0] + "_v0"
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    game_dir = os.path.join(current_path, "gym_gvgai", "envs", "games", game_name)

    vgdl_rule_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                           if f.endswith(".txt") and "lvl" not in f), None)
    level_layout_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                              if f.endswith(".txt") and "lvl" in f), None)

    if not vgdl_rule_file or not level_layout_file:
        raise FileNotFoundError("No file detected")

    with open(vgdl_rule_file, "r") as f:
        vgdl_rules = f.read()

    try:
        step = 0
        frame_interval = 3  # 每3步捕获一次画面
        max_steps = 50

        while step < max_steps:
            # 获取游戏状态
            try:
                raw_state = env.unwrapped.get_observation()
                if isinstance(raw_state, tuple):
                    game_state = f"Avatar position: {raw_state[1]}\nGrid:\n" + "\n".join(
                        ["".join(row) for row in raw_state[0]])
                else:
                    game_state = str(raw_state)
            except AttributeError:
                grid, pos = parse_vgdl_level(os.path.join(game_dir, f"{game_name}_lvl0.txt"))
                game_state = f"Avatar starts at: {pos}\nGrid:\n" + "\n".join(["".join(row) for row in grid])

            # 定期捕获画面
            image_b64 = None
            if step % frame_interval == 0:
                image_b64 = capture_game_frame(env, step, f"Reward: {reward_system.total_reward}", game_state)

            # 构建请求消息
            if image_b64:
                messages = build_vision_message(vgdl_rules, game_state, image_b64)
            else:
                messages = [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"当前游戏状态（无画面）：\n{game_state}"
                    }]
                }]

            # 获取模型响应
            response, error = vision_client.get_vision_response(messages)

            if error:
                print(f"API错误：{error}")
                action = 0
                analysis = "API请求失败"
            else:
                action, analysis = parse_response(response)
                action = min(max(action, 0), env.action_space.n - 1)  # 确保动作在有效范围内

            # 执行动作
            next_state, reward, done, _ = env.step(action)
            reward_system.update(action, reward)
            reflection_mgr.add_reflection(analysis)

            print(f"\nStep {step} - 动作: {action}")
            print(f"分析: {analysis[:200]}...")
            print(f"奖励: {reward} | 累计奖励: {reward_system.total_reward}")

            step += 1
            if done:
                break

    finally:
        env.close()
        print("\n运行统计：")
        print(f"总步数: {step}")
        print(f"最终奖励: {reward_system.total_reward}")
        print(f"API使用统计: {vision_client.usage_log[-1]}")
