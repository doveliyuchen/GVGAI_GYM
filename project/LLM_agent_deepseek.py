import os
import gym
import gym_gvgai as gvgai
import numpy as np
import re
import pygame
import matplotlib.pyplot as plt
from llm.client import LLMClient
from collections import defaultdict, Counter
from typing import Iterable, Tuple, Optional
import imageio
import time
import time

def create_directory(base_dir='imgs'):
    """
    Create a directory, and generate a new directory name (e.g., imgs_1, imgs_2, etc.) if the directory already exists.

    Args:
        base_dir (str): The base directory name, default is 'imgs'.

    Returns:
        str: The final created directory path.
    """
    if os.path.exists(base_dir):
        # 如果目录已存在，生成新的目录名
        index = 1
        while True:
            new_dir = f"{base_dir}_{index}"
            if not os.path.exists(new_dir):
                base_dir = new_dir
                break
            index += 1

    # 创建目录
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


class show_state_gif():
    def __init__(self):
        self.frames = []
    def __call__(self, env):
        self.frames.append(env.render(mode='rgb_array'))

    def save(self, game_name):
        gif_name = game_name + '.gif'
        imageio.mimsave(gif_name, self.frames, 'GIF', duration = 0.1)


def show_state(env, step, name, info, directory, vgdl_representation=None):
    """
    Render the environment state and save it as an image file.

    Args:
        env: Environment object used to render the image.
        step (int): Current step.
        name (str): Image name.
        info (str): Additional information to display in the title.
        directory (str): Target directory to save the image.
        vgdl_representation: Optional parameter to override the rendering logic.

    Returns:
        str: The file path of the saved image; returns None if saving fails.
    """
    # Render the image
    plt.figure(3)
    plt.clf()
    img = env.render(mode='rgb_array')
    plt.imshow(img)

    img = env.render(mode='rgb_array')
    plt.imshow(img)



    plt.title(f"{name} | Step: {step} {info}")
    plt.axis("off")


    path = f'{directory}/{name}_{len(os.listdir(directory)) + 1}.png'
    plt.savefig(path)


    return path if os.path.exists(path) else None



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
    # 如果输入是字符串，则按换行符拆分成多行（换行符会被删除）
    if isinstance(vgdl_level, str):
        vgdl_level = vgdl_level.splitlines()

    # 如果输入是字符串，则按换行符拆分成多行（换行符会被删除）
    if isinstance(vgdl_level, str):
        vgdl_level = vgdl_level.splitlines()

    max_width = max(len(row) for row in vgdl_level)
    padded_level = [row.ljust(max_width, ".") for row in vgdl_level]

    avatar_pos = None
    for y, row in enumerate(padded_level):
        if 'A' in row:
            x = row.index('A')
            avatar_pos = (x, y)
            break

    # 每一行被转换为一个字符列表，从而构成一个二维矩阵
    return np.array([list(row) for row in padded_level]), avatar_pos
    # 每一行被转换为一个字符列表，从而构成一个二维矩阵
    return np.array([list(row) for row in padded_level]), avatar_pos




class RewardSystem:
    def __init__(self):
        self.action_history = []
        self.reward_history = []
        self.action_efficacy = defaultdict(list)
        self.consecutive_zero_threshold = 3
        # self.window_size = window_size
        self.total_reward = 0.0

    def update(self, action: int, reward: float):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.action_efficacy[action].append(reward)
        self.total_reward += reward



    def generate_guidance(self) -> str:
        return self._performance_summary()

    def get_zero_streak(self) -> int:
        return next(
            (i for i, r in enumerate(reversed(self.reward_history)) if r != 0),
            len(self.reward_history))




def build_enhanced_prompt(vgdl_rules: str,
                          state: str,
                          last_state: str,
                          action_map: dict,
                          reward_system: RewardSystem,
                          reflection_mgr: ReflectionManager,
                          current_image_path: Optional[str] = None,
                          last_image_path: Optional[str] = None,
                          reflection = False, reward = False ) -> str:
    """Build prompt with layout and reflection history"""

    last_action = None
    last_reward = None
    if reward_system.action_history:
        last_action = reward_system.action_history[-1]
        last_reward = reward_system.reward_history[-1]
    last_action_desc = action_map.get(last_action, "None") if last_action is not None else "None"
    last_reward_desc = action_map.get(last_reward, "None") if last_reward is not None else "None"

    formating = f'''
    You are controlling avatar A, try to win the game with *meaningful action*.\n
    Goal: Try to interact with the game by analyzing the game state and learn to play and win it. 
    Respond in this format with only *ONE* action with a sentence of analysis of your current position:\n
    ``` Action:<action number> ``` \n'''
    # You cannot ask questions after your chosen action.

    reflection_format =  ""
    guidance = ""
    reflection_section = ""
    if reflection:
        reflection_format = ''' Reflection: ```<your strategy reflection>```  '''
        if reflection_mgr.history:
            reflection_section = f"\n=== Reflection History ===\n{reflection_mgr.get_formatted_history()}"

    base =f'''
    === Game Rules ===
    {vgdl_rules}


    === Last State ===
    {last_state}

    === Current State ===
    You are "avatar"
    {state}
   
    === Last Action ===
    {last_action}

    === Available Actions ===
    {chr(10).join(f'{k}: {v}' for k, v in action_map.items())}
    '''

    if reward:
        reward_prompt = f'''
         === Last Reward ===
         {last_reward} ({last_reward_desc})
        '''
    else:
        reward_prompt = ''

#     guidance = f'''
# State Awareness: Recognize the differences between states and identify the optimal action to transition to a desired state. Consider this process operates within a fully defined Markov framework.
# Action Consistency: Be mindful that your current decision may negate the effects of the previous action. Aim to maintain consistency and avoid contradictory moves.
# Meaningful Decisions: Ensure that your actions are purposeful. For instance, moving against a wall is unproductive and should be avoided.
# '''

    return f"{formating}{reflection_format}{base}{reward_prompt}{reflection_section}\n{guidance}\n"

# ====action match======


# 修改为优先匹配 "action" + 任意符号 + 动作（数字 或 ACTION_XXX），而不是 "ANSWER:"

def parse_action_from_response(response: str, action_map: dict) -> Tuple[int, str]:
    reverse_action_dict = {v: k for k, v in action_map.items()}

    keyword_to_action = {}
    for aid, aname in action_map.items():
        for word in aname.replace("ACTION_", "").lower().split("_"):
            keyword_to_action[word] = aid

    # 1. 优先匹配 action + 任意符号 + 动作值（数字或ACTION_XXX）
    action_stmt_match = re.findall(r"\baction\s*[:=~\-]?\s*(\d+|ACTION_[A-Z_]+)", response, re.IGNORECASE)
    for val in reversed(action_stmt_match):
        if val.isdigit():
            num = int(val)
            if num in action_map:
                return num, action_map[num]
        elif val.upper() in reverse_action_dict:
            return reverse_action_dict[val.upper()], val.upper()

    # 2. 明确表达 ACTION_NIL
    nil_match = re.findall(r"ACTION[_\\]*NIL", response, re.IGNORECASE)
    if nil_match:
        return 0, action_map[0]

    # 3. 代码块内容（数字、ACTION、关键词）
    code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response)
    for block in reversed(code_blocks):
        num_match = re.findall(r"\b(\d+)\b", block)
        for val in reversed(num_match):
            num = int(val)
            if num in action_map:
                return num, action_map[num]

        action_words = re.findall(r"ACTION_[A-Z_]+", block)
        for act_name in reversed(action_words):
            if act_name in reverse_action_dict:
                return reverse_action_dict[act_name], act_name

        word_matches = re.findall(r"\b(" + "|".join(re.escape(k) for k in keyword_to_action) + r")\b", block.lower())
        for word in reversed(word_matches):
            aid = keyword_to_action[word]
            return aid, action_map[aid]

    # 4. 全局 ACTION_XXX
    action_words = re.findall(r"ACTION_[A-Z_]+", response)
    for act_name in reversed(action_words):
        if act_name in reverse_action_dict:
            return reverse_action_dict[act_name], act_name

    # 5. 键值对格式 "3: ACTION_DOWN"
    full_pairs = re.findall(r"(\d+)\s*[:=]\s*(ACTION_[A-Z_]+)", response)
    for num, act_name in reversed(full_pairs):
        if act_name in reverse_action_dict:
            return reverse_action_dict[act_name], act_name

    # 6. standalone 数字
    number_matches = re.findall(r"\b(\d+)\b", response)
    for num in reversed(number_matches):
        val = int(num)
        if val in action_map:
            return val, action_map[val]

    # 7. move/go + direction
    smart_matches = re.findall(r"\b(?:move|go|walk|run|head|step|proceed)[\s_]*(left|right|up|down|use|nothing|nil)\b", response.lower())
    if smart_matches:
        keyword = smart_matches[-1]
        if keyword in keyword_to_action:
            return keyword_to_action[keyword], action_map[keyword_to_action[keyword]]

    # 8. keyword fallback
    keyword_matches = re.findall(r"\b(" + "|".join(re.escape(k) for k in keyword_to_action) + r")\b", response.lower())
    if keyword_matches:
        keyword = keyword_matches[-1]
        return keyword_to_action[keyword], action_map[keyword_to_action[keyword]]

    return 0, action_map[0]



def query_llm(llm_client: LLMClient,
              vgdl_rules: str,
              current_state: str,
              last_state: str,
              action_map: dict,
              reward_system: RewardSystem,
              reflection_mgr: ReflectionManager,
              step: int,
              current_image_path: Optional[str] = None,
              last_image_path: Optional[str] = None,
              reflection = False,
              reward = False) -> Tuple[int, str]:
    prompt = build_enhanced_prompt(vgdl_rules, current_state, last_state, action_map,
                                   reward_system, reflection_mgr, current_image_path, last_image_path,reflection,reward)

    try:
        # time.sleep(1)
        # print("====prompt====")
        # print(prompt)
        response = llm_client.query(prompt, image_path=current_image_path)
        print("====response====")
        print(response)

        reflection_match = re.search(r"Reflection:\s*```(.*?)```", response, re.DOTALL)


        # 初始化 action 变量
        action = 0  # 默认值

        action, action_name = parse_action_from_response(response, action_map)
  
        reflection = reflection_match.group(1).strip() if reflection_match else ""

        # 查找 Reflection（策略思考）
        reflection_match = re.search(r"Reflection:\s*```(.*?)```", response, re.DOTALL)
        reflection_text = reflection_match.group(1).strip() if reflection_match else ""

        # 打印调试信息
        print(f"\n=== Step {step} ===")
        print(f"Selected Action: {action} ({action_name})")
        if reflection:
            print(f"Strategy Reflection: {reflection[:700]}...")

        return action, reflection_text

    except Exception as e:
        print(f"LLM query error: {str(e)}")
        return 0, " "
        
def generate_report(system: RewardSystem, step: int, dir) -> str:
    print(f"\n=== Game analysis ===")
    print(f"Total steps: {step}")
    print(f"Total reward: {system.total_reward}")
    print(f"Zero Streak: {system.get_zero_streak()}")

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(system.reward_history)
    plt.title("Reward trend")

    plt.subplot(122)
    action_dist = Counter(system.action_history)
    plt.bar(action_dist.keys(), action_dist.values())
    plt.title("Action distribution")
    plt.savefig(dir+"game_analysis.png")


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(os.path.dirname(current_path), "gym_gvgai", "envs", "games")
    llm_list= ['deepseek']
    # model = 'deepseek-r1:14b'

    for game in os.listdir(full_path)[103:]:
        print(game)
        env_name = "gvgai-"+game[:-3]+"-lvl0-v0"

        env = gvgai.make(env_name)
        state = env.reset()
        done = False

        game_dir = os.path.join(os.path.dirname(current_path), "gym_gvgai", "envs", "games", game)

        vgdl_rule_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                               if f.endswith(".txt") and "lvl" not in f), None)
        level_layout_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                                  if f.endswith(".txt") and "lvl" in f), None)



        with open(vgdl_rule_file, "r") as f:
            vgdl_rules = f.read().splitlines()
        with open(level_layout_file, "r") as f:
            level_layout = f.read()

        vgdl_grid, avatar_pos = parse_vgdl_level(level_layout)
        h, w = vgdl_grid.shape

        available_actions = list(range(env.action_space.n))
        try:
            action_mapping = {i: env.unwrapped.get_action_meanings()[i] for i in available_actions}
        except AttributeError:
            action_mapping = {i: f"Action {i}" for i in available_actions}


        for llm in llm_list:

            llm_client = LLMClient(llm)
            state = env.reset()
            done = False
            reflection_mgr = ReflectionManager()
            reward_system = RewardSystem()
            total_reward = 0
            step_count = 0
            info = None
            image_path = None
            img = show_state_gif()
            last_state = None
            game_state = vgdl_grid
            last_state_img = None
            game_state_img = None
            dir = create_directory("img_text3/"+game)

            try:

                while not done:

                    action, reflection = query_llm(llm_client, vgdl_rules,game_state, last_state, action_mapping, reward_system,
                                                   reflection_mgr, step_count, reflection = False)
                    next_state, reward, done, info = env.step(action)
                    reward_system.update(action, reward)
                    last_state = game_state
                    game_state = info["ascii"]


                    total_reward += reward
                    print(f"Received Reward: {reward}")

                    img(env)
                    winner = info['winner']
                    step_count += 1


            finally:

                env.close()
                try:
                    img.save(dir+"_"+llm)
                except:
                    print("cannot save")
                with open(f"game_logs_text_{llm}.txt", mode="a") as f:
                    f.write(f"game_name: {game}, step_count: {step_count}, winner: {winner}, api: {llm}\n")
                generate_report(reward_system, step_count,dir+"_"+llm)


    

