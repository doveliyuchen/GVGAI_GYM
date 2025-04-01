import os
import gym
import gym_gvgai as gvgai
import numpy as np
import re
import pygame
import matplotlib.pyplot as plt
import imageio
import time
from collections import defaultdict, Counter
from typing import Iterable, Tuple, Optional, Dict, List
import ast
from llm.client import LLMClient
###########################
# 辅助函数：读取文件、提取映射
###########################

def load_file(filename):
    """读取文件内容，返回字符串"""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def extract_level_mapping(rule_text):
    """
    从 rule.txt 中提取 LevelMapping 部分的映射，
    返回一个字典：sprite_name -> symbol
    例如：'base' -> '0', 'avatar' -> 'A'
    
    示例输入：
        LevelMapping
            r > background rock
            g > background ground
            . > background
            A > background avatar

    期望输出：
        {'rock': 'r', 'ground': 'g', 'background': '.', 'avatar': 'A'}
    """
    lines = rule_text.splitlines()
    mapping_section_found = False
    level_mapping = {}
    for line in lines:
        if "LevelMapping" in line:
            mapping_section_found = True
            continue  # 跳过 LevelMapping 标记行
        if mapping_section_found:
            # 若遇到无缩进的行，则认为 LevelMapping 部分结束
            if line and not line.startswith(" "):
                break
            stripped = line.strip()
            if not stripped:
                continue
            if ">" not in stripped:
                continue
            # 例如： "r > background rock"
            symbol, sprite_str = stripped.split(">", 1)
            symbol = symbol.strip()
            sprite_tokens = sprite_str.strip().split()
            # 如果右侧有多个 token 且第一个是 "background"，则跳过第一个 token
            if len(sprite_tokens) > 1 and sprite_tokens[0] == "background":
                sprite_tokens = sprite_tokens[1:]
            for sprite in sprite_tokens:
                if sprite not in level_mapping:
                    level_mapping[sprite] = symbol
    return level_mapping


def extract_sprite_set_keys(rule_text):
    """
    从 rule.txt 中的 SpriteSet 部分提取所有 sprite 的名称，
    返回一个集合，包含所有非空的 sprite key。
    """
    lines = rule_text.splitlines()
    sprite_keys = set()
    in_sprite_set = False
    sprite_indent = None
    for line in lines:
        if "SpriteSet" in line:
            in_sprite_set = True
            sprite_indent = len(line) - len(line.lstrip())
            continue
        if in_sprite_set:
            current_indent = len(line) - len(line.lstrip())
            # 当缩进回退到与 SpriteSet 同级或更低时退出
            if line.strip() and current_indent <= sprite_indent:
                break
            stripped = line.strip()
            if not stripped or ">" not in stripped:
                continue
            key = stripped.split(">", 1)[0].strip()
            if key:
                sprite_keys.add(key)
    return sprite_keys

def generate_mapping(rule_filename, state):
    """
    综合 rule.txt 与当前 state 中出现的 sprite，
    生成最终的 sprite -> symbol 映射。
    先提取 LevelMapping 和 SpriteSet 中定义的 sprite，
    对于缺失映射的部分，根据 sprite 首字母（冲突时追加数字）自动分配；
    如果 state 中出现 rule.txt 未涉及的 sprite，则从预定义字符中分配。
    """
    rule_text = load_file(rule_filename)
    level_mapping = extract_level_mapping(rule_text)
    sprite_keys = extract_sprite_set_keys(rule_text)
    final_mapping = dict(level_mapping)
    for sprite in sprite_keys:
        if sprite not in final_mapping:
            default = sprite[0].upper()
            new_symbol = default
            counter = 1
            while new_symbol in final_mapping.values():
                new_symbol = f"{default}{counter}"
                counter += 1
            final_mapping[sprite] = new_symbol

    # 检查 state 中可能出现但 rule.txt 未涉及的 sprite
    unique_sprites = set()
    for row in state:
        for cell in row:
            if cell:
                unique_sprites.add(cell)
    available_symbols = list("1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    used_symbols = set(final_mapping.values())
    available_symbols = [sym for sym in available_symbols if sym not in used_symbols]
    for sprite in unique_sprites:
        if sprite not in final_mapping:
            if available_symbols:
                final_mapping[sprite] = available_symbols.pop(0)
            else:
                final_mapping[sprite] = "?"
    return final_mapping

def convert_state_to_string(state, mapping):
    """
    根据 mapping 将 state 中的 sprite 名称转换为符号表示，
    空 cell 用 '.' 表示
    """
    converted = []
    for row in state:
        row_str = "".join(mapping.get(cell, ".") if cell else "." for cell in row)
        converted.append(row_str)
    return converted

###########################
# 以下为你已有的部分代码（部分函数未修改）
###########################

def create_directory(base_dir='imgs'):
    """
    Create a directory, and generate a new directory name (e.g., imgs_1, imgs_2, etc.) if the directory already exists.
    """
    if os.path.exists(base_dir):
        index = 1
        while True:
            new_dir = f"{base_dir}_{index}"
            if not os.path.exists(new_dir):
                base_dir = new_dir
                break
            index += 1
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
    plt.figure(3)
    plt.clf()
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
        if reflection:
            self.history.append(reflection)
            if len(self.history) > self.max_history:
                self.history.pop(0)
    def get_formatted_history(self) -> str:
        return "\n".join(f"[History reflection{i + 1}] {r}"
                         for i, r in enumerate(self.history))

def parse_vgdl_level(vgdl_level):
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
    return np.array([list(row) for row in padded_level]), avatar_pos

class RewardSystem:
    def __init__(self):
        self.action_history = []
        self.reward_history = []
        self.action_efficacy = defaultdict(list)
        self.consecutive_zero_threshold = 3
        self.total_reward = 0.0
    def update(self, action: int, reward: float):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.action_efficacy[action].append(reward)
        self.total_reward += reward
    def generate_guidance(self) -> str:
        return "Placeholder guidance..."
    def get_zero_streak(self) -> int:
        return next(
            (i for i, r in enumerate(reversed(self.reward_history)) if r != 0),
            len(self.reward_history))
    

def build_mapping_text(sprite_mapping: Optional[Dict[str, str]], game_state: List[str]) -> str:
    """
    将 sprite 映射字典转换为文本说明，格式为：
      symbol -> sprite
    只包括出现在当前 game_state 中的映射。

    参数:
      sprite_mapping: 可选的字典，例如 {'alienBlue': 'A3', 'base': '0', ...}
      game_state: 字符串列表，每个元素代表一行的状态（已由 convert_state_to_string 得到）

    返回:
      每行形如 "A3 -> alienBlue" 的说明字符串，如果未提供 sprite_mapping 则返回空字符串。
    """
    if sprite_mapping is None:
        return ""
    mapping_lines = []
    # 对于 mapping 中的每个项，检查 symbol 是否出现在任意一行中
    for sprite, symbol in sprite_mapping.items():
        if any(symbol in row for row in game_state):
            mapping_lines.append(f"{symbol} -> {sprite}")
    return "\n".join(mapping_lines)


def build_enhanced_prompt(vgdl_rules: str,
                          state: str,
                          last_state: str,
                          action_map: dict,
                          reward_system: RewardSystem,
                          reflection_mgr: ReflectionManager,
                          sprite_mapping: Optional[Dict[str, str]],
                          current_image_path: Optional[str] = None,
                          last_image_path: Optional[str] = None,
                          reflection = False, reward = False ) -> str:
    last_action = reward_system.action_history[-1] if reward_system.action_history else None
    last_reward = reward_system.reward_history[-1] if reward_system.reward_history else None
    last_action_desc = action_map.get(last_action, "") if last_action is not None else ""
    last_reward_desc = action_map.get(last_reward, "") if last_reward is not None else ""
    mapping_text = build_mapping_text(sprite_mapping, state) if sprite_mapping else ""
    formating = f'''
    You are controlling avatar A, try to win the game with *meaningful action*.
    Goal: Try to interact with the game by analyzing the game state and learn to play and win it. 
    Respond in this format with only *ONE* action with a sentence of analysis of your current position:
    ``` Action:<action number> ``` 
    '''
    reflection_format =  ""
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
    {state}
   
    === Last Action ===
    {last_action}

    === Representation Mapping ===
    {mapping_text}

    === Available Actions ===
    {chr(10).join(f'{k}: {v}' for k, v in action_map.items())}
    '''
    reward_prompt = f'''
         === Last Reward ===
         {last_reward} ({last_reward_desc})
        ''' if reward else ''
    guidance = "\nState guidance placeholder...\n"
    return f"{formating}{reflection_format}{base}{reward_prompt}{reflection_section}\n{guidance}\n"

def parse_action_from_response(response: str, action_map: dict) -> Tuple[int, str]:
    reverse_action_dict = {v: k for k, v in action_map.items()}
    keyword_to_action = {}
    for aid, aname in action_map.items():
        for word in aname.replace("ACTION_", "").lower().split("_"):
            keyword_to_action[word] = aid
    action_stmt_match = re.findall(r"\baction\s*[:=~\-]?\s*(\d+|ACTION_[A-Z_]+)", response, re.IGNORECASE)
    for val in reversed(action_stmt_match):
        if val.isdigit():
            num = int(val)
            if num in action_map:
                return num, action_map[num]
        elif val.upper() in reverse_action_dict:
            return reverse_action_dict[val.upper()], val.upper()
    nil_match = re.findall(r"ACTION[_\\]*NIL", response, re.IGNORECASE)
    if nil_match:
        return 0, action_map[0]
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
    action_words = re.findall(r"ACTION_[A-Z_]+", response)
    for act_name in reversed(action_words):
        if act_name in reverse_action_dict:
            return reverse_action_dict[act_name], act_name
    full_pairs = re.findall(r"(\d+)\s*[:=]\s*(ACTION_[A-Z_]+)", response)
    for num, act_name in reversed(full_pairs):
        if act_name in reverse_action_dict:
            return reverse_action_dict[act_name], act_name
    number_matches = re.findall(r"\b(\d+)\b", response)
    for num in reversed(number_matches):
        val = int(num)
        if val in action_map:
            return val, action_map[val]
    smart_matches = re.findall(r"\b(?:move|go|walk|run|head|step|proceed)[\s_]*(left|right|up|down|use|nothing|nil)\b", response.lower())
    if smart_matches:
        keyword = smart_matches[-1]
        if keyword in keyword_to_action:
            return keyword_to_action[keyword], action_map[keyword_to_action[keyword]]
    keyword_matches = re.findall(r"\b(" + "|".join(re.escape(k) for k in keyword_to_action) + r")\b", response.lower())
    if keyword_matches:
        keyword = keyword_matches[-1]
        return keyword_to_action[keyword], action_map[keyword_to_action[keyword]]
    return 0, action_map[0]

def query_llm(llm_client,
              vgdl_rules: str,
              current_state: str,
              last_state: str,
              action_map: dict,
              reward_system: RewardSystem,
              reflection_mgr: ReflectionManager,
              step: int,
              current_image_path: Optional[str] = None,
              last_image_path: Optional[str] = None,
              sprite_mapping: Optional[Dict[str, str]] = None,
              reflection = False,
              reward = False) -> Tuple[int, str]:
    prompt = build_enhanced_prompt(vgdl_rules, current_state, last_state, action_map,
                                   reward_system, reflection_mgr, current_image_path, last_image_path,sprite_mapping, reflection, reward)
    try:
        response = llm_client.query(prompt, image_path=current_image_path)
        print("====response====")
        print(response)
        reflection_match = re.search(r"Reflection:\s*```(.*?)```", response, re.DOTALL)
        action, action_name = parse_action_from_response(response, action_map)
        reflection_text = reflection_match.group(1).strip() if reflection_match else ""
        print(f"\n=== Step {step} ===")
        print(f"Selected Action: {action} ({action_name})")
        if reflection_text:
            print(f"Strategy Reflection: {reflection_text[:700]}...")
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

###########################
# 主程序入口：整合环境、规则、映射与状态转换
###########################

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(os.path.dirname(current_path), "gym_gvgai", "envs", "games")
    llm_list = {"deepseek":["deepseek"]}

    for game in os.listdir(full_path):
        game="aliens_v0"
        env_name = "gvgai-"+game[:-3]+"-lvl0-v0"
        env = gvgai.make(env_name)
        state = env.reset()
        done = False

        # 获取 VGDL 规则与关卡布局文件
        game_name = env.spec.id.replace("gvgai-", "").split("-")[0] + "_v0"
        game_dir = os.path.join(os.path.dirname(current_path), "gym_gvgai", "envs", "games", game_name)
        vgdl_rule_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                               if f.endswith(".txt") and "lvl" not in f), None)
        level_layout_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                                  if f.endswith(".txt") and "lvl" in f), None)
        with open(vgdl_rule_file, "r") as f:
            vgdl_rules_lines = f.read().splitlines()
        with open(level_layout_file, "r") as f:
            level_layout = f.read()

        vgdl_grid, avatar_pos = parse_vgdl_level(level_layout)
        h, w = vgdl_grid.shape

        try:
            action_mapping = {i: env.unwrapped.get_action_meanings()[i] for i in range(env.action_space.n)}
        except AttributeError:
            action_mapping = {i: f"Action {i}" for i in range(env.action_space.n)}

        # 初始化 llm 与其他管理器
        for model in llm_list["deepseek"]:
            llm, = llm_list.keys()
            llm_client = LLMClient(llm)
            state = env.reset()
            done = False
            reflection_mgr = ReflectionManager()
            reward_system = RewardSystem()
            total_reward = 0
            step_count = 0
            info = None
            img = show_state_gif()
            last_state = None
            game_state = vgdl_grid  # 初始状态采用关卡布局
            last_state_img = None
            game_state_img = None
            llm_dir = re.search(r"(.*?):", model)
            if llm_dir:
                llm_dir = llm_dir.group(1).strip()
            dir_path = create_directory(f"img_{llm_dir}/"+game_name)
            
            # 这里利用 rule 文件与初始 state 生成 sprite 映射
            sprite_mapping = generate_mapping(vgdl_rule_file, game_state)
            print("生成的 sprite 映射：")
            for sprite, sym in sprite_mapping.items():
                print(f"{sprite}: {sym}")
            
            try:
                while not done:
                    # 将当前 state（通常为 info["ascii"] 格式）转换为二维列表
                    if "ascii" in info if info else False:
                        raw_state = [row.split(',') for row in info["ascii"].splitlines()]
                        # 检查 raw_state 中所有 cell 去除空白后长度是否都为1
                        if all(len(cell.strip()) == 1 for row in raw_state for cell in row):
                            # 如果每个 cell 长度都为1，就不需要转换
                            game_state = raw_state
                        else:
                            # 否则执行转换，比如调用 convert_state_to_string 函数
                            game_state = convert_state_to_string(raw_state, sprite_mapping)

                    print(game_state)
                    action, reflection = query_llm(llm_client, "\n".join(vgdl_rules_lines), 
                                                     game_state, 
                                                     last_state,
                                                     action_mapping, reward_system,
                                                     reflection_mgr, step_count,
                                                     reflection=False, sprite_mapping=sprite_mapping)
                    next_state, reward, done, info = env.step(action)
                    reward_system.update(action, reward)
                    last_state = game_state
                    game_state = [row.split(',') for row in info["ascii"].splitlines()]
                    total_reward += reward
                    print(f"Received Reward: {reward}")
                    img(env)
                    winner = info.get('winner', "None")
                    step_count += 1
            finally:
                env.close()
                try:
                    img.save(dir_path+"_"+llm)
                    with open(f"game_logs_text_{model}.txt", mode="a") as f:
                        f.write(f"game_name: {game_name}, step_count: {step_count}, winner: {winner}, api: {llm}, total reward: {total_reward}\n")
                except Exception as e:
                    print("无法保存图片：", e)
                with open(f"game_logs_text_{model}.txt", mode="a") as f:
                    f.write(f"game_name: {game_name}, step_count: {step_count}, winner: {winner}, api: {llm}, total reward: {total_reward}\n")
                generate_report(reward_system, step_count, dir_path+"_"+llm)
