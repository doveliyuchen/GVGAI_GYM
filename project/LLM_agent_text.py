import os
import imageio
from collections import defaultdict
import re
import string

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
        return self._performance_summary()

    def get_zero_streak(self) -> int:
        return next(
            (i for i, r in enumerate(reversed(self.reward_history)) if r != 0),
            len(self.reward_history))


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
        return "\n".join(f"[History reflection{i + 1}] {r}" for i, r in enumerate(self.history))


class show_state_gif:
    def __init__(self):
        self.frames = []

    def __call__(self, env):
        self.frames.append(env.render(mode='rgb_array'))

    def save(self, game_name):
        gif_name = game_name + '.gif'
        imageio.mimsave(gif_name, self.frames, 'GIF', duration=0.1)


def create_directory(base_dir='imgs'):
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


# VGDL State Parsing

def parse_vgdl(vgdl_text):
    sprite_names = set()
    level_mapping = defaultdict(list)

    if isinstance(vgdl_text, list):
        vgdl_lines = vgdl_text
        vgdl_string = '\n'.join(vgdl_text)
    else:
        vgdl_string = vgdl_text
        vgdl_lines = vgdl_text.split('\n')

    sprite_section = re.search(r"SpriteSet(.*?)LevelMapping", vgdl_string, re.DOTALL)
    if sprite_section:
        lines = sprite_section.group(1).split('\n')
        parent_stack = []
        for line in lines:
            line = line.rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            while parent_stack and parent_stack[-1][0] >= indent:
                parent_stack.pop()
            parts = line.strip().split('>')
            name = parts[0].strip()
            if parent_stack:
                full_name = parent_stack[-1][1] + "." + name
            else:
                full_name = name
            sprite_names.add(full_name)
            parent_stack.append((indent, full_name))

    in_level_mapping = False
    for line in vgdl_lines:
        if 'LevelMapping' in line:
            in_level_mapping = True
            continue
        if 'TerminationSet' in line:
            break
        if in_level_mapping:
            parts = line.strip().split('>')
            if len(parts) == 2:
                char = parts[0].strip()
                sprites = parts[1].strip().split()
                level_mapping[char].extend(sprites)

    return sprite_names, dict(level_mapping)


def convert_state(state, sprite_to_char):
    result = []
    for row in state:
        line = ''
        for cell in row:
            chosen = '.'
            if cell:
                for sprite in cell.split():
                    if sprite in sprite_to_char:
                        chosen = sprite_to_char[sprite]
                        break
            line += chosen
        result.append(line)
    return '\n'.join(result)


def generate_mapping_and_ascii(state, vgdl_text):
    sprite_names, level_mapping = parse_vgdl(vgdl_text)

    sprite_to_char = {}
    for char, sprite_list in level_mapping.items():
        for sprite in sprite_list:
            sprite_to_char[sprite] = char

    sprite_to_char['background'] = '.'

    all_leaf_sprites = set()
    for full_name in sprite_names:
        leaf = full_name.split('.')[-1]
        all_leaf_sprites.add(leaf)

    used_chars = set(sprite_to_char.values())
    available_chars = [c for c in (string.digits + string.ascii_uppercase + "@#$%&*") if c not in used_chars]
    for sprite in sorted(all_leaf_sprites):
        if sprite not in sprite_to_char and available_chars:
            sprite_to_char[sprite] = available_chars.pop(0)

    ascii_level = convert_state(state, sprite_to_char)
    return sprite_to_char, ascii_level


def extract_avatar_position_from_state(ascii_lines, sprite_to_char):
    avatar_chars = [char for sprite, char in sprite_to_char.items() if 'avatar' in sprite.lower() or sprite.lower() == 'a']
    if not avatar_chars:
        return None

    avatar_char_set = set(avatar_chars)
    for y, row in enumerate(ascii_lines):
        for x, ch in enumerate(row):
            if ch in avatar_char_set:
                return (y, x)
    return None
