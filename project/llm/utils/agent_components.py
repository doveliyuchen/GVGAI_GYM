import os
import imageio
from collections import defaultdict
import re
import string
import csv
import string
from io import StringIO

from typing import List, Dict, Tuple, Optional, Union




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


# VGDL SpriteSet + LevelMapping Parser
def parse_vgdl(vgdl_text: Union[str, List[str]]) -> Tuple[set, dict]:
    sprite_names = set()
    level_mapping = {}

    if isinstance(vgdl_text, list):
        vgdl_lines = vgdl_text
        vgdl_string = '\n'.join(vgdl_text)
    else:
        vgdl_string = vgdl_text
        vgdl_lines = vgdl_text.split('\n')

    # === Parse SpriteSet ===
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

    # === Parse LevelMapping ===
    in_level_mapping = False
    for line in vgdl_lines:
        line = line.strip()
        if line.startswith('LevelMapping'):
            in_level_mapping = True
            continue
        if line.startswith('InteractionSet') or line.startswith('TerminationSet'):
            break
        if in_level_mapping and '>' in line:
            parts = line.split('>', 1)
            char = parts[0].strip()
            sprite_list = parts[1].strip().split()
            level_mapping[char] = sprite_list
            sprite_names.update(sprite_list)

    return sprite_names, level_mapping


def convert_state(state, sprite_to_char, debug: bool = False):
    result = []
    for row_idx, row in enumerate(state):
        line = ''
        for col_idx, cell in enumerate(row):
            chosen = '.'
            sprite = cell.strip()
            reason = ''

            if sprite in sprite_to_char:
                chosen = sprite_to_char[sprite]
                reason = 'direct match'
            else:
                for part in sprite.split():
                    if part in sprite_to_char:
                        chosen = sprite_to_char[part]
                        reason = f'fallback to "{part}"'
                        break
            if debug:
                print(f"[{row_idx:02d},{col_idx:02d}] '{sprite}' → '{chosen}' ({reason})")
            line += chosen
        result.append(line)
    return '\n'.join(result)

def get_available_chars(sprite_to_char: Dict[str, str], sprite_names: set) -> list:
    used = set(sprite_to_char.values())

    # Step 1: symbols
    symbols = "@#$%&*"
    symbol_pool = [c for c in symbols if c not in used]

    # Step 2: first-letter of each sprite
    letters = []
    seen_letters = set()
    for name in sorted(sprite_names):
        if not name:
            continue
        first = name.strip()[0].lower()
        if first not in used and first not in seen_letters and first.isalpha():
            letters.append(first)
            seen_letters.add(first)

    # Step 3: fallback: remaining lowercase, uppercase, digits
    fallback_pool = [
        c for c in (string.ascii_lowercase + string.ascii_uppercase + string.digits)
        if c not in used and c not in letters
    ]

    return symbol_pool + letters + fallback_pool
    
def check_unique_mapping(sprite_to_char: dict):
    inverse = {}
    for sprite, char in sprite_to_char.items():
        if char in inverse:
            print(f"[DUPLICATE] Character '{char}' used for both '{inverse[char]}' and '{sprite}'")
        inverse[char] = sprite



def normalize_sprite(sprite: str) -> str:
    """
    Normalize a sprite string:
    - Remove leading/trailing spaces
    - Deduplicate tokens
    - Sort tokens alphabetically
    """
    tokens = sprite.strip().split()
    unique_tokens = sorted(set(tokens))
    return ' '.join(unique_tokens)




def detect_input_type(state_str: str) -> str:
    lines = state_str.strip().splitlines()
    csv_like = sum(',' in line for line in lines)
    ascii_like = sum(all(c in string.printable for c in line) for line in lines)

    if csv_like >= len(lines) // 2:
        return "csv"
    elif ascii_like and csv_like == 0:
        return "ascii"
    return "unknown"


def ascii_to_pseudo_grid(state_str: str) -> str:
    lines = state_str.strip().splitlines()
    csv_grid = [','.join(list(line)) for line in lines]
    return '\n'.join(csv_grid)


def generate_mapping_and_ascii(
    state_str: str,
    vgdl_text: str,
    existing_mapping: Optional[dict] = None,
    debug: bool = False
) -> Tuple[dict, str]:
    # Step 0: Detect and convert input format
    # if detect_input_type(state_str) == "ascii":
    #     state_str = ascii_to_pseudo_grid(state_str)

    # Step 1: Parse VGDL rules
    sprite_names, level_mapping = parse_vgdl(vgdl_text)

    # Step 2: Parse CSV into grid
    reader = csv.reader(StringIO(state_str))
    state_grid = []
    all_leaf_sprites = set()

    for row in reader:
        row_data = []
        for cell in row:
            raw_sprite = cell.strip()
            sprite = normalize_sprite(raw_sprite) if raw_sprite else ''
            row_data.append(sprite)
            if sprite:
                all_leaf_sprites.add(sprite)
        state_grid.append(row_data)

    # Step 3: Build sprite-to-char mapping
    sprite_to_char = dict(existing_mapping) if existing_mapping else {}
    sprite_to_char.setdefault('avatar', 'a')
    if 'background' in all_leaf_sprites:
        sprite_to_char.setdefault('background', '.')
    elif 'floor' in all_leaf_sprites:
        sprite_to_char.setdefault('floor', '.')

    available_chars = get_available_chars(sprite_to_char, all_leaf_sprites)

    for char, sprite_list in level_mapping.items():
        key = ' '.join(sprite_list)
        if key in sprite_to_char:
            continue
        if 'avatar' in sprite_list and sprite_to_char.get('avatar') == 'a':
            continue
        if char not in sprite_to_char.values():
            sprite_to_char[key] = char

    for sprite in sorted(all_leaf_sprites):
        if sprite in sprite_to_char:
            continue
        if 'avatar' in sprite and sprite_to_char.get('avatar') == 'a':
            continue
        if available_chars:
            sprite_to_char[sprite] = available_chars.pop(0)
        else:
            raise ValueError("Ran out of characters to assign.")

    if debug:
        check_unique_mapping(sprite_to_char)

    ascii_level = convert_state(state_grid, sprite_to_char, debug=debug)
    ascii_flipped_y ='\n'.join(reversed(ascii_level.splitlines()))

    return sprite_to_char, ascii_level, ascii_flipped_y


def extract_avatar_position_from_state(
    ascii_lines: Union[str, List[str]],
    sprite_to_char: Dict[str, str],
    flip_vertical: bool = False
) -> Optional[Tuple[int, int]]:
    # 自动处理字符串输入
    if isinstance(ascii_lines, str):
        ascii_lines = ascii_lines.splitlines()

    # 防止误传 list of characters
    if isinstance(ascii_lines, list) and all(isinstance(x, str) and len(x) == 1 for x in ascii_lines):
        ascii_lines = ''.join(ascii_lines).splitlines()

    avatar_char = sprite_to_char.get('avatar', 'a')
    height = len(ascii_lines)

    for y, row in enumerate(ascii_lines):
        if avatar_char in row:
            x = row.index(avatar_char)
            actual_y = (height - 1 - y) if flip_vertical else y
            return ( actual_y, x)

    return None


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
