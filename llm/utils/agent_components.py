import os
try:
    import imageio.v2 as imageio  # imageio >= 2.28 (Python 3.11+)
except ImportError:
    import imageio  # type: ignore[no-redef]
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
        # gymnasium render() returns RGB array directly; legacy gym needs mode kwarg
        try:
            frame = env.render()
        except TypeError:
            frame = env.render(mode='rgb_array')  # type: ignore[call-arg]
        if frame is not None:
            self.frames.append(frame)

    def save(self, game_name):
        gif_name = game_name + '.gif'
        imageio.mimsave(gif_name, self.frames, format='GIF', duration=0.1)


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
) -> Tuple[dict, str, str]:
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


import re
import json
from typing import Tuple

def parse_action_from_response(response: str, action_map: dict):
    import re, json

    # Strip full <think>...</think> blocks
    base_text = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.IGNORECASE).strip()
    # If model emits reasoning without opening <think>, grab only text after </think>
    if not base_text:
        base_text = response
    elif re.search(r"</think>", response, re.IGNORECASE):
        after = re.split(r"</think>", response, flags=re.IGNORECASE)[-1].strip()
        if after:
            base_text = after

    # Prefer last fenced code block; fall back to full text
    code_blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]+\s*\n)?([\s\S]*?)```", base_text)
    text = code_blocks[-1].strip() if code_blocks else base_text

    reverse_action = {v: k for k, v in action_map.items()}
    keyword_to_action = {
        word: aid
        for aid, name in action_map.items()
        for word in name.replace("ACTION_", "").lower().split("_")
    }

    def by_id(n):
        v = int(n)
        return (v, action_map[v]) if v in action_map else None

    def by_kw(w):
        aid = keyword_to_action.get(w.lower())
        return (aid, action_map[aid]) if aid is not None else None

    # 1. Action: N  (last match wins)
    for n in reversed(re.findall(r"\baction\s*[:=\s]*?(\d+)", text, re.IGNORECASE)):
        if r := by_id(n): return r

    # 2. JSON {"action": "word"}
    try:
        m = re.search(r'{\s*"action"\s*:\s*".*?"[\s\S]*?}', text)
        if m:
            w = json.loads(m.group(0)).get("action", "")
            if r := by_kw(w): return r
    except Exception:
        pass

    # 3. Action: word / ACTION_XXX lines
    for block in reversed(re.findall(r"(?:^|\n)\s*(?:Action\s*[:=]|ACTION_[A-Z_]+)([^\n]*)", text, re.IGNORECASE)):
        for act in reversed(re.findall(r"ACTION_[A-Z_]+", block.upper())):
            if act in reverse_action:
                return reverse_action[act], act
        for w in reversed(re.findall(r"\b(left|right|up|down|use|nil|nothing)\b", block.lower())):
            if r := by_kw(w): return r

    # 4. Boxed / symbol numbers
    for n in reversed(re.findall(r"(?:\\boxed{|\(|\[|\*\*|\*)\s*(\d+)\s*(?:}|\)|\]|\*\*|\*)", text)):
        if r := by_id(n): return r

    # 5. Last line direction word
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        for w in reversed(re.findall(r"\b(left|right|up|down|use|nil|nothing)\b", lines[-1].lower())):
            if r := by_kw(w): return r

    # 6. Plain numbers — skip list markers like "1." / "2)" at line start
    for n in reversed(re.findall(r"(?<![.\w])(\d+)(?![.\w])", text)):
        if r := by_id(n): return r

    # 7. Natural language
    for w in reversed(re.findall(r"\b(?:move|go|walk|run|head|step|proceed)?\s*(left|right|up|down|use|nothing|nil)\b", text.lower())):
        if r := by_kw(w): return r

    return 0, action_map[0]



if __name__ == "__main__":
   
    action_map = {
    0: "ACTION_NIL",
    1: "ACTION_USE",
    2: "ACTION_UP",
    3: "ACTION_DOWN",
    4: "ACTION_LEFT",
    5: "ACTION_RIGHT"
    }
    test_responses = [
        '''```Action:1```
Feedback: I am using the fire action to launch a missile upward at column 17 to destroy the alien at (0, 17), as I have a clear path and there are no immediate threats in my current column.
''',
        'ACTION_DOWN (4)',
        '{"action": "use"}',
        'I will move left now.',
        'The action is 3.',
        'Let me **2** this time.',
        'Response: Action:1'
    ]
    for resp in test_responses:
        aid, aname = parse_action_from_response(resp, action_map)
        print(f"Response: {resp}  -->  Parsed Action: {aid} ({aname})")
