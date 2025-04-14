import os
import gym
import gym_gvgai as gvgai
import re
from dotenv import load_dotenv
import numpy
import string
from collections import defaultdict
from typing import Union

load_dotenv('.env')
    
def get_available_games(full_name=True):
    """Get all available games in the gym_gvgai environment using registry."""
    try:
        # Get all environments that start with 'gvgai'
        gvgai_envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai')]
        
        if full_name: return gvgai_envs

        # Extract unique game names from environment IDs
        game_names = set()
        for env_id in gvgai_envs:
            # Parse the game name from the environment ID (format: gvgai-{game}-lvl{level}-v{version})
            parts = env_id.split('-')
            if len(parts) >= 3:
                game_name = parts[1]
                game_names.add(game_name)
        
        return sorted(game_names)
            
    except Exception as e:
        print(f"Error listing games: {e}")
        return []
    
def extract_game_dir(env_id, gvgai_game_dir):

    # gvgai-x-racer-lvl0-v0 → x-racer_v0
    match = re.match(r"gvgai-(.+)-lvl\d+-v\d+", env_id)
    if match:
        game_name = match.group(1)
        game_dir = os.path.join(gvgai_game_dir, game_name + "_v0")
        return game_dir
    else:
        raise ValueError(f"Invalid environment ID format: {env_id}")
    
# returns env object, vgdl_rules, level_layout
def get_game_env(env_id):

    gvgai_game_dir = os.getenv("GVGAI_GAME_DIR")
    env = gvgai.make(env_id)
    game_dir = extract_game_dir(env_id, gvgai_game_dir)

    if not os.path.exists(game_dir):
        raise FileNotFoundError(f"Game directory not found: {game_dir}")

    # Extract game name and level index
    match = re.match(r"gvgai-(.*)-lvl(\d+)-v\d+", env_id)
    if not match:
        raise ValueError(f"env_id format incorrect: {env_id}")
    game_name, level_idx = match.groups()
    level_file_expected = f"{game_name}_lvl{level_idx}.txt"
    rule_file_expected = f"{game_name}.txt"

    vgdl_rule_file = os.path.join(game_dir, rule_file_expected)
    level_layout_file = os.path.join(game_dir, level_file_expected)

    if not os.path.exists(vgdl_rule_file):
        raise FileNotFoundError(f"VGDL rule file not found: {vgdl_rule_file}")
    if not os.path.exists(level_layout_file):
        raise FileNotFoundError(f"Level layout file not found: {level_layout_file}")

    try:
        with open(vgdl_rule_file, "r") as f:
            vgdl_rules = f.read()
        with open(level_layout_file, "r") as f:
            level_layout = f.read()
    except PermissionError:
        raise PermissionError("Permission denied when reading game files")
    except UnicodeDecodeError:
        raise UnicodeDecodeError("Error decoding game files - check file encoding")

    return env, vgdl_rules, level_layout



def parse_vgdl(vgdl_text: Union[str, list]):
    sprite_names = set()
    level_mapping = defaultdict(list)

    # Handle input type
    if isinstance(vgdl_text, list):
        vgdl_lines = vgdl_text
        vgdl_string = '\n'.join(vgdl_text)
    else:
        vgdl_string = vgdl_text
        vgdl_lines = vgdl_text.split('\n')

    # Extract SpriteSet using string
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

    # Extract LevelMapping using lines
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

    # If state is a string, treat it as pre-rendered ASCII
    if isinstance(state, str):
        return state

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


def generate_mapping_and_ascii(state, vgdl_text: Union[str, list]):
    sprite_names, level_mapping = parse_vgdl(vgdl_text)

    sprite_to_char = {}
    for char, sprite_list in level_mapping.items():
        for sprite in sprite_list:
            sprite_to_char[sprite] = char

    sprite_to_char['background'] = '.'

    # Flatten leaf sprites (supports deep nesting)
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


if __name__== "__main__":

    load_dotenv('.env')
    # lst = get_available_games()
    # for item in lst:
    #     print(get_game_env(item))

    env, vgdl_rules, lvl_layout = get_game_env('gvgai-aliens-lvl0-v0')

    mapping, ascii_map = generate_mapping_and_ascii(lvl_layout, vgdl_rules)

    print("\n=== SPRITE MAPPING ===")
    for sprite, char in mapping.items():
        print(f"{sprite:15} => '{char}'")

    print("\n=== ASCII LEVEL ===")
    print(ascii_map)

    print(lvl_layout)
