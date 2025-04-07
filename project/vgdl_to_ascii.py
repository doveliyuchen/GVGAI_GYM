# vgdl_state_to_ascii.py

import re
import string
from collections import defaultdict
from typing import Union

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

if __name__ == "__main__":
    import argparse
    import json
    import numpy as np

    parser = argparse.ArgumentParser(description="Convert VGDL state to ASCII using VGDL rules.")
    parser.add_argument("--state", type=str, required=True, help="Path to state .json or .npy file")
    parser.add_argument("--vgdl", type=str, required=True, help="Path to VGDL rules .txt file")

    args = parser.parse_args()

    if args.state.endswith(".json"):
        with open(args.state, "r") as f:
            state = json.load(f)
    else:
        state = np.load(args.state, allow_pickle=True)

    with open(args.vgdl, "r") as f:
        vgdl_text = f.read()

    mapping, ascii_map = generate_mapping_and_ascii(state, vgdl_text)

    print("\n=== SPRITE MAPPING ===")
    for sprite, char in mapping.items():
        print(f"{sprite:15} => '{char}'")

    print("\n=== ASCII LEVEL ===")
    print(ascii_map)
