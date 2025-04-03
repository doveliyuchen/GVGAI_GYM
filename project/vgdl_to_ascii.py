# vgdl_state_to_ascii.py

import re
import string
from collections import defaultdict

def parse_vgdl(vgdl_text):
    sprite_names = set()
    level_mapping = defaultdict(list)

    # Extract SpriteSet
    sprite_section = re.search(r"SpriteSet(.*?)LevelMapping", vgdl_text, re.DOTALL)
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

    # Extract LevelMapping
    lines = vgdl_text.split('\n')
    in_level_mapping = False
    for line in lines:
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
