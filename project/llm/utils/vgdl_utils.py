# llm/utils/vgdl_utils.py

import os
import gym
from typing import Tuple, List
from pathlib import Path # Ensure Path is imported

# Determine the project root directory.
# Assuming vgdl_utils.py is in GVGAI_LLM/project/llm/utils/
# Path(__file__) is the path to this file.
# .resolve() makes it absolute.
# .parent moves up one directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent 
GVGAI_GAMES_BASE_DIR = PROJECT_ROOT / "gym_gvgai" / "envs" / "games"

def parse_env_name(env_name: str) -> Tuple[str, str]:
    """
    Splits env_name like 'gvgai-x-ray-lvl0-v0' into ('x-ray', 'lvl0')
    """
    if not env_name.startswith("gvgai-") or not env_name.endswith("-v0"):
        raise ValueError(f"Invalid env_name format: {env_name}")
    name_core = env_name[len("gvgai-"):-len("-v0")]
    if "-lvl" not in name_core:
        # Allow games that might not have levels explicitly in name, assume lvl0
        # For example, if env_name is 'gvgai-aliens-v0', it implies level 0.
        # This case needs to be handled if parse_env_name is strictly for -lvlX names.
        # For now, let's assume -lvl is always present if levels are distinct.
        # If a game like "gvgai-bait-v0" is passed, and it implies level 0,
        # then parse_env_name might need adjustment or the caller handles it.
        # The error message indicates "bait_v0/bait.txt", so game name is "bait".
        # If env_name was "gvgai-bait-lvl0-v0", parse_env_name works.
        # If env_name was "gvgai-bait-v0", it would fail here.
        # Let's assume main_4.24.py always constructs env_name with -lvlX.
        raise ValueError(f"Expected '-lvl' in env_name core: {name_core}")
    game, level_str = name_core.rsplit("-lvl", 1)
    return game, f"lvl{level_str}"

def get_game_paths(env_name: str) -> Tuple[str, str]:
    """
    Constructs paths for VGDL and level files using GVGAI_GAMES_BASE_DIR.
    Note: This function's level path construction (game_level.txt) might differ
    from the typical lvlX.txt. load_level_map uses lvlX.txt directly.
    """
    game, level_str_part = parse_env_name(env_name) # e.g., ("bait", "lvl0")
    
    game_dir_abs = GVGAI_GAMES_BASE_DIR / f"{game}_v0"
    vgdl_path_abs = game_dir_abs / f"{game}.txt"
    
    # This level path construction might be specific or an alternative naming
    level_path_abs = game_dir_abs / f"{game}_{level_str_part}.txt" # e.g., bait_lvl0.txt
    
    return str(vgdl_path_abs), str(level_path_abs)

def get_available_games(full_name: bool = True) -> List[str]:
    """List all gvgai environments registered in gym."""
    try:
        gvgai_envs = [env.id for env in gym.envs.registry.all() if env.id.startswith('gvgai')]
        if full_name:
            return sorted(gvgai_envs)

        game_names = set()
        for env_id in gvgai_envs:
            parts = env_id.split('-')
            if len(parts) >= 3:
                game_name = parts[1] # This extracts 'aliens' from 'gvgai-aliens-lvl0-v0'
                game_names.add(game_name)
        return sorted(list(game_names))

    except Exception as e:
        print(f"Error listing games: {e}")
        return []

def load_vgdl_rules(env_name: str) -> str:
    """
    Loads VGDL rules text. Paths are constructed relative to the project root.
    :param env_name: e.g., 'gvgai-zelda-lvl0-v0'
    """
    game, _ = parse_env_name(env_name) 
    game_specific_dir = GVGAI_GAMES_BASE_DIR / f"{game}_v0"
    vgdl_file_path = game_specific_dir / f"{game}.txt"

    with open(vgdl_file_path, 'r') as f:
        return f.read()

def load_level_map(env_name: str, level_idx_1_based: int) -> str:
    """
    Loads VGDL level layout. Paths are constructed relative to the project root.
    :param env_name: e.g., 'gvgai-zelda-lvl0-v0'. Used to get game base name.
    :param level_idx_1_based: 1-based index of the level to load (e.g., 1 for lvl0.txt)
    """
    game, _ = parse_env_name(env_name) # game = "angelsdemons" from "gvgai-angelsdemons-lvlX-v0"
    
    game_specific_dir = GVGAI_GAMES_BASE_DIR / f"{game}_v0" # e.g., .../angelsdemons_v0
    
    # Construct the level string part like "lvl0", "lvl1"
    level_num_0_based = level_idx_1_based - 1
    level_str_part_for_filename = f"lvl{level_num_0_based}" # e.g., "lvl0"

    # Primary pattern: game_lvlX.txt (e.g., angelsdemons_lvl0.txt)
    primary_level_filename = f"{game}_{level_str_part_for_filename}.txt"
    primary_level_path = game_specific_dir / primary_level_filename

    if primary_level_path.exists():
        with open(primary_level_path, 'r') as f:
            return f.read()
    else:
        # Fallback pattern: lvlX.txt (e.g., lvl0.txt)
        fallback_level_filename = f"lvl{level_num_0_based}.txt"
        fallback_level_path = game_specific_dir / fallback_level_filename
        if fallback_level_path.exists():
            print(f"[vgdl_utils] Primary level file '{primary_level_filename}' not found. Using fallback '{fallback_level_filename}'.")
            with open(fallback_level_path, 'r') as f:
                return f.read()
        else:
            print(f"[vgdl_utils] Level file not found using pattern '{primary_level_filename}' or fallback '{fallback_level_filename}' in {game_specific_dir}. Attempted primary: {primary_level_path}. Returning None.")
            return None
