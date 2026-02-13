import os
import json
import pandas as pd
import gym_gvgai as gvgai
from datetime import datetime
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import gc
import psutil
import signal
import sys
import traceback

from llm.agent.llm_agent import LLMPlayer
from llm.agent.llm_translator import LLMTranslator
from llm.utils.agent_components import show_state_gif, generate_mapping_and_ascii, extract_avatar_position_from_state
from llm.utils.vgdl_utils import load_level_map, load_vgdl_rules
from dotenv import load_dotenv
from llm.utils.config import get_profile_config

# --- Helper functions for managing run directories (inspired by llm_agent_loop.py) ---
def get_game_name_simple(env_name_full):
    """Extracts a simplified game name like 'zelda-lvl1' from 'gvgai-zelda-lvl1-v0'."""
    match = re.search(r'gvgai-(.*?)-v0', env_name_full)
    if match:
        return match.group(1)
    return env_name_full # Fallback

def get_model_name_simple(model_name_full):
    """Returns the model name, handling 'portkey-' prefix for directory naming."""
    return model_name_full

def get_run_dir_path(base_dir, model_name_full, env_name_full, run_id):
    """Get the directory path for a specific run."""
    model_simple = get_model_name_simple(model_name_full)
    game_simple = get_game_name_simple(env_name_full)
    return os.path.join(base_dir, model_simple, game_simple, f"run_{run_id}")

def check_run_dir_is_taken(base_dir, model_name_full, env_name_full, run_id):
    """Check if the specified run directory is taken (has a completed benchmark_analysis.json or is currently running)."""
    run_dir = get_run_dir_path(base_dir, model_name_full, env_name_full, run_id)
    analysis_file_path = os.path.join(run_dir, "benchmark_analysis.json")
    temp_file_path = os.path.join(run_dir, ".running_temp")
    
    # A run is "taken" if:
    # 1. The benchmark_analysis.json directory exists (indicating successful completion)
    # 2. The .running_temp file exists (indicating another worker is currently running this task)
    return os.path.exists(analysis_file_path) or os.path.isfile(temp_file_path)

def find_next_available_run_id(base_dir, model_name_full, env_name_full, initial_run_id):
    """Find the next available run ID where no successful completion exists (no benchmark_analysis.json)."""
    run_id = initial_run_id
    while check_run_dir_is_taken(base_dir, model_name_full, env_name_full, run_id):
        print(f"Run_id {run_id} (Game: {env_name_full}, Model: {model_name_full}) already has successful completion (benchmark_analysis.json exists). Trying next run ID.")
        run_id += 1
    return run_id

def check_memory_usage(threshold_percent=85):
    """Check if memory usage is above threshold"""
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > threshold_percent:
        print(f"WARNING: High memory usage detected: {memory_percent}%")
        return True
    return False

def safe_cleanup(obj, obj_name="object"):
    """Safely cleanup an object with error handling"""
    try:
        if obj is not None:
            if hasattr(obj, 'close'):
                obj.close()
            elif hasattr(obj, 'cleanup'):
                obj.cleanup()
            del obj
    except Exception as e:
        print(f"Warning: Error cleaning up {obj_name}: {e}")

def run_single_game_task(env_name_full: str, mode: str, model_name_full: str, requested_run_id: int, 
                         base_output_dir: str, max_steps: int, portkey_virtual_key: str = None,
                         force_rerun: bool = False):
    """
    Worker function to run a single game instance with improved error handling and resource management.
    """
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"Process {os.getpid()}: Received signal {signum}, attempting graceful shutdown...")
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    actual_model_for_agent = model_name_full
    
    # Log information about virtual key usage for this task
    if portkey_virtual_key:
        print(f"Process {os.getpid()}: Task for {model_name_full} received specific virtual key.")
    elif model_name_full.startswith("portkey-") or model_name_full == "gemini":
        print(f"Warning: Task for Portkey-type model {model_name_full} did not receive a specific virtual key.")
    
    actual_run_id_to_use: int
    if force_rerun:
        actual_run_id_to_use = requested_run_id
        print(f"Force rerun enabled. Using run_id: {actual_run_id_to_use} for {env_name_full}, {model_name_full}, {mode}.")
    else:
        actual_run_id_to_use = find_next_available_run_id(
            base_output_dir, model_name_full, env_name_full, requested_run_id
        )
        if actual_run_id_to_use != requested_run_id:
            print(f"Requested run_id {requested_run_id} was taken. Using next available run_id: {actual_run_id_to_use}")
        else:
            print(f"Using requested run_id: {actual_run_id_to_use}")

    run_dir = get_run_dir_path(base_output_dir, model_name_full, env_name_full, actual_run_id_to_use)
    temp_file_path = os.path.join(run_dir, ".running_temp")
    
    # Create directory and temporary file to mark this task as running
    os.makedirs(run_dir, exist_ok=True)
    try:
        with open(temp_file_path, 'w') as f:
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            f.write(f"Process ID: {os.getpid()}\n")
            f.write(f"Game: {env_name_full}\n")
            f.write(f"Model: {model_name_full}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Run ID: {actual_run_id_to_use}\n")
        print(f"Created temporary file: {temp_file_path}")
    except Exception as e:
        print(f"Warning: Could not create temporary file {temp_file_path}: {e}")
    
    print(f"\n==== Starting game: {env_name_full} | Mode: {mode} | Model: {model_name_full} | Run: {actual_run_id_to_use} ====")

    # Initialize variables for proper cleanup
    env = None
    gif_saver = None
    player = None
    translator = None
    step_count = 0
    info = {}
    game_successful = False
    result_summary = ""
    
    try:
        # Check memory before starting
        if check_memory_usage(80):
            gc.collect()  # Force garbage collection
        
        # Create environment with timeout and error handling
        try:
            env = gvgai.make(env_name_full)
            if env is None:
                raise RuntimeError(f"Failed to create environment for {env_name_full}")
        except Exception as e:
            raise RuntimeError(f"Environment creation failed for {env_name_full}: {e}")
        
        # Load VGDL rules with error handling
        try:
            vgdl_rules = load_vgdl_rules(env_name_full)
            if not vgdl_rules:
                print(f"Warning: Empty VGDL rules for {env_name_full}")
        except Exception as e:
            print(f"Warning: Could not load VGDL rules for {env_name_full}: {e}")
            vgdl_rules = ""
        
        # Determine level and load level layout
        level_match_in_env_name = re.search(r'-lvl(\d+)-v\d+$', env_name_full)
        level_idx_0_based = 0
        if level_match_in_env_name:
            level_idx_0_based = int(level_match_in_env_name.group(1))
        
        try:
            level_layout = load_level_map(env_name_full, level_idx_0_based + 1)
        except Exception as e:
            print(f"Warning: Could not load level layout for {env_name_full}: {e}")
            level_layout = None

        if level_layout is None:
            print(f"Warning: No level layout found for {env_name_full} (level {level_idx_0_based})")

        # Reset environment with error handling
        try:
            state = env.reset()
        except Exception as e:
            raise RuntimeError(f"Environment reset failed for {env_name_full}: {e}")
        
        # Create translator with error handling and timeout
        try:
            translator = LLMTranslator(model_name=actual_model_for_agent)
            translation_text = translator.translate(vgdl_rules=vgdl_rules, level_layout=level_layout)
            translated_rules = f"Game rules in natural language:\n{translation_text}"
        except Exception as e:
            print(f"Warning: Translation failed for {env_name_full}: {e}. Using raw VGDL rules.")
            translated_rules = f"Game rules (raw VGDL):\n{vgdl_rules}"
            # Don't fail the entire task for translation issues
        
        # Create player with error handling
        try:
            player = LLMPlayer(
                model_name=actual_model_for_agent,
                env=env,
                vgdl_rules=translated_rules,
                initial_state=level_layout,
                mode=mode,
                rotate_state=False,
                expand_state=True,
                log_dir=run_dir,
            )
        except Exception as e:
            raise RuntimeError(f"Player creation failed for {env_name_full}: {e}")

        # Initialize GIF saver
        try:
            gif_saver = show_state_gif()
        except Exception as e:
            print(f"Warning: GIF saver initialization failed: {e}")
            gif_saver = None
        
        # Reset environment again for clean start
        try:
            current_observation_pixels = env.reset()
            if gif_saver:
                gif_saver(env)
        except Exception as e:
            print(f"Warning: Environment reset or GIF capture failed: {e}")
        
        # Get initial state
        try:
            _, _, _, info_init = env.step(0)
            ascii_state = info_init.get('ascii', '')
            if not ascii_state:
                print(f"Warning: Initial ASCII state is empty for {env_name_full}")
        except Exception as e:
            print(f"Warning: Initial step failed: {e}")
            ascii_state = ""

        done = False
        sprite_map = {}

        # Main game loop with improved error handling
        while not done and (max_steps is None or step_count < max_steps):
            try:
                # Check memory periodically
                if step_count % 50 == 0 and check_memory_usage(90):
                    print(f"High memory usage at step {step_count}, forcing garbage collection")
                    gc.collect()
                
                print(f"== {env_name_full} | Mode: {mode} | Model: {model_name_full} | Run: {actual_run_id_to_use} | Step {step_count+1}/{max_steps if max_steps is not None else 'inf'} ==")

                # Generate current state representation
                try:
                    current_sprite_map, current_ascii_state_str, _ = generate_mapping_and_ascii(
                        state_str=ascii_state,
                        vgdl_text=vgdl_rules,
                        existing_mapping=sprite_map 
                    )
                    sprite_map.update(current_sprite_map)
                except Exception as e:
                    print(f"Warning: State generation failed at step {step_count}: {e}")
                    current_ascii_state_str = ascii_state or ""

                try:
                    current_position = extract_avatar_position_from_state(
                        ascii_lines=current_ascii_state_str,
                        sprite_to_char=sprite_map,
                        flip_vertical=False
                    )
                except Exception as e:
                    print(f"Warning: Position extraction failed at step {step_count}: {e}")
                    current_position = None

                # Select action with timeout and error handling
                try:
                    action = player.select_action(
                        current_state=current_ascii_state_str,
                        current_position=current_position,
                        sprite_map=sprite_map,
                    )
                except Exception as e:
                    print(f"Warning: Action selection failed at step {step_count}: {e}. Using default action 0.")
                    action = 0

                # Execute action
                try:
                    _, reward, done, info = env.step(action)
                    winner_info = info.get('winner', None)
                    player.update(action=action, reward=reward, winner=winner_info)
                    ascii_state = info.get('ascii', '')
                    if gif_saver:
                        gif_saver(env)
                    step_count += 1
                except Exception as e:
                    print(f"Warning: Environment step failed at step {step_count}: {e}")
                    # Try to continue with next step
                    step_count += 1
                    if step_count >= (max_steps or 1000):  # Prevent infinite loops
                        break
                    
            except Exception as e:
                print(f"Error in game loop at step {step_count}: {e}")
                traceback.print_exc()
                break  # Exit game loop on serious errors
        
        # Determine game outcome
        win_status = info.get('winner', 'UNKNOWN') if info else 'UNKNOWN'
        total_reward_val = player.total_reward if player else 0
        game_successful = done or (max_steps is not None and step_count >= max_steps)
        
        result_summary = f"Finished: {env_name_full}, {model_name_full}, {mode}, run {actual_run_id_to_use}, steps {step_count}, reward {total_reward_val}, win {win_status}, proper_end: {game_successful}"

    except Exception as e:
        error_msg = f"ERROR during game: {env_name_full}, Model: {model_name_full}, Mode: {mode}, Run: {actual_run_id_to_use}"
        print(f"!!! {error_msg} !!!")
        print(f"Error type: {type(e).__name__}, Message: {e}")
        traceback.print_exc()
        result_summary = f"Failed: {env_name_full}, {model_name_full}, {mode}, run {actual_run_id_to_use} - {type(e).__name__}: {str(e)}"
        
    finally:
        # Comprehensive cleanup with error handling
        print(f"Starting cleanup for {env_name_full}, run {actual_run_id_to_use}")
        
        # Clean up environment
        safe_cleanup(env, "environment")
        
        # Save logs and analysis if game was successful
        if game_successful and player:
            try:
                player.save_logs()
                analysis_file_path = os.path.join(run_dir, "benchmark_analysis.json")
                player.export_analysis(analysis_file_path)
                print(f"Analysis saved to {analysis_file_path}")
            except Exception as e:
                print(f"Warning: Could not save player logs/analysis: {e}")
        elif player:
            print(f"Game was not successful. Skipping log saving.")

        # Save GIF if available
        if gif_saver and step_count > 0:
            try:
                gif_path = os.path.join(run_dir, "gameplay.gif")
                gif_saver.save(gif_path)
                print(f"GIF saved to {gif_path}")
            except Exception as e:
                print(f"Warning: Could not save GIF: {e}")

        # Clean up objects
        safe_cleanup(player, "player")
        safe_cleanup(translator, "translator")
        safe_cleanup(gif_saver, "gif_saver")
        
        # Final status print
        if player:
            print(f"[{mode.upper()}] Game: {env_name_full} Model: {model_name_full} Run: {actual_run_id_to_use} "
                  f"ended after {step_count} steps. Total reward: {player.total_reward}. Success: {game_successful}")
        
        # Clean up temporary file
        try:
            if os.path.isfile(temp_file_path):
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file_path}: {e}")
        
        # Force garbage collection
        gc.collect()
        
        print(f"Cleanup completed for {env_name_full}, run {actual_run_id_to_use}")
    
    return result_summary


def generate_tasks_prioritized(game_list_to_process, models, modes, num_runs, base_output_dir, max_steps, portkey_virtual_keys_loaded, force_rerun):
    """
    Generate tasks with prioritized ordering: complete all games for run 1, then run 2, etc.
    """
    tasks = []
    
    for current_run_num in range(1, num_runs + 1):
        print(f"Preparing tasks for run {current_run_num}...")
        
        for game_short_name_cli in game_list_to_process:
            game_base_name_for_level_iteration = game_short_name_cli
            game_version_for_level_iteration = "0"

            match_versioned_cli = re.match(r"(.+)_v(\d+)", game_short_name_cli)
            if match_versioned_cli:
                game_base_name_for_level_iteration = match_versioned_cli.group(1)
                game_version_for_level_iteration = match_versioned_cli.group(2)
            
            game_base_name_for_level_iteration = re.sub(r'-lvl\d+', '', game_base_name_for_level_iteration)

            levels_to_process = []
            game_dir_name = f"{game_base_name_for_level_iteration}_v{game_version_for_level_iteration}"

            script_dir = Path(__file__).parent
            possible_game_paths = [
                Path(f"../gym_gvgai/envs/games/{game_dir_name}"),
                Path(f"gym_gvgai/envs/games/{game_dir_name}"),
            ]
            
            game_levels_path = None
            for path in possible_game_paths:
                if path.is_dir():
                    game_levels_path = path
                    break
            
            if game_levels_path is None:
                print(f"Warning: Game directory not found for {game_dir_name}. Will attempt level 0.")
                levels_to_process.append(0)
                continue
                
            print(f"Discovering levels for {game_dir_name} in path: {game_levels_path.resolve()}")
            if game_levels_path.is_dir():
                found_level_indices = set()
                for f_path in sorted(game_levels_path.glob("*.txt")):
                    filename = f_path.name
                    if filename.lower() == f"{game_base_name_for_level_iteration}.txt".lower():
                        continue
                    
                    primary_pattern = rf"{re.escape(game_base_name_for_level_iteration)}_lvl(\d+)\.txt"
                    level_match_primary = re.match(primary_pattern, filename, re.IGNORECASE)
                    
                    if level_match_primary:
                        found_level_indices.add(int(level_match_primary.group(1)))
                    else:
                        fallback_pattern = r'lvl(\d+)\.txt'
                        level_match_fallback = re.match(fallback_pattern, filename, re.IGNORECASE)
                        if level_match_fallback:
                            found_level_indices.add(int(level_match_fallback.group(1)))
                
                if found_level_indices:
                    levels_to_process = sorted(list(found_level_indices))
                    print(f"Found levels for {game_dir_name}: {levels_to_process}")
                else:
                    print(f"Warning: No level files found for {game_dir_name}. Will attempt level 0.")
                    levels_to_process.append(0)
            else:
                print(f"Warning: Game directory {game_levels_path} not found. Will attempt level 0.")
                levels_to_process.append(0)

            if not levels_to_process:
                continue

            for level_num in levels_to_process:
                env_name_full = f'gvgai-{game_base_name_for_level_iteration}-lvl{level_num}-v{game_version_for_level_iteration}'

                for model_name_full in models:
                    for mode in modes:
                        task_args = {
                            "env_name_full": env_name_full,
                            "mode": mode,
                            "model_name_full": model_name_full,
                            "requested_run_id": current_run_num,
                            "base_output_dir": base_output_dir,
                            "max_steps": max_steps,
                            "force_rerun": force_rerun
                        }
                        
                        if model_name_full in portkey_virtual_keys_loaded:
                            task_args["portkey_virtual_key"] = portkey_virtual_keys_loaded[model_name_full]
                        
                        if not force_rerun and check_run_dir_is_taken(base_output_dir, model_name_full, env_name_full, current_run_num):
                            print(f"Run {current_run_num} already exists for {env_name_full}, {model_name_full}, {mode}. Skipping.")
                            continue
                        tasks.append(task_args)
    
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Run LLM Agent on GVGAI games with improved error handling')
    parser.add_argument('--games', nargs='*', default=None, help='List of game names')
    parser.add_argument('--models', nargs='+', required=True, help='List of models')
    parser.add_argument('--modes', nargs='+', default=['zero-shot', 'contextual'], help='List of modes')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs per game/model/mode')
    parser.add_argument('--base_output_dir', type=str, default='llm_agent_runs_output', help='Base output directory')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum parallel workers (default: 4, reduced for stability)')
    parser.add_argument('--force_rerun', action='store_true', help='Force rerun existing tasks')
    parser.add_argument('--reverse', action='store_true', help='Process games in reverse order')
    parser.add_argument('--resume_game', type=str, default=None, help='Resume from specific game')
    parser.add_argument('--specific_level', type=int, default=None, help='Process only specific level')

    args = parser.parse_args()

    # Reduce max_workers if system has limited resources
    available_memory_gb = psutil.virtual_memory().total / (1024**3)
    if available_memory_gb < 16:  # Less than 16GB RAM
        recommended_workers = max(1, min(args.max_workers, 2))
        print(f"System has {available_memory_gb:.1f}GB RAM. Reducing max_workers to {recommended_workers} for stability.")
        args.max_workers = recommended_workers

    # Load Portkey virtual keys
    portkey_virtual_keys_loaded = {}
    
    try:
        script_dir = Path(__file__).parent
        dotenv_path = script_dir / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path=str(dotenv_path))
            print(f"Loaded .env file from: {dotenv_path.resolve()}")
        else:
            print(f"Warning: .env file not found at {dotenv_path.resolve()}")

        for model_cli_name in args.models:
            try:
                profile_data = get_profile_config(model_cli_name)
                virtual_key_env_var_name = profile_data.get("virtual_key_env_var")

                if virtual_key_env_var_name:
                    virtual_key_value = os.getenv(virtual_key_env_var_name)
                    if virtual_key_value:
                        portkey_virtual_keys_loaded[model_cli_name] = virtual_key_value
                        print(f"Loaded virtual key for '{model_cli_name}'")
                    else:
                        print(f"Warning: Virtual key env var '{virtual_key_env_var_name}' not found for '{model_cli_name}'")
                elif model_cli_name.startswith("portkey-"):
                    # Fallback for older naming
                    key_suffix = model_cli_name.replace('portkey-', '').upper().replace('-', '_')
                    env_var_name_fallback = f'PORTKEY_VIRTUAL_KEY_{key_suffix}'
                    key_fallback_val = os.getenv(env_var_name_fallback)
                    if key_fallback_val:
                        portkey_virtual_keys_loaded[model_cli_name] = key_fallback_val
                        print(f"Loaded virtual key for '{model_cli_name}' using fallback")
                    else:
                        print(f"Warning: No virtual key found for '{model_cli_name}'")

            except Exception as e:
                print(f"Error processing model '{model_cli_name}' for keys: {e}")
    
    except Exception as e:
        print(f"Error during Portkey key setup: {e}")

    # Determine games to process
    game_list_to_process = []
    if args.games:
        game_list_to_process = args.games
        print(f"Processing user-specified games: {game_list_to_process}")
    else:
        print("Scanning for all available games...")
        possible_paths = [
            Path("../gym_gvgai/envs/games/"),
            Path("gym_gvgai/envs/games/"),
        ]
        
        all_games_dir = None
        for path in possible_paths:
            if path.is_dir():
                all_games_dir = path
                break
        
        if all_games_dir is None:
            print(f"Error: Could not find games directory. Please specify games via --games.")
            return
            
        if all_games_dir.is_dir():
            for game_path in sorted(all_games_dir.iterdir()):
                if game_path.is_dir():
                    game_name = game_path.name
                    if "testgame" not in game_name.lower():
                        game_list_to_process.append(game_name)
            if game_list_to_process:
                print(f"Found {len(game_list_to_process)} games: {game_list_to_process}")
            else:
                print(f"Warning: No games found after excluding testgames.")
                return
        else:
            print(f"Error: Games directory {all_games_dir} not found. Please specify games via --games.")
            return
            
    # Apply resume_game and reverse filters
    if args.resume_game:
        try:
            start_index = game_list_to_process.index(args.resume_game)
            game_list_to_process = game_list_to_process[start_index:]
            print(f"Resuming from game: {args.resume_game}. Games to process: {game_list_to_process}")
        except ValueError:
            print(f"Warning: Resume game '{args.resume_game}' not found in the game list. Processing all games.")
    
    if args.reverse:
        game_list_to_process.reverse()
        print(f"Processing games in reverse order: {game_list_to_process}")

    # Generate tasks with prioritized ordering
    print(f"\n=== Generating tasks with prioritized execution ===")
    print(f"Strategy: Complete all games for run 1, then run 2, etc.")
    print(f"This ensures faster initial results across all games.")
    
    tasks = generate_tasks_prioritized(
        game_list_to_process=game_list_to_process,
        models=args.models,
        modes=args.modes,
        num_runs=args.num_runs,
        base_output_dir=args.base_output_dir,
        max_steps=args.max_steps,
        portkey_virtual_keys_loaded=portkey_virtual_keys_loaded,
        force_rerun=args.force_rerun
    )

    if not tasks:
        print("No tasks to run.")
        return

    print(f"\nPrepared {len(tasks)} tasks to run with {args.max_workers} workers.")
    print(f"Task execution order: Run 1 (all games) -> Run 2 (all games) -> ... -> Run {args.num_runs} (all games)")
    
    # Using ProcessPoolExecutor with improved error handling
    completed_count = 0
    total_tasks = len(tasks)
    failed_tasks = []
    
    try:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(run_single_game_task, **task): task for task in tasks}
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed_count += 1
                
                try:
                    result = future.result(timeout=30)  # 30 second timeout for getting results
                    print(f"[{completed_count}/{total_tasks}] Task completed: {result}")
                except Exception as e:
                    failed_task_info = f"{task['env_name_full']}, {task['model_name_full']}, {task['mode']}, run_{task['requested_run_id']}"
                    failed_tasks.append(failed_task_info)
                    print(f"[{completed_count}/{total_tasks}] !!! Task failed: {failed_task_info} - {type(e).__name__}: {e} !!!")
                    import traceback
                    traceback.print_exc()
                    
    except KeyboardInterrupt:
        print("\n!!! Keyboard interrupt received (Ctrl+C) !!!")
        print("Attempting to cancel remaining tasks...")
        
        # Cancel all pending futures
        for future in future_to_task.keys():
            if not future.done():
                cancelled = future.cancel()
                if cancelled:
                    print(f"Cancelled pending task")
                else:
                    print(f"Could not cancel running task")
        
        # Wait for graceful shutdown
        import time
        time.sleep(2)
        
        print("Shutdown complete. Some tasks may have been interrupted.")
        return
    except Exception as e:
        print(f"!!! Unexpected error in main execution: {e} !!!")
        traceback.print_exc()
        return

    # Summary
    print(f"\n=== Execution Summary ===")
    print(f"Total tasks: {total_tasks}")
    print(f"Completed: {completed_count}")
    print(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print("\nFailed tasks:")
        for failed_task in failed_tasks:
            print(f"  - {failed_task}")
    
    print(f"=== All tasks processed ===")


if __name__ == "__main__":
    main()
