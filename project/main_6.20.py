import os
import json
import pandas as pd
import gym_gvgai as gvgai
from datetime import datetime
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path # Added for Path object

from llm.agent.llm_agent import LLMPlayer
from llm.agent.llm_translator import LLMTranslator
from llm.utils.agent_components import show_state_gif, generate_mapping_and_ascii, extract_avatar_position_from_state
from llm.utils.vgdl_utils import load_level_map, load_vgdl_rules
from dotenv import load_dotenv
from llm.utils.config import get_profile_config # To read llm_config.json for each model
# --- Helper functions for managing run directories (inspired by llm_agent_loop.py) ---
def get_game_name_simple(env_name_full):
    """Extracts a simplified game name like 'zelda-lvl1' from 'gvgai-zelda-lvl1-v0'."""
    match = re.search(r'gvgai-(.*?)-v0', env_name_full)
    if match:
        return match.group(1)
    return env_name_full # Fallback

def get_model_name_simple(model_name_full):
    """Returns the model name, handling 'portkey-' prefix for directory naming."""
    # For directory structure, we might want to keep the full portkey model name
    # e.g., portkey-gpt-4o-mini, not just portkey.
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
# --- End of helper functions ---

def run_single_game_task(env_name_full: str, mode: str, model_name_full: str, requested_run_id: int, 
                         base_output_dir: str, max_steps: int, portkey_virtual_key: str = None,
                         force_rerun: bool = False):
    """
    Worker function to run a single game instance.
    Handles setting environment variables for Portkey if needed.
    """
    
    actual_model_for_agent = model_name_full # Use the full name like "portkey-4o-mini" or "gemini"

    # Log information about virtual key usage for this task
    if portkey_virtual_key:
        print(f"Process {os.getpid()}: Task for {model_name_full} received specific virtual key.")
    elif model_name_full.startswith("portkey-") or model_name_full == "gemini":
        # This warning is important if main() failed to load a key for a Portkey-type model
        print(f"Warning: Task for Portkey-type model {model_name_full} did not receive a specific virtual key. "
              "Client creation might fail or use a default if not configured correctly in llm_config.json / .env.")
    
    # The actual_model_for_agent (which is model_name_full) will be used by LLMPlayer/LLMTranslator
    # to call create_client_from_config. That function will then use this profile name
    # to look up the necessary details (including which env var holds the virtual key)
    # from llm_config.json and instantiate PortkeyClient correctly.
    # No need to set a generic os.environ['PORTKEY_VIRTUAL_KEY'] here.

    actual_run_id_to_use: int
    if force_rerun:
        actual_run_id_to_use = requested_run_id
        print(f"Force rerun enabled. Using run_id: {actual_run_id_to_use} for {env_name_full}, {model_name_full}, {mode}.")
    else:
        # Check if the requested_run_id directory is taken. If so, find the next available one.
        # If requested_run_id directory is not taken (non-existent or empty), it will be used.
        actual_run_id_to_use = find_next_available_run_id(
            base_output_dir, model_name_full, env_name_full, requested_run_id
        )
        if actual_run_id_to_use != requested_run_id:
            print(f"Requested run_id {requested_run_id} was taken or would overwrite. Using next available run_id: {actual_run_id_to_use} for {env_name_full}, {model_name_full}, {mode}.")
        else:
            # This means the requested_run_id directory was either non-existent or empty, so it's fine to use.
            print(f"Using requested run_id: {actual_run_id_to_use} for {env_name_full}, {model_name_full}, {mode}.")


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
    
    print(f"\n==== Starting game: {env_name_full} | Mode: {mode} | Model: {model_name_full} | Run: {actual_run_id_to_use} | Output: {run_dir} ====")

    env = None
    gif_saver = None
    player = None
    step_count = 0
    info = {} # Initialize info to ensure it exists
    game_successful = False # Flag to track successful completion
    
    try:
        env = gvgai.make(env_name_full)
        vgdl_rules = load_vgdl_rules(env_name_full)
        
        # Dynamically determine level for loading map. load_level_map expects 1-based index.
        level_match_in_env_name = re.search(r'-lvl(\d+)-v\d+$', env_name_full)
        level_idx_0_based = 0 # Default to level 0 if not found in name
        if level_match_in_env_name:
            level_idx_0_based = int(level_match_in_env_name.group(1))
        
        level_layout = load_level_map(env_name_full, level_idx_0_based + 1) # This now returns None if file not found

        if level_layout is None:
            print(f"Warning: No explicit level layout file found for {env_name_full} (level index {level_idx_0_based}). "
                  "Proceeding without specific layout for translator/player initial_state. "
                  "The game environment will use its default for this level.")
            # translator and player will receive level_layout=None or initial_state=None

        state = env.reset()
        # Initial step to get ASCII, if needed before translator
        # _, _, _, info = env.step(0) # Taking an action might not be desired before planning
        # ascii_state = info['ascii']
        
        # Generate initial state representation for translator
        # This part might need adjustment based on how translator expects initial state
        # For now, assume translator works with VGDL and level layout directly.

        translator = LLMTranslator(
            model_name=actual_model_for_agent
            # If LLMTranslator/Portkey client can take virtual_key directly, pass it here
            # virtual_key=portkey_virtual_key if model_name_full.startswith('portkey-') else None
        )
        # The translation should ideally use the specific level if applicable.
        # For now, using general rules and a representative level layout.
        translation_text = translator.translate(vgdl_rules=vgdl_rules, level_layout=level_layout)
        # Using only translated rules as per original logic
        translated_rules = f"Game rules in natural language:\n{translation_text}"

        # Create run directory only after successful translation
        os.makedirs(run_dir, exist_ok=True)

        player = LLMPlayer(
            model_name=actual_model_for_agent,
            env=env,
            vgdl_rules=translated_rules, # Use translated rules
            initial_state=level_layout, # This is the map, not the dynamic state
            mode=mode,
            rotate_state=False,
            expand_state=True,
            log_dir=run_dir, # Player logs go into the run-specific directory
            # If LLMPlayer/Portkey client can take virtual_key directly, pass it here
            # virtual_key=portkey_virtual_key if model_name_full.startswith('portkey-') else None
        )

        gif_saver = show_state_gif()
        
        # Reset again to ensure clean start for the agent after setup
        current_observation_pixels = env.reset() 
        gif_saver(env) # Save initial frame
        
        # Get initial ASCII state for the agent's first observation
        # An initial action (e.g., NOOP) might be needed to get the first ASCII state from info
        _, _, _, info_init = env.step(0) # Assuming action 0 is NOOP or safe
        ascii_state = info_init['ascii']
        if not ascii_state:
            print(f"Warning: Initial ASCII state is empty for {env_name_full}. Agent might not work correctly.")
            # Fallback or error handling needed if ascii_state is crucial and not available
            # For now, we proceed, but this could be an issue.

        done = False
        sprite_map = {} # Initialize sprite_map for the loop

        while not done and (max_steps is None or step_count < max_steps):
            print(f"== {env_name_full} | Mode: {mode} | Model: {model_name_full} | Run: {actual_run_id_to_use} | Step {step_count+1}/{max_steps if max_steps is not None else 'inf'} ==")

            # Generate current state representation for the agent
            # Ensure vgdl_rules here is the original VGDL, not the translated one, if generate_mapping_and_ascii expects it.
            # The original `vgdl_rules` variable holds the raw VGDL.
            current_sprite_map, current_ascii_state_str, _ = generate_mapping_and_ascii(
                state_str=ascii_state, # This is the current dynamic ASCII state
                vgdl_text=vgdl_rules,  # Original VGDL for parsing sprites
                existing_mapping=sprite_map 
            )
            sprite_map.update(current_sprite_map) # Persist discovered mappings

            current_position = extract_avatar_position_from_state(
                ascii_lines=current_ascii_state_str,
                sprite_to_char=sprite_map,
                flip_vertical=False # Assuming standard GVGAI coordinate system
            )

            action = player.select_action(
                current_state=current_ascii_state_str, # Pass the string representation
                current_position=current_position,
                sprite_map=sprite_map,
            )

            _, reward, done, info = env.step(action)
            # Extract winner information from info - always pass it, not just when done
            winner_info = info.get('winner', None)
            player.update(action=action, reward=reward, winner=winner_info)
            ascii_state = info['ascii'] # Update for next iteration
            gif_saver(env)
            step_count += 1
        
        win_status = info.get('winner', 'UNKNOWN') if info else 'UNKNOWN'
        total_reward_val = player.total_reward if player else 0

        # Only mark as successful if game ended properly:
        # 1. Game naturally ended (done=True, win/lose)
        # 2. Reached max steps limit
        game_successful = done or (max_steps is not None and step_count >= max_steps)
        
        result_summary = f"Finished: {env_name_full}, {model_name_full}, {mode}, run {actual_run_id_to_use}, steps {step_count}, reward {total_reward_val}, win {win_status}, proper_end: {game_successful}"

    except Exception as e:
        print(f"!!! ERROR during game: {env_name_full}, Model: {model_name_full}, Mode: {mode}, Run: {actual_run_id_to_use} !!!")
        print(f"Error type: {type(e).__name__}, Message: {e}")
        import traceback
        traceback.print_exc()
        return f"Failed: {env_name_full}, {model_name_full}, {mode}, run {actual_run_id_to_use} - {type(e).__name__}"
    finally:
        if env:
            env.close()
        
        if game_successful and player: # Only save logs if game completed successfully
            player.save_logs() 
            analysis_file_path = os.path.join(run_dir, "benchmark_analysis.json")
            player.export_analysis(analysis_file_path) 
            print(f"Analysis saved to {analysis_file_path}")
        elif player: # Player exists but game was not successful
            print(f"Game run for {env_name_full}, {model_name_full}, Run {actual_run_id_to_use} was not successful. Skipping log saving for LLMPlayer.")

        if gif_saver and step_count > 0: # Save GIF regardless of success, if frames were captured
            gif_path = os.path.join(run_dir, "gameplay.gif")
            gif_saver.save(gif_path)
            print(f"GIF saved to {gif_path}")
        
        if player: # Ensure player object exists for final status print
            print(f"[{mode.upper()}] Game: {env_name_full} Model: {model_name_full} Run: {actual_run_id_to_use} "
                  f"ended after {step_count} steps. Total reward: {player.total_reward}. Success: {game_successful}")
        
        # Clean up temporary file
        try:
            if os.path.isfile(temp_file_path):
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file_path}: {e}")
        
        # Clean up environment variable if set for Portkey
        if model_name_full.startswith('portkey-') and portkey_virtual_key:
            if 'PORTKEY_VIRTUAL_KEY' in os.environ and os.environ['PORTKEY_VIRTUAL_KEY'] == portkey_virtual_key:
                del os.environ['PORTKEY_VIRTUAL_KEY']
                print(f"Process {os.getpid()}: Cleared PORTKEY_VIRTUAL_KEY for {model_name_full}")
    
    return result_summary


def generate_tasks_prioritized(game_list_to_process, models, modes, num_runs, base_output_dir, max_steps, portkey_virtual_keys_loaded, force_rerun):
    """
    Generate tasks with prioritized ordering: complete all games for run 1, then run 2, etc.
    This ensures we get a complete picture of all games before moving to additional iterations.
    """
    tasks = []
    
    # Group tasks by run number to prioritize completion of all games per run
    for current_run_num in range(1, num_runs + 1):
        print(f"Preparing tasks for run {current_run_num}...")
        
        for game_short_name_cli in game_list_to_process:
            # Determine the base game name and version for level iteration
            game_base_name_for_level_iteration = game_short_name_cli
            game_version_for_level_iteration = "0" # Default version

            match_versioned_cli = re.match(r"(.+)_v(\d+)", game_short_name_cli)
            if match_versioned_cli:
                game_base_name_for_level_iteration = match_versioned_cli.group(1)
                game_version_for_level_iteration = match_versioned_cli.group(2)
            
            # Remove any explicit -lvlX from the base name for iteration purposes
            game_base_name_for_level_iteration = re.sub(r'-lvl\d+', '', game_base_name_for_level_iteration)

            levels_to_process = []
            game_dir_name = f"{game_base_name_for_level_iteration}_v{game_version_for_level_iteration}"

            # Default: Process all discovered levels for this game
            # Use the same logic as above to find the correct path
            script_dir = Path(__file__).parent
            possible_game_paths = [
                Path(f"../gym_gvgai/envs/games/{game_dir_name}"),  # When running from project/
                Path(f"gym_gvgai/envs/games/{game_dir_name}"),  # When running from root
            ]
            
            game_levels_path = None
            for path in possible_game_paths:
                if path.is_dir():
                    game_levels_path = path
                    break
            
            if game_levels_path is None:
                print(f"Warning: Game directory not found for {game_dir_name}. Tried: {[str(p) for p in possible_game_paths]}. Will attempt to run default level 0 if game environment can be created.")
                levels_to_process.append(0)
                continue
            print(f"Discovering all levels for {game_dir_name} in path: {game_levels_path.resolve()}")
            if game_levels_path.is_dir():
                found_level_indices = set() # Use a set to store 0-based level indices to avoid duplicates
                for f_path in sorted(game_levels_path.glob("*.txt")):
                    filename = f_path.name
                    if filename.lower() == f"{game_base_name_for_level_iteration}.txt".lower():
                        continue  # Skip main game VGDL file
                    
                    # Try primary pattern: game_lvlX.txt (e.g., angelsdemons_lvl0.txt)
                    # game_base_name_for_level_iteration is like "angelsdemons"
                    primary_pattern = rf"{re.escape(game_base_name_for_level_iteration)}_lvl(\d+)\.txt"
                    level_match_primary = re.match(primary_pattern, filename, re.IGNORECASE)
                    
                    if level_match_primary:
                        found_level_indices.add(int(level_match_primary.group(1)))
                    else:
                        # Try fallback pattern: lvlX.txt (e.g., lvl0.txt)
                        fallback_pattern = r'lvl(\d+)\.txt'
                        level_match_fallback = re.match(fallback_pattern, filename, re.IGNORECASE)
                        if level_match_fallback:
                            found_level_indices.add(int(level_match_fallback.group(1)))
                
                if found_level_indices:
                    levels_to_process = sorted(list(found_level_indices)) # Convert set to sorted list
                    print(f"Found levels for {game_dir_name}: {levels_to_process}")
                else:
                    print(f"Warning: No level files (e.g., lvl0.txt) found in {game_levels_path} for {game_dir_name}. Will attempt to run default level 0 if game environment can be created.")
                    levels_to_process.append(0) # Default to level 0 if no files found but game dir exists
            else:
                print(f"Warning: Game directory {game_levels_path} not found for {game_dir_name}. Will attempt to run default level 0 if game environment can be created.")
                levels_to_process.append(0) # Default to level 0 if game dir doesn't exist

            if not levels_to_process:
                continue

            for level_num in levels_to_process:
                # Construct the full environment name for this specific level
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
                        
                        # Pass the loaded virtual key string if this model is in our dict
                        if model_name_full in portkey_virtual_keys_loaded:
                            task_args["portkey_virtual_key"] = portkey_virtual_keys_loaded[model_name_full]
                        
                        # Only schedule the run if it is missing (not completed and not running)
                        if check_run_dir_is_taken(base_output_dir, model_name_full, env_name_full, current_run_num):
                            print(f"Run {current_run_num} already exists for {env_name_full}, {model_name_full}, {mode}. Skipping.")
                            continue
                        tasks.append(task_args)
    
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Run LLM Agent on GVGAI games in parallel with prioritized execution.')
    parser.add_argument('--games', nargs='*', default=None, help='Optional: List of game names (e.g., zelda_v0 aliens_v0). If not provided, all games (excluding testgames) will be processed.')
    parser.add_argument('--models', nargs='+', required=True, help='List of models (e.g., deepseek openai portkey-gpt-4o-mini)')
    parser.add_argument('--modes', nargs='+', default=['zero-shot', 'contextual'], help='List of modes (default: zero-shot contextual)')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for each game/model/mode combination (default: 1)')
    parser.add_argument('--base_output_dir', type=str, default='llm_agent_runs_output', help='Base directory for all results')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum steps per episode (default: None, runs until game done)')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of parallel workers (default: 10)')
    parser.add_argument('--force_rerun', action='store_true', help='Force rerun tasks even if output directory exists and is not empty.')
    parser.add_argument('--reverse', action='store_true', help='Process games in reverse order.')
    parser.add_argument('--resume_game', type=str, default=None, help='Game name (short form like zelda_v0) to resume processing from.')
    parser.add_argument('--specific_level', type=int, default=None, help='Optional: Process only this specific level number for all games. If not set, all discovered levels are processed.')

    args = parser.parse_args()

    # Load Portkey virtual keys from .env if portkey models are requested
    portkey_virtual_keys_loaded = {} # Stores CLI model name -> its virtual key string
    
    # Load .env once
    try:
        # Use Path for cross-platform compatibility
        script_dir = Path(__file__).parent  # project/
        dotenv_path = script_dir / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path=str(dotenv_path))
            print(f"Loaded .env file from: {dotenv_path.resolve()}")
        else:
            print(f"Warning: .env file not found at {dotenv_path.resolve()}. Portkey/Gemini models might fail if keys not in environment.")

        for model_cli_name in args.models: # e.g., "gemini", "portkey-o3-mini"
            try:
                # Load the llm_config.json profile for this specific model
                profile_data = get_profile_config(model_cli_name)
                
                # Check if this profile uses Portkey
                virtual_key_env_var_name = profile_data.get("virtual_key_env_var")

                if virtual_key_env_var_name: # If the profile specifies how to get its virtual key
                    virtual_key_value = os.getenv(virtual_key_env_var_name)
                    if virtual_key_value:
                        portkey_virtual_keys_loaded[model_cli_name] = virtual_key_value
                        print(f"Loaded virtual key for CLI model '{model_cli_name}' from env var '{virtual_key_env_var_name}'.")
                    else:
                        print(f"Warning: Environment variable '{virtual_key_env_var_name}' (for virtual key of '{model_cli_name}') not found.")
                elif model_cli_name.startswith("portkey-"): # Fallback for older portkey- naming if no explicit config
                    key_suffix = model_cli_name.replace('portkey-', '').upper().replace('-', '_')
                    env_var_name_fallback = f'PORTKEY_VIRTUAL_KEY_{key_suffix}'
                    key_fallback_val = os.getenv(env_var_name_fallback)
                    if key_fallback_val:
                         portkey_virtual_keys_loaded[model_cli_name] = key_fallback_val
                         print(f"Loaded virtual key for '{model_cli_name}' using fallback env var name '{env_var_name_fallback}'.")
                    else:
                        print(f"Warning: No virtual key found for '{model_cli_name}' using explicit config or fallback naming.")

            except KeyError:
                print(f"Warning: Profile for model '{model_cli_name}' not found in llm_config.json. Cannot determine Portkey virtual key.")
            except Exception as e:
                print(f"Error processing model '{model_cli_name}' for Portkey keys: {e}")
    
    except ImportError:
        print("Warning: python-dotenv or llm.utils.config not available. Cannot load .env or llm_config.json for Portkey keys.")
    except Exception as e:
        print(f"Error during initial Portkey key setup: {e}")

    game_list_to_process = []
    if args.games: # If user provided specific games
        game_list_to_process = args.games
        print(f"Processing user-specified games: {game_list_to_process}")
    else:
        # Default: Scan for all games in gym_gvgai/envs/games/ and exclude testgames
        print("No specific games provided via --games. Scanning for all available games (excluding 'testgame' variants)...")
        # Handle both running from root directory and from project directory
        possible_paths = [
            Path("../gym_gvgai/envs/games/"),  # When running from project/
            Path("gym_gvgai/envs/games/"),  # When running from root
        ]
        
        all_games_dir = None
        for path in possible_paths:
            if path.is_dir():
                all_games_dir = path
                break
        
        if all_games_dir is None:
            print(f"Error: Could not find games directory. Tried: {[str(p) for p in possible_paths]}. Please specify games via --games. Exiting.")
            return
        if all_games_dir.is_dir():
            for game_path in sorted(all_games_dir.iterdir()):
                if game_path.is_dir(): # Each game is a directory like 'aliens_v0'
                    game_name = game_path.name
                    if "testgame" not in game_name.lower():
                        game_list_to_process.append(game_name)
            if game_list_to_process:
                print(f"Found {len(game_list_to_process)} games to process (excluding testgames): {game_list_to_process}")
            else:
                print(f"Warning: No games found in {all_games_dir} after excluding testgames. Exiting.")
                return
        else:
            print(f"Error: Default games directory {all_games_dir} not found. Please specify games via --games. Exiting.")
            return
            
    # Prepare game list based on args (resume_game, reverse apply to the determined list)
    if args.resume_game:
        try:
            start_index = game_list_to_process.index(args.resume_game)
            game_list_to_process = game_list_to_process[start_index:]
            print(f"Resuming from game: {args.resume_game}. Games to process: {game_list_to_process}")
        except ValueError:
            print(f"Warning: Resume game '{args.resume_game}' not found in the game list. Processing all determined games.")
    
    if args.reverse:
        game_list_to_process.reverse()
        print(f"Processing games in reverse order. Order: {game_list_to_process}")

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
    
    # Using ProcessPoolExecutor for parallelism
    try:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(run_single_game_task, **task) for task in tasks]
            
            completed_count = 0
            total_tasks = len(tasks)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    completed_count += 1
                    print(f"[{completed_count}/{total_tasks}] Task completed: {result}")
                except Exception as e:
                    completed_count += 1
                    print(f"[{completed_count}/{total_tasks}] !!! Task generated an exception: {e} !!!")
                    import traceback
                    traceback.print_exc()
    except KeyboardInterrupt:
        print("\n!!! Keyboard interrupt received (Ctrl+C) !!!")
        print("Attempting to cancel remaining tasks...")
        
        # Cancel all pending futures
        for future in futures:
            if not future.done():
                future.cancel()
        
        # Wait a bit for graceful shutdown
        import time
        time.sleep(2)
        
        print("Shutdown complete. Some tasks may have been interrupted.")
        return

    print(f"\n=== All {total_tasks} tasks processed ===")

if __name__ == "__main__":
    main()
