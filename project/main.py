import os
import json
import pandas as pd
import gym_gvgai as gvgai
from datetime import datetime

from llm.agent.llm_agent import LLMPlayer, LLMPlanner, LLMEvaluator
from llm.agent.llm_translator import LLMTranslator
from llm.utils.agent_components import (
    create_directory, 
    show_state_gif, 
    generate_mapping_and_ascii, 
    extract_avatar_position_from_state
)
from llm.utils.vgdl_utils import load_level_map, load_vgdl_rules

def run_full_loop(game_list):
    llm_lst = ['openai','deepseek','qwen']
    # 从.env文件中读取Portkey的virtual keys
    portkey_virtual_keys = {
        'o3-mini': os.getenv('PORTKEY_VIRTUAL_KEY_O3_MINI'),
        'gpt-4o-mini': os.getenv('PORTKEY_VIRTUAL_KEY_GPT4O_MINI'),
        'vertex-ai': os.getenv('PORTKEY_VIRTUAL_KEY_VERTEX_AI')
    }
    
    for env_name in game_list:
        for mode in ['zero-shot', 'contextual']:
            # # 处理非Portkey模型
            # for model in llm_lst:
            #     run_single_game(env_name, mode, model)
            
            # 处理Portkey的三个模型
            for model_name, virtual_key in portkey_virtual_keys.items():
                # 修改.env文件中的PORTKEY_VIRTUAL_KEY
                # update_portkey_virtual_key(virtual_key)
                # 使用模型名称作为目录名
                run_single_game(env_name, mode, f"portkey-{model_name}")

# def update_portkey_virtual_key(virtual_key):
#     """更新PORTKEY_VIRTUAL_KEY环境变量"""
#     env_path = "/.env"
    
#     # 读取.env文件
#     with open(env_path, 'r') as f:
#         lines = f.readlines()
    
#     # 修改PORTKEY_VIRTUAL_KEY行
#     for i, line in enumerate(lines):
#         if line.startswith('PORTKEY_VIRTUAL_KEY='):
#             lines[i] = f'PORTKEY_VIRTUAL_KEY={virtual_key}\n'
#             break
    
#     # 写回.env文件
#     with open(env_path, 'w') as f:
#         f.writelines(lines)
    
#     # 更新环境变量
    os.environ['PORTKEY_VIRTUAL_KEY'] = virtual_key

def run_single_game(game_name: str, mode: str, model: str):
    # 处理portkey-{model_name}格式的模型名称
    actual_model = model
    if model.startswith('portkey-'):
        actual_model = 'portkey'
    
    env_name = 'gvgai-' + game_name[:-3] + '-lvl2-v0'
    print(f"\n==== Starting game: {env_name} | Mode: {mode} | Model: {model} ====")

    env = None
    gif_saver = None
    player = None
    planner = None
    evaluator = None
    step_count = 0
    ascii_state = None
    action_history = []
    current_position = []
    current_state = ''
    sprite_map = {}
    evaluation = ''  # 累积 evaluator 输出

    try:
        # === 初始化环境和数据 ===
        env = gvgai.make(env_name)
        vgdl_rules = load_vgdl_rules(env_name)
        level_layout = load_level_map(env_name, 1)

        state = env.reset()
        _, _, _, info = env.step(0)
        ascii_state = info['ascii']

        sprite_map, current_state, _ = generate_mapping_and_ascii(
            state_str=ascii_state,
            vgdl_text=vgdl_rules
        )

        # === Translator（用对应 model）===
        translator = LLMTranslator(model_name=actual_model)
        translation = translator.translate(vgdl_rules=vgdl_rules, level_layout=level_layout)
        # 不重复包含原始规则，只使用翻译后的规则
        translation = f"Game rules in natural language:\n{translation}"

        # === Planner ===
        planner = LLMPlanner(model_name=actual_model, vgdl_rules=vgdl_rules)

        # 创建目录
        current_date = datetime.now().strftime("%Y-%m-%d")
        dir = create_directory(f"dual_reward{current_date}/{game_name}_{mode}_{model}")

        # === Player ===
        player = LLMPlayer(
            model_name=actual_model,
            env=env,
            vgdl_rules=translation,
            initial_state=level_layout,
            extra_prompt=None,
            mode=mode,
            rotate_state=False,  # 禁用rotate_state
            expand_state=True,   # 启用expand_state
            log_dir=dir
        )

        # === Evaluator ===
        evaluator = LLMEvaluator(model_name=actual_model)

        gif_saver = show_state_gif()
        gif_saver(env)


        # === 主循环 ===
        done = False
        while not done:
            print(f"== {env_name} | Mode: {mode} | Model: {model} | Step {step_count} ==")

            sprite_map, current_state, _ = generate_mapping_and_ascii(
                state_str=ascii_state,
                vgdl_text=vgdl_rules,
                existing_mapping=sprite_map
            )

            current_position = extract_avatar_position_from_state(
                ascii_lines=current_state,
                sprite_to_char=sprite_map,
                flip_vertical=False
            )

            # === Planning ===
            # print(current_position,sprite_map,current_state)
            plan = planner.query(
                prompt=translation + evaluation,
                current_state=current_state,
                action_history=action_history,
                current_position=current_position,
                sprite_mapping=sprite_map
            )

            # === Player 选动作 ===
            action = player.select_action(
                current_state=current_state,
                extra_prompt=evaluation,
                current_position=current_position,
                sprite_map=sprite_map,
                plan=plan
            )
            action_history.append(action)

            # === 环境步进 ===
            _, reward, done, info = env.step(action)

            player.update(action=action, reward=reward, winner=info.get('winner', None))

            ascii_state = info['ascii']

            # === 评估当前行为 ===
            evaluation = evaluator.query(
                action_taken=action,
                current_state=ascii_state,
                reward=reward,
                done=done,
                current_position=current_position,
                sprite_mapping=sprite_map
            )

            gif_saver(env)
            step_count += 1

    finally:
        if env:
            env.close()
        if player:
            player.save_logs()
        if gif_saver:
            gif_saver.save(dir)
        if player:
            print(f"[{mode.upper()}] [{model}] Game finished in {step_count} steps. Total reward: {player.total_reward}")
            output_dir = os.path.join("benchmark_results", f"{env_name.replace(':', '_')}_{mode}_{model}")
            player.export_analysis(output_dir)

if __name__ == "__main__":
    difficulty_df = pd.read_csv("gvgai_game_difficulty_scores.csv")
    difficulty_groups = {
        label: difficulty_df[difficulty_df['difficulty_label'] == label]['game_name'].tolist()
        for label in difficulty_df['difficulty_label'].unique()
    }

    selected_games = [ 'boulderdash_v0', 'realsokoban_v0', 'escape_v0', 'sokoban_v0']
    print("Selected games:", selected_games)
    run_full_loop(selected_games)
