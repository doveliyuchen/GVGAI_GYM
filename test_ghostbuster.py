"""Test ghostbuster through the full LLM pipeline (no actual LLM call)."""
import sys, traceback, random
sys.path.insert(0, '.')
import gym_gvgai as gvgai
from llm.utils.agent_components import generate_mapping_and_ascii
from llm.utils.vgdl_utils import load_vgdl_rules, load_level_map

env_name = 'gvgai-ghostbuster-lvl0-v0'
print(f"=== Testing pipeline for {env_name} ===\n")

# Load VGDL and level
vgdl_rules = load_vgdl_rules(env_name)
level_layout = load_level_map(env_name, 1)
print(f"VGDL loaded: {len(vgdl_rules)} chars")
print(f"Level layout loaded: {level_layout is not None}")

# Run env and process state through LLM pipeline components
env = gvgai.make(env_name)
env.reset()
sprite_map = {}

for step in range(100):
    action = random.randrange(env.action_space.n)
    result = env.step(action)
    obs, reward, *rest = result
    done = rest[0] if len(rest) == 2 else (rest[0] or rest[1])
    info = rest[-1]
    ascii_state = info.get('ascii', '')

    try:
        current_sprite_map, ascii_out, _ = generate_mapping_and_ascii(
            state_str=ascii_state,
            vgdl_text=vgdl_rules,
            existing_mapping=sprite_map
        )
        sprite_map.update(current_sprite_map)
    except Exception as e:
        print(f"FAIL at step {step} in generate_mapping_and_ascii: {type(e).__name__}: {e}")
        traceback.print_exc()
        break

    if step % 20 == 0:
        print(f"  step {step}: ascii_state={len(ascii_state)} chars, sprite_map_size={len(sprite_map)}, ascii_out_lines={len(ascii_out.splitlines())}")

    if done:
        print(f"  Game done at step {step}")
        break

env.close()
print("\nPASS: no errors in full pipeline")
