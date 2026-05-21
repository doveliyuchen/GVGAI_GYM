"""
Test every game (all actual levels, 20 steps) to find runtime crashes.
Level count is read from the actual game directory, not hardcoded.
"""
import sys, traceback, re
sys.path.insert(0, '.')
import gym_gvgai as gvgai
from pathlib import Path

games_dir = Path("gym_gvgai/envs/games")
games = sorted(
    d for d in games_dir.iterdir()
    if d.is_dir() and "testgame" not in d.name.lower()
)

print(f"Testing {len(games)} games (20 steps per level)...\n")

ok_games, failed_games = [], []

for game_path in games:
    game_dir = game_path.name
    base = game_dir.split('_v')[0]
    version = game_dir.split('_v')[-1]

    # Discover actual level files
    level_indices = set()
    for f in game_path.glob("*.txt"):
        m = re.search(r'lvl(\d+)', f.name, re.IGNORECASE)
        if m:
            level_indices.add(int(m.group(1)))
    levels = sorted(level_indices)
    if not levels:
        print(f"  SKIP {base} (no level files found)")
        continue

    game_fail = []
    for lvl in levels:
        env_id = f"gvgai-{base}-lvl{lvl}-v{version}"
        env = None
        try:
            env = gvgai.make(env_id)
            env.reset()
            for step in range(20):
                result = env.step(step % env.action_space.n)
                if len(result) == 5:
                    _, _, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    _, _, done, _ = result
                if done:
                    break
        except Exception as e:
            game_fail.append((env_id, type(e).__name__, str(e)[:120]))
            print(f"  FAIL {env_id}  ->  {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
        finally:
            if env is not None:
                try: env.close()
                except: pass

    if game_fail:
        failed_games.append((base, game_fail))
    else:
        ok_games.append(base)
        print(f"  OK   {base} (levels {levels})")

print(f"\n=== Results ===")
print(f"OK games:     {len(ok_games)}")
print(f"Failed games: {len(failed_games)}")
for gname, fails in failed_games:
    print(f"\n  {gname}:")
    for env_id, etype, emsg in fails:
        print(f"    {env_id}: {etype}: {emsg}")
