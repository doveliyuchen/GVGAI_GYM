import matplotlib.pyplot as plt
from collections import Counter
import json
import os
import csv

def generate_reward_report(reflection_manager, output_dir, winner=None):
    """Generate and save reward trend and action distribution from reflection_manager."""
    reward_history = [entry["reward"] for entry in reflection_manager.step_log]
    action_history = [entry["action"] for entry in reflection_manager.step_log]

    print(f"\n=== Game analysis ===")
    print(f"Total steps: {len(reflection_manager.step_log)}")
    print(f"Total reward: {sum(reward_history)}")
    if winner is not None:
        print(f"Winner: {winner}")

    plt.figure(figsize=(12, 5))

    # Reward trend
    plt.subplot(121)
    plt.plot(reward_history)
    plt.title("Reward Trend")

    # Action distribution
    plt.subplot(122)
    action_dist = Counter(action_history)
    plt.bar(action_dist.keys(), action_dist.values())
    plt.title("Action Distribution")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "game_analysis.png"))
    plt.close()

def save_step_metrics_json(path, step_flags, positions, key="meaningful", winner=None):
    """Save boolean list and avatar positions as step-wise metrics json."""
    if len(step_flags) == 0:
        step_ratio = 0.0
    else:
        step_ratio = sum(step_flags) / len(step_flags)
    metrics = {
        f"{key}_steps": step_flags,
        f"{key}_step_ratio": step_ratio,
        "avatar_positions": [[pos] for pos in positions] # Transform into n*1 shape
    }
    if winner is not None:
        metrics["winner"] = winner
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def analyze_meaningful_steps(states, step_log):
    def extract_avatar_pos(state):
        # Removed vertical flipping logic and debug print
        for y, row in enumerate(state.splitlines()): # Split lines here directly
            for x, ch in enumerate(row):
                if 'a' in ch.lower() or 'avatar' in ch.lower():
                    return (y, x) # Return raw y
        return None

    def detect_entity_disappearance(s_t, s_tp1):
        def flatten_exclude_avatar(state):
            return [ch for row in state for ch in row if not ('a' in ch.lower() or 'avatar' in ch.lower())]
        count_t = Counter(flatten_exclude_avatar(s_t))
        count_tp1 = Counter(flatten_exclude_avatar(s_tp1))
        return any(count_tp1.get(k, 0) < count_t[k] for k in count_t)

    flags = []
    pos_prev = None

    max_steps = min(len(step_log), len(states) - 1)

    for t in range(max_steps):
        entry = step_log[t]
        s_t = states[t]
        s_tp1 = states[t + 1]
        a_t = entry["action"]
        r_tp1 = entry["reward"]

        pos_t = extract_avatar_pos(s_t)
        pos_tp1 = extract_avatar_pos(s_tp1)

        reward_triggered = r_tp1 != 0
        entity_triggered = detect_entity_disappearance(s_t, s_tp1)
        canceling = (pos_prev == pos_tp1 and pos_t != pos_tp1)

        if a_t == 0 or (pos_t == pos_tp1 and not reward_triggered and not entity_triggered) or canceling:
            meaningful = False
        else:
            meaningful = True

        flags.append(meaningful)
        pos_prev = pos_t
        # Removed debug prints for individual step positions

    positions = [extract_avatar_pos(states[t]) for t in range(max_steps)] # Collect pos_t for each step
    # Removed debug prints for the final list
    return flags, sum(flags) / len(flags) if flags else 0.0, positions


def save_step_metrics_csv(states, step_log, output_path, winner=None):
    """Save full step-by-step metrics to CSV for analysis."""

    def extract_avatar_pos(state):
        # Removed vertical flipping logic and debug print
        for y, row in enumerate(state.splitlines()): # Split lines here directly
            for x, ch in enumerate(row):
                if 'a' in ch.lower() or 'avatar' in ch.lower():
                    return (y, x) # Return raw y
        return None

    def detect_entity_disappearance(s_t, s_tp1):
        def flatten_exclude_avatar(state):
            return [ch for row in state for ch in row if not ('a' in ch.lower() or 'avatar' in ch.lower())]
        count_t = Counter(flatten_exclude_avatar(s_t))
        count_tp1 = Counter(flatten_exclude_avatar(s_tp1))
        return any(count_tp1.get(k, 0) < count_t[k] for k in count_t)

    rows = []
    pos_prev = None

    max_steps = min(len(step_log), len(states) - 1)
    
    # 确定所有可能的字段
    fieldnames = ["step", "action", "reward", "avatar_pos_before", "avatar_pos_after", 
                 "reward_triggered", "entity_triggered", "meaningful"]
    if winner is not None:
        fieldnames.append("winner")

    for t in range(max_steps):
        entry = step_log[t]
        s_t = states[t]
        s_tp1 = states[t + 1]
        a_t = entry["action"]
        r_tp1 = entry["reward"]

        pos_t = extract_avatar_pos(s_t)
        pos_tp1 = extract_avatar_pos(s_tp1)

        reward_triggered = r_tp1 != 0
        entity_triggered = detect_entity_disappearance(s_t, s_tp1)
        canceling = (pos_prev == pos_tp1 and pos_t != pos_tp1)

        meaningful = not (
            a_t == 0 or
            (pos_t == pos_tp1 and not reward_triggered and not entity_triggered) or
            canceling
        )

        row_data = {
            "step": t,
            "action": a_t,
            "reward": r_tp1,
            "avatar_pos_before": pos_t,
            "avatar_pos_after": pos_tp1,
            "reward_triggered": reward_triggered,
            "entity_triggered": entity_triggered,
            "meaningful": meaningful
        }
        
        # 如果有winner信息，添加到每一行
        if winner is not None:
            row_data["winner"] = winner if t == max_steps - 1 else ""
            
        rows.append(row_data)

        pos_prev = pos_t

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_full_analysis_report(
    reflection_manager,
    states,
    output_dir,
    winner=None
):
    """Master function to generate all outputs from reflection_manager + states."""
    os.makedirs(output_dir, exist_ok=True)

    generate_reward_report(reflection_manager, output_dir, winner)

    step_log = reflection_manager.step_log
    step_flags, _, positions = analyze_meaningful_steps(states, step_log) # Capture positions
    save_step_metrics_json(os.path.join(output_dir, "step_metrics.json"), step_flags, positions, winner=winner) # Pass positions

    save_step_metrics_csv(
        states=states,
        step_log=step_log,
        output_path=os.path.join(output_dir, "step_metrics.csv"),
        winner=winner
    )
