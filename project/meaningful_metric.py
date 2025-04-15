from collections import Counter

def extract_avatar_token(state):
    for row in state:
        for ch in row:
            if isinstance(ch, str) and ('a' in ch.lower() or 'avatar' in ch.lower()):
                return ch
    return None

def detect_entity_disappearance(state_t, state_tp1):
    avatar_token = extract_avatar_token(state_t)
    
    def flatten_and_filter(state, ignore_token):
        return [ch for row in state for ch in row if ch != ignore_token]

    flat_t = flatten_and_filter(state_t, avatar_token)
    flat_tp1 = flatten_and_filter(state_tp1, avatar_token)

    count_t = Counter(flat_t)
    count_tp1 = Counter(flat_tp1)

    for entity in count_t:
        if count_tp1.get(entity, 0) < count_t[entity]:
            return True
    return False

def extract_avatar_pos(state):
    for y, row in enumerate(state):
        for x, ch in enumerate(row):
            if 'a' in ch.lower() or 'avatar' in ch.lower():
                return (y, x)
    return None

def analyze_meaningful_steps(states, actions, rewards):
    """
    Parameters:
        states: List of 2D grid states (e.g., [['w','w','w'], ['w','A','w'], ...])
        actions: List of int actions (0 = nil, 1=left, etc.)
        rewards: List of rewards after each step
    Returns:
        meaningful_flags: List of booleans for each step
        ratio: meaningful step ratio
    """
    meaningful_flags = []
    pos_prev = None

    for t in range(len(actions)):
        s_t = states[t]
        s_tp1 = states[t + 1]
        a_t = actions[t]
        r_t = rewards[t + 1]  # reward after this action

        pos_t = extract_avatar_pos(s_t)
        pos_tp1 = extract_avatar_pos(s_tp1)

        # 判断是否机制触发
        triggered = r_t != 0 or detect_entity_disappearance(s_t, s_tp1)

        # 判断是否cancel动作
        canceling = (pos_prev == pos_tp1 and pos_t != pos_tp1)

        # 判断 meaningful
        if a_t == 0:
            meaningful = False
        elif pos_t == pos_tp1 and not triggered:
            meaningful = False
        elif canceling:
            meaningful = False
        else:
            meaningful = True

        meaningful_flags.append(meaningful)
        pos_prev = pos_t

    ratio = sum(meaningful_flags) / len(meaningful_flags)
    return meaningful_flags, ratio
