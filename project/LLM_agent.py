import openai
import gym
import gym_gvgai as gvgai
import numpy as np
import re

# 初始化 GVGAI 环境
env = gvgai.make("gvgai-aliens-lvl0-v0")
state = env.reset()
done = False

vgdl_level = [
    "1.............................",
    "000...........................",
    "000...........................",
    "..............................",
    "..............................",
    "..............................",
    "..............................",
    "....000......000000.....000...",
    "...00000....00000000...00000..",
    "...0...0....00....00...00000..",
    "................A............."
]


# 解析 VGDL Level 成 2D 数组
def parse_vgdl_level(vgdl_level):
    return np.array([list(row) for row in vgdl_level])

vgdl_grid = parse_vgdl_level(vgdl_level)
h, w = vgdl_grid.shape  # 计算网格大小
print("VGDL 关卡网格大小:", h, "x", w)

# 位置到 VGDL 符号的映射
def position_to_vgdl(state, vgdl_grid):
    state_np = np.array(state)  # 确保 state 是 numpy 数组
    h, w, _ = state_np.shape  # 提取游戏画面大小
    vgdl_output = []

    for i in range(min(h, vgdl_grid.shape[0])):  # 以较小尺寸匹配
        vgdl_row = []
        for j in range(min(w, vgdl_grid.shape[1])):
            vgdl_row.append(vgdl_grid[i, j])  # 直接用 VGDL 的字符
        vgdl_output.append("".join(vgdl_row))

    return "\n".join(vgdl_output)

# 转换 `state` 到 `VGDL`
vgdl_representation = position_to_vgdl(state, vgdl_grid)
print(vgdl_representation)  # 打印游戏状态



def query_llm(state_text):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are an AI game-playing agent."},
                  {"role": "user", "content": f"Game state:\n{state_text}\nChoose an action (integer)."}],
        temperature=0,
        max_tokens=10
    )

    action = response["choices"][0]["message"]["content"].strip()
    print("LLM Response:", action)

    try:
        return int(action)
    except ValueError:
        print("Error: LLM returned non-integer response.")
        return 0


while not done:
    vgdl_representation = position_to_vgdl(state, vgdl_grid)  # 解析游戏状态
    print(vgdl_representation)  # 可视化当前游戏状态

    response = query_llm(vgdl_representation)  # 传给 LLM
    print("LLM Response:", response)  # 打印 LLM 反馈，检查格式


    response_str = str(response)

    # 使用正则表达式提取 LLM 返回的第一个数字
    match = re.search(r"\d+", response_str)
    if match:
        action = int(match.group(0))
    else:
        print("LLM 生成了无效动作，默认 action = 0")
        action = 0  # 如果没有找到数字，默认执行动作 0

    state, reward, done, info = env.step(action)  # 执行 LLM 选的动作

env.close()

