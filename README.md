# GVGAI-JPype: LLM Benchmark for General Video Game AI

A benchmark for evaluating **Large Language Models (LLMs)** on the [GVGAI](http://www.gvgai.net) game suite. Games are defined in **VGDL** (Video Game Description Language). This fork replaces the original subprocess + TCP-socket architecture with a **JPype direct-JVM bridge**, eliminating port management and serialisation overhead while keeping the Gymnasium API identical.

Originally forked from [GVGAI Gym](https://github.com/rubenrtorrado/GVGAI_GYM).

---

## Architecture

```
Python (gymnasium env)
        │
        │  JPype  (direct JVM call-in, no sockets)
        ▼
Java GVGAIBridge
        ├── reset(levelIdx, seed)
        ├── step(actionName)
        ├── renderToBytes()           →  pixel image (PNG via Java2D)
        ├── getObservationString()    →  discrete grid (itypeKey symbols)
        └── getGameScore / isGameOver / getWinner
```

The JVM starts once per Python process. All `gvgai.make()` calls within that process share it.

---

## State Representation

Every `step()` returns **both** representations simultaneously:

| Field | Type | Description |
|---|---|---|
| `obs` | `(H, W, 4)` uint8 | **Pixel** — RGBA image rendered by Java2D. This is what `observation_space` describes. |
| `info["ascii"]` | string | **Discrete** — grid of `itypeKey` sprite-type tokens; rows by `\n`, columns by `,` |
| `info["grid"]` | numpy object array | 2-D parsed version of `info["ascii"]` |

The LLM agent uses `info["ascii"]` as its primary input (`text` mode). The pixel `obs` is used for GIF recording and `vision`/`multimodal` prompting.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.11 | conda env `gvgai_jpype` |
| Java JDK | 9+ | `javac` must be on PATH |
| numpy | < 2.0 | numpy 2.x causes a DLL crash on Windows; use 1.26.4 |
| gymnasium | 1.x | |
| jpype1 | 1.x | |
| pillow, psutil, python-dotenv | any recent | |

---

## Installation

### 1. Build Java

```bash
git clone <this-repo>
cd GVGAI_jpype
python build.py
```

Compiles all sources under `gym_gvgai/envs/gvgai/src/` into `gym_gvgai/envs/gvgai/GVGAI_Build/`.

### 2. Install Python package

```bash
conda activate gvgai_jpype
pip install -e .
```

> **Windows gotcha** — if you see exit code `-1066598273` on any import, numpy 2.x is installed. Fix with:
> ```bash
> pip install "numpy<2.0" --force-reinstall
> ```

### 3. Configure API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
PORTKEY_API_KEY=...               # NYU AI Gateway
PORTKEY_VIRTUAL_KEY_VERTEX_AI=... # Gemini / Llama via Vertex
PORTKEY_VIRTUAL_KEY_O3_MINI=...
```

LLM profiles are defined in [`llm_config.json`](llm_config.json):

| Profile | Backend | Model |
|---|---|---|
| `4o-mini` | OpenAI | gpt-4o-mini |
| `gemini-pro` | Portkey → Vertex AI | gemini-2.5-pro |
| `gemini3-flash` | Gemini API | gemini-3-flash-preview |
| `deepseek` | DeepSeek | deepseek-chat |
| `deepseek-r3.2` | DeepSeek | deepseek-reasoner |
| `o3-mini` | Portkey → OpenAI | o3-mini |
| `llama3.1` | Portkey → Vertex AI | Llama-3.1-405B |
| `qwen3` | Ollama (local) | qwen3:32b |
| `vllm-local` | vLLM (local) | configurable |

---

## Running Experiments

```bash
python run_llm_gvgai.py \
    --games aliens zelda \
    --models gemini3-flash \
    --modes zero-shot contextual \
    --input_modes text \
    --max_steps 200 \
    --num_runs 1
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--games` | all 119 | Space-separated game names |
| `--models` | *(required)* | Profile names from `llm_config.json` |
| `--modes` | `zero-shot contextual` | Agent reasoning mode |
| `--input_modes` | `text` | `text` / `vision` / `multimodal` |
| `--max_steps` | 2000 | Max steps per episode |
| `--num_runs` | 1 | Runs per game × model × mode combination |
| `--force_rerun` | off | Re-run even if results already exist |
| `--resume_game` | — | Skip games before this one (alphabetical) |

Results are saved under `llm_agent_runs_output/<model>/<game>/<mode>/run_<n>/`.

Each run produces:
- `benchmark_analysis.json` — per-step log + summary statistics
- `gameplay.gif` — rendered frames

### Input modes

| Mode | What the LLM sees |
|---|---|
| `text` | ASCII symbolic grid only |
| `vision` | Screenshot (pixel image) only |
| `multimodal` | ASCII grid + screenshot |

### Quick sanity test (no LLM)

```python
import gym_gvgai as gvgai, random

env = gvgai.make('gvgai-aliens-lvl0-v0')
env.reset()
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(random.randrange(env.action_space.n))
    print(reward, info['winner'], len(info['ascii']))
env.close()
```

---

## Games

**119 playable games**, all confirmed working across all available levels. Environment IDs follow the pattern `gvgai-<game>-lvl<n>-v<version>`.

Most games have 5 levels (`lvl0`–`lvl4`). A few have fewer:

| Game | Levels |
|---|---|
| flower, invest, investdie, waferthinmints, waferthinmintsexit | 1 (lvl0 only) |
| bravekeeper, cec1, cec2, cec3, golddigger, greedymouse, sistersavior, trappedhero, treasurekeeper, waterpuzzle | 2 (lvl0–lvl1) |

Three directories (`testgame1_v0`, `testgame2_v0`, `testgame3_v0`) are internal engine tests and are automatically excluded by the experiment runner.

---

## Project Structure

```
GVGAI_jpype/
├── build.py                     # Java compilation script
├── run_llm_gvgai.py             # Main experiment runner
├── llm_config.json              # LLM profile definitions
├── .env                         # API keys (not committed)
│
├── gym_gvgai/
│   ├── __init__.py              # Gymnasium env registration (auto-discovers games)
│   └── envs/
│       ├── gvgai_env_jpype.py   # Gymnasium env — JPype backend
│       ├── games/               # 119 VGDL game definitions (+ 3 testgames)
│       └── gvgai/
│           ├── src/             # Java source: GVGAIBridge + VGDL engine
│           └── GVGAI_Build/     # Compiled .class files (generated by build.py)
│
└── llm/
    ├── agent/
    │   ├── llm_agent.py         # LLMPlayer: action selection, context, logging
    │   └── llm_translator.py    # VGDL → natural language translation
    ├── visual/                  # LLM client implementations (OpenAI, Gemini, vLLM, …)
    ├── utils/
    │   ├── config.py            # Profile loader (llm_config.json)
    │   ├── build_prompt.py      # Prompt construction
    │   ├── agent_components.py  # ASCII state generation, GIF saving, action parsing
    │   └── vgdl_utils.py        # VGDL / level file I/O
    └── analysis/                # Post-hoc metrics and visualisation
```

---

## References

1. Torrado et al., *Deep Reinforcement Learning for General Video Game AI*, IEEE CIG 2018. ([paper](https://arxiv.org/abs/1806.02448))
2. Perez-Liebana et al., *General Video Game AI: A Multitrack Framework*, IEEE Transactions on Games. ([paper](https://arxiv.org/pdf/1802.10363))
3. [Original GVGAI Gym](https://github.com/rubenrtorrado/GVGAI_GYM) — Ruben Rodriguez Torrado et al.
4. Brockman et al., *OpenAI Gym*, 2016.
