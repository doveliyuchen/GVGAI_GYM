# GVGAI Benchmark for LLMs

This project is an ongoing effort to create a **benchmark for Large Language Models (LLMs)** within the **Generic Video Game AI (GVGAI)** framework. Originally forked from [GVGAI Gym](https://github.com/rubenrtorrado/GVGAI_GYM), this repository has been adapted to evaluate how LLMs perform in video game environments described using the **Video Game Description Language (VGDL)**.

The GVGAI framework has been widely used since 2019 for competitions like the **GVGAI Learning Competition** and research in reinforcement learning and planning agents. While traditional approaches relied on access to forward models or reinforcement learning, this project explores how LLMs can interact with game states—both text-based and image-based—to make decisions in these environments.

For more details about the original competition rules, rankings, and related resources, visit the [AI in Games website](http://www.aingames.cn/), maintained by [Hao Tong](https://github.com/HawkTom) and [Jialin Liu](https://github.com/ljialin).

---

## Key Features

- **LLM Integration**: The framework now supports sending **text-based states** and **image-based states** to LLMs, enabling them to interpret and act within GVGAI games.
- **Benchmarking**: Evaluate LLM performance across various GVGAI game levels, comparing their decision-making capabilities to traditional planning and RL agents.
- **Ongoing Development**: This project is actively being developed to refine the benchmark and explore new ways to integrate LLMs into game AI research.

---


## Installation

### Option 1: Using Docker
Follow the step-by-step [guidelines](https://github.com/SUSTechGameAI/GVGAI_GYM/blob/master/docs/Guidelines-Docker-GVGAI-RLbaselines.md) to set up the framework with Docker (GPU or CPU support).

### Option 2: Manual Setup
1. Clone this repository to your local machine.
2. Install dependencies using `pip install -e <package-location>`.
3. Install a Java compiler (e.g., `sudo apt install openjdk-9-jdk-headless`).

---

## Requirements

- **Anaconda**: Version published after 2019.10 is recommended.
- **Java**: JDK 9 is recommended.
- **Python**: Python 3 (3.6 or 3.7 recommended). **Python 2 is not supported.**

---

# Resources

[GVGAI website](http://www.gvgai.net)

[original GVGAI-Gym (master branch)](https://github.com/rubenrtorrado/GVGAI_GYM) 

[Demo video on YouTube](https://youtu.be/O84KgRt6AJI)

[AI in Games website for more about competition updates](http://www.aingames.cn/#sources)

[*Deep Reinforcement Learning for General Video Game AI*](https://arxiv.org/abs/1806.02448) published at IEEE CIG2018

---
# References

1. [G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, “Openai gym,” 2016.](https://github.com/openai/gym)
2. [A. Hill, A. Raffin, M. Ernestus, A. Gleave, A. Kanervisto, R. Traore, P. Dhariwal, C. Hesse, O. Klimov, A. Nichol, M. Plappert, A. Radford, J. Schulman, S. Sidor, and Y. Wu, “Stable baselines,” https://github.com/hill-a/stable-baselines, 2018.](https://github.com/hill-a/stable-baselines)
3. [R. R. Torrado, P. Bontrager, J. Togelius, J. Liu, and D. Perez-Liebana, “Deep reinforcement learning for general video game AI,” in Computational Intelligence and Games (CIG), 2018 IEEE Conference on. IEEE, 2018.](https://github.com/rubenrtorrado/GVGAI_GYM)
4. [D Perez-Liebana, J Liu, A Khalifa, RD Gaina, J Togelius, SM Lucas, "General video game AI: A multitrack framework for evaluating agents, games, and content generation algorithms," in IEEE Transactions on Games, 11(3), 195-214.](https://arxiv.org/pdf/1802.10363)
