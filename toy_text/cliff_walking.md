# 🧗 CliffWalking 环境使用指南（CliffWalking-v0）

CliffWalking 是 OpenAI Gym 提供的经典强化学习环境，用于研究代理在**高风险路径选择中的行为策略**。该环境以 4×12 的网格表示 Cliff Walking 问题，训练目标是智能体从起点移动至终点，同时避免掉入悬崖区，尽可能缩短路径。

---

## 🗺️ 1. 环境概述（Environment Description）

**网格结构：**
- **尺寸：** 4 行 × 12 列，共 48 个状态
- **起点（S）：** 左下角 `(3, 0)`，状态编号为 `36`
- **终点（G）：** 右下角 `(3, 11)`，状态编号为 `47`
- **悬崖区域（C）：** 第 4 行，第 1~10 列（状态编号 `37~46`），坠落即重置

```

+----+----+----+----+----+----+----+----+----+----+----+----+
\|    |    |    |    |    |    |    |    |    |    |    |    |
+----+----+----+----+----+----+----+----+----+----+----+----+
\|    |    |    |    |    |    |    |    |    |    |    |    |
+----+----+----+----+----+----+----+----+----+----+----+----+
\|    |    |    |    |    |    |    |    |    |    |    |    |
+----+----+----+----+----+----+----+----+----+----+----+----+
\| S  | C  | C  | C  | C  | C  | C  | C  | C  | C  | C  | G  |
+----+----+----+----+----+----+----+----+----+----+----+----+

````

> 图示符号：S = Start, C = Cliff, G = Goal

---

## ⚙️ 2. 环境参数（Core Parameters）

### 🎯 状态空间（Observation Space）

- 类型：`Discrete(48)`
- 表示：每个状态是一个 0~47 的唯一编号，对应网格位置

### 🧭 动作空间（Action Space）

- 类型：`Discrete(4)`
- 动作编号与方向映射如下：

| 编号 | 方向 | 坐标变换   |
|------|------|------------|
| 0    | 上   | (-1, 0)    |
| 1    | 右   | (0, +1)    |
| 2    | 下   | (+1, 0)    |
| 3    | 左   | (0, -1)    |

---

## 🏆 3. 奖励机制（Reward Function）

| 行为类型       | 奖励值 | 是否终止 |
|----------------|--------|----------|
| 正常移动       | -1     | 否 ❌     |
| 坠落悬崖       | -100   | 是 ✅（重置） |
| 成功到达终点   | 0      | 是 ✅     |

---

## ⚡ 4. 安装与使用（Installation & Basic Usage）

### 🔧 安装依赖

```bash
pip install gym numpy
````

### 🧪 环境初始化与交互示例

```python
import gym

env = gym.make('CliffWalking-v0')

print("状态空间:", env.observation_space)  # Discrete(48)
print("动作空间:", env.action_space)      # Discrete(4)

state = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    print(f"状态: {state}, 动作: {action}, 奖励: {reward}, 新状态: {next_state}")
    state = next_state

print("回合总奖励:", total_reward)
env.close()
```

---

## 🧠 5. Q-Learning 训练示例（Q-learning Example）

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.9, epsilon=0.1):
        self.Q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.Q_table[state])

    def update(self, state, action, reward, next_state):
        td_target = reward + self.gamma * np.max(self.Q_table[next_state])
        td_error = td_target - self.Q_table[state][action]
        self.Q_table[state][action] += self.lr * td_error
```

### 🔁 训练循环

```python
agent = QLearningAgent(n_states=48, n_actions=4)

for episode in range(1000):
    state = env.reset()
    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        if done:
            break
```

---

## 📊 6. 可视化（Rendering）

```python
env = gym.make('CliffWalking-v0', render_mode="human")
state = env.reset()
env.render()
```

### 渲染图例：

| 元素     | 表示色 |
| ------ | --- |
| 🟦 智能体 | 蓝色  |
| 🟩 终点  | 绿色  |
| 🟥 悬崖  | 红色  |
| ⬜ 路径   | 白色  |

> 若图形窗口未出现，请确保系统支持 GUI，并安装 `pygame`

---

## 📈 7. 最优策略示例（Optimal Path）

### 路径动作序列：

→ → → → → → → → → ↓ →

智能体通常采取绕开底部悬崖边缘的策略，从起点向右移动至倒数第二列，再向下抵达终点。

---

## 📚 8. 延伸阅读（References）

* [Gym 官方文档 - CliffWalking-v0](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/)
* [强化学习基础：Q-Learning 示例](https://spinningup.openai.com/)
* \[清华大学《强化学习导论》课程笔记]
* \[Sutton & Barto. Reinforcement Learning: An Introduction]

---

> 📌 本文档结合环境定义、算法实现、可视化渲染等要素，适用于强化学习课程教学、策略验证、算法基准测试等场景。
