### 🚗 山地车环境（MountainCar Environment） - 自定义 Gymnasium 实现指南

---

## 🧭 1. 项目概述（Overview）

本项目自定义实现了经典强化学习任务 **MountainCar** 环境，兼容 OpenAI Gymnasium 接口，使用 `Python` 语言与 `PyGame` 进行可视化。

该环境模拟一个处于 **正弦形山谷底部的小车**，其目标是利用反复加速，最终爬上 **右侧高坡的目标位置**，体现动力学控制与策略优化的挑战。

---

## 🧩 2. 环境定义与核心特性

### ✅ 马尔可夫决策过程（MDP）建模：

* **状态空间**（Observation Space）：二维连续值 `[位置, 速度]`
* **动作空间**（Action Space）：离散动作集合 `{0: 向左, 1: 不动, 2: 向右}`
* **奖励函数**（Reward Function）：

  * 每步固定惩罚 `-1`
  * 达成目标位置奖励为 `0`，并终止回合

### 📐 参数约束：

| 项目   | 范围 / 值         | 描述        |
| ---- | -------------- | --------- |
| 位置   | \[-1.2, 0.6]   | 水平位置边界    |
| 速度   | \[-0.07, 0.07] | 最大速度限制    |
| 最大步数 | 200（可配置）       | 回合自动截断    |
| 动作集  | \[0, 1, 2]     | 离散：左、不动、右 |

---

## 🧠 3. 物理动力模型（Physics Model）

动力学遵循简化牛顿第二定律：

```python
velocity += force * scale_factor - gravity * cos(3 * position)
position += velocity
```

* **力值定义**：

  * 向左 `-1`，不动 `0`，向右 `+1`
* **重力常数**：`0.0025`
* **力缩放因子**：`0.001`
* **速度限制**：在 `[-0.07, 0.07]` 内剪裁
* **位置限制**：超出边界后强制剪裁，最左位置速度归零

---

## 🎮 4. 渲染特性（Visualization via PyGame）

使用 `PyGame` 实现细致动态渲染效果：

| 元素       | 说明                     |
| -------- | ---------------------- |
| ⛰ 地形     | 正弦曲线山谷，绿色背景，动态网格       |
| 🚗 小车模型  | 红色车身 + 灰色车窗，根据坡度自动旋转   |
| 🔰 目标区域  | 金色目标点，带有 "TARGET" 标签   |
| 📍 轨迹追踪  | 记录最近 100 步的小车位置，绘制蓝色路径 |
| ↗ 速度向量箭头 | 根据速度方向绘制箭头，表示运动趋势      |
| 🕒 信息面板  | 显示当前位置、速度和当前步数等实时数据    |

---

## 🧪 5. 安装与依赖（Installation）

### ✅ 环境要求：

* Python >= 3.8
* PyGame >= 2.1
* NumPy >= 1.21

### 🧰 安装依赖：

```bash
pip install numpy pygame
```

---

## 🧑‍💻 6. 使用示例（Usage Examples）

### 🟢 基础使用（随机策略）

```python
from mountain_car import MountainCarEnv

env = MountainCarEnv(render_mode="human")
state = env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    print(f"位置: {state[0]:.3f}, 速度: {state[1]:.5f}, 奖励: {reward}")

env.close()
```

---

### 🧠 简单策略示例：基于速度方向

```python
def simple_policy(state):
    pos, vel = state
    return 2 if vel >= 0 else 0

env = MountainCarEnv(render_mode="human")
state = env.reset()
done = False

while not done:
    env.render()
    action = simple_policy(state)
    state, reward, done, _ = env.step(action)

env.close()
```

---

## ⚙️ 7. 自定义参数（Configurable Parameters）

```python
env = MountainCarEnv(
    render_mode="human",     # 渲染模式：None / 'human' / 'rgb_array'
    max_steps=500,           # 每回合最大步数
    window_size=(1024, 768)  # 渲染窗口大小
)
```

---

## 📘 8. API 文档（API Reference）

### 类定义

```python
class MountainCarEnv:
    def __init__(self, render_mode=None, max_steps=200, window_size=(800, 600))
```

| 参数           | 类型               | 默认值        | 说明                     |
| ------------ | ---------------- | ---------- | ---------------------- |
| render\_mode | Optional\[str]   | None       | 渲染模式（human/rgb\_array） |
| max\_steps   | int              | 200        | 每回合步数限制                |
| window\_size | Tuple\[int, int] | (800, 600) | 窗口大小                   |

### 方法说明

```python
def reset() -> np.ndarray
"""
重置环境，返回初始状态
"""

def step(action: int) -> Tuple[np.ndarray, float, bool, dict]
"""
执行动作，返回：新状态、奖励、是否终止、额外信息
"""

def render() -> None
"""
渲染当前环境状态
"""

def close() -> None
"""
关闭并释放资源
"""
```

---

## 🧪 9. 测试覆盖（Verification Tests）

已测试功能包括：

* 左边界速度归零验证
* 成功到达目标状态终止
* 超过最大步数自动截断
* 奖励函数正确性校验
* 渲染可视化精度测试

运行示例：

```bash
python mountain_car.py
```

输出：

```
位置: -0.582, 速度: -0.0007, 奖励: -1
位置: -0.583, 速度: -0.0014, 奖励: -1
...
位置: 0.501, 速度: 0.0234, 奖励: 0
Episode completed in 182 steps!
```

---

## 🧾 10. 应用场景（Use Cases）

本环境适用于：

* ✅ 强化学习算法训练：Q-Learning、DQN、Sarsa 等
* ✅ 策略梯度方法测试：REINFORCE、PPO、A2C 等
* ✅ 演化学习验证：遗传算法、NEAT 等
* ✅ 控制策略设计：控制理论教学与实验
* ✅ 可视化教学：强化学习课程演示平台

---

## 📁 11. 项目结构（Project Files）

| 文件名               | 说明               |
| ----------------- | ---------------- |
| `mountain_car.py` | 主环境实现            |
| `README.md`       | 当前项目说明文档         |
| `assets/`         | 包含 PyGame 所需图像资源 |

---

## 🧑‍🔬 12. 开发标准与说明

* ✅ **从零实现**：完全自定义无依赖 Gym 模块
* ✅ **接口兼容**：符合 `gym.Env` 接口规范
* ✅ **风格一致**：遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
* ✅ **文档规范**：所有函数含 `PEP 257` 风格 DocString
* ✅ **边界测试**：包含位置极限、速度剪裁等单元测试

---

## 📜 13. 开源许可证（License）

本项目使用 MIT 开源协议，允许自由使用、修改和分发。