# 🎯 CartPole 平衡小车系统说明文档

本项目从零实现了经典控制环境 **CartPole**，完全遵循 OpenAI Gym 接口规范，并使用 **Pygame** 实现高质量的图形渲染，适用于强化学习训练、控制算法测试与物理建模演示。

---

## 🧩 1. 环境描述（Environment Overview）

CartPole 是强化学习领域最常见的控制问题之一。一个带有自由铰接杆的小车在一维轨道上移动，智能体需通过施加 **左右方向的力**，保持杆子在垂直方向平衡。

```

+------------------------------------------------------+
\|                                                      |
\|         |       ⚖️                                   |
\|      ← 🔲 →      |                                   |
\|     Cart      Red pole (stick)                       |
\|                                                      |
+------------------------------------------------------+

````

### 📌 环境目标

保持杆子 **不倒下**，即尽可能长时间维持其在垂直方向的动态平衡状态。

---

## 🧮 2. 状态空间（Observation Space）

每个状态为 4 个连续变量组成的向量：

| 变量名称   | 含义描述               | 数值范围         |
|------------|------------------------|------------------|
| 车位置     | 小车在轨道上的水平位置 | [-2.4, 2.4]      |
| 车速度     | 小车的水平速度         | (-∞, +∞)         |
| 杆角度     | 杆子偏离垂直的角度     | [-0.418, +0.418] |
| 杆角速度   | 杆子当前角速度         | (-∞, +∞)         |

> 单位角度范围约为 ±24°，超出则视为失败。

---

## 🎮 3. 动作空间（Action Space）

动作为离散的两个方向：

| 动作编号 | 含义          | 作用力方向 |
|----------|---------------|------------|
| 0        | 向左施加力    | ←          |
| 1        | 向右施加力    | →          |

默认施力大小为 **±10.0 牛顿**。

---

## ❌ 4. 终止条件（Episode Termination）

回合将在以下任一条件满足时终止：

- 小车位置超出 ±2.4 的轨道边界
- 杆子角度超过 ±12°（即 ±0.418 弧度）
- 仿真步数超过 **1000 步**（默认最大步长）

---

## 🏆 5. 奖励机制（Reward System）

- 每个有效步给予奖励 **+1**
- 若回合终止（失败），无额外惩罚（reward = 0）

该设置鼓励智能体维持尽可能 **长时间的平衡控制**。

---

## ⚙️ 6. 物理系统参数（Physics Configuration）

| 参数名称         | 数值    | 单位   |
|------------------|---------|--------|
| 重力加速度       | 9.8     | m/s²   |
| 小车质量         | 1.0     | kg     |
| 杆子质量         | 0.1     | kg     |
| 杆子半长         | 0.5     | m      |
| 作用力大小       | 10.0    | N      |
| 时间步长         | 0.02    | s      |

> 模拟以牛顿运动定律为基础，使用欧拉积分更新状态。

---

## 🚀 7. 安装与运行（Installation & Execution）

### 🔧 安装依赖

```bash
pip install numpy gym pygame
````

### ▶️ 运行主程序

```bash
python cartpole.py
```

> 程序将自动以 **随机策略** 运行环境，并显示杆子的动态变化。

---

## 🧠 8. 自定义使用示例（Example Usage）

```python
import cartpole

env = cartpole.CartPoleEnv()
state = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    env.render()

    if done:
        state = env.reset()

env.close()
```

---

## 🧱 9. 关键模块组件（Core Modules）

### 🔹 `CartPoleEnv` 类

实现了标准 Gym 接口，具备以下方法：

* `__init__()`：初始化系统参数、状态空间和图形引擎
* `reset()`：将环境重置为初始状态
* `step(action)`：执行动作并更新状态
* `render(mode)`：使用 Pygame 渲染当前场景
* `close()`：释放资源、关闭窗口

### 🔹 物理引擎（Physics Engine）

采用欧拉法近似计算动态更新：

```python
# 伪公式表达
θ_acc = (g * sinθ - cosθ * temp) / (l * (4/3 - m * cos²θ / total_mass))
x_acc = temp - m * l * θ_acc * cosθ / total_mass
```

---

## 🖼️ 10. 可视化设计（Pygame Rendering）

图形元素说明：

* **蓝色矩形**：小车
* **红色线段**：杆子
* **灰色轨道线**：运行轨迹
* **红字提示**：终止状态出现时提示文字

> 渲染帧率可调，适配演示与训练场景

---

## 📦 11. 应用场景（Application Scenarios）

本环境可广泛用于：

* 强化学习算法训练（如 DQN、PPO、A2C）
* 控制系统教学与演示
* 机器人运动稳定性测试
* 经典动力学系统仿真

---

## 🔭 12. 可扩展方向（Possible Extensions）

* ✅ 实现双杆系统（Double CartPole）
* ✅ 引入高斯噪声模拟现实不确定性
* ✅ 添加 CLI / 字符图渲染模式
* ✅ 接入 TensorBoard / Weights & Biases 记录训练数据

---

## 📚 参考资料（References）

* [OpenAI Gym 官方 CartPole 文档](https://gym.openai.com/envs/CartPole-v1/)
* [Sutton & Barto《强化学习导论》](http://incompleteideas.net/book/the-book.html)
* [MIT OCW: Underactuated Robotics - CartPole Lecture](https://underactuated.mit.edu/)

> 本项目旨在提供一个**可扩展、可视化、可复用**的 CartPole 学习环境框架。

---
