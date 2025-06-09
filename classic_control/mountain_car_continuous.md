# 🚗 MountainCarContinuous 连续山地车环境使用说明（Gymnasium 兼容）

本环境模拟了一辆位于 **正弦形谷地** 中的小车，目标是通过施加 **连续控制力**，驱动车辆冲上右侧高坡顶端。

本环境遵循 Gymnasium 接口规范，适用于强化学习训练、控制系统建模与动力学仿真等任务。

---

## 🧩 1. 核心属性（Key Properties）

| 属性             | 值                   | 描述说明                            |
|------------------|----------------------|-------------------------------------|
| 目标位置         | `0.45`               | 到达右侧坡顶的水平坐标               |
| 目标速度         | `0.0`                | 到达目标位置时的最小速度要求         |
| 位置范围         | `[-1.2, 0.6]`        | 小车允许的水平位置边界              |
| 速度范围         | `[-0.07, 0.07]`      | 小车允许的水平速度边界              |
| 物理参数         | `power=0.0015`，`gravity=0.0025` | 控制力与重力的物理常量          |
| 最多步长         | `200`                | 超过步数后自动截断（truncated）     |
| 渲染模式         | `human`，`rgb_array` | 支持可视化窗口或 RGB 数组输出       |

---

## 🔍 2. 状态空间（Observation Space）

连续二维状态向量 `[位置, 速度]`：

```python
Box(low=[-1.2, -0.07], high=[0.6, 0.07], dtype=np.float32)
````

| 索引 | 名称 | 数值范围            | 含义说明             |
| -- | -- | --------------- | ---------------- |
| 0  | 位置 | `[-1.2, 0.6]`   | 小车当前在谷地中的位置      |
| 1  | 速度 | `[-0.07, 0.07]` | 小车当前水平速度（负值表示向左） |

---

## 🎮 3. 动作空间（Action Space）

连续一维动作 `[力]`：

```python
Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
```

| 数值范围 | 作用方向  | 说明                   |
| ---- | ----- | -------------------- |
| 负数   | 向左推   | 力大小 × power          |
| 正数   | 向右推   | 力大小 × power          |
| 绝对值  | 推动力强度 | 动作会被剪裁至 `[-1, 1]` 范围 |

---

## ⚖️ 4. 奖励机制（Reward System）

* **基础惩罚**：
  `reward = -0.1 × (action^2)`
  👉 惩罚较大的力，鼓励精细控制

* **成功奖励**：
  若小车在位置 `≥ 0.45` 且速度 `≥ 0.0`，给予奖励 `+100.0`

* **完整奖励函数**：

```python
reward = -0.1 * (action**2) + (100 if position >= 0.45 and velocity >= 0 else 0)
```

---

## 🚥 5. 回合终止条件（Termination）

| 条件类型  | 标志                | 描述说明                  |
| ----- | ----------------- | --------------------- |
| 成功终止  | `terminated=True` | 小车到达目标位置，且速度符合条件      |
| 步数截断  | `truncated=True`  | 连续控制步数达到最大值（默认 200）   |
| 左边界触发 | -                 | 小车撞到最左边界时速度被强制清零（未终止） |

---

## 🖼️ 6. 可视化渲染（Visualization）

通过 `render()` 实现实时渲染：

```python
def render(self) -> Optional[np.ndarray]
```

### 🔎 可视化要素：

* **地形地貌**：正弦函数地形 `y = sin(3x) * 0.4 + 0.5`
* **小车造型**：红色圆形，附带轮子
* **速度箭头**：表示当前速度方向
* **目标标记**：红色三角指示目标位置 `(0.45)`
* **信息面板**：显示当前步数与速度

---

## 🔄 7. 核心逻辑方法（Core Methods）

### 🏗️ 状态推进：`step(action)`

```python
def step(self, action) -> Tuple[obs, reward, terminated, truncated, info]
```

执行流程：

1. 动作剪裁至 `[-1, 1]`
2. 更新速度（考虑推力与坡度重力）
3. 更新位置（考虑速度与边界）
4. 奖励计算
5. 判断是否终止
6. 渲染（如启用）

---

### 🔬 物理更新核心逻辑：`_physics_update()`

```python
def _physics_update(position, velocity, action) -> Tuple[new_pos, new_vel]
```

伪公式如下：

```python
force = clip(action, -1, 1) * power
velocity += force - gravity * cos(3 * position)
velocity = clip(velocity, -0.07, 0.07)

position += velocity
position = clip(position, -1.2, 0.6)
```

> 若小车撞到最左边界，速度会被清零

---

## 🏁 8. 基本使用示例（Usage Example）

```python
from gymnasium.envs.classic_control import MountainCarContinuousEnv

env = MountainCarContinuousEnv(render_mode="human")
obs, info = env.reset()

while True:
    action = agent(obs)  # 自定义策略或随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        break

env.close()
```

---

## ✅ 9. 验证测试（Verification Tests）

本环境附带多个测试用例，用于确保行为一致性：

* 左边界反弹时速度归零
* 成功终止条件触发判断
* 200 步截断行为是否触发
* 奖励计算公式验证
* 渲染图像是否可视化正确

运行测试：

```python
test_environment(episodes=3)
```

---

## 📚 10. 参考资料（References）

* Brockman, G. et al. (2016). [OpenAI Gym](https://arxiv.org/abs/1606.01540)
* Moore, A. (1990). *Efficient Memory-based Learning for Robot Control*
* Singh, S. et al. (2000). *Convergence of Single-Step On-Policy RL Algorithms*

---

> 本环境适用于连续控制策略（如 DDPG、TD3、SAC）训练与测试，适合从简到深的控制学习任务演示。
