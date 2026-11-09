# gym-envs · 经典强化学习教学环境

面向初学者的 Gymnasium 环境/小游戏教学合集（经典控制与 Toy Text）。**它是用于研究和教学的可复现环境，不是直接浏览器试玩目录。**

## P3 · 现代环境接口与可复现性（2026-11-09）

原有 FrozenLake 和 Blackjack 示例曾使用已退役的 Gym **4 元组** `step()`、未分离的全局随机源；Blackjack 会在 `reset()` 中提前标记终局而没有告知调用方。本轮将以下两个环境迁移至现代 Gymnasium 接口：

- `reset(seed=...)` 返回 `(observation, info)`；
- `step(action)` 返回 `(observation, reward, terminated, truncated, info)`；
- 所有随机行为从环境私有 `np_random` 获取，避免污染其他试验。
- 终局后拒绝继续执行动作，必须显式 `reset()`；
- FrozenLake 支持可控滑动、确定性路径、ANSI 与低分辨率 RGB 数组显示，不再要求 Pygame 安装；
- Blackjack 支持 **hit/stand**、A 计分与完整庄家终局；自然 21 在玩家 stand 时判定，保证 reset 后仍可合法行动。

其他 `classic_control/` 与 `toy_text/` 的旧示例**尚未统一迁移**，不要将它们全部宣称 Gymnasium 已兼容。

### 安装与验证

```bash
python -m pip install "gymnasium>=1.0,<1.4" "numpy>=1.24,<3"
python -m unittest discover -s tests -v
```

包含 FrozenLake 完整获胜路径、Blackjack 完整结算、固定种子重现、全局随机源隔离、非法终局动作和 Gymnasium `check_env` 兼容性检查。CI 通过只说明这两个示例的协议与规则测试成功，不代表其他环境或训练算法经过复现。

### 程序用法

```python
from toy_text.frozen_lake import FrozenLakeEnv
from toy_text.blackjack import ToyTextBlackjackEnv

lake = FrozenLakeEnv(is_slippery=False, render_mode="ansi")
obs, info = lake.reset(seed=42)
obs, reward, terminated, truncated, info = lake.step(1)

blackjack = ToyTextBlackjackEnv()
obs, info = blackjack.reset(seed=42)
obs, reward, terminated, truncated, info = blackjack.step(0)  # stand
```

本仓库主要用作教育；使用更广泛的 Gymnasium 官方环境时应参考对应官方版本文档，而不要把本仓库中的教学实现当作官方同名环境。
