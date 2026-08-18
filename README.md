# gym-envs

> 经典 Gymnasium 环境的**教学示例合集**（classic_control + toy_text）。

## 📌 项目概述

`gym-envs` 收录了 Gymnasium 经典环境的**实现 + 图文教程**，以「代码 + Markdown 讲解」结构组织，帮助理解强化学习标准环境与接口约定。

## 📁 目录结构

```
gym-envs/
├─ classic_control/   # 经典控制类
│   ├─ cart_pole.{py,md}
│   ├─ mountain_car.{py,md}
│   ├─ mountain_car_continuous.{py,md}
│   └─ pendulum.{py,md}
├─ toy_text/          # 文本类
│   ├─ frozen_lake.{py,md}
│   ├─ blackjack.py
│   └─ cliff_walking.{py,md}
└─ README.md
```

每个环境同时提供可运行的 `.py` 实现与 Markdown 图文说明。

## 🚀 使用

```bash
pip install gymnasium
# 运行某一环境示例
python classic_control/cart_pole.py
```

## 📄 License

MIT