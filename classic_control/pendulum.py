"""
Inverted Pendulum Environment Implementation

This module implements the classic inverted pendulum control problem using Gymnasium.
The environment includes physics modeling, observation/action spaces, reward calculation, 
and visualization using Pygame.

Copyright (c) 2023 Industrial Control Systems. All rights reserved.
"""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class InvertedPendulumEnv(gym.Env):
    """
    倒立摆环境实现 (Inverted Pendulum Environment Implementation)
    
    基于Gymnasium框架的经典倒立摆控制问题模拟。目标是通过扭矩控制使摆杆保持竖直向上状态。
    系统使用物理动力学方程建模，并通过Pygame实现可视化。

    观察空间 (Observation Space):
        Box([-1, -1, -8], [1, 1, 8], (3,), float32)
        包含: [cos(θ), sin(θ), 角速度]

    动作空间 (Action Space):
        Box(-2.0, 2.0, (1,), float32) - 代表扭矩

    动力学方程 (Dynamics):
        θ'' = (3g/2l) * sin(θ) + (3/(ml²)) * torque
    """
    
    # 元数据配置 (Metadata Configuration)
    metadata: Dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }
    
    def __init__(self, render_mode: Optional[str] = None) -> None:
        """
        初始化倒立摆环境 (Initialize inverted pendulum environment)
        
        Args:
            render_mode: 渲染模式, 支持 'human' 或 'rgb_array'
        """
        super().__init__()
        
        # 物理参数 (Physical parameters)
        self.gravity: float = 9.81  # 重力加速度 (m/s²)
        self.mass: float = 1.0      # 摆杆质量 (kg)
        self.length: float = 1.0    # 摆杆长度 (m)
        self.dt: float = 0.02       # 仿真时间步长 (s)
        self.max_speed: float = 8.0  # 最大角速度 (rad/s)
        self.max_torque: float = 2.0  # 最大扭矩 (N·m)
        
        # 计算力矩参数 (Calculate torque parameters)
        self.inertia: float = self.mass * self.length**2  # 转动惯量
        self.g_term: float = (3 * self.gravity) / (2 * self.length)
        self.torque_term: float = 3.0 / (self.mass * self.length**2)
        
        # 定义观察空间 (Define observation space)
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        low = np.array([-1.0, -1.0, -self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # 定义动作空间 (Define action space)
        self.action_space = spaces.Box(
            low=np.array([-self.max_torque], dtype=np.float32),
            high=np.array([self.max_torque], dtype=np.float32),
            dtype=np.float32
        )
        
        # 渲染设置 (Rendering setup)
        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.screen_size: Tuple[int, int] = (600, 600)
        self.pivot_pos: Tuple[float, float] = (300, 300)
        self.scale_factor: float = 200.0  # 像素/米 (Pixels per meter)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        重置环境状态 (Reset environment state)
        
        Returns:
            observation: 初始观察值
            info: 环境信息
        """
        super().reset(seed=seed)
        
        # 随机初始化角度和角速度 (Random initialize angle and angular velocity)
        theta = self.np_random.uniform(-np.pi, np.pi)
        theta_dot = self.np_random.uniform(-1, 1)
        
        # 保存状态 (Save state)
        self.state: np.ndarray = np.array([theta, theta_dot], dtype=np.float32)
        
        # 渲染初始化 (Rendering initialization)
        if self.render_mode == "human":
            self._init_render()
        
        # 返回初始观察值 (Return initial observation)
        return self._get_obs(), {}
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行单步仿真 (Perform single simulation step)
        
        Args:
            action: 施加的扭矩
            
        Returns:
            observation: 新观察值
            reward: 奖励值
            terminated: 是否终止
            truncated: 是否截断
            info: 环境信息
        """
        # 解析当前状态 (Parse current state)
        theta, theta_dot = self.state
        
        # 确保扭矩在合法范围 (Ensure torque within valid range)
        torque = np.clip(action, -self.max_torque, self.max_torque)[0]
        
        # 计算角加速度 (Calculate angular acceleration)
        gravity_torque = self.g_term * np.sin(theta)
        torque_accel = self.torque_term * torque
        theta_dot_dot = gravity_torque + torque_accel
        
        # 欧拉积分更新状态 (Update state with Euler integration)
        new_theta_dot = theta_dot + theta_dot_dot * self.dt
        new_theta = theta + new_theta_dot * self.dt
        
        # 限制角速度范围 (Constrain angular velocity)
        new_theta_dot = np.clip(
            new_theta_dot, 
            -self.max_speed, 
            self.max_speed
        )
        
        # 更新状态 (Update state)
        self.state = np.array([new_theta, new_theta_dot], dtype=np.float32)
        
        # 计算奖励 (Calculate reward)
        reward = self._calculate_reward(theta, theta_dot, torque)
        
        # 环境永不自动终止 (Environment never terminates automatically)
        terminated = truncated = False
        
        # 渲染更新 (Rendering update)
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _get_obs(self) -> np.ndarray:
        """
        获取当前观察值 (Get current observation)
        
        Returns:
            [cos(θ), sin(θ), 角速度]
        """
        theta, theta_dot = self.state
        return np.array([
            np.cos(theta), 
            np.sin(theta), 
            theta_dot
        ], dtype=np.float32)
    
    def _calculate_reward(
        self, 
        theta: float, 
        theta_dot: float,
        torque: float
    ) -> float:
        """
        计算奖励函数 (Calculate reward function)
        
        奖励函数设计原则:
        - 鼓励竖直向上状态 (cos(θ)=1, sin(θ)=0)
        - 惩罚高速运动和过度控制
        
        R = -(θ_normalized² + 0.1*θ_dot² + 0.001*torque²)
        """
        # 规范化角度到[-π, π] (Normalize angle to [-π, π])
        theta_norm = (theta + np.pi) % (2 * np.pi) - np.pi
        
        # 计算成本 (Calculate cost)
        angle_cost = theta_norm ** 2
        velocity_cost = 0.1 * (theta_dot ** 2)
        torque_cost = 0.001 * (torque ** 2)
        
        return float(-(angle_cost + velocity_cost + torque_cost))
    
    def _init_render(self) -> None:
        """初始化Pygame渲染 (Initialize Pygame rendering)"""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Inverted Pendulum Control")
            self.clock = pygame.time.Clock()
    
    def _render_frame(self) -> None:
        """渲染当前帧 (Render current frame)"""
        if self.screen is None:
            self._init_render()
        
        # 清空屏幕 (Clear screen)
        self.screen.fill((255, 255, 255))  # 白色背景 (White background)
        
        # 获取状态 (Get state)
        theta, _ = self.state
        
        # 计算端点位置 (Calculate end point position)
        end_x = self.pivot_pos[0] + self.scale_factor * np.sin(theta)
        end_y = self.pivot_pos[1] - self.scale_factor * np.cos(theta)
        
        # 绘制摆杆 (Draw pendulum)
        pygame.draw.line(
            surface=self.screen,
            color=(0, 0, 0),  # 黑色摆杆 (Black pendulum)
            start_pos=self.pivot_pos,
            end_pos=(end_x, end_y),
            width=4
        )
        
        # 绘制支点 (Draw pivot)
        pygame.draw.circle(
            surface=self.screen,
            color=(255, 0, 0),  # 红色支点 (Red pivot)
            center=self.pivot_pos,
            radius=8
        )
        
        # 绘制目标位置 (Draw target position - vertical upward)
        pygame.draw.line(
            surface=self.screen,
            color=(0, 255, 0),  # 绿色目标线 (Green target line)
            start_pos=self.pivot_pos,
            end_pos=(self.pivot_pos[0], self.pivot_pos[1] - self.scale_factor),
            width=1
        )
        
        # 更新显示 (Update display)
        pygame.event.pump()
        pygame.display.flip()
        
        # 控制帧率 (Control frame rate)
        self.clock.tick(self.metadata["render_fps"])
    
    def render(self) -> Optional[np.ndarray]:
        """渲染环境 (Render environment)"""
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            self._render_frame()
        return None
    
    def _render_rgb_array(self) -> np.ndarray:
        """生成RGB数组 (Generate RGB array)"""
        if self.screen is None:
            self._init_render()
        self._render_frame()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)),
            axes=(1, 0, 2)
        )
    
    def close(self) -> None:
        """关闭环境及渲染资源 (Close environment and rendering resources)"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

# 主函数与测试代码 (Main function and testing code)
if __name__ == "__main__":
    import time
    
    def test_environment() -> None:
        """测试环境功能 (Test environment functionality)"""
        # 初始化环境 (Initialize environment)
        env = InvertedPendulumEnv(render_mode="human")
        
        print("="*60)
        print("倒立摆环境测试 (Inverted Pendulum Environment Test)")
        print(f"观察空间 (Observation space): {env.observation_space}")
        print(f"动作空间 (Action space): {env.action_space}")
        print("="*60)
        
        try:
            # 运行测试周期 (Run test episodes)
            for episode in range(3):
                state, _ = env.reset()
                total_reward = 0.0
                
                print(f"测试周期 {episode+1} (Test Episode {episode+1})")
                print(f"初始状态 (Initial state): [cosθ={state[0]:.3f}, sinθ={state[1]:.3f}, ω={state[2]:.3f}]")
                
                # 运行单周期 (Run single episode)
                for step in range(200):
                    # 随机控制策略 (Random control policy)
                    action = env.action_space.sample()
                    
                    # 执行动作 (Take action)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    
                    # 状态转移 (State transition)
                    state = next_state
                    
                    # 显示进度 (Display progress)
                    if (step+1) % 50 == 0:
                        print(f"步骤 {step+1}: 奖励={reward:.3f}, 累计奖励={total_reward:.3f}")
                    
                    time.sleep(0.01)
                
                print(f"周期结束 (Episode finished): 总奖励={total_reward:.3f}")
                print("-"*60)
        
        finally:
            env.close()
            print("测试完成 (Test completed)")
    
    # 执行测试 (Execute test)
    test_environment()