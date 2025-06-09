import math
import numpy as np
import gym
from gym import spaces
import pygame
import pygame.gfxdraw
from typing import Tuple, Optional, Dict, Any


class MountainCarContinuousEnv(gym.Env):
    """连续山地车环境实现。
    
    模拟一个车在正弦波地形山谷中行驶的物理环境，使用连续动力系统模拟。
    目标是通过连续动力控制使车到达右侧山顶目标位置。
    
    观察空间: [位置, 速度] 
        position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]
    动作空间: [-1.0, 1.0] 的连续值，表示推动小车的力
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, 
                 goal_velocity: float = 0.0,
                 render_mode: Optional[str] = None) -> None:
        # 物理系统参数
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # 目标位置
        self.goal_velocity = goal_velocity  # 目标速度
        self.power = 0.0015  # 动力系数
        self.gravity = 0.0025  # 重力系数

        # 渲染系统参数
        self.screen_width = 800
        self.screen_height = 600
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # 观察空间: [位置, 速度]
        self.low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        # 动作空间: [-1, 1] 连续控制
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 初始化状态
        self.state = None
        self.steps = 0
        self.max_episode_steps = 200

    def _physics_update(self, 
                        current_position: float, 
                        current_velocity: float, 
                        action: np.ndarray) -> Tuple[float, float]:
        """应用物理规则更新系统状态。
        
        Args:
            current_position: 当前车辆位置
            current_velocity: 当前车辆速度
            action: 施加的控制力 [-1.0, 1.0]
        
        Returns:
            更新后的位置和速度
        """
        force = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        
        # 更新速度
        velocity = current_velocity + force * self.power - self.gravity * math.cos(3 * current_position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        
        # 更新位置
        position = current_position + velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        # 左边界速度归零
        if position <= self.min_position and velocity < 0:
            velocity = 0.0
            
        return position, velocity

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步环境动态更新。
        
        Args:
            action: 控制动作向量
            
        Returns:
            observation: 新状态观测值
            reward: 获得的奖励
            terminated: 是否终止
            truncated: 是否被截断（超时）
            info: 额外信息
        """
        self.steps += 1
        position, velocity = self.state
        new_position, new_velocity = self._physics_update(position, velocity, action)
        self.state = np.array([new_position, new_velocity], dtype=np.float32)
        
        # 计算奖励
        # 基础奖励: 惩罚过大控制动作
        reward = -0.1 * (action[0] ** 2)  
        # 成功奖励
        if new_position >= self.goal_position and new_velocity >= self.goal_velocity:
            reward += 100.0
        
        # 终止条件检查
        terminated = bool(
            new_position >= self.goal_position and new_velocity >= self.goal_velocity
        )
        
        # 步数限制截断
        truncated = bool(self.steps >= self.max_episode_steps)
        
        # 渲染更新
        if self.render_mode == "human":
            self.render()
            
        return self.state, reward, terminated, truncated, {"original_reward": reward + 0.1*(action[0]**2)}

    def reset(self, 
             seed: Optional[int] = None, 
             options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境到初始状态。
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            初始状态和额外信息
        """
        super().reset(seed=seed)
        # 初始位置在小山谷中，初始速度为0
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def _render_setup(self) -> None:
        """初始化Pygame渲染系统。"""
        if self.screen is None and self.render_mode is not None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height)
                )
            pygame.display.set_caption("MountainCar Continuous")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _position_to_screen_coords(self, position: float) -> Tuple[int, int]:
        """将物理位置转换为屏幕坐标。
        
        Args:
            position: 物理位置
            
        Returns:
            屏幕坐标 (x, y)
        """
        # X坐标映射
        norm_x = (position - self.min_position) / (self.max_position - self.min_position)
        screen_x = norm_x * (self.screen_width - 100) + 50
        
        # Y坐标由正弦波地形定义
        terrain_height = math.sin(3 * position) * 0.4 + 0.5
        screen_y = self.screen_height * terrain_height - 100
        return int(screen_x), int(screen_y)

    def render(self) -> Optional[np.ndarray]:
        """渲染环境状态。
        
        Returns:
            如果是rgb_array模式，返回图像数组；否则返回None
        """
        self._render_setup()
        
        # 清空屏幕
        self.screen.fill((255, 255, 255))
        
        # 绘制地形背景
        terrain_points = []
        for i in range(self.screen_width):
            pos = self.min_position + (i / self.screen_width) * (self.max_position - self.min_position)
            x, y = self._position_to_screen_coords(pos)
            terrain_points.append((x, y))
        
        # 绘制主地形
        if len(terrain_points) > 1:
            pygame.draw.lines(self.screen, (70, 130, 180), False, terrain_points, 3)
        
        # 绘制起点和终点标记
        start_x, start_y = self._position_to_screen_coords(-0.5)
        pygame.draw.circle(self.screen, (50, 205, 50), (start_x, start_y), 8)
        
        goal_x, goal_y = self._position_to_screen_coords(self.goal_position)
        pygame.draw.polygon(self.screen, (220, 20, 60), 
                           [(goal_x, goal_y), (goal_x - 15, goal_y - 30), 
                            (goal_x + 15, goal_y - 30)])
        
        # 绘制物理小车
        if self.state is not None:
            car_x, car_y = self._position_to_screen_coords(self.state[0])
            pygame.draw.circle(self.screen, (220, 20, 60), (car_x, car_y), 15)
            pygame.draw.circle(self.screen, (30, 30, 30), (car_x - 8, car_y + 12), 5)
            pygame.draw.circle(self.screen, (30, 30, 30), (car_x + 8, car_y + 12), 5)
            
            # 速度方向指示器
            velocity_ind = -int(self.state[1] * 500)
            pygame.draw.line(self.screen, (30, 30, 30), 
                            (car_x, car_y), 
                            (car_x + velocity_ind, car_y - 10), 2)
        
        # 信息显示
        font = pygame.font.SysFont('Arial', 24)
        step_text = font.render(f"Steps: {self.steps}/{self.max_episode_steps}", True, (0, 0, 0))
        self.screen.blit(step_text, (20, 20))
        
        if self.state is not None:
            vel_text = font.render(f"Velocity: {self.state[1]:.4f}", True, (0, 0, 0))
            self.screen.blit(vel_text, (20, 50))

        # 根据渲染模式处理
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        """关闭渲染资源。"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


def test_environment(episodes: int = 3) -> None:
    """测试山地车环境的功能。
    
    Args:
        episodes: 测试次数
    """
    env = MountainCarContinuousEnv(render_mode="human")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            # 随机策略
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            
        print(f"Episode {episode+1} finished. Reward: {total_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    # 测试环境运行
    test_environment()