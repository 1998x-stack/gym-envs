import pygame
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
from enum import Enum

class MountainCarEnv:
    """
    A custom implementation of the MountainCar environment.
    Uses discrete actions with PyGame for rendering.
    
    Observation space: (position, velocity)
    Action space: Discrete(3) - [left, noop, right]
    """
    
    class Action(Enum):
        """Discrete actions available in the environment"""
        LEFT = 0
        NOOP = 1
        RIGHT = 2
    
    def __init__(self, 
                 render_mode: Optional[str] = None, 
                 max_steps: int = 200, 
                 window_size: Tuple[int, int] = (800, 600)) -> None:
        """
        Initialize MountainCar environment.
        
        Args:
            render_mode: Rendering mode (None, 'human', 'rgb_array')
            max_steps: Max steps per episode
            window_size: Render window dimensions
        """
        # 环境参数 (Environment parameters)
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.force_scale = 0.001
        self.gravity = 0.0025
        self.max_steps = max_steps
        
        # 动作/状态空间 (Action/Observation Space)
        self.action_space = 3  # Discrete(3)
        self.observation_space = np.array([
            [self.min_position, -self.max_speed],
            [self.max_position, self.max_speed]
        ])
        
        # 渲染设置 (Rendering setup)
        self.render_mode = render_mode
        self.window_size = window_size
        self._trail = []  # 轨迹点存储 (Trail points)
        self.window = None
        self.clock = None
        
        # 初始状态 (Initial state)
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        重置环境到初始状态 (Reset environment to initial state)
        
        Returns:
            初始观察值 (Initial observation)
        """
        # 在斜坡底部附近随机初始化位置 (Random position near valley bottom)
        self.state[0] = np.random.uniform(low=-0.6, high=-0.4)
        self.state[1] = 0.0
        self.steps = 0
        self._trail = []
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行动作并更新环境状态 (Execute action and update environment)
        
        Args:
            action: 离散动作索引 (Discrete action index)
        
        Returns:
            (obs, reward, done, info)
        """
        # 解析动作 (Parse action)
        force = action - 1  # 将动作转换为[-1, 0, 1] (Map action to [-1,0,1])
        
        # 物理计算 (Physics calculations)
        position, velocity = self.state
        velocity += (
            force * self.force_scale 
            - self.gravity * math.cos(3 * position)
        )
        
        # 速度钳制 (Velocity clamping)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        
        # 位置钳制 (Position clamping)
        if position <= self.min_position and velocity < 0:
            velocity = 0
        position = np.clip(position, self.min_position, self.max_position)
        
        # 更新状态 (Update state)
        self.state = np.array([position, velocity])
        self.steps += 1
        
        # 检查终止条件 (Check termination)
        done = position >= self.goal_position or self.steps >= self.max_steps
        reward = -1.0  # 每一步固定惩罚 (Fixed penalty per step)
        
        # 记录轨迹 (Record trail for rendering)
        self._trail.append((position, math.sin(3 * position)))
        
        return self.state.copy(), reward, done, {}
    
    def render(self) -> None:
        """使用PyGame渲染环境状态 (Render environment state using PyGame)"""
        if self.render_mode is None:
            return
        
        # 初始化Pygame (Initialize PyGame)
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("MountainCar")
        
        # 计算缩放参数 (Calculate scaling parameters)
        map_width = self.max_position - self.min_position
        screen_w, screen_h = self.window_size
        scale_x = screen_w / map_width
        baseline_y = screen_h * 0.7
        scale_y = 100
        
        # 清除屏幕 (Clear screen)
        self.window.fill((0, 0, 0))
        
        # 绘制网格背景 (Draw grid background)
        for x in np.arange(self.min_position, self.max_position, 0.2):
            px = int((x - self.min_position) * scale_x)
            pygame.draw.line(self.window, (30, 30, 30), (px, 0), (px, screen_h), 1)
        
        # 绘制山脉曲线 (Draw mountain curve)
        points = []
        for x in np.linspace(self.min_position, self.max_position, 100):
            wx = (x - self.min_position) * scale_x
            wy = baseline_y - math.sin(3 * x) * scale_y
            points.append((wx, wy))
        pygame.draw.lines(self.window, (100, 200, 100), False, points, 3)
        
        # 绘制目标位置 (Draw target position)
        goal_x = (self.goal_position - self.min_position) * scale_x
        goal_y = baseline_y - math.sin(3 * self.goal_position) * scale_y
        pygame.draw.circle(self.window, (255, 215, 0), (int(goal_x), int(goal_y)), 10)
        font = pygame.font.Font(None, 36)
        target_text = font.render("TARGET", True, (255, 215, 0))
        self.window.blit(target_text, (goal_x - 40, goal_y - 40))
        
        # 绘制轨迹 (Draw trail)
        if len(self._trail) > 1:
            trail_points = []
            for (x, y) in self._trail[-100:]:
                tx = (x - self.min_position) * scale_x
                ty = baseline_y - y * scale_y
                trail_points.append((tx, ty))
            pygame.draw.lines(self.window, (70, 130, 180), False, trail_points, 2)
        
        # 绘制车辆 (Draw car)
        pos, vel = self.state
        car_x = (pos - self.min_position) * scale_x
        terrain_y = math.sin(3 * pos)
        car_y = baseline_y - terrain_y * scale_y
        
        # 计算车辆角度 (Calculate car angle)
        slope = 3 * math.cos(3 * pos)
        angle = math.degrees(math.atan(-slope))
        
        # 创建车辆表面 (Create car surface)
        car_surface = pygame.Surface((30, 15), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (220, 20, 60), (0, 0, 30, 15), border_radius=3)
        pygame.draw.rect(car_surface, (50, 50, 50), (5, 5, 20, 5))
        rotated_car = pygame.transform.rotate(car_surface, angle)
        
        # 定位车辆 (Position car)
        car_rect = rotated_car.get_rect(center=(car_x, car_y))
        self.window.blit(rotated_car, car_rect)
        
        # 绘制速度向量 (Draw velocity vector)
        if vel != 0:
            direction = 1 if vel > 0 else -1
            end_x = car_x + direction * 20
            pygame.draw.line(
                self.window, 
                (0, 191, 255), 
                (car_x, car_y - 20), 
                (end_x, car_y - 20), 
                3
            )
        
        # 显示状态数据 (Display state data)
        info_text = font.render(
            f"Position: {pos:.3f} | Velocity: {vel:.4f} | Steps: {self.steps}", 
            True, (255, 255, 255))
        self.window.blit(info_text, (10, 10))
        
        # 刷新屏幕 (Update display)
        pygame.event.pump()
        pygame.display.flip()
        
        # 控制帧率 (Control frame rate)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.clock.tick(30)
    
    def close(self) -> None:
        """清理渲染资源 (Clean up rendering resources)"""
        if self.window is not None:
            pygame.display.quit()
            self.window = None
            pygame.quit()

# 演示环境功能 (Demo environment functionality)
if __name__ == "__main__":
    import time
    
    # 创建并重置环境 (Create and reset environment)
    env = MountainCarEnv(render_mode="human")
    state = env.reset()
    done = False
    
    # 简单控制策略 (Simple control policy)
    def simple_policy(state):
        """简单爬山策略 (Simple hill-climbing policy)"""
        pos, vel = state
        return MountainCarEnv.Action.RIGHT.value if vel >= 0 else MountainCarEnv.Action.LEFT.value
    
    # 主循环 (Main loop)
    try:
        while not done:
            # 渲染环境 (Render environment)
            env.render()
            
            # 选择并执行动作 (Select and execute action)
            action = simple_policy(state)
            state, reward, done, _ = env.step(action)
            
            # 显示状态信息 (Display state info)
            print(f"Position: {state[0]:.3f}, Velocity: {state[1]:.5f}, Reward: {reward}")
            
            # 小延迟以观察 (Small delay for observation)
            time.sleep(0.02)
    finally:
        env.close()
    print("Episode completed!")