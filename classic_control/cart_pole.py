import math
import numpy as np
import pygame
import random
from typing import Tuple, Optional, Dict, Any
import gym
from gym import spaces

class CartPoleEnv(gym.Env):
    """
    一个从头实现的经典CartPole环境，使用OpenAI Gym接口，并使用Pygame进行可视化。

    Attributes:
        state (Tuple[float, float, float, float]): 当前状态 [车位置, 车速, 杆角度, 杆角速度]
        steps_count (int): 当前episode的步数计数器
        done (bool): 标记episode是否结束
        metadata (dict): 环境元数据（如渲染模式）
        screen (pygame.Surface): Pygame窗口表面
        clock (pygame.time.Clock): Pygame时钟控制帧率
    """
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self) -> None:
        super().__init__()
        # 物理参数
        self.gravity = 9.8  # 重力加速度 (m/s^2)
        self.cart_mass = 1.0  # 车质量 (kg)
        self.pole_mass = 0.1  # 杆质量 (kg)
        self.total_mass = self.cart_mass + self.pole_mass
        self.length = 0.5  # 杆的半长 (m)
        self.polemass_length = self.pole_mass * self.length
        self.force_mag = 10.0  # 作用力大小 (N)
        self.dt = 0.02  # 积分时间步长 (s)
        
        # 阈值（用于终止条件）
        self.x_threshold = 2.4  # 车水平位置阈值 (m)
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  # 杆角度阈值 (rad)
        
        # 状态和动作空间定义
        high = np.array([
            self.x_threshold * 2.0,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2.0,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        # 初始化状态
        self.state = None
        self.steps_count = 0
        self.done = False
        
        # Pygame可视化相关
        self.screen = None
        self.clock = None
        self.isopen = True
        
        # 屏幕尺寸参数
        self.screen_width = 800
        self.screen_height = 600
        self.cart_width = 0.4  # 车宽度 (m)
        self.cart_height = 0.2  # 车高度 (m)
        self.pole_length = 1.0  # 杆的全长 (m)
        self.scale = 200  # 像素每米：用于将物理坐标映射到屏幕像素

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一个动作，更新环境状态。
        
        Args:
            action (int): 要执行的动作，0 表示向左推，1 表示向右推。
            
        Returns:
            observation: 当前状态。
            reward: 该步的奖励（总是1）。
            done: 该步是否导致episode结束。
            info: 附加信息（空字典）。
        """
        assert self.action_space.contains(action), f"无效动作: {action}"
        if self.state is None:
            raise RuntimeError("请先调用reset()方法")
        
        # 解析状态变量
        x, x_dot, theta, theta_dot = self.state
        
        # 根据动作确定作用力方向
        force = self.force_mag if action == 1 else -self.force_mag
        
        # 计算角加速度和线加速度
        sintheta = math.sin(theta)
        costheta = math.cos(theta)
        # 中间计算：临时变量
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        # 角加速度 (θ'')
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.pole_mass * costheta ** 2 / self.total_mass)
        )
        # 线加速度 (x'')
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass
        
        # 通过欧拉法更新状态
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        
        # 更新状态
        self.state = (x, x_dot, theta, theta_dot)
        self.steps_count += 1
        
        # 检查终止条件
        self.done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        # 计算奖励：未终止时返回1，终止返回0
        reward = 1.0 if not self.done else 0.0
        
        return np.array(self.state, dtype=np.float32), reward, self.done, {}

    def reset(self) -> np.ndarray:
        """
        重置环境到初始随机状态。
        
        Returns:
            初始状态数组。
        """
        self.state = (
            random.uniform(-0.05, 0.05),  # 车位置
            0.0,  # 车速度
            random.uniform(-0.05, 0.05),  # 杆角度
            0.0   # 杆角速度
        )
        self.steps_count = 0
        self.done = False
        return np.array(self.state, dtype=np.float32)

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        使用Pygame渲染环境。
        
        Args:
            mode (str): 渲染模式（'human' 或 'rgb_array'）。
            
        Returns:
            如果模式是 'rgb_array' 则返回RGB数组，否则返回None。
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("CartPole")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # 屏幕中心对应于坐标 (0, 0) 在物理世界中的位置
        center_x = self.screen_width // 2
        ground_y = self.screen_height - 50
        
        # 获取状态变量
        if self.state is None:
            return
        x, _, theta, _ = self.state
        
        # 计算车在屏幕上的坐标
        cart_x = x * self.scale + center_x
        cart_y = ground_y
        
        # 计算杆端点坐标（在车中心上方）
        pole_top_x = cart_x + self.pole_length * self.scale * math.sin(theta)
        pole_top_y = cart_y - self.pole_length * self.scale * math.cos(theta)
        
        # 清屏（白色背景）
        self.screen.fill((255, 255, 255))
        
        # 绘制轨道（一条横贯屏幕的水平线）
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (0, ground_y),
            (self.screen_width, ground_y),
            2
        )
        
        # 绘制车（矩形）
        cart_rect = pygame.Rect(
            cart_x - self.cart_width * self.scale / 2,
            cart_y - self.cart_height * self.scale / 2,
            self.cart_width * self.scale,
            self.cart_height * self.scale
        )
        pygame.draw.rect(self.screen, (100, 100, 200), cart_rect)
        
        # 绘制杆（从车中心到杆顶点的线）
        pygame.draw.line(
            self.screen,
            (200, 100, 100),
            (cart_x, cart_y),
            (pole_top_x, pole_top_y),
            6
        )
        
        # 如果episode结束，显示终止信息
        if self.done:
            font = pygame.font.Font(None, 36)
            text = font.render("Episode Terminated", True, (255, 0, 0))
            self.screen.blit(text, (center_x - 120, 50))
        
        # 刷新屏幕显示
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
        # 处理Pygame事件（如关闭窗口）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.isopen = False
        
        # 返回RGB数组（如果模式为'rgb_array'）
        if mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        return None

    def close(self) -> None:
        """
        关闭环境并清理Pygame资源。
        """
        if self.screen is not None:
            pygame.quit()
            self.isopen = False
            self.screen = None
            self.clock = None

if __name__ == "__main__":
    # 创建环境并运行简单测试
    env = CartPoleEnv()
    state = env.reset()
    total_reward = 0
    steps = 0
    
    # 运行1000步或直到episode结束
    while steps < 1000:
        action = env.action_space.sample()  # 随机动作
        next_state, reward, done, _ = env.step(action)
        env.render()
        steps += 1
        total_reward += reward
        if done:
            break
    
    # 输出结果
    print(f"测试结束，总步数: {steps}, 总奖励: {total_reward}")
    env.close()