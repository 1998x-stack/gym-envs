import pygame
import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, List, Optional

class ToyTaxiEnv(gym.Env):
    """
    A custom Taxi environment implemented from scratch with PyGame visualization.
    The environment follows the classic Taxi problem dynamics: 
    - 5x5 grid world with barriers
    - Taxi picks up passenger at one location and drops at destination
    - 6 possible actions: move in 4 directions + pickup/dropoff

    State Representation:
        (taxi_row, taxi_col, passenger_location, destination)
    
    Actions:
        0 = South, 1 = North, 2 = East, 3 = West, 4 = Pickup, 5 = Dropoff
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super(ToyTaxiEnv, self).__init__()
        # 定义环境尺寸和特殊位置
        self.grid_size = (5, 5)
        self.passenger_locations = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B
        self.destination_locations = [(0, 0), (0, 4), (4, 0), (4, 3)]

        # 障碍物定义 (阻挡移动方向)
        self.horizontal_barriers = {(0, 1), (0, 3), (2, 1), (2, 3)}
        self.vertical_barriers = {(1, 1), (1, 3), (3, 1), (3, 3)}

        # 状态和动作空间
        self.state_space = spaces.Discrete(5 * 5 * 5 * 4)  # 500种可能状态
        self.action_space = spaces.Discrete(6)  # 6种动作
        self.observation_space = self.state_space
        self.action_meaning = {0: "South", 1: "North", 2: "East", 3: "West", 4: "Pickup", 5: "Dropoff"}

        # 渲染配置
        self.render_mode = render_mode
        self.cell_size = 60  # 每个网格的像素大小
        self.window_size = (
            self.grid_size[1] * self.cell_size,
            self.grid_size[0] * self.cell_size,
        )
        self.colors = {
            'background': (255, 255, 255),
            'taxi_empty': (200, 200, 200),
            'taxi_occupied': (170, 240, 170),
            'wall': (0, 0, 0),
            'locations': [
                (255, 0, 0),    # R - Red
                (0, 255, 0),    # G - Green
                (255, 255, 0),  # Y - Yellow
                (0, 0, 255)     # B - Blue
            ],
            'destination': (128, 0, 128)  # Purple
        }
        
        # PyGame 变量
        self.window = None
        self.clock = None
        self.font = None

        # 初始化状态
        self.reset()

    def encode_state(self, taxi_row: int, taxi_col: int, pass_loc: int, dest_idx: int) -> int:
        """
        将状态元组编码为单一整数值
        Args:
            taxi_row: 出租车行位置 (0-4)
            taxi_col: 出租车列位置 (0-4)
            pass_loc: 乘客位置 (0-4) 0-3=位置,4=在车上
            dest_idx: 目的地索引 (0-3)
        Returns:
            int: 编码后的状态值
        """
        return ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx

    def decode_state(self, state: int) -> Tuple[int, int, int, int]:
        """
        将单一整数值解码为状态元组
        Args:
            state: 编码后的状态值
        Returns:
            tuple: (taxi_row, taxi_col, pass_loc, dest_idx)
        """
        dest_idx = state % 4
        state //= 4
        pass_loc = state % 5
        state //= 5
        taxi_col = state % 5
        taxi_row = state // 5
        return taxi_row, taxi_col, pass_loc, dest_idx

    def reset(self) -> int:
        """
        重置环境到随机初始状态
        Returns:
            int: 初始状态编码值
        """
        # 随机初始化出租车位置
        self.taxi_row, self.taxi_col = np.random.randint(0, 5), np.random.randint(0, 5)
        
        # 随机设置乘客位置和目的地 (确保起始位置和目的地不同)
        self.passenger_loc = np.random.randint(0, 4)
        self.destination_idx = np.random.randint(0, 4)
        while self.passenger_loc == self.destination_idx:
            self.destination_idx = np.random.randint(0, 4)
        
        self.state = self.encode_state(
            self.taxi_row, self.taxi_col, self.passenger_loc, self.destination_idx
        )
        self.last_action = None
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return self.state

    def move_taxi(self, action: int) -> bool:
        """
        尝试移动出租车并处理障碍物
        Args:
            action: 移动方向 (0-3)
        Returns:
            bool: 是否移动成功
        """
        new_row, new_col = self.taxi_row, self.taxi_col

        if action == 0:  # South
            if self.taxi_row < 4 and (self.taxi_row, self.taxi_col) not in self.horizontal_barriers:
                new_row += 1
        elif action == 1:  # North
            if self.taxi_row > 0 and (self.taxi_row - 1, self.taxi_col) not in self.horizontal_barriers:
                new_row -= 1
        elif action == 2:  # East
            if self.taxi_col < 4 and (self.taxi_row, self.taxi_col) not in self.vertical_barriers:
                new_col += 1
        elif action == 3:  # West
            if self.taxi_col > 0 and (self.taxi_row, self.taxi_col - 1) not in self.vertical_barriers:
                new_col -= 1

        # 更新位置
        moved = (new_row != self.taxi_row) or (new_col != self.taxi_col)
        self.taxi_row, self.taxi_col = new_row, new_col
        return moved

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        执行环境步进
        Args:
            action: 要执行的动作 (0-5)
        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.last_action = action
        done = False
        reward = -1  # 默认移动惩罚
        
        # 移动动作处理
        if action < 4:
            moved = self.move_taxi(action)
            if not moved:
                reward = -2  # 额外障碍物碰撞惩罚
                
        # 拾取动作处理
        elif action == 4:
            passenger_at_location = (
                self.passenger_loc < 4 and 
                (self.taxi_row, self.taxi_col) == self.passenger_locations[self.passenger_loc]
            )
            if passenger_at_location:
                self.passenger_loc = 4  # 乘客在车上
            else:
                reward = -10  # 无效拾取惩罚
                
        # 放下动作处理
        elif action == 5:
            if self.passenger_loc == 4:
                if (self.taxi_row, self.taxi_col) == self.passenger_locations[self.destination_idx]:
                    done = True
                    reward = 20  # 成功奖励
                    self.passenger_loc = self.destination_idx
                else:
                    reward = -10  # 无效放下惩罚

        # 更新状态
        self.state = self.encode_state(
            self.taxi_row, self.taxi_col, self.passenger_loc, self.destination_idx
        )
        
        # 渲染当前帧
        if self.render_mode == 'human':
            self._render_frame()
        
        return self.state, reward, done, {}

    def _render_frame(self) -> None:
        """
        使用 PyGame 渲染当前环境状态
        """
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Toy Taxi Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 18)
        
        if self.clock is not None:
            self.clock.tick(self.metadata['render_fps'])
        
        canvas = pygame.Surface(self.window_size)
        canvas.fill(self.colors['background'])
        
        # 绘制网格和障碍物
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                rect = (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)
                
                # 绘制水平障碍物
                if (row, col) in self.horizontal_barriers:
                    pygame.draw.line(
                        canvas, 
                        self.colors['wall'], 
                        (col * self.cell_size, (row + 1) * self.cell_size),
                        ((col + 1) * self.cell_size, (row + 1) * self.cell_size),
                        3
                    )
                
                # 绘制垂直障碍物
                if (row, col) in self.vertical_barriers:
                    pygame.draw.line(
                        canvas, 
                        self.colors['wall'], 
                        ((col + 1) * self.cell_size, row * self.cell_size),
                        ((col + 1) * self.cell_size, (row + 1) * self.cell_size),
                        3
                    )
        
        # 绘制特殊位置 (R, G, Y, B)
        for idx, (row, col) in enumerate(self.passenger_locations):
            center = (
                col * self.cell_size + self.cell_size // 2,
                row * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(canvas, self.colors['locations'][idx], center, 10)
            if idx == self.destination_idx:
                pygame.draw.circle(canvas, self.colors['destination'], center, 15, 2)
        
        # 绘制出租车
        taxi_color = (
            self.colors['taxi_occupied'] if self.passenger_loc == 4 
            else self.colors['taxi_empty']
        )
        taxi_rect = (
            self.taxi_col * self.cell_size + 5,
            self.taxi_row * self.cell_size + 5,
            self.cell_size - 10,
            self.cell_size - 10
        )
        pygame.draw.rect(canvas, taxi_color, taxi_rect)
        pygame.draw.rect(canvas, (0, 0, 0), taxi_rect, 2)
        
        # 绘制乘客状态
        text = ""
        if self.passenger_loc < 4:
            text = f"Passenger at: {['R','G','Y','B'][self.passenger_loc]}"
        else:
            text = "Passenger in taxi"
        
        text += f" | Destination: {['R','G','Y','B'][self.destination_idx]}"
        
        if self.last_action is not None:
            text += f" | Action: {self.action_meaning[self.last_action]}"
        
        text_surface = self.font.render(text, True, (0, 0, 0))
        canvas.blit(text_surface, (5, 5))
        
        # 更新显示
        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), 
                axes=(1, 0, 2)
            )

    def render(self) -> None:
        """环境渲染接口方法"""
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        elif self.render_mode == 'human':
            self._render_frame()

    def close(self) -> None:
        """清理环境资源"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    # 验证环境正确性
    import time
    
    # 创建并初始化环境
    env = ToyTaxiEnv(render_mode='human')
    state = env.reset()
    
    # 运行测试用例
    test_passed = True
    
    # 测试1: 状态编码/解码一致性
    for _ in range(10):
        taxi_row, taxi_col = np.random.randint(0, 5, size=2)
        pass_loc = np.random.randint(0, 5)
        dest_idx = np.random.randint(0, 4)
        encoded = env.encode_state(taxi_row, taxi_col, pass_loc, dest_idx)
        decoded = env.decode_state(encoded)
        if (taxi_row, taxi_col, pass_loc, dest_idx) != decoded:
            print(f"编码/解码失败: 原始({taxi_row},{taxi_col},{pass_loc},{dest_idx}) "
                  f"-> 解码后({decoded})")
            test_passed = False
    
    # 测试2: 移动动作障碍检测
    # 设置出租车到(0,0)，测试向北移动应失败
    _, _, pass_loc, dest_idx = env.decode_state(state)
    env.taxi_row, env.taxi_col = 0, 1
    state = env.encode_state(0, 1, pass_loc, dest_idx)
    
    _, reward, _, _ = env.step(1)  # 向北移动
    if reward > -2:  # 应触发障碍物碰撞
        print("障碍物检测失败: 北向移动应被阻挡")
        test_passed = False
    
    # 测试3: 有效拾取
    env.taxi_row, env.taxi_col = env.passenger_locations[env.passenger_loc]
    passenger_loc = env.passenger_loc
    env.step(4)  # 执行拾取
    if env.passenger_loc != 4:
        print(f"拾取失败: 乘客状态={env.passenger_loc} (应为4)")
        test_passed = False
    
    # 测试4: 无效拾取
    # 将出租车移动到其他位置
    env.taxi_row, env.taxi_col = 0, 0
    # 尝试拾取
    _, reward, _, _ = env.step(4)
    if reward != -10:
        print("无效拾取检测失败")
        test_passed = False
    
    # 测试5: 有效放下
    env.taxi_row, env.taxi_col = env.passenger_locations[env.destination_idx]
    _, reward, done, _ = env.step(5)
    if not done or reward != 20:
        print(f"有效放下失败: done={done}, reward={reward}")
        test_passed = False
    
    # 输出测试结果
    print("\n===== 环境测试结果 =====")
    print(f"测试通过: {test_passed}")
    print(f"当前状态: {env.decode_state(env.state)}")
    print(f"最后动作: {env.action_meaning.get(env.last_action, 'None')}")
    
    # 手动演示
    if test_passed:
        print("\n启动手动演示...")
        env.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 0
                    elif event.key == pygame.K_LEFT:
                        action = 3
                    elif event.key == pygame.K_RIGHT:
                        action = 2
                    elif event.key == pygame.K_p:
                        action = 4
                    elif event.key == pygame.K_d:
                        action = 5
                    else:
                        continue
                    
                    _, _, done, _ = env.step(action)
                    if done:
                        print("任务完成! 重置环境...")
                        time.sleep(2)
                        env.reset()
        env.close()