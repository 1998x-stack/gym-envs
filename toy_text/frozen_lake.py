import gym
from gym import spaces
import numpy as np
import sys
import pygame
from pygame import gfxdraw
from typing import Optional, List, Tuple, Dict, Any

class FrozenLakeEnv(gym.Env):
    """自定义FrozenLake环境，模拟冰冻湖面导航问题。"""
    
    metadata = {"render.modes": ["human", "ansi", "rgb_array"], "render_fps": 4}
    
    # 预定义地图
    MAPS = {
        "4x4": [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ],
        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG"
        ],
    }
    
    # 动作对应的方向变化：右、下、左、上
    ACTION_DIRECTION = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def __init__(self, desc: Optional[List[str]] = None, map_name: str = "4x4", 
                 is_slippery: bool = True) -> None:
        super(FrozenLakeEnv, self).__init__()
        
        # 加载地图
        if desc is None and map_name in self.MAPS:
            desc = self.MAPS[map_name]
        elif desc is None:
            raise ValueError("必须提供地图描述或有效的预定义地图名称")
        
        self.desc = desc
        self.nrow = len(desc)
        self.ncol = len(desc[0])
        self.is_slippery = is_slippery
        
        # 初始化动作和状态空间
        self.action_space = spaces.Discrete(4)  # 4个动作
        self.observation_space = spaces.Discrete(self.nrow * self.ncol)  # 网格中的所有单元格
        
        # 初始化Pygame渲染相关变量
        self.window = None
        self.window_size = None
        self.cell_size = 60
        self.clock = None
        
        # 重置环境
        self.reset()
    
    def reset(self) -> int:
        """重置环境到初始状态。"""
        # 找到起始位置（标记为'S'）
        start_pos = None
        for i, row in enumerate(self.desc):
            for j, cell in enumerate(row):
                if cell == 'S':
                    start_pos = (i, j)
                    break
            if start_pos is not None:
                break
        if start_pos is None:
            raise ValueError("地图中没有起始位置（'S'）")
        
        # 状态表示为行优先的一维索引
        self.state = start_pos[0] * self.ncol + start_pos[1]
        self.done = False
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """执行一个动作并更新环境状态。
        
        参数:
            action: 要执行的动作（0: 右, 1: 下, 2: 左, 3: 上）
        
        返回:
            next_state: 新状态
            reward: 获得的奖励
            done: 是否结束
            info: 额外信息
        """
        if self.done:
            return self.state, 0.0, True, {}
        
        # 将状态索引转换为二维坐标
        current_row = self.state // self.ncol
        current_col = self.state % self.ncol
        
        # 处理滑动：随机选择实际动作（含偏移）
        if self.is_slippery:
            action = np.random.choice([action, (action - 1) % 4, (action + 1) % 4])
        
        # 获取动作对应的方向变化
        d_row, d_col = self.ACTION_DIRECTION[action]
        next_row = current_row + d_row
        next_col = current_col + d_col
        
        # 边界检查：如果超出则保持原位
        if next_row < 0 or next_row >= self.nrow or next_col < 0 or next_col >= self.ncol:
            next_row, next_col = current_row, current_col
        
        # 计算新状态的索引
        next_state_idx = next_row * self.ncol + next_col
        self.state = next_state_idx
        
        # 获取新位置的地图标记
        cell_value = self.desc[next_row][next_col]
        
        # 根据单元格类型判断结束状态和奖励
        if cell_value == 'H':  # 掉入洞穴
            self.done = True
            reward = 0.0
        elif cell_value == 'G':  # 到达目标
            self.done = True
            reward = 1.0
        else:  # 安全或冻结表面
            self.done = False
            reward = 0.0
        
        return next_state_idx, reward, self.done, {}
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """渲染当前环境状态。"""
        if mode in ("human", "rgb_array"):
            return self._render_gui(mode)
        elif mode == "ansi":
            return self._render_text()
        else:
            super().render(mode=mode)
    
    def _render_text(self) -> None:
        """在终端输出文本形式的网格。"""
        outfile = sys.stdout
        desc = self.desc.copy()
        
        # 当前状态的位置
        current_row = self.state // self.ncol
        current_col = self.state % self.ncol
        
        # 临时替换当前位置为智能体的标记
        row_list = list(desc[current_row])
        row_list[current_col] = 'A'
        desc[current_row] = ''.join(row_list)
        
        # 打印网格
        print("\n".join(desc), end="\n\n")
    
    def _render_gui(self, mode: str) -> Optional[np.ndarray]:
        """使用Pygame渲染图形界面。"""
        if self.window is None and mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.ncol * self.cell_size, self.nrow * self.cell_size))
            pygame.display.set_caption("Frozen Lake")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # 创建临时Surface用于所有绘图
        canvas = pygame.Surface((self.ncol * self.cell_size, self.nrow * self.cell_size))
        canvas.fill((255, 255, 255))  # 白色背景
        
        # 颜色定义
        colors = {
            'S': (144, 238, 144),  # 起始位置：浅绿
            'F': (240, 255, 255),  # 冻结表面：白色
            'H': (255, 99, 71),    # 洞穴：红色
            'G': (50, 205, 50)     # 目标：绿色
        }
        
        # 绘制每个单元格
        for i in range(self.nrow):
            for j in range(self.ncol):
                cell_value = self.desc[i][j]
                color = colors.get(cell_value, (255, 255, 255))
                
                # 绘制单元格背景
                pygame.draw.rect(
                    canvas,
                    color,
                    (j * self.cell_size, i * self.cell_size, 
                     self.cell_size, self.cell_size)
                )
                
                # 绘制网格边框
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),  # 黑色边框
                    (j * self.cell_size, i * self.cell_size, 
                     self.cell_size, self.cell_size),
                    width=1
                )
                
                # 在当前位置绘制智能体（黑色圆形）
                if i == self.state // self.ncol and j == self.state % self.ncol:
                    center_x = j * self.cell_size + self.cell_size // 2
                    center_y = i * self.cell_size + self.cell_size // 2
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 0),  # 黑色
                        (center_x, center_y),
                        self.cell_size // 3
                    )
        
        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array模式
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), 
                axes=(1, 0, 2)
            )
    
    def close(self) -> None:
        """清理环境，特别是Pygame资源。"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


# 测试环境是否正确
if __name__ == "__main__":
    env = FrozenLakeEnv(map_name="4x4")
    print("初始地图：")
    env.render(mode="ansi")  # 文本模式显示初始地图
    
    # 重置并开始模拟
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    print(f"初始状态：{state}")
    
    # 执行100步或直到结束
    while not done and step_count < 100:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1
        print(f"动作: {action} -> 新状态: {next_state}, 奖励: {reward}, 结束: {done}")
        
        # 图形渲染
        env.render(mode="human")  # 图形模式
        
    # 关闭环境
    env.close()
    
    # 输出统计结果
    print(f"总步数: {step_count}, 总奖励: {total_reward}")