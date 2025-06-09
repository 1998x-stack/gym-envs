import gym
from gym import spaces
from typing import Tuple, Optional
import random

class ToyTextBlackjackEnv(gym.Env):
    """
    一个使用Gym接口实现的21点(Blackjack)游戏环境。
    
    游戏规则:
    - 玩家和庄家初始各发两张牌，庄家一张牌面向上(可见)，一张牌面向下(不可见)
    - 玩家可以"要牌"(hit=1)或"停牌"(stand=0)
    - 玩家目标：使手牌点数接近21点而不超过
    - A可计为1或11，J/Q/K计为10
    - 玩家停牌后，庄家必须持续要牌直到点数≥17
    - 胜负判定：
        - 玩家爆牌(>21)：庄家胜
        - 庄家爆牌：玩家胜
        - 双方均未爆牌：点数大者胜
        - 点数相同：平局
    - 初始发牌后玩家直接获得21点（A+10）称为"自然黑杰克"
        - 若庄家也是自然黑杰克：平局
        - 否则：玩家立即获胜
    
    观察空间: (玩家当前点数, 庄家明牌, 玩家是否有可用A)
    动作空间: 0(停牌) 或 1(要牌)
    奖励: 赢+1, 输-1, 平局0
    """

    def __init__(self) -> None:
        """初始化21点游戏环境"""
        super().__init__()
        
        # 定义动作空间：0=停牌, 1=要牌
        self.action_space = spaces.Discrete(2)
        
        # 定义观察空间：三元组(玩家点数, 庄家明牌, 是否有可用A)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),   # 玩家点数范围0-31
            spaces.Discrete(11),   # 庄家明牌值1-10（0位置不使用）
            spaces.Discrete(2)     # 是否有可用A: 0或1
        ))
        
        # 游戏状态变量
        self.player_hand: list = []       # 玩家手牌
        self.dealer_hand: list = []        # 庄家手牌
        self.done: bool = True             # 游戏是否结束标志
        
        # 初始化随机数生成器
        self.seed()

    def seed(self, seed: Optional[int] = None) -> list:
        """设置随机数生成器种子"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[int, int, int]:
        """重置游戏状态并返回初始观察"""
        # 清空手牌
        self.player_hand = []
        self.dealer_hand = []
        self.done = False
        
        # 初始发牌：玩家2张，庄家2张（1张明牌）
        self.player_hand.extend([self._draw_card(), self._draw_card()])
        self.dealer_hand.extend([self._draw_card(), self._draw_card()])
        
        # 检查玩家是否自然黑杰克
        player_total, usable_ace = self._calculate_hand_value(self.player_hand)
        if player_total == 21 and len(self.player_hand) == 2:
            dealer_total, _ = self._calculate_hand_value(self.dealer_hand)
            # 庄家也是自然黑杰克则平局，否则玩家获胜
            if dealer_total == 21 and len(self.dealer_hand) == 2:
                self.done = True
            else:
                self.done = True
        
        return self._get_observation()

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, dict]:
        """
        执行玩家动作并返回环境反馈
        
        参数:
            action: 0=停牌, 1=要牌
        
        返回:
            observation: 新状态 (玩家点数, 庄家明牌, 可用A)
            reward: 奖励值
            done: 游戏是否结束
            info: 附加信息
        """
        assert self.action_space.contains(action), "无效动作"
        if self.done:
            raise ValueError("游戏已结束，请重置环境")
        
        # 玩家选择要牌
        if action == 1:
            self.player_hand.append(self._draw_card())
            player_total, _ = self._calculate_hand_value(self.player_hand)
            
            # 玩家爆牌，游戏结束
            if player_total > 21:
                self.done = True
                return self._get_observation(), -1.0, True, {}
            # 玩家未爆牌，游戏继续
            return self._get_observation(), 0.0, False, {}
        
        # 玩家选择停牌，庄家回合开始
        self.done = True
        dealer_total, _ = self._calculate_hand_value(self.dealer_hand)
        
        # 庄家要牌规则：点数<17必须要牌
        while dealer_total < 17:
            self.dealer_hand.append(self._draw_card())
            dealer_total, _ = self._calculate_hand_value(self.dealer_hand)
        
        # 计算最终点数
        player_total, _ = self._calculate_hand_value(self.player_hand)
        
        # 庄家爆牌：玩家胜
        if dealer_total > 21:
            return self._get_observation(), 1.0, True, {}
        
        # 点数比较
        if player_total > dealer_total:
            return self._get_observation(), 1.0, True, {}
        elif player_total < dealer_total:
            return self._get_observation(), -1.0, True, {}
        else:
            return self._get_observation(), 0.0, True, {}

    def render(self, mode: str = 'human') -> None:
        """以文本方式渲染当前游戏状态"""
        player_total, usable_ace = self._calculate_hand_value(self.player_hand)
        print(f"玩家手牌: {self.player_hand} 点数={player_total} 可用A={usable_ace}")
        
        if self.done:
            dealer_total, _ = self._calculate_hand_value(self.dealer_hand)
            print(f"庄家手牌: {self.dealer_hand} 点数={dealer_total}")
        else:
            print(f"庄家手牌: [{self.dealer_hand[0]}, ?]")

    def _draw_card(self) -> int:
        """随机抽取一张牌并返回牌值 (1-10)"""
        card = random.randint(1, 13)  # 1=A, 11/12/13=J/Q/K
        return min(card, 10)  # J/Q/K计为10

    def _calculate_hand_value(self, hand: list) -> Tuple[int, int]:
        """
        计算手牌点数和可用A标志
        
        参数:
            hand: 手牌列表
        
        返回:
            total: 手牌最佳点数
            usable_ace: 是否有可用A (0/1)
        """
        total = 0
        num_aces = 0
        
        # 第一轮计算：A计为11，其他按面值
        for card in hand:
            if card == 1:  # A
                total += 11
                num_aces += 1
            else:
                total += card
        
        # 调整A值：若爆牌且手中有A，将A从11改为1
        while total > 21 and num_aces > 0:
            total -= 10  # 将1个A从11改为1
            num_aces -= 1
        
        # 检查是否有仍计为11的A
        usable_ace = 1 if num_aces > 0 else 0
        return total, usable_ace

    def _get_observation(self) -> Tuple[int, int, int]:
        """获取当前观察值：(玩家点数, 庄家明牌, 可用A)"""
        player_total, usable_ace = self._calculate_hand_value(self.player_hand)
        dealer_showing = self.dealer_hand[0]  # 庄家第一张牌（明牌）
        return (player_total, dealer_showing, usable_ace)


def verify_environment() -> None:
    """验证环境实现正确性的测试函数"""
    env = ToyTextBlackjackEnv()
    env.seed(42)  # 固定随机种子
    
    print("="*50)
    print("开始环境验证测试...")
    print("="*50)
    
    # 测试1：环境初始化
    state = env.reset()
    print("\n测试1: 环境初始化")
    print(f"初始状态: {state}")
    env.render()
    assert len(env.player_hand) == 2, "玩家应有2张牌"
    assert len(env.dealer_hand) == 2, "庄家应有2张牌"
    
    # 测试2：玩家爆牌
    print("\n测试2: 玩家爆牌")
    env.reset()
    # 强制设置手牌使玩家爆牌
    env.player_hand = [10, 5, 7]  # 10+5+7=22
    state, reward, done, _ = env.step(1)  # 要牌动作
    env.render()
    print(f"状态: {state}, 奖励: {reward}, 结束: {done}")
    assert reward == -1 and done, "玩家爆牌应输-1"
    
    # 测试3：庄家爆牌
    print("\n测试3: 庄家爆牌")
    env.reset()
    env.player_hand = [10, 9]  # 19点
    env.dealer_hand = [10, 5, 7]  # 10+5+7=22
    state, reward, done, _ = env.step(0)  # 停牌
    env.render()
    print(f"状态: {state}, 奖励: {reward}, 结束: {done}")
    assert reward == 1 and done, "庄家爆牌玩家应赢+1"
    
    # 测试4：自然黑杰克（玩家）
    print("\n测试4: 玩家自然黑杰克")
    env.reset()
    env.player_hand = [1, 10]  # A+10=21
    env.dealer_hand = [9, 5]    # 14点
    state, reward, done, _ = env.step(0)  # 任何动作都结束
    env.render()
    print(f"状态: {state}, 奖励: {reward}, 结束: {done}")
    assert reward == 1 and done, "玩家自然黑杰克应赢+1"
    
    # 测试5：双方自然黑杰克
    print("\n测试5: 双方自然黑杰克")
    env.reset()
    env.player_hand = [1, 10]  # 21
    env.dealer_hand = [1, 10]  # 21
    state, reward, done, _ = env.step(0)
    env.render()
    print(f"状态: {state}, 奖励: {reward}, 结束: {done}")
    assert reward == 0 and done, "双方自然黑杰克应平局"
    
    # 测试6：点数比较
    print("\n测试6: 点数比较 (玩家胜)")
    env.reset()
    env.player_hand = [10, 8]  # 18
    env.dealer_hand = [10, 7]  # 17
    state, reward, done, _ = env.step(0)
    env.render()
    print(f"状态: {state}, 奖励: {reward}, 结束: {done}")
    assert reward == 1 and done, "玩家18>庄家17应赢+1"
    
    print("\n" + "="*50)
    print("所有基础测试通过！")
    print("="*50)


if __name__ == "__main__":
    # 启动环境验证
    verify_environment()