"""Environment contracts: two complete games, seed isolation and terminal safety."""
import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from toy_text.blackjack import ToyTextBlackjackEnv
from toy_text.frozen_lake import FrozenLakeEnv


class GymnasiumContracts(unittest.TestCase):
    def test_frozen_lake_fixed_path_reward_and_post_terminal(self):
        env=FrozenLakeEnv(map_name="4x4", is_slippery=False,render_mode="ansi")
        obs, info=env.reset(seed=42)
        self.assertEqual(obs,0)
        self.assertIn("A",env.render())
        for action in [1,1,0,1,0,1]:  # down, down, right, down, right, down
            obs, reward, terminated, truncated, info=env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(reward,1.0)
        with self.assertRaises(RuntimeError): env.step(0)
        env.close()

    def test_slippery_seed_and_global_rng_isolation(self):
        import random
        random.seed(31)
        baseline=random.random()
        random.seed(31)
        left=FrozenLakeEnv(is_slippery=True)
        right=FrozenLakeEnv(is_slippery=True)
        left.reset(seed=11);right.reset(seed=11)
        a=[];b=[]
        for _ in range(20):
            a.append(left.step(0)[0]);b.append(right.step(0)[0])
            if left._terminated or right._terminated: break
        self.assertEqual(a,b)
        self.assertEqual(random.random(),baseline)

    def test_blackjack_seed_and_terminal_behavior(self):
        first=ToyTextBlackjackEnv()
        second=ToyTextBlackjackEnv()
        obs,_=first.reset(seed=3)
        same,_=second.reset(seed=3)
        self.assertEqual(obs,same)
        self.assertEqual(first.dealer_hand,second.dealer_hand)
        _,reward,done,truncated,_=first.step(0)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertIn(reward,(-1.0,0.0,1.0))
        with self.assertRaises(RuntimeError): first.step(1)
        first.reset(seed=3)
        self.assertEqual(first.player_hand,second.player_hand)

    def test_blackjack_ace_values(self):
        score=ToyTextBlackjackEnv._calculate_hand_value
        self.assertEqual(score([1,10]),(21,1))
        self.assertEqual(score([1,1,9]),(21,1))
        self.assertEqual(score([1,1,10,10]),(22,0))

    def test_env_checker(self):
        check_env(FrozenLakeEnv(is_slippery=False),skip_render_check=True)
        check_env(ToyTextBlackjackEnv(),skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
