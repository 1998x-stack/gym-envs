"""Seedable Gymnasium Blackjack teaching environment, no global RNG."""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces


class ToyTextBlackjackEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("unsupported render mode")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)  # 0 stand, 1 hit
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2)
        ))
        self.player_hand = []
        self.dealer_hand = []
        self._terminated = True

    def _draw_card(self):
        return min(int(self.np_random.integers(1, 14)), 10)

    @staticmethod
    def _calculate_hand_value(hand):
        total = sum(11 if card == 1 else card for card in hand)
        usable = sum(card == 1 for card in hand)
        while total > 21 and usable:
            total -= 10
            usable -= 1
        return total, int(usable > 0)

    def _obs(self):
        total, usable = self._calculate_hand_value(self.player_hand)
        return (total, self.dealer_hand[0], usable)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        self._terminated = False
        # Natural blackjack is adjudicated on stand, not during reset.
        return self._obs(), {}

    def step(self, action):
        if not self.player_hand:
            raise RuntimeError("call reset() before step()")
        if self._terminated:
            raise RuntimeError("episode ended; call reset()")
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        if int(action) == 1:
            self.player_hand.append(self._draw_card())
            bust = self._calculate_hand_value(self.player_hand)[0] > 21
            self._terminated = bust
            return self._obs(), (-1.0 if bust else 0.0), bust, False, {}
        self._terminated = True
        dealer_total = self._calculate_hand_value(self.dealer_hand)[0]
        while dealer_total < 17:
            self.dealer_hand.append(self._draw_card())
            dealer_total = self._calculate_hand_value(self.dealer_hand)[0]
        player_total = self._calculate_hand_value(self.player_hand)[0]
        reward = 1.0 if dealer_total > 21 or player_total > dealer_total else (
            -1.0 if player_total < dealer_total else 0.0)
        return self._obs(), reward, True, False, {}

    def render(self):
        if self.render_mode is None:
            return None
        total, usable = self._calculate_hand_value(self.player_hand)
        other = self.dealer_hand if self._terminated else [self.dealer_hand[0], "?"]
        msg = f"Player {self.player_hand}: {total} (usable ace={usable}); dealer {other}"
        if self.render_mode == "human":
            print(msg)
            return None
        return msg
