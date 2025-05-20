import gymnasium as gym
import numpy as np
from collections import deque
import random

class ERSRiskSlapEnv(gym.Env):
    """
    A custom Gymnasium environment for a simplified Egyptian Ratscrew card game,
    focused on modeling the risk-slapping decision.

    Instance Variables:
    -------------------
    burn_penalty : int
        The number of cards a player must burn after an incorrect slap.
    
    deck : list[int]
        A full shuffled 52-card deck represented with values 2–14 (Ace = 14).
    
    action_space : gym.spaces.Discrete
        Action space with two options: 0 (do not slap), 1 (slap).
    
    observation_space : gym.spaces.Box
        Observation space: a 6-dimensional vector representing the top 3 cards
        of the central stack (or 0 if not present), stack size, player hand size,
        and opponent hand size.
    
    player_hand : collections.deque
        The current cards held by the player agent.
    
    opponent_hand : collections.deque
        The current cards held by the opponent bot.
    
    central_stack : collections.deque
        The shared central stack where cards are played and slapped upon.
    """
    def __init__(self, burn_penalty=1):
        super(ERSRiskSlapEnv,  self).__init__()
        self.burn_penalty = burn_penalty
        self.deck = self.generate_deck()
        self.action_space = gym.spaces.Discrete(2)  # 0 = don't slap, 1 = slap
        self.observation_space = gym.spaces.Box(low=0, high=14, shape=(6,), dtype=np.int32)
        self.reset()

    def generate_deck(self):
        return [i for i in range(2, 15)] * 4  # 2–14 (Ace = 14), 4 suits

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        deck = self.generate_deck()
        random.shuffle(deck)
        half = len(deck) // 2
        self.player_hand = deque(deck[:half])
        self.opponent_hand = deque(deck[half:])
        self.central_stack = deque()

        # Opponent plays first card
        self.play_card(self.opponent_hand)
        return self.get_observation(), {}

    def play_card(self, hand):
        if hand:
            card = hand.popleft()
            self.central_stack.append(card)

    def is_valid_combo(self):
        stack = self.central_stack
        if len(stack) < 2:
            return False

        top = stack[-1]
        second = stack[-2]
        third = stack[-3] if len(stack) >= 3 else None
        bottom = stack[0] if stack else None

        # 1. Doubles
        if top == second:
            return True

        # 2. Sandwich
        if third is not None and top == third:
            return True

        # 3. Tens (Ace treated as 1)
        top_val = 1 if top == 14 else top
        second_val = 1 if second == 14 else second
        if top_val + second_val == 10:
            return True

        # 4. Straights
        if third is not None:
            cards = [top, second, third]
            sorted_cards = sorted(cards)
            if sorted_cards[2] - sorted_cards[1] == 1 and sorted_cards[1] - sorted_cards[0] == 1:
                return True

        # 5. Top-Bottom
        if top == bottom and len(stack) >= 2:
            return True

        # 6. Marriage
        if (top == 13 and second == 12) or (top == 12 and second == 13):
            return True

        return False

    def get_observation(self):
        top = self.central_stack[-1] if len(self.central_stack) >= 1 else 0
        second = self.central_stack[-2] if len(self.central_stack) >= 2 else 0
        third = self.central_stack[-3] if len(self.central_stack) >= 3 else 0
        return np.array([top, second, third, len(self.central_stack),
                         len(self.player_hand), len(self.opponent_hand)], dtype=np.int32)

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if action == 1:
            self.play_card(self.opponent_hand)
            self.play_card(self.player_hand)

            if self.is_valid_combo():
                self.player_hand.extend(self.central_stack)
                self.central_stack.clear()
                reward = 1
            else:
                burn_count = min(self.burn_penalty, len(self.player_hand))
                for _ in range(burn_count):
                    if self.player_hand:
                        self.central_stack.append(self.player_hand.popleft())
                reward = -self.burn_penalty

        else:
            self.play_card(self.opponent_hand)
            self.play_card(self.player_hand)

            if self.is_valid_combo():
                self.opponent_hand.extend(self.central_stack)
                self.central_stack.clear()

        if not self.player_hand and not self.opponent_hand:
            terminated = True
            reward = 0
        elif not self.player_hand:
            terminated = True
            reward = -5
        elif not self.opponent_hand:
            terminated = True
            reward = 5

        return self.get_observation(), reward, terminated, truncated, info
