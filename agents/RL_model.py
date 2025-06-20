import numpy as np
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game.engine.dealer import Dealer
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
from RL_player import RLAgentPlayer  

class PokerEnv:
    def __init__(self, agent=None):
        # 定義 action space: 0 = fold, 1 = call, 2 = raise
        self.num_actions = 3
        self.obs_dim = 5

        self.dealer = Dealer(small_blind_amount=20, initial_stack=1000)

        if agent is None:
            self.agent = RLAgentPlayer()
        else:
            self.agent = agent

        # baseline ai 列表
        self.baseline_ai_list = [
            baseline0_ai,
            baseline1_ai,
            baseline2_ai,
            baseline3_ai,
            baseline4_ai,
            baseline5_ai,
            baseline6_ai,
            baseline7_ai
        ]

        # 一開始就隨機選一個 opponent
        self.opponent_ai = random.choice(self.baseline_ai_list)
        self.opponent = self.opponent_ai()

        self.dealer.register_player("agent", self.agent)
        self.dealer.register_player("opponent", self.opponent)

    def reset(self):
        # Reset agent internal state
        self.agent.reset_episode()

        # Reset Dealer (要重新建立 Dealer 才能 reset 遊戲)
        self.dealer = Dealer(small_blind_amount=20, initial_stack=1000)

        # 每次 reset → 隨機選一個 opponent
        self.opponent_ai = random.choice(self.baseline_ai_list)
        self.opponent = self.opponent_ai()

        self.dealer.register_player("agent", self.agent)
        self.dealer.register_player("opponent", self.opponent)

        # Run one game to deal cards and set initial obs
        self.dealer.set_verbose(0)
        self.dealer.start_game(max_round=1)

        obs = self.agent.current_obs
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # Set action for agent
        self.agent.set_next_action(action)

        # Run one game step → 目前設計是一整局打一局
        self.dealer.set_verbose(0)
        self.dealer.start_game(max_round=1)

        # reward = 最終 stack 差異
        final_stack = self.agent.final_stack
        reward = final_stack - 1000  # initial stack
        done = True

        obs = self.agent.current_obs

        return np.array(obs, dtype=np.float32), reward, done, {}
