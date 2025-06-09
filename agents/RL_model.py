import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from game.engine.dealer import Dealer
from baseline0 import setup_ai as baseline0_ai
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

        self.opponent = baseline0_ai()

        self.dealer.register_player("agent", self.agent)
        self.dealer.register_player("opponent", self.opponent)


