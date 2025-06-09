import gym
from gym import spaces
import numpy as np
from game.engine.dealer import Dealer
from baseline0 import setup_ai as baseline0_ai
from RL_player import RLAgentPlayer  


class PokerEnv(gym.Env):
    def __init__(self):
        super(PokerEnv, self).__init__()

        # Action space: 0 = fold, 1 = call, 2 = raise
        self.action_space = spaces.Discrete(3)

        # Observation space: 自訂，簡單設為 5 維
        self.observation_space = spaces.Box(low=0, high=10000, shape=(5,), dtype=np.float32)

        self.dealer = Dealer(small_blind_amount=20, initial_stack=1000)
        self.agent = RLAgentPlayer()  # 你要訓練的 agent
        self.opponent = baseline0_ai() # baseline 對手

        self.dealer.register_player("agent", self.agent)
        self.dealer.register_player("opponent", self.opponent)

    def reset(self):
        self.agent.reset_episode()
        self.dealer.set_verbose(0)
        self.dealer.start_game(max_round=1)
        obs = self.agent.current_obs
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # 這裡 agent 會用 action 決定 respond_to_ask 裡的行為
        self.agent.set_next_action(action)
        
        # 讓 dealer 自己跑完這一局
        self.dealer.start_game(max_round=1)

        # reward = 最終 stack 差異
        final_stack = self.agent.final_stack
        reward = final_stack - 1000  # initial stack
        done = True
        obs = self.agent.current_obs  # 可用不上

        return np.array(obs, dtype=np.float32), reward, done, {}

