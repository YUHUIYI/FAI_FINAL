# MC_player.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import copy
from game.players import BasePokerPlayer
from game.engine.dealer import Dealer
from baseline0 import setup_ai as baseline0_ai

class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self, num_simulations=50):
        self.num_simulations = num_simulations

    def declare_action(self, valid_actions, hole_card, round_state):
        # 對每個 action 模擬 N 次 → 算 EV
        action_EVs = []
        for action_info in valid_actions:
            action = action_info["action"]
            amount = action_info["amount"]

            ev = self._estimate_action_EV(action, amount, hole_card, round_state)
            action_EVs.append((action, amount, ev))

        # 選 EV 最大的 action 出
        best_action = max(action_EVs, key=lambda x: x[2])
        print(f"[MC Player] Action EVs: {action_EVs}, Selected: {best_action}")
        return best_action[0], best_action[1]

    def _estimate_action_EV(self, action, amount, hole_card, round_state):
        total_reward = 0
        for _ in range(self.num_simulations):
            # 每次模擬一局
            reward = self._simulate_game(action, amount, hole_card, round_state)
            total_reward += reward

        avg_reward = total_reward / self.num_simulations
        return avg_reward

    def _simulate_game(self, action, amount, hole_card, round_state):
        # 初始化 dealer → 用 baseline0_ai 當對手 → 簡單可靠 baseline
        dealer = Dealer(small_blind_amount=20, initial_stack=1000)

        # 自己用一個 DummyPlayer → 只會固定做 action
        mc_dummy_player = FixedActionPlayer(action, amount)
        opponent = baseline0_ai()

        dealer.register_player("mc_dummy_player", mc_dummy_player)
        dealer.register_player("opponent", opponent)

        dealer.set_verbose(0)
        dealer.start_game(max_round=1)

        # reward = 自己 final stack - initial stack
        final_stack = mc_dummy_player.final_stack
        reward = final_stack - 1000
        return reward

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

# 一個固定做某個 action 的 Dummy Player → 給 Monte Carlo 模擬用
class FixedActionPlayer(BasePokerPlayer):
    def __init__(self, fixed_action, fixed_amount):
        self.fixed_action = fixed_action
        self.fixed_amount = fixed_amount
        self.final_stack = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        return self.fixed_action, self.fixed_amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        for player in round_state["seats"]:
            if player["uuid"] == self.uuid:
                self.final_stack = player["stack"]

def setup_ai():
    return MonteCarloPlayer(num_simulations=50)  # 你可以調 N 模擬次數
