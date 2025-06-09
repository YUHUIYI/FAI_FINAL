# RF_player.py
import sys
import os
import pickle
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game.players import BasePokerPlayer

class RFPlayer(BasePokerPlayer):
    def __init__(self, model_path="poker_policy_rf.pkl"):
        self.observation_history = []
        self.action_history = []
        try:
            with open(model_path, "rb") as f:
                self.rf_clf = pickle.load(f)
        except FileNotFoundError:
            # 如果模型文件不存在，创建一个新的随机森林分类器
            from sklearn.ensemble import RandomForestClassifier
            self.rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    def declare_action(self, valid_actions, hole_card, round_state):
        obs = self._state_to_feature(hole_card, round_state)
        self.observation_history.append(obs)  # 记录观察
        
        obs = np.array(obs).reshape(1, -1)
        action_idx = self.rf_clf.predict(obs)[0]
        self.action_history.append(action_idx)  # 记录动作

        action_info = valid_actions[action_idx]
        action, amount = action_info["action"], action_info["amount"]

        if action == "raise" and isinstance(amount, dict):
            amount = amount["max"]

        return action, amount

    def _state_to_feature(self, hole_card, round_state):
        card_rank_dict = {
            "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
            "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
        }
        hole_ranks = [card_rank_dict[c[1]] for c in hole_card]
        pot_size = round_state["pot"]["main"]["amount"]
        num_players = len(round_state["seats"])
        street_dict = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
        street = round_state["street"]
        street_num = street_dict.get(street, 0)
        return hole_ranks + [pot_size, num_players, street_num]

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

def setup_ai():
    return RFPlayer()
