# RF_player.py
import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game.players import BasePokerPlayer

class RFPlayer(BasePokerPlayer):
    def __init__(self):
        self.observation_history = []
        self.opponent_action_history = []

    def declare_action(self, valid_actions, hole_card, round_state):
        # 記錄 obs → feature
        obs = self._state_to_feature(hole_card, round_state)
        self.observation_history.append(obs)

        # 自己隨便選，不重要 → 因為要學 opponent 的行為
        action_info = valid_actions[1]  # call
        return action_info["action"], action_info["amount"]

    def receive_game_update_message(self, action, round_state):
        # 如果是 opponent 出 action → 記錄下來當 label
        if action["player_uuid"] != self.uuid:
            action_map = {"fold": 0, "call": 1, "raise": 2}
            action_idx = action_map.get(action["action"], 1)
            self.opponent_action_history.append(action_idx)

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

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return RFPlayer()
