from game.players import BasePokerPlayer
import torch
import torch.nn as nn
import numpy as np

# 和 train_RL.py 裡的 PolicyNet 保持一致
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RLAgentPlayer(BasePokerPlayer):
    def __init__(self, model_path=None):
        self.next_action = 1  # 預設 call
        self.current_obs = None
        self.final_stack = 0

        # 如果有 model_path → 載入 model
        if model_path:
            self.policy_net = PolicyNet(obs_dim=5, num_actions=3)
            self.policy_net.load_state_dict(torch.load(model_path))
            self.policy_net.eval()
        else:
            self.policy_net = None

    def set_next_action(self, action_idx):
        self.next_action = action_idx

    def reset_episode(self):
        self.current_obs = None
        self.final_stack = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        obs = self._state_to_feature(hole_card, round_state)
        self.current_obs = obs

        if self.policy_net:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(obs_tensor)
            action_idx = torch.argmax(q_values).item()
        else:
            action_idx = self.next_action

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
        for player in round_state["seats"]:
            if player["name"] == "agent":
                self.final_stack = player["stack"]
