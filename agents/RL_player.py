from game.players import BasePokerPlayer

class RLAgentPlayer(BasePokerPlayer):
    def __init__(self):
        self.next_action = 1  # 預設 call
        self.current_obs = None
        self.final_stack = 0

    def set_next_action(self, action_idx):
        self.next_action = action_idx

    def reset_episode(self):
        self.current_obs = None
        self.final_stack = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        # 這裡將 round_state + hole_card 轉 obs
        obs = self._state_to_feature(hole_card, round_state)
        self.current_obs = obs

        # 直接用 self.next_action 選 action
        action_info = valid_actions[self.next_action]
        action, amount = action_info["action"], action_info["amount"]

        if action == "raise" and isinstance(amount, dict):
            amount = amount["max"]

        return action, amount

    def _state_to_feature(self, hole_card, round_state):
        # 同前例
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

    def setup_ai():
        return RLAgentPlayer()