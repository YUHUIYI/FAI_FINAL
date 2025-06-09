from game.players import BasePokerPlayer

class HeuristicPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        hole_strength = self._evaluate_hole_card(hole_card)

        if hole_strength >= 20:
            raise_action_info = valid_actions[2]
            action, amount = raise_action_info["action"], raise_action_info["amount"]
            if isinstance(amount, dict):
                amount = amount["max"]
            return action, amount
        elif hole_strength >= 14:
            call_action_info = valid_actions[1]
            action, amount = call_action_info["action"], call_action_info["amount"]
            return action, amount
        else:
            fold_action_info = valid_actions[0]
            action, amount = fold_action_info["action"], fold_action_info["amount"]
            return action, amount

    def _evaluate_hole_card(self, hole_card):
        card_rank_dict = {
            "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
            "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
        }
        ranks = [card_rank_dict[c[1]] for c in hole_card]
        return sum(ranks)

    def receive_game_start_message(self, game_info):
        print(f"[Agent] I am using HeuristicPlayer.")

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return HeuristicPlayer()
