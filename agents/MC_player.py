import random
import itertools
from collections import Counter
from math import exp
import game.visualize_utils as U
from game.players import BasePokerPlayer


class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self, num_simulations=1000, verbose=False):
        self.num_simulations = num_simulations
        self.verbose = verbose
        self.card_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.card_suits = ['C', 'D', 'H', 'S']  # Clubs, Diamonds, Hearts, Spades

    def declare_action(self, valid_actions, hole_card, round_state):
        win_probability = self._calculate_win_probability(hole_card, round_state)
        action, amount = self._decide_action(valid_actions, win_probability, round_state)

        if self.verbose:
            print(f"Monte Carlo分析:")
            print(f"手牌: {hole_card}")
            print(f"獲勝機率: {win_probability:.2%}")
            print(f"決定動作: {action}, 金額: {amount}")
            print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))

        return action, amount

    def receive_game_start_message(self, game_info):
        if self.verbose:
            print(U.visualize_game_start(game_info, self.uuid))

    def receive_round_start_message(self, round_count, hole_card, seats):
        if self.verbose:
            print(U.visualize_round_start(round_count, hole_card, seats, self.uuid))

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def _calculate_win_probability(self, hole_cards, round_state):
        community_cards = round_state.get('community_card', [])
        num_players = len([seat for seat in round_state.get('seats', []) if seat['state'] != 'folded'])

        wins = 0

        for _ in range(self.num_simulations):
            remaining_deck = self._get_remaining_deck(hole_cards, community_cards)
            opponent_hands = self._simulate_opponent_hands(remaining_deck, num_players - 1)
            simulated_community = self._complete_community_cards(remaining_deck, community_cards)

            my_hand_strength = self._evaluate_hand(hole_cards + simulated_community)
            opponent_strengths = [self._evaluate_hand(hand + simulated_community) for hand in opponent_hands]

            if all(my_hand_strength > opp_strength for opp_strength in opponent_strengths):
                wins += 1

        return wins / self.num_simulations

    def _get_remaining_deck(self, hole_cards, community_cards):
        all_cards = [rank + suit for rank in self.card_ranks for suit in self.card_suits]
        used_cards = set(hole_cards + community_cards)
        return [card for card in all_cards if card not in used_cards]

    def _simulate_opponent_hands(self, deck, num_opponents):
        available_cards = deck.copy()
        opponent_hands = []

        for _ in range(num_opponents):
            if len(available_cards) >= 2:
                hand = random.sample(available_cards, 2)
                opponent_hands.append(hand)
                for card in hand:
                    available_cards.remove(card)

        return opponent_hands

    def _complete_community_cards(self, deck, existing_community):
        cards_needed = 5 - len(existing_community)
        if cards_needed <= 0:
            return existing_community

        available_cards = [card for card in deck if card not in existing_community]
        additional_cards = random.sample(available_cards, min(cards_needed, len(available_cards)))
        return existing_community + additional_cards

    def _evaluate_hand(self, seven_cards):
        if len(seven_cards) < 5:
            return 0

        best_score = 0
        for five_cards in itertools.combinations(seven_cards, 5):
            score = self._score_hand(list(five_cards))
            best_score = max(best_score, score)

        return best_score

    def _score_hand(self, five_cards):
        ranks = [self._rank_to_number(card[0]) for card in five_cards]
        suits = [card[1] for card in five_cards]

        ranks.sort(reverse=True)
        rank_counts = Counter(ranks)
        is_flush = len(set(suits)) == 1
        is_straight = self._is_straight(ranks)

        if is_straight and is_flush:
            if ranks == [14, 13, 12, 11, 10]:
                return 9000 + max(ranks)
            else:
                return 8000 + max(ranks)
        elif 4 in rank_counts.values():
            four_kind = [rank for rank, count in rank_counts.items() if count == 4][0]
            return 7000 + four_kind
        elif 3 in rank_counts.values() and 2 in rank_counts.values():
            three_kind = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair = [rank for rank, count in rank_counts.items() if count == 2][0]
            return 6000 + three_kind * 20 + pair
        elif is_flush:
            return 5000 + sum(ranks)
        elif is_straight:
            return 4000 + max(ranks)
        elif 3 in rank_counts.values():
            three_kind = [rank for rank, count in rank_counts.items() if count == 3][0]
            return 3000 + three_kind
        elif list(rank_counts.values()).count(2) == 2:
            pairs = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)
            return 2000 + pairs[0] * 20 + pairs[1]
        elif 2 in rank_counts.values():
            pair = [rank for rank, count in rank_counts.items() if count == 2][0]
            return 1000 + pair
        else:
            score = 0
            for i, rank in enumerate(ranks):
                score += rank * (10 ** (4 - i))
            return score

    def _rank_to_number(self, rank):
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(rank, 0)

    def _is_straight(self, ranks):
        ranks = sorted(set(ranks), reverse=True)
        if len(ranks) < 5:
            return False

        for i in range(len(ranks) - 4):
            if ranks[i] - ranks[i + 4] == 4:
                return True

        if set([14, 5, 4, 3, 2]).issubset(set(ranks)):
            return True

        return False

    def _decide_action(self, valid_actions, win_probability, round_state):
        fold_action = valid_actions[0]
        call_action = valid_actions[1]
        can_raise = len(valid_actions) > 2 and valid_actions[2]["amount"]["min"] != -1

        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        call_amount = call_action["amount"]
        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0

        # Bluff: 如果勝率略低於 pot_odds，但有機會偷雞
        if win_probability < pot_odds and random.random() < 0.05 and can_raise:
            raise_amount = self._calculate_raise_amount(valid_actions[2], 0.6)
            return valid_actions[2]["action"], raise_amount

        # 正常策略
        margin = win_probability - pot_odds
        if margin > 0.15 and can_raise:
            raise_amount = self._calculate_raise_amount(valid_actions[2], win_probability)
            return valid_actions[2]["action"], raise_amount
        elif margin > -0.05:
            return call_action["action"], call_action["amount"]
        else:
            return fold_action["action"], fold_action["amount"]

    def _calculate_raise_amount(self, raise_action, win_probability):
        min_raise = raise_action["amount"]["min"]
        max_raise = raise_action["amount"]["max"]
        if max_raise == -1:
            max_raise = min_raise * 10

        # 用 sigmoid 平滑決定加注幅度
        scaled = 1 / (1 + exp(-12 * (win_probability - 0.65)))
        return int(min_raise + scaled * (max_raise - min_raise))


def setup_ai():
    return MonteCarloPlayer(num_simulations=1000, verbose=True)
