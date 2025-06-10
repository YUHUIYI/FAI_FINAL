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
        """评估7张牌中最好的5张牌组合"""
        if len(seven_cards) < 5:
            return 0

        # 定义牌型常量
        HIGHCARD = 0
        ONEPAIR = 1 << 8
        TWOPAIR = 1 << 9
        THREECARD = 1 << 10
        STRAIGHT = 1 << 11
        FLASH = 1 << 12
        FULLHOUSE = 1 << 13
        FOURCARD = 1 << 14
        STRAIGHTFLASH = 1 << 15

        best_score = 0
        # 从7张牌中选出5张牌的所有组合
        for five_cards in itertools.combinations(seven_cards, 5):
            score = self._score_hand(list(five_cards))
            best_score = max(best_score, score)

        return best_score

    def _score_hand(self, five_cards):
        """为5张牌组合评分"""
        # 定义牌型常量
        HIGHCARD = 0
        ONEPAIR = 1 << 8
        TWOPAIR = 1 << 9
        THREECARD = 1 << 10
        STRAIGHT = 1 << 11
        FLASH = 1 << 12
        FULLHOUSE = 1 << 13
        FOURCARD = 1 << 14
        STRAIGHTFLASH = 1 << 15

        # 获取牌面值和花色
        ranks = [self._rank_to_number(card[0]) for card in five_cards]
        suits = [card[1] for card in five_cards]

        # 按牌面值排序
        ranks.sort(reverse=True)
        rank_counts = Counter(ranks)
        
        # 检查同花
        is_flush = len(set(suits)) == 1
        
        # 检查顺子
        is_straight = self._is_straight(ranks)

        # 计算牌型分数
        if is_straight and is_flush:
            # 同花顺
            if ranks == [14, 13, 12, 11, 10]:  # 皇家同花顺
                return STRAIGHTFLASH | (14 << 4)
            else:
                return STRAIGHTFLASH | (max(ranks) << 4)
        elif 4 in rank_counts.values():
            # 四条
            four_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            kicker = [rank for rank in ranks if rank != four_rank][0]
            return FOURCARD | (four_rank << 4) | kicker
        elif 3 in rank_counts.values() and 2 in rank_counts.values():
            # 葫芦
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            return FULLHOUSE | (three_rank << 4) | pair_rank
        elif is_flush:
            # 同花
            return FLASH | (ranks[0] << 4) | (ranks[1] << 3) | (ranks[2] << 2) | (ranks[3] << 1) | ranks[4]
        elif is_straight:
            # 顺子
            return STRAIGHT | (max(ranks) << 4)
        elif 3 in rank_counts.values():
            # 三条
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            kickers = sorted([rank for rank in ranks if rank != three_rank], reverse=True)
            return THREECARD | (three_rank << 4) | (kickers[0] << 3) | (kickers[1] << 2)
        elif list(rank_counts.values()).count(2) == 2:
            # 两对
            pairs = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)
            kicker = [rank for rank in ranks if rank not in pairs][0]
            return TWOPAIR | (pairs[0] << 4) | (pairs[1] << 3) | kicker
        elif 2 in rank_counts.values():
            # 一对
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            kickers = sorted([rank for rank in ranks if rank != pair_rank], reverse=True)
            return ONEPAIR | (pair_rank << 4) | (kickers[0] << 3) | (kickers[1] << 2) | (kickers[2] << 1)
        else:
            # 高牌
            return HIGHCARD | (ranks[0] << 4) | (ranks[1] << 3) | (ranks[2] << 2) | (ranks[3] << 1) | ranks[4]

    def _rank_to_number(self, rank):
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(rank, 0)

    def _is_straight(self, ranks):
        """检查是否为顺子"""
        ranks = sorted(set(ranks), reverse=True)
        if len(ranks) < 5:
            return False

        # 检查普通顺子
        for i in range(len(ranks) - 4):
            if ranks[i] - ranks[i+4] == 4:
                return True

        # 检查A-2-3-4-5顺子
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
            raise_amount = self._calculate_raise_amount(valid_actions[2], 0.6, round_state)
            return valid_actions[2]["action"], raise_amount

        # 正常策略
        margin = win_probability - pot_odds
        if margin > 0.15 and can_raise:
            raise_amount = self._calculate_raise_amount(valid_actions[2], win_probability, round_state)
            return valid_actions[2]["action"], raise_amount
        elif margin > -0.05:
            return call_action["action"], call_action["amount"]
        else:
            return fold_action["action"], fold_action["amount"]

    def _calculate_raise_amount(self, raise_action, win_probability, round_state):
        min_raise = raise_action["amount"]["min"]
        max_raise = raise_action["amount"]["max"]
        if max_raise == -1:
            max_raise = min_raise * 10

        # 取得我方與其他玩家的籌碼量
        my_stack = 0
        opp_max_stack = 0
        for seat in round_state.get('seats', []):
            if seat['uuid'] == self.uuid:
                my_stack = seat['stack']
            elif seat['state'] != 'folded':
                opp_max_stack = max(opp_max_stack, seat['stack'])

        # 比例計算
        if opp_max_stack > 0:
            stack_ratio = my_stack / opp_max_stack
            stack_ratio = max(0.1, min(stack_ratio, 5.0))  # 限制範圍避免極端
        else:
            stack_ratio = 1.0

        # stack_ratio 越大 → 越保守，越小 → 越激進
        aggressiveness_factor = 2 / stack_ratio  # 高 stack_ratio 時 factor < 1，低 stack_ratio 時 factor > 1
        aggressiveness_factor = max(0.5, min(aggressiveness_factor, 2.0))

        # 用 sigmoid 平滑決定加注幅度
        scaled = 1 / (1 + exp(-12 * (win_probability - 0.65)))

        # 最終 raise
        target_raise = min_raise + scaled * (max_raise - min_raise)
        target_raise *= aggressiveness_factor
        target_raise = max(min_raise, min(target_raise, max_raise))

        return int(target_raise)


def setup_ai():
    return MonteCarloPlayer(num_simulations=1000, verbose=True)