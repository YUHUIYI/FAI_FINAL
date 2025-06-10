import random
import itertools
from collections import Counter
import game.visualize_utils as U
from game.players import BasePokerPlayer


class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self, num_simulations=1000, verbose=False):
        self.num_simulations = num_simulations
        self.verbose = verbose
        self.card_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.card_suits = ['C', 'D', 'H', 'S']  # Clubs, Diamonds, Hearts, Spades
        
    def declare_action(self, valid_actions, hole_card, round_state):
        # 計算獲勝機率
        win_probability = self._calculate_win_probability(hole_card, round_state)
        
        # 計算期望值並決定動作
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
        #if self.verbose:
           # print(U.visualize_street_start(street, round_state, self.uuid))
           pass

    def receive_game_update_message(self, new_action, round_state):
        #if self.verbose:
           # print(U.visualize_game_update(new_action, round_state, self.uuid))
           pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        #if self.verbose:
            #print(U.visualize_round_result(winners, hand_info, round_state, self.uuid))
            pass

    def _calculate_win_probability(self, hole_cards, round_state):
        """使用Monte Carlo模擬計算獲勝機率"""
        community_cards = round_state.get('community_card', [])
        num_players = len([seat for seat in round_state.get('seats', []) if seat['state'] != 'folded'])
        
        wins = 0
        
        for _ in range(self.num_simulations):
            # 創建剩餘牌組
            remaining_deck = self._get_remaining_deck(hole_cards, community_cards)
            
            # 模擬對手手牌
            opponent_hands = self._simulate_opponent_hands(remaining_deck, num_players - 1)
            
            # 完成公共牌
            simulated_community = self._complete_community_cards(remaining_deck, community_cards)
            
            # 評估手牌強度
            my_hand_strength = self._evaluate_hand(hole_cards + simulated_community)
            opponent_strengths = [
                self._evaluate_hand(hand + simulated_community) 
                for hand in opponent_hands
            ]
            
            # 檢查是否獲勝
            if all(my_hand_strength > opp_strength for opp_strength in opponent_strengths):
                wins += 1
        
        return wins / self.num_simulations

    def _get_remaining_deck(self, hole_cards, community_cards):
        """獲取剩餘牌組"""
        all_cards = [rank + suit for rank in self.card_ranks for suit in self.card_suits]
        used_cards = set(hole_cards + community_cards)
        return [card for card in all_cards if card not in used_cards]

    def _simulate_opponent_hands(self, deck, num_opponents):
        """模擬對手手牌"""
        available_cards = deck.copy()
        opponent_hands = []
        
        for _ in range(num_opponents):
            if len(available_cards) >= 2:
                hand = random.sample(available_cards, 2)
                opponent_hands.append(hand)
                # 移除已使用的牌
                for card in hand:
                    available_cards.remove(card)
        
        return opponent_hands

    def _complete_community_cards(self, deck, existing_community):
        """完成公共牌到5張"""
        cards_needed = 5 - len(existing_community)
        if cards_needed <= 0:
            return existing_community
        
        available_cards = [card for card in deck if card not in existing_community]
        additional_cards = random.sample(available_cards, min(cards_needed, len(available_cards)))
        return existing_community + additional_cards

    def _evaluate_hand(self, seven_cards):
        """評估7張牌中最好的5張牌組合"""
        if len(seven_cards) < 5:
            return 0
        
        best_score = 0
        # 從7張牌中選出5張牌的所有組合
        for five_cards in itertools.combinations(seven_cards, 5):
            score = self._score_hand(list(five_cards))
            best_score = max(best_score, score)
        
        return best_score

    def _score_hand(self, five_cards):
        """為5張牌組合評分"""
        ranks = [self._rank_to_number(card[0]) for card in five_cards]
        suits = [card[1] for card in five_cards]
        
        ranks.sort(reverse=True)
        rank_counts = Counter(ranks)
        is_flush = len(set(suits)) == 1
        is_straight = self._is_straight(ranks)
        
        # 手牌類型評分 (數字越大越好)
        if is_straight and is_flush:
            if ranks == [14, 13, 12, 11, 10]:  # 皇家同花順
                return 9000 + max(ranks)
            else:  # 同花順
                return 8000 + max(ranks)
        elif 4 in rank_counts.values():  # 四條
            four_kind = [rank for rank, count in rank_counts.items() if count == 4][0]
            return 7000 + four_kind
        elif 3 in rank_counts.values() and 2 in rank_counts.values():  # 葫蘆
            three_kind = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair = [rank for rank, count in rank_counts.items() if count == 2][0]
            return 6000 + three_kind * 20 + pair
        elif is_flush:  # 同花
            return 5000 + sum(ranks)
        elif is_straight:  # 順子
            return 4000 + max(ranks)
        elif 3 in rank_counts.values():  # 三條
            three_kind = [rank for rank, count in rank_counts.items() if count == 3][0]
            return 3000 + three_kind
        elif list(rank_counts.values()).count(2) == 2:  # 兩對
            pairs = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)
            return 2000 + pairs[0] * 20 + pairs[1]
        elif 2 in rank_counts.values():  # 一對
            pair = [rank for rank, count in rank_counts.items() if count == 2][0]
            return 1000 + pair
        else:  # 高牌（強化版，五張 kicker 全考慮）
            score = 0
            for i, rank in enumerate(ranks):
                score += rank * (100 ** (4 - i))
            return score

    def _rank_to_number(self, rank):
        """將牌面轉換為數字"""
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(rank, 0)

    def _is_straight(self, ranks):
        """檢查是否為順子"""
        ranks = sorted(set(ranks), reverse=True)
        if len(ranks) < 5:
            return False
        
        # 檢查普通順子
        for i in range(len(ranks) - 4):
            if ranks[i] - ranks[i+4] == 4:
                return True
        
        # 檢查A-2-3-4-5順子
        if set([14, 5, 4, 3, 2]).issubset(set(ranks)):
            return True
        
        return False

    def _decide_action(self, valid_actions, win_probability, round_state):
        """基於獲勝機率決定動作"""
        # 獲取可用動作
        fold_action = valid_actions[0]
        call_action = valid_actions[1]
        
        # 檢查是否可以加注
        can_raise = len(valid_actions) > 2 and valid_actions[2]["amount"]["min"] != -1
        
        # 計算底池賠率
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        call_amount = call_action["amount"]
        
        if call_amount == 0:  # 可以免費看牌
            if win_probability > 0.7 and can_raise:
                # 強牌加注
                raise_amount = self._calculate_raise_amount(valid_actions[2], win_probability)
                return valid_actions[2]["action"], raise_amount
            else:
                return call_action["action"], call_action["amount"]
        
        # 計算期望值
        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 1
        
        if win_probability > pot_odds + 0.1:  # 有足夠優勢
            if win_probability > 0.7 and can_raise:
                # 強牌加注
                raise_amount = self._calculate_raise_amount(valid_actions[2], win_probability)
                return valid_actions[2]["action"], raise_amount
            else:
                return call_action["action"], call_action["amount"]
        elif win_probability > pot_odds - 0.05:  # 邊緣情況跟注
            return call_action["action"], call_action["amount"]
        else:  # 棄牌
            return fold_action["action"], fold_action["amount"]

    def _calculate_raise_amount(self, raise_action, win_probability):
        """計算加注金額"""
        min_raise = raise_action["amount"]["min"]
        max_raise = raise_action["amount"]["max"]
        
        # 根據獲勝機率調整加注大小
        if win_probability > 0.9:
            # 極強牌大注
            return min(max_raise, min_raise * 3)
        elif win_probability > 0.8:
            # 強牌中等加注
            return min(max_raise, min_raise * 2)
        else:
            # 最小加注
            return min_raise


def setup_ai():
    """設置AI，可以調整參數"""
    return MonteCarloPlayer(num_simulations=1000, verbose=True)