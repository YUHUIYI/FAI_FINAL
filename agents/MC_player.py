import random
import itertools
from collections import Counter
from math import exp
import game.visualize_utils as U
from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card


class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self, num_simulations=1000, verbose=False):
        self.num_simulations = num_simulations
        self.verbose = verbose
        self.card_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.card_suits = ['C', 'D', 'H', 'S']  # Clubs, Diamonds, Hearts, Spades

    def _convert_to_card(self, card_str):
        """将字符串转换为Card对象"""
        rank = card_str[0]
        suit = card_str[1]
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        return Card(rank_map[rank], suit_map[suit])

    def _calculate_win_probability(self, hole_cards, round_state):
        """使用Monte Carlo模擬計算獲勝機率"""
        community_cards = round_state.get('community_card', [])
        
        # 转换手牌和公共牌为Card对象
        hole_cards = [self._convert_to_card(card) for card in hole_cards]
        community_cards = [self._convert_to_card(card) for card in community_cards]
        
        # 获取当前未弃牌的玩家数量
        active_players = [seat for seat in round_state.get('seats', []) if seat['state'] != 'folded']
        num_players = len(active_players)
        
        # 如果只剩一个玩家，直接返回100%胜率
        if num_players == 1:
            return 1.0
            
        # 获取当前玩家的位置信息
        my_seat = next((seat for seat in active_players if seat['uuid'] == self.uuid), None)
        if not my_seat:
            return 0.0
            
        wins = 0
        ties = 0
        
        for _ in range(self.num_simulations):
            # 创建剩余牌组
            remaining_deck = self._get_remaining_deck(hole_cards, community_cards)
            
            # 模拟对手手牌
            opponent_hands = self._simulate_opponent_hands(remaining_deck, num_players - 1)
            
            # 完成公共牌
            simulated_community = self._complete_community_cards(remaining_deck, community_cards)
            
            # 使用HandEvaluator评估手牌强度
            my_hand_strength = HandEvaluator.eval_hand(hole_cards, simulated_community)
            opponent_strengths = [
                HandEvaluator.eval_hand(hand, simulated_community) 
                for hand in opponent_hands
            ]
            
            # 检查是否获胜或平局
            if all(my_hand_strength > opp_strength for opp_strength in opponent_strengths):
                wins += 1
            elif all(my_hand_strength == opp_strength for opp_strength in opponent_strengths):
                ties += 1
        
        # 计算胜率（包括平局）
        win_rate = (wins + ties/2) / self.num_simulations
        
        # 根据位置调整胜率
        position = my_seat.get('position', '')
        if position == 'BB':  # 大盲位
            win_rate = min(0.95, win_rate * 1.1)  # 提高10%的胜率
        elif position == 'SB':  # 小盲位
            win_rate = min(0.95, win_rate * 1.05)  # 提高5%的胜率
            
        # 确保胜率在合理范围内
        win_rate = max(0.2, min(0.9, win_rate))
        
        if self.verbose:
            print(f"[MC Player] 模拟结果: 胜={wins}, 平={ties}, 总={self.num_simulations}")
            print(f"[MC Player] 原始胜率: {(wins + ties/2) / self.num_simulations:.2%}")
            print(f"[MC Player] 调整后胜率: {win_rate:.2%}")
            
        return win_rate

    def _get_remaining_deck(self, hole_cards, community_cards):
        """获取剩余牌组"""
        all_cards = [rank + suit for rank in self.card_ranks for suit in self.card_suits]
        used_cards = set(str(card.rank) + self.card_suits[card.suit] for card in hole_cards + community_cards)
        return [card for card in all_cards if card not in used_cards]

    def _simulate_opponent_hands(self, deck, num_opponents):
        """模拟对手手牌"""
        available_cards = deck.copy()
        opponent_hands = []
        
        for _ in range(num_opponents):
            if len(available_cards) >= 2:
                hand = random.sample(available_cards, 2)
                opponent_hands.append([self._convert_to_card(card) for card in hand])
                # 移除已使用的牌
                for card in hand:
                    available_cards.remove(card)
        
        return opponent_hands

    def _complete_community_cards(self, deck, existing_community):
        """完成公共牌到5张"""
        cards_needed = 5 - len(existing_community)
        if cards_needed <= 0:
            return existing_community
        
        available_cards = [card for card in deck if card not in existing_community]
        additional_cards = random.sample(available_cards, min(cards_needed, len(available_cards)))
        return existing_community + [self._convert_to_card(card) for card in additional_cards]

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