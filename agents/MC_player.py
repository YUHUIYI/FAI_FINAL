import random
import itertools
from collections import Counter
from math import exp
import game.visualize_utils as U
from game.players import BasePokerPlayer
from game.engine.hand_evaluator_ver1 import HandEvaluator
from game.engine.card import Card


class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self, num_simulations=1000, verbose=False):
        self.num_simulations = num_simulations
        self.verbose = verbose
        self.card_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.card_suits = ['C', 'D', 'H', 'S']  # Clubs, Diamonds, Hearts, Spades

    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            win_probability = self._calculate_win_probability(hole_card, round_state)
            action, amount = self._decide_action(valid_actions, win_probability, round_state)

            if self.verbose:
                print(f"Monte Carlo分析:")
                print(f"手牌: {hole_card}")
                print(f"獲勝機率: {win_probability:.2%}")
                print(f"決定動作: {action}, 金額: {amount}")
                print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))

            return action, amount
        except Exception as e:
            if self.verbose:
                print(f"錯誤: {str(e)}")
            # 如果發生錯誤，選擇最安全的動作
            return valid_actions[0]["action"], valid_actions[0]["amount"]

    def receive_game_start_message(self, game_info):
        if self.verbose:
            print(U.visualize_game_start(game_info, self.uuid))

    def receive_round_start_message(self, round_count, hole_card, seats):
        if self.verbose:
            print(U.visualize_round_start(round_count, hole_card, seats, self.uuid))

    def receive_street_start_message(self, street, round_state):
        if self.verbose:
            print(U.visualize_street_start(street, round_state, self.uuid))

    def receive_game_update_message(self, new_action, round_state):
        if self.verbose:
            print(U.visualize_game_update(new_action, round_state, self.uuid))

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.verbose:
            print(U.visualize_round_result(winners, hand_info, round_state, self.uuid))

    def _calculate_win_probability(self, hole_cards, round_state):
        community_cards = round_state.get('community_card', [])
        num_players = len([seat for seat in round_state.get('seats', []) if seat['state'] != 'folded'])

        # 轉換牌型格式
        try:
            # 注意：Card.from_str()需要花色在前，點數在後
            hole_cards = [Card.from_str(card) for card in hole_cards]  # 直接使用原始格式
            community_cards = [Card.from_str(card) for card in community_cards]  # 直接使用原始格式
        except Exception as e:
            if self.verbose:
                print(f"牌型轉換錯誤: {str(e)}")
            return 0.0

        wins = 0
        for _ in range(self.num_simulations):
            try:
                remaining_deck = self._get_remaining_deck(hole_cards, community_cards)
                opponent_hands = self._simulate_opponent_hands(remaining_deck, num_players - 1)
                simulated_community = self._complete_community_cards(remaining_deck, community_cards)

                my_hand_strength = HandEvaluator.eval_hand(hole_cards, simulated_community)
                opponent_strengths = [HandEvaluator.eval_hand(hand, simulated_community) for hand in opponent_hands]

                if all(my_hand_strength > opp_strength for opp_strength in opponent_strengths):
                    wins += 1
            except Exception as e:
                if self.verbose:
                    print(f"模擬錯誤: {str(e)}")
                continue

        return wins / self.num_simulations

    def _get_remaining_deck(self, hole_cards, community_cards):
        all_cards = []
        for rank in self.card_ranks:
            for suit in self.card_suits:
                try:
                    # 直接使用原始格式
                    card_str = f"{suit}{rank}"
                    card = Card.from_str(card_str)
                    if card not in hole_cards and card not in community_cards:
                        all_cards.append(card)
                except Exception as e:
                    if self.verbose:
                        print(f"創建牌錯誤: {card_str} - {str(e)}")
                    continue
        return all_cards

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

    def _decide_action(self, valid_actions, win_probability, round_state):
        fold_action = valid_actions[0]
        call_action = valid_actions[1]
        can_raise = len(valid_actions) > 2 and valid_actions[2]["amount"]["min"] != -1

        # 獲取當前回合信息
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        call_amount = call_action["amount"]
        street = round_state.get('street', '')
        community_cards = round_state.get('community_card', [])
        
        # 計算pot odds
        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0

        # 獲取玩家信息
        my_stack = 0
        opp_stacks = []
        for seat in round_state.get('seats', []):
            if seat['uuid'] == self.uuid:
                my_stack = seat['stack']
            elif seat['state'] != 'folded':
                opp_stacks.append(seat['stack'])

        # 計算籌碼相關指標
        stack_ratio = my_stack / max(opp_stacks) if opp_stacks else 1.0
        stack_ratio = max(0.1, min(stack_ratio, 5.0))  # 限制範圍避免極端
        
        # 根據不同階段調整策略
        if street == 'preflop':
            # 翻牌前策略
            if win_probability < 0.3:  # 手牌太差
                return fold_action["action"], fold_action["amount"]
            elif win_probability > 0.7 and can_raise:  # 手牌很好
                raise_amount = self._calculate_raise_amount(valid_actions[2], win_probability, round_state)
                return valid_actions[2]["action"], raise_amount
            elif win_probability > 0.4:  # 手牌一般但可跟注
                return call_action["action"], call_action["amount"]
            else:
                return fold_action["action"], fold_action["amount"]
        else:
            # 翻牌後策略
            margin = win_probability - pot_odds
            
            # 根據勝率和pot odds的差距決定行動
            if margin > 0.2 and can_raise:  # 明顯優勢
                raise_amount = self._calculate_raise_amount(valid_actions[2], win_probability, round_state)
                return valid_actions[2]["action"], raise_amount
            elif margin > -0.1:  # 接近平衡
                return call_action["action"], call_action["amount"]
            else:  # 劣勢
                # 考慮偷雞機會
                if len(community_cards) >= 4 and random.random() < 0.1 and can_raise:
                    raise_amount = self._calculate_raise_amount(valid_actions[2], 0.6, round_state)
                    return valid_actions[2]["action"], raise_amount
                return fold_action["action"], fold_action["amount"]

    def _calculate_raise_amount(self, raise_action, win_probability, round_state):
        min_raise = raise_action["amount"]["min"]
        max_raise = raise_action["amount"]["max"]
        if max_raise == -1:
            max_raise = min_raise * 10

        # 獲取玩家信息
        my_stack = 0
        opp_max_stack = 0
        for seat in round_state.get('seats', []):
            if seat['uuid'] == self.uuid:
                my_stack = seat['stack']
            elif seat['state'] != 'folded':
                opp_max_stack = max(opp_max_stack, seat['stack'])

        # 計算籌碼比例
        if opp_max_stack > 0:
            stack_ratio = my_stack / opp_max_stack
            stack_ratio = max(0.1, min(stack_ratio, 5.0))
        else:
            stack_ratio = 1.0

        # 根據勝率和籌碼比例調整加注幅度
        if win_probability > 0.8:  # 非常強的牌
            aggressiveness = 0.8
        elif win_probability > 0.6:  # 較強的牌
            aggressiveness = 0.6
        else:  # 一般牌力
            aggressiveness = 0.4

        # 根據籌碼比例調整激進程度
        if stack_ratio > 2:  # 籌碼優勢
            aggressiveness *= 1.2
        elif stack_ratio < 0.5:  # 籌碼劣勢
            aggressiveness *= 0.8

        # 計算最終加注金額
        target_raise = min_raise + aggressiveness * (max_raise - min_raise)
        target_raise = max(min_raise, min(target_raise, max_raise))

        return int(target_raise)


def setup_ai():
    return MonteCarloPlayer(num_simulations=1000, verbose=True)