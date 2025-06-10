import random
from math import exp
from collections import defaultdict
from game.players import BasePokerPlayer
from game.engine.hand_evaluator_ver1 import HandEvaluator
from game.engine.card import Card
import game.visualize_utils as U


# ---------- 工具函式 ----------
def chen_formula_score(card1, card2):
    """根據 Chen Formula 計算起手牌評分（簡化版）"""
    rank_map = {'A':10, 'K':8, 'Q':7, 'J':6, 'T':5,
                '9':4.5, '8':4, '7':3.5, '6':3, '5':2.5,
                '4':2, '3':1.5, '2':1}
    r1, r2 = card1[1], card2[1]  # Card string 例如 'SA' → S 花色, A 點數
    v1, v2 = rank_map[r1], rank_map[r2]
    high, low = max(v1, v2), min(v1, v2)
    score = high

    # pair
    if r1 == r2:
        score = max(5, high*2)  # 最低給 5
        if high < 5:            # 小對子扣 1
            score -= 1

    # suited
    if card1[0] == card2[0]:
        score += 2

    # gap / connectors
    gap = abs("23456789TJQKA".index(r1) - "23456789TJQKA".index(r2)) - 1
    if gap == 0:
        score += 1
    elif gap == 1:
        score += 0   # 不加不減
    elif 2 <= gap <= 3:
        score -= 1
    else:  # gap ≥ 4
        score -= 2

    # small bonus for A-2 suited wheel
    if {"A", "2"} <= {r1, r2} and card1[0] == card2[0]:
        score += 1

    return max(score, 0)


def estimate_fold_equity(pot, raise_amt):
    """極簡估計對手棄牌率；僅供示意"""
    if pot <= 0:
        return 0.0
    pct = raise_amt / pot
    # 半池下注→約 30% 棄牌；全池下注→約 45%；全下上衝 60%
    return max(0.2, min(0.6, 0.3 + 0.3 * pct))


# ---------- 主體 ----------
class MonteCarloPlayer(BasePokerPlayer):
    CARD_RANKS = '23456789TJQKA'
    CARD_SUITS = 'CDHS'  # Clubs Diamonds Hearts Spades

    def __init__(self,
                 base_simulations: int = 1_000,
                 preflop_threshold: float = 7.0,
                 verbose: bool = False):
        self.base_simulations = base_simulations
        self.preflop_threshold = preflop_threshold
        self.verbose = verbose
        self.STATIC_DECK = [Card.from_str(s + r) for s in self.CARD_SUITS for r in self.CARD_RANKS]

    # -------- 介面回呼（保留可視化） --------
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            street = round_state.get('street', 'preflop')
            pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
            call_amt = valid_actions[1]['amount']
            chen_score = chen_formula_score(hole_card[0], hole_card[1])

            # ---- Pre-flop：弱牌直接應對，強牌才跑 MC ----
            if street == 'preflop' and chen_score < self.preflop_threshold:
                # 只在大盲可過牌時跟注，否則棄牌
                if call_amt == 0:
                    return valid_actions[1]['action'], call_amt
                return valid_actions[0]['action'], valid_actions[0]['amount']

            win_prob = self._calculate_win_probability(hole_card, round_state)

            action, amount = self._decide_action(valid_actions,
                                               win_prob,
                                               pot_size,
                                               call_amt,
                                               round_state)

            if self.verbose:
                print(f"\n=== Monte-Carlo Player ===")
                print(f"Street = {street}")
                print(f"Hole  = {hole_card}")
                print(f"Chen  = {chen_score:.1f}")
                print(f"Win%  = {win_prob:.3f}")
                print(f"選擇  = {action}, {amount}")
                print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))

            return action, amount

        except Exception as e:
            if self.verbose:
                print("declare_action error:", e)
            return valid_actions[0]['action'], valid_actions[0]['amount']

    # 其餘 receive_* 保留 (省略) -------------------------
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

    # -------- Monte-Carlo 勝率估計 --------
    def _calculate_win_probability(self, hole_card, round_state):
        street = round_state.get('street', 'preflop')
        community = round_state.get('community_card', [])

        # 模擬次數：前街多，後街少
        sims = { 'preflop': self.base_simulations,
                 'flop'   : self.base_simulations,
                 'turn'   : self.base_simulations,
                 'river'  : self.base_simulations }.get(street, self.base_simulations)

        # 轉換格式
        try:
            my_cards = [Card.from_str(c) for c in hole_card]
            comm_cards = [Card.from_str(c) for c in community]
        except Exception:
            return 0.0

        num_alive = sum(seat['state'] != 'folded'
                        for seat in round_state.get('seats', []))

        wins = ties = 0
        deck = [c for c in self.STATIC_DECK
                if c not in my_cards and c not in comm_cards]

        for _ in range(sims):
            try:
                random.shuffle(deck)
                idx = 0

                # 補對手手牌
                opp_hands = []
                for _ in range(num_alive - 1):
                    opp_hands.append([deck[idx], deck[idx+1]])
                    idx += 2

                # 補公共牌
                needed = 5 - len(comm_cards)
                fake_community = comm_cards + deck[idx: idx+needed]

                my_score = HandEvaluator.eval_hand(my_cards, fake_community)
                opp_scores = [HandEvaluator.eval_hand(h, fake_community) for h in opp_hands]

                best_opp = max(opp_scores)
                if my_score > best_opp:
                    wins += 1
                elif my_score == best_opp:
                    ties += 1
            except Exception:
                continue

        return (wins + 0.5*ties) / sims if sims else 0.0

    # -------- 行動決策 (加 fold-equity & 三檔下注) --------
    def _decide_action(self,
                       valid_actions,
                       win_prob: float,
                       pot: int,
                       call_amt: int,
                       round_state):

        fold_act = valid_actions[0]
        call_act = valid_actions[1]
        raise_act = valid_actions[2] if len(valid_actions) > 2 else None
        can_raise = raise_act and raise_act['amount']['min'] != -1

        pot_odds = call_amt / (pot + call_amt) if pot + call_amt else 0
        margin = win_prob - pot_odds

        # ---- 嘗試加注 / 偷雞 ----
        if can_raise:
            raise_amt = self._select_raise_amount(raise_act,
                                                  win_prob,
                                                  pot,
                                                  round_state)

            # Fold-equity only for post-flop aggressive action
            street = round_state.get('street', 'preflop')
            if street in ('turn', 'river') and raise_amt:
                fold_eq = estimate_fold_equity(pot, raise_amt)
                eff_win_prob = win_prob + fold_eq*(1 - win_prob)
            else:
                eff_win_prob = win_prob

            # 只有當有效勝率高於 Pot Odds + 0.05 才考慮加注
            if eff_win_prob - pot_odds > 0.05 and raise_amt:
                return raise_act['action'], raise_amt

            # 偷雞：勝率略低於 Pot Odds 但隨機 5% 嘗試
            if win_prob < pot_odds and random.random() < 0.05 and raise_amt:
                return raise_act['action'], raise_amt

        # ---- 跟注 / 棄牌 ----
        if margin > -0.03:               # 容忍 3% 負 EV
            return call_act['action'], call_amt
        return fold_act['action'], fold_act['amount']

    # ---- 選擇三檔 Raise 金額 ----
    def _select_raise_amount(self, raise_act, win_prob, pot, round_state):
        min_r, max_r = raise_act['amount']['min'], raise_act['amount']['max']
        if max_r == -1:  # 無上限，取對手最大籌碼或自己籌碼
            my_stack = next(seat['stack'] for seat in round_state['seats']
                            if seat['uuid'] == self.uuid)
            max_r = my_stack

        # 三檔閾值
        if win_prob >= 0.80:          # 超強牌 → All-in
            target = max_r
        elif win_prob >= 0.60:        # 強牌 → 1 pot raise
            target = pot
        elif win_prob >= 0.45:        # 可觀察牌 → 0.5 pot raise
            target = pot * 0.5
        else:
            return None

        # 籌碼深度調節
        my_stack = next(seat['stack'] for seat in round_state['seats']
                        if seat['uuid'] == self.uuid)
        opp_max = max(seat['stack'] for seat in round_state['seats']
                      if seat['uuid'] != self.uuid and seat['state'] != 'folded')
        stack_ratio = my_stack / opp_max if opp_max else 1.0
        aggressiveness_factor = 0.5 + 1.5 * min(stack_ratio, 2.0)
        target *= aggressiveness_factor

        # 套用限制
        target = max(min_r, min(int(target), max_r))
        return target

# -------- PyPokerEngine 掛載函式 --------
def setup_ai():
    # 可調 base_simulations、verbose
    return MonteCarloPlayer(base_simulations=1_000, verbose=True)
