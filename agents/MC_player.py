import random
from collections import defaultdict
from game.players import BasePokerPlayer
from game.engine.hand_evaluator_ver1 import HandEvaluator
from game.engine.card import Card
import game.visualize_utils as U


# ---------- 工具函式 ----------
def chen_formula_score(card1, card2):
    """Chen Formula（簡化）計算起手牌分數"""
    rank_map = {'A': 10, 'K': 8, 'Q': 7, 'J': 6, 'T': 5,
                '9': 4.5, '8': 4, '7': 3.5, '6': 3, '5': 2.5,
                '4': 2, '3': 1.5, '2': 1}
    r1, r2 = card1[1], card2[1]
    v1, v2 = rank_map[r1], rank_map[r2]
    score = max(v1, v2)

    # pair
    if r1 == r2:
        score = max(5, score * 2)
        if score < 5:
            score -= 1

    # suited
    if card1[0] == card2[0]:
        score += 2

    # gap
    gap = abs("23456789TJQKA".index(r1) - "23456789TJQKA".index(r2)) - 1
    if gap == 0:
        score += 1
    elif gap in (2, 3):
        score -= 1
    elif gap >= 4:
        score -= 2

    # wheel bonus
    if {"A", "2"} <= {r1, r2} and card1[0] == card2[0]:
        score += 1

    return max(score, 0)


def estimate_fold_equity(pot, raise_amt):
    """極簡棄牌率模型：raise_amt 與 pot 比例愈大 → 棄牌率愈高"""
    if pot <= 0:
        return 0.0
    pct = raise_amt / pot
    return max(0.2, min(0.6, 0.3 + 0.3 * pct))


# ---------- 主體 ----------
class MonteCarloPlayer(BasePokerPlayer):
    # ------------------------
    # 可調參數
    # ------------------------
    BASE_SIMS = 1_000         # Turn / River 模擬次數
    PREFLOP_CHEN_MIN = 8.0    # 低於門檻直接 fold/limp
    VERBOSE = True

    # 牌組常量
    CARD_RANKS = '23456789TJQKA'
    CARD_SUITS = 'CDHS'

    def __init__(self):
        self.STATIC_DECK = [Card.from_str(s + r) for s in self.CARD_SUITS for r in self.CARD_RANKS]

    # ------------------------
    # 入口
    # ------------------------
    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            street = round_state.get("street", "preflop")
            pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)
            call_amt = valid_actions[1]["amount"]
            chen = chen_formula_score(hole_card[0], hole_card[1])

            # ---------- Pre-flop 弱牌：不跑 MC ----------
            if street == "preflop" and chen < self.PREFLOP_CHEN_MIN:
                if call_amt == 0:  # big blind 過牌
                    return valid_actions[1]["action"], call_amt
                return valid_actions[0]["action"], valid_actions[0]["amount"]

            win_prob = self._calc_win_prob(hole_card, round_state)

            action, amount = self._decide(valid_actions,
                                          win_prob,
                                          pot,
                                          call_amt,
                                          round_state)

            if self.VERBOSE:
                print(f"\n=== Monte-Carlo Player ===")
                print(f"Street = {street}")
                print(f"Hole   = {hole_card}")
                print(f"Chen   = {chen:.1f}")
                print(f"Win%   = {win_prob:.3f}")
                print(f"Action = {action} {amount}")
                print(U.visualize_declare_action(valid_actions, hole_card,
                                                 round_state, self.uuid))
            return action, amount

        except Exception as e:
            if self.VERBOSE:
                print("declare_action error:", e)
            return valid_actions[0]["action"], valid_actions[0]["amount"]

    # -- 以下五個回呼保持空實作 --
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): 
        print(f"receive_game_update_message: {new_action}, {round_state}")
    def receive_round_result_message(self, winners, hand_info, round_state): pass

    # ===========================================================
    # 1) 勝率估計：依街口自適應模擬 + random.sample
    # ===========================================================
    def _calc_win_prob(self, hole_card, round_state):
        street = round_state.get("street", "preflop")
        comm_cards = [Card.from_str(c) for c in round_state.get("community_card", [])]
        my_cards = [Card.from_str(c) for c in hole_card]

        sims = {
            "preflop": self.BASE_SIMS ,
            "flop":    self.BASE_SIMS ,
            "turn":    self.BASE_SIMS ,
            "river":   self.BASE_SIMS
        }.get(street, self.BASE_SIMS)

        alive = sum(s["state"] != "folded" for s in round_state["seats"])
        wins = ties = 0

        deck = [c for c in self.STATIC_DECK if c not in my_cards + comm_cards]
        need_board = 5 - len(comm_cards)
        need_opp   = 2 * (alive - 1)

        for _ in range(sims):
            draw = random.sample(deck, need_board + need_opp)
            faux_board = comm_cards + draw[:need_board]
            idx = need_board
            opp_hands = [draw[idx + 2*i: idx + 2*i + 2] for i in range(alive - 1)]

            my_score = HandEvaluator.eval_hand(my_cards, faux_board)
            opp_best = max(HandEvaluator.eval_hand(h, faux_board) for h in opp_hands)

            if my_score > opp_best:
                wins += 1
            elif my_score == opp_best:
                ties += 1

        return (wins + 0.5 * ties) / sims if sims else 0.0

    # ===========================================================
    # 2) 行動決策：EV-based Call、三檔 Raise、FoldEquity
    # ===========================================================
    def _decide(self, valid_actions, win_prob, pot, call_amt, rs):
        fold_act, call_act = valid_actions[:2]
        raise_act = valid_actions[2] if len(valid_actions) > 2 else None
        can_raise = raise_act and raise_act["amount"]["min"] != -1

        pot_odds = call_amt / (pot + call_amt) if pot + call_amt else 0
        margin   = win_prob - pot_odds

        # ---------- 嘗試加注 ----------
        if can_raise:
            raise_amt = self._select_raise(raise_act, win_prob, pot, rs)
            if raise_amt:  # 若 None 代表不加注
                street = rs.get("street", "preflop")
                eff_wp = win_prob
                if street in ("turn", "river"):
                    fe = estimate_fold_equity(pot, raise_amt)
                    eff_wp = win_prob + fe * (1 - win_prob)

                if eff_wp - pot_odds > 0.05:
                    return raise_act["action"], raise_amt
                if win_prob < pot_odds and random.random() < 0.05:
                    return raise_act["action"], raise_amt

        # ---------- EV-based Call / Fold ----------
        ev = win_prob * (pot + call_amt) - (1 - win_prob) * call_amt
        if ev > 0 and margin > -0.02:
            return call_act["action"], call_amt
        return fold_act["action"], fold_act["amount"]

    # ===========================================================
    # 3) 三檔 Raise Sizing：Pre-flop 特調，Post-flop Pot 基準
    # ===========================================================
    def _select_raise(self, raise_act, win_prob, pot, rs):
        min_r, max_r = raise_act["amount"]["min"], raise_act["amount"]["max"]

        # 無上限 → all-in 上限 = 自己籌碼
        if max_r == -1:
            me = next(s for s in rs["seats"] if s["uuid"] == self.uuid)
            max_r = me["stack"]

        street = rs.get("street", "preflop")
        big_blind = rs.get("table", {}).get("big_blind", 10)
        target = None

        # ---- Pre-flop ----
        if street == "preflop":
            if win_prob >= 0.80:
                target = 4 * big_blind
            elif win_prob >= 0.65:
                target = 3 * big_blind
            elif win_prob >= 0.55:
                target = 2.5 * big_blind

        # ---- Post-flop ----
        else:
            if win_prob >= 0.75:
                target = pot              # 1 pot
            elif win_prob >= 0.55:
                target = 0.5 * pot        # 0.5 pot

        if target is None:
            return None

        # -------- Aggressiveness Factor (籌碼深度) --------
        me_stack = next(s["stack"] for s in rs["seats"] if s["uuid"] == self.uuid)
        opp_stack = max(s["stack"] for s in rs["seats"]
                        if s["uuid"] != self.uuid and s["state"] != "folded")
        ratio = me_stack / opp_stack if opp_stack else 1.0
        target *= (1.0 + 0.75 * min(ratio - 1, 1)) if ratio > 1 else 1.0

        # 套用邊界
        target = int(max(min_r, min(target, max_r)))
        return target if target >= min_r else None


# ---------- PyPokerEngine 掛載 ----------
def setup_ai():
    return MonteCarloPlayer()
