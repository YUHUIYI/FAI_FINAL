# MC_player.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# fast_mc_player.py
import random
import numpy as np
from game.players import BasePokerPlayer
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator
from game.engine.poker_constants import PokerConstants as Const

class FastMonteCarloPlayer(BasePokerPlayer):
    """
    Monte-Carlo Search-based Texas-Hold'em agent (fold / call / shove)
    - 每次決策：
        1. 依目前 street & 已知牌，用 Monte Carlo 抽樣對手 hole 牌 + 剩餘公共牌
        2. 計算自己 vs 對手勝率 (equity)
        3. 估算 fold / call / raise EV，三擇一
    """
    def __init__(self, num_simulations:int = 3000, raise_threshold:float = 0.65):
        self.N = num_simulations
        self.raise_thr = raise_threshold     # equity > 0.65 則考慮 raise

    # ---------------- core decision ---------------- #
    def declare_action(self, valid_actions, hole_card, round_state):
        equity = self._estimate_equity(hole_card, round_state)  # 0~1
        pot_size   = round_state["pot"]["main"]["amount"]
        call_amt   = next(e["amount"] for e in valid_actions if e["action"]=="call")
        my_invest  = call_amt                                    # 才需要再投入的籌碼

        # --- EV 計算（簡化版）--- #
        ev_fold = 0.0                                           # 折牌 EV = 0
        ev_call = equity * pot_size  -  (1 - equity) * my_invest
        ev_raise = -np.inf
        raise_info = next(a for a in valid_actions if a["action"]=="raise")
        if raise_info["amount"]["max"] != -1:
            shove_amt = raise_info["amount"]["max"]
            # 假設對手 50% 跟注、50% 棄牌作近似
            ev_if_called = equity * (pot_size + shove_amt) - (1 - equity) * shove_amt
            ev_if_fold   = pot_size
            ev_raise     = 0.5 * ev_if_called + 0.5 * ev_if_fold

        # --- 取最大 EV 的動作 --- #
        evs = [("fold", 0, ev_fold), ("call", call_amt, ev_call), ("raise", raise_info["amount"]["max"], ev_raise)]
        best_action, best_amt, _ = max(evs, key=lambda x: x[2])

        # 也可設定簡單 heuristic：若 equity > raise_thr 才允許 raise
        if best_action=="raise" and equity < self.raise_thr:
            best_action, best_amt = "call", call_amt

        # print debug
        print(f"[FastMC] equity={equity:.3f}, EVs={[(e[0],round(e[2],1)) for e in evs]}, choose={best_action}")
        return best_action, best_amt

    # ---------------- Monte Carlo Equity ---------------- #
    def _estimate_equity(self, hole_card, round_state):
        # 1. 已知牌
        board_cards = [Card.from_str(c) for c in round_state["community_card"]]
        my_cards    = [Card.from_str(c) for c in hole_card]
        known_ids   = {c.to_id() for c in board_cards+my_cards}

        # 2. 建立牌庫 & 隨機抽樣
        deck_ids = [cid for cid in range(1,53) if cid not in known_ids]
        sims = np.random.choice(deck_ids, size=(self.N, 2 + (5-len(board_cards))), replace=False)

        wins = 0
        for draw in sims:
            opp_hole   = [Card.from_id(draw[0]), Card.from_id(draw[1])]
            future_ids = draw[2:]
            future_board = board_cards + [Card.from_id(cid) for cid in future_ids]

            my_score  = HandEvaluator.eval_hand(my_cards, future_board)
            opp_score = HandEvaluator.eval_hand(opp_hole, future_board)
            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                wins += 0.5  # split pot

        return wins / self.N
    def receive_game_start_message(self, game_info):
        print("[MC Player] I am using MonteCarloPlayer.")
        

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

# ----------------------------------------------------- #
def setup_ai():
    """讓 start_game.py 可以直接 import 這個 AI"""
    return FastMonteCarloPlayer(num_simulations=3000, raise_threshold=0.65)
