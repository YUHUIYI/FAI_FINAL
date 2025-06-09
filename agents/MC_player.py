# agents/MC_player.py
"""
Fast Monte-Carlo search​-based Texas Hold’em agent
-------------------------------------------------
‣ 每回合僅考慮 3 個動作：fold / call / all-in(raise)
‣ Monte-Carlo 依「目前街別」補齊餘下公共牌，估算自己勝率 (equity)
‣ 用簡單 EV = equity × 總彩池 − (1-equity) × 需投入計算
‣ 取 EV 最大的動作；equity 高於 raise_threshold 才考慮 all-in
"""

import os, sys, random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from game.players import BasePokerPlayer
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator

# ---- Monte-Carlo player ---------------------------------------------------- #
class FastMonteCarloPlayer(BasePokerPlayer):
    def __init__(self,
                 num_simulations: int = 2000,
                 raise_threshold: float = .65,
                 rng_seed: int = None):
        self.N          = num_simulations
        self.raise_thr  = raise_threshold
        self.rng        = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------ #
    # 主要決策
    # ------------------------------------------------------------------ #
    def declare_action(self, valid_actions, hole_card, round_state):
        # 1. Monte-Carlo 估 equity
        equity = self._estimate_equity(hole_card, round_state)   # 0-1

        # 2. 取得彩池 & call / raise 資訊
        pot_size   = round_state["pot"]["main"]["amount"]
        call_amt   = next(a["amount"]  for a in valid_actions if a["action"] == "call")

        raise_info = next(a for a in valid_actions if a["action"] == "raise")
        can_raise  = (isinstance(raise_info["amount"], dict) and
                      raise_info["amount"]["max"] != -1)

        # 3. EV (極簡估計)
        ev_fold  = 0.0
        ev_call  = equity * pot_size - (1 - equity) * call_amt

        ev_raise = -np.inf
        raise_amt = 0
        if can_raise and equity >= self.raise_thr:
            raise_amt = raise_info["amount"]["max"]
            # 假設對手 p_call = 0.5
            ev_if_called = equity * (pot_size + raise_amt) - (1 - equity) * raise_amt
            ev_if_fold   = pot_size
            ev_raise     = 0.5 * ev_if_called + 0.5 * ev_if_fold

        # 4. 選 EV 最大動作
        cand = [("fold", 0, ev_fold),
                ("call", call_amt, ev_call),
                ("raise", raise_amt, ev_raise)]
        best_action, best_amt, _ = max(cand, key=lambda x: x[2])

        # 如果 raise 不可行，就退回 call / fold
        if best_action == "raise" and not can_raise:
            best_action, best_amt = ("call", call_amt) if ev_call > 0 else ("fold", 0)

        # debug
        print(f"[FastMC] equity={equity:.3f}  pot={pot_size}  "
              f"EV(f/c/r)={[round(ev_fold,1), round(ev_call,1), round(ev_raise,1)]}  "
              f"→ {best_action.upper()}")

        return best_action, best_amt

    # ------------------------------------------------------------------ #
    # Monte-Carlo equity 估計
    # ------------------------------------------------------------------ #
    def _estimate_equity(self, hole_card, round_state) -> float:
        board = [Card.from_str(c) for c in round_state["community_card"]]
        my    = [Card.from_str(c) for c in hole_card]

        known_ids = {c.to_id() for c in board + my}
        deck_ids  = [cid for cid in range(1, 53) if cid not in known_ids]

        # 根據 street 補幾張牌
        n_board_needed = 5 - len(board)          # 0/2/1/0 for river/turn/flop/preflop?
        draw_size      = 2 + n_board_needed      # 2 for opp hole, 0-3 future community

        # 一次把需要的 id 全抽出來 (不重複)
        draws = self.rng.choice(deck_ids,
                                size=(self.N, draw_size),
                                replace=False)

        win = 0.0
        for line in draws:
            opp_hole_ids = line[:2]
            future_ids   = line[2:]

            opp_hole = [Card.from_id(i) for i in opp_hole_ids]
            full_board = board + [Card.from_id(i) for i in future_ids]

            my_score  = HandEvaluator.eval_hand(my,  full_board)
            opp_score = HandEvaluator.eval_hand(opp_hole, full_board)

            if   my_score >  opp_score: win += 1
            elif my_score == opp_score: win += .5

        return win / self.N

    # 其餘 callback 保留
    def receive_game_start_message(self, game_info):             pass
    def receive_round_start_message(self, round_count, hole_card,seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state):  pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

# 工廠函式
def setup_ai():
    return FastMonteCarloPlayer(num_simulations=500,
                                raise_threshold=.35,
                                rng_seed=None)
