# agents/MC_player.py
"""
Monte-Carlo player —— 只會 CALL 或 ALL-IN（永不 fold）
compatible with Python 3.8+
"""
import os, sys
import numpy as np
from typing import Optional          #  ←  新增

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from game.players import BasePokerPlayer
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator


class FastMonteCarloPlayer(BasePokerPlayer):
    def __init__(self,
                 num_simulations: int = 2000,
                 raise_threshold: float = 0.55,
                 rng_seed: Optional[int] = None):      # ←  使用 Optional
        self.N   = num_simulations
        self.thr = raise_threshold
        self.rng = np.random.default_rng(rng_seed)

    # ─────────────────────────────────────────────────────────── #
    def declare_action(self, valid_actions, hole_card, round_state):
        equity = self._equity(hole_card, round_state)

        pot      = round_state["pot"]["main"]["amount"]
        call_amt = next(a["amount"] for a in valid_actions if a["action"] == "call")

        raise_info = next(a for a in valid_actions if a["action"] == "raise")
        can_raise  = isinstance(raise_info["amount"], dict) and raise_info["amount"]["max"] != -1

        ev_call  = equity * pot - (1 - equity) * call_amt

        ev_raise = -np.inf
        raise_amt = 0
        if can_raise:
            raise_amt  = raise_info["amount"]["max"]         # 直接 all-in
            opp_call_p = max(0.2, 1 - equity)                # 粗估對手跟注率
            ev_called  = equity * (pot + raise_amt) - (1 - equity) * raise_amt
            ev_folded  = pot
            ev_raise   = opp_call_p * ev_called + (1 - opp_call_p) * ev_folded

        # 永遠不 fold
        if can_raise and equity >= self.thr and ev_raise > ev_call:
            action, amount = "raise", raise_amt
        else:
            action, amount = "call",  call_amt

        print(f"[FastMC] equity={equity:.3f}  pot={pot}  "
              f"EV(call)={ev_call:.1f}  EV(raise)={ev_raise:.1f} → {action.upper()}")
        return action, amount

    # ─────────────────────────────────────────────────────────── #
    def _equity(self, hole_card, round_state) -> float:
        board = [Card.from_str(c) for c in round_state["community_card"]]
        me    = [Card.from_str(c) for c in hole_card]

        known = {c.to_id() for c in board + me}
        deck  = [cid for cid in range(1, 53) if cid not in known]

        need   = 5 - len(board)            # 還缺幾張公共牌
        sample = self.rng.choice(deck, size=(self.N, 2 + need), replace=False)

        wins = 0.0
        for row in sample:
            opp    = [Card.from_id(i) for i in row[:2]]
            future = [Card.from_id(i) for i in row[2:]]
            full_b = board + future

            my_s  = HandEvaluator.eval_hand(me,  full_b)
            opp_s = HandEvaluator.eval_hand(opp, full_b)
            if   my_s >  opp_s: wins += 1
            elif my_s == opp_s: wins += 0.5

        return wins / self.N

    # 其它 callback 保留空函式
    def receive_game_start_message(self, *a):    pass
    def receive_round_start_message(self, *a):   pass
    def receive_street_start_message(self, *a):  pass
    def receive_game_update_message(self, *a):   pass
    def receive_round_result_message(self, *a):  pass


def setup_ai():
    return FastMonteCarloPlayer(num_simulations=2000,
                                raise_threshold=0.55,
                                rng_seed=None)
