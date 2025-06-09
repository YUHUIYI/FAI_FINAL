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
        # 添加调试信息
        print(f"[MC Player] Valid actions: {valid_actions}")
        print(f"[MC Player] Hole cards: {hole_card}")
        print(f"[MC Player] Round state: {round_state}")
        
        # 1. Monte-Carlo 估 equity
        equity = self._estimate_equity(hole_card, round_state)   # 0-1
        
        # 根据位置调整equity
        position_bonus = 0.05 if self.position == "BB" else 0.0
        adjusted_equity = min(0.95, equity + position_bonus)

        # 2. 取得彩池 & call / raise 資訊
        pot_size = round_state["pot"]["main"]["amount"]
        
        # 直接使用索引获取动作信息
        call_action_info = valid_actions[1]  # call是第二个动作
        raise_action_info = valid_actions[2]  # raise是第三个动作
        
        call_amt = call_action_info["amount"]
        can_raise = (isinstance(raise_action_info["amount"], dict) and
                    raise_action_info["amount"]["max"] != -1)

        # 3. EV 计算（改进版）
        # 改进call的EV计算
        pot_odds = call_amt / (pot_size + call_amt)
        ev_call = adjusted_equity * pot_size - (1 - adjusted_equity) * call_amt
        
        # 如果pot odds很好，大幅提高call的EV
        if pot_odds < 0.3:  # 好的pot odds
            ev_call *= 1.5
        elif pot_odds < 0.5:  # 一般的pot odds
            ev_call *= 1.2

        ev_raise = -np.inf
        raise_amt = 0
        if can_raise:
            raise_amt = raise_action_info["amount"]["max"]
            # 根据equity动态调整对手跟注概率
            call_prob = max(0.2, min(0.8, 1 - adjusted_equity))
            ev_if_called = adjusted_equity * (pot_size + raise_amt) - (1 - adjusted_equity) * raise_amt
            ev_if_fold = pot_size
            ev_raise = call_prob * ev_if_called + (1 - call_prob) * ev_if_fold

        # 4. 决策逻辑（改进版）- 禁止fold
        # 首先检查是否可以raise
        if can_raise and adjusted_equity >= self.thr:
            best_action, best_amt = "raise", raise_amt
            print(f"[MC Player] Choosing RAISE because adjusted_equity ({adjusted_equity:.3f}) >= raise_thr ({self.thr})")
        # 否则就call
        else:
            best_action, best_amt = "call", call_amt
            print(f"[MC Player] Choosing CALL because adjusted_equity ({adjusted_equity:.3f}) < raise_thr ({self.thr}) or cannot raise")

        # debug
        print(f"[MC Player] pos={self.position} equity={equity:.3f}(adj={adjusted_equity:.3f}) "
              f"pot={pot_size} pot_odds={pot_odds:.2f} "
              f"EV(c/r)={[round(ev_call,1), round(ev_raise,1)]} "
              f"→ {best_action.upper()}")

        return best_action, best_amt

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
