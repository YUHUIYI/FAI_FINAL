# agents/MC_player.py
"""
Fast Monte-Carlo search​-based Texas Hold'em agent
-------------------------------------------------
‣ 每回合僅考慮 2 個動作：call / all-in(raise)
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
                 num_simulations: int = 1000,
                 raise_threshold: float = 0.35,
                 call_threshold: float = 0.10,
                 rng_seed: int = None):
        self.N = num_simulations
        self.raise_thr = raise_threshold
        self.call_thr = call_threshold
        self.rng = np.random.default_rng(rng_seed)
        self.position = None
        print(f"[MC Player] Initialized with raise_thr={raise_threshold}, call_thr={call_threshold}")

    def receive_game_start_message(self, game_info):
        print("[MC Player] Game started")
        self.position = None

    def receive_round_start_message(self, round_count, hole_card, seats):
        # 在每轮开始时更新位置信息
        for seat in seats:
            if seat["uuid"] == self.uuid:
                self.position = "SB" if seat["stack"] == 980 else "BB"  # 假设小盲是20
                print(f"[MC Player] Round {round_count} started, position: {self.position}")
                break

    # ------------------------------------------------------------------ #
    # 主要決策
    # ------------------------------------------------------------------ #
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
        call_amt = next(a["amount"] for a in valid_actions if a["action"] == "call")
        
        raise_info = next(a for a in valid_actions if a["action"] == "raise")
        can_raise = (isinstance(raise_info["amount"], dict) and
                    raise_info["amount"]["max"] != -1)

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
            raise_amt = raise_info["amount"]["max"]
            # 根据equity动态调整对手跟注概率
            call_prob = max(0.2, min(0.8, 1 - adjusted_equity))
            ev_if_called = adjusted_equity * (pot_size + raise_amt) - (1 - adjusted_equity) * raise_amt
            ev_if_fold = pot_size
            ev_raise = call_prob * ev_if_called + (1 - call_prob) * ev_if_fold

        # 4. 决策逻辑（改进版）- 禁止fold
        # 首先检查是否可以raise
        if can_raise and adjusted_equity >= self.raise_thr:
            best_action, best_amt = "raise", raise_amt
            print(f"[MC Player] Choosing RAISE because adjusted_equity ({adjusted_equity:.3f}) >= raise_thr ({self.raise_thr})")
        # 否则就call
        else:
            best_action, best_amt = "call", call_amt
            print(f"[MC Player] Choosing CALL because adjusted_equity ({adjusted_equity:.3f}) < raise_thr ({self.raise_thr}) or cannot raise")

        # debug
        print(f"[MC Player] pos={self.position} equity={equity:.3f}(adj={adjusted_equity:.3f}) "
              f"pot={pot_size} pot_odds={pot_odds:.2f} "
              f"EV(c/r)={[round(ev_call,1), round(ev_raise,1)]} "
              f"→ {best_action.upper()}")

        # 确保不会返回fold
        if best_action == "fold":
            print("[MC Player] WARNING: Somehow got fold action, forcing call instead")
            best_action, best_amt = "call", call_amt

        return best_action, best_amt

    # ------------------------------------------------------------------ #
    # Monte-Carlo equity 估計
    # ------------------------------------------------------------------ #
    def _estimate_equity(self, hole_card, round_state) -> float:
        board = [Card.from_str(c) for c in round_state["community_card"]]
        my = [Card.from_str(c) for c in hole_card]

        known_ids = {c.to_id() for c in board + my}
        deck_ids = [cid for cid in range(1, 53) if cid not in known_ids]

        # 根據 street 補幾張牌
        n_board_needed = 5 - len(board)          # 0/2/1/0 for river/turn/flop/preflop?
        draw_size = 2 + n_board_needed      # 2 for opp hole, 0-3 future community

        # 一次把需要的 id 全抽出來 (不重複)
        draws = self.rng.choice(deck_ids,
                              size=(self.N, draw_size),
                              replace=False)

        win = 0.0
        for line in draws:
            opp_hole_ids = line[:2]
            future_ids = line[2:]

            opp_hole = [Card.from_id(i) for i in opp_hole_ids]
            full_board = board + [Card.from_id(i) for i in future_ids]

            my_score = HandEvaluator.eval_hand(my, full_board)
            opp_score = HandEvaluator.eval_hand(opp_hole, full_board)

            if my_score > opp_score: win += 1
            elif my_score == opp_score: win += .5

        equity = win / self.N
        print(f"[MC Player] Raw equity calculation: {equity:.3f} (wins={win}, total={self.N})")
        return equity

    def receive_street_start_message(self, street, round_state):
        print(f"[MC Player] Street {street} started")

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        print(f"[MC Player] Round ended. Winners: {winners}")

# 工廠函式
def setup_ai():
    return FastMonteCarloPlayer(num_simulations=1000,
                              raise_threshold=0.35,
                              call_threshold=0.10,
                              rng_seed=None)
