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
    def __init__(self, num_simulations:int = 3000, raise_threshold:float = 0.55):
        self.N = num_simulations
        self.raise_thr = raise_threshold
        self.position = None
        self.hand_history = []

    # ---------------- core decision ---------------- #
    def declare_action(self, valid_actions, hole_card, round_state):
        # 更新位置信息
        if self.position is None:
            self.position = "SB" if round_state["small_blind_pos"] == round_state["current_player"] else "BB"

        equity = self._estimate_equity(hole_card, round_state)
        pot_size = round_state["pot"]["main"]["amount"]
        call_amt = next(e["amount"] for e in valid_actions if e["action"]=="call")
        my_invest = call_amt

        # 改进的EV计算
        ev_fold = 0.0
        ev_call = equity * pot_size - (1 - equity) * my_invest

        # 根据位置和equity调整raise策略
        position_factor = 1.2 if self.position == "BB" else 1.0
        adjusted_equity = equity * position_factor

        ev_raise = -np.inf
        raise_info = next(a for a in valid_actions if a["action"]=="raise")
        if raise_info["amount"]["max"] != -1:
            shove_amt = raise_info["amount"]["max"]
            # 根据equity调整对手跟注概率
            call_prob = max(0.2, min(0.8, 1 - adjusted_equity))
            ev_if_called = adjusted_equity * (pot_size + shove_amt) - (1 - adjusted_equity) * shove_amt
            ev_if_fold = pot_size
            ev_raise = call_prob * ev_if_called + (1 - call_prob) * ev_if_fold

        # 记录决策历史
        self.hand_history.append({
            "equity": equity,
            "pot_size": pot_size,
            "position": self.position,
            "action": None  # 将在选择动作后更新
        })

        # 选择最佳动作
        evs = [("fold", 0, ev_fold), ("call", call_amt, ev_call), ("raise", raise_info["amount"]["max"], ev_raise)]
        best_action, best_amt, best_ev = max(evs, key=lambda x: x[2])

        # 改进的raise策略
        if best_action == "raise":
            if adjusted_equity < self.raise_thr:
                # 如果equity不够高，考虑call而不是直接fold
                if ev_call > ev_fold:
                    best_action, best_amt = "call", call_amt
                else:
                    best_action, best_amt = "fold", 0
            elif adjusted_equity > 0.8:  # 非常强的牌
                # 考虑更大的加注
                if raise_info["amount"]["max"] > 2 * pot_size:
                    best_amt = min(raise_info["amount"]["max"], 2 * pot_size)

        # 更新决策历史
        self.hand_history[-1]["action"] = best_action
        self.hand_history[-1]["ev"] = best_ev

        print(f"[FastMC] equity={equity:.3f}, pos={self.position}, EVs={[(e[0],round(e[2],1)) for e in evs]}, choose={best_action}")
        return best_action, best_amt

    # ---------------- Monte Carlo Equity ---------------- #
    def _estimate_equity(self, hole_card, round_state):
        board_cards = [Card.from_str(c) for c in round_state["community_card"]]
        my_cards = [Card.from_str(c) for c in hole_card]
        known_ids = {c.to_id() for c in board_cards+my_cards}

        # 考虑位置因素
        position_bonus = 0.05 if self.position == "BB" else 0.0

        deck_ids = [cid for cid in range(1,53) if cid not in known_ids]
        sims = np.random.choice(deck_ids, size=(self.N, 2 + (5-len(board_cards))), replace=False)

        wins = 0
        for draw in sims:
            opp_hole = [Card.from_id(draw[0]), Card.from_id(draw[1])]
            future_ids = draw[2:]
            future_board = board_cards + [Card.from_id(cid) for cid in future_ids]

            my_score = HandEvaluator.eval_hand(my_cards, future_board)
            opp_score = HandEvaluator.eval_hand(opp_hole, future_board)
            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                wins += 0.5

        equity = wins / self.N
        return min(0.95, equity + position_bonus)  # 添加位置奖励，但限制最大equity

    def receive_game_start_message(self, game_info):
        print("[MC Player] I am using MonteCarloPlayer.")
        self.position = None
        self.hand_history = []

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # 分析这一轮的结果
        if self.hand_history:
            last_decision = self.hand_history[-1]
            print(f"[MC Player] Round result - Action: {last_decision['action']}, "
                  f"Equity: {last_decision['equity']:.3f}, EV: {last_decision.get('ev', 'N/A')}")

# ----------------------------------------------------- #
def setup_ai():
    """讓 start_game.py 可以直接 import 這個 AI"""
    return FastMonteCarloPlayer(num_simulations=3000, raise_threshold=0.55)
