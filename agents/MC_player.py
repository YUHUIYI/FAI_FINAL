"""agents/MC_player.py
A lightweight Monte‑Carlo search Texas‑Hold’em agent.

Highlights
----------
* **Never folds** – the agent will always choose **call** or **raise**.
* **Equity driven** – estimates win‑rate (equity) against one unknown opponent by
  Monte‑Carlo sampling of opponent hole cards and future community cards.
* **Simple EV model** – decides between *call* and *all‑in raise* using
  expected‑value given the current pot and the amount that must be invested.
* **Console‑friendly** – implementation follows the same callback style used by
  `ConsolePlayer` (declare_action / receive_* methods).
"""

from __future__ import annotations

import os
import sys
import random
from typing import List, Tuple

import numpy as np

# Project root on the import path ------------------------------------------------
SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if SYS_DIR not in sys.path:
    sys.path.append(SYS_DIR)

from game.players import BasePokerPlayer
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator


# ---------------------------------------------------------------------------
# Monte‑Carlo helper
# ---------------------------------------------------------------------------

def _estimate_equity(my_hole: List[str], community: List[str], n_sim: int, rng: np.random.Generator) -> float:
    """Return the probability that *my hand* wins at showdown against a single
    random opponent.

    Parameters
    ----------
    my_hole : list[str]
        Two strings like 'SA', 'D3'.
    community : list[str]
        Already revealed community cards (0‑5 strings).
    n_sim : int
        Number of Monte‑Carlo iterations.
    rng : numpy Random Generator
    """
    my_cards = [Card.from_str(s) for s in my_hole]
    board = [Card.from_str(s) for s in community]

    known = {c.to_id() for c in my_cards + board}
    deck_ids = [cid for cid in range(1, 53) if cid not in known]

    # How many community cards still to be dealt?
    need_board = 5 - len(board)
    draw_size = 2 + need_board  # 2 cards opp hole + remaining board

    wins = 0.0
    draws = rng.choice(deck_ids, size=(n_sim, draw_size), replace=False)
    for row in draws:
        opp_hole_ids = row[:2]
        future_ids = row[2:]

        opp_hole = [Card.from_id(int(cid)) for cid in opp_hole_ids]
        final_board = board + [Card.from_id(int(cid)) for cid in future_ids]

        my_score = HandEvaluator.eval_hand(my_cards, final_board)
        opp_score = HandEvaluator.eval_hand(opp_hole, final_board)

        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5  # split pot

    return wins / n_sim


# ---------------------------------------------------------------------------
# Main player
# ---------------------------------------------------------------------------

class MonteCarloPlayer(BasePokerPlayer):
    """A very small Monte‑Carlo poker bot that never folds."""

    def __init__(self, n_simulations: int = 1500, raise_threshold: float = 0.60, seed: int | None = None):
        super().__init__()
        self.n_sim = n_simulations
        self.raise_thr = raise_threshold
        self.rng = np.random.default_rng(seed if seed is not None else random.randrange(2**32))

    # ------------------------------------------------------------------
    #  Core decision logic
    # ------------------------------------------------------------------
    def declare_action(self, valid_actions: List[dict], hole_card: List[str], round_state: dict) -> Tuple[str, int]:
        """Choose between *call* and *raise* based on equity.

        The function **never returns `fold`.** If a raise is not allowed, it will
        always fall back to call.
        """
        # 1. Compute win probability
        equity = _estimate_equity(hole_card, round_state["community_card"], self.n_sim, self.rng)

        # 2. Extract action details
        call_info = next(a for a in valid_actions if a["action"] == "call")
        call_amount = int(call_info["amount"])

        raise_info = next(a for a in valid_actions if a["action"] == "raise")
        can_raise = isinstance(raise_info["amount"], dict) and raise_info["amount"]["max"] != -1
        max_raise = int(raise_info["amount"]["max"]) if can_raise else -1

        pot = round_state["pot"]["main"]["amount"]

        # 3. Simple EV comparison
        ev_call = equity * pot - (1 - equity) * call_amount
        ev_raise = float("-inf")
        if can_raise:
            # Go all‑in with max_raise for simplicity. Model opp calling 50%.
            invest = max_raise
            ev_if_called = equity * (pot + invest) - (1 - equity) * invest
            ev_if_fold = pot  # opp folds, we win pot
            ev_raise = 0.5 * ev_if_called + 0.5 * ev_if_fold

        # 4. Decision – never fold.
        if can_raise and equity >= self.raise_thr and ev_raise >= ev_call:
            action, amount = "raise", max_raise
        else:
            action, amount = "call", call_amount

        # Debug print (comment out to silence)
        print(f"[MC] equity={equity:.3f} pot={pot} → {action.upper()} {amount}")
        return action, amount

    # ------------------------------------------------------------------
    # Callbacks – keep console‑style signatures
    # ------------------------------------------------------------------

    def receive_game_start_message(self, game_info):
        print("[MC] Game start – stacks:", {p['name']: p['stack'] for p in game_info['seats']})

    def receive_round_start_message(self, round_count, hole_card, seats):
        print(f"[MC] >>> ROUND {round_count} – hole: {hole_card}")

    def receive_street_start_message(self, street, round_state):
        print(f"[MC] --- street {street} community: {round_state['community_card']}")

    def receive_game_update_message(self, action, round_state):
        pass  # we keep silent – could be used for future learning

    def receive_round_result_message(self, winners, hand_info, round_state):
        print("[MC] Round result – stacks:", {p['name']: p['stack'] for p in round_state['seats']})


# Factory function ----------------------------------------------------------------

def setup_ai():
    """Entry‑point used by the framework."""
    return MonteCarloPlayer()
