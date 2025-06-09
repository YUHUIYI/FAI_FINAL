import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from agents.Heuristic_player import setup_ai as heuristic_ai
from agents.RL_player import setup_ai as RL_ai

def evaluate_RL_agent(num_games=100):
    p2_wins = 0

    for game_idx in range(num_games):
        print(f"\n===== Game {game_idx+1}/{num_games} =====")

        # 設定 config
        config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)

        # p1 = baseline0 AI
        config.register_player(name="p1", algorithm=baseline1_ai())
        # p2 = RL AI
        config.register_player(name="p2", algorithm=RL_ai())

        # 跑一局 → 不 verbose
        game_result = start_poker(config, verbose=0)

        # 取 stack
        p1_stack = None
        p2_stack = None
        for player_info in game_result["players"]:
            if player_info["name"] == "p1":
                p1_stack = player_info["stack"]
            elif player_info["name"] == "p2":
                p2_stack = player_info["stack"]

        # 判斷勝負
        if p2_stack is not None and p1_stack is not None:
            if p2_stack > p1_stack:
                p2_wins += 1
                result_str = "p2 win"
            elif p2_stack < p1_stack:
                result_str = "p1 win"
            else:
                result_str = "draw"

            print(f"Result: p1 stack = {p1_stack}, p2 stack = {p2_stack} → {result_str}")

    # 印出總結
    win_rate = p2_wins / num_games * 100
    print(f"\n===== Summary after {num_games} games =====")
    print(f"p2 (RL agent) win rate: {win_rate:.2f}%")

if __name__ == "__main__":
    evaluate_RL_agent(num_games=100)
