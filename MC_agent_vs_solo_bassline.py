import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai
from agents.MC_player import setup_ai as MC_ai

# baseline list
baseline_ai_list = [
    #("baseline0", baseline0_ai),
    #("baseline1", baseline1_ai),
    #("baseline2", baseline2_ai),
    ("baseline3", baseline3_ai),
    #("baseline4", baseline4_ai),
    #("baseline5", baseline5_ai),
    #("baseline6", baseline6_ai),
    #("baseline7", baseline7_ai)
]

def evaluate_MC_agent_vs_baselines(num_games_per_baseline):
    # summary 結果
    results_summary = {}

    for baseline_name, baseline_ai in baseline_ai_list:
        print(f"\n===== Evaluating MC agent vs {baseline_name} =====")

        p2_wins = 0

        for game_idx in range(num_games_per_baseline):
            print(f"\n--- Game {game_idx+1}/{num_games_per_baseline} vs {baseline_name} ---")

            # 設定 config
            config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)

            # p1 = baseline AI
            config.register_player(name="p1", algorithm=baseline_ai())
            # p2 = MC AI
            config.register_player(name="p2", algorithm=MC_ai())

            # 跑一局
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

        # 計算 baseline 對應 win rate
        win_rate = p2_wins / num_games_per_baseline * 100
        results_summary[baseline_name] = win_rate

    # 總結所有 baseline 的結果
    print(f"\n===== Final Summary after {num_games_per_baseline} games per baseline =====")
    for baseline_name, win_rate in results_summary.items():
        print(f"MC agent vs {baseline_name}: win rate = {win_rate:.2f}%")

if __name__ == "__main__":
    evaluate_MC_agent_vs_baselines(num_games_per_baseline=5)
