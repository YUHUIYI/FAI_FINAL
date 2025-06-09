from RL_model import PokerEnv
import numpy as np

# 建立環境
env = PokerEnv()

# reset → 取得初始 observation
obs = env.reset()
print("Initial observation:", obs)

# 單局 episode
done = False
step_count = 0
while not done:
    # 隨機選 action 測試 (0=fold, 1=call, 2=raise)
    action = np.random.choice(env.num_actions)
    print(f"\nStep {step_count} - Action taken: {action}")

    obs, reward, done, info = env.step(action)

    print("New observation:", obs)
    print("Reward:", reward)
    print("Done:", done)

    step_count += 1

print("\nTest complete.")
