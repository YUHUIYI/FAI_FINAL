from RL_player import RLAgentPlayer
from RL_model import PokerEnv
import numpy as np
import torch

# 建立 RLAgentPlayer → 用學好的 model
agent = RLAgentPlayer(model_path="poker_policy_net.pth")

# 建立環境 → 傳入 agent
env = PokerEnv(agent=agent)

# reset → 取得初始 observation
obs = env.reset()
print("Initial observation:", obs)

# 單局 episode
done = False
step_count = 0
while not done:
    # 這次不要 random → 直接由 RLAgentPlayer 出 action
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q_values = agent.policy_net(obs_tensor)
    action = torch.argmax(q_values).item()

    print(f"\nStep {step_count} - Action taken: {action}")

    obs, reward, done, info = env.step(action)

    print("New observation:", obs)
    print("Reward:", reward)
    print("Done:", done)

    step_count += 1

print("\nTest complete.")
