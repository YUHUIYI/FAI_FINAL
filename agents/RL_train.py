from RL_model import PokerEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# policy network (for DQN → 你也可以改 PPO)
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 超參數
num_episodes = 1000
gamma = 0.99
learning_rate = 1e-3

# 建立 env
env = PokerEnv()

# 建立 policy network
policy_net = PolicyNet(env.obs_dim, env.num_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# 訓練 loop
for episode in range(num_episodes):
    obs = env.reset()
    done = False

    total_reward = 0
    while not done:
        # 轉換 obs → tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        # 用 policy net 選 action
        q_values = policy_net(obs_tensor)
        action = torch.argmax(q_values).item()

        # ε-greedy 探索
        epsilon = max(0.1, 1 - episode / 2000)
        if np.random.rand() < epsilon:
            action = np.random.choice(env.num_actions)

        # 執行 step
        next_obs, reward, done, _ = env.step(action)

        total_reward += reward

        # 計算 target Q value
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        next_q_values = policy_net(next_obs_tensor)
        max_next_q = torch.max(next_q_values).detach()

        target_q = reward + gamma * max_next_q * (0 if done else 1)

        # 計算 loss
        q_value = q_values[0, action]
        loss = loss_fn(q_value, target_q)

        # 更新 policy net
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 下一步
        obs = next_obs

    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

# 儲存 model
torch.save(policy_net.state_dict(), "poker_policy_net.pth")
print("Model saved to poker_policy_net.pth")
