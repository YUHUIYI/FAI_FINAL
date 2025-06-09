# RF_train.py
import numpy as np
import sys
import os
import random
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from RL_model import PokerEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 參數
num_episodes = 5000

# 資料集
obs_list = []
action_list = []

# 環境
env = PokerEnv()

print("Collecting data...")
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        # baseline 隨機選 action 當 label (也可以改 heuristic_ai 出 action)
        action_idx = random.choice([0, 1, 2])

        # 存資料
        obs_list.append(obs)
        action_list.append(action_idx)

        obs, reward, done, _ = env.step(action_idx)

    if (episode+1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes} collected")

# 轉成 numpy
X = np.array(obs_list)
y = np.array(action_list)

# 分 train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練 RF
print("\nTraining Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate
y_pred = rf_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 儲存 model
with open("poker_policy_rf.pkl", "wb") as f:
    pickle.dump(rf_clf, f)
print("\nModel saved to poker_policy_rf.pkl")
