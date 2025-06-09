# RF_train.py
import numpy as np
import sys
import os
import pickle
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from game.engine.dealer import Dealer
from RF_player import RFPlayer

# baseline AI 列表
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai

baseline_ai_list = [
    baseline0_ai,
    baseline1_ai,
    baseline2_ai,
    baseline3_ai,
    baseline4_ai,
    baseline5_ai,
    baseline6_ai,
    baseline7_ai
]

# 參數
num_episodes = 5000

# 資料集
obs_list = []
action_list = []

print("Collecting data...")
for episode in range(num_episodes):
    # 每局隨機 baseline0~7
    baseline_opponent_ai = random.choice(baseline_ai_list)()

    dealer = Dealer(small_blind_amount=20, initial_stack=1000)
    rf_player = RFPlayer()
    dealer.register_player("rf_player", rf_player)
    dealer.register_player("baseline_opponent", baseline_opponent_ai)

    dealer.set_verbose(0)
    dealer.start_game(max_round=1)

    # 收 baseline AI 出的 action
    obs_list.extend(rf_player.observation_history)
    action_list.extend(rf_player.opponent_action_history)

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
